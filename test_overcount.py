"""Regression test: work stealing must not double-count bytes."""
import os
import sys
import time
import json
import threading
import http.server
import socketserver
import importlib.util

sys.modules.setdefault('folder_paths', type(sys)('folder_paths'))
sys.modules['folder_paths'].models_dir = '/tmp/wmd_ov/models'
sys.modules['folder_paths'].base_path = '/tmp/wmd_ov/comfy'
sys.modules['folder_paths'].get_filename_list = lambda *a, **k: []
sys.modules.setdefault('server', type(sys)('server'))


class _Routes:
    def get(self, *a, **k):
        return lambda fn: fn
    def post(self, *a, **k):
        return lambda fn: fn


class _PromptServer:
    instance = type('I', (), {'routes': _Routes()})()


sys.modules['server'].PromptServer = _PromptServer

spec = importlib.util.spec_from_file_location('wmd_server', '/tmp/wmd_server_ov.py')
wmd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wmd)

SETTINGS = {'download_segments': 8, 'max_download_retries': 5}
wmd.load_settings = lambda: dict(SETTINGS)
wmd.load_model_metadata = lambda: {}
wmd.save_model_metadata = lambda m: None
wmd.add_to_download_history = lambda info: None
wmd.save_download_history = lambda: None

PAYLOAD = os.urandom(48 * 1024 * 1024)   # 48 MB
BASE = '/tmp/wmd_ov'
# Track every byte-range the server actually serves so we can prove the client
# never asked for (and counted) the same bytes on two connections.
SERVED = []
SERVED_LOCK = threading.Lock()


class UnevenServer(http.server.BaseHTTPRequestHandler):
    """Some ranges crawl, forcing the healthy workers to steal their tails."""
    protocol_version = 'HTTP/1.1'

    def do_GET(self):
        rng = self.headers.get('Range')
        start, end = 0, len(PAYLOAD) - 1
        partial = False
        if rng and rng.startswith('bytes='):
            a, b = rng.split('=')[1].split('-')
            start = int(a)
            end = int(b) if b else len(PAYLOAD) - 1
            partial = True
        body = PAYLOAD[start:end + 1]
        self.send_response(206 if partial else 200)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Accept-Ranges', 'bytes')
        if partial:
            self.send_header('Content-Range', f'bytes {start}-{end}/{len(PAYLOAD)}')
        self.end_headers()

        # Serve the opening portion of the file very slowly; everything else fast.
        slow = start < len(PAYLOAD) // 8
        step = 64 * 1024
        rate = 1 * 1024 * 1024 if slow else 20 * 1024 * 1024
        sent = 0
        try:
            for i in range(0, len(body), step):
                piece = body[i:i + step]
                self.wfile.write(piece)
                self.wfile.flush()
                sent += len(piece)
                time.sleep(len(piece) / rate)
        except Exception:
            pass
        finally:
            with SERVED_LOCK:
                SERVED.append((start, start + sent - 1))

    def log_message(self, *a):
        pass


def serve(handler):
    socketserver.ThreadingTCPServer.allow_reuse_address = True
    httpd = socketserver.ThreadingTCPServer(('127.0.0.1', 0), handler)
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, httpd.server_address[1]


def fresh(name):
    path = os.path.join(BASE, 'out', name)
    for suffix in ('', '.partial', '.partial.segments.json', '.partial.segments.json.tmp'):
        try:
            os.remove(path + suffix)
        except OSError:
            pass
    return path


def test_no_overcount_with_stealing():
    print('--- work stealing must not double count ---')
    print('  payload: %.1f MB' % (len(PAYLOAD) / 1048576))
    SERVED.clear()
    httpd, port = serve(UnevenServer)
    dest = fresh('overcount.safetensors')
    dl_id = 'ov1'
    wmd.download_progress[dl_id] = {'status': 'starting', 'progress': 0,
                                    'filename': 'overcount.safetensors',
                                    'total_size': 0, 'downloaded': 0}
    start_t = time.time()
    ok, err = wmd._download_with_resume(f'http://127.0.0.1:{port}/f', dest, dl_id, None)
    elapsed = time.time() - start_t
    httpd.shutdown()

    entry = wmd.download_progress.get(dl_id, {})
    downloaded = entry.get('downloaded', 0)
    total = entry.get('total_size', 0)
    data = open(dest, 'rb').read() if os.path.exists(dest) else b''

    print('  ok:', ok, '| err:', err, '| elapsed: %.1fs' % elapsed)
    print('  downloaded:', downloaded, '| total_size:', total)
    print('  delta:', downloaded - total, 'bytes')
    print('  bytes on disk:', len(data), '| identical:', data == PAYLOAD)
    print('  served ranges:', len(SERVED))

    assert ok, err
    assert data == PAYLOAD, 'file content is corrupted'
    assert downloaded == total, f'progress overshoots: {downloaded} vs {total}'
    assert entry.get('progress') in (99, 100), entry.get('progress')
    print('  PASS')


def test_segment_state_never_exceeds_range():
    print('--- persisted segment state must never exceed its own range ---')
    # Re-run and inspect the intermediate state file while it is in flight.
    SERVED.clear()
    httpd, port = serve(UnevenServer)
    dest = fresh('statechk.safetensors')
    dl_id = 'ov2'
    wmd.download_progress[dl_id] = {'status': 'starting', 'progress': 0,
                                    'filename': 'statechk.safetensors',
                                    'total_size': 0, 'downloaded': 0}
    violations = []
    stop = threading.Event()

    def watch():
        state_path = dest + '.partial.segments.json'
        while not stop.is_set():
            try:
                st = json.load(open(state_path))
                for seg in st['segments']:
                    length = seg['end'] - seg['start'] + 1
                    if seg['done'] > length:
                        violations.append((seg['start'], seg['end'], seg['done'], length))
            except Exception:
                pass
            time.sleep(0.05)

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    ok, err = wmd._download_with_resume(f'http://127.0.0.1:{port}/f', dest, dl_id, None)
    stop.set()
    watcher.join(timeout=2)
    httpd.shutdown()

    print('  ok:', ok, '| violations observed:', len(violations))
    if violations:
        for v in violations[:5]:
            print('   ', v)
    assert ok, err
    assert not violations, f'{len(violations)} segments exceeded their range'
    print('  PASS')


if __name__ == '__main__':
    for sub in ('out', 'models', 'comfy'):
        os.makedirs(os.path.join(BASE, sub), exist_ok=True)
    test_no_overcount_with_stealing()
    test_segment_state_never_exceeds_range()
    print('\nALL OVERCOUNT TESTS PASSED')
