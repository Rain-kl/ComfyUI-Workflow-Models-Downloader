"""Deterministic reproduction: shrink a segment's end mid-download.

The old worker cached `end` in a local at entry and never refreshed it, so when
another worker stole the tail (shrinking current['end']) the original worker
kept downloading to the stale boundary. It therefore wrote the stolen bytes too
and reported done larger than its own range.
"""
import os
import sys
import time
import threading
import http.server
import socketserver
import importlib.util

sys.modules.setdefault('folder_paths', type(sys)('folder_paths'))
sys.modules['folder_paths'].models_dir = '/tmp/wmd_shrink/models'
sys.modules['folder_paths'].base_path = '/tmp/wmd_shrink/comfy'
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

SPEC_PATH = os.environ.get('WMD_SERVER', '/tmp/wmd_server_ov.py')
spec = importlib.util.spec_from_file_location('wmd_server', SPEC_PATH)
wmd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wmd)

wmd.load_settings = lambda: {'download_segments': 8, 'max_download_retries': 3}
wmd.load_model_metadata = lambda: {}
wmd.save_model_metadata = lambda m: None
wmd.add_to_download_history = lambda info: None
wmd.save_download_history = lambda: None

PAYLOAD = os.urandom(8 * 1024 * 1024)
BASE = '/tmp/wmd_shrink'


class SlowServer(http.server.BaseHTTPRequestHandler):
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
        step = 16 * 1024
        try:
            for i in range(0, len(body), step):
                self.wfile.write(body[i:i + step])
                self.wfile.flush()
                time.sleep(0.01)          # ~1.6 MB/s, slow enough to interrupt
        except Exception:
            pass

    def log_message(self, *a):
        pass


def serve():
    socketserver.ThreadingTCPServer.allow_reuse_address = True
    httpd = socketserver.ThreadingTCPServer(('127.0.0.1', 0), SlowServer)
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, httpd.server_address[1]


def main():
    for sub in ('out', 'models', 'comfy'):
        os.makedirs(os.path.join(BASE, sub), exist_ok=True)
    dest = os.path.join(BASE, 'out', 'shrink.safetensors')
    for suffix in ('', '.partial', '.partial.segments.json', '.partial.segments.json.tmp'):
        try:
            os.remove(dest + suffix)
        except OSError:
            pass

    total = len(PAYLOAD)
    half = total // 2
    # One segment covering the whole file, already claiming half the bytes are
    # present so the worker starts in the middle.
    state = {'total_size': total,
             'segments': [{'start': 0, 'end': total - 1, 'done': half}]}
    state_path = dest + '.partial.segments.json'
    import json
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest + '.partial', 'wb') as f:
        f.truncate(total)
    with open(state_path, 'w') as f:
        json.dump(state, f)

    httpd, port = serve()
    url = f'http://127.0.0.1:{port}/f'
    dl_id = 'shrink1'
    wmd.download_progress[dl_id] = {'status': 'starting', 'progress': 0,
                                    'filename': 'shrink.safetensors',
                                    'total_size': total, 'downloaded': half}

    segment = state['segments'][0]
    counters = {'downloaded': half}
    error_box = []
    lock = threading.Lock()
    fd = os.open(dest + '.partial', os.O_WRONLY)

    # Shrink the range mid-flight, exactly like a thief taking the tail.
    shrink_at = half + 256 * 1024
    def shrink():
        while True:
            with lock:
                cur = segment['start'] + segment.get('done', 0)
            if cur >= shrink_at:
                break
            time.sleep(0.01)
        with lock:
            segment['end'] = shrink_at + 512 * 1024 - 1   # cut off the remaining tail
        return

    t = threading.Thread(target=wmd._segment_worker,
                         args=(url, fd, segment, {}, dl_id, counters, error_box,
                               state, state_path, lock),
                         daemon=True)
    t.start()
    shrinker = threading.Thread(target=shrink, daemon=True)
    shrinker.start()
    t.join(timeout=40)
    shrinker.join(timeout=5)
    httpd.shutdown()
    os.close(fd)

    shrunk_end = segment['end']
    range_len = shrunk_end - segment['start'] + 1
    done = segment.get('done', 0)
    print('SPEC:', SPEC_PATH)
    print('shrunk range length:', range_len)
    print('segment done:', done)
    print('overshoot:', done - range_len)
    print('counters total:', counters['downloaded'])
    print('sum of segment done:', sum(s.get('done', 0) for s in state['segments']))

    problems = []
    if done > range_len:
        problems.append(f'done {done} exceeds range {range_len} by {done - range_len}')
    if counters['downloaded'] > total:
        problems.append(f'counters report {counters["downloaded"]} > total {total}')
    if problems:
        print('FAIL:', '; '.join(problems))
        return 1
    print('PASS')
    return 0


if __name__ == '__main__':
    sys.exit(main())
