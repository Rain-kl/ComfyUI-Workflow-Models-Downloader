"""Deterministic reproduction of the work-stealing double-count bug.

Instead of hoping a race happens, this drives the exact interleaving that the
old code got wrong: worker A is mid-download of a segment when worker B steals
A's tail. In the old implementation A kept writing and counting up to its
cached `end` (the pre-steal value), so the stolen bytes were counted twice.
"""
import os
import sys
import time
import json
import threading
import http.server
import socketserver
import importlib.util

sys.modules.setdefault('folder_paths', type(sys)('folder_paths'))
sys.modules['folder_paths'].models_dir = '/tmp/wmd_det/models'
sys.modules['folder_paths'].base_path = '/tmp/wmd_det/comfy'
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

SETTINGS = {'download_segments': 8, 'max_download_retries': 5}
wmd.load_settings = lambda: dict(SETTINGS)
wmd.load_model_metadata = lambda: {}
wmd.save_model_metadata = lambda m: None
wmd.add_to_download_history = lambda info: None
wmd.save_download_history = lambda: None
# Keep splits possible but not frequent, so the interleaving is predictable.
wmd.DOWNLOAD_SEGMENT_MIN_SIZE = 64 * 1024

PAYLOAD = os.urandom(24 * 1024 * 1024)   # 24 MB
BASE = '/tmp/wmd_det'

# A single range that is served very slowly so a thief can act mid-download.
SLOW_START = 0
SLOW_END = 12 * 1024 * 1024 - 1
SLOW_BPS = 700 * 1024


class ControlledServer(http.server.BaseHTTPRequestHandler):
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

        slow = start <= SLOW_END
        step = 32 * 1024
        rate = SLOW_BPS if slow else 50 * 1024 * 1024
        try:
            for i in range(0, len(body), step):
                self.wfile.write(body[i:i + step])
                self.wfile.flush()
                if slow:
                    time.sleep(len(body[i:i + step]) / rate)
        except Exception:
            pass

    def log_message(self, *a):
        pass


def serve():
    socketserver.ThreadingTCPServer.allow_reuse_address = True
    httpd = socketserver.ThreadingTCPServer(('127.0.0.1', 0), ControlledServer)
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, httpd.server_address[1]


def main():
    for sub in ('out', 'models', 'comfy'):
        os.makedirs(os.path.join(BASE, sub), exist_ok=True)
    dest = os.path.join(BASE, 'out', 'det.safetensors')
    for suffix in ('', '.partial', '.partial.segments.json', '.partial.segments.json.tmp'):
        try:
            os.remove(dest + suffix)
        except OSError:
            pass

    httpd, port = serve()
    url = f'http://127.0.0.1:{port}/f'
    dl_id = 'det1'
    wmd.download_progress[dl_id] = {'status': 'starting', 'progress': 0,
                                    'filename': 'det.safetensors',
                                    'total_size': 0, 'downloaded': 0}
    try:
        ok, err = wmd._download_with_resume(url, dest, dl_id, None)
    finally:
        httpd.shutdown()

    entry = wmd.download_progress.get(dl_id, {})
    downloaded = entry.get('downloaded', 0)
    total = entry.get('total_size', 0)
    data = open(dest, 'rb').read() if os.path.exists(dest) else b''

    print('SPEC:', SPEC_PATH)
    print('ok:', ok, '| err:', err)
    print('downloaded:', downloaded)
    print('total_size:', total)
    print('delta:', downloaded - total)
    print('disk bytes:', len(data), '| identical:', data == PAYLOAD)

    problems = []
    if not ok:
        problems.append(f'download failed: {err}')
    if data != PAYLOAD:
        problems.append('content mismatch')
    if downloaded != total:
        problems.append(f'progress overshoot by {downloaded - total} bytes')

    if problems:
        print('FAIL:', '; '.join(problems))
        return 1
    print('PASS')
    return 0


if __name__ == '__main__':
    sys.exit(main())
