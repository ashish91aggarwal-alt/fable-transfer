"""Tiny file server for the pod. Every URL must start with /<SECRET>/ ; GET reads, PUT writes under ROOT.
Run: SECRET=xxxx ROOT=/w/share python3 fileserver.py 8000   (expose 8000/http on the pod)"""
import os, sys, http.server, urllib.parse
SECRET = os.environ["SECRET"]; ROOT = os.path.realpath(os.environ.get("ROOT", "/w/share")); os.makedirs(ROOT, exist_ok=True)
class H(http.server.BaseHTTPRequestHandler):
    def _path(self):
        p = urllib.parse.unquote(urllib.parse.urlparse(self.path).path).lstrip("/")
        if not p.startswith(SECRET + "/"): return None
        full = os.path.realpath(os.path.join(ROOT, p[len(SECRET) + 1:]))
        return full if full.startswith(ROOT + os.sep) else None
    def do_GET(self):
        f = self._path()
        if not f or not os.path.isfile(f): self.send_response(404); self.end_headers(); return
        self.send_response(200); self.send_header("Content-Length", str(os.path.getsize(f))); self.end_headers()
        with open(f, "rb") as fh: self.wfile.write(fh.read())
    def do_PUT(self):
        f = self._path()
        if not f: self.send_response(403); self.end_headers(); return
        os.makedirs(os.path.dirname(f), exist_ok=True); n = int(self.headers.get("Content-Length", 0))
        with open(f, "wb") as fh: fh.write(self.rfile.read(n))
        self.send_response(201); self.end_headers()
        try: os.utime("/w/activity")
        except Exception: pass
    def log_message(self, *a): pass
http.server.ThreadingHTTPServer(("0.0.0.0", int(sys.argv[1]) if len(sys.argv) > 1 else 8000), H).serve_forever()
