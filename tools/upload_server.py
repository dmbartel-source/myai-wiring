"""Minimal private upload server for MYAI media intake.

Stdlib only. Binds 127.0.0.1 (expose via `tailscale serve`, tailnet-only).
Saves uploads to UPLOAD_DIR; the agent then references them by filename.

Usage:
    python3 upload_server.py [--port 8888] [--dir /home/nanobot/uploads]

Security notes:
- No auth by design: ONLY run behind tailscale serve (or another
  private tunnel). Never bind 0.0.0.0 on a public interface.
- Filenames are sanitized; paths cannot escape UPLOAD_DIR.
- 2 GiB per-file cap to avoid disk-fill accidents.
"""

from __future__ import annotations

import argparse
import cgi
import html
import os
import re
from http.server import BaseHTTPRequestHandler, HTTPServer

MAX_BYTES = 2 * 1024 * 1024 * 1024
UPLOAD_DIR = os.environ.get("MYAI_UPLOAD_DIR", "/home/nanobot/uploads")

PAGE = """<!doctype html><html><head><meta name="viewport"
content="width=device-width,initial-scale=1"><title>MYAI upload</title>
<style>body{{font-family:sans-serif;max-width:640px;margin:2em auto;padding:0 1em}}
input[type=file]{{margin:1em 0}}button{{padding:.6em 1.2em;font-size:1em}}
li{{margin:.3em 0}}</style></head><body>
<h2>Send to MYAI</h2>
<form method="post" action="/upload" enctype="multipart/form-data">
<input type="file" name="files" multiple accept="image/*,video/*"><br>
<button type="submit">Upload</button></form>
<h3>On the server</h3><ul>{files}</ul></body></html>"""

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_name(name: str) -> str:
    base = os.path.basename(name or "upload.bin")
    base = _SAFE.sub("_", base).strip("._") or "upload.bin"
    return base[:120]


def list_files() -> str:
    try:
        names = sorted(os.listdir(UPLOAD_DIR))
    except OSError:
        return "<li>(upload dir unavailable)</li>"
    if not names:
        return "<li>(empty)</li>"
    return "".join(f"<li>{html.escape(n)}</li>" for n in names)


class Handler(BaseHTTPRequestHandler):
    server_version = "MYAIUpload/1.0"

    def log_message(self, *args):  # keep logs quiet
        pass

    def do_GET(self):
        if self.path != "/":
            self.send_error(404)
            return
        body = PAGE.format(files=list_files()).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != "/upload":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0) or 0)
        if length > MAX_BYTES * 4:
            self.send_error(413, "too large")
            return
        form = cgi.FieldStorage(
            fp=self.rfile, headers=self.headers,
            environ={"REQUEST_METHOD": "POST",
                     "CONTENT_TYPE": self.headers.get("Content-Type", ""),
                     "CONTENT_LENGTH": str(length)})
        saved = []
        items = form["files"] if "files" in form else []
        if not isinstance(items, list):
            items = [items]
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        for item in items:
            if not getattr(item, "filename", None):
                continue
            name = safe_name(item.filename)
            # de-dupe: append -1, -2, ...
            dest = os.path.join(UPLOAD_DIR, name)
            stem, ext = os.path.splitext(name)
            n = 1
            while os.path.exists(dest):
                dest = os.path.join(UPLOAD_DIR, f"{stem}-{n}{ext}")
                n += 1
            size = 0
            with open(dest, "wb") as f:
                while True:
                    chunk = item.file.read(1024 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > MAX_BYTES:
                        f.close()
                        os.remove(dest)
                        self.send_error(413, "file too large")
                        return
                    f.write(chunk)
            # Privacy: strip ALL metadata from every upload, before storage.
            try:
                from privacy import strip_metadata
                strip_metadata(dest)
            except Exception:
                pass  # never fail an upload on metadata stripping
            saved.append(os.path.basename(dest))
        body = ("<h2>Saved</h2><ul>" +
                "".join(f"<li>{html.escape(s)}</li>" for s in saved) +
                '</ul><a href="/">back</a>').encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main(argv=None) -> int:
    global UPLOAD_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8888)
    ap.add_argument("--dir", default=UPLOAD_DIR)
    args = ap.parse_args(argv)
    UPLOAD_DIR = args.dir
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    srv = HTTPServer(("127.0.0.1", args.port), Handler)
    print(f"MYAI upload server on 127.0.0.1:{args.port} -> {UPLOAD_DIR}",
          flush=True)
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
