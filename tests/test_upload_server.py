"""Tests for tools/upload_server.py. Stdlib only, no network."""

import io
import os
import sys
import tempfile
import threading
import time
import urllib.request

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

import upload_server as us


def test_safe_name():
    assert us.safe_name("../../etc/passwd") == "passwd"
    assert us.safe_name("my video (1).MP4") == "my_video_1_.MP4"
    assert us.safe_name("") == "upload.bin"
    assert len(us.safe_name("a" * 200 + ".mp4")) <= 120


def _start_server(tmpdir):
    us.UPLOAD_DIR = tmpdir
    srv = us.HTTPServer(("127.0.0.1", 0), us.Handler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    time.sleep(0.2)
    return srv, port


def _multipart(fields):
    boundary = "TESTBOUNDARY123"
    body = io.BytesIO()
    for name, filename, data in fields:
        body.write(f"--{boundary}\r\n".encode())
        body.write(
            f'Content-Disposition: form-data; name="{name}"; '
            f'filename="{filename}"\r\n'.encode())
        body.write(b"Content-Type: application/octet-stream\r\n\r\n")
        body.write(data + b"\r\n")
    body.write(f"--{boundary}--\r\n".encode())
    return body.getvalue(), boundary


def test_upload_roundtrip():
    with tempfile.TemporaryDirectory() as td:
        srv, port = _start_server(td)
        try:
            # GET /
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/") as r:
                assert r.status == 200
                assert b"MYAI" in r.read()
            # POST a file
            data, boundary = _multipart(
                [("files", "clip.mp4", b"FAKEMP4DATA" * 100)])
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/upload", data=data,
                headers={"Content-Type":
                         f"multipart/form-data; boundary={boundary}"})
            with urllib.request.urlopen(req) as r:
                assert r.status == 200
                assert b"clip.mp4" in r.read()
            assert os.path.getsize(os.path.join(td, "clip.mp4")) > 0
            with open(os.path.join(td, "clip.mp4"), "rb") as f:
                assert f.read(11) == b"FAKEMP4DATA"
            # path traversal attempt is neutralized
            data2, b2 = _multipart([("files", "../../evil.sh", b"x")])
            req2 = urllib.request.Request(
                f"http://127.0.0.1:{port}/upload", data=data2,
                headers={"Content-Type":
                         f"multipart/form-data; boundary={b2}"})
            with urllib.request.urlopen(req2):
                pass
            assert os.path.exists(os.path.join(td, "evil.sh"))
            assert not os.path.exists(os.path.join(td, "..", "evil.sh"))
        finally:
            srv.shutdown()
