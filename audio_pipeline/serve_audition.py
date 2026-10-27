#!/usr/bin/env python3
"""
serve_audition.py - tiny static file server for the audio stem audition booth.

Serves the audio_pipeline/ folder (the parent of this script's folder) as the
webroot, so the booth page can fetch stems with absolute paths like:

    /separated_stems/htdemucs/<TRACK>/<stem>.wav

and the booth page itself lives at:

    /audition_booth/booth.html

Adds two things stdlib http.server doesn't give you for free:
  1. HTTP Range support (206 partial content) — needed so <audio> elements can
     seek/scrub inside multi-minute, 80-150MB stem WAVs without re-downloading
     the whole file from byte 0 every time.
  2. POST /pick — the booth's "PICK THIS BED" button posts {track, command, ts}
     here and it gets appended as a line to AUDITION_PICK.txt at the
     audio_pipeline root.

No third-party dependencies — stdlib only, so it runs with whatever `python`
is on PATH.

Usage:
    python serve_audition.py [port]      (default port 8777)
"""

import functools
import http.server
import json
import os
import re
import socketserver
import sys

DEFAULT_PORT = 8777
HERE = os.path.dirname(os.path.abspath(__file__))
SERVE_ROOT = os.path.dirname(HERE)                      # audio_pipeline/
PICK_FILE = os.path.join(SERVE_ROOT, "AUDITION_PICK.txt")

RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")


class RangeRequestHandler(http.server.SimpleHTTPRequestHandler):
    """SimpleHTTPRequestHandler + byte-range GET/HEAD + POST /pick."""

    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map, **{
        ".wav": "audio/wav",
        ".html": "text/html; charset=utf-8",
    }}

    def send_head(self):
        path = self.translate_path(self.path)
        if os.path.isdir(path):
            return super().send_head()
        if not os.path.exists(path):
            self.send_error(404, "File not found")
            return None
        ctype = self.guess_type(path)
        try:
            f = open(path, "rb")
        except OSError:
            self.send_error(404, "File not found")
            return None

        fs = os.fstat(f.fileno())
        file_len = fs.st_size
        range_header = self.headers.get("Range")
        self._range = None

        if range_header:
            m = RANGE_RE.match(range_header)
            if m:
                start_s, end_s = m.groups()
                if start_s == "" and end_s == "":
                    start, end = 0, file_len - 1
                elif start_s == "":
                    length = int(end_s)
                    start, end = max(0, file_len - length), file_len - 1
                else:
                    start = int(start_s)
                    end = int(end_s) if end_s else file_len - 1
                end = min(end, file_len - 1)
                if start > end or start >= file_len:
                    f.close()
                    self.send_response(416)
                    self.send_header("Content-Range", "bytes */%d" % file_len)
                    self.end_headers()
                    return None
                length = end - start + 1
                self.send_response(206)
                self.send_header("Content-type", ctype)
                self.send_header("Content-Range", "bytes %d-%d/%d" % (start, end, file_len))
                self.send_header("Content-Length", str(length))
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Last-Modified", self.date_time_string(int(fs.st_mtime)))
                self.end_headers()
                f.seek(start)
                self._range = (start, length)
                return f

        self.send_response(200)
        self.send_header("Content-type", ctype)
        self.send_header("Content-Length", str(file_len))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Last-Modified", self.date_time_string(int(fs.st_mtime)))
        self.end_headers()
        return f

    def copyfile(self, source, outputfile):
        if getattr(self, "_range", None):
            start, length = self._range
            remaining = length
            bufsize = 64 * 1024
            while remaining > 0:
                chunk = source.read(min(bufsize, remaining))
                if not chunk:
                    break
                try:
                    outputfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    return
                remaining -= len(chunk)
        else:
            try:
                super().copyfile(source, outputfile)
            except (BrokenPipeError, ConnectionResetError):
                return

    def do_POST(self):
        if self.path != "/pick":
            self.send_error(404, "not found")
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length else b"{}"
            data = json.loads(raw.decode("utf-8") or "{}")
        except Exception as e:
            self.send_error(400, "bad request: %s" % e)
            return

        track = str(data.get("track", "?"))
        command = str(data.get("command", ""))
        ts = str(data.get("ts", ""))
        line = "%s | %s | %s\n" % (ts, track, command)

        try:
            with open(PICK_FILE, "a", encoding="utf-8") as fh:
                fh.write(line)
        except OSError as e:
            self.send_error(500, "write failed: %s" % e)
            return

        body = json.dumps({"ok": True, "written_to": PICK_FILE}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))


class ReuseAddrTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PORT
    handler = functools.partial(RangeRequestHandler, directory=SERVE_ROOT)
    with ReuseAddrTCPServer(("127.0.0.1", port), handler) as httpd:
        url = "http://127.0.0.1:%d/audition_booth/booth.html" % port
        print("Audition booth server")
        print("Serving root : %s" % SERVE_ROOT)
        print("Picks logged : %s" % PICK_FILE)
        print("Open         : %s" % url)
        print("Ctrl+C to stop.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down.")


if __name__ == "__main__":
    main()
