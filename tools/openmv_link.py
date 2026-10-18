"""Host-side link to an OpenMV cam running MicroPython over USB CDC.

Talks to the board's raw REPL, so it can run arbitrary code, read stdout,
capture frames and (re)write files on the camera's filesystem without the
OpenMV IDE.  Used only for debugging; the car itself uses UART3.
"""
import base64
import time

import serial


class OpenMV:
    def __init__(self, port="/dev/ttyACM0", baud=115200):
        self.s = serial.Serial(port, baud, timeout=0.05)
        self.s.reset_input_buffer()
        time.sleep(0.2)

    # ------------------------------------------------------------ plumbing
    def _read_until(self, marker, timeout=10.0, maxbytes=4 * 1024 * 1024):
        end = time.time() + timeout
        buf = bytearray()
        while time.time() < end:
            chunk = self.s.read(4096)
            if chunk:
                buf.extend(chunk)
                if buf.endswith(marker):
                    return bytes(buf)
                if len(buf) > maxbytes:
                    break
        return bytes(buf)

    def _drain(self, quiet=0.15):
        out = bytearray()
        last = time.time()
        while time.time() - last < quiet:
            chunk = self.s.read(4096)
            if chunk:
                out.extend(chunk)
                last = time.time()
        return bytes(out)

    def interrupt(self):
        """Stop whatever script is running and land in the friendly REPL."""
        self.s.write(b"\r\x03\x03")
        time.sleep(0.3)
        return self._drain(0.2)

    def raw_repl(self):
        """Interrupt any running script, then enter the raw REPL."""
        self.s.write(b"\r\x03\x03")
        time.sleep(0.15)
        self.s.reset_input_buffer()
        self.s.write(b"\x01")  # Ctrl-A
        got = self._read_until(b">", timeout=3.0)
        if b"raw REPL" not in got:
            # Not in raw mode (e.g. mid soft-reboot): try once more.
            time.sleep(0.3)
            self.s.write(b"\x03\x01")
            got += self._read_until(b">", timeout=3.0)
        return got

    def friendly_repl(self):
        self.s.write(b"\x02")  # Ctrl-B
        time.sleep(0.2)
        return self._drain(0.2)

    def soft_reset(self):
        """Reset the board; it will run main.py again."""
        self.friendly_repl()
        self.s.write(b"\x04")  # Ctrl-D
        time.sleep(1.0)

    # --------------------------------------------------------------- exec
    def exec(self, code, timeout=10.0):
        """Run `code` in the raw REPL, return (stdout, stderr)."""
        if isinstance(code, str):
            code = code.encode()
        self.s.reset_input_buffer()
        self.s.write(code + b"\x04")
        end = time.time() + timeout
        buf = bytearray()
        while time.time() < end:
            chunk = self.s.read(4096)
            if chunk:
                buf.extend(chunk)
                if buf.endswith(b"\x04>"):
                    break
        raw = bytes(buf)
        # Format: \r\nOK\r\n <stdout> \x04 <stderr> \x04 >
        if raw.startswith(b"\r\nOK\r\n"):
            raw = raw[6:]
        elif raw.startswith(b"OK"):
            raw = raw[2:]
        parts = raw.split(b"\x04")
        out = parts[0]
        err = parts[1] if len(parts) > 1 else b""
        return out, err

    def eval(self, expr, timeout=10.0):
        out, err = self.exec("print(repr(%s))" % expr, timeout)
        text = out.decode(errors="replace").strip()
        if err.strip():
            raise RuntimeError(err.decode(errors="replace"))
        return eval(text)

    # -------------------------------------------------------------- files
    def listdir(self, path="/"):
        return self.eval("__import__('os').listdir(%r)" % path)

    def read_file(self, path):
        out, err = self.exec(
            "import ubinascii\n"
            "f=open(%r,'rb')\n"
            "d=f.read(); f.close()\n"
            "print(ubinascii.b2a_base64(d).decode())\n" % path,
            timeout=30.0)
        if err.strip():
            raise RuntimeError(err.decode(errors="replace"))
        text = out.decode(errors="replace").strip()
        return base64.b64decode(text)

    def write_file(self, path, data):
        b64 = base64.b64encode(data).decode()
        # Write in chunks so the REPL line buffer never overflows.
        self.exec("import ubinascii\n"
                  "f=open(%r,'wb')\n" % path)
        for i in range(0, len(b64), 512):
            chunk = b64[i:i + 512]
            out, err = self.exec(
                "f.write(ubinascii.a2b_base64(%r))\n" % chunk)
            if err.strip():
                raise RuntimeError(err.decode(errors="replace"))
        out, err = self.exec("f.close()\nprint('WROTE')\n")
        if b"WROTE" not in out:
            raise RuntimeError("write failed: %r %r" % (out, err))

    # ------------------------------------------------------------- frames
    def capture_jpeg(self, quality=90, settle_ms=600):
        code = (
            "import sensor, image, time, sys, ubinascii\n"
            "sensor.reset()\n"
            "sensor.set_pixformat(sensor.RGB565)\n"
            "sensor.set_framesize(sensor.VGA)\n"
            "sensor.set_auto_gain(False)\n"
            "sensor.set_auto_whitebal(False)\n"
            "sensor.skip_frames(time=%d)\n"
            "img = sensor.snapshot()\n"
            "img.compress(quality=%d)\n"
            "data = bytes(img.bytearray())\n"
            "print('LEN', len(data))\n"
            "print(ubinascii.b2a_base64(data).decode())\n"
        ) % (settle_ms, quality)
        out, err = self.exec(code, timeout=30.0)
        if err.strip():
            raise RuntimeError(err.decode(errors="replace"))
        lines = out.decode(errors="replace").strip().splitlines()
        if not lines or not lines[0].startswith("LEN"):
            raise RuntimeError("bad capture output: %r" % out[:200])
        return base64.b64decode(lines[1])

    def close(self):
        self.s.close()
