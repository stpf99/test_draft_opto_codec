#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dec2mp4.py - strumień OPTO (.zst) -> plik mp4

  python3 dec2mp4.py out/orig_def.zst out/orig_def.mp4            # oryginalny orig_dec.py (OPTO50)
  python3 dec2mp4.py out/orig_def.zst out/orig_def_core.mp4 --core  # opto_core.py (OPTO50 i OPTO51)
  python3 dec2mp4.py strumien.zst wynik.mp4 --crf 0 --444           # bez strat dodatkowych (domyślnie crf 14, 4:2:0 = ok. -0.3 dB PSNR)

Strumień nie niesie chrominancji (enkoder koduje tylko Y), więc wynik jest czarno-biały.
Połóż skrypt obok orig_dec.py / opto_core.py.
"""
import os, sys, struct, shutil, subprocess, argparse, importlib.util
import numpy as np
import cv2
import zstandard as zstd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
HDR = ">6sHHHIHB"


class Mp4Out:
    """ffmpeg (libx264) przez pipe; bez ffmpeg w PATH -> cv2.VideoWriter (mp4v)."""

    def __init__(self, path, w, h, f100, crf=14, y444=False):
        self.n, self.p, self.vw = 0, None, None
        if shutil.which("ffmpeg"):
            cmd = ["ffmpeg", "-y", "-loglevel", "error",
                   "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{f100}/100", "-i", "-",
                   "-an", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                   "-c:v", "libx264", "-preset", "slow", "-crf", str(crf), "-pix_fmt", "yuv444p" if y444 else "yuv420p",
                   "-movflags", "+faststart", path]
            self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        else:
            print("brak ffmpeg w PATH -> cv2.VideoWriter (mp4v)" + ("; --crf/--444 działają tylko z ffmpeg" if (crf != 14 or y444) else ""))
            self.vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), f100 / 100.0, (w, h))

    def write(self, bgr):
        bgr = np.ascontiguousarray(bgr)
        self.n += 1
        if self.p:
            self.p.stdin.write(bgr.tobytes())
        else:
            self.vw.write(bgr)

    def close(self):
        if self.p:
            self.p.stdin.close()
            self.p.wait()
        else:
            self.vw.release()


def read_stream(path):
    raw = zstd.ZstdDecompressor().decompress(open(path, "rb").read())
    magic, w, h, f100, total, gop, bs = struct.unpack(HDR, raw[:struct.calcsize(HDR)])
    return raw, magic, w, h, f100


def via_orig_dec(stream, out, f100, crf=14, y444=False):
    """Oryginalny dekoder: play() tylko woła cv2.imshow - przechwytujemy klatki (jak decode_orig w common.py)."""
    spec = importlib.util.spec_from_file_location("orig_dec", os.path.join(HERE, "orig_dec.py"))
    dec = importlib.util.module_from_spec(spec)
    sys.modules["orig_dec"] = dec
    spec.loader.exec_module(dec)

    sink = {}

    def imshow(_name, img):                      # img = BGR (H,W,3) uint8
        if "o" not in sink:
            h, w = img.shape[:2]
            sink["o"] = Mp4Out(out, w, h, f100, crf, y444)
        sink["o"].write(img)

    saved = (cv2.namedWindow, cv2.imshow, cv2.waitKey, cv2.destroyAllWindows)
    cv2.namedWindow, cv2.imshow = (lambda *a, **k: None), imshow
    cv2.waitKey, cv2.destroyAllWindows = (lambda d=0: -1), (lambda: None)
    try:
        dec.OptoDecoderEngineV53(stream).play()
    finally:
        cv2.namedWindow, cv2.imshow, cv2.waitKey, cv2.destroyAllWindows = saved
        if "o" in sink:
            sink["o"].close()
    return sink["o"].n if "o" in sink else 0


def via_core(raw, magic, w, h, f100, out, crf=14, y444=False):
    """Dekoder na wspólnym rdzeniu opto_core (OPTO50/51/52, z chromą gdy jest)."""
    import opto_core as oc
    mp4 = Mp4Out(out, w, h, f100, crf, y444)
    for planes in oc.decode_frames(raw):
        if len(planes) == 3:
            y, cr, cb = planes
            ycc = np.stack([y, cr, cb], axis=-1)
            bgr = cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)
        else:
            bgr = cv2.cvtColor(planes[0], cv2.COLOR_GRAY2BGR)
        mp4.write(bgr)
    mp4.close()
    return mp4.n


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="strumień OPTO (.zst) -> mp4")
    ap.add_argument("stream")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--core", action="store_true", help="dekoduj przez opto_core.py zamiast orig_dec.py")
    ap.add_argument("--crf", type=int, default=14, help="x264 CRF (0 = bezstratnie; domyślnie 14)")
    ap.add_argument("--444", dest="y444", action="store_true", help="bez podpróbkowania chromy (yuv444p); większy plik, brak straty kolorów")
    a = ap.parse_args()
    out = a.out or os.path.splitext(a.stream)[0] + ".mp4"

    raw, magic, w, h, f100 = read_stream(a.stream)
    use_core = a.core or magic != b"OPTO50"       # orig_dec.py zna tylko OPTO50
    if use_core and not a.core:
        print(f"{magic.decode()} -> orig_dec.py tego nie czyta, używam opto_core")
    if use_core:
        n = via_core(raw, magic, w, h, f100, out, a.crf, a.y444)
    else:
        del raw
        n = via_orig_dec(a.stream, out, f100, a.crf, a.y444)
    print(f"{n} klatek ({w}x{h}, {f100 / 100:g} fps) -> {out}")
