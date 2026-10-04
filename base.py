import os, sys, struct, time
import numpy as np, cv2, zstandard as zstd
from common import *

if __name__ == "__main__":
    os.makedirs("out", exist_ok=True)
    if not os.path.exists("clip.avi"):
        make_clip("clip.avi")
    ysrc = read_y("clip.avi")
    print("clip:", ysrc.shape)
    enc = load_module("orig_enc.py", "orig_enc")
    dec = load_module("orig_dec.py", "orig_dec")

    from numba import njit
    @njit(fastmath=True)
    def sad_fixed(b1, b2):
        s = 0.0
        h, w = b1.shape
        for i in range(h):
            for j in range(w):
                d = np.int64(b1[i, j]) - np.int64(b2[i, j])
                s += d if d >= 0 else -d
        return s / float(h * w)
    orig_sad = enc.calc_sad_y_channel
    cfgs = {"def": (36.0, 110.0, False), "sane": (6.0, 20.0, False), "def_fix": (36.0, 110.0, True), "sane_fix": (6.0, 20.0, True)}
    for name, (dbd, thr, fix) in cfgs.items():
        enc.calc_sad_y_channel = sad_fixed if fix else orig_sad
        t0 = time.time()
        with enc.OptoCodecEngineV5(block_size=32, deadband=dbd, threshold=thr, zstd_level=12, gop=12) as e:
            e.encode("clip.avi", f"out/orig_{name}.zst")
        print(f"[{name}] encode time {time.time()-t0:.1f}s")
        fr = decode_orig(f"out/orig_{name}.zst", dec)
        print(f"[{name}] decoded frames: {len(fr)} / {len(ysrc)}")
        n = min(len(fr), len(ysrc))
        ps = [psnr(ysrc[i], fr[i]) for i in range(n)]
        print(f"[{name}] PSNR(Y) kl.0-14:", " ".join(f"{p:.1f}" for p in ps[:14]))
        print(f"[{name}] mean PSNR {np.mean(ps):.2f} dB;  grid score src={np.mean([grid_score(ysrc[i]) for i in range(n)]):.3f}  dec={np.mean([grid_score(fr[i]) for i in range(n)]):.3f}")
        np.save(f"out/orig_{name}_frames.npy", fr)
        print(f"[{name}] stream bytes: {os.path.getsize(f'out/orig_{name}.zst')}")

        # ---- analiza strumienia: saturacja gain, MOTION niezerowe ----
        raw = zstd.ZstdDecompressor().decompress(open(f"out/orig_{name}.zst", "rb").read())
        magic, w, h, f100, tot, gop, bs = struct.unpack(">6sHHHIHB", raw[:19])
        pos = 19
        nb = (h // 32) * (w // 32)
        cnt = {"SKIP": 0, "MOTION": 0, "MOTION_nz": 0, "DICT": 0}
        sat = tot_g = 0
        gains = []
        i = 0
        nfr = 0
        while pos < len(raw):
            i = 0
            while i < nb:
                tag = raw[pos]; pos += 1
                if tag == 254: pos += 4
                elif tag == 240: cnt["SKIP"] += raw[pos]; i += raw[pos]; pos += 1
                elif tag == 251:
                    dx, dy = struct.unpack("bb", raw[pos:pos+2]); pos += 2
                    cnt["MOTION"] += 1; cnt["MOTION_nz"] += (dx != 0 or dy != 0); i += 1
                elif tag == 253:
                    for k in range(4):
                        g = raw[pos + 3*k + 1]; gains.append(g)
                    pos += 12; cnt["DICT"] += 1; i += 1
                else:
                    raise SystemExit(f"bad tag {tag}")
            nfr += 1
        gains = np.array(gains)
        print(f"[{name}] frames parsed {nfr}; blocks {cnt}")
        print(f"[{name}] gain saturated (0 or 255): {100*np.mean((gains==0)|(gains==255)):.1f}% of DICT tiles; |alpha|>=32: {100*np.mean(np.abs(gains.astype(int)-128)>=64):.1f}%")
