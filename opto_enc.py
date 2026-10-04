#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
====================================================================
 [OPTO-ENC V5.5 - ENKODER ZAMKNIĘTEJ PĘTLI NA opto_core]
 - referencja = ZREKONSTRUOWANA poprzednia klatka (to samo co widzi
   dekoder: brak dryfu), rekonstrukcja z opto_core.synth_planes
 - słownik V5.1 (DCT + krawędzie), gain kompandowany, padding do 32 px
 - kolor: DC chrominancji (Cr, Cb) na siatce kafli 16x16
 - do K atomów na kafel (matching pursuit) przycinanych kryterium R-D
 - SKIP / MOTION / RES / DICT wybierane minimalizacją  SSE + lam * bajty
   (RES = kompensacja ruchu + dokodowanie reszty DC/atomami - kumuluje detal z klatki na klatkę)
 Format: OPTO52 (K=1 bez chromy == OPTO51)

 python3 opto_enc.py clip.avi out/new.zst [--lam 800] [--atoms 8] [--gop 60]
 python3 dec2mp4.py out/new.zst out/new.mp4
====================================================================
"""

import os
import time
import struct
import argparse
import multiprocessing
import concurrent.futures
import numpy as np
import cv2
import zstandard as zstd
import opto_core as oc

B = oc.BLOCK


# ------------------------------------------------------------------
# Wejście
# ------------------------------------------------------------------
def load_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Nie można otworzyć: {path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2YCrCb))      # kanały: Y, Cr, Cb
    cap.release()
    return np.array(frames, np.uint8), w, h, fps


def pad_planes(ycc, H, W):
    h, w = ycc.shape[:2]
    return [np.ascontiguousarray(np.pad(ycc[:, :, c], ((0, H - h), (0, W - w)), mode="edge")) for c in range(3)]


# ------------------------------------------------------------------
# Analiza atomów (matching pursuit + przycinanie R-D)
# ------------------------------------------------------------------
def match_atoms(src_f, dc_grid, is_dict_tile, prof, K, lam):
    """
    Do K atomów na kafel DICT, dopasowywanych do reszty względem TEJ SAMEJ powierzchni DC, którą zbuduje
    dekoder. Atom wchodzi, gdy jego zysk (ok. 4*alpha^2 w SSE pełnej rozdzielczości) pokrywa koszt 2 B:
        4*alpha^2 >= lam * 2     (lam = SSE na bajt)
    Zwraca idx (K,gh,gw), gain (K,gh,gw), used (gh,gw) - liczba użytych atomów na kafel.
    """
    H, W = src_f.shape
    gh, gw = is_dict_tile.shape
    S = oc.upsample_dc(dc_grid.astype(np.float32), H, W)
    Rd = (src_f - S).reshape(H // 2, 2, W // 2, 2).mean(axis=(1, 3))
    P = Rd.reshape(gh, oc.ATOM, gw, oc.ATOM).transpose(0, 2, 1, 3).reshape(gh * gw, oc.ATOM * oc.ATOM)
    sel = np.flatnonzero(is_dict_tile.reshape(-1))
    idx = np.zeros((K, gh * gw), np.uint8)
    gain = np.full((K, gh * gw), 128, np.uint8)
    used = np.zeros(gh * gw, np.int32)
    if len(sel):
        r = P[sel].astype(np.float32)
        ar = np.arange(len(sel))
        alive = np.ones(len(sel), bool)
        thr = lam * 2.0 / 4.0
        for k in range(K):
            dots = r @ prof.cb_flat.T
            j = np.argmax(np.abs(dots), axis=1)
            q = oc.quantize_gain(dots[ar, j])
            aq = prof.gain[q]
            alive &= (aq * aq >= thr)
            if not alive.any():
                break
            s = sel[alive]
            idx[k, s] = j[alive]
            gain[k, s] = q[alive]
            used[s] += 1
            r[alive] -= aq[alive, None] * prof.cb_flat[j[alive]]
    return idx.reshape(K, gh, gw), gain.reshape(K, gh, gw), used.reshape(gh, gw)


# ------------------------------------------------------------------
# Kodowanie klatki
# ------------------------------------------------------------------
def write_frame(key, frame_idx, mode, mv, pay):
    out = bytearray()
    if key:
        out.append(oc.TAG_KEYFRAME)
        out += struct.pack(">I", frame_idx)
    run = 0
    nby, nbx = mode.shape
    for by in range(nby):
        for bx in range(nbx):
            m = mode[by, bx]
            if m == oc.MODE_SKIP:
                run += 1
                if run == 255:
                    out += bytes((oc.TAG_SKIP_RUN, 255))
                    run = 0
                continue
            if run:
                out += bytes((oc.TAG_SKIP_RUN, run))
                run = 0
            if m == oc.MODE_MOTION:
                out.append(oc.TAG_MOTION)
                out += struct.pack("bb", int(mv[by, bx, 0]), int(mv[by, bx, 1]))
            elif m == oc.MODE_RES:
                out.append(oc.TAG_RES)
                out += struct.pack("bb", int(mv[by, bx, 0]), int(mv[by, bx, 1]))
                out += pay[by, bx].tobytes()
            else:
                out.append(oc.TAG_DICT_WIN)
                out += pay[by, bx].tobytes()
    if run:
        out += bytes((oc.TAG_SKIP_RUN, run))
    return out


def tile_mask(blocks):
    return np.repeat(np.repeat(blocks, 2, axis=0), 2, axis=1)


def block_sum(x):
    nby, nbx = x.shape[0] // B, x.shape[1] // B
    return x.reshape(nby, B, nbx, B).sum(axis=(1, 3))


def encode_frame(src, prev, key, prm, prof, frame_idx):
    """src: lista planów uint8 (H,W) [Y(,Cr,Cb)]; prev: lista planów poprzedniej REKONSTRUKCJI albo None."""
    K, chroma, R = prm["K"], prm["chroma"], prm["search"]
    lam = prm["lam"] * (prm["key_boost"] if key else 1.0)
    H, W = src[0].shape
    nby, nbx = H // B, W // B
    gh, gw = H // oc.TILE, W // oc.TILE
    nplanes = len(src)
    srcf = [s.astype(np.float32) for s in src]
    dc_src = [np.clip(np.rint(oc.tile_means(s)), 0, 255).astype(np.float32) for s in srcf]   # węzły DC ze źródła
    all_tiles = np.ones((gh, gw), bool)
    zmv = np.zeros((nby, nbx, 2), np.int8)
    extra = 2 if chroma else 0                                    # bajty chromy na kafel

    if key or prev is None:
        key = True
        prev = [np.full((H, W), 128, np.uint8) for _ in range(nplanes)]
        mode = np.full((nby, nbx), oc.MODE_DICT, np.uint8)
        mv = zmv
        idx, gain, used = match_atoms(srcf[0], dc_src[0], all_tiles, prof, K, lam)
        dcs = [d.astype(np.uint8) for d in dc_src]
        is_dict_tile, is_res_tile = all_tiles, ~all_tiles
    else:
        prevY = prev[0]

        # --- (a) kandydat DICT dla całej klatki: koszt błędu i bajtów do decyzji R-D
        idx_i, gain_i, used_i = match_atoms(srcf[0], dc_src[0], all_tiles, prof, K, lam)
        D = oc.synth_atoms(np.zeros((H, W), np.float32), dc_src[0], np.full((nby, nbx), oc.MODE_DICT, np.uint8),
                           idx_i, gain_i, prof)
        sse_d = block_sum((srcf[0] - D) ** 2)
        bytes_d = 1 + (1 + extra + 2 * used_i).reshape(nby, 2, nbx, 2).sum(axis=(1, 3))

        # --- (b) SKIP (wektor zerowy) i MOTION (pełne przeszukanie +-R na ZREKONSTRUOWANEJ klatce)
        sse_s = block_sum((srcf[0] - prevY.astype(np.float32)) ** 2)
        sse_m = np.full((nby, nbx), np.inf, np.float32)
        mv_best = zmv.copy()
        for by in range(nby):
            for bx in range(nbx):
                if sse_s[by, bx] == 0:
                    continue
                y0, x0 = by * B, bx * B
                ya, yb = max(0, y0 - R), min(H, y0 + B + R)
                xa, xb = max(0, x0 - R), min(W, x0 + B + R)
                res = cv2.matchTemplate(np.ascontiguousarray(prevY[ya:yb, xa:xb]), src[0][y0:y0 + B, x0:x0 + B], cv2.TM_SQDIFF)
                iy, ix = divmod(int(np.argmin(res)), res.shape[1])
                dy, dx = ya + iy - y0, xa + ix - x0
                mv_best[by, bx] = (dx, dy)
                if dx or dy:
                    sse_m[by, bx] = res[iy, ix]

        # --- (c) kandydat RES: najlepsze MC + znakowane DC poprawki + atomy na reszcie po MC
        cur_mc = oc.apply_copies(prevY, np.full((nby, nbx), oc.MODE_MOTION, np.uint8), mv_best)
        Rres = srcf[0] - cur_mc
        delta = np.clip(np.rint(oc.tile_means(Rres)), -127, 127).astype(np.float32)
        idx_r, gain_r, used_r = match_atoms(Rres, delta, all_tiles, prof, K, lam)
        rec_r = np.clip(np.rint(cur_mc + oc.res_layer(delta, np.ones((nby, nbx), bool), idx_r, gain_r, prof, H, W)), 0, 255)
        sse_r = block_sum((srcf[0] - rec_r) ** 2)
        bytes_r = 3 + (1 + extra + 2 * used_r).reshape(nby, 2, nbx, 2).sum(axis=(1, 3))

        # --- decyzja R-D:  SSE + lam * bajty
        J = np.stack([sse_s + lam * 0.1, sse_m + lam * 3.0, sse_r + lam * bytes_r, sse_d + lam * bytes_d])
        mode = np.array([oc.MODE_SKIP, oc.MODE_MOTION, oc.MODE_RES, oc.MODE_DICT], np.uint8)[np.argmin(J, axis=0)]
        has_mv = (mode == oc.MODE_MOTION) | (mode == oc.MODE_RES)
        mv = np.where(has_mv[:, :, None], mv_best, 0).astype(np.int8)

        # --- właściwa analiza dla wybranych bloków (referencja = kopia po MC, jak u dekodera)
        mc_mode = np.where(mode == oc.MODE_RES, oc.MODE_MOTION, mode).astype(np.uint8)
        is_dict_tile = tile_mask(mode == oc.MODE_DICT)
        is_res_tile = tile_mask(mode == oc.MODE_RES)
        cur = [oc.apply_copies(prev[p], mc_mode, mv) for p in range(nplanes)]
        grid0 = np.where(is_dict_tile, dc_src[0], oc.tile_means(cur[0]))
        idx_d, gain_d, used_d = match_atoms(srcf[0], grid0, is_dict_tile, prof, K, lam)
        Rres = srcf[0] - cur[0]
        delta0 = np.where(is_res_tile, np.clip(np.rint(oc.tile_means(Rres)), -127, 127), 0).astype(np.float32)
        idx_r, gain_r, used_r = match_atoms(Rres, delta0, is_res_tile, prof, K, lam)
        idx = np.where(is_dict_tile[None], idx_d, np.where(is_res_tile[None], idx_r, 0)).astype(np.uint8)
        gain = np.where(is_dict_tile[None], gain_d, np.where(is_res_tile[None], gain_r, 128)).astype(np.uint8)
        used = used_d + used_r
        dcs = []
        for p in range(nplanes):
            dl = delta0 if p == 0 else np.clip(np.rint(oc.tile_means(srcf[p] - cur[p])), -127, 127)
            b = np.zeros((gh, gw), np.uint8)
            b[is_dict_tile] = dc_src[p][is_dict_tile].astype(np.uint8)          # DICT: DC absolutne
            b[is_res_tile] = dl[is_res_tile].astype(np.int8).view(np.uint8)     # RES: poprawka DC (int8)
            dcs.append(b)

    pay = oc.payload_from_tiles_x(dcs, idx, gain)
    data = write_frame(key, frame_idx, mode, mv, pay)
    recon = oc.synth_planes(prev, mode, mv, dcs, idx, gain, prof)       # DOKŁADNIE to, co zrobi dekoder

    coded = is_dict_tile | is_res_tile
    st = dict(SKIP=int((mode == oc.MODE_SKIP).sum()), MOTION=int((mode == oc.MODE_MOTION).sum()),
              RES=int((mode == oc.MODE_RES).sum()), DICT=int((mode == oc.MODE_DICT).sum()),
              tiles=int(coded.sum()), atoms=int(used[coded].sum()), key=bool(key), bytes=len(data))
    return data, recon, st


def mse(a, b):
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))


def encode_gop(job):
    frames, start, prm, keep = job
    prof = oc.Profile(oc.MAGIC_V52)
    h, w = frames.shape[1:3]
    cw, ch = prof.coded_dims(w, h)
    nplanes = 3 if prm["chroma"] else 1
    out = bytearray()
    prev = None
    recon_keep = []
    tot = dict(SKIP=0, MOTION=0, RES=0, DICT=0, tiles=0, atoms=0, key_bytes=0, p_bytes=0)
    mses = [[] for _ in range(nplanes)]
    for i in range(len(frames)):
        src = pad_planes(frames[i], ch, cw)[:nplanes]
        data, prev, st = encode_frame(src, prev, i == 0, prm, prof, start + i)
        out += data
        for k in ("SKIP", "MOTION", "RES", "DICT", "tiles", "atoms"):
            tot[k] += st[k]
        tot["key_bytes" if st["key"] else "p_bytes"] += st["bytes"]
        for p in range(nplanes):
            mses[p].append(mse(frames[i][:, :, p], prev[p][:h, :w]))
        if keep:
            recon_keep.append(tuple(p[:h, :w].copy() for p in prev))
    return start, bytes(out), tot, mses, recon_keep


# ------------------------------------------------------------------
# Enkoder
# ------------------------------------------------------------------
def psnr_from_mse(m):
    return 99.0 if m <= 0 else 10 * np.log10(255.0 ** 2 / m)


def encode(path_in, path_out, prm, jobs=None, verify=False):
    t0 = time.time()
    src_mb = os.path.getsize(path_in) / (1024 * 1024)
    frames, w, h, fps = load_video(path_in)
    n = len(frames)
    gop = prm["gop"]
    chunks = [(frames[i:i + gop], i, prm, verify) for i in range(0, n, gop)]
    jobs = jobs or multiprocessing.cpu_count()

    print("============================================================")
    print(" [OPTO-ENC V5.5 - ZAMKNIĘTA PĘTLA, OPTO52]")
    print("============================================================")
    print(f" Plik źródłowy:  {path_in} ({src_mb:.2f} MB), {w}x{h}, {n} klatek, {fps:g} fps")
    print(f" lam={prm['lam']:g} (keyframe x{prm['key_boost']:g})  atomy<= {prm['K']}  "
          f"kolor={'tak' if prm['chroma'] else 'nie'}  GOP={gop}  przeszukiwanie +-{prm['search']} px")

    if jobs > 1 and len(chunks) > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=min(jobs, len(chunks))) as ex:
            results = list(ex.map(encode_gop, chunks))
    else:
        results = [encode_gop(c) for c in chunks]
    results.sort(key=lambda r: r[0])

    flags = 1 if prm["chroma"] else 0
    raw = bytearray(struct.pack(oc.HEADER_FMT, oc.MAGIC_V52, w, h, int(round(fps * 100)), n, gop, B))
    raw += struct.pack(oc.EXT_FMT, prm["K"], flags)
    tot = dict(SKIP=0, MOTION=0, RES=0, DICT=0, tiles=0, atoms=0, key_bytes=0, p_bytes=0)
    nplanes = 3 if prm["chroma"] else 1
    mses = [[] for _ in range(nplanes)]
    recon = []
    for _, data, t, ms, rk in results:
        raw += data
        for k in tot:
            tot[k] += t[k]
        for p in range(nplanes):
            mses[p] += ms[p]
        recon += rk
    comp = zstd.ZstdCompressor(level=prm["zstd"]).compress(bytes(raw))
    with open(path_out, "wb") as f:
        f.write(comp)

    out_mb = len(comp) / (1024 * 1024)
    kbps = len(comp) * 8 / (n / fps) / 1000.0
    names = ["Y", "Cr", "Cb"]
    print("------------------------------------------------------------")
    print(f" Wyjście:        {path_out}: {len(comp)} B ({out_mb:.3f} MB), {kbps:.0f} kbps, {src_mb / out_mb:.1f}x mniejszy")
    print(f" Bloki:          SKIP {tot['SKIP']} | MOTION {tot['MOTION']} | RES {tot['RES']} | DICT {tot['DICT']}"
          f"  (atomów/kafel DICT+RES: {tot['atoms'] / max(1, tot['tiles']):.2f})")
    print(f" Surowo:         keyframe'y {tot['key_bytes']} B | P-klatki {tot['p_bytes']} B")
    print(" PSNR vs źródło: " + "  ".join(f"{names[p]} {psnr_from_mse(np.mean(mses[p])):.2f} dB" for p in range(nplanes))
          + f"   (Y min/klatkę {min(psnr_from_mse(m) for m in mses[0]):.2f})")
    print(f" Czas:           {time.time() - t0:.1f} s")

    if verify:
        got = list(oc.decode_frames(zstd.ZstdDecompressor().decompress(comp)))
        ok = len(got) == len(recon) and all(all(np.array_equal(a, b) for a, b in zip(x, y)) for x, y in zip(got, recon))
        print(f" Weryfikacja:    dekoder == rekonstrukcja enkodera (bit w bit): {'OK' if ok else 'BŁĄD!'}")
        if not ok:
            raise SystemExit(1)
    print("============================================================")
    return dict(bytes=len(comp), psnr=[psnr_from_mse(np.mean(m)) for m in mses], tot=tot,
                psnr_y_frames=[psnr_from_mse(m) for m in mses[0]])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="OPTO-ENC V5.5 (zamknięta pętla, OPTO52)")
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--lam", type=float, default=800.0, help="koszt R-D: SSE na bajt (mniejsze = lepsza jakość, większy plik)")
    ap.add_argument("--key-boost", type=float, default=0.25, help="mnożnik lam dla keyframe'ów (<1 = więcej bitów na I-klatki)")
    ap.add_argument("--atoms", type=int, default=8, help="maks. atomów na kafel 16x16")
    ap.add_argument("--gop", type=int, default=60)
    ap.add_argument("--search", type=int, default=16, help="zakres szukania ruchu +-px (maks. 127)")
    ap.add_argument("--no-chroma", action="store_true", help="tylko luminancja (czarno-biały)")
    ap.add_argument("--zstd", type=int, default=19)
    ap.add_argument("--jobs", type=int, default=0, help="procesy (domyślnie: liczba CPU; GOP-y kodują się niezależnie)")
    ap.add_argument("--verify", action="store_true", help="po kodowaniu zdekoduj i porównaj bit w bit z rekonstrukcją enkodera")
    a = ap.parse_args()
    prm = dict(lam=a.lam, key_boost=a.key_boost, K=a.atoms, gop=a.gop, search=min(a.search, 127),
               chroma=not a.no_chroma, zstd=a.zstd)
    encode(a.input, a.output, prm, jobs=a.jobs or None, verify=a.verify)
