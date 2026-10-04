#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
====================================================================
 [OPTO-CODEC V5 ENGINE - MULTIPROCESSING + DICTIONARY & CARRIER]
 Codebook VQ (256 Atoms) + Carrier Wave Modulation + ZSTD L12
====================================================================
"""

import os
import sys
import time
import struct
import concurrent.futures
import multiprocessing
import numpy as np
import cv2
import zstandard as zstd
from numba import njit

TAG_SKIP_RUN        = 240
TAG_MOTION          = 251
TAG_DICT_WIN        = 253  # Zastępuje stary CDF_WIN (Codebook + Carrier)
TAG_KEYFRAME        = 254

MAGIC_HEADER = b"OPTO50"

# ==================================================================
# DETERMINISTYCZNY GENERATOR SŁOWNIKA ATOMÓW (256 ATOMÓW 8x8)
# ==================================================================

def build_universal_codebook(atom_size=8, num_atoms=256, seed=42):
    """
    Tworzy znormalizowaną bazę wzorców tekstur/krawędzi (256 atomów 8x8).
    Stałe ziarno deterministycznie odtwarza słownik po stronie dekodera.
    """
    np.random.seed(seed)
    codebook = np.zeros((num_atoms, atom_size, atom_size), dtype=np.float32)
    idx = 0

    # 1. Bazy DCT (krawędzie i częstotliwości)
    for u in range(4):
        for v in range(4):
            if idx >= num_atoms: break
            for x in range(atom_size):
                for y in range(atom_size):
                    codebook[idx, y, x] = np.cos((2*x + 1)*u*np.pi / 16.0) * np.cos((2*y + 1)*v*np.pi / 16.0)
            norm = np.linalg.norm(codebook[idx])
            if norm > 0: codebook[idx] /= norm
            idx += 1

    # 2. Gradienty kątowe (32 kierunki)
    angles = np.linspace(0, 2*np.pi, 32, endpoint=False)
    for ang in angles:
        if idx >= num_atoms: break
        dx, dy = np.cos(ang), np.sin(ang)
        for x in range(atom_size):
            for y in range(atom_size):
                codebook[idx, y, x] = x * dx + y * dy
        codebook[idx] -= np.mean(codebook[idx])
        norm = np.linalg.norm(codebook[idx])
        if norm > 0: codebook[idx] /= norm
        idx += 1

    # 3. Szum i losowe bazy ortogonalne dla pozostałych atomów
    while idx < num_atoms:
        rand_atom = np.random.randn(atom_size, atom_size).astype(np.float32)
        rand_atom -= np.mean(rand_atom)
        norm = np.linalg.norm(rand_atom)
        if norm > 0: rand_atom /= norm
        codebook[idx] = rand_atom
        idx += 1

    return codebook

# ==================================================================
# NUMBA JIT & CARRIER MODULATION
# ==================================================================

@njit(fastmath=True)
def calc_sad_y_channel(blk1_y, blk2_y):
    sad_sum = 0.0
    h, w = blk1_y.shape
    for i in range(h):
        for j in range(w):
            diff = int(blk1_y[i, j]) - int(blk2_y[i, j])
            sad_sum += diff if diff >= 0 else -diff
    return sad_sum / float(h * w)


@njit(fastmath=True)
def match_atom_and_carrier(block_patch, codebook):
    """
    Szybkie wyznaczanie najlepszego atomu oraz skwantowanych parametrów nośnej (Gain, DC).
    """
    dc_offset = 0.0
    for r in range(8):
        for c in range(8):
            dc_offset += block_patch[r, c]
    dc_offset /= 64.0

    best_idx = 0
    best_err = 1e9
    best_alpha = 0.0

    num_atoms = codebook.shape[0]
    for i in range(num_atoms):
        dot_prod = 0.0
        for r in range(8):
            for c in range(8):
                dot_prod += (block_patch[r, c] - dc_offset) * codebook[i, r, c]

        err = 0.0
        for r in range(8):
            for c in range(8):
                diff = (block_patch[r, c] - dc_offset) - dot_prod * codebook[i, r, c]
                err += diff * diff

        if err < best_err:
            best_err = err
            best_idx = i
            best_alpha = dot_prod

    q_dc = min(255, max(0, int(dc_offset)))
    q_gain = min(255, max(0, int(best_alpha * 2.0 + 128.0)))
    return np.uint8(best_idx), np.uint8(q_gain), np.uint8(q_dc)

# ==================================================================
# GOP WORKER FUNCTION
# ==================================================================

def encode_gop_v5_worker(gop_frames, start_frame_idx, block_size, threshold, deadband, codebook):
    num_frames, height, width, _ = gop_frames.shape
    chunk_bytes = bytearray()
    stats = {'SKIP': 0, 'MOTION': 0, 'DICT_WIN': 0}

    prev_frame = None

    for local_idx in range(num_frames):
        global_idx = start_frame_idx + local_idx
        frame = gop_frames[local_idx]
        is_keyframe = (local_idx == 0)

        frame_bytes = bytearray()
        if is_keyframe:
            frame_bytes.append(TAG_KEYFRAME)
            frame_bytes.extend(struct.pack(">I", global_idx))

        skip_run = 0

        for y in range(0, height - block_size + 1, block_size):
            for x in range(0, width - block_size + 1, block_size):
                curr_blk = frame[y:y+block_size, x:x+block_size]

                # 1. Test SKIP
                if prev_frame is not None and not is_keyframe:
                    ref_blk = prev_frame[y:y+block_size, x:x+block_size]
                    sad_y = calc_sad_y_channel(curr_blk[:, :, 0], ref_blk[:, :, 0])
                    if sad_y <= deadband:
                        skip_run += 1
                        stats['SKIP'] += 1
                        if skip_run == 255:
                            frame_bytes.append(TAG_SKIP_RUN)
                            frame_bytes.append(skip_run)
                            skip_run = 0
                        continue

                if skip_run > 0:
                    frame_bytes.append(TAG_SKIP_RUN)
                    frame_bytes.append(skip_run)
                    skip_run = 0

                # 2. Keyframe
                if is_keyframe or prev_frame is None:
                    frame_bytes.append(TAG_DICT_WIN)
                    y_channel = curr_blk[:, :, 0]
                    for row in range(2):
                        for col in range(2):
                            sub_patch = cv2.resize(y_channel[row*16:(row+1)*16, col*16:(col+1)*16], (8, 8)).astype(np.float32)
                            idx, gain, dc = match_atom_and_carrier(sub_patch, codebook)
                            frame_bytes.extend([idx, gain, dc])
                    stats['DICT_WIN'] += 1
                    continue

                # 3. Szukanie ruchu (Diamond Search)
                best_dx, best_dy = 0, 0
                best_sad = calc_sad_y_channel(curr_blk[:, :, 0], ref_blk[:, :, 0])

                search_offsets = [(-16,0), (16,0), (0,-16), (0,16), (-8,-8), (8,8), (-8,8), (8,-8)]
                for dx, dy in search_offsets:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx <= width - block_size and 0 <= ny <= height - block_size:
                        sad = calc_sad_y_channel(curr_blk[:, :, 0], prev_frame[ny:ny+block_size, nx:nx+block_size, 0])
                        if sad < best_sad:
                            best_sad = sad
                            best_dx, best_dy = dx, dy

                if best_sad <= threshold:
                    frame_bytes.append(TAG_MOTION)
                    frame_bytes.extend(struct.pack("b b", best_dx, best_dy))
                    stats['MOTION'] += 1
                    continue

                # 4. Fallback Słownikowy (12 Bajtów na Blok 32x32)
                frame_bytes.append(TAG_DICT_WIN)
                y_channel = curr_blk[:, :, 0]
                for row in range(2):
                    for col in range(2):
                        sub_patch = cv2.resize(y_channel[row*16:(row+1)*16, col*16:(col+1)*16], (8, 8)).astype(np.float32)
                        idx, gain, dc = match_atom_and_carrier(sub_patch, codebook)
                        frame_bytes.extend([idx, gain, dc])
                stats['DICT_WIN'] += 1

        if skip_run > 0:
            frame_bytes.append(TAG_SKIP_RUN)
            frame_bytes.append(skip_run)

        prev_frame = frame
        chunk_bytes.extend(frame_bytes)

    return start_frame_idx, chunk_bytes, stats

# ==================================================================
# KLASA ENGINU OPTO-CODEC V5
# ==================================================================

class OptoCodecEngineV5:
    def __init__(self, block_size=32, deadband=36.0, threshold=110.0, zstd_level=12, gop=120):
        self.block_size = block_size
        self.deadband = deadband
        self.threshold = threshold
        self.zstd_level = zstd_level
        self.gop = gop
        
        # Konstruktor generuje deterministyczny słownik i konfigurowalne konteksty
        self.codebook = build_universal_codebook(atom_size=8, num_atoms=256, seed=42)
        self.cctx = zstd.ZstdCompressor(level=self.zstd_level)
        self._raw_stream = bytearray()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()

    def close(self):
        """Deterministyczny destruktor zwalniający bufory surowe."""
        if hasattr(self, '_raw_stream') and self._raw_stream is not None:
            self._raw_stream.clear()
            self._raw_stream = None

    def _load_video(self, input_file):
        cap = cv2.VideoCapture(input_file)
        if not cap.isOpened():
            raise RuntimeError(f"Nie można otworzyć: {input_file}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        frames_yuv = []
        while True:
            ret, frame = cap.read()
            if not ret: break
            yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
            frames_yuv.append(yuv)
        cap.release()

        return np.array(frames_yuv, dtype=np.uint8), width, height, fps, len(frames_yuv)

    def encode(self, input_file, output_file):
        src_size_mb = os.path.getsize(input_file) / (1024 * 1024)
        target_size_mb = src_size_mb / 3.0
        num_cpus = multiprocessing.cpu_count()

        print("============================================================")
        print(" [OPTO-CODEC V5 ENGINE - DICTIONARY & CARRIER CODEC]")
        print("============================================================")
        print(f" Plik źródłowy:         {input_file} ({src_size_mb:.2f} MB)")
        print(f" Rozmiar Bloku:         {self.block_size}x{self.block_size} px")
        print(f" Deadband / Threshold:  {self.deadband} / {self.threshold}")
        print(f" Słownik Atomów:        256 Wzorców (12 B / Blok 32x32)")
        print(f" ZSTD Kompresja:        Level {self.zstd_level}")
        print("============================================================")

        t0 = time.time()
        video_yuv, width, height, fps, total_frames = self._load_video(input_file)
        print(f" Załadowano {total_frames} klatek w {time.time() - t0:.2f} s.")

        gop_chunks = [(video_yuv[i:i+self.gop], i) for i in range(0, total_frames, self.gop)]

        self._raw_stream = bytearray()
        file_header = struct.pack(">6sHHHIHB", MAGIC_HEADER, width, height, int(fps * 100), total_frames, self.gop, self.block_size)
        self._raw_stream.extend(file_header)

        global_stats = {'SKIP': 0, 'MOTION': 0, 'DICT_WIN': 0}
        results = []

        with concurrent.futures.ProcessPoolExecutor(max_workers=num_cpus) as executor:
            futures = [
                executor.submit(encode_gop_v5_worker, chunk, start_idx, self.block_size, self.threshold, self.deadband, self.codebook)
                for chunk, start_idx in gop_chunks
            ]
            for future in concurrent.futures.as_completed(futures):
                start_idx, chunk_bytes, chunk_stats = future.result()
                results.append((start_idx, chunk_bytes))
                for k, v in chunk_stats.items():
                    global_stats[k] += v

        # Chronologiczne scalenie GOP
        results.sort(key=lambda x: x[0])
        for _, chunk_bytes in results:
            self._raw_stream.extend(chunk_bytes)

        print(f" Rozmiar surowego strumienia: {len(self._raw_stream) / (1024*1024):.2f} MB")
        print(f" Kompresja ZSTD Level {self.zstd_level}...")

        compressed_payload = self.cctx.compress(bytes(self._raw_stream))

        with open(output_file, "wb") as f:
            f.write(compressed_payload)

        out_size_mb = os.path.getsize(output_file) / (1024 * 1024)
        bitrate_kbps = (os.path.getsize(output_file) * 8) / (total_frames / fps) / 1000.0

        print("\n============================================================")
        print(" [OPTO-CODEC V5 PODSUMOWANIE]")
        print("============================================================")
        print(f" Rozmiar Wyjściowy:     {out_size_mb:.2f} MB")
        print(f" Bitrate Wyjściowy:     {bitrate_kbps:.2f} kbps ({bitrate_kbps/1000:.2f} Mbps)")
        print(f" Stosunek vs MP4 Input: {(src_size_mb / out_size_mb):.2f}x mniejszy")
        print(f" Podział Bloków:        SKIP: {global_stats['SKIP']} | MOTION: {global_stats['MOTION']} | DICT: {global_stats['DICT_WIN']}")
        print(f" Status Celu (3x):      {'OSIĄGNIĘTY!' if out_size_mb <= target_size_mb else 'Niespełniony'}")
        print("============================================================")

if __name__ == "__main__":
    with OptoCodecEngineV5(block_size=32, deadband=36.0, threshold=110.0, zstd_level=12) as engine:
        engine.encode("inputfile.mp4", "stream_v5.zst")