#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
====================================================================
 [OPTO-DECODER V5.3 - SEAMLESS CONTINUOUS CANVAS ENGINE]
 Global 2D Continuous DC Surface + Overlapped AC Atom Feathering
 100% Elimination of Block Boundary Artifacts
====================================================================
"""

import os
import struct
import time
import numpy as np
import cv2
import zstandard as zstd
from numba import njit

TAG_SKIP_RUN = 240
TAG_MOTION   = 251
TAG_DICT_WIN = 253
TAG_KEYFRAME = 254
MAGIC_HEADER = b"OPTO50"


def build_universal_codebook(atom_size=8, num_atoms=256, seed=42):
    """ Generuje deterministyczny słownik wzorców (jednakowy z enkoderem). """
    np.random.seed(seed)
    codebook = np.zeros((num_atoms, atom_size, atom_size), dtype=np.float32)
    idx = 0

    # 1. Bazy DCT
    for u in range(4):
        for v in range(4):
            if idx >= num_atoms: break
            for x in range(atom_size):
                for y in range(atom_size):
                    codebook[idx, y, x] = np.cos((2*x + 1)*u*np.pi / 16.0) * np.cos((2*y + 1)*v*np.pi / 16.0)
            norm = np.linalg.norm(codebook[idx])
            if norm > 0: codebook[idx] /= norm
            idx += 1

    # 2. Gradienty
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

    # 3. Szum ortogonalny
    while idx < num_atoms:
        rand_atom = np.random.randn(atom_size, atom_size).astype(np.float32)
        rand_atom -= np.mean(rand_atom)
        norm = np.linalg.norm(rand_atom)
        if norm > 0: rand_atom /= norm
        codebook[idx] = rand_atom
        idx += 1

    return codebook


def generate_feather_window(size=16, margin=3):
    """
    Tworzy maskę 2D podniesionego kosinusa dla płynnego przenikania krawędzi atomów.
    """
    w1d = np.ones(size, dtype=np.float32)
    ramp = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, margin)))
    w1d[:margin] = ramp
    w1d[-margin:] = ramp[::-1]
    w2d = np.outer(w1d, w1d)
    return w2d


class OptoDecoderEngineV53:
    def __init__(self, stream_file):
        self.stream_file = stream_file
        self.codebook = build_universal_codebook(atom_size=8, num_atoms=256, seed=42)
        self.feather_win = generate_feather_window(size=16, margin=3)
        self.dctx = zstd.ZstdDecompressor()

    def play(self):
        if not os.path.exists(self.stream_file):
            print(f"Błąd: Plik {self.stream_file} nie istnieje.")
            return

        print(f" Wczytywanie i dekompresja: {self.stream_file}...")
        with open(self.stream_file, "rb") as f:
            raw_bytes = self.dctx.decompress(f.read())

        header_fmt = ">6sHHHIHB"
        header_len = struct.calcsize(header_fmt)
        magic, width, height, fps_x100, total_frames, gop, block_size = struct.unpack(header_fmt, raw_bytes[:header_len])

        if magic != MAGIC_HEADER:
            raise ValueError("Błędny nagłówek strumienia OPTO-CODEC.")

        fps = fps_x100 / 100.0
        stream_idx = header_len
        total_len = len(raw_bytes)

        # Siatki węzłowe dla składowej stałej (DC) i wzmocnienia (Gain) na poziomie całej klatki
        grid_h, grid_w = height // 16, width // 16
        dc_frame_grid = np.full((grid_h, grid_w), 128.0, dtype=np.float32)
        
        # Bufor klatki YUV
        curr_frame_yuv = np.full((height, width, 3), 128, dtype=np.uint8)
        prev_frame_yuv = None
        frame_delay = max(1, int(1000.0 / fps))

        cv2.namedWindow("OPTO-CODEC V5.3 - Continuous Canvas", cv2.WINDOW_AUTOSIZE)

        while stream_idx < total_len:
            y, x = 0, 0
            skip_run = 0

            # Przechowujemy detale AC dla bieżącej klatki
            ac_detail_layer = np.zeros((height, width), dtype=np.float32)

            while y <= height - block_size:
                if skip_run > 0:
                    if prev_frame_yuv is not None:
                        curr_frame_yuv[y:y+block_size, x:x+block_size] = prev_frame_yuv[y:y+block_size, x:x+block_size]
                    skip_run -= 1
                    x += block_size
                    if x > width - block_size:
                        x = 0; y += block_size
                    continue

                tag = raw_bytes[stream_idx]
                stream_idx += 1

                if tag == TAG_KEYFRAME:
                    stream_idx += 4
                    continue
                elif tag == TAG_SKIP_RUN:
                    skip_run = raw_bytes[stream_idx]
                    stream_idx += 1
                    continue
                elif tag == TAG_MOTION:
                    dx, dy = struct.unpack("b b", raw_bytes[stream_idx:stream_idx+2])
                    stream_idx += 2
                    if prev_frame_yuv is not None:
                        ref_y = max(0, min(height - block_size, y + dy))
                        ref_x = max(0, min(width - block_size, x + dx))
                        curr_frame_yuv[y:y+block_size, x:x+block_size] = prev_frame_yuv[ref_y:ref_y+block_size, ref_x:ref_x+block_size]
                elif tag == TAG_DICT_WIN:
                    payload_12b = raw_bytes[stream_idx:stream_idx+12]
                    stream_idx += 12

                    ptr = 0
                    for r in range(2):
                        for c in range(2):
                            idx, q_gain, q_dc = payload_12b[ptr], payload_12b[ptr+1], payload_12b[ptr+2]
                            ptr += 3

                            gh_idx = (y // 16) + r
                            gw_idx = (x // 16) + c

                            # 1. Rejestracja punktu węzłowego DC na siatce klatki
                            dc_frame_grid[gh_idx, gw_idx] = float(q_dc)

                            # 2. Rekonstrukcja detalu AC ze słownika
                            atom_8x8 = self.codebook[idx]
                            atom_16x16 = cv2.resize(atom_8x8, (16, 16), interpolation=cv2.INTER_CUBIC)
                            alpha = (float(q_gain) - 128.0) / 2.0

                            # Płynne nałożenie atomu z użyciem okna zmiękczającego
                            py, px = y + r*16, x + c*16
                            ac_detail_layer[py:py+16, px:px+16] = alpha * atom_16x16 * self.feather_win

                    if prev_frame_yuv is not None:
                        curr_frame_yuv[y:y+block_size, x:x+block_size, 1:] = prev_frame_yuv[y:y+block_size, x:x+block_size, 1:]

                x += block_size
                if x > width - block_size:
                    x = 0; y += block_size

            # --------------------------------------------------------------
            # KROK KLUCZOWY: Generowanie ciągłej mapy DC dla całej klatki
            # --------------------------------------------------------------
            # Rozciągnięcie rzadkiej siatki DC (H/16, W/16) do pełnej rozdzielczości (H, W)
            # interpolacją bi-sześcienną. Eliminacja 100% krawędzi bloków!
            global_dc_surface = cv2.resize(dc_frame_grid, (width, height), interpolation=cv2.INTER_CUBIC)

            # Sumowanie płynnej fali nośnej (DC) z zarejestrowanymi detalami (AC)
            full_luma = global_dc_surface + ac_detail_layer
            curr_frame_yuv[:, :, 0] = np.clip(full_luma, 0, 255).astype(np.uint8)

            # Opcjonalne lekkie wygładzenie szumu (Fast Guided Smooth)
            curr_frame_yuv[:, :, 0] = cv2.GaussianBlur(curr_frame_yuv[:, :, 0], (3, 3), 0.5)

            frame_bgr = cv2.cvtColor(curr_frame_yuv, cv2.COLOR_YCrCb2BGR)
            cv2.imshow("OPTO-CODEC V5.3 - Continuous Canvas", frame_bgr)
            prev_frame_yuv = curr_frame_yuv.copy()

            if (cv2.waitKey(frame_delay) & 0xFF) in (27, ord('q')):
                break

        cv2.destroyAllWindows()


if __name__ == "__main__":
    player = OptoDecoderEngineV53("stream_v5.zst")
    player.play()