# ==================================================================
# OPTO CORE  –  wspólny rdzeń enkodera i dekodera (OPTO50 / OPTO51 / OPTO52)
#   tryby bloków 32x32: SKIP / MOTION / DICT / RES   (RES tylko w OPTO52, enkoder V5.5)
# ==================================================================
import struct
import numpy as np
import cv2

# ------------------------------------------------------------------
# Stałe protokołu
# ------------------------------------------------------------------
BLOCK = 32
TILE  = 16
ATOM  = 8
WIN   = 16          # rozmiar okna feather (atom 8x8 → 16x16)

MODE_SKIP   = 0
MODE_MOTION = 1
MODE_DICT   = 2
MODE_RES    = 3          # MC + korekta DC (int8, ze znakiem) + atomy na reszcie  (OPTO52)

TAG_SKIP_RUN = 240
TAG_MOTION   = 251
TAG_RES      = 252       # dx, dy (int8) + payload jak w DICT_WIN
TAG_DICT_WIN = 253
TAG_KEYFRAME = 254

MAGIC_V50 = b"OPTO50"
MAGIC_V51 = b"OPTO51"
MAGIC_V52 = b"OPTO52"

HEADER_FMT = ">6sHHHIHB"          # magic, w, h, f100, total, gop, block_size
HEADER_LEN = struct.calcsize(HEADER_FMT)
EXT_FMT    = ">BB"                # K, flags  (tylko OPTO52)
EXT_LEN    = struct.calcsize(EXT_FMT)

# ------------------------------------------------------------------
# Słownik atomów (identyczny z orig_enc / orig_dec, seed=42)
# ------------------------------------------------------------------
def build_codebook(atom_size=8, num_atoms=256, seed=42):
    np.random.seed(seed)
    cb = np.zeros((num_atoms, atom_size, atom_size), np.float32)
    idx = 0
    # 1. DCT
    for u in range(4):
        for v in range(4):
            if idx >= num_atoms:
                break
            for x in range(atom_size):
                for y in range(atom_size):
                    cb[idx, y, x] = (np.cos((2 * x + 1) * u * np.pi / 16.0) *
                                     np.cos((2 * y + 1) * v * np.pi / 16.0))
            n = np.linalg.norm(cb[idx])
            if n > 0:
                cb[idx] /= n
            idx += 1
    # 2. Gradienty
    for ang in np.linspace(0, 2 * np.pi, 32, endpoint=False):
        if idx >= num_atoms:
            break
        dx, dy = np.cos(ang), np.sin(ang)
        for x in range(atom_size):
            for y in range(atom_size):
                cb[idx, y, x] = x * dx + y * dy
        cb[idx] -= cb[idx].mean()
        n = np.linalg.norm(cb[idx])
        if n > 0:
            cb[idx] /= n
        idx += 1
    # 3. Szum ortogonalny
    while idx < num_atoms:
        a = np.random.randn(atom_size, atom_size).astype(np.float32)
        a -= a.mean()
        n = np.linalg.norm(a)
        if n > 0:
            a /= n
        cb[idx] = a
        idx += 1
    return cb


def generate_feather_window(size=16, margin=3):
    w1d = np.ones(size, np.float32)
    ramp = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, margin)))
    w1d[:margin] = ramp
    w1d[-margin:] = ramp[::-1]
    return np.outer(w1d, w1d)


_CODEBOOK = build_codebook()
_FEATHER  = generate_feather_window(WIN, 3)

# Okno feather ma zera na skrajnych pikselach ([0, .5, 1 … 1, .5, 0]). Przy stride == TILE (kafle się nie nakładają)
# normalizacja przez wsum kasuje okno wszędzie POZA tymi zerami: AC znika na 1-px ramce każdego kafla 16x16
# (60 z 256 px = 23%) -> widoczna kratka i sufit PSNR(Y) ~24 dB nawet przy nieskończenie wielu atomach.
# FEATHER_FLOOR > 0 usuwa zera (dla nienakładających się kafli wynik = czysty atom).
# 0.0 = stare zachowanie. Enkoder i dekoder importują ten sam moduł, więc wartość jest wspólna;
# strumienie zakodowane przy innej wartości dekodują się z lekkim dryfem.
FEATHER_FLOOR = 0.05
_WIN2     = np.maximum(_FEATHER, FEATHER_FLOOR)      # używane w overlap-add


# ------------------------------------------------------------------
# Kwantyzacja gainu  (α → 0..255, skala 2.0 jak w orig)
# ------------------------------------------------------------------
def quantize_gain(alpha):
    """alpha (float lub ndarray) → uint8 0..255"""
    q = np.rint(np.asarray(alpha) * 2.0 + 128.0)
    return np.clip(q, 0, 255).astype(np.uint8)


def dequantize_gain(q):
    """uint8 → float α"""
    return (np.asarray(q, np.float32) - 128.0) * 0.5


# ------------------------------------------------------------------
# Profile – zależny od magica (padding, słownik, tabela gain)
# ------------------------------------------------------------------
class Profile:
    def __init__(self, magic):
        if isinstance(magic, str):
            magic = magic.encode()
        self.magic = magic
        self.cb = _CODEBOOK                                   # (256, 8, 8)
        # atom 8x8 → 16x16 bicubic (jak w orig_dec)
        self.cb_ext = np.stack([
            cv2.resize(a, (WIN, WIN), interpolation=cv2.INTER_CUBIC)
            for a in self.cb
        ])                                                    # (256, 16, 16)
        self.cb_flat = self.cb.reshape(256, -1)               # do matching pursuit
        # tabela dekwantyzacji gainu
        self.gain = dequantize_gain(np.arange(256))           # (256,)

    def coded_dims(self, w, h):
        """OPTO50: obcięcie w dół do wielokrotności BLOCK.
           OPTO51/52: padding w górę do wielokrotności BLOCK."""
        if self.magic == MAGIC_V50:
            return (w // BLOCK) * BLOCK, (h // BLOCK) * BLOCK
        # V51 / V52 – pad
        cw = ((w + BLOCK - 1) // BLOCK) * BLOCK
        ch = ((h + BLOCK - 1) // BLOCK) * BLOCK
        return cw, ch


# ------------------------------------------------------------------
# Pomocnicze operacje na siatce
# ------------------------------------------------------------------
def tile_means(img):
    """Średnie 16x16 → siatka (H/16, W/16) float32"""
    H, W = img.shape
    return img.reshape(H // TILE, TILE, W // TILE, TILE).mean(axis=(1, 3)).astype(np.float32)


def upsample_dc(dc_grid, H, W):
    """Siatka DC (gh,gw) → pełna rozdzielczość bicubic (H,W)"""
    return cv2.resize(dc_grid.astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)


def overlap_add(tiles, gh, gw):
    """
    tiles: (gh, gw, WIN, WIN)  – atomy z oknem feather
    Zwraca (gh*TILE, gw*TILE). WIN==TILE==16 → bez przestrzennego overlapu.
    """
    stride = TILE
    out_h = (gh - 1) * stride + WIN
    out_w = (gw - 1) * stride + WIN
    out = np.zeros((out_h, out_w), np.float32)
    for iy in range(gh):
        for ix in range(gw):
            y0, x0 = iy * stride, ix * stride
            out[y0:y0 + WIN, x0:x0 + WIN] += tiles[iy, ix]
    return out[:gh * TILE, :gw * TILE]


def dict_mask(dict_blocks, H, W):
    """
    Miękka maska dla bloków DICT.
    dict_blocks: (nby, nbx) bool  – bloki 32x32
    Zwraca (H, W) float32 ∈ [0,1]
    """
    hard = np.repeat(np.repeat(dict_blocks.astype(np.float32), 2, axis=0), 2, axis=1)
    mask = cv2.resize(hard, (W, H), interpolation=cv2.INTER_LINEAR)
    return np.clip(mask, 0.0, 1.0)


def _blend(cur, X, M):
    """
    cur + M*(X - cur), ale DOKŁADNIE X tam, gdzie M == 1.
    W float32 `cur + (X - cur)` bywa o 1 ulp od X zależnie od cur; przy dokładnych remisach x.5 (bikubik z całkowitych
    węzłów DC) przestawia to rint(). Keyframe w enkoderze liczy się od prev=128, a w dekoderze od ostatniej klatki
    poprzedniego GOP-u - bez tego dekoder != rekonstrukcja enkodera na pojedynczych pikselach (--verify: BŁĄD).
    """
    return np.where(M >= 1.0, X, cur + M * (X - cur))


# ------------------------------------------------------------------
# Kopiowanie bloków (SKIP / MOTION)
# ------------------------------------------------------------------
def apply_copies(prev, mode, mv):
    """
    prev : (H, W) uint8
    mode : (nby, nbx)  MODE_*
    mv   : (nby, nbx, 2) int8  (dx, dy)
    """
    H, W = prev.shape
    nby, nbx = mode.shape
    out = prev.copy()
    for by in range(nby):
        for bx in range(nbx):
            m = mode[by, bx]
            if m == MODE_DICT or m == MODE_SKIP:
                continue
            y0, x0 = by * BLOCK, bx * BLOCK
            dx, dy = int(mv[by, bx, 0]), int(mv[by, bx, 1])
            sy = max(0, min(H - BLOCK, y0 + dy))
            sx = max(0, min(W - BLOCK, x0 + dx))
            out[y0:y0 + BLOCK, x0:x0 + BLOCK] = prev[sy:sy + BLOCK, sx:sx + BLOCK]
    return out


# ------------------------------------------------------------------
# Parsowanie strumienia
# ------------------------------------------------------------------
def parse_frame(raw, pos, nby, nbx, plen=None):
    """
    Parsuje jedną klatkę.
    plen – długość payloadu bloku DICT (domyślnie 12 = K=1).
    Zwraca: nowy_pos, is_keyframe, mode, mv, pay
    """
    if plen is None:
        plen = 12
    mode = np.full((nby, nbx), MODE_SKIP, np.uint8)
    mv   = np.zeros((nby, nbx, 2), np.int8)
    pay  = np.zeros((nby, nbx, plen), np.uint8)
    key  = False
    i = 0
    total = nby * nbx
    while i < total:
        if pos >= len(raw):
            break
        tag = raw[pos]
        pos += 1
        if tag == TAG_KEYFRAME:
            pos += 4
            key = True
            continue
        if tag == TAG_SKIP_RUN:
            run = raw[pos]
            pos += 1
            i += run
            continue
        by, bx = divmod(i, nbx)
        if tag == TAG_MOTION:
            dx, dy = struct.unpack("bb", raw[pos:pos + 2])
            pos += 2
            mode[by, bx] = MODE_MOTION
            mv[by, bx] = (dx, dy)
            i += 1
        elif tag == TAG_RES:
            dx, dy = struct.unpack("bb", raw[pos:pos + 2])
            pos += 2
            mode[by, bx] = MODE_RES
            mv[by, bx] = (dx, dy)
            pay[by, bx] = np.frombuffer(raw[pos:pos + plen], np.uint8)
            pos += plen
            i += 1
        elif tag == TAG_DICT_WIN:
            mode[by, bx] = MODE_DICT
            pay[by, bx] = np.frombuffer(raw[pos:pos + plen], np.uint8)
            pos += plen
            i += 1
        else:
            raise ValueError(f"nieznany tag {tag} przy pos={pos - 1}")
    return pos, key, mode, mv, pay


# ------------------------------------------------------------------
# Payload ↔ kafelki (OPTO50/51 – K=1)
# ------------------------------------------------------------------
def tiles_from_payload(pay):
    """pay (nby,nbx,12) → idx, gain, dc  każdy (gh,gw)"""
    nby, nbx, _ = pay.shape
    t = pay.reshape(nby, nbx, 2, 2, 3).transpose(0, 2, 1, 3, 4).reshape(nby * 2, nbx * 2, 3)
    return t[..., 0], t[..., 1], t[..., 2]


def payload_from_tiles(idx, gain, dc):
    gh, gw = idx.shape
    t = np.stack([idx, gain, dc], axis=-1)
    return t.reshape(gh // 2, 2, gw // 2, 2, 3).transpose(0, 2, 1, 3, 4).reshape(gh // 2, gw // 2, 12)


# ------------------------------------------------------------------
# OPTO52 – K atomów + opcjonalna chroma
# ------------------------------------------------------------------
def rec_len(K, chroma):
    return 3 + 2 * (K - 1) + (2 if chroma else 0)


def tiles_from_payload_x(pay, K, chroma):
    """pay (nby,nbx,4*rec) → (dcs, idx (K,gh,gw), gain (K,gh,gw))"""
    nby, nbx, _ = pay.shape
    rec = rec_len(K, chroma)
    t = pay.reshape(nby, nbx, 2, 2, rec).transpose(0, 2, 1, 3, 4).reshape(nby * 2, nbx * 2, rec)
    idx = np.stack([t[..., 0]] + [t[..., 3 + 2 * (k - 1)] for k in range(1, K)])
    gain = np.stack([t[..., 1]] + [t[..., 4 + 2 * (k - 1)] for k in range(1, K)])
    dcs = [t[..., 2]]
    if chroma:
        dcs += [t[..., 3 + 2 * (K - 1)], t[..., 4 + 2 * (K - 1)]]
    return dcs, idx, gain


def payload_from_tiles_x(dcs, idx, gain):
    K = idx.shape[0]
    gh, gw = dcs[0].shape
    fields = [idx[0], gain[0], dcs[0]]
    for k in range(1, K):
        fields += [idx[k], gain[k]]
    if len(dcs) == 3:
        fields += [dcs[1], dcs[2]]
    t = np.stack(fields, axis=-1)
    return t.reshape(gh // 2, 2, gw // 2, 2, -1).transpose(0, 2, 1, 3, 4).reshape(gh // 2, gw // 2, -1)


# ------------------------------------------------------------------
# Synteza
# ------------------------------------------------------------------
def synth_frame(cur, dc_grid, mode, idx, gain, prof):
    """OPTO50/51 – jeden atom na kafel."""
    return synth_atoms(cur, dc_grid, mode, idx[np.newaxis], gain[np.newaxis], prof)


def _ac_layer(is_tile, aidx, again, prof, H, W):
    """AC z K atomów na kaflach is_tile (gh,gw) bool -> (H,W) float32 (okno feather kompensowane sumą wag)."""
    gh, gw = is_tile.shape
    T = np.zeros((gh, gw, WIN, WIN), np.float32)
    for k in range(aidx.shape[0]):
        alpha = prof.gain[again[k]] * is_tile
        T += prof.cb_ext[aidx[k]] * alpha[:, :, None, None]
    T *= _WIN2
    ac = overlap_add(T, gh, gw)
    wsum = overlap_add(is_tile[:, :, None, None].astype(np.float32) * _WIN2, gh, gw)
    ac = ac / np.maximum(wsum, 1e-3)
    return ac[:H, :W]


def res_layer(delta, res_blocks, aidx, again, prof, H, W):
    """
    Korekta bloków RES, DODAWANA do kopii po kompensacji ruchu:
        M * ( powierzchnia DC z delta (ze znakiem) + atomy AC )
    delta      : (gh, gw) poprawka DC kafla (ze znakiem; poza RES ignorowana)
    res_blocks : (nby, nbx) bool – bloki 32x32 w trybie RES
    aidx/again : (K, gh, gw) albo None (plany chrominancji: tylko DC)
    Zwraca (H, W) float32.
    """
    is_res_tile = np.repeat(np.repeat(res_blocks, 2, axis=0), 2, axis=1)
    D = upsample_dc(np.where(is_res_tile, delta, 0.0).astype(np.float32), H, W)
    if aidx is not None:
        D = D + _ac_layer(is_res_tile, aidx, again, prof, H, W)
    return dict_mask(res_blocks, H, W) * D


def synth_atoms(cur, dc_grid, mode, aidx, again, prof):
    """K atomów na kafel (aidx/again: (K,gh,gw)). K=1 ≡ synth_frame."""
    H, W = cur.shape[:2]
    dict_blocks = (mode == MODE_DICT)
    if not dict_blocks.any():
        return cur.astype(np.uint8)
    is_dict_tile = np.repeat(np.repeat(dict_blocks, 2, axis=0), 2, axis=1)
    S = upsample_dc(dc_grid, H, W)
    ac = _ac_layer(is_dict_tile, aidx, again, prof, H, W)
    M = dict_mask(dict_blocks, H, W)
    out = _blend(cur.astype(np.float32), S + ac, M)
    return np.rint(np.clip(out, 0, 255)).astype(np.uint8)


def synth_dc(cur, dc_grid, mode):
    """Plan chrominancji: tylko powierzchnia DC."""
    H, W = cur.shape
    dict_blocks = (mode == MODE_DICT)
    if not dict_blocks.any():
        return cur.astype(np.uint8)
    S = upsample_dc(dc_grid, H, W)
    M = dict_mask(dict_blocks, H, W)
    return np.rint(np.clip(_blend(cur.astype(np.float32), S, M), 0, 255)).astype(np.uint8)


def synth_planes(prev, mode, mv, dcs, aidx, again, prof):
    """Rekonstrukcja całej klatki (Y + opcjonalnie Cr,Cb).
    DICT: dcs = DC absolutne (uint8). RES: dcs = poprawka DC względem kopii po MC (int8 w bajcie)."""
    is_dict_tile = np.repeat(np.repeat(mode == MODE_DICT, 2, axis=0), 2, axis=1)
    res_blocks = (mode == MODE_RES)
    has_res = bool(res_blocks.any())
    out = []
    for p in range(len(prev)):
        cur = apply_copies(prev[p], mode, mv)               # SKIP/DICT: bez zmian, MOTION i RES: kopia z wektorem
        grid = np.where(is_dict_tile, dcs[p].astype(np.float32), tile_means(cur))
        y = (synth_atoms(cur, grid, mode, aidx, again, prof) if p == 0
             else synth_dc(cur, grid, mode))
        if has_res:
            H, W = y.shape
            delta = dcs[p].view(np.int8).astype(np.float32)
            lay = res_layer(delta, res_blocks, aidx if p == 0 else None, again, prof, H, W)
            y = np.rint(np.clip(y.astype(np.float32) + lay, 0, 255)).astype(np.uint8)
        out.append(y)
    return out


# ------------------------------------------------------------------
# Nagłówek + generator klatek
# ------------------------------------------------------------------
def read_header(raw):
    magic, w, h, f100, total, gop, bs = struct.unpack_from(HEADER_FMT, raw, 0)
    pos, K, chroma = HEADER_LEN, 1, False
    if magic == MAGIC_V52:
        K, flags = struct.unpack_from(EXT_FMT, raw, pos)
        pos += EXT_LEN
        chroma = bool(flags & 1)
    return dict(magic=magic, w=w, h=h, f100=f100, total=total, gop=gop,
                K=K, chroma=chroma, pos=pos)


def fit_plane(p, w, h):
    p = p[:h, :w]
    if p.shape != (h, w):
        return np.pad(p, ((0, h - p.shape[0]), (0, w - p.shape[1])), mode="edge")
    return p


def decode_frames(raw):
    """Generator klatek: krotka planów uint8 (h,w) – (Y,) albo (Y,Cr,Cb)."""
    hd = read_header(raw)
    prof = Profile(hd["magic"])
    w, h, K, chroma = hd["w"], hd["h"], hd["K"], hd["chroma"]
    cw, ch = prof.coded_dims(w, h)
    nby, nbx = ch // BLOCK, cw // BLOCK
    prev = [np.full((ch, cw), 128, np.uint8) for _ in range(3 if chroma else 1)]
    pos = hd["pos"]
    plen = 4 * rec_len(K, chroma)
    while pos < len(raw):
        pos, _key, mode, mv, pay = parse_frame(raw, pos, nby, nbx, plen)
        dcs, aidx, again = tiles_from_payload_x(pay, K, chroma)
        prev = synth_planes(prev, mode, mv, dcs, aidx, again, prof)
        yield tuple(fit_plane(p, w, h) for p in prev)
