import sys, os, importlib.util
import numpy as np
import cv2


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def make_clip(path, w=640, h=360, n=36, fps=24):
    from skimage import data
    a = cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)             # 512x512
    c = cv2.cvtColor(data.coffee(), cv2.COLOR_RGB2BGR)                # 400x600
    c = cv2.resize(c, (768, 512), interpolation=cv2.INTER_CUBIC)
    canvas = np.concatenate([a, c], axis=1)                           # 512x1280
    cat = cv2.cvtColor(data.chelsea(), cv2.COLOR_RGB2BGR)
    cat = cv2.resize(cat, (150, 100), interpolation=cv2.INTER_AREA)
    yy, xx = np.mgrid[0:100, 0:150].astype(np.float32)
    alpha = np.clip(1.0 - (((xx - 75) / 75.0) ** 2 + ((yy - 50) / 50.0) ** 2) * 1.0, 0, 1)
    alpha = np.clip(alpha * 3.0, 0, 1)[:, :, None]
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'MJPG'), fps, (w, h))
    for t in range(n):
        x0, y0 = 20 + 2.5 * t, 90 + 1.0 * t
        M = np.float32([[1, 0, -x0], [0, 1, -y0]])
        fr = cv2.warpAffine(canvas, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        ox, oy = int(30 + 6 * t), int(120 + 40 * np.sin(t / 6.0))
        ox = min(ox, w - 150)
        roi = fr[oy:oy + 100, ox:ox + 150].astype(np.float32)
        fr[oy:oy + 100, ox:ox + 150] = (roi * (1 - alpha) + cat.astype(np.float32) * alpha).astype(np.uint8)
        vw.write(fr)
    vw.release()


def read_y(path):
    cap = cv2.VideoCapture(path)
    ys = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        ys.append(cv2.cvtColor(fr, cv2.COLOR_BGR2YCrCb)[:, :, 0])
    cap.release()
    return np.array(ys)


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def grid_score(img, period=16):
    """(max-min)/mean średniego gradientu w funkcji (x mod 16): ~0 = brak siatki, duże = widoczna siatka bloków"""
    f = img.astype(np.float32)
    dx = np.abs(np.diff(f, axis=1)).mean(axis=0)
    dy = np.abs(np.diff(f, axis=0)).mean(axis=1)
    def prof(d):
        n = len(d) // period * period
        p = d[:n].reshape(-1, period).mean(axis=0)
        return (p.max() - p.min()) / p.mean()
    return 0.5 * (prof(dx) + prof(dy))


def decode_orig(stream_path, dec_module):
    """Uruchamia ORYGINALNY dekoder bez GUI i przechwytuje klatki."""
    frames = []
    cv2.namedWindow = lambda *a, **k: None
    cv2.imshow = lambda name, img: frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)[:, :, 0].copy())
    cv2.waitKey = lambda d=0: -1
    cv2.destroyAllWindows = lambda: None
    dec_module.OptoDecoderEngineV53(stream_path).play()
    return np.array(frames)


def montage(rows, path, crop=None, scale=2, labels=None):
    """rows: lista list obrazów (gray) – każdy wiersz = jedna klatka, kolumny = warianty"""
    out = []
    for r in rows:
        ims = []
        for im in r:
            if crop is not None:
                y0, y1, x0, x1 = crop
                im = im[y0:y1, x0:x1]
            im = cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            ims.append(cv2.cvtColor(im, cv2.COLOR_GRAY2BGR))
        out.append(np.concatenate(ims, axis=1))
    img = np.concatenate(out, axis=0)
    if labels:
        wcol = img.shape[1] // len(labels)
        for i, l in enumerate(labels):
            cv2.putText(img, l, (i * wcol + 6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.imwrite(path, img)
