"""Data provider — real-first with a reproducible synthetic fallback.

Real sources (no authentication):
  * OpenCV sample images (BSD-3-Clause) for tasks 1-3 base scenes.
  * EuroSAT land-use dataset (torchvision auto-download) for task 4.

Distinct from the sibling project: different base images (home / aerial) and a
different classification dataset (EuroSAT satellite land-use, not ground-level).
"""

import urllib.request
from pathlib import Path

import cv2
import numpy as np

from .settings import CFG
from .io_tools import say, write_image

_OPENCV = "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data"
_UA = {"User-Agent": "ccp571/1.0"}


def _grab(url, dest, retries=4):
    dest = Path(dest)
    if dest.exists():
        return dest
    last = None
    for k in range(retries):
        try:
            req = urllib.request.Request(url, headers=_UA)
            with urllib.request.urlopen(req, timeout=30) as r:
                data = r.read()
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            return dest
        except Exception as e:
            last = e
            import time
            time.sleep(1.5 * (k + 1))
    raise last


def _sample(name, dest):
    _grab(f"{_OPENCV}/{name}", dest)
    say(f"  fetched real image {name}")
    return dest


# --------------------------------------------------------------------------- #
# synthetic fallbacks (different look from the sibling project)
# --------------------------------------------------------------------------- #
def _synthetic_street(h=440, w=720):
    rng = np.random.default_rng(CFG.seed)
    img = np.full((h, w, 3), 60, np.uint8)
    img[: h // 3] = (150, 130, 110)
    img += rng.integers(0, 10, img.shape, dtype=np.int16).astype(np.uint8)
    # buildings flanking a road with converging kerbs
    for x in range(20, w // 2 - 60, 46):
        cv2.rectangle(img, (x, 90), (x + 34, h // 3), (90, 90, 95), -1)
    vx = w // 2
    cv2.line(img, (int(0.12 * w), h), (vx, h // 3), (240, 240, 240), 3)
    cv2.line(img, (int(0.88 * w), h), (vx, h // 3), (240, 240, 240), 3)
    for i in range(8):
        y0 = int(h - i / 8 * (2 * h / 3))
        y1 = int(h - (i + 0.45) / 8 * (2 * h / 3))
        cv2.line(img, (vx, y0), (vx, y1), (60, 220, 220), 3)
    return img


def _synthetic_rich(h=460, w=640, seed=CFG.seed):
    rng = np.random.default_rng(seed)
    img = rng.integers(70, 140, (h, w, 3), dtype=np.uint8)
    for _ in range(70):
        p = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        q = (p[0] + int(rng.integers(15, 55)), p[1] + int(rng.integers(15, 55)))
        cv2.rectangle(img, p, q, tuple(int(v) for v in rng.integers(0, 255, 3)), -1)
    for _ in range(50):
        c = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        cv2.circle(img, c, int(rng.integers(3, 12)),
                   tuple(int(v) for v in rng.integers(0, 255, 3)), 2)
    return img


def _homography_view(img, angle=-11, tx=-45, ty=15, gain=1.12):
    h, w = img.shape[:2]
    a = np.deg2rad(angle)
    cx, cy = w / 2, h / 2
    rot = np.array([[np.cos(a), -np.sin(a), cx - (np.cos(a) * cx - np.sin(a) * cy)],
                    [np.sin(a), np.cos(a), cy - (np.sin(a) * cx + np.cos(a) * cy)],
                    [0, 0, 1]])
    persp = np.array([[1, 0, tx], [0, 1, ty], [-2.5e-4, 1.5e-4, 1]])
    H = persp @ rot
    out = cv2.warpPerspective(img, H, (w, h))
    out = np.clip(out.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    return out, H


# --------------------------------------------------------------------------- #
# public provider
# --------------------------------------------------------------------------- #
class DataProvider:
    """Supplies inputs for every task, honoring CFG.data_mode."""

    def __init__(self, mode=None):
        self.mode = mode or CFG.data_mode

    def road_scene(self):
        """Return list of (name, path) outdoor images with strong lines."""
        out = []
        if self.mode in ("auto", "real"):
            try:
                out.append(("house", _sample("home.jpg", CFG.raw_dir / "t1_house.jpg")))
            except Exception as e:
                if self.mode == "real":
                    raise
                say(f"  [t1] download failed ({e}); synthetic only")
        road = CFG.raw_dir / "t1_street.png"
        write_image(road, _synthetic_street())
        out.append(("street", road))
        return out

    def stereo_pair(self):
        """Two views of one scene; returns (path_a, path_b, gt_H_or_None)."""
        if self.mode in ("auto", "real"):
            try:
                base = _sample("aero3.jpg", CFG.raw_dir / "t2_base.jpg")
                img = cv2.imread(str(base), cv2.IMREAD_COLOR)
                view, H = _homography_view(img)
                a = CFG.raw_dir / "t2_view_a.png"
                b = CFG.raw_dir / "t2_view_b.png"
                write_image(a, img)
                write_image(b, view)
                return a, b, H
            except Exception as e:
                if self.mode == "real":
                    raise
                say(f"  [t2] download failed ({e}); synthetic pair")
        base = _synthetic_rich()
        view, H = _homography_view(base)
        a, b = CFG.raw_dir / "t2_view_a.png", CFG.raw_dir / "t2_view_b.png"
        write_image(a, base)
        write_image(b, view)
        return a, b, H

    def mosaic_frames(self, n=None):
        n = n or CFG.mosaic_frames
        src = CFG.raw_dir / "t3_source.png"
        got = False
        if self.mode in ("auto", "real"):
            try:
                _sample("graf1.png", src)
                got = True
            except Exception as e:
                if self.mode == "real":
                    raise
                say(f"  [t3] download failed ({e}); synthetic wide scene")
        if not got:
            write_image(src, _synthetic_rich(h=440, w=1180, seed=99))
        wide = cv2.imread(str(src), cv2.IMREAD_COLOR)
        h, w = wide.shape[:2]
        win = int(w / (1 + (n - 1) * 0.5))       # ~50% overlap
        step = int((w - win) / (n - 1))
        paths = []
        for i in range(n):
            x0 = min(i * step, w - win)
            frame = wide[:, x0:x0 + win].copy()
            frame = np.clip(frame.astype(np.float32) * (0.92 + 0.05 * i), 0, 255).astype(np.uint8)
            p = CFG.raw_dir / f"t3_frame{i + 1}.png"
            write_image(p, frame)
            paths.append(p)
        return paths

    # -- task 4 dataset (EuroSAT via torchvision) --------------------------- #
    def scene_dataset(self):
        train_dir = CFG.scene_dir / "train"
        val_dir = CFG.scene_dir / "val"
        if self._ds_ready(train_dir, val_dir):
            say("  [t4] scene dataset already present")
            return train_dir, val_dir, CFG.scene_classes
        if self.mode in ("auto", "real"):
            try:
                self._build_eurosat(train_dir, val_dir)
                return train_dir, val_dir, CFG.scene_classes
            except Exception as e:
                if self.mode == "real":
                    raise
                say(f"  [t4] EuroSAT unavailable ({e}); synthetic land-use fallback")
        self._build_synthetic_scene(train_dir, val_dir)
        return train_dir, val_dir, CFG.scene_classes

    def _ds_ready(self, train_dir, val_dir):
        if not train_dir.exists() or not val_dir.exists():
            return False
        for c in CFG.scene_classes:
            if len(list((train_dir / c).glob("*.png"))) < CFG.train_per_class:
                return False
            if len(list((val_dir / c).glob("*.png"))) < CFG.val_per_class:
                return False
        return True

    def _build_eurosat(self, train_dir, val_dir):
        import torchvision

        say("  [t4] downloading EuroSAT (torchvision)...")
        ds = torchvision.datasets.EuroSAT(root=str(CFG.data_dir / "_eurosat"), download=True)
        # bucket indices by our target class names
        name_to_idx = {n: i for i, n in enumerate(ds.classes)}
        for c in CFG.scene_classes:
            if c not in name_to_idx:
                raise RuntimeError(f"EuroSAT missing class {c} (has {ds.classes})")
        buckets = {c: [] for c in CFG.scene_classes}
        wanted = {name_to_idx[c]: c for c in CFG.scene_classes}
        for i, (_, label) in enumerate(ds.samples if hasattr(ds, "samples") else []):
            if label in wanted:
                buckets[wanted[label]].append(i)
        # torchvision EuroSAT exposes .samples (path,label); fall back to iteration
        if not any(buckets.values()):
            for i in range(len(ds)):
                _, label = ds[i]
                if label in wanted:
                    buckets[wanted[label]].append(i)
        rng = np.random.default_rng(CFG.seed)
        for c in CFG.scene_classes:
            idxs = buckets[c]
            rng.shuffle(idxs)
            need = CFG.train_per_class + CFG.val_per_class
            if len(idxs) < need:
                raise RuntimeError(f"not enough EuroSAT images for {c}")
            (train_dir / c).mkdir(parents=True, exist_ok=True)
            (val_dir / c).mkdir(parents=True, exist_ok=True)
            for k, idx in enumerate(idxs[:need]):
                img, _ = ds[idx]                       # PIL image
                arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                is_train = k < CFG.train_per_class
                split = train_dir if is_train else val_dir
                j = k if is_train else k - CFG.train_per_class
                write_image(split / c / f"{j:04d}.png", arr)
            say(f"  [t4] {c}: {CFG.train_per_class} train + {CFG.val_per_class} val (EuroSAT)")

    def _build_synthetic_scene(self, train_dir, val_dir):
        rng = np.random.default_rng(CFG.seed)
        size = CFG.img_size

        def make(cls, seed):
            r = np.random.default_rng(seed)
            img = np.zeros((size, size, 3), np.uint8)
            if cls == "Forest":
                img[:] = (30, 100, 30)
                for _ in range(500):
                    c = (int(r.integers(0, size)), int(r.integers(0, size)))
                    cv2.circle(img, c, int(r.integers(1, 5)),
                               (int(r.integers(10, 60)), int(r.integers(80, 150)), int(r.integers(10, 60))), -1)
            elif cls == "Highway":
                img[:] = (90, 90, 95)
                cv2.rectangle(img, (size // 2 - 12, 0), (size // 2 + 12, size), (60, 60, 65), -1)
                for y in range(0, size, 20):
                    cv2.line(img, (size // 2, y), (size // 2, y + 8), (240, 240, 240), 2)
            elif cls == "Residential":
                img[:] = (120, 120, 125)
                for _ in range(40):
                    x, y = r.integers(0, size - 14), r.integers(0, size - 14)
                    cv2.rectangle(img, (x, y), (x + 10, y + 10), (40, 40, 60), -1)
            else:  # River
                img[:] = (120, 90, 40)
                cv2.rectangle(img, (0, size // 2 - 18), (size, size // 2 + 18), (150, 110, 60), -1)
            img = np.clip(img.astype(np.int16) + r.integers(-16, 16, img.shape), 0, 255).astype(np.uint8)
            return img

        for split, per in (("train", CFG.train_per_class), ("val", CFG.val_per_class)):
            for c in CFG.scene_classes:
                d = (train_dir if split == "train" else val_dir) / c
                d.mkdir(parents=True, exist_ok=True)
                for i in range(per):
                    write_image(d / f"{i:04d}.png", make(c, int(rng.integers(0, 1 << 30))))
        say("  [t4] built synthetic land-use dataset (fallback)")
