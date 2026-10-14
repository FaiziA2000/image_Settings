"""IO, plotting and small utilities (OpenCV + NumPy + Matplotlib only)."""

import math
import random
import time
from pathlib import Path

import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# --------------------------------------------------------------------------- #
# reproducibility & timing
# --------------------------------------------------------------------------- #
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


class Clock:
    """Tiny stopwatch used for progress messages."""

    def __init__(self):
        self.t0 = time.monotonic()

    def stamp(self, msg):
        print(f"[{time.monotonic() - self.t0:6.1f}s] {msg}", flush=True)


CLOCK = Clock()


def say(msg):
    CLOCK.stamp(msg)


# --------------------------------------------------------------------------- #
# image io (BGR internally)
# --------------------------------------------------------------------------- #
def read_color(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"cannot read {path}")
    return img


def read_gray(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"cannot read {path}")
    return img


def to_rgb(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def write_image(path, bgr):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), bgr)


def write_rgb(path, rgb):
    write_image(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


# --------------------------------------------------------------------------- #
# plotting — a flexible grid (auto rows x cols), unlike a fixed single row
# --------------------------------------------------------------------------- #
def panel(images, titles, save_to, cols=None, suptitle=None, cmaps=None):
    n = len(images)
    cols = cols or min(n, 3)
    rows = math.ceil(n / cols)
    if cmaps is None:
        cmaps = [None] * n
    fig, axes = plt.subplots(rows, cols, figsize=(4.4 * cols, 4.0 * rows))
    axes = np.atleast_1d(axes).ravel()
    for i, ax in enumerate(axes):
        if i >= n:
            ax.axis("off")
            continue
        im, cmap = images[i], cmaps[i]
        if im.ndim == 2 and cmap is None:
            cmap = "gray"
        ax.imshow(im, cmap=cmap)
        ax.set_title(titles[i], fontsize=10)
        ax.axis("off")
    if suptitle:
        fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    fig.tight_layout()
    Path(save_to).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_to, dpi=130, bbox_inches="tight")
    plt.close(fig)


def render_table(rows, headers, save_to, title=None):
    fig, ax = plt.subplots(figsize=(min(2.1 * len(headers), 12), 0.5 + 0.4 * (len(rows) + 1)))
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    tbl = ax.table(cellText=[[str(c) for c in r] for r in rows],
                   colLabels=headers, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.5)
    for j in range(len(headers)):
        cell = tbl[0, j]
        cell.set_facecolor("#33691e")   # olive header (distinct palette from v1)
        cell.set_text_props(color="white", fontweight="bold")
    fig.tight_layout()
    Path(save_to).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_to, dpi=140, bbox_inches="tight")
    plt.close(fig)
