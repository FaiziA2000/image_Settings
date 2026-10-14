"""Central settings for the vision pipeline (dataclass style).

A single frozen ``Settings`` instance, ``CFG``, holds every path and hyper-parameter.
This keeps the pipeline reproducible and puts all tunables in one obvious place.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    # -- paths ----------------------------------------------------------------
    root: Path = _ROOT
    data_dir: Path = _ROOT / "data"
    raw_dir: Path = _ROOT / "data" / "raw"
    scene_dir: Path = _ROOT / "data" / "scene_ds"
    results_dir: Path = _ROOT / "results"
    paper_dir: Path = _ROOT / "paper"

    # -- reproducibility ------------------------------------------------------
    seed: int = 1234

    # -- Task 1: edges & lines ------------------------------------------------
    blur_kernel: int = 5                 # bilateral / Gaussian window
    auto_canny_sigma: float = 0.33       # median-based automatic threshold spread
    canny_manual: Tuple[Tuple[int, int], ...] = ((40, 120), (80, 200))
    hough_std_threshold: int = 140       # accumulator votes for standard HoughLines
    hough_prob_threshold: int = 45
    hough_prob_min_len: int = 45
    hough_prob_max_gap: int = 12

    # -- Task 2: SIFT + brute-force matching ----------------------------------
    lowe_ratio: float = 0.75             # brief-mandated ratio test
    top_matches: int = 50
    ratio_grid: Tuple[float, ...] = (0.5, 0.6, 0.7, 0.75, 0.8, 0.9)

    # -- Task 3: mosaic / panorama --------------------------------------------
    mosaic_frames: int = 4               # >= 3 overlapping frames
    ransac_thresh: float = 4.0           # cv2.findHomography reprojection tolerance
    min_pair_matches: int = 15
    cylindrical_focal: float = 800.0     # focal length for cylindrical warp

    # -- Task 4: MobileNetV2 scene classifier ---------------------------------
    scene_classes: List[str] = field(default_factory=lambda:
                                     ["Forest", "Highway", "Residential", "River"])
    train_per_class: int = 200           # >= 100 required
    val_per_class: int = 60
    img_size: int = 160                  # MobileNetV2 default input
    batch_size: int = 32
    epochs: int = 6
    lr: float = 1.5e-3
    weight_decay: float = 1e-4
    target_acc: float = 0.70
    imagenet_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    imagenet_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    # -- data sourcing --------------------------------------------------------
    data_mode: str = "auto"              # auto | synthetic | real

    def all_dirs(self):
        return (self.raw_dir, self.scene_dir, self.results_dir, self.paper_dir)

    def task_dir(self, name: str) -> Path:
        d = self.results_dir / name
        d.mkdir(parents=True, exist_ok=True)
        return d


CFG = Settings()
