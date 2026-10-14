"""Task 1 — Edge & Line Detection.

Alternative approach to the sibling project:
  * Gaussian smoothing, then a MEDIAN-BASED automatic Canny threshold (adaptive)
    alongside two manual (low, high) configs -> three configs on the same image.
  * BOTH the standard Hough transform (cv2.HoughLines) and the probabilistic Hough
    transform (cv2.HoughLinesP, required by the brief), drawn in distinct colours.
"""

import cv2
import numpy as np

from .settings import CFG
from .io_tools import panel, render_table, say, to_rgb, write_image


class EdgePipeline:
    def __init__(self, cfg=CFG):
        self.cfg = cfg

    def _smooth(self, gray):
        k = self.cfg.blur_kernel
        return cv2.GaussianBlur(gray, (k, k), 0)

    def _auto_canny(self, gray):
        """Median-based automatic thresholds (Otsu-style heuristic)."""
        v = float(np.median(gray))
        s = self.cfg.auto_canny_sigma
        lo, hi = int(max(0, (1 - s) * v)), int(min(255, (1 + s) * v))
        return cv2.Canny(gray, lo, hi, L2gradient=True), (lo, hi)

    @staticmethod
    def _std_hough(edges, threshold):
        """Standard Hough -> convert (rho, theta) to drawable segments."""
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold)
        segs = []
        if lines is not None:
            for rho, theta in lines.reshape(-1, 2):
                a, b = np.cos(theta), np.sin(theta)
                x0, y0 = a * rho, b * rho
                segs.append((int(x0 - 1000 * b), int(y0 + 1000 * a),
                             int(x0 + 1000 * b), int(y0 - 1000 * a)))
        return segs

    def _prob_hough(self, edges):
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, self.cfg.hough_prob_threshold,
                                minLineLength=self.cfg.hough_prob_min_len,
                                maxLineGap=self.cfg.hough_prob_max_gap)
        return [] if lines is None else [tuple(int(v) for v in seg) for seg in lines.reshape(-1, 4)]

    @staticmethod
    def _draw(img, segs, color):
        out = img.copy()
        for x1, y1, x2, y2 in segs:
            cv2.line(out, (int(x1), int(y1)), (int(x2), int(y2)), color, 2, cv2.LINE_AA)
        return out

    def process(self, name, path):
        out_dir = self.cfg.task_dir("task1")
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        gray = self._smooth(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

        # three Canny configs on the same image
        configs, canny_rows, edge_maps, labels = [], [], [], []
        auto_edges, (alo, ahi) = self._auto_canny(gray)
        edge_maps.append(auto_edges)
        labels.append(f"auto ({alo},{ahi})")
        canny_rows.append([name, "auto (median)", f"{alo}/{ahi}", f"{100 * (auto_edges > 0).mean():.2f}%"])
        for lo, hi in self.cfg.canny_manual:
            e = cv2.Canny(gray, lo, hi, L2gradient=True)
            edge_maps.append(e)
            labels.append(f"manual ({lo},{hi})")
            canny_rows.append([name, "manual", f"{lo}/{hi}", f"{100 * (e > 0).mean():.2f}%"])
        panel([to_rgb(img)] + edge_maps, ["original"] + labels,
              out_dir / f"{name}_canny_configs.png", cols=2,
              suptitle=f"Task 1 — Canny configs ({name})")

        # Hough on the adaptive (auto) edge map: standard + probabilistic
        std_segs = self._std_hough(auto_edges, self.cfg.hough_std_threshold)
        prob_segs = self._prob_hough(auto_edges)
        std_vis = self._draw(img, std_segs, (0, 140, 255))     # orange (BGR)
        prob_vis = self._draw(img, prob_segs, (255, 90, 0))    # blue (BGR)
        write_image(out_dir / f"{name}_hough_standard.png", std_vis)
        write_image(out_dir / f"{name}_hough_prob.png", prob_vis)
        panel([to_rgb(img), auto_edges, to_rgb(std_vis), to_rgb(prob_vis)],
              ["original", "Canny (auto)", f"standard Hough ({len(std_segs)})",
               f"prob. Hough ({len(prob_segs)})"],
              out_dir / f"{name}_pipeline.png", cols=2,
              suptitle=f"Task 1 — edge & line pipeline ({name})")
        hough_rows = [[name, "standard", self.cfg.hough_std_threshold, len(std_segs)],
                      [name, "probabilistic",
                       f"{self.cfg.hough_prob_threshold}/{self.cfg.hough_prob_min_len}/{self.cfg.hough_prob_max_gap}",
                       len(prob_segs)]]
        return canny_rows, hough_rows

    def run(self, provider):
        say("TASK 1 — Edge & Line Detection")
        all_canny, all_hough = [], []
        for name, path in provider.road_scene():
            c, h = self.process(name, path)
            all_canny += c
            all_hough += h
        out_dir = self.cfg.task_dir("task1")
        render_table(all_canny, ["image", "config", "low/high", "edge density"],
                     out_dir / "task1_canny_table.png", title="Task 1 — Canny experiments")
        render_table(all_hough, ["image", "hough", "params", "#lines"],
                     out_dir / "task1_hough_table.png", title="Task 1 — Hough experiments")
        say(f"  wrote Task 1 outputs to {out_dir}")
        return {"canny": all_canny, "hough": all_hough}
