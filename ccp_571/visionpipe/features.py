"""Task 2 — SIFT Feature Extraction & Matching.

Alternative approach: SIFT + a brute-force (BFMatcher) matcher. We keep Lowe's ratio
test (0.75, required) as the primary filter, compare it against a cross-check matcher,
and add a geometric verification (findHomography inliers) to quantify match quality.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

from .settings import CFG
from .io_tools import say, to_rgb, write_rgb


class FeatureMatcher:
    def __init__(self, cfg=CFG):
        self.cfg = cfg
        self.sift = cv2.SIFT_create()

    def _ratio_good(self, knn, ratio):
        good = []
        for pair in knn:
            if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance:
                good.append(pair[0])
        return good

    def run(self, provider):
        say("TASK 2 — SIFT Feature Extraction & Matching")
        out_dir = self.cfg.task_dir("task2")
        pa, pb, _ = provider.stereo_pair()
        img_a, img_b = cv2.imread(str(pa)), cv2.imread(str(pb))
        gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)

        kp_a, des_a = self.sift.detectAndCompute(gray_a, None)
        kp_b, des_b = self.sift.detectAndCompute(gray_b, None)
        say(f"  keypoints: A={len(kp_a)} B={len(kp_b)}")

        flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        write_rgb(out_dir / "keypoints_a.png",
                  to_rgb(cv2.drawKeypoints(img_a, kp_a, None, (255, 0, 255), flags)))
        write_rgb(out_dir / "keypoints_b.png",
                  to_rgb(cv2.drawKeypoints(img_b, kp_b, None, (255, 0, 255), flags)))

        # primary: brute-force knn + Lowe ratio test
        bf = cv2.BFMatcher(cv2.NORM_L2)
        knn = bf.knnMatch(des_a, des_b, k=2)
        good = sorted(self._ratio_good(knn, self.cfg.lowe_ratio), key=lambda m: m.distance)
        match_ratio = len(good) / max(1, min(len(kp_a), len(kp_b)))

        # comparison: symmetric cross-check matcher
        bf_cc = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        cross = bf_cc.match(des_a, des_b)
        say(f"  good (ratio {self.cfg.lowe_ratio})={len(good)}  cross-check={len(cross)}")

        # geometric verification via homography inliers
        inliers = 0
        if len(good) >= 4:
            src = np.float32([kp_a[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst = np.float32([kp_b[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            _, mask = cv2.findHomography(src, dst, cv2.RANSAC, self.cfg.ransac_thresh)
            inliers = int(mask.sum()) if mask is not None else 0

        top = good[: self.cfg.top_matches]
        vis = cv2.drawMatches(img_a, kp_a, img_b, kp_b, top, None,
                              matchColor=(0, 255, 0), singlePointColor=(255, 0, 255),
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        write_rgb(out_dir / "top_matches.png", to_rgb(vis))

        # ratio sweep -> precision/recall trade-off
        counts = [len(self._ratio_good(knn, r)) for r in self.cfg.ratio_grid]
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        ax.plot(self.cfg.ratio_grid, counts, "D-", color="#33691e", lw=2)
        ax.axvline(self.cfg.lowe_ratio, color="#c62828", ls="--",
                   label=f"chosen {self.cfg.lowe_ratio}")
        ax.set_xlabel("Lowe ratio"); ax.set_ylabel("# good matches")
        ax.set_title("Task 2 — ratio vs match count (low=precision, high=recall)")
        ax.grid(alpha=0.3); ax.legend()
        fig.tight_layout(); fig.savefig(out_dir / "ratio_sweep.png", dpi=140); plt.close(fig)

        from .io_tools import render_table
        rows = [["keypoints A", len(kp_a)], ["keypoints B", len(kp_b)],
                [f"good matches (ratio {self.cfg.lowe_ratio})", len(good)],
                ["cross-check matches", len(cross)],
                ["homography inliers", inliers],
                ["match quality ratio", f"{match_ratio:.3f}"]]
        render_table(rows, ["metric", "value"], out_dir / "task2_stats.png",
                     title="Task 2 — SIFT matching statistics")
        say(f"  wrote Task 2 outputs to {out_dir}")
        return {"kp_a": len(kp_a), "kp_b": len(kp_b), "good": len(good),
                "cross_check": len(cross), "inliers": inliers, "match_ratio": match_ratio}
