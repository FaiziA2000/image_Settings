"""Task 3 — Panorama / Mosaic construction (manual pipeline, no cv2.Stitcher).

Alternative approach to the sibling project:
  * Each frame is first projected onto a CYLINDER (so a horizontal pan is close to a
    pure translation).
  * Consecutive frames are aligned with cv2.findHomography(RANSAC) — inlier ratio is
    read from the RANSAC mask.
  * Homographies are accumulated against the LEFT-most frame; frames are warped onto a
    common canvas and combined with distance-transform FEATHER blending.
"""

import cv2
import numpy as np

from .settings import CFG
from .io_tools import panel, render_table, say, to_rgb, write_image, write_rgb


class MosaicBuilder:
    def __init__(self, cfg=CFG):
        self.cfg = cfg
        self.sift = cv2.SIFT_create()

    # -- cylindrical projection --------------------------------------------- #
    def _cylindrical(self, img):
        h, w = img.shape[:2]
        f = self.cfg.cylindrical_focal
        cx, cy = w / 2.0, h / 2.0
        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        theta = (xx - cx) / f
        hh = (yy - cy) / f
        X, Y, Z = np.sin(theta), hh, np.cos(theta)
        map_x = (f * X / Z + cx).astype(np.float32)
        map_y = (f * Y / Z + cy).astype(np.float32)
        warped = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT)
        return warped

    # -- pairwise homography (findHomography + RANSAC) ---------------------- #
    def _pair_homography(self, gray_a, gray_b):
        """H mapping B -> A, plus inlier ratio and match count."""
        kp_a, des_a = self.sift.detectAndCompute(gray_a, None)
        kp_b, des_b = self.sift.detectAndCompute(gray_b, None)
        bf = cv2.BFMatcher(cv2.NORM_L2)
        knn = bf.knnMatch(des_b, des_a, k=2)   # query = B, train = A  (map B->A)
        good = [m for m, n in (p for p in knn if len(p) == 2)
                if m.distance < self.cfg.lowe_ratio * n.distance]
        if len(good) < self.cfg.min_pair_matches:
            return None, 0.0, len(good)
        src = np.float32([kp_b[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp_a[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, self.cfg.ransac_thresh)
        if H is None:
            return None, 0.0, len(good)
        ratio = float(mask.mean()) if mask is not None else 0.0
        return H, ratio, len(good)

    @staticmethod
    def _corners(w, h):
        return np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)

    def run(self, provider):
        say("TASK 3 — Mosaic / Panorama (manual)")
        out_dir = self.cfg.task_dir("task3")
        paths = provider.mosaic_frames()
        frames = [self._cylindrical(cv2.imread(str(p))) for p in paths]
        grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
        n = len(frames)
        panel([to_rgb(f) for f in frames], [f"frame {i+1}" for i in range(n)],
              out_dir / "frames.png", cols=n, suptitle="Task 3 — cylindrical frames")

        # pairwise B->A homographies
        L, stats = [], []
        for i in range(n - 1):
            H, ratio, nmatch = self._pair_homography(grays[i], grays[i + 1])
            if H is None:
                say(f"  [pair {i+1}-{i+2}] EDGE CASE: only {nmatch} matches")
                stats.append([f"{i+1}-{i+2}", nmatch, "insufficient overlap"])
                L.append(None)
            else:
                stats.append([f"{i+1}-{i+2}", nmatch, f"{100*ratio:.1f}%"])
                say(f"  [pair {i+1}-{i+2}] matches={nmatch} inlier ratio={100*ratio:.1f}%")
                L.append(H)

        if any(h is None for h in L):
            render_table(stats, ["pair", "#matches", "RANSAC inliers"],
                         out_dir / "task3_stats.png", title="Task 3 — pairwise stats")
            say("  aborted: insufficient overlap in a pair")
            return {"stats": stats, "panorama": None}

        # accumulate to LEFT anchor (frame 0)
        M = [np.eye(3)]
        for i in range(1, n):
            M.append(M[i - 1] @ L[i - 1])

        all_c = np.vstack([cv2.perspectiveTransform(
            self._corners(frames[j].shape[1], frames[j].shape[0]), M[j]).reshape(-1, 2)
            for j in range(n)])
        xmin, ymin = np.floor(all_c.min(0)).astype(int)
        xmax, ymax = np.ceil(all_c.max(0)).astype(int)
        offset = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]], np.float64)
        size = (int(xmax - xmin), int(ymax - ymin))

        warped, weights, masks = [], [], []
        for j in range(n):
            Hf = offset @ M[j]
            warped.append(cv2.warpPerspective(frames[j], Hf, size))
            # validity mask = actual (non-black) cylindrical content, not the full rectangle,
            # so the thin black cylindrical borders never bleed into the seams
            content = (frames[j].max(axis=2) > 0).astype(np.uint8) * 255
            m = cv2.warpPerspective(content, Hf, size)
            masks.append(m > 0)
            weights.append(cv2.distanceTransform((m > 0).astype(np.uint8), cv2.DIST_L2, 5))
        wsum = np.sum(weights, axis=0) + 1e-8
        pano = np.zeros((size[1], size[0], 3), np.float32)
        for img, wt in zip(warped, weights):
            pano += img.astype(np.float32) * (wt / wsum)[..., None]
        pano = np.clip(pano, 0, 255).astype(np.uint8)

        # crop empty borders
        cover = np.any(masks, axis=0).astype(np.uint8)
        ys, xs = np.where(cover)
        pano = pano[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        write_image(out_dir / "panorama.png", pano)

        with open(out_dir / "homographies.txt", "w") as fh:
            for i, H in enumerate(L):
                fh.write(f"H[{i}] (frame {i+2} -> frame {i+1}):\n{np.array2string(H, precision=4)}\n\n")

        render_table(stats, ["pair", "#matches", "RANSAC inliers"],
                     out_dir / "task3_stats.png", title="Task 3 — pairwise stats")
        say(f"  panorama {pano.shape[1]}x{pano.shape[0]} -> {out_dir}")
        mean_inlier = np.mean([float(r[2].rstrip('%')) for r in stats if r[2].endswith('%')])
        return {"stats": stats, "panorama": list(pano.shape), "mean_inlier_ratio": mean_inlier,
                "method": "findHomography+cylindrical+feather"}
