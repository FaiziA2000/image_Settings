# CS-571 CCP — Technical Report (alternative implementation)
## 1. Executive Summary
Four-stage outdoor scene pipeline: auto-Canny + standard/probabilistic Hough (T1),
SIFT + brute-force + ratio test + homography verification (T2), manual cylindrical
RANSAC mosaic with feather blending (T3), MobileNetV2 fine-tuned on EuroSAT (T4,
best val accuracy **96.2%**). T5 evaluates all stages.

## 2. Pipeline Architecture
Frames -> auto-Canny + Hough -> SIFT/BF matching -> cylindrical + RANSAC mosaic ->
MobileNetV2 scene label. See paper/technical_report.pdf Figure 1.

## 3. Quantitative Results
| Task | Metric | Value |
|---|---|---|
| T2 | keypoints A/B, good, inliers | 3033/3421, 1825, 1814 |
| T3 | mean RANSAC inlier ratio | 70.8% (findHomography+cylindrical+feather) |
| T4 | best / final val accuracy | 96.2% / 96.2% |

### T4 per-class metrics
| class | precision | recall | F1 | support |
|---|---|---|---|---|
| Forest | 1.00 | 1.00 | 1.00 | 60 |
| Highway | 0.96 | 0.88 | 0.92 | 60 |
| Residential | 1.00 | 0.98 | 0.99 | 60 |
| River | 0.89 | 0.98 | 0.94 | 60 |

## 4. Failure Analysis (>= 2 per task)
See paper/technical_report.pdf page 3 (auto-threshold texture response; standard-Hough
line merging; ambiguous SIFT matches / ratio trade-off; low-overlap RANSAC instability
and cylindrical focal errors; River-vs-Highway confusion; Grad-CAM background focus).

## 5. Classical vs Deep-Learning Trade-offs
Classical (T1-T3): CPU real-time, exact under valid geometry, transparent, no data, but
brittle to low texture/overlap. Deep (T4): strong semantics but opaque, data-hungry,
GPU-trained, confuses similar classes.

## 6. Limitations & Future Work
Overlapping-crop mosaic vs multi-camera parallax; EuroSAT satellite vs ground-level;
CPU-bound SIFT/BF; fuse geometry + semantics into a temporal scene map.

*Reproducible via `python main.py` (fixed seed). Data cited in README.*
