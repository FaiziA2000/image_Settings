# CS-571 — Computer Vision Complex Computing Problem
## Autonomous Visual Intelligence System (alternative implementation)

A five-stage outdoor scene-understanding pipeline: edge/line detection, SIFT feature
matching, manual panorama construction, MobileNetV2 scene classification, and a
critical technical report.

## Quick start
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py                 # all tasks (real data) + report
python main.py --task 3        # a single task (1|2|3|4)
python main.py --data synthetic  # fully offline
```

## Approach (per stage)
| Stage | Module | Method |
|---|---|---|
| 1 | `visionpipe/edges.py` | Gaussian smoothing, **median-based auto-Canny** (+2 manual configs), **standard + probabilistic Hough** |
| 2 | `visionpipe/features.py` | **SIFT** + **BFMatcher** + Lowe's ratio (0.75) + cross-check + homography inlier verification, top-50 matches |
| 3 | `visionpipe/mosaic.py` | **cylindrical projection** + `cv2.findHomography(RANSAC)` + left-anchor accumulation + **feather blending** (no `cv2.Stitcher`) |
| 4 | `visionpipe/classifier.py` | **MobileNetV2** transfer learning on **EuroSAT**, augmentation, curves, confusion, per-class P/R, **Grad-CAM** |
| 5 | `visionpipe/report.py` | data-driven PDF + Markdown report |

## Layout
```
main.py                     single-command entry point
visionpipe/                 settings, io_tools, datasets, edges, features, mosaic, classifier, report
results/task1..4/           generated visualizations
paper/technical_report.*    Task-5 report
```

## Data & citations
- **Tasks 1–3** — OpenCV sample images (BSD-3-Clause): <https://github.com/opencv/opencv> (`samples/data`). Task 2's second view is created with a known homography; Task 3 frames are overlapping crops of a wide scene.
- **Task 4** — EuroSAT land-use dataset (Sentinel-2), auto-downloaded via `torchvision.datasets.EuroSAT`: <https://github.com/phelber/EuroSAT>.
- No images are self-captured. Offline fallback: `python main.py --data synthetic`.

## Notes
- Allowed libraries only (OpenCV, NumPy, Matplotlib, PyTorch/torchvision, scikit-learn); downloads use the Python standard library.
- Panorama stitching is implemented manually — no `cv2.Stitcher`.
- Reproducible via a fixed seed in `visionpipe/settings.py`.
