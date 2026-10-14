"""Task 5 — Critical Evaluation & Technical Report.

Produces report/paper as a Markdown source plus a multi-page PDF (matplotlib
PdfPages, an allowed library). Data-driven from the metrics returned by tasks 1-4.
"""

import json
import textwrap

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from .settings import CFG
from .io_tools import say

A4 = (8.27, 11.69)
INK = "#1b5e20"   # olive/green heading colour (distinct palette)


def _wrap(fig, y, s, size=10, weight="normal", color="#111", x=0.07, width=112, gap=0.006):
    s = "\n".join(textwrap.fill(ln, width) if ln else "" for ln in s.split("\n"))
    fig.text(x, y, s, fontsize=size, fontweight=weight, color=color, va="top")
    return y - ((s.count("\n") + 1) * size * 1.7 / 792) - gap


def _line(fig, y):
    fig.add_artist(plt.Line2D([0.07, 0.93], [y, y], color=INK, lw=1.3, transform=fig.transFigure))
    return y - 0.012


def _img(fig, rect, path, cap=None):
    try:
        im = plt.imread(str(path))
    except FileNotFoundError:
        return
    ax = fig.add_axes(rect); ax.imshow(im); ax.axis("off")
    if cap:
        ax.set_title(cap, fontsize=8, color="#333")


class ReportWriter:
    def __init__(self, cfg=CFG):
        self.cfg = cfg

    def _diagram(self, fig, rect):
        ax = fig.add_axes(rect); ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis("off")
        steps = ["Frames", "Edges &\nLines", "SIFT\nmatch", "Cylindrical\nmosaic", "MobileNetV2\nscene"]
        for i, s in enumerate(steps):
            x = 0.3 + i * 1.95
            fc = "#e8f5e9" if 0 < i < 4 else ("#dcedc8" if i == 0 else "#ffe0b2")
            ax.add_patch(mpatches.FancyBboxPatch((x, 1.4), 1.5, 1.2, boxstyle="round,pad=0.08",
                         fc=fc, ec=INK, lw=1.2))
            ax.text(x + 0.75, 2.0, s, ha="center", va="center", fontsize=7.4)
            if i:
                ax.annotate("", xy=(x - 0.02, 2.0), xytext=(x - 0.45, 2.0),
                            arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))

    def _p1(self, pdf, R):
        fig = plt.figure(figsize=A4)
        fig.text(0.07, 0.955, "CS-571 — Computer Vision Complex Computing Problem",
                 fontsize=13.5, fontweight="bold", color=INK)
        y = _wrap(fig, 0.93, "Autonomous Visual Intelligence System (alternative implementation)",
                  size=11, weight="bold")
        y = _line(fig, y)
        y = _wrap(fig, y, "1. Executive Summary", size=12, weight="bold", color=INK)
        acc = R.get("task4", {}).get("best_acc", 0.0)
        y = _wrap(fig, y, (
            "This system builds an outdoor-scene understanding pipeline from four stages. "
            "Edges and lines are found with Gaussian smoothing, a median-based automatic "
            "Canny threshold, and both standard and probabilistic Hough transforms. Cross-"
            "view correspondence uses SIFT with a brute-force matcher, Lowe's ratio test and "
            "homography-based geometric verification. A panorama is built manually by "
            "projecting frames onto a cylinder, aligning them with RANSAC homographies and "
            "feather-blending them. Finally a pretrained MobileNetV2 is fine-tuned on EuroSAT "
            f"land-use imagery, reaching a best validation accuracy of {acc:.1%} (target 70%)."),
            size=9.5)
        y = _wrap(fig, y - 0.005, "2. Pipeline Architecture", size=12, weight="bold", color=INK)
        self._diagram(fig, [0.06, 0.52, 0.88, 0.16])
        fig.text(0.07, 0.50, "Figure 1 — five-stage data flow (geometry stages feed the learned stage).",
                 fontsize=8, color="#333")
        t2, t3 = R.get("task2", {}), R.get("task3", {})
        _wrap(fig, 0.46, (f"Headline: T2 keypoints A/B={t2.get('kp_a','-')}/{t2.get('kp_b','-')}, "
              f"good matches={t2.get('good','-')}, homography inliers={t2.get('inliers','-')}  |  "
              f"T3 mean inlier ratio={t3.get('mean_inlier_ratio', float('nan')):.1f}%  |  "
              f"T4 best acc={acc:.1%}"), size=9, weight="bold", color=INK, width=88)
        pdf.savefig(fig); plt.close(fig)

    def _p2(self, pdf, R):
        fig = plt.figure(figsize=A4)
        y = _wrap(fig, 0.955, "3. Quantitative Results", size=13, weight="bold", color=INK)
        _line(fig, y)
        _img(fig, [0.07, 0.71, 0.40, 0.17], self.cfg.results_dir / "task1/task1_hough_table.png", "T1 — Hough experiments")
        _img(fig, [0.53, 0.71, 0.40, 0.17], self.cfg.results_dir / "task2/task2_stats.png", "T2 — SIFT stats")
        _img(fig, [0.07, 0.49, 0.40, 0.17], self.cfg.results_dir / "task3/task3_stats.png", "T3 — RANSAC inliers")
        _img(fig, [0.53, 0.49, 0.40, 0.17], self.cfg.results_dir / "task4/task4_summary.png", "T4 — summary")
        _img(fig, [0.09, 0.25, 0.40, 0.19], self.cfg.results_dir / "task4/training_curves.png", "T4 — curves")
        _img(fig, [0.54, 0.25, 0.38, 0.19], self.cfg.results_dir / "task4/confusion_matrix.png", "T4 — confusion")
        _img(fig, [0.30, 0.04, 0.40, 0.17], self.cfg.results_dir / "task4/per_class_metrics.png", "T4 — per-class")
        pdf.savefig(fig); plt.close(fig)

    def _p3(self, pdf, R):
        fig = plt.figure(figsize=A4)
        y = _wrap(fig, 0.955, "4. Failure Analysis (>= 2 per task)", size=13, weight="bold", color=INK)
        y = _line(fig, y)
        y = _wrap(fig, y, (
            "Task 1. (a) A median-based auto threshold adapts to global brightness but still "
            "over-responds to fine texture, adding spurious short Hough segments. (b) The "
            "standard Hough returns infinite lines that ignore segment endpoints, so it merges "
            "distinct co-linear structures; the probabilistic variant is preferred for real edges.\n"
            "Task 2. (a) Repetitive / self-similar texture yields ambiguous descriptors and wrong "
            "matches that only geometric (homography) verification removes. (b) Ratio 0.9 admits "
            "false positives (low precision); 0.5 starves recall — quantified by the ratio sweep.\n"
            "Task 3. (a) Too little overlap makes RANSAC homography unstable (flagged as an edge "
            "case). (b) Wrong cylindrical focal length under/over-curves frames, leaving residual "
            "seams that feathering only partly hides.\n"
            "Task 4. (a) Spectrally similar land-use classes (e.g. River vs Highway) are confused — "
            "see the off-diagonal confusion cells. (b) The Grad-CAM incorrect sample shows the net "
            "focusing on texture/background rather than the class-defining region."), size=8.8, width=118)
        _img(fig, [0.07, 0.20, 0.40, 0.16], self.cfg.results_dir / "task2/ratio_sweep.png", "T2 — ratio sweep")
        _img(fig, [0.53, 0.20, 0.40, 0.16], self.cfg.results_dir / "task4/confusion_matrix.png", "T4 — confusion")
        _img(fig, [0.20, 0.02, 0.60, 0.16], self.cfg.results_dir / "task4/gradcam.png", "T4 — Grad-CAM")
        pdf.savefig(fig); plt.close(fig)

    def _p4(self, pdf, R):
        fig = plt.figure(figsize=A4)
        y = _wrap(fig, 0.955, "5. Classical vs. Deep-Learning Trade-offs", size=13, weight="bold", color=INK)
        _line(fig, y)
        ax = fig.add_axes([0.07, 0.60, 0.86, 0.28]); ax.axis("off")
        rows = [["Speed", "Real-time on CPU", "GPU/MPS to train"],
                ["Accuracy", "Exact under valid geometry", "High semantics, distribution-sensitive"],
                ["Interpretability", "Transparent maths", "Opaque; needs Grad-CAM"],
                ["Data need", "None (tuned params)", ">=100 labelled images/class"],
                ["Failure", "Low texture / overlap", "Similar-class confusion"]]
        t = ax.table(cellText=rows, colLabels=["Dimension", "Classical (T1-T3)", "Deep learning (T4)"],
                     cellLoc="left", loc="center", colWidths=[0.2, 0.38, 0.42])
        t.auto_set_font_size(False); t.set_fontsize(8.6); t.scale(1, 1.6)
        for j in range(3):
            t[0, j].set_facecolor(INK); t[0, j].set_text_props(color="white", fontweight="bold")
        y = _wrap(fig, 0.57, "6. Limitations & Future Work", size=13, weight="bold", color=INK)
        y = _line(fig, y)
        _wrap(fig, y, (
            "- Mosaic frames are overlapping crops of a wide scene; genuine multi-camera capture "
            "with parallax would need bundle adjustment.\n"
            "- EuroSAT is satellite land-use; a ground-level dataset would better match a driving "
            "robot, and the full taxonomy would test generalisation.\n"
            "- SIFT + brute-force matching is CPU-bound; a learned detector or GPU matcher would "
            "raise throughput.\n"
            "- Future work: fuse the geometric mosaic with the semantic labels into one scene map "
            "and add temporal consistency.\n\n"
            "Reproducible via 'python main.py' (fixed seed). Data cited in README (OpenCV samples, "
            "BSD-3; EuroSAT land-use)."), size=9, width=112)
        pdf.savefig(fig); plt.close(fig)

    def _markdown(self, R):
        t2, t3, t4 = R.get("task2", {}), R.get("task3", {}), R.get("task4", {})
        acc = t4.get("best_acc", 0.0)
        pc = "\n".join(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} |" for r in t4.get("per_class", []))
        md = f"""# CS-571 CCP — Technical Report (alternative implementation)
## 1. Executive Summary
Four-stage outdoor scene pipeline: auto-Canny + standard/probabilistic Hough (T1),
SIFT + brute-force + ratio test + homography verification (T2), manual cylindrical
RANSAC mosaic with feather blending (T3), MobileNetV2 fine-tuned on EuroSAT (T4,
best val accuracy **{acc:.1%}**). T5 evaluates all stages.

## 2. Pipeline Architecture
Frames -> auto-Canny + Hough -> SIFT/BF matching -> cylindrical + RANSAC mosaic ->
MobileNetV2 scene label. See paper/technical_report.pdf Figure 1.

## 3. Quantitative Results
| Task | Metric | Value |
|---|---|---|
| T2 | keypoints A/B, good, inliers | {t2.get('kp_a','-')}/{t2.get('kp_b','-')}, {t2.get('good','-')}, {t2.get('inliers','-')} |
| T3 | mean RANSAC inlier ratio | {t3.get('mean_inlier_ratio', float('nan')):.1f}% ({t3.get('method','-')}) |
| T4 | best / final val accuracy | {acc:.1%} / {t4.get('final_acc',0):.1%} |

### T4 per-class metrics
| class | precision | recall | F1 | support |
|---|---|---|---|---|
{pc}

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
"""
        (self.cfg.paper_dir / "technical_report.md").write_text(md)

    def run(self, results=None):
        say("TASK 5 — technical report")
        if results is None:
            results = json.loads((self.cfg.results_dir / "results.json").read_text())
        self.cfg.paper_dir.mkdir(parents=True, exist_ok=True)
        self._markdown(results)
        with PdfPages(str(self.cfg.paper_dir / "technical_report.pdf")) as pdf:
            self._p1(pdf, results); self._p2(pdf, results)
            self._p3(pdf, results); self._p4(pdf, results)
        say(f"  wrote {self.cfg.paper_dir}/technical_report.pdf + .md")
