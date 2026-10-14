"""Task 4 — Scene Classification via MobileNetV2 transfer learning.

Alternative approach to the sibling project:
  * Backbone: pretrained MobileNetV2 (torchvision), last feature blocks + classifier
    fine-tuned on EuroSAT land-use classes.
  * Augmentation: horizontal flip, rotation, colour jitter.
  * Reports: training/validation curves, confusion matrix, per-class precision/recall.
  * Grad-CAM on the last MobileNet feature block for 2 correct + 1 incorrect samples.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

from .settings import CFG
from .io_tools import render_table, say


class _GradCAM:
    def __init__(self, model, layer):
        self.model = model
        self.a = None
        self.g = None
        layer.register_forward_hook(lambda m, i, o: setattr(self, "a", o.detach()))
        layer.register_full_backward_hook(lambda m, gi, go: setattr(self, "g", go[0].detach()))

    def map(self, x, cls):
        self.model.zero_grad()
        out = self.model(x)
        out[0, cls].backward()
        w = self.g.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((w * self.a).sum(1)).squeeze(0)
        cam -= cam.min()
        cam /= cam.max() + 1e-8
        return cam.cpu().numpy()


class SceneClassifier:
    def __init__(self, cfg=CFG):
        self.cfg = cfg
        self.device = ("cuda" if torch.cuda.is_available()
                       else "mps" if torch.backends.mps.is_available() else "cpu")

    def _loaders(self, train_dir, val_dir):
        c = self.cfg
        norm = transforms.Normalize(c.imagenet_mean, c.imagenet_std)
        train_tf = transforms.Compose([
            transforms.Resize((c.img_size, c.img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(12),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.ToTensor(), norm])
        eval_tf = transforms.Compose([
            transforms.Resize((c.img_size, c.img_size)), transforms.ToTensor(), norm])
        tr = ImageFolder(str(train_dir), transform=train_tf)
        va = ImageFolder(str(val_dir), transform=eval_tf)
        return (DataLoader(tr, batch_size=c.batch_size, shuffle=True),
                DataLoader(va, batch_size=c.batch_size, shuffle=False), tr.classes)

    def _model(self, n):
        m = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        for p in m.parameters():
            p.requires_grad = False
        for p in m.features[-3:].parameters():   # fine-tune last feature blocks
            p.requires_grad = True
        m.classifier[1] = nn.Linear(m.last_channel, n)
        return m.to(self.device)

    def _epoch(self, model, loader, crit, opt=None):
        train = opt is not None
        model.train(train)
        tot = corr = 0
        loss_sum = 0.0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            with torch.set_grad_enabled(train):
                out = model(x)
                loss = crit(out, y)
                if train:
                    opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += loss.item() * x.size(0)
            corr += (out.argmax(1) == y).sum().item()
            tot += x.size(0)
        return loss_sum / tot, corr / tot

    def _denorm(self, x):
        m, s = np.array(self.cfg.imagenet_mean), np.array(self.cfg.imagenet_std)
        return np.clip(x.cpu().numpy().transpose(1, 2, 0) * s + m, 0, 1)

    def _gradcam_panel(self, model, val_ld, classes, out_path):
        import cv2
        import matplotlib.pyplot as plt

        cam = _GradCAM(model, model.features[-1])
        model.eval()
        ok, bad = [], []
        for x, y in val_ld:
            for i in range(x.size(0)):
                xi = x[i:i + 1].to(self.device)
                pred = model(xi).argmax(1).item()
                rec = (xi, int(y[i]), pred)
                if pred == y[i].item() and len(ok) < 2:
                    ok.append(rec)
                elif pred != y[i].item() and len(bad) < 1:
                    bad.append(rec)
            if len(ok) >= 2 and len(bad) >= 1:
                break
        samples = [("correct", s) for s in ok] + [("incorrect", s) for s in bad]
        fig, axes = plt.subplots(1, len(samples), figsize=(4.0 * len(samples), 4.4))
        axes = np.atleast_1d(axes)
        for ax, (kind, (xi, t, p)) in zip(axes, samples):
            heat = cam.map(xi, p)
            img = self._denorm(xi[0])
            heat = cv2.resize(heat, (img.shape[1], img.shape[0]))
            hc = cv2.applyColorMap(np.uint8(255 * heat), cv2.COLORMAP_JET)[:, :, ::-1] / 255.0
            ax.imshow(np.clip(0.55 * img + 0.45 * hc, 0, 1))
            ax.set_title(f"{kind.upper()}\ntrue={classes[t]} pred={classes[p]}",
                         color="#1b5e20" if kind == "correct" else "#c62828", fontsize=10)
            ax.axis("off")
        fig.suptitle("Task 4 — Grad-CAM (2 correct + 1 incorrect)", fontweight="bold")
        fig.tight_layout(); fig.savefig(out_path, dpi=140, bbox_inches="tight"); plt.close(fig)
        return len(samples)

    def run(self, provider):
        say("TASK 4 — Scene Classification (MobileNetV2)")
        say(f"  device = {self.device}")
        out_dir = self.cfg.task_dir("task4")
        train_dir, val_dir, _ = provider.scene_dataset()
        train_ld, val_ld, classes = self._loaders(train_dir, val_dir)
        say(f"  classes = {classes}")

        model = self._model(len(classes))
        crit = nn.CrossEntropyLoss()
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                               lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        hist = {k: [] for k in ("tl", "ta", "vl", "va")}
        best = 0.0
        for ep in range(1, self.cfg.epochs + 1):
            tl, ta = self._epoch(model, train_ld, crit, opt)
            vl, va = self._epoch(model, val_ld, crit)
            for k, v in zip(("tl", "ta", "vl", "va"), (tl, ta, vl, va)):
                hist[k].append(v)
            best = max(best, va)
            say(f"  epoch {ep}/{self.cfg.epochs}  train_acc={ta:.3f}  val_acc={va:.3f}")

        import matplotlib.pyplot as plt
        ep = range(1, self.cfg.epochs + 1)
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
        a1.plot(ep, hist["tl"], "o-", label="train"); a1.plot(ep, hist["vl"], "s-", label="val")
        a1.set_title("Loss"); a1.set_xlabel("epoch"); a1.legend(); a1.grid(alpha=0.3)
        a2.plot(ep, hist["ta"], "o-", label="train"); a2.plot(ep, hist["va"], "s-", label="val")
        a2.axhline(self.cfg.target_acc, color="#c62828", ls="--", label="target 70%")
        a2.set_title("Accuracy"); a2.set_xlabel("epoch"); a2.legend(); a2.grid(alpha=0.3)
        fig.suptitle("Task 4 — training / validation curves", fontweight="bold")
        fig.tight_layout(); fig.savefig(out_dir / "training_curves.png", dpi=140); plt.close(fig)

        model.eval()
        yt, yp = [], []
        with torch.no_grad():
            for x, y in val_ld:
                yp += model(x.to(self.device)).argmax(1).cpu().tolist()
                yt += y.tolist()
        cm = confusion_matrix(yt, yp, labels=list(range(len(classes))))
        fig, ax = plt.subplots(figsize=(5.2, 4.6))
        im = ax.imshow(cm, cmap="Greens")
        ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=45, ha="right")
        ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.set_title("Task 4 — confusion matrix", fontweight="bold")
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        fig.colorbar(im, fraction=0.046, pad=0.04)
        fig.tight_layout(); fig.savefig(out_dir / "confusion_matrix.png", dpi=140); plt.close(fig)

        rep = classification_report(yt, yp, labels=list(range(len(classes))),
                                    target_names=classes, output_dict=True, zero_division=0)
        rows = [[c, f"{rep[c]['precision']:.2f}", f"{rep[c]['recall']:.2f}",
                 f"{rep[c]['f1-score']:.2f}", int(rep[c]['support'])] for c in classes]
        render_table(rows, ["class", "precision", "recall", "F1", "support"],
                     out_dir / "per_class_metrics.png", title="Task 4 — per-class metrics")

        n_cam = self._gradcam_panel(model, val_ld, classes, out_dir / "gradcam.png")
        render_table([["backbone", "mobilenet_v2"], ["dataset", "EuroSAT (land-use)"],
                      ["classes", ", ".join(classes)],
                      ["images/class (train)", self.cfg.train_per_class],
                      ["best val accuracy", f"{best:.3f}"],
                      [f"target {self.cfg.target_acc:.0%} met?", "YES" if best >= self.cfg.target_acc else "NO"],
                      ["Grad-CAM samples", n_cam]],
                     ["metric", "value"], out_dir / "task4_summary.png", title="Task 4 — summary")
        torch.save(model.state_dict(), out_dir / "mobilenet_scene.pth")
        say(f"  best val acc={best:.3f} -> {out_dir}")
        return {"best_acc": best, "final_acc": hist["va"][-1], "classes": classes,
                "per_class": rows, "confusion": cm.tolist(),
                "target_met": bool(best >= self.cfg.target_acc)}
