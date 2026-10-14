"""CS-571 CCP — single-command entry point.

Examples:
    python main.py                 # all tasks (real data) + report
    python main.py --task 3        # one task
    python main.py --data synthetic  # fully offline
"""

import argparse
import json

from visionpipe.settings import CFG, Settings
from visionpipe.io_tools import say, seed_everything
from visionpipe.datasets import DataProvider
from visionpipe.edges import EdgePipeline
from visionpipe.features import FeatureMatcher
from visionpipe.mosaic import MosaicBuilder
from visionpipe.classifier import SceneClassifier
from visionpipe.report import ReportWriter


def _json_safe(o):
    import numpy as np
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return o


class Pipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        self.provider = DataProvider(cfg.data_mode)

    def run(self, which):
        results = {}
        if which in ("all", "1"):   
            results["task1"] = EdgePipeline(self.cfg).run(self.provider)
        if which in ("all", "2"):
            results["task2"] = FeatureMatcher(self.cfg).run(self.provider)
        if which in ("all", "3"):
            results["task3"] = MosaicBuilder(self.cfg).run(self.provider)
        if which in ("all", "4"):
            results["task4"] = SceneClassifier(self.cfg).run(self.provider)
        (self.cfg.results_dir / "results.json").write_text(json.dumps(_json_safe(results), indent=2))
        say(f"wrote {self.cfg.results_dir}/results.json")
        if which == "all":
            ReportWriter(self.cfg).run(results)
        return results


def main():
    ap = argparse.ArgumentParser(description="CS-571 CCP pipeline")
    ap.add_argument("--task", default="all", choices=["all", "1", "2", "3", "4"])
    ap.add_argument("--data", default=None, choices=["auto", "synthetic", "real"])
    args = ap.parse_args()

    cfg = CFG if not args.data else Settings(data_mode=args.data)
    seed_everything(cfg.seed)
    for d in cfg.all_dirs():
        d.mkdir(parents=True, exist_ok=True)
    say(f"CS-571 CCP | task={args.task} | data_mode={cfg.data_mode}")
    Pipeline(cfg).run(args.task)
    say("DONE.")


if __name__ == "__main__":
    main()
