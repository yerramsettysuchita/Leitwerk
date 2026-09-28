"""Score the shipped pipeline on the 30 labelled rows in sample_messages.csv.

    python code/evaluation/main.py

A thin wrapper. It runs the same pipeline.run() that writes output.csv and
scores it with validate.score(), so there is no second scoring path that could
drift from `python code/main.py --validate`. Writes nothing except the media
cache, which perceive.py rewrites unchanged on every run.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from calibrate import Calibrator
from loader import load_dataset
from pipeline import RunConfig, run
from retrieve import EvidenceIndex
from validate import report, score

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def main() -> int:
    ds = load_dataset()
    cfg = RunConfig()
    predictions, _ = run(ds.samples, ds, EvidenceIndex(ds), cfg, Calibrator.load())
    print(report(score(predictions, ds.samples),
                 f"EVALUATION [{cfg.label()}] - sample_messages.csv"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
