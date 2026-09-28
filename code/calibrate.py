"""S7 calibration - LEITWERK.md sec 8.

The scored axis is "reasonable confidence calibration". Most systems emit an
LLM-guessed 0.8 for everything. We emit a number derived from how decisive the
decision actually was, mapped through a single fitted temperature.

    confidence(a, raw) = clamp( anchor[a] + swing[a] * tanh((raw - 0.5) / T) )

One scalar, T, fit once on the 30 labelled samples by grid search on mean
absolute error against the gold confidences, then frozen into calibration.json
and logged. Per-action anchors and bands come from the observed gold
distribution: every gold confidence lies in [0.78, 0.91], and the band is
stratified by action.

The clamp is not cosmetic. A confidence of 0.99 on a routing decision is a
claim no evidence in this dataset supports, and the gold labels never make it.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import config as C

CALIBRATION_PATH = C.REPO_ROOT / "calibration.json"

DEFAULT_TEMPERATURE = 0.35


@dataclass
class Calibrator:
    temperature: float = DEFAULT_TEMPERATURE

    def __call__(self, action: str, raw: float) -> float:
        low, high = C.CONF_ACTION_BAND[action]
        anchor = C.CONF_ANCHOR[action]
        swing = max(high - anchor, anchor - low)
        value = anchor + swing * math.tanh((raw - 0.5) / max(self.temperature, 1e-3))
        value = max(low, min(high, value))
        return round(max(C.CONF_FLOOR, min(C.CONF_CEIL, value)), 2)

    def save(self, path=CALIBRATION_PATH) -> None:
        path.write_text(
            json.dumps(
                {
                    "temperature": round(self.temperature, 4),
                    "action_bands": {k: list(v) for k, v in C.CONF_ACTION_BAND.items()},
                    "anchors": C.CONF_ANCHOR,
                    "fitted_on": "dataset/sample_messages.csv (30 labelled rows)",
                    "note": (
                        "Single temperature scalar; per-action bands are the observed "
                        "gold distribution, not tuned."
                    ),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path=CALIBRATION_PATH) -> "Calibrator":
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return cls(temperature=float(data.get("temperature", DEFAULT_TEMPERATURE)))
        return cls()


def fit(samples: list[tuple[str, float, float]]) -> tuple[Calibrator, float]:
    """samples: (action, raw_posterior, gold_confidence). Returns the fitted
    calibrator and its mean absolute error."""
    best = (DEFAULT_TEMPERATURE, float("inf"))
    for step in range(1, 201):
        t = step * 0.01
        cal = Calibrator(temperature=t)
        mae = sum(abs(cal(a, raw) - gold) for a, raw, gold in samples) / max(len(samples), 1)
        if mae < best[1] - 1e-9:
            best = (t, mae)
    return Calibrator(temperature=best[0]), best[1]


def reliability(predictions: list[dict], gold_rows: list[dict], bins: int = 4) -> list[dict]:
    """Reliability table: within each confidence bin, does stated confidence
    track empirical accuracy? This is the calibration axis made visible."""
    gold_by_id = {g["message_id"]: g for g in gold_rows}
    buckets: dict[int, list[tuple[float, bool]]] = {}

    for p in predictions:
        g = gold_by_id.get(p["message_id"])
        if g is None:
            continue
        conf = float(p["confidence"])
        correct = p["action"] == g["action"]
        idx = min(
            int((conf - C.CONF_FLOOR) / max(C.CONF_CEIL - C.CONF_FLOOR, 1e-9) * bins),
            bins - 1,
        )
        buckets.setdefault(idx, []).append((conf, correct))

    width = (C.CONF_CEIL - C.CONF_FLOOR) / bins
    table = []
    for idx in sorted(buckets):
        rows = buckets[idx]
        mean_conf = sum(c for c, _ in rows) / len(rows)
        accuracy = sum(1 for _, ok in rows if ok) / len(rows)
        table.append(
            {
                "bin": f"{C.CONF_FLOOR + idx * width:.2f}-{C.CONF_FLOOR + (idx + 1) * width:.2f}",
                "n": len(rows),
                "mean_confidence": round(mean_conf, 3),
                "accuracy": round(accuracy, 3),
                "gap": round(mean_conf - accuracy, 3),
            }
        )
    return table


def expected_calibration_error(table: list[dict]) -> float:
    total = sum(row["n"] for row in table)
    if not total:
        return 0.0
    return sum(row["n"] * abs(row["gap"]) for row in table) / total
