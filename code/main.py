"""LEITWERK - the steering unit for attention.

Entry point for the HackerRank Orchestrate (August 2026) Message Notification
Router submission.

    python code/main.py --run         write output.csv for all 110 dataset rows
    python code/main.py --validate    score on the 30 labelled samples
    python code/main.py --calibrate   refit the confidence temperature
    python code/main.py --all         calibrate, run, then validate
    python code/main.py --engine bayes --validate

Reads only from dataset/. Deterministic: no sampling anywhere, fixed sort
orders, cached media perception keyed by file hash. Two consecutive runs
produce byte-identical output.csv.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as C
from calibrate import (
    Calibrator,
    expected_calibration_error,
    fit,
    reliability,
)
from loader import Dataset, load_dataset
from pipeline import RunConfig, process, run
from retrieve import EvidenceIndex
from validate import report, score

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def write_output(predictions: list[dict[str, str]], path: Path = C.OUTPUT_CSV) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=C.OUTPUT_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(predictions)


def write_trace(traces: list[dict], path: Path = C.TRACE_JSONL) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for t in traces:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Contract gate
# ---------------------------------------------------------------------------

def verify_contract(predictions: list[dict[str, str]], ds: Dataset) -> list[str]:
    """Hard gate on the output contract. Returns a list of violations."""
    problems: list[str] = []

    expected = ds.output_ids
    got = [p["message_id"] for p in predictions]
    if len(got) != len(expected):
        problems.append(f"row count {len(got)} != required {len(expected)}")
    if got != expected:
        missing, extra = set(expected) - set(got), set(got) - set(expected)
        if missing:
            problems.append(f"missing message_ids: {sorted(missing)[:5]}")
        if extra:
            problems.append(f"unexpected message_ids: {sorted(extra)[:5]}")
        if not missing and not extra:
            problems.append("message_ids present but out of template order")

    known_history = set(ds.history_by_id)
    for p in predictions:
        if p["action"] not in C.ACTIONS:
            problems.append(f"{p['message_id']}: illegal action {p['action']!r}")
        if p["message_type"] not in C.MESSAGE_TYPES:
            problems.append(f"{p['message_id']}: illegal type {p['message_type']!r}")
        if not p["reason"].strip():
            problems.append(f"{p['message_id']}: empty reason")
        try:
            conf = float(p["confidence"])
            if not 0.0 <= conf <= 1.0:
                problems.append(f"{p['message_id']}: confidence {conf} out of [0,1]")
        except ValueError:
            problems.append(f"{p['message_id']}: non-numeric confidence")
        ev = p["evidence_message_ids"]
        if ev != C.NO_EVIDENCE:
            for mid in ev.split(C.EVIDENCE_SEPARATOR):
                if mid.strip() not in known_history:
                    problems.append(f"{p['message_id']}: fabricated evidence id {mid!r}")

    return problems


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_calibrate(ds: Dataset, index: EvidenceIndex, cfg: RunConfig) -> Calibrator:
    """Fit the single temperature scalar on the labelled samples."""
    from envelope import build_envelopes
    from perceive import perceive, save_cache

    envelopes = build_envelopes(ds.samples, ds)
    for env in envelopes:
        env.perception = perceive(env, enabled=cfg.perception)
    save_cache()

    gold_by_id = {g["message_id"]: g for g in ds.samples}
    points: list[tuple[str, float, float]] = []
    for env in envelopes:
        r = process(env, ds, index, cfg)
        gold = gold_by_id[env.message_id]
        points.append((r.decision.action, r.raw_posterior, float(gold["confidence"])))

    calibrator, mae = fit(points)
    calibrator.save()
    print(
        f"\ncalibration: temperature T={calibrator.temperature:.2f} "
        f"fit on {len(points)} labelled rows (MAE {mae:.4f}) -> calibration.json"
    )
    return calibrator


def cmd_run(ds: Dataset, index: EvidenceIndex, cfg: RunConfig, cal: Calibrator) -> None:
    predictions, traces = run(ds.messages, ds, index, cfg, cal)
    problems = verify_contract(predictions, ds)

    write_output(predictions)
    write_trace(traces)

    print(f"\nengine: {cfg.label()}")
    print(f"wrote {C.OUTPUT_CSV.name}  ({len(predictions)} rows)")
    print(f"wrote {C.TRACE_JSONL.name}  ({len(traces)} traces)")

    dist: dict[str, int] = {}
    tdist: dict[str, int] = {}
    ev_none = forced = perceived = 0
    for p, t in zip(predictions, traces):
        dist[p["action"]] = dist.get(p["action"], 0) + 1
        tdist[p["message_type"]] = tdist.get(p["message_type"], 0) + 1
        ev_none += p["evidence_message_ids"] == C.NO_EVIDENCE
        forced += bool(t.get("safety_forced"))
        perceived += t.get("perception_source", "none") != "none"

    print("\naction: " + "  ".join(f"{a}={dist.get(a, 0)}" for a in C.ACTIONS))
    print("type:   " + "  ".join(f"{t}={tdist[t]}" for t in C.MESSAGE_TYPES if tdist.get(t)))
    print(f"evidence `none`: {ev_none}/{len(predictions)}   "
          f"media perceived: {perceived}   safety-forced mutes: {forced}")

    confs = [float(p["confidence"]) for p in predictions]
    print(f"confidence: min={min(confs):.2f} mean={sum(confs)/len(confs):.3f} max={max(confs):.2f}")

    if problems:
        print("\nCONTRACT VIOLATIONS:")
        for p in problems:
            print(f"  - {p}")
        raise SystemExit(2)
    print("\ncontract OK: row count and order, exact columns, legal values, real evidence ids")


def cmd_validate(ds: Dataset, index: EvidenceIndex, cfg: RunConfig, cal: Calibrator) -> None:
    predictions, _ = run(ds.samples, ds, index, cfg, cal)
    sc = score(predictions, ds.samples)
    print(report(sc, f"VALIDATION [{cfg.label()}] - sample_messages.csv"))

    table = reliability(predictions, ds.samples)
    print("\n  reliability (does stated confidence track accuracy?)")
    print(f"    {'bin':<14}{'n':>4}{'mean conf':>12}{'accuracy':>11}{'gap':>8}")
    for row in table:
        print(f"    {row['bin']:<14}{row['n']:>4}{row['mean_confidence']:>12.3f}"
              f"{row['accuracy']:>11.3f}{row['gap']:>8.3f}")
    print(f"    expected calibration error: {expected_calibration_error(table):.4f}\n")


def main() -> int:
    parser = argparse.ArgumentParser(prog="leitwerk", description=__doc__)
    parser.add_argument("--run", action="store_true", help="write output.csv")
    parser.add_argument("--validate", action="store_true", help="score on labelled samples")
    parser.add_argument("--calibrate", action="store_true", help="refit confidence temperature")
    parser.add_argument("--all", action="store_true", help="calibrate, run, validate")
    parser.add_argument("--engine", default="hybrid", choices=("rules", "bayes", "hybrid"))
    parser.add_argument("--no-perception", action="store_true", help="skip OCR/ASR")
    args = parser.parse_args()

    if not (args.run or args.validate or args.calibrate or args.all):
        parser.print_help()
        return 1

    cfg = RunConfig(engine=args.engine, perception=not args.no_perception)
    ds = load_dataset()
    index = EvidenceIndex(ds)

    if args.calibrate or args.all:
        cal = cmd_calibrate(ds, index, cfg)
    else:
        cal = Calibrator.load()

    if args.run or args.all:
        cmd_run(ds, index, cfg, cal)
    if args.validate or args.all:
        cmd_validate(ds, index, cfg, cal)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
