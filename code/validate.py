"""Per-axis scoring against the 30 labelled rows in sample_messages.csv.

The five scored axes (LEITWERK.md sec 2) each get their own number so we can
see which layer moves which axis. No axis is averaged away into one figure.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import config as C


@dataclass
class AxisScores:
    n: int = 0
    action_correct: int = 0
    type_correct: int = 0
    both_correct: int = 0
    reason_exact: int = 0
    evidence_hit: int = 0
    evidence_exact: int = 0
    evidence_applicable: int = 0
    abs_conf_error: float = 0.0
    in_band: int = 0
    confusion: dict[tuple[str, str], int] = field(default_factory=dict)
    misses: list[dict] = field(default_factory=list)

    # ---- rates -------------------------------------------------------------
    @property
    def action_acc(self) -> float:
        return self.action_correct / self.n if self.n else 0.0

    @property
    def type_acc(self) -> float:
        return self.type_correct / self.n if self.n else 0.0

    @property
    def joint_acc(self) -> float:
        return self.both_correct / self.n if self.n else 0.0

    @property
    def reason_acc(self) -> float:
        return self.reason_exact / self.n if self.n else 0.0

    @property
    def evidence_recall(self) -> float:
        return (
            self.evidence_hit / self.evidence_applicable
            if self.evidence_applicable
            else 0.0
        )

    @property
    def mae_conf(self) -> float:
        return self.abs_conf_error / self.n if self.n else 0.0

    @property
    def band_rate(self) -> float:
        return self.in_band / self.n if self.n else 0.0


def score(predictions: list[dict[str, str]], gold_rows: list[dict[str, str]]) -> AxisScores:
    gold_by_id = {g["message_id"]: g for g in gold_rows}
    sc = AxisScores()

    for pred in predictions:
        gold = gold_by_id.get(pred["message_id"])
        if gold is None:
            continue
        sc.n += 1

        a_ok = pred["action"] == gold["action"]
        t_ok = pred["message_type"] == gold["message_type"]
        sc.action_correct += a_ok
        sc.type_correct += t_ok
        sc.both_correct += a_ok and t_ok

        key = (gold["action"], pred["action"])
        sc.confusion[key] = sc.confusion.get(key, 0) + 1

        # Reason: exact string match against the gold template bank. This is a
        # deliberately harsh proxy - the real axis is graded on usefulness and
        # consistency - but exact match is the only objective signal available
        # and it tracks the template-bank hypothesis directly.
        sc.reason_exact += pred["reason"].strip() == gold["reason"].strip()

        gold_ev = {
            e.strip()
            for e in gold["evidence_message_ids"].split(C.EVIDENCE_SEPARATOR)
            if e.strip() and e.strip() != C.NO_EVIDENCE
        }
        pred_ev = {
            e.strip()
            for e in pred["evidence_message_ids"].split(C.EVIDENCE_SEPARATOR)
            if e.strip() and e.strip() != C.NO_EVIDENCE
        }
        if gold_ev:
            sc.evidence_applicable += 1
            if gold_ev & pred_ev:
                sc.evidence_hit += 1
            if gold_ev == pred_ev:
                sc.evidence_exact += 1
        elif not pred_ev:
            # Gold said `none` and so did we. Counts as an exact agreement.
            sc.evidence_exact += 1

        try:
            gold_conf = float(gold["confidence"])
            pred_conf = float(pred["confidence"])
        except ValueError:
            gold_conf = pred_conf = 0.0
        sc.abs_conf_error += abs(gold_conf - pred_conf)
        sc.in_band += C.CONF_FLOOR <= pred_conf <= C.CONF_CEIL

        if not (a_ok and t_ok):
            sc.misses.append(
                {
                    "message_id": pred["message_id"],
                    "gold": f"{gold['action']}/{gold['message_type']}",
                    "pred": f"{pred['action']}/{pred['message_type']}",
                    "text": (gold.get("message_text") or "")[:88].replace("\n", " "),
                    "media": gold.get("media_type") or "text",
                }
            )

    return sc


def report(sc: AxisScores, title: str = "SAMPLE VALIDATION") -> str:
    lines = [
        "",
        "=" * 72,
        f"  {title}  (n={sc.n})",
        "=" * 72,
        f"  action accuracy        {sc.action_acc:6.1%}   ({sc.action_correct}/{sc.n})",
        f"  message_type accuracy  {sc.type_acc:6.1%}   ({sc.type_correct}/{sc.n})",
        f"  joint (both correct)   {sc.joint_acc:6.1%}   ({sc.both_correct}/{sc.n})",
        f"  reason exact match     {sc.reason_acc:6.1%}   ({sc.reason_exact}/{sc.n})",
        f"  evidence recall@2      {sc.evidence_recall:6.1%}   "
        f"({sc.evidence_hit}/{sc.evidence_applicable} rows with gold evidence)",
        f"  evidence exact set     {sc.evidence_exact}/{sc.n}",
        f"  confidence MAE         {sc.mae_conf:6.3f}",
        f"  confidence in band     {sc.band_rate:6.1%}   "
        f"[{C.CONF_FLOOR}, {C.CONF_CEIL}]",
        "-" * 72,
        "  action confusion (gold -> pred):",
    ]
    for gold_a in C.ACTIONS:
        row = "    " + f"{gold_a:>7} -> " + "  ".join(
            f"{p}:{sc.confusion.get((gold_a, p), 0)}" for p in C.ACTIONS
        )
        lines.append(row)

    if sc.misses:
        lines.append("-" * 72)
        lines.append(f"  misses ({len(sc.misses)}):")
        for m in sc.misses:
            lines.append(
                f"    {m['message_id']:<16} gold={m['gold']:<22} pred={m['pred']:<22} "
                f"[{m['media']}]"
            )
            if m["text"]:
                lines.append(f"        {m['text']}")
    lines.append("=" * 72)
    return "\n".join(lines)
