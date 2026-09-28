"""Phase 3 - the orchestration surface (LEITWERK.md sec 10).

Exposes each of the seven stages as an HTTP endpoint so an external orchestrator
(n8n, Lyzr, or anything else that speaks JSON) can render the pipeline as a
visible DAG and drive it step by step.

    python code/serve.py --port 8420

**The non-negotiable rule**: this layer WRAPS the core, it never becomes a
dependency of it. `python code/main.py --run` produces the submission with this
server stopped, uninstalled, or on fire. Nothing in code/ imports serve.py.
Phase 3 exists for the demo and for the "Orchestrate" framing; it is not on the
path to output.csv.

Endpoints, all POST with a JSON body `{"message_id": "msg_001"}`:

    /s1/perceive     normalized text + OCR/ASR facts
    /s2/featurize    the deterministic signal vector
    /s3/retrieve     evidence ids with scores
    /s4/believe      importance, risk, type posterior
    /s5/defend       prosecution and defence cases, verdict
    /s6/decide       expected losses and chosen action
    /s7/express      the final output row
    /route           all seven, one call, with the full trace
    /health          liveness + what the core loaded
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, is_dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from believe import believe
from decide import decide as decide_rules
from defend import adjudicate
from envelope import build_envelope
from express import build_row
from features import extract
from loader import load_dataset
from losses import decide_by_loss
from perceive import perceive
from pipeline import RunConfig, process
from retrieve import EvidenceIndex, format_evidence

_DS = None
_INDEX = None
_ROWS: dict[str, dict] = {}
_CFG = RunConfig()


def _boot() -> None:
    global _DS, _INDEX, _ROWS
    _DS = load_dataset()
    _INDEX = EvidenceIndex(_DS)
    _ROWS = {r["message_id"]: r for r in (_DS.messages + _DS.samples)}


def _envelope(message_id: str):
    row = _ROWS.get(message_id)
    if row is None:
        raise KeyError(message_id)
    env = build_envelope(row, _DS)
    env.perception = perceive(env, enabled=_CFG.perception)
    return env


def _plain(obj):
    """Make dataclasses and tuples JSON-safe without leaking objects."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _plain(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_plain(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    return obj


# ---------------------------------------------------------------------------
# Stage handlers
# ---------------------------------------------------------------------------

def h_perceive(mid: str) -> dict:
    env = _envelope(mid)
    return {
        "message_id": mid,
        "text": env.text,
        "media_type": env.media_type,
        "perception_source": env.perception.source,
        "ocr_text": env.perception.ocr_text,
        "transcript": env.perception.transcript,
        "effective_text": env.effective_text,
    }


def h_featurize(mid: str) -> dict:
    env = _envelope(mid)
    sig = extract(env, _DS)
    return {"message_id": mid, "fired": sig.fired, "signals": _plain(sig)}


def h_retrieve(mid: str) -> dict:
    env = _envelope(mid)
    sig = extract(env, _DS)
    ids = _INDEX.select(env, sig)
    return {
        "message_id": mid,
        "evidence_message_ids": format_evidence(ids),
        "evidence": [
            {"id": i, "text": (_DS.history_by_id.get(i, {}).get("message_text") or "")[:200]}
            for i in ids
        ],
    }


def h_believe(mid: str) -> dict:
    env = _envelope(mid)
    sig = extract(env, _DS)
    b = believe(env, sig, use_decay=_CFG.decay, use_fatigue=_CFG.fatigue)
    return {
        "message_id": mid,
        "importance": round(b.importance, 4),
        "risk": round(b.risk, 4),
        "top_type": b.top_type,
        "type_entropy": round(b.type_entropy, 4),
        "type_posterior": {k: round(v, 4) for k, v in sorted(
            b.type_posterior.items(), key=lambda kv: -kv[1])},
        "importance_terms": b.importance_terms,
        "risk_terms": b.risk_terms,
    }


def h_defend(mid: str) -> dict:
    env = _envelope(mid)
    sig = extract(env, _DS)
    b = believe(env, sig, use_decay=_CFG.decay, use_fatigue=_CFG.fatigue)
    v = adjudicate(env, sig, b)
    return {
        "message_id": mid,
        "adjudicated_risk": round(v.risk, 4),
        "safety_forced": v.forced,
        "forced_action": v.forced_action,
        "forced_type": v.forced_type,
        "prosecution": [{"claim": c, "weight": round(w, 3)} for c, w in v.prosecution],
        "defence": [{"claim": c, "weight": round(w, 3)} for c, w in v.defence],
        "note": v.note,
    }


def h_decide(mid: str) -> dict:
    env = _envelope(mid)
    sig = extract(env, _DS)
    b = believe(env, sig, use_decay=_CFG.decay, use_fatigue=_CFG.fatigue)
    v = adjudicate(env, sig, b)
    lo = decide_by_loss(env, sig, b, v)
    rules = decide_rules(env, sig)
    return {
        "message_id": mid,
        "expected_loss": {k: round(x, 4) for k, x in lo.expected_loss.items()},
        "loss_argmin": lo.action,
        "loss_margin": round(lo.margin, 4),
        "ambiguous": lo.ambiguous,
        "tiebreaker": lo.tiebreaker,
        "safety_forced": lo.forced,
        "cascade_rule": rules.rule,
        "cascade_action": rules.action,
        "cascade_type": rules.message_type,
    }


def h_express(mid: str) -> dict:
    env = _envelope(mid)
    r = process(env, _DS, _INDEX, _CFG)
    from calibrate import Calibrator

    conf = Calibrator.load()(r.decision.action, r.raw_posterior)
    return build_row(r.env, r.sig, r.decision, r.evidence, conf)


def h_route(mid: str) -> dict:
    env = _envelope(mid)
    r = process(env, _DS, _INDEX, _CFG)
    from calibrate import Calibrator

    conf = Calibrator.load()(r.decision.action, r.raw_posterior)
    return {
        "row": build_row(r.env, r.sig, r.decision, r.evidence, conf),
        "stages": {
            "s1_perceive": h_perceive(mid),
            "s2_featurize": {"fired": r.sig.fired},
            "s3_retrieve": {"evidence": r.evidence},
            "s4_believe": {
                "importance": round(r.belief.importance, 4),
                "risk": round(r.belief.risk, 4),
                "top_type": r.belief.top_type,
            },
            "s5_defend": {
                "adjudicated_risk": round(r.verdict.risk, 4),
                "forced": r.verdict.forced,
                "prosecution": [c for c, _ in r.verdict.prosecution],
                "defence": [c for c, _ in r.verdict.defence],
            },
            "s6_decide": {
                "expected_loss": {k: round(x, 4) for k, x in r.loss.expected_loss.items()},
                "rule": r.decision.rule,
            },
            "s7_express": {"raw_posterior": round(r.raw_posterior, 4), "confidence": conf},
        },
    }


ROUTES = {
    "/s1/perceive": h_perceive,
    "/s2/featurize": h_featurize,
    "/s3/retrieve": h_retrieve,
    "/s4/believe": h_believe,
    "/s5/defend": h_defend,
    "/s6/decide": h_decide,
    "/s7/express": h_express,
    "/route": h_route,
}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(_plain(payload), ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._send(200, {
                "status": "ok",
                "messages": len(_DS.messages),
                "samples": len(_DS.samples),
                "history": len(_DS.history),
                "stages": sorted(ROUTES),
                "note": "wraps the core; output.csv never depends on this server",
            })
        else:
            self._send(404, {"error": "unknown path", "stages": sorted(ROUTES)})

    def do_POST(self) -> None:  # noqa: N802
        handler = ROUTES.get(self.path)
        if handler is None:
            self._send(404, {"error": "unknown stage", "stages": sorted(ROUTES)})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            mid = body.get("message_id")
            if not mid:
                self._send(400, {"error": "message_id is required"})
                return
            self._send(200, handler(mid))
        except KeyError as exc:
            self._send(404, {"error": f"unknown message_id: {exc}"})
        except Exception as exc:  # pragma: no cover
            self._send(500, {"error": type(exc).__name__, "detail": str(exc)})

    def log_message(self, fmt, *args) -> None:
        sys.stderr.write(f"  [serve] {fmt % args}\n")


def main() -> int:
    parser = argparse.ArgumentParser(prog="leitwerk-serve", description=__doc__)
    parser.add_argument("--port", type=int, default=8420)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    _boot()
    print(f"LEITWERK stages on http://{args.host}:{args.port}")
    print(f"  stages: {', '.join(sorted(ROUTES))}")
    print("  this server wraps the core. output.csv does not depend on it.")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
