"""S7 EXPRESS - confidence placement, row emission, and the audit trace.

Phase 1 confidence is a band-anchored heuristic: the action determines the band,
the firing rule's signal strength places the value inside it. Phase 2 replaces
the placement with a temperature-scaled posterior fit on the 30 labelled samples
(calibrate.py); the band clamp stays either way, because every gold confidence
lies in [0.78, 0.91] and going outside it is off-distribution.
"""

from __future__ import annotations

import hashlib
import json

import config as C
from decide import Decision
from envelope import MessageEnvelope
from features import Signals
from reasons import render


def place_confidence(decision: Decision) -> float:
    """Position confidence inside the action's observed gold band."""
    low, high = C.CONF_ACTION_BAND[decision.action]
    anchor = C.CONF_ANCHOR[decision.action]
    # strength 0.5 sits on the anchor; 0 and 1 swing to the band edges.
    value = anchor + (decision.strength - 0.5) * 2 * C.CONF_SIGNAL_SWING
    value = max(low, min(high, value))
    value = max(C.CONF_FLOOR, min(C.CONF_CEIL, value))
    return round(value, 2)


def fingerprint(env: MessageEnvelope, sig: Signals, decision: Decision) -> str:
    """Stable 8-char hash of the inputs that produced this row, so any decision
    can be reproduced and diffed across runs."""
    payload = json.dumps(
        {
            "message_id": env.message_id,
            "text": env.effective_text,
            "fired": sig.fired,
            "rule": decision.rule,
            "action": decision.action,
            "type": decision.message_type,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]


def build_row(
    env: MessageEnvelope,
    sig: Signals,
    decision: Decision,
    evidence: str,
    confidence: float | None = None,
) -> dict[str, str]:
    assert decision.action in C.ACTIONS, f"illegal action {decision.action!r}"
    assert decision.message_type in C.MESSAGE_TYPES, (
        f"illegal message_type {decision.message_type!r}"
    )
    value = place_confidence(decision) if confidence is None else confidence
    return {
        "message_id": env.message_id,
        "action": decision.action,
        "message_type": decision.message_type,
        "reason": render(decision.reason_code),
        "confidence": f"{value:.2f}",
        "evidence_message_ids": evidence,
    }


def build_trace(
    env: MessageEnvelope,
    sig: Signals,
    decision: Decision,
    evidence: str,
    confidence: float | None = None,
) -> dict:
    value = place_confidence(decision) if confidence is None else confidence
    return {
        "message_id": env.message_id,
        "fingerprint": fingerprint(env, sig, decision),
        "user_id": env.user_id,
        "conversation_type": env.conversation_type,
        "media_type": env.media_type,
        "perceived": env.is_perceived,
        "rule": decision.rule,
        "action": decision.action,
        "message_type": decision.message_type,
        "reason_code": decision.reason_code,
        "strength": round(decision.strength, 3),
        "confidence": round(value, 2),
        "evidence": evidence,
        "signals_fired": sig.fired,
        "engagement_rate": round(sig.engagement_rate, 3),
        "domain_similarity": round(sig.domain_similarity, 3),
    }
