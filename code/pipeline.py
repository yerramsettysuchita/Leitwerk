"""The seven-stage pipeline, parameterized.

One code path, three decision engines, every advanced layer independently
switchable. main.py and ablate.py both call this, so the ablation table
measures exactly the system that ships - there is no separate "eval" pipeline
that could drift from the real one.

Engines:
  rules   S1-S3 + the ordered rule cascade. The Phase 1 baseline.
  bayes   S1-S5 + argmin expected loss over the type posterior. The pure
          decision-theoretic arm.
  hybrid  the cascade decides, the defence layer's hard safety constraint can
          override it, and the belief posterior supplies confidence. Ships.

`hybrid` is not a hedge. The cascade encodes high-precision structure that a
smooth posterior blurs; the posterior contributes exactly the two things the
cascade cannot produce - a computed confidence and a safety constraint that
cannot be outvoted. The ablation table is what decided this, not taste.
"""

from __future__ import annotations

from dataclasses import dataclass

from believe import Belief, believe
from decide import Decision, decide as decide_rules
from defend import Verdict, adjudicate
from envelope import MessageEnvelope, build_envelopes
from express import build_row, build_trace
from features import Signals, extract
from loader import Dataset
import critic
from losses import AMBIGUITY_MARGIN, LossOutcome, decide_by_loss
from perceive import perceive, save_cache
from retrieve import EvidenceIndex, format_evidence

# Null result (see _blend_posterior): evidence-weighted confidence
# measurably worsened MAE against gold, so it ships disabled.
USE_EVIDENCE_WEIGHTED_CONFIDENCE = False

# Cascade branches that express "no specific rule applied" rather than positive
# knowledge. These are the rows the posterior is allowed to override.
GENERIC_FALLBACK_RULES = frozenset({
    "E8_media_unperceived",
    "E10_trusted_not_urgent",
    "E11_default",
})


@dataclass(frozen=True)
class RunConfig:
    engine: str = "hybrid"          # rules | bayes | hybrid
    perception: bool = True         # S1 OCR / ASR
    retrieval: bool = True          # S3 evidence
    defence: bool = True            # S5 hard safety constraint
    matrix: bool = True             # S6 integrate over the posterior
    decay: bool = True              # sec 4.5 temporal relevance
    budget: bool = True             # sec 6 attention budget
    calibration: bool = True        # sec 8 temperature scaling
    fatigue: bool = True            # sec 4.6
    semantic: bool = True           # S2 embedding intents (semantic.py)
    critic: bool = True             # S6b LLM critic (no-op without a key)

    def label(self) -> str:
        off = [
            name
            for name, on in (
                ("perception", self.perception),
                ("retrieval", self.retrieval),
                ("defence", self.defence),
                ("matrix", self.matrix),
                ("decay", self.decay),
                ("budget", self.budget),
                ("calibration", self.calibration),
                ("fatigue", self.fatigue),
                ("semantic", self.semantic),
                ("critic", self.critic),
            )
            if not on
        ]
        suffix = f" -{','.join(off)}" if off else ""
        return f"{self.engine}{suffix}"


@dataclass
class StageResult:
    """Everything the seven stages produced for one message."""

    env: MessageEnvelope
    sig: Signals
    belief: Belief
    verdict: Verdict
    loss: LossOutcome
    decision: Decision
    evidence: str
    critic_note: str
    raw_posterior: float


def _blend_posterior(
    decision: Decision,
    loss: LossOutcome,
    belief: Belief,
    sig: Signals | None = None,
    evidence: str = "",
) -> float:
    """The raw, uncalibrated confidence in the chosen action.

    Five sources of decisiveness. The first three say how cleanly the decision
    itself resolved; the last two say how much this user's history actually
    supports it, which is what stops a first-contact call from being stated as
    confidently as a well-evidenced one.

      rule strength      how cleanly the firing rule matched
      loss margin        gap to the runner-up action in expected loss
      type certainty     concentration of the type posterior
      evidence support   how much relevant history was retrieved
      relationship depth how much behavioural history exists at all

    MEASURED RESULT: the evidence-count and relationship-depth terms are
    implemented below but DISABLED, because adding them was a measured
    regression - confidence MAE against the gold values moved 0.017 -> 0.019
    while no other axis improved. Widening the emitted band to let them create
    real spread was worse still (0.031). The gold confidence distribution is
    genuinely narrow, so matching it means being narrow, and "more spread is
    better calibration" does not survive contact with this target. Kept visible
    rather than deleted so the null result is auditable.
    """
    type_certainty = max(belief.type_posterior.values())
    parts = [decision.strength, loss.posterior, type_certainty]

    if sig is not None and USE_EVIDENCE_WEIGHTED_CONFIDENCE:
        n_evidence = 0 if (not evidence or evidence == "none") else len(evidence.split(";"))
        parts.append(min(n_evidence / 2.0, 1.0))
        # Beta concentration: how many recorded interactions back the prior.
        history = sig.opens + sig.replies + sig.dismissals + sig.muted_after + sig.reports
        parts.append(min(history / 8.0, 1.0))

    return sum(parts) / len(parts)


def process(
    env: MessageEnvelope,
    ds: Dataset,
    index: EvidenceIndex | None,
    cfg: RunConfig,
) -> StageResult:
    sig = extract(env, ds)                                            # S2
    evidence_ids = index.select(env, sig) if (index and cfg.retrieval) else []
    evidence = format_evidence(evidence_ids)                          # S3
    belief = believe(env, sig, use_decay=cfg.decay, use_fatigue=cfg.fatigue)  # S4
    verdict = adjudicate(env, sig, belief)                            # S5

    loss = decide_by_loss(                                            # S6
        env, sig, belief, verdict,
        use_matrix=cfg.matrix,
        use_defence=cfg.defence,
        use_decay=cfg.decay,
        use_budget=cfg.budget,
    )
    rules = decide_rules(env, sig)
    critic_note = ""

    if cfg.engine == "rules":
        decision = rules
    elif cfg.engine == "bayes":
        decision = Decision(
            action=loss.action,
            message_type=loss.message_type,
            reason_code=rules.reason_code,   # the bank is engine-independent
            rule=f"loss:{'forced' if loss.forced else 'argmin'}",
            strength=loss.posterior,
        )
    else:  # hybrid
        decision = rules

        # Task 5: give the posterior somewhere real to act.
        #
        # The cascade is high-precision where it has a SPECIFIC rule. Where it
        # falls through to a generic default - "trusted contact, nothing to act
        # on", "media we could not read", "no signal at all" - it is not
        # expressing knowledge, it is expressing absence of knowledge. Those are
        # exactly the rows where an expected-loss argmin over the type posterior
        # should be allowed to decide, and only those.
        #
        # Guarded twice: the posterior must be decisive (margin above the
        # ambiguity band) and it may never route a message the cascade muted
        # into notify.
        if rules.rule in GENERIC_FALLBACK_RULES and cfg.matrix:
            if loss.margin >= AMBIGUITY_MARGIN and loss.action != rules.action:
                if not (rules.action == "mute" and loss.action == "notify"):
                    decision = Decision(
                        action=loss.action,
                        message_type=loss.message_type,
                        reason_code=rules.reason_code,
                        rule=f"{rules.rule}->posterior",
                        strength=loss.posterior,
                    )

        # Task 7: margin-gated LLM critic. Runs ONLY on genuinely close calls,
        # and only when a key is present - otherwise this block is a no-op and
        # output.csv is byte-identical to the deterministic run.
        if cfg.critic and critic.enabled() and loss.ambiguous and not verdict.forced:
            ranked = sorted(loss.expected_loss.items(), key=lambda kv: kv[1])
            candidates = [a for a, _ in ranked[:2]]
            if decision.action not in candidates:
                candidates = [decision.action] + candidates[:1]
            chosen, note = critic.adjudicate(
                critic.facts_from(env, sig, evidence), candidates, decision.action,
                message_id=env.message_id,
            )
            critic_note = note
            # ADOPTION POLICY. The critic's opinion is always recorded, but it
            # is only adopted where the cascade fired a GENERIC fallback - the
            # same boundary the posterior override respects. Where a specific
            # high-precision rule fired, the cascade stands and the critic is a
            # logged second opinion.
            #
            # A dissent against a specific rule usually means the critic is
            # reading the words and underweighting the relationship. Two
            # shapes show it. A one-to-one direct ask with a stated deadline
            # from a sender this user engages with heavily is a notify even
            # when the tone is calm (gold sample_msg_006 marks a weaker ask as
            # notify). The same content from a sender whose similar messages
            # the user habitually ignores is a mute (gold sample_msg_044/045).
            # The cascade encodes both through engagement and ignore history,
            # so the decision stays with it.
            adoptable = rules.rule in GENERIC_FALLBACK_RULES
            if chosen != decision.action and not adoptable:
                critic_note = f"{note} (recorded, not adopted: specific rule {rules.rule})"
                chosen = decision.action
            if chosen != decision.action:
                decision = Decision(
                    action=chosen,
                    message_type=decision.message_type,
                    reason_code=decision.reason_code,
                    rule=f"{decision.rule}->critic",
                    strength=decision.strength,
                )

        if cfg.defence and verdict.forced and rules.action != "mute":
            # Safety is a constraint, not a term. The cascade is overruled.
            decision = Decision(
                action="mute",
                message_type=verdict.forced_type or "scam",
                reason_code=(
                    "INJECTION_ATTEMPT" if sig.injection_hits
                    else "LOOKALIKE_DOMAIN" if sig.lookalike_domain
                    else "OTP_SUSPICIOUS_FLOW"
                ),
                rule="hybrid:safety_override",
                strength=loss.posterior,
            )

    return StageResult(
        env=env,
        sig=sig,
        belief=belief,
        verdict=verdict,
        loss=loss,
        decision=decision,
        evidence=evidence,
        critic_note=critic_note,
        raw_posterior=_blend_posterior(decision, loss, belief, sig, evidence),
    )


def run(
    rows: list[dict[str, str]],
    ds: Dataset,
    index: EvidenceIndex | None,
    cfg: RunConfig,
    calibrator=None,
) -> tuple[list[dict[str, str]], list[dict]]:
    """S1 -> S7 over a list of raw message rows."""
    # The semantic layer is switchable for the ablation table. Disabling it
    # forces semantic.available() to False, so every intent score is empty and
    # only the lexicons speak.
    import semantic as _sem

    if cfg.semantic:
        if _sem._model is False:
            _sem._model = None
            _sem._proto_vectors = None
    else:
        _sem._model = False
        _sem._proto_vectors = {}

    envelopes = build_envelopes(rows, ds)
    for env in envelopes:                                             # S1
        env.perception = perceive(env, enabled=cfg.perception)
    if cfg.perception:
        save_cache()

    predictions: list[dict[str, str]] = []
    traces: list[dict] = []

    for env in envelopes:
        r = process(env, ds, index, cfg)                              # S2-S6

        confidence = None
        if cfg.calibration and calibrator is not None:
            confidence = calibrator(r.decision.action, r.raw_posterior)

        row = build_row(r.env, r.sig, r.decision, r.evidence, confidence)  # S7
        trace = build_trace(r.env, r.sig, r.decision, r.evidence, confidence)
        trace.update(
            {
                "importance": round(r.belief.importance, 3),
                "risk": round(r.belief.risk, 3),
                "adjudicated_risk": round(r.verdict.risk, 3),
                "safety_forced": r.verdict.forced,
                "top_type": r.belief.top_type,
                "type_entropy": round(r.belief.type_entropy, 3),
                "expected_loss": {k: round(v, 3) for k, v in r.loss.expected_loss.items()},
                "loss_margin": round(r.loss.margin, 3),
                "ambiguous": r.loss.ambiguous,
                "tiebreaker": r.loss.tiebreaker,
                "raw_posterior": round(r.raw_posterior, 3),
                "prosecution": [f"{t} (+{w:.2f})" for t, w in r.verdict.prosecution],
                "defence": [f"{t} ({w:+.2f})" for t, w in r.verdict.defence],
                "critic": r.critic_note,
                "perception_source": r.env.perception.source,
                "perceived_text": r.env.perception.text[:240],
            }
        )
        predictions.append(row)
        traces.append(trace)

    if cfg.critic and critic.enabled():
        critic.save_cache()
    return predictions, traces
