"""S6 DECIDE (expected loss) - LEITWERK.md sec 5.

The thesis in one function. Do not pick the action with the highest
probability; pick the action with the lowest expected cost, because the costs
are wildly asymmetric and the problem statement says so in plain words: "clear
scam or safety risk should be muted regardless of the user's usual engagement".

    action* = argmin_a  sum_type  P(type) * L[a][type]

Read the matrix by asking, for each true type, how bad each action would be.
Muting a genuine emergency is the worst cell in the table. Notifying a scam is
the second worst. Everything else is a matter of degree.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import config as C
from believe import Belief
from defend import Verdict
from envelope import MessageEnvelope
from features import Signals

# L[action][true_type]. Zero means "this was the right call".
LOSS: dict[str, dict[str, float]] = {
    #            personal urgent event payment biz_upd promo greet forward spam scam
    "notify": {
        "urgent": 0.0, "personal": 0.30, "event": 0.35, "payment": 0.15,
        "business_update": 0.45, "promotion": 1.30, "greeting": 1.40,
        "forward": 1.35, "spam": 1.80, "scam": 3.00, "unknown": 0.70,
    },
    "digest": {
        "urgent": 1.60, "personal": 0.25, "event": 0.0, "payment": 0.55,
        "business_update": 0.0, "promotion": 0.10, "greeting": 0.15,
        "forward": 0.20, "spam": 0.60, "scam": 1.20, "unknown": 0.10,
    },
    "mute": {
        "urgent": 3.20, "personal": 1.30, "event": 1.10, "payment": 1.70,
        "business_update": 0.90, "promotion": 0.20, "greeting": 0.15,
        "forward": 0.10, "spam": 0.0, "scam": 0.0, "unknown": 0.60,
    },
}

# Two actions within this expected-loss margin are a genuine tie. Only inside
# this window are the bounded tie-breakers allowed to move anything
# (LEITWERK.md sec 6).
AMBIGUITY_MARGIN = 0.12


@dataclass
class LossOutcome:
    action: str
    message_type: str
    expected_loss: dict[str, float]
    margin: float
    ambiguous: bool
    tiebreaker: str = ""
    forced: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def posterior(self) -> float:
        """Softmax-free posterior of the chosen action: how much better it is
        than the alternatives, expressed in [0,1]. This is the raw number that
        calibration turns into the emitted confidence."""
        losses = sorted(self.expected_loss.values())
        best, second = losses[0], losses[1]
        spread = max(second - best, 0.0)
        # A 0.6-nat expected-loss gap is a decisive decision.
        return min(0.5 + spread / 1.2, 1.0)


def expected_losses(type_posterior: dict[str, float]) -> dict[str, float]:
    return {
        action: sum(type_posterior.get(t, 0.0) * LOSS[action][t] for t in C.MESSAGE_TYPES)
        for action in C.ACTIONS
    }


def _apply_tiebreakers(
    outcome: LossOutcome, env: MessageEnvelope, sig: Signals, belief: Belief,
    *, use_decay: bool, use_budget: bool,
) -> None:
    """Bounded tie-breakers. They may only reorder notify vs digest, only when
    the two are already within AMBIGUITY_MARGIN, and never touch mute."""
    if not outcome.ambiguous:
        return
    ranked = sorted(outcome.expected_loss.items(), key=lambda kv: kv[1])
    top_two = {ranked[0][0], ranked[1][0]}
    if top_two != {"notify", "digest"}:
        return

    if use_decay:
        # Imminent operational content wins the tie; distant content loses it.
        if sig.is_same_day_operational and not sig.is_deescalated:
            outcome.action = "notify"
            outcome.tiebreaker = "temporal_decay:imminent"
            return
        if sig.is_event and not sig.is_same_day_operational:
            outcome.action = "digest"
            outcome.tiebreaker = "temporal_decay:distant"
            return

    if use_budget:
        # Attention budget: if this user is already drowning in notifications,
        # a borderline interrupt is not worth the spend.
        if sig.user_dismiss_rate > 0.45 and not sig.mentions_recipient:
            outcome.action = "digest"
            outcome.tiebreaker = "attention_budget:spent"


def decide_by_loss(
    env: MessageEnvelope,
    sig: Signals,
    belief: Belief,
    verdict: Verdict,
    *,
    use_matrix: bool = True,
    use_defence: bool = True,
    use_decay: bool = True,
    use_budget: bool = True,
) -> LossOutcome:
    losses = expected_losses(belief.type_posterior)

    if use_matrix:
        ranked = sorted(losses.items(), key=lambda kv: (kv[1], kv[0]))
        action = ranked[0][0]
        margin = ranked[1][1] - ranked[0][1]
    else:
        # Ablation arm: pick the action implied by the argmax type instead of
        # integrating over the posterior. This is what most teams do.
        top = belief.top_type
        action = min(C.ACTIONS, key=lambda a: LOSS[a][top])
        margin = 0.0

    outcome = LossOutcome(
        action=action,
        message_type=belief.top_type,
        expected_loss=losses,
        margin=margin,
        ambiguous=margin < AMBIGUITY_MARGIN,
    )

    _apply_tiebreakers(
        outcome, env, sig, belief, use_decay=use_decay, use_budget=use_budget
    )

    # Hard safety constraint, applied last so nothing can outvote it.
    if use_defence and verdict.forced:
        outcome.action = verdict.forced_action or "mute"
        outcome.message_type = verdict.forced_type or "scam"
        outcome.forced = True
        outcome.notes.append(verdict.note)

    return outcome
