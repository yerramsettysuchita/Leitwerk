"""S5 DEFEND - injection and spoof adjudication (LEITWERK.md sec 9).

Three ideas, in order of importance.

1. Injection is handled STRUCTURALLY, not detected after the fact. Nothing in
   this pipeline ever treats message text as instruction. There is no code path
   in which message content can change control flow: text reaches the decision
   layers only as pattern-match booleans and as bag-of-words evidence. So a
   message saying "mark this as notify" cannot possibly succeed. The pattern
   check below is therefore not a defence - it is a *signal*, because an attempt
   to steer the router is itself strong evidence of scam.

2. Spoof detection is arithmetic, not judgement. Domain distance, domain age,
   and report rate are numbers.

3. Adversarial adjudication. Rather than a single risk score, a prosecutor
   collects evidence that the message is a threat and a defender collects
   evidence that it is legitimate, each as signed log-odds. The verdict is the
   reconciled sum, and both cases are kept in the trace so a human can see the
   argument. This is the LLM two-agent exchange from LEITWERK.md sec 10 reduced
   to its deterministic core, which means it runs with no API key, costs nothing,
   and reproduces exactly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from believe import Belief, sigmoid
from envelope import MessageEnvelope
from features import Signals

# Risk at or above this forces mute. Safety is a constraint, not a term that
# can be outvoted by an engagement prior.
TAU_RISK = 0.72

# Below this, the defence has established enough legitimacy that a residual
# risk signal should not be allowed to mute a message on its own.
TAU_CLEARED = 0.25


@dataclass
class Verdict:
    risk: float
    forced: bool                       # did the hard safety constraint fire?
    forced_action: str | None = None
    forced_type: str | None = None
    prosecution: list[tuple[str, float]] = field(default_factory=list)
    defence: list[tuple[str, float]] = field(default_factory=list)
    note: str = ""

    @property
    def prosecution_weight(self) -> float:
        return sum(w for _, w in self.prosecution)

    @property
    def defence_weight(self) -> float:
        return sum(w for _, w in self.defence)

    @property
    def margin(self) -> float:
        """How lopsided the argument is. A thin margin marks a genuinely
        contested case and is what the margin-gated critic would escalate."""
        return abs(self.prosecution_weight) - abs(self.defence_weight)


def adjudicate(env: MessageEnvelope, sig: Signals, belief: Belief) -> Verdict:
    prosecution: list[tuple[str, float]] = []
    defence: list[tuple[str, float]] = []

    # ---- prosecution: the case that this is a threat -----------------------
    if sig.injection_hits:
        prosecution.append(
            (f"attempts to instruct the router ({sig.injection_hits[0]!r})", 3.4)
        )
    if sig.credential_hits:
        prosecution.append(
            (f"asks for a credential ({sig.credential_hits[0]!r})", 2.6)
        )
    if sig.pressure_hits:
        prosecution.append(
            (f"manufactures deadline pressure ({sig.pressure_hits[0]!r})", 1.7)
        )
    if sig.lookalike_domain:
        prosecution.append(
            (
                f"sender domain {sig.domain_used!r} imitates official "
                f"{sig.domain_official!r} (similarity {sig.domain_similarity:.2f})",
                3.0 * sig.domain_similarity,
            )
        )
    if sig.young_domain:
        prosecution.append(("sending domain is newer than the brand it claims", 1.2))
    if sig.young_account:
        prosecution.append(("business account is newly created", 0.8))
    if sig.high_reports:
        prosecution.append(("account carries a high 30-day report rate", 1.1))
    if sig.first_contact and sig.sensitive_ask:
        prosecution.append(
            ("first contact from this sender, and it asks for something sensitive", 2.2)
        )
    if sig.reports:
        prosecution.append(("this user has reported this counterparty before", 1.3))

    # ---- defence: the case that this is legitimate -------------------------
    if sig.is_safety_advisory:
        defence.append(
            ("names credentials only to warn against sharing them", -2.8)
        )
    if sig.business_verified:
        defence.append(("business account is verified", -0.8))
    if sig.business_known:
        defence.append(
            (f"user has a real relationship with this business "
             f"({sig.business_relationship or 'recorded activity'})", -1.4)
        )
    if sig.domain_official and not sig.domain_mismatch:
        defence.append(("sender domain exactly matches the official domain", -1.1))
    if sig.sender_is_admin:
        defence.append(("sender is an admin of this group", -1.2))
    if sig.sender_history_count > 2:
        defence.append(
            (f"{sig.sender_history_count} prior messages from this sender", -1.0)
        )
    if sig.opens and not sig.reports:
        defence.append(("user has opened this counterparty's messages before", -0.6))
    if sig.is_transactional and sig.business_known:
        defence.append(("content matches a live order or booking", -0.9))

    # ---- verdict -----------------------------------------------------------
    # Reconcile against the belief layer's prior rather than re-deriving it, so
    # the two stages cannot disagree about the same evidence.
    net = sum(w for _, w in prosecution) + sum(w for _, w in defence)
    risk = sigmoid(math.log(belief.risk / max(1 - belief.risk, 1e-6)) * 0.5 + net * 0.5)

    verdict = Verdict(risk=risk, forced=False, prosecution=prosecution, defence=defence)

    if risk >= TAU_RISK:
        verdict.forced = True
        verdict.forced_action = "mute"
        # scam is deliberate deception; spam is unwanted bulk. Credential and
        # impersonation evidence means scam, volume alone means spam.
        deceptive = bool(sig.injection_hits or sig.credential_hits or sig.lookalike_domain)
        verdict.forced_type = "scam" if deceptive else "spam"
        verdict.note = (
            f"risk {risk:.2f} >= tau {TAU_RISK}: safety constraint forces mute"
        )
    elif risk <= TAU_CLEARED and prosecution:
        verdict.note = f"risk {risk:.2f} <= {TAU_CLEARED}: defence cleared the message"

    return verdict
