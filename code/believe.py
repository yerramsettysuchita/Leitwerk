"""S4 BELIEVE - the dual Bayesian posterior (LEITWERK.md sec 4).

Two orthogonal latent variables, not one verdict:

    Importance  I in [0,1]   how much this user would value being interrupted now
    Risk        R in [0,1]   probability the message is scam, spam, or unsafe

Keeping them orthogonal is what lets "high importance but high risk" resolve to
mute, which is the payment-ask-from-a-stranger case.

Both are built the same way: a Beta-Binomial behavioral prior converted to
log-odds, plus a sum of log-likelihood ratios from deterministic signals. Every
weight lives in one table at the top of this module so the whole belief layer is
inspectable and ablatable in one screen.

A third output, the type posterior, is a distribution over all 11 legal
message_type values. It is deliberately a distribution and not an argmax,
because the loss matrix in decide.py integrates over it - an uncertain type
spreads risk across actions rather than committing wrongly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import config as C
from envelope import MessageEnvelope
from features import Signals

# ---------------------------------------------------------------------------
# Weights. Log-likelihood-ratio contributions, in nats.
#
# Tuned against the 30 labelled samples only, never against the hidden set.
# Signs are the design: positive raises the latent variable, negative lowers it.
# ---------------------------------------------------------------------------

W_IMPORTANCE = {
    "mention": 2.2,           # a direct @mention of this user
    "deadline": 1.8,          # stated dependency or escalation
    "same_day": 0.9,          # operational content for today
    "admin_operational": 1.5,  # group admin in a society/school/safety group
    "direct_ask": 1.0,        # asks this user for a response
    "work_context": 0.6,
    "transactional": 0.8,     # matches a live order/booking relationship
    "relationship": 1.2,      # engagement prior deviation from neutral
    "decay": 0.8,             # temporal relevance (sec 4.5), flag-gated
    "fatigue": -0.7,          # user drowning in notifications (sec 4.6)
    "repetition": -1.6,       # near-duplicate of history this user ignored
    "opted_out": -2.4,
    "group_muted": -1.5,
    "promotional": -1.1,
    "greeting": -1.3,
    "chain_forward": -1.7,
    "unperceived_media": -0.3,  # honest humility, not a penalty on the message
}

W_RISK = {
    "injection": 3.4,          # steering the router is itself evidence of scam
    "credential_ask": 2.6,
    "pressure": 1.7,
    "payment_ask": 0.9,
    "lookalike_domain": 3.0,
    "young_domain": 1.2,
    "young_account": 0.8,
    "high_reports": 1.1,
    "unverified_business": 0.9,
    "first_contact": 0.7,
    "first_contact_sensitive": 2.2,  # interaction term, not a sum of parts
    "chain_forward": 0.5,
    "safety_advisory": -2.8,   # a brand warning about fraud is not fraud
    "known_relationship": -1.0,
    "verified_business": -0.8,
    "admin_sender": -1.2,
}

# Prior log-odds when no evidence at all is present.
PRIOR_LOGIT_IMPORTANCE = -0.4   # most messages do not deserve an interruption
PRIOR_LOGIT_RISK = -2.6         # most messages are not scams

# Type posterior: deterministic guards seed mass before any likelihood applies.
TYPE_PRIOR = 0.02  # floor so no legal type has zero mass


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


@dataclass
class Belief:
    importance: float
    risk: float
    type_posterior: dict[str, float]
    importance_terms: dict[str, float] = field(default_factory=dict)
    risk_terms: dict[str, float] = field(default_factory=dict)

    @property
    def top_type(self) -> str:
        return max(self.type_posterior.items(), key=lambda kv: (kv[1], kv[0]))[0]

    @property
    def type_entropy(self) -> float:
        """Nats. High entropy means the type is genuinely unresolved, which the
        loss matrix should see rather than have hidden behind an argmax."""
        return -sum(p * math.log(p) for p in self.type_posterior.values() if p > 0)


# ---------------------------------------------------------------------------
# Beta-Binomial behavioral prior (sec 4.2)
# ---------------------------------------------------------------------------

def engagement_prior(sig: Signals) -> tuple[float, float]:
    """Posterior mean of p_engage under Beta(alpha0+a, beta0+b), and the
    strength of the evidence behind it.

    Returns (mean, concentration). Concentration is a+b: it is how much history
    exists. A first-contact sender has concentration 0, so the mean sits at the
    weak prior 0.5 and contributes almost nothing to the log-odds. That humility
    is where calibrated confidence comes from.
    """
    a = sig.opens + sig.replies
    b = sig.dismissals + sig.muted_after + sig.reports
    mean = (C.BETA_ALPHA0 + a) / (C.BETA_ALPHA0 + a + C.BETA_BETA0 + b)
    return mean, float(a + b)


def _shrink(mean: float, concentration: float, k: float = 6.0) -> float:
    """Pull the prior mean toward 0.5 when there is little history behind it.
    Without this, one open and zero dismissals would read as certainty."""
    weight = concentration / (concentration + k)
    return 0.5 + (mean - 0.5) * weight


# ---------------------------------------------------------------------------
# Temporal relevance decay (sec 4.5), bounded and flag-gated
# ---------------------------------------------------------------------------

def temporal_relevance(env: MessageEnvelope, sig: Signals, now=None) -> float:
    """exp(-ln2 * age_hours / H). Same-day operational content has a short
    half-life; scheduled future events have a long one."""
    if env.created_at is None:
        return 0.5
    half_life = 4.0 if sig.is_same_day_operational else 48.0
    reference = now or max(
        (e for e in (env.created_at,) if e), default=env.created_at
    )
    age_hours = max((reference - env.created_at).total_seconds() / 3600.0, 0.0)
    return math.exp(-math.log(2) * age_hours / half_life)


# ---------------------------------------------------------------------------
# Type posterior (sec 4.4)
# ---------------------------------------------------------------------------

def type_posterior(env: MessageEnvelope, sig: Signals, risk: float) -> dict[str, float]:
    """Deterministic guards seed unnormalized mass over the 11 legal types."""
    mass = {t: TYPE_PRIOR for t in C.MESSAGE_TYPES}

    # --- risk-bearing types -------------------------------------------------
    if sig.injection_hits:
        mass["scam"] += 6.0
    if sig.credential_hits:
        mass["scam"] += 3.5
    if sig.lookalike_domain:
        mass["scam"] += 4.0
    if sig.pressure_hits and sig.payment_hits:
        mass["scam"] += 2.0
    if sig.first_contact and sig.sensitive_ask:
        mass["scam"] += 3.0

    # Unverified bulk sender: spam rather than scam. Verification is the gold
    # discriminator between the two (sample_msg_043 vs sample_msg_047).
    if sig.business_id_present and not sig.business_verified:
        mass["spam"] += 1.6 + 1.4 * float(sig.is_promotional)
        if sig.high_reports:
            mass["spam"] += 0.8

    # --- benign types -------------------------------------------------------
    if sig.is_promotional or sig.is_marketplace_listing:
        mass["promotion"] += 2.6 if sig.business_verified or not sig.business_id_present else 1.0
    if sig.is_transactional and sig.business_id_present:
        mass["business_update"] += 2.4
    if sig.is_safety_advisory and sig.business_verified:
        mass["business_update"] += 3.0
    if sig.is_event:
        mass["event"] += 2.2
    if sig.is_greeting:
        mass["greeting"] += 2.4
    if sig.is_forward_chain and not sig.is_greeting:
        mass["forward"] += 2.3
    if sig.payment_hits and sig.business_known and sig.business_verified:
        mass["payment"] += 1.8
    if sig.mentions_recipient and (sig.has_deadline or sig.is_same_day_operational):
        mass["urgent"] += 3.0
    if sig.has_deadline:
        mass["urgent"] += 1.6
    if sig.sender_is_admin and sig.is_operational_group and sig.is_same_day_operational:
        mass["urgent"] += 1.4
        mass["event"] += 0.8
    if env.conversation_type in ("personal", "group") and not sig.business_id_present:
        mass["personal"] += 1.3
    if sig.asks_directly and sig.known_sender:
        mass["personal"] += 0.9

    # An unfamiliar sender is `unknown` regardless of topic: what matters about
    # a stranger is that they are a stranger (sample_msg_049).
    if sig.first_contact and not sig.business_id_present and not sig.sensitive_ask:
        mass["unknown"] += 2.6

    # High risk pulls mass onto the unsafe types so the loss matrix routes it.
    if risk > 0.5:
        pull = (risk - 0.5) * 6.0
        if sig.business_id_present and not sig.business_verified and not sig.credential_hits:
            mass["spam"] += pull
        else:
            mass["scam"] += pull

    total = sum(mass.values())
    return {t: m / total for t, m in mass.items()}


# ---------------------------------------------------------------------------
# The belief step
# ---------------------------------------------------------------------------

def believe(
    env: MessageEnvelope,
    sig: Signals,
    *,
    use_decay: bool = True,
    use_fatigue: bool = True,
) -> Belief:
    # ---- Risk first: the type posterior needs it ---------------------------
    r_terms: dict[str, float] = {}

    def add_risk(name: str, on: bool, scale: float = 1.0) -> None:
        if on and scale:
            r_terms[name] = W_RISK[name] * scale

    add_risk("injection", bool(sig.injection_hits))
    add_risk("credential_ask", bool(sig.credential_hits))
    add_risk("pressure", bool(sig.pressure_hits))
    add_risk("payment_ask", bool(sig.payment_hits))
    add_risk("lookalike_domain", sig.lookalike_domain, sig.domain_similarity)
    add_risk("young_domain", sig.young_domain)
    add_risk("young_account", sig.young_account)
    add_risk("high_reports", sig.high_reports)
    add_risk("unverified_business", sig.business_id_present and not sig.business_verified)
    add_risk("first_contact", sig.first_contact)
    add_risk("first_contact_sensitive", sig.first_contact and sig.sensitive_ask)
    add_risk("chain_forward", sig.is_chain_forward)
    add_risk("safety_advisory", sig.is_safety_advisory)
    add_risk("known_relationship", sig.business_known or sig.sender_history_count > 2)
    add_risk("verified_business", sig.business_verified)
    add_risk("admin_sender", sig.sender_is_admin)

    risk = sigmoid(PRIOR_LOGIT_RISK + sum(r_terms.values()))

    # ---- Importance --------------------------------------------------------
    i_terms: dict[str, float] = {}
    prior_mean, concentration = engagement_prior(sig)
    shrunk = _shrink(prior_mean, concentration)
    i_terms["relationship"] = W_IMPORTANCE["relationship"] * logit(shrunk)

    def add_imp(name: str, on: bool, scale: float = 1.0) -> None:
        if on and scale:
            i_terms[name] = W_IMPORTANCE[name] * scale

    add_imp("mention", sig.mentions_recipient)
    add_imp("deadline", sig.has_deadline)
    add_imp("same_day", sig.is_same_day_operational)
    add_imp("admin_operational", sig.sender_is_admin and sig.is_operational_group)
    add_imp("direct_ask", sig.asks_directly)
    add_imp("work_context", sig.is_work_context)
    add_imp("transactional", sig.is_transactional and sig.business_known)
    add_imp("repetition", sig.habitually_ignored)
    add_imp("opted_out", sig.opted_out)
    add_imp("group_muted", sig.group_muted and not sig.mentions_recipient)
    add_imp("promotional", sig.is_promotional)
    add_imp("greeting", sig.is_greeting)
    add_imp("chain_forward", sig.is_chain_forward)
    add_imp("unperceived_media", sig.has_media and not sig.perceived)

    if use_fatigue and sig.user_dismiss_rate > 0:
        add_imp("fatigue", True, min(sig.user_dismiss_rate * 2.0, 1.0))
    if use_decay:
        add_imp("decay", True, temporal_relevance(env, sig) - 0.5)

    importance = sigmoid(PRIOR_LOGIT_IMPORTANCE + sum(i_terms.values()))

    return Belief(
        importance=importance,
        risk=risk,
        type_posterior=type_posterior(env, sig, risk),
        importance_terms={k: round(v, 3) for k, v in i_terms.items()},
        risk_terms={k: round(v, 3) for k, v in r_terms.items()},
    )
