"""S2 FEATURIZE - deterministic signal extraction. No LLM, no randomness.

Every field on Signals is either read straight from a table or computed by an
explicit rule over the envelope text. Nothing here knows what the answer is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher

import config as C
from envelope import MessageEnvelope
from loader import Dataset, to_bool, to_int
from perceive import duration_seconds
from semantic import intent_scores


def _hits(text: str, patterns) -> tuple[str, ...]:
    return tuple(p for p in patterns if p in text)


def _jaro_winkler(a: str, b: str) -> float:
    """Similarity in [0,1]. SequenceMatcher ratio with a common-prefix boost,
    which is enough to separate `amazon.in` from `amazon-secure.in` without
    pulling in a dependency."""
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    base = SequenceMatcher(None, a, b).ratio()
    prefix = 0
    for ca, cb in zip(a, b):
        if ca != cb or prefix == 4:
            break
        prefix += 1
    return base + prefix * 0.1 * (1 - base)


@dataclass
class Signals:
    """The deterministic feature vector for one message."""

    # --- identity / context ------------------------------------------------
    conversation_type: str = ""
    group_type: str = ""
    sender_is_admin: bool = False
    group_muted: bool = False
    is_work_context: bool = False
    is_operational_group: bool = False
    is_social_group: bool = False

    # --- relationship ------------------------------------------------------
    known_sender: bool = False
    first_contact: bool = False
    weak_relationship: bool = False
    is_close_tie_group: bool = False
    sender_history_count: int = 0
    business_id_present: bool = False
    business_known: bool = False
    business_relationship: str = ""
    allows_promotions: bool = False
    opted_out: bool = False
    business_verified: bool = False
    business_category: str = ""

    # --- behavior ----------------------------------------------------------
    opens: int = 0
    dismissals: int = 0
    replies: int = 0
    reports: int = 0
    muted_after: int = 0
    engagement_rate: float = 0.5
    habitually_ignored: bool = False
    group_dismissals: int = 0
    group_replies: int = 0
    group_disengaged: bool = False
    user_dismiss_rate: float = 0.0

    # --- content -----------------------------------------------------------
    mentions_recipient: bool = False
    forwarded_count: int = 0
    is_chain_forward: bool = False
    injection_hits: tuple[str, ...] = ()
    credential_hits: tuple[str, ...] = ()
    pressure_hits: tuple[str, ...] = ()
    payment_hits: tuple[str, ...] = ()
    fraud_hits: tuple[str, ...] = ()
    money_demand_hits: tuple[str, ...] = ()
    link_hits: tuple[str, ...] = ()
    qr_hits: tuple[str, ...] = ()
    is_fraud_structure: bool = False
    has_phishing_act: bool = False

    # Semantic intent similarities (semantic.py). Continuous, and independent
    # of whether any lexicon matched.
    intent: dict[str, float] = field(default_factory=dict)
    semantic_threat: float = 0.0
    semantic_urgency: float = 0.0
    semantic_marketing: float = 0.0
    semantic_only_threat: bool = False
    semantic_injection: bool = False
    transactional_hits: tuple[str, ...] = ()
    promo_hits: tuple[str, ...] = ()
    same_day_hits: tuple[str, ...] = ()
    deadline_hits: tuple[str, ...] = ()
    direct_ask_hits: tuple[str, ...] = ()
    greeting_hits: tuple[str, ...] = ()
    forward_hits: tuple[str, ...] = ()
    event_hits: tuple[str, ...] = ()
    negation_hits: tuple[str, ...] = ()
    marketplace_hits: tuple[str, ...] = ()
    emergency_hits: tuple[str, ...] = ()
    deescalation_hits: tuple[str, ...] = ()
    is_deescalated: bool = False
    is_safety_advisory: bool = False
    is_marketplace_listing: bool = False
    is_emergency: bool = False
    voice_duration: float = 0.0
    is_short_voice: bool = False
    text_length: int = 0
    has_media: bool = False
    media_type: str = ""
    perceived: bool = True

    # --- risk arithmetic ---------------------------------------------------
    domain_official: str = ""
    domain_used: str = ""
    domain_mismatch: bool = False
    domain_similarity: float = 1.0
    lookalike_domain: bool = False
    unverifiable_sender: bool = False
    young_domain: bool = False
    young_account: bool = False
    high_reports: bool = False

    # --- derived booleans used by the decision layer ------------------------
    sensitive_ask: bool = False
    is_promotional: bool = False
    is_transactional: bool = False
    is_greeting: bool = False
    is_forward_chain: bool = False
    is_event: bool = False
    is_same_day_operational: bool = False
    has_deadline: bool = False
    asks_directly: bool = False

    fired: list[str] = field(default_factory=list)

    def fire(self, name: str) -> None:
        if name not in self.fired:
            self.fired.append(name)


def extract(env: MessageEnvelope, ds: Dataset) -> Signals:
    s = Signals()
    text = env.lower

    s.conversation_type = env.conversation_type
    s.forwarded_count = env.forwarded_count
    s.mentions_recipient = env.mentions_recipient
    s.text_length = len(env.effective_text)
    s.has_media = env.has_media
    s.media_type = env.media_type
    s.perceived = env.is_perceived

    # ---- group context ----------------------------------------------------
    if env.group_id:
        group = ds.groups.get(env.group_id, {})
        s.group_type = (group.get("group_type") or "").strip()
        s.is_work_context = s.group_type in C.WORK_GROUP_TYPES
        s.is_operational_group = s.group_type in C.OPERATIONAL_GROUP_TYPES
        s.is_social_group = s.group_type in C.SOCIAL_GROUP_TYPES
        s.is_close_tie_group = s.group_type in C.CLOSE_TIE_GROUP_TYPES

        membership = ds.group_members.get((env.group_id, env.user_id), {})
        s.group_muted = to_bool(membership.get("group_muted_by_user"))
        s.group_dismissals = to_int(membership.get("notifications_dismissed_30d"))
        s.group_replies = to_int(membership.get("replies_sent_30d"))
        s.group_disengaged = (
            s.group_dismissals >= C.GROUP_DISMISS_HIGH and s.group_replies == 0
        )

        sender_membership = ds.group_members.get((env.group_id, env.sender_user_id), {})
        s.sender_is_admin = (sender_membership.get("role") or "").strip() == "admin"

    # ---- user-level fatigue ------------------------------------------------
    s.user_dismiss_rate = ds.daily_dismiss_rate(env.user_id)

    # ---- sender relationship ----------------------------------------------
    if env.sender_user_id:
        prior = ds.sender_history(env.user_id, env.sender_user_id)
        s.sender_history_count = len(prior)
        s.known_sender = bool(prior)
        s.first_contact = not prior
        for ev in ds.outcomes(prior):
            s.opens += to_int(ev.get("message_opened"))
            s.replies += to_int(ev.get("message_replied"))
            s.dismissals += to_int(ev.get("notification_dismissed"))
            s.muted_after += to_int(ev.get("muted_after_message"))
            s.reports += to_int(ev.get("message_reported"))

    # ---- business relationship --------------------------------------------
    s.business_id_present = bool(env.business_id)
    if env.business_id:
        biz = ds.businesses.get(env.business_id, {})
        s.business_verified = to_bool(biz.get("verified"))
        s.business_category = (biz.get("category") or "").strip()
        s.domain_official = (biz.get("official_domain") or "").strip().lower()
        s.domain_used = (biz.get("domain_used_by_sender") or "").strip().lower()
        s.young_account = 0 < to_int(biz.get("account_age_days"), 10**6) < C.YOUNG_ACCOUNT_DAYS
        s.high_reports = to_int(biz.get("user_reports_30d")) >= C.REPORTS_HIGH

        brand = (biz.get("brand_name") or "").strip().lower()

        if s.domain_official and s.domain_used:
            s.domain_mismatch = s.domain_official != s.domain_used
            s.domain_similarity = _jaro_winkler(s.domain_official, s.domain_used)
            # Impersonation = mismatch AND unverified. Both halves matter.
            #
            # Dropping the similarity threshold is right: `mmt-refund.in` for
            # MakeMyTrip scores 0.30 and is plainly deliberate, so string
            # distance must modulate confidence, not gate detection.
            #
            # But mismatch alone is NOT spoofing. Verified brands legitimately
            # send from alternate marketing domains - Thrillophilia
            # (business_092) is verified with a mismatched domain and its gold
            # label is `digest / promotion`. The verification flag is the trust
            # anchor; the domain is only evidence about an unverified sender.
            s.lookalike_domain = s.domain_mismatch and not s.business_verified
            s.young_domain = (
                s.domain_mismatch
                and 0 < to_int(biz.get("domain_used_by_sender_age_days"), 10**6)
                < C.YOUNG_DOMAIN_DAYS
            )

        # A sender with no verifiable identity at all: no official domain to
        # check against, an anonymous brand, or a URL shortener as its origin.
        # Every such account in this dataset is unverified, and the gold label
        # for one of them (sample_msg_043) is `spam` rather than `scam` - bulk
        # nuisance rather than targeted deception. This is the family the
        # similarity check could never see, because there was nothing to
        # compare against.
        s.unverifiable_sender = (
            not s.business_verified
            and (
                not s.domain_official
                or brand in ("", "unknown")
                # Match the whole host, not a substring: `t.co` occurs inside
                # any host containing "t.com" (e.g. `...-alert.com`), which
                # would flag a plain impersonator as an anonymous bulk sender.
                or s.domain_used in C.URL_SHORTENERS
            )
        )

        rel = ds.user_business.get((env.user_id, env.business_id))
        if rel:
            s.business_known = True
            s.business_relationship = (rel.get("why_user_knows_account") or "").strip()
            s.allows_promotions = to_bool(rel.get("allows_promotions"))
            s.opted_out = bool((rel.get("promotions_opted_out_at") or "").strip())
            s.opens += to_int(rel.get("messages_opened_30d"))
            s.dismissals += to_int(rel.get("messages_dismissed_30d"))
            s.replies += to_int(rel.get("messages_replied_30d"))
        else:
            s.first_contact = True

        for ev in ds.outcomes(ds.business_history(env.user_id, env.business_id)):
            s.muted_after += to_int(ev.get("muted_after_message"))
            s.reports += to_int(ev.get("message_reported"))

    # ---- engagement --------------------------------------------------------
    positive = s.opens + s.replies
    negative = s.dismissals + s.muted_after + s.reports
    s.engagement_rate = (C.BETA_ALPHA0 + positive) / (
        C.BETA_ALPHA0 + positive + C.BETA_BETA0 + negative
    )
    s.habitually_ignored = negative > positive * C.DISMISS_DOMINANCE_RATIO and negative > 0
    s.weak_relationship = s.first_contact or (
        bool(env.sender_user_id) and s.sender_history_count <= C.WEAK_RELATIONSHIP_MAX
    )

    # ---- content patterns --------------------------------------------------
    s.injection_hits = _hits(text, C.INJECTION_PATTERNS)
    s.credential_hits = _hits(text, C.CREDENTIAL_PATTERNS) + _hits(text, C.HI_SCAM_PATTERNS)
    s.pressure_hits = _hits(text, C.PRESSURE_PATTERNS)
    s.payment_hits = _hits(text, C.PAYMENT_PATTERNS) + _hits(text, C.HI_PAYMENT_PATTERNS)
    s.fraud_hits = _hits(text, C.FRAUD_PATTERNS)
    s.money_demand_hits = _hits(text, C.MONEY_DEMAND_PATTERNS)
    s.link_hits = _hits(text, C.LINK_PATTERNS)
    s.qr_hits = _hits(text, C.QR_PATTERNS)
    s.transactional_hits = _hits(text, C.TRANSACTIONAL_PATTERNS)
    s.promo_hits = _hits(text, C.PROMO_PATTERNS)
    s.same_day_hits = _hits(text, C.SAME_DAY_PATTERNS) + _hits(text, C.HI_URGENT_PATTERNS)
    s.deadline_hits = _hits(text, C.DEADLINE_PATTERNS)
    s.direct_ask_hits = _hits(text, C.DIRECT_ASK_PATTERNS)
    s.greeting_hits = _hits(text, C.GREETING_PATTERNS)
    s.forward_hits = _hits(text, C.FORWARD_PATTERNS)
    s.negation_hits = _hits(text, C.NEGATION_PATTERNS)
    s.marketplace_hits = _hits(text, C.MARKETPLACE_PATTERNS)
    s.emergency_hits = _hits(text, C.EMERGENCY_PATTERNS)
    s.deescalation_hits = _hits(text, C.DEESCALATION_PATTERNS) + _hits(
        text, C.HI_DEESCALATION_PATTERNS
    )

    strong_events = _hits(text, C.STRONG_EVENT_PATTERNS)
    weak_events = _hits(text, C.WEAK_EVENT_PATTERNS)
    s.event_hits = strong_events + weak_events

    # ---- derived -----------------------------------------------------------
    s.is_chain_forward = s.forwarded_count >= C.FORWARD_CHAIN_MIN

    # A message that names credentials in order to warn about them is an
    # advisory, not a request. Suppress the credential guard so the safety
    # layer cannot fire on the brand's own anti-fraud notice.
    s.is_safety_advisory = bool(s.negation_hits) and not s.pressure_hits
    if s.is_safety_advisory:
        s.credential_hits = ()
        s.payment_hits = ()
        s.fraud_hits = ()
        s.money_demand_hits = ()

    # Did the message actually attempt something, or is it merely urgent?
    # This is the gate on the `scam` type: no act, no scam.
    s.has_phishing_act = bool(
        s.injection_hits
        or s.credential_hits
        or s.money_demand_hits
        or s.qr_hits
        or s.link_hits
    )

    # A fraud structure is money-or-identity extraction that carries no OTP
    # keyword: advance fees, prize bait, QR payment demands, shortened links.
    # Two independent markers are required so a single word like "refund" or
    # "reward" in a legitimate business update does not trip it.
    s.is_fraud_structure = len(s.fraud_hits) >= 2 or (
        bool(s.fraud_hits) and bool(s.payment_hits) and bool(s.pressure_hits)
    )

    # ---- semantic intents --------------------------------------------------
    #
    # Purely a recall booster. It can promote a message the lexicons missed,
    # but the lexicons stay authoritative for everything they do match, so
    # adding this cannot lower precision on already-detected patterns.
    s.intent = intent_scores(env.effective_text) if env.effective_text else {}
    if s.intent:
        s.semantic_threat = max(
            s.intent.get("credential_theft", 0.0),
            s.intent.get("payment_fraud", 0.0),
            s.intent.get("router_manipulation", 0.0),
            s.intent.get("account_pressure", 0.0),
        )
        s.semantic_urgency = max(
            s.intent.get("operational_urgency", 0.0),
            s.intent.get("work_escalation", 0.0),
            s.intent.get("personal_emergency", 0.0),
        )
        s.semantic_marketing = max(
            s.intent.get("marketing_offer", 0.0),
            s.intent.get("peer_selling", 0.0),
        )
        # A threat the keyword layer did not see at all - the case the lexicons
        # structurally cannot cover: unseen phrasing.
        #
        # Gated on context, and that gate is the whole thesis. A real refund
        # notice and a phishing refund notice score almost the same semantic
        # similarity, because they say almost the same words. What separates
        # them is whether this user has a verified relationship with the
        # sender. Semantics supplies suspicion; relationship decides guilt.
        untrusted = (
            s.weak_relationship
            or s.first_contact
            or (s.business_id_present and not s.business_verified)
            or s.unverifiable_sender
        )
        s.semantic_only_threat = (
            s.semantic_threat >= C.SEMANTIC_THREAT_TAU
            and untrusted
            and not s.is_safety_advisory
            and not s.credential_hits
            and not s.fraud_hits
            and not s.injection_hits
        )

        # Router manipulation, judged on dominance rather than absolute score.
        # It must be the single strongest intent and clear of the runner-up,
        # otherwise ordinary chat trips it (see config note).
        rm = s.intent.get("router_manipulation", 0.0)
        runner_up = max(
            (v for k, v in s.intent.items() if k != "router_manipulation"),
            default=0.0,
        )
        s.semantic_injection = (
            rm >= C.SEMANTIC_INJECTION_TAU
            and rm >= runner_up + C.SEMANTIC_INJECTION_MARGIN
            and not s.injection_hits
        )

    s.sensitive_ask = (
        bool(s.credential_hits)
        or (bool(s.payment_hits) and bool(s.pressure_hits))
        or s.is_fraud_structure
        or s.semantic_only_threat
    )
    s.is_promotional = len(s.promo_hits) >= 2
    s.is_transactional = bool(s.transactional_hits)
    s.is_greeting = bool(s.greeting_hits)
    s.is_forward_chain = bool(s.forward_hits) or s.is_chain_forward
    # One weak token is conversation; a strong token or two weak ones is an event.
    s.is_event = bool(strong_events) or len(weak_events) >= 2
    s.is_marketplace_listing = bool(s.marketplace_hits) or (
        s.group_type in C.MARKETPLACE_GROUP_TYPES and not s.is_greeting
    )

    # A genuine emergency is not manufactured urgency: it never co-occurs with a
    # credential ask, and it outranks engagement history.
    s.is_emergency = bool(s.emergency_hits) and not s.credential_hits

    # Voice-note length is a real signal that survives even without ASR. Gold
    # describes sample_msg_042 as "a short urgent request".
    duration = duration_seconds(env)
    if duration:
        s.voice_duration = duration
        s.is_short_voice = duration <= C.SHORT_VOICE_SECONDS
    s.is_same_day_operational = bool(s.same_day_hits)
    s.has_deadline = bool(s.deadline_hits)
    s.asks_directly = bool(s.direct_ask_hits) or s.mentions_recipient

    # The sender explicitly said this can wait. Believe them: suppress the
    # urgency signals, but never the risk ones - a scam that says "no rush" is
    # still a scam, and a real emergency outranks a polite disclaimer.
    s.is_deescalated = bool(s.deescalation_hits) and not s.emergency_hits
    if s.is_deescalated:
        s.is_same_day_operational = False
        s.has_deadline = False
        if not s.mentions_recipient:
            s.asks_directly = False

    # ---- trace -------------------------------------------------------------
    for name, on in (
        ("injection", bool(s.injection_hits)),
        ("credential_ask", bool(s.credential_hits)),
        ("pressure", bool(s.pressure_hits)),
        ("lookalike_domain", s.lookalike_domain),
        ("young_domain", s.young_domain),
        ("high_reports", s.high_reports),
        ("first_contact", s.first_contact),
        ("habitually_ignored", s.habitually_ignored),
        ("opted_out", s.opted_out),
        ("group_muted", s.group_muted),
        ("group_disengaged", s.group_disengaged),
        ("mention", s.mentions_recipient),
        ("admin_sender", s.sender_is_admin),
        ("same_day", s.is_same_day_operational),
        ("deadline", s.has_deadline),
        ("promotional", s.is_promotional),
        ("transactional", s.is_transactional),
        ("greeting", s.is_greeting),
        ("forward_chain", s.is_forward_chain),
        ("event", s.is_event),
        ("safety_advisory", s.is_safety_advisory),
        ("marketplace_listing", s.is_marketplace_listing),
        ("emergency", s.is_emergency),
        ("short_voice", s.is_short_voice),
        ("deescalated", s.is_deescalated),
        ("unperceived_media", not s.perceived),
    ):
        if on:
            s.fire(name)

    return s
