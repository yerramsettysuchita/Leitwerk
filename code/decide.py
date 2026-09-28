"""S6 DECIDE - Phase 1 rules-first baseline.

This is deliberately NOT the Bayesian layer. It is an ordered rule cascade that
produces a legal, measurable decision for every row so we are on the board with
a scoring-valid submission before any math is built (LEITWERK.md sec 12).

Cascade order is load-bearing:

  A. SAFETY      - risk is a constraint, not a term. Fires first, always.
  B. URGENT PULL - a direct mention with a deadline beats a muted group,
                   because the problem statement says it must.
  C. BEHAVIORAL  - opted out, habitually ignored, muted, chain-forwarded.
  D. NOTIFY      - admin operational, work deadline, matched business update.
  E. DIGEST      - the safe middle, with the type chosen by content.

Every branch records the rule that fired, so the trace explains the row.
"""

from __future__ import annotations

from dataclasses import dataclass

from envelope import MessageEnvelope
from features import Signals


@dataclass
class Decision:
    action: str
    message_type: str
    reason_code: str
    rule: str
    # Signal strength in [0,1]: how cleanly the firing rule matched. Feeds
    # confidence placement inside the action's band (config.CONF_ACTION_BAND).
    strength: float = 0.5


def _content_type(env: MessageEnvelope, s: Signals, default: str = "personal") -> str:
    """What the message IS, independent of why it was routed.

    Suppression rules kept typing from their own trigger - a muted group made
    everything `personal`, a high forward count made everything `forward` -
    which mislabelled real promotions and event notices. The action answers
    "should this interrupt"; the type must answer "what is this", and the two
    are decided separately.
    """
    if s.is_greeting:
        return "greeting"
    if s.is_promotional or s.is_marketplace_listing:
        return "promotion" if (s.business_verified or not s.business_id_present) else "spam"
    if s.is_event:
        return "event"
    if s.is_transactional and s.business_id_present:
        return "business_update"
    if s.is_forward_chain:
        return "forward"
    return default


def _scam_reason(env: MessageEnvelope, s: Signals, default: str = "OTP_SUSPICIOUS_FLOW") -> str:
    """Name the mechanism this message actually used.

    Ordered most-specific first. The gold templates for the three scam shapes
    the samples DO contain (injection, fake-support pressure, first-contact
    sensitive ask) keep priority so their exact wording is preserved; the newer
    mechanism templates cover the shapes the samples never showed.
    """
    text = env.lower

    if s.injection_hits or s.semantic_injection:
        return "INJECTION_ATTEMPT"

    # Gold wording, preserved: fake support desk + account-blocking pressure.
    if s.credential_hits and s.pressure_hits and any(
        w in text for w in ("support", "helpdesk", "customer care", "profile will be")
    ):
        return "FAKE_SUPPORT_PRESSURE"

    # Gold wording, preserved: a stranger asking for something sensitive.
    if s.weak_relationship and env.conversation_type == "personal" and (
        s.credential_hits or s.money_demand_hits
    ):
        return "FIRST_CONTACT_SENSITIVE"

    if s.credential_hits:
        return "OTP_SUSPICIOUS_FLOW"
    if s.lookalike_domain:
        return "LOOKALIKE_DOMAIN"

    hit = " ".join(s.fraud_hits + s.money_demand_hits)
    if any(w in hit for w in ("processing fee", "reactivation fee", "loan approved",
                              "amount will be released", "token amount", "remit")):
        return "ADVANCE_FEE"
    if any(w in hit for w in ("reward", "prize", "gift card", "draw", "you won",
                              "was selected", "voucher expires", "claim your")):
        return "PRIZE_BAIT"
    if any(w in hit for w in ("redelivery charge", "reattempt charge", "held at the depot")):
        return "DELIVERY_CHARGE_SCAM"
    if any(w in hit for w in ("account number", "bank details", "card details",
                              "verify wallet", "share your account")):
        return "BANK_DETAIL_REQUEST"
    if s.qr_hits or s.link_hits:
        return "PAYMENT_LINK_DEMAND"

    return default


def _strength(*conditions: bool) -> float:
    """Fraction of corroborating conditions that held. A rule that fires on one
    weak signal reports lower strength than one backed by several."""
    if not conditions:
        return 0.5
    return sum(1 for c in conditions if c) / len(conditions)


# ---------------------------------------------------------------------------
# A. SAFETY
# ---------------------------------------------------------------------------

def _safety(env: MessageEnvelope, s: Signals) -> Decision | None:
    # An attempt to steer the router is itself evidence of scam.
    if s.injection_hits:
        return Decision(
            "mute", "scam", "INJECTION_ATTEMPT", "A1_injection",
            _strength(True, bool(s.credential_hits), bool(s.pressure_hits)),
        )

    # The same attempt, phrased in a way no keyword list anticipated. Caught by
    # semantic similarity to the concept of addressing an automated triage
    # system rather than to any particular wording.
    if s.semantic_injection:
        return Decision(
            "mute", "scam", "INJECTION_ATTEMPT", "A1b_semantic_injection",
            _strength(True, s.semantic_threat >= 0.4, s.weak_relationship),
        )

    # Impersonation: the brand has an official domain and this is not it.
    if s.lookalike_domain:
        return Decision(
            "mute", "scam", "LOOKALIKE_DOMAIN", "A2_lookalike_domain",
            _strength(True, s.young_domain, s.high_reports, not s.business_verified),
        )

    # A sender with no verifiable identity: anonymous brand, no official domain,
    # or a URL shortener as its origin. Split by intent, which is the same
    # scam-vs-spam distinction the gold labels draw. Mimicking a real
    # transaction ("your prescription is ready") is deception; blasting offers
    # at people is nuisance.
    if s.unverifiable_sender:
        impersonating = s.is_transactional or s.sensitive_ask or bool(s.pressure_hits)
        if impersonating:
            return Decision(
                "mute", "scam", _scam_reason(env, s, "LOOKALIKE_DOMAIN"), "A2b_unverifiable_impersonation",
                _strength(True, bool(s.pressure_hits), s.high_reports),
            )
        return Decision(
            "mute", "spam",
            "MARKETING_OPTED_OUT" if (s.opted_out or s.habitually_ignored) else "BULK_PROMOTION",
            "A2c_unverifiable_bulk",
            _strength(True, s.opted_out, s.high_reports, s.young_account),
        )

    # A stranger asking for a credential or payment. Checked before the
    # pressure branch: when a message is BOTH a first contact and a sensitive
    # ask, the relationship gap is the more informative fact, and it is the
    # framing gold uses (sample_msg_052).
    if s.sensitive_ask and s.weak_relationship and env.conversation_type == "personal":
        return Decision(
            "mute", "scam", "FIRST_CONTACT_SENSITIVE", "A4_first_contact_sensitive",
            _strength(True, True, bool(s.pressure_hits)),
        )

    # Fake support language plus account-blocking pressure.
    if s.credential_hits and s.pressure_hits:
        support_flavored = any(
            w in env.lower for w in ("support", "helpdesk", "customer care", "profile will be")
        )
        code = "FAKE_SUPPORT_PRESSURE" if support_flavored else "OTP_SUSPICIOUS_FLOW"
        return Decision(
            "mute", "scam", code, "A3_credential_pressure",
            _strength(True, True, s.first_contact, not s.business_verified),
        )

    # Credential harvesting without the pressure framing.
    if s.credential_hits:
        return Decision(
            "mute", "scam", _scam_reason(env, s), "A5_credential_ask",
            _strength(True, s.first_contact, not s.business_verified),
        )

    # Money-or-identity extraction that never says "OTP": advance-fee loans,
    # prize bait, QR payment demands, shortened verification links. These carry
    # no credential keyword, so without this branch they fall through to the
    # promotion and forward rules and can even reach `digest`.
    if s.is_fraud_structure:
        return Decision(
            "mute", "scam", _scam_reason(env, s), "A7_fraud_structure",
            _strength(True, bool(s.pressure_hits), bool(s.payment_hits), s.weak_relationship),
        )

    # A credential or payment extraction the lexicons never saw, from a sender
    # this user has no established relationship with. Neither half is
    # sufficient alone: the semantics alone would also flag a genuine refund
    # notice, and the weak relationship alone flags every new contact.
    # Money demanded by someone this user has no established relationship with.
    #
    # The same relationship gate already applied to credentials, extended to
    # payments. Legitimate money requests come from a verified business the
    # user transacts with, or from a contact they actually know. A stranger
    # asking you to settle a charge is the single most common scam shape there
    # is, and it needs no scam vocabulary to work: "a redelivery charge is
    # pending on your parcel, settle it now" contains nothing a keyword list
    # would flag.
    if (
        s.money_demand_hits
        and s.weak_relationship
        and not s.business_verified
        and not s.sender_is_admin
        and not s.is_safety_advisory
    ):
        return Decision(
            "mute", "scam", _scam_reason(env, s, "FIRST_CONTACT_SENSITIVE"), "A10_stranger_payment_demand",
            _strength(True, bool(s.pressure_hits), s.first_contact, bool(s.fraud_hits)),
        )

    if s.semantic_only_threat:
        return Decision(
            "mute", "scam",
            _scam_reason(env, s),
            "A9_semantic_threat",
            _strength(True, s.semantic_threat >= 0.6, s.first_contact),
        )

    # A payment demand routed through a link or QR by an ordinary member of a
    # society or school group. Dues are collected by admins through official
    # channels; a resident asking others to pay a link and send a screenshot is
    # the standard community-payment scam. The dataset plants the contrast
    # explicitly: the admin's own notice in the same group says "don't use any
    # payment link shared by residents".
    if (
        s.payment_hits
        and s.is_operational_group
        and not s.sender_is_admin
        and any(w in env.lower for w in ("link", "qr", "screenshot", "scan"))
    ):
        return Decision(
            "mute", "scam", "UNOFFICIAL_PAYMENT_CHANNEL", "A8_unofficial_payment_channel",
            _strength(True, bool(s.pressure_hits), s.weak_relationship),
        )

    # An unverified business account asking for money or identity.
    if s.business_id_present and s.sensitive_ask and not s.business_verified:
        return Decision(
            "mute", "scam", "UNVERIFIED_SENSITIVE", "A6_unverified_sensitive",
            _strength(True, s.young_account, s.high_reports),
        )

    return None


# ---------------------------------------------------------------------------
# B. URGENT PULL - beats behavioral mutes
# ---------------------------------------------------------------------------

def _urgent_pull(env: MessageEnvelope, s: Signals) -> Decision | None:
    # A blessing chain does not become urgent by naming someone in it. An
    # @mention inside a forward-this-to-everyone greeting is not a direct ask.
    if s.is_greeting or s.is_chain_forward:
        return None
    if not s.mentions_recipient:
        return None

    if s.has_deadline or (s.is_work_context and s.asks_directly):
        return Decision(
            "notify", "urgent", "WORK_DEADLINE", "B1_mention_deadline",
            _strength(True, s.has_deadline, s.is_work_context, s.is_same_day_operational),
        )

    if s.group_muted:
        return Decision(
            "notify", "personal", "MENTION_IN_MUTED_GROUP", "B2_mention_in_muted_group",
            _strength(True, s.asks_directly, s.known_sender),
        )

    if s.asks_directly:
        return Decision(
            "notify", "personal", "DIRECT_ASK", "B3_direct_mention_ask",
            _strength(True, s.known_sender, s.engagement_rate > 0.5),
        )

    return None


# ---------------------------------------------------------------------------
# C. BEHAVIORAL SUPPRESSION
# ---------------------------------------------------------------------------

def _behavioral(env: MessageEnvelope, s: Signals) -> Decision | None:
    # Exemption, checked first: a verified brand's own anti-fraud advisory is
    # legitimate communication. It is neither scam (the safety layer already
    # suppressed the credential keywords) nor marketing, so it must not be
    # swept up by the bulk-promotion rule below (sample_msg_048).
    if s.is_safety_advisory and s.business_verified:
        return Decision(
            "digest", "business_update", "VERIFIED_LEGIT_NOT_IMMEDIATE",
            "C0_verified_advisory",
            _strength(True, True, not s.is_promotional),
        )

    # Exemption: an admin's operational post in a society, school, or safety
    # group is never suppressed by forward-count or mute-state heuristics. The
    # problem statement is explicit that a muted group can still carry
    # something the user must see. An admin's time-critical notice is often
    # reshared widely, so a high forward count says nothing about its value.
    # Greetings from admins are not exempt - they are still chatter.
    if s.sender_is_admin and s.is_operational_group and not s.is_greeting:
        return None

    # Marketing the user has explicitly or effectively opted out of.
    #
    # Type is decided by the sender's verification status, not by whether a
    # relationship row exists. A verified brand blasting an offer is an unwanted
    # `promotion`; an unverified account doing the same is `spam`. Getting this
    # backwards was the sample_msg_043 / sample_msg_047 error.
    if s.is_promotional or (env.business_id and not s.is_transactional):
        marketing_type = "promotion" if s.business_verified else "spam"

        if s.opted_out or (s.business_known and not s.allows_promotions and s.habitually_ignored):
            return Decision(
                "mute", marketing_type, "MARKETING_OPTED_OUT", "C1_marketing_opted_out",
                _strength(True, s.opted_out, s.habitually_ignored, not s.allows_promotions),
            )
        # Bulk marketing from an account the user has no relationship with.
        if env.business_id and not s.business_known:
            code = "MARKETING_OPTED_OUT" if s.business_verified else "BULK_PROMOTION"
            return Decision(
                "mute", marketing_type, code, "C2_bulk_promotion",
                _strength(True, s.high_reports, s.young_account, not s.business_verified),
            )

    # Chain forwards and blessing chains this user routinely ignores.
    #
    # `forward` describes content with no purpose beyond being passed on. It is
    # NOT a label for "arrived with a high forward count" - a forwarded
    # marketplace listing is still a promotion and a forwarded society notice
    # is still an event. Typing from the forward flag rather than the content
    # mislabelled six rows.
    if s.is_forward_chain and (s.habitually_ignored or s.is_chain_forward):
        muted_type = (
            "greeting" if s.is_greeting
            else "promotion" if (s.is_promotional or s.is_marketplace_listing)
            else "event" if s.is_event
            else "forward"
        )
        return Decision(
            "mute", muted_type, "FORWARD_HABIT_IGNORED", "C3_forward_habit",
            _strength(True, s.is_chain_forward, s.habitually_ignored, s.is_greeting),
        )

    # Content is fine, but this user has ignored near-identical history.
    # This is the sample_msg_044 / sample_msg_045 divergence.
    if s.habitually_ignored and not s.is_same_day_operational and not s.has_deadline:
        mtype = (
            "promotion" if (s.is_promotional or s.is_marketplace_listing)
            else "greeting" if s.is_greeting
            else "business_update" if env.business_id
            else "personal"
        )
        return Decision(
            "mute", mtype, "SIMILAR_IGNORED", "C4_similar_ignored",
            _strength(True, s.dismissals > s.opens, s.muted_after > 0),
        )

    # A muted group with nothing directed at this user.
    if s.group_muted and not s.mentions_recipient:
        return Decision(
            "mute", _content_type(env, s), "GROUP_MUTED_LOW_VALUE", "C5_group_muted",
            _strength(True, not s.is_same_day_operational, s.group_dismissals > 0),
        )

    # The user reads little of this group and dismisses most of its pings.
    if s.group_disengaged and not s.is_same_day_operational and not s.sender_is_admin:
        return Decision(
            "mute", _content_type(env, s), "GROUP_DISENGAGED", "C6_group_disengaged",
            _strength(True, s.group_replies == 0),
        )

    return None


# ---------------------------------------------------------------------------
# D. NOTIFY
# ---------------------------------------------------------------------------

def _notify(env: MessageEnvelope, s: Signals) -> Decision | None:
    # A health, safety, or family emergency from someone the user knows. This
    # outranks engagement history by design: notification fatigue must never
    # suppress "dad is unwell, we are going to the clinic" (sample_msg_042).
    if s.is_emergency and (s.known_sender or s.sender_is_admin):
        if s.is_short_voice or s.asks_directly or s.is_same_day_operational:
            return Decision(
                "notify", "urgent", "CLOSE_CONTACT_URGENT", "D0_emergency",
                _strength(True, s.is_short_voice or s.asks_directly, s.known_sender),
            )

    # Work escalation with a stated dependency.
    if (s.is_work_context or env.conversation_type == "personal") and s.has_deadline:
        return Decision(
            "notify", "urgent", "WORK_DEADLINE", "D1_work_deadline",
            _strength(True, s.is_same_day_operational, s.asks_directly, s.known_sender),
        )

    # A school admin's circular is operational by construction: it carries
    # timings, consent, and absences that parents act on the same day, even when
    # the text itself contains no explicit urgency token (sample_msg_046).
    if s.sender_is_admin and s.group_type == "school_group" and (
        s.is_same_day_operational or s.is_event or s.is_transactional
    ):
        return Decision(
            "notify", "event", "SCHOOL_SAME_DAY", "D2_school_same_day",
            _strength(True, s.is_same_day_operational, s.is_event),
        )

    # Group admin pushing a same-day operational update.
    if s.sender_is_admin and s.is_operational_group and s.is_same_day_operational:
        return Decision(
            "notify", "urgent", "ADMIN_TIME_SENSITIVE", "D3_admin_time_sensitive",
            _strength(True, True, s.asks_directly),
        )

    # Verified business whose message matches a live relationship.
    if env.business_id and s.business_verified and s.business_known and s.is_transactional:
        booking_like = any(
            w in s.business_relationship
            for w in ("booking", "appointment", "reservation", "prescription", "clinic", "flight", "hotel")
        ) or any(
            w in env.lower for w in ("appointment", "booking", "reservation", "prescription", "check-in")
        )
        if booking_like:
            return Decision(
                "notify", "event", "VERIFIED_MATCHES_BOOKING", "D4_verified_booking",
                _strength(True, s.is_same_day_operational, s.allows_promotions or s.business_known),
            )
        order_like = any(
            w in s.business_relationship
            for w in ("delivery", "order", "grocery", "pickup", "return", "ride")
        ) or any(w in env.lower for w in ("your order", "order ending", "delivery", "out for delivery"))
        if order_like:
            return Decision(
                "notify", "business_update", "VERIFIED_MATCHES_ORDER", "D5_verified_order",
                _strength(True, s.is_same_day_operational, True),
            )

    # A live payment obligation with a real relationship behind it. Marketing
    # copy that happens to quote a price is not a payment obligation.
    if (
        env.business_id
        and s.business_verified
        and s.business_known
        and s.payment_hits
        and not s.is_promotional
    ):
        if s.is_same_day_operational and not s.sensitive_ask:
            return Decision(
                "notify", "payment", "PAYMENT_DUE_KNOWN_BUSINESS", "D6_payment_due",
                _strength(True, True, s.business_category in ("bank", "payments", "utilities")),
            )

    # A trusted contact asking this user directly, one-to-one.
    if env.conversation_type == "personal" and s.known_sender and s.asks_directly:
        if s.is_same_day_operational:
            return Decision(
                "notify", "urgent", "CLOSE_CONTACT_URGENT", "D7_close_contact_urgent",
                _strength(True, s.engagement_rate > 0.5, True),
            )
        return Decision(
            "notify", "personal", "DIRECT_ASK", "D8_personal_direct_ask",
            _strength(True, s.engagement_rate > 0.5, False),
        )

    # A known contact in a group asking this user for something time-bound.
    if s.asks_directly and s.known_sender and s.is_same_day_operational and not s.is_promotional:
        return Decision(
            "notify", "personal", "DIRECT_ASK", "D9_group_direct_ask",
            _strength(True, s.engagement_rate > 0.5, s.sender_history_count > 1),
        )

    return None


# ---------------------------------------------------------------------------
# E. DIGEST - the safe middle
# ---------------------------------------------------------------------------

def _digest(env: MessageEnvelope, s: Signals) -> Decision:
    # Promotional but wanted.
    if s.is_promotional:
        if s.allows_promotions or s.business_known:
            return Decision(
                "digest", "promotion", "PROMO_OPTED_IN", "E1_promo_opted_in",
                _strength(s.allows_promotions, s.business_known, s.engagement_rate > 0.5),
            )
        if env.conversation_type == "group":
            return Decision(
                "digest", "promotion", "OFFER_LOW_PRIORITY", "E2_group_offer",
                _strength(True, not s.habitually_ignored),
            )
        return Decision(
            "digest", "promotion", "INTEREST_LOW_PRIORITY", "E3_promo_interest",
            _strength(s.business_known, not s.habitually_ignored),
        )

    # Peer-to-peer selling. Structurally a promotion even though a person sent
    # it. A dedicated marketplace group is a known-interest channel; the same
    # listing posted elsewhere is just an offer.
    if s.is_marketplace_listing and not env.business_id:
        code = (
            "INTEREST_LOW_PRIORITY"
            if s.group_type == "marketplace"
            else "OFFER_LOW_PRIORITY"
        )
        return Decision(
            "digest", "promotion", code, "E3b_marketplace_listing",
            _strength(True, not s.habitually_ignored, s.known_sender),
        )

    # Legitimate business traffic with no urgency.
    if env.business_id:
        return Decision(
            "digest", "business_update", "VERIFIED_NON_URGENT", "E4_business_non_urgent",
            _strength(s.business_verified, s.business_known, s.is_transactional),
        )

    # Greetings and blessing posts that are not chain forwards.
    if s.is_greeting:
        return Decision(
            "digest", "greeting", "HARMLESS_GREETING", "E5_greeting",
            _strength(True, not s.is_chain_forward, not s.habitually_ignored),
        )

    # A stranger, but a benign one. This is checked before topic typing: what
    # matters about an unfamiliar sender is that they are unfamiliar, so the
    # type is `unknown` regardless of what the message happens to be about
    # (sample_msg_049).
    if s.first_contact and not env.business_id:
        return Decision(
            "digest", "unknown", "UNFAMILIAR_BENIGN", "E9_unfamiliar_benign",
            _strength(True, not s.sensitive_ask, not s.is_promotional),
        )

    # Scheduled events with no same-day pressure.
    if s.is_event:
        code = "GROUP_USEFUL_NOT_URGENT" if env.group_id else "EVENT_NOT_IMMINENT"
        return Decision(
            "digest", "event", code, "E6_event_not_imminent",
            _strength(True, not s.is_same_day_operational, s.sender_is_admin),
        )

    # Forwarded content that this user does not habitually ignore.
    if env.forwarded_count > 0:
        return Decision(
            "digest", _content_type(env, s, default="forward"),
            "GROUP_USEFUL_NOT_URGENT", "E7_forward_benign",
            _strength(True, not s.is_chain_forward),
        )

    # Media with nothing urgent attached to it.
    if s.has_media and not s.perceived:
        return Decision(
            "digest", "personal", "MEDIA_NO_URGENCY", "E8_media_unperceived",
            _strength(s.known_sender, not s.habitually_ignored, False),
        )

    # A trusted contact with nothing to act on.
    #
    # Two gold templates cover this, split by who the message is aimed at.
    # Broadcast chatter in a social group is "safe casual chat"; something
    # addressed to this user by someone close is "the sender is trusted, but".
    if s.known_sender:
        close_tie = (
            env.conversation_type == "personal"
            or s.is_close_tie_group
            or s.asks_directly
        )
        code = "TRUSTED_NOT_URGENT" if close_tie else "CASUAL_CHAT"
        return Decision(
            "digest", "personal", code, "E10_trusted_not_urgent",
            _strength(True, s.known_sender, s.engagement_rate > 0.5),
        )

    return Decision(
        "digest", "unknown", "UNKNOWN_LOW_SIGNAL", "E11_default",
        _strength(False, False, True),
    )


# ---------------------------------------------------------------------------
# Cascade
# ---------------------------------------------------------------------------

def _repair_type(env: MessageEnvelope, s: Signals, d: Decision) -> Decision:
    """A message muted because it looked dangerous must be TYPED as dangerous.

    The cascade could reach `mute` through a behavioural branch (muted group,
    habitually ignored) while risk signals were also firing, and then type the
    row `personal`. That is internally inconsistent - the row says "we
    suppressed this as ordinary chat" about a message containing a credential
    request - and it was the single largest source of harness misses, where the
    action was right and only the type was wrong.
    """
    if d.action != "mute" or d.message_type in ("scam", "spam"):
        return d
    risky = bool(
        s.injection_hits
        or s.semantic_injection
        or s.credential_hits
        or s.money_demand_hits
        or s.qr_hits
        or s.is_fraud_structure
        or s.semantic_only_threat
        or s.lookalike_domain
    )
    if not risky:
        return d

    deceptive = bool(
        s.injection_hits or s.semantic_injection or s.credential_hits
        or s.lookalike_domain or s.qr_hits or s.is_fraud_structure
    )
    return Decision(
        action="mute",
        message_type="scam" if deceptive else "spam",
        reason_code=_scam_reason(env, s),
        rule=f"{d.rule}+type_repair",
        strength=d.strength,
    )


def decide(env: MessageEnvelope, s: Signals) -> Decision:
    for stage in (_safety, _urgent_pull, _behavioral, _notify):
        decision = stage(env, s)
        if decision is not None:
            return _repair_type(env, s, decision)
    return _repair_type(env, s, _digest(env, s))
