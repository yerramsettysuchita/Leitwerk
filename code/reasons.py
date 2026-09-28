"""S7 reason bank (LEITWERK.md sec 7).

The gold `reason` values in sample_messages.csv come from a small closed
template bank and repeat verbatim across rows. We do not generate prose. We
select the template whose firing condition matches the signals that actually
moved the decision, so the reason can never contradict the row.

Templates marked GOLD are copied verbatim from observed gold labels. Templates
marked NEW cover firing conditions the 30 samples do not exercise; they match
the gold register exactly - one declarative sentence, no hedging, referencing
the signal rather than restating the content.
"""

REASONS: dict[str, str] = {
    # ---- notify (GOLD) -----------------------------------------------------
    "ADMIN_TIME_SENSITIVE": "A trusted group admin sent a time-sensitive update that should interrupt the user.",
    "SCHOOL_SAME_DAY": "A school admin sent a same-day operational update that the user is likely to need immediately.",
    "WORK_DEADLINE": "The message is from a work context and contains a direct deadline or meeting dependency.",
    "DIRECT_ASK": "The sender directly asks this user for a response or action.",
    "CLOSE_CONTACT_URGENT": "A close contact sent a short urgent request that should interrupt the user.",
    "VERIFIED_MATCHES_ORDER": "A verified business is sending an update that matches the user's recent order history.",
    "VERIFIED_MATCHES_BOOKING": "A verified business is sending a reminder that matches the user's recent booking history.",
    # ---- digest (GOLD) -----------------------------------------------------
    "GROUP_USEFUL_NOT_URGENT": "The message is useful group information, but it is not urgent enough to interrupt the user.",
    "HARMLESS_GREETING": "The message is a harmless greeting that can be read later.",
    "CASUAL_CHAT": "The message is safe casual chat with no urgent action required.",
    "TRUSTED_NOT_URGENT": "The sender is trusted, but the message has no urgent action or safety relevance.",
    "VERIFIED_NON_URGENT": "A verified business is sending a legitimate but non-urgent update.",
    "VERIFIED_LEGIT_NOT_IMMEDIATE": "The verified business message is legitimate but does not require immediate attention.",
    "PROMO_OPTED_IN": "The message is promotional but matches a topic or business the user has opted into.",
    "OFFER_LOW_PRIORITY": "The offer is potentially relevant, but it does not need immediate attention.",
    "INTEREST_LOW_PRIORITY": "The message matches the user's known interests but is still low priority.",
    "UNFAMILIAR_BENIGN": "The sender is unfamiliar, but the message does not show urgency, payment pressure, or safety risk.",
    # ---- mute (GOLD) -------------------------------------------------------
    "FORWARD_HABIT_IGNORED": "The sender has a pattern of repeated forwards or greetings that the user usually ignores.",
    "MARKETING_OPTED_OUT": "The user has opted out of or repeatedly dismissed similar marketing messages.",
    "SIMILAR_IGNORED": "Similar historical messages were ignored, dismissed, or muted by this user.",
    "OTP_SUSPICIOUS_FLOW": "The message asks for urgent OTP or account verification through a suspicious flow.",
    "FAKE_SUPPORT_PRESSURE": "The message uses fake support language and account-blocking pressure to push the user into action.",
    "FIRST_CONTACT_SENSITIVE": "This is the first message from the sender and it asks for sensitive verification or payment.",
    "INJECTION_ATTEMPT": "The message tries to instruct the router, but the routing decision should be based on the actual content and risk.",
    # ---- scam mechanisms ---------------------------------------------------
    # The `reason` axis is scored on usefulness AND consistency. One generic
    # OTP sentence was being stamped on QR-payment, advance-fee, and prize-bait
    # scams that contain no OTP at all - a reason that contradicts its own
    # message is worse than a vague one, because it is checkably false. Each
    # template below names the mechanism actually detected.
    "PAYMENT_LINK_DEMAND": "The message pushes the user to pay through an unofficial link or QR code.",
    "ADVANCE_FEE": "The message demands an upfront fee before releasing a promised amount.",
    "PRIZE_BAIT": "The message claims the user has won something and asks them to claim it quickly.",
    "BANK_DETAIL_REQUEST": "The message asks the user to share bank account or card details.",
    "DELIVERY_CHARGE_SCAM": "The message invents a pending delivery charge to extract a payment.",
    "UNOFFICIAL_PAYMENT_CHANNEL": "A payment is being collected through a personal channel rather than the official one.",

    # ---- NEW: firing conditions the 30 samples do not exercise --------------
    "LOOKALIKE_DOMAIN": "The sender uses a lookalike domain that does not match the brand's official domain.",
    "UNVERIFIED_SENSITIVE": "An unverified business account is asking for payment or verification details.",
    "GROUP_MUTED_LOW_VALUE": "The user has muted this group and the message carries no direct mention or urgent action.",
    "GROUP_DISENGAGED": "The user reads little of this group and dismisses most of its notifications.",
    "MENTION_IN_MUTED_GROUP": "The user muted this group, but the message directly mentions them and needs a response.",
    "PAYMENT_DUE_KNOWN_BUSINESS": "A business the user actively pays is sending a due-payment reminder that needs attention today.",
    "EVENT_NOT_IMMINENT": "The message is about a scheduled event that is not immediate enough to interrupt the user.",
    "BULK_PROMOTION": "The message is bulk marketing from an account the user has no active relationship with.",
    "MEDIA_NO_URGENCY": "The attached media carries no urgent action or safety relevance for this user.",
    "OPERATIONAL_UPDATE": "A group admin sent an operational update that is useful but does not need to interrupt the user.",
    "UNKNOWN_LOW_SIGNAL": "The message carries no clear urgency, relationship, or risk signal for this user.",
}


def render(code: str) -> str:
    """Resolve a reason code to its sentence. Unknown codes fail loudly rather
    than silently emitting a placeholder into the submission."""
    try:
        return REASONS[code]
    except KeyError:  # pragma: no cover - guards against typos in decide.py
        raise KeyError(f"unknown reason code: {code!r}") from None
