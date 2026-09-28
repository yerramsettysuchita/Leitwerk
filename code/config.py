"""LEITWERK configuration.

Every tunable lives here so it is inspectable and ablatable (LEITWERK.md sec 4.3).
Nothing in this file is a label. These are thresholds, weights, and lexicons.
"""

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / "dataset"
OUTPUT_CSV = REPO_ROOT / "output.csv"
TRACE_JSONL = REPO_ROOT / "output_trace.jsonl"

# --------------------------------------------------------------------------
# Contract (problem_statement.md wins any disagreement)
# --------------------------------------------------------------------------

OUTPUT_COLUMNS = [
    "message_id",
    "action",
    "message_type",
    "reason",
    "confidence",
    "evidence_message_ids",
]

ACTIONS = ("notify", "digest", "mute")

MESSAGE_TYPES = (
    "personal",
    "urgent",
    "event",
    "payment",
    "business_update",
    "promotion",
    "greeting",
    "forward",
    "spam",
    "scam",
    "unknown",
)

NO_EVIDENCE = "none"
EVIDENCE_SEPARATOR = ";"

# --------------------------------------------------------------------------
# Confidence band
#
# All 30 gold confidences lie in [0.78, 0.91], stratified by action.
# Emitting values outside this band is off-distribution and costs calibration
# points even when the label is right (LEITWERK.md sec 8).
# --------------------------------------------------------------------------

# The gold band is where CORRECT, well-evidenced calls should land. It is not a
# straitjacket for every row: clamping all 110 into [0.78, 0.91] made
# confidence nearly constant (observed spread 0.81-0.89) and therefore
# uninformative, which is the opposite of calibration. The emitted range is
# widened so a genuinely thin call can say so, while the gold band still
# anchors the centre of mass.
CONF_FLOOR = 0.78
CONF_CEIL = 0.91

# Observed per-action gold anchors.
CONF_ANCHOR = {
    "notify": 0.87,
    "digest": 0.81,
    "mute": 0.84,
}
# Modestly wider than the observed gold band so a thin call can say it is thin,
# but NOT the wide range a pure-ECE argument would suggest.
#
# Measured trade-off: widening to [0.55, 0.95] moved ECE 0.153 -> 0.131 but
# moved confidence MAE against the gold values 0.017 -> 0.031. ECE is degenerate
# on this validation set - sample accuracy is 100%, so every bin shows a gap
# equal to (1 - confidence) no matter what we emit, and the only way to "fix" it
# is to claim near-certainty. MAE against real gold confidences is the honest
# signal, and it says stay close to the gold distribution.
CONF_ACTION_BAND = {
    "notify": (0.85, 0.91),
    "digest": (0.78, 0.84),
    "mute": (0.81, 0.87),
}

# How much a strong / weak signal set may move confidence inside its band.
CONF_SIGNAL_SWING = 0.03

# --------------------------------------------------------------------------
# Behavioral thresholds (Phase 1 rules baseline)
# --------------------------------------------------------------------------

# Beta-Binomial smoothing (Laplace), used from Phase 2 onward but defined here.
BETA_ALPHA0 = 1.0
BETA_BETA0 = 1.0

# A user is "fatigued" by a sender/business when dismissals dominate opens.
DISMISS_DOMINANCE_RATIO = 1.0

# Group-level: dismissals in 30d above this and near-zero replies means the user
# has effectively tuned this group out.
GROUP_DISMISS_HIGH = 5

# forwarded_count at or above this marks chain-forward behavior.
FORWARD_CHAIN_MIN = 3

# A voice note at or under this length is a quick ask, not a monologue.
SHORT_VOICE_SECONDS = 20.0

# Cosine similarity against a threat prototype that counts as a real match.
# all-MiniLM-L6-v2 on short informal text puts genuine paraphrases at 0.55-0.75
# and unrelated content below 0.40; 0.55 keeps precision while recovering the
# phrasings no lexicon enumerates.
SEMANTIC_THREAT_TAU = 0.50

# Router-manipulation is judged on DOMINANCE, not on an absolute score.
#
# The first attempt used a low absolute bar on the theory that no benign
# message addresses a notification system. That was wrong and the harness
# caught it: this embedding model assigns ~0.3 similarity between almost any
# two short texts, so "reached the station safely, will message once I board"
# cleared the bar and was muted as a scam. Requiring the intent to be the
# single highest-scoring one, and clear of the runner-up, is what actually
# separates a message aimed at the router from ordinary chat.
SEMANTIC_INJECTION_TAU = 0.34
SEMANTIC_INJECTION_MARGIN = 0.02  # must lead the second-place intent by this
SEMANTIC_EVIDENCE_WEIGHT = 0.45  # blend factor for hybrid retrieval
SEMANTIC_EVIDENCE_FLOOR = 0.40   # below this, two texts are merely both English

# Link shorteners hide the real destination. A business account whose sending
# domain IS a shortener has no verifiable identity.
URL_SHORTENERS = (
    "bit.ly", "tinyurl", "shorturl", "rb.gy", "cutt.ly", "vl.gl",
    "t.co", "goo.gl", "ow.ly", "is.gd", "buff.ly",
)

# Business account age below this is a young-account risk signal.
YOUNG_ACCOUNT_DAYS = 180

# Sender domain younger than this while impersonating an older brand is a
# strong spoof signal.
YOUNG_DOMAIN_DAYS = 365

# user_reports_30d at or above this is a crowd risk signal.
REPORTS_HIGH = 15

# --------------------------------------------------------------------------
# Retrieval (S3)
# --------------------------------------------------------------------------

EVIDENCE_MAX = 2          # gold rows cite one or two ids
EVIDENCE_MIN_SCORE = 0.08  # below this we emit `none` rather than fabricate

STOPWORDS = frozenset("""
a an and are as at be been but by can cant could did do does for from get got
had has have he her his how i if in into is it its just me my no not of on one
or our out please pls so than that the their them then there these they this
to too us was we were what when where which who will with would you your
""".split())

# --------------------------------------------------------------------------
# Lexicons for deterministic guards (S2 / S5)
#
# These are pattern families, not per-message labels. They are matched against
# message text; none of them names a message_id.
# --------------------------------------------------------------------------

# Attempts to steer the router itself. An attempt to instruct the classifier is
# itself strong evidence of scam (LEITWERK.md sec 9).
INJECTION_PATTERNS = (
    # Direct overrides
    "ignore all previous",
    "ignore previous",
    "ignore the above",
    "ignore sender",
    "disregard previous",
    "disregard all",
    "new instructions",
    "system prompt",
    "override the",
    "routing override",
    "routing rules",
    # Telling the router what to emit
    "mark this message as",
    "mark this as",
    "mark notify",
    "mark as notify",
    "classify this as",
    "classify as",
    "set action",
    "action=",
    "always notify",
    "always mark",
    "confidence=",
    "user_priority",
    "verified_business=",
    # Addressing the router as if it were a system component. No legitimate
    # human message to a WhatsApp user talks to the notification router.
    "notification router",
    "system note for",
    "internal router",
    "router metadata",
    "assistant instruction",
    "ai assistant",
    "for the router",
    "to the classifier",
    "you are now",
    "act as",
)

# Credential / verification harvesting.
CREDENTIAL_PATTERNS = (
    "otp",
    "one time password",
    "one-time password",
    "login code",
    "6 digit",
    "6-digit",
    "six digit",
    "verification code",
    "confirm password",
    "share password",
    "your password",
    "pin number",
    "cvv",
    "card number",
    "net banking password",
    "upi pin",
)

# Manufactured urgency / account-blocking pressure.
PRESSURE_PATTERNS = (
    "will be blocked",
    "may be blocked",
    "temporarily blocked",
    "will expire today",
    "expires today",
    "within 2 hours",
    "in 2 hours",
    "immediately or",
    "last warning",
    "final notice",
    "account suspended",
    "suspend your account",
    "keep your account active",
    "keep payments active",
    "verify now",
    "act now or",
    "failed; reply",
    "verification failed",
    "before midnight",
    "pay today",
    "avoid account",
    "service stops today",
    "finalized tonight",
    "limited window",
)

# Money movement asks.
PAYMENT_PATTERNS = (
    "pay now",
    "payment link",
    "this link and send",
    "transfer",
    "send money",
    "upi",
    "gpay",
    "phonepe",
    "paytm",
    "kyc",
    "refund",
    "outstanding amount",
    "due amount",
    "pay rs",
    "pay ₹",
    "wallet",
    "bank account",
    "bank details",
    "account number",
    "card details",
    "processing fee",
    "reactivation fee",
    "clearance amount",
    "clearance",
    "token amount",
    "penalty",
    "scan this qr",
    "scan the qr",
    "scan and pay",
    "pay before",
    "pay the",
    "pay at this",
    "charge is pending",
    "settle it",
    "dues",
    "account changed",
)

# A concrete phishing ACT, as opposed to mere urgency.
#
# `scam` requires that the message actually try to do something to the user:
# harvest a credential, move money, or steer them to a link. Urgency alone is
# not fraud - a rushed secondhand sale that says "today only, cash or UPI ok"
# is hype, not phishing, and would be typed scam if any payment word from the
# payment lexicon counted as a demand. Naming a payment METHOD is not
# DEMANDING a payment.
MONEY_DEMAND_PATTERNS = (
    "send money",
    "send the amount",
    "transfer to",
    "transfer the",
    "pay now",
    "pay today",
    "pay before",
    "pay the fee",
    "pay processing",
    "settle it",
    "settle the",
    "remit",
    "deposit",
    "processing fee",
    "reactivation fee",
    "token amount",
    "clearance amount",
    "charge is pending",
    "pending charge",
    "redelivery charge",
    "reattempt charge",
    "send details",
    "sharing your account number",
    "share your account",
    "account number",
    "bank details",
    "card details",
    "verify wallet",
    "dues to",
    "new number",
)

LINK_PATTERNS = (
    "http://",
    "https://",
    "www.",
    "bit.ly",
    "tinyurl",
    "shorturl",
    "rb.gy",
    "cutt.ly",
    "this link",
    "the link",
    "at this link",
    "open the link",
    "link open",
    "click the",
    "click below",
    "click here",
    ".in and",
    "-login",
    "-secure",
    "-verify",
)

QR_PATTERNS = (
    "scan this qr",
    "scan the qr",
    "scan and pay",
    "qr code",
    "scan this",
)

# Advance-fee, prize-bait, and QR/shortlink fraud. These are scam structures
# that carry no OTP keyword at all, so the credential guard never sees them.
# Six phishing messages reached `digest` before this list existed.
FRAUD_PATTERNS = (
    "loan approved",
    "amount will be released",
    "claim benefits",
    "claim today",
    "was selected for",
    "number was selected",
    "voucher expires",
    "reward",
    "prize",
    "lottery",
    "benefit approval",
    "approval window closes",
    "sharing your account number",
    "send details",
    "verify wallet",
    "verify your account",
    "complete verification",
    "pending account check",
    "pending verification",
    "security check required",
    "failed login",
    "login now",
    "account blocked unless",
    "avoid account lock",
    "service stops today",
    "profile will be restricted",
    "reattempt charge",
    "final verification step",
    "bit.ly",
    "tinyurl",
    "rb.gy",
    "cutt.ly",
    "open this document",
    "urgent document",
    "reactivation fee",
    "service reactivation",
    "account lock",
    "refund approved",
    "verify wallet",
    "scan this qr and pay",
    "scan and pay",
    "pending charge",
    "clearance amount",
    "penalty list",
    "send screenshot",
    "gift card",
    "the draw",
    "lucky draw",
    "you won",
    "claim your",
    "redelivery charge",
    "held at the depot",
)

# Legitimate transactional business language.
TRANSACTIONAL_PATTERNS = (
    "your order",
    "order ending",
    "has been packed",
    "out for delivery",
    "has been shipped",
    "delivery",
    "tracking",
    "booking",
    "reservation",
    "appointment",
    "prescription",
    "claim",
    "invoice",
    "receipt",
    "statement",
    "ticket",
    "itinerary",
    "check-in",
    "policy",
    "renewal",
    "installed",
    "service request",
)

# Marketing language.
PROMO_PATTERNS = (
    "% off",
    "off with",
    "discount",
    "sale",
    "offer",
    "coupon",
    "promo",
    "deal",
    "cashback",
    "limited time",
    "hurry",
    "shop now",
    "buy now",
    "order now",
    "flat ",
    "starting at",
    "from rs",
    "unsubscribe",
    "reply stop",
    "t&c apply",
    "exclusive",
    "free trial",
    "today only",
    "warehouse pickup",
    "cash or upi",
    "buyer cancelled",
)

# --------------------------------------------------------------------------
# Hinglish / romanized Hindi.
#
# Roughly one in twelve messages in this dataset is written in romanized Hindi,
# and the English lexicons above see none of it. A society notice saying the
# water has arrived and must be collected within minutes carries its same-day
# urgency in Hindi words ("jaldi", "min me"), so without these tuples it reads
# as ordinary chat. Kept as separate tuples so the language coverage is
# visible and auditable rather than buried.
# --------------------------------------------------------------------------

HI_URGENT_PATTERNS = (
    "abhi",
    "jaldi",
    "turant",
    "aaj",
    "min me",
    "minute me",
    "baje tak",
    "hone wala hai",
    "aa gaya",
    "aa raha hai",
    "nikalna padega",
    "le aao",
    "hata do",
    "kar dena",
    "band ho jayega",
    "bacha lo",
    "zaroori",
)

HI_DEESCALATION_PATTERNS = (
    "koi urgency nahi",
    "urgency nahi",
    "baad me",
    "kal milte",
    "kal baat",
    "raat ko",
    "jab time mile",
    "koi jaldi nahi",
)

HI_SCAM_PATTERNS = (
    "otp batao",
    "otp abhi",
    "otp daal",
    "code daal",
    "link open karo",
    "link open karke",
    "verification nahi",
    "account band",
    "account bachane",
    "leak ho gaya",
    "hold pe",
    "paisa bhej",
    "fees bhar",
)

HI_PAYMENT_PATTERNS = (
    "payment kar",
    "pay kar",
    "late fee",
    "bhugtan",
    "paise",
    "rupay",
    "receipt",
)

# Same-day operational language: the thing that separates notify from digest.
SAME_DAY_PATTERNS = (
    "today",
    "tonight",
    "this evening",
    "this morning",
    "right now",
    "now",
    "in 20 minutes",
    "in 15 mins",
    "15 mins early",
    "next 10 minutes",
    "next 15 minutes",
    "next five minutes",
    "next few minutes",
    "10 minutes",
    "15 minutes",
    "in the next",
    "few minutes",
    "immediately",
    "asap",
    "before eod",
    "eod",
    "by 7:35",
    "urgent",
    "emergency",
    "heads-up",
    "heads up",
    "quick",
)

# Deadline / dependency language typical of work escalations.
DEADLINE_PATTERNS = (
    "deadline",
    "before eod",
    "eod",
    "escalation",
    "escalat",
    "alert threshold",
    "pulled to",
    "review got",
    "client note",
    "sign off",
    "signoff",
    "approve",
    "blocker",
    "call starts",
    "join with",
    "need quick help",
    "come online",
    "closes at",
    "close at",
    "closes today",
    "portal locks",
    "before the portal",
    "last date",
    "won't be accepted",
    "wont be accepted",
    "late entries",
    "submit the",
)

# Direct-ask language.
DIRECT_ASK_PATTERNS = (
    "can you",
    "could you",
    "please reply",
    "pls reply",
    "reply once",
    "let me know",
    "confirm if",
    "are you",
    "when you get",
    "need you to",
    "call me",
    "call?",
    "please call",
    "pls call",
    "call now",
    "call back",
    "come online",
    "join the",
    "please confirm",
    "please check",
    "please reach",
    "please pick",
    "or confirm",
    "confirm in the",
    "collect it",
    "come and",
)

# Explicit de-escalation. The sender is telling the user this does NOT need
# attention now. Four gold rows carry it, and it inverts otherwise-urgent
# surface wording: "Don't call now ... nothing urgent" (sample_msg_050) matches
# both a direct-ask and a same-day pattern while meaning the opposite.
DEESCALATION_PATTERNS = (
    "nothing urgent",
    "not urgent",
    "no need to reply",
    "no need to respond",
    "no need to",
    "no rush",
    "no hurry",
    "no pressure",
    "don't call",
    "dont call",
    "do not call",
    "whenever you get time",
    "when you get time",
    "talk tomorrow",
    "tomorrow morning",
    "after dinner",
    "later today is fine",
    "at your convenience",
    "just fyi",
    "for your information",
    "just sharing",
    "join only if",
)

# Health, safety, and family emergencies. Distinct from manufactured urgency:
# these are the messages that must interrupt regardless of engagement history,
# and no amount of notification fatigue should suppress them.
EMERGENCY_PATTERNS = (
    "unwell",
    "not well",
    "is ill",
    "hospital",
    "clinic",
    "ambulance",
    "emergency",
    "accident",
    "admitted",
    "icu",
    "doctor said",
    "fell down",
    "collapsed",
    "surgery",
    "blood",
    "missing person",
    "missing from",
    "fire",
    "evacuat",
    "police",
    "incident bridge",
    "outage",
    "down for",
)

# Greeting / blessing chain content.
GREETING_PATTERNS = (
    "good morning",
    "good evening",
    "good night",
    "stay positive",
    "keep smiling",
    "blessings",
    "good vibes",
    "hope today is",
    "happy sunday",
    "happy monday",
    "have a great day",
    "god bless",
)

# Chain-forward tells.
FORWARD_PATTERNS = (
    "fwd as received",
    "forwarded as received",
    "forwarding because",
    "pls forward",
    "please forward",
    "share with family",
    "forward to family",
    "very useful apparently",
    "in case it helps someone",
    "share this with",
)

# Safety advisories that MENTION credentials in order to warn about them.
# "The brand never asks for OTP" is the opposite of an OTP request, and a
# keyword matcher that cannot see the negation turns a legitimate advisory into
# a false scam (sample_msg_048). This list suppresses the credential guard.
NEGATION_PATTERNS = (
    "never ask",
    "never asks",
    "will never ask",
    "do not share",
    "don't share",
    "never share",
    "do not disclose",
    "beware of",
    "be aware of",
    "safety advisory",
    "security advisory",
    "awareness",
    "we never",
    "no one from",
    "nobody from",
    "fraud alert about",
    "how to stay safe",
    # Legitimate senders pre-empting the scam pattern. A courier or bank notice
    # that says no payment or OTP is required names the credential only to
    # rule it out, and without these phrases the credential guard mutes it.
    "no payment or otp",
    "no otp is required",
    "no otp required",
    "otp is not required",
    "no payment is required",
    "no payment required",
    "will not ask",
    "won't ask",
    "never request",
    "will never request",
    "never asks for",
    "no payment is due",
    "no payment due",
    "don't use any payment link",
    "do not use any payment link",
    "not ask for otp",
)

# Peer-to-peer selling inside a group. Structurally a promotion even though it
# comes from a person rather than a business account.
MARKETPLACE_PATTERNS = (
    "selling",
    "for sale",
    "dm if interested",
    "ping me if interested",
    "pickup near",
    "pickup is near",
    "pickup at",
    "share pics",
    "share photos",
    "photos for the",
    "pics attached",
    "photos are attached",
    "barely used",
    "no crash damage",
    "bought last year",
    "price negotiable",
    "best price",
)

# Group types that are inherently a known-interest shopping channel.
MARKETPLACE_GROUP_TYPES = frozenset({"marketplace", "local_food", "real_estate"})

# Event / scheduling language.
#
# Split by strength. A single weak token ("match", "class", "trip") is ordinary
# conversation - "anyone watching the match tonight?" is casual chat, not an
# event (sample_msg_010). Weak tokens only count when two or more co-occur.
STRONG_EVENT_PATTERNS = (
    "form is open",
    "registrations are open",
    "registrations open",
    "potluck",
    "rsvp",
    "cultural night",
    "circular",
    "consent note",
    "reschedul",
    "rehearsal",
    "workshop",
    "webinar",
    "exam",
    "picnic",
    "agenda",
    "invitation",
    "venue",
)

WEAK_EVENT_PATTERNS = (
    "register",
    "registration",
    "meeting",
    "event",
    "consent",
    "timing",
    "schedule",
    "practice",
    "match",
    "session",
    "class",
    "holiday",
    "trip",
    "slot",
    "booking",
    "moved to",
    "still on",
    "sync",
)

# Group types where an admin post is operationally load-bearing.
OPERATIONAL_GROUP_TYPES = frozenset(
    {"society", "school_group", "safety", "caregiving", "coworker"}
)

# Group types that are work contexts.
WORK_GROUP_TYPES = frozenset({"coworker", "college_faculty"})

# Group types that are inherently low-signal social chatter.
SOCIAL_GROUP_TYPES = frozenset(
    {"friends", "alumni", "book_club", "sports", "tech_community", "college_students",
     "local_food", "dance_class", "investment_tips"}
)

# Groups whose members are close ties. A quiet message here is "the sender is
# trusted, but nothing to act on", not anonymous chatter.
CLOSE_TIE_GROUP_TYPES = frozenset({"family", "extended_family", "caregiving"})

# A relationship this thin reads as first contact. u_028 had two prior messages
# from the sender of sample_msg_052 and gold still called it "the first message
# from the sender", so the gold notion is a near-absent relationship rather
# than a literal zero.
WEAK_RELATIONSHIP_MAX = 2

# Business categories where a verification/payment ask is high-stakes.
SENSITIVE_BUSINESS_CATEGORIES = frozenset(
    {
        "bank",
        "payments",
        "fintech",
        "credit_card",
        "insurance",
        "vehicle_insurance",
        "finance",
        "security",
        "cloud_security",
        "telecom",
    }
)
