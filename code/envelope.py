"""S1 PERCEIVE - normalize text, image, and voice into one MessageEnvelope.

Text is normalized here; media is resolved to a file path and handed to
`perceive.py`, which fills the `perception` slot with OCR and transcription
results. Every downstream stage reads `effective_text`, so a message is the same
shape to the rest of the pipeline whether its content arrived as text, as pixels
in a poster, or as speech in a voice note.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from config import DATASET_DIR
from loader import Dataset, to_dt, to_int

_WHITESPACE = re.compile(r"\s+")
_MENTION = re.compile(r"@(u_\d+)")


def normalize_text(raw: str | None) -> str:
    """Unicode-normalize and collapse whitespace. Deterministic, lossless enough
    for matching while keeping the text human-readable in traces."""
    if not raw:
        return ""
    text = unicodedata.normalize("NFKC", raw)
    text = text.replace("​", "").replace("\xa0", " ")
    return _WHITESPACE.sub(" ", text).strip()


@dataclass
class Perception:
    """Facts extracted from media. Empty until perceive.py fills it."""

    ocr_text: str = ""
    transcript: str = ""
    caption: str = ""
    source: str = "none"  # none | rapidocr | faster-whisper

    @property
    def text(self) -> str:
        return normalize_text(" ".join(p for p in (self.ocr_text, self.transcript, self.caption) if p))


@dataclass
class MessageEnvelope:
    """One incoming message plus everything S2 onward needs, in one object."""

    message_id: str
    user_id: str
    conversation_type: str
    group_id: str
    business_id: str
    sender_user_id: str
    created_at: datetime | None
    created_at_raw: str
    text: str
    media_type: str
    media_id: str
    media_path: Path | None
    forwarded_count: int
    mentions: tuple[str, ...] = ()
    perception: Perception = field(default_factory=Perception)

    @property
    def effective_text(self) -> str:
        """Text the decision layers reason over: literal text plus any media
        facts recovered in S1. For voice notes the literal text is empty, so
        this is the transcript alone."""
        parts = [p for p in (self.text, self.perception.text) if p]
        return " ".join(parts)

    @property
    def lower(self) -> str:
        return self.effective_text.lower()

    @property
    def has_media(self) -> bool:
        return bool(self.media_type)

    @property
    def is_perceived(self) -> bool:
        """True when media content was actually inspected, not just referenced."""
        return not self.has_media or self.perception.source != "none"

    @property
    def mentions_recipient(self) -> bool:
        return self.user_id in self.mentions


def build_envelope(row: dict[str, str], ds: Dataset) -> MessageEnvelope:
    text = normalize_text(row.get("message_text"))
    media_type = (row.get("media_type") or "").strip()
    media_id = (row.get("media_id") or "").strip()

    # file_path columns are relative to dataset/, e.g. `media/images/img_001.jpg`.
    media_path: Path | None = None
    if media_type == "image" and media_id in ds.images:
        media_path = DATASET_DIR / ds.images[media_id]
    elif media_type == "voice" and media_id in ds.voice_notes:
        media_path = DATASET_DIR / ds.voice_notes[media_id]

    return MessageEnvelope(
        message_id=row["message_id"],
        user_id=row["user_id"],
        conversation_type=(row.get("conversation_type") or "").strip(),
        group_id=(row.get("group_id") or "").strip(),
        business_id=(row.get("business_id") or "").strip(),
        sender_user_id=(row.get("sender_user_id") or "").strip(),
        created_at=to_dt(row.get("created_at")),
        created_at_raw=(row.get("created_at") or "").strip(),
        text=text,
        media_type=media_type,
        media_id=media_id,
        media_path=media_path,
        forwarded_count=to_int(row.get("forwarded_count")),
        mentions=tuple(_MENTION.findall(text)),
    )


def build_envelopes(rows: list[dict[str, str]], ds: Dataset) -> list[MessageEnvelope]:
    return [build_envelope(row, ds) for row in rows]
