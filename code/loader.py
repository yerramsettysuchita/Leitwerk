"""Dataset loading and joining.

Reads only from `dataset/`. No organizer files. Everything is keyed into plain
dicts so downstream stages stay pure and testable.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from config import DATASET_DIR


def _read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def to_int(value: Any, default: int = 0) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def to_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def to_dt(value: Any) -> datetime | None:
    text = (str(value) if value is not None else "").strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


@dataclass
class Dataset:
    """Every participant-facing table, joined and indexed."""

    messages: list[dict[str, str]] = field(default_factory=list)
    output_ids: list[str] = field(default_factory=list)
    samples: list[dict[str, str]] = field(default_factory=list)

    users: dict[str, dict] = field(default_factory=dict)
    groups: dict[str, dict] = field(default_factory=dict)
    group_members: dict[tuple[str, str], dict] = field(default_factory=dict)
    businesses: dict[str, dict] = field(default_factory=dict)
    user_business: dict[tuple[str, str], dict] = field(default_factory=dict)

    history: list[dict[str, str]] = field(default_factory=list)
    history_by_id: dict[str, dict] = field(default_factory=dict)
    history_by_user: dict[str, list[dict]] = field(default_factory=dict)
    events_by_id: dict[str, dict] = field(default_factory=dict)

    images: dict[str, str] = field(default_factory=dict)
    voice_notes: dict[str, str] = field(default_factory=dict)
    daily_load: dict[str, list[dict]] = field(default_factory=dict)

    # ---- derived indices ---------------------------------------------------

    def sender_history(self, user_id: str, sender_user_id: str) -> list[dict]:
        """Past messages this user received from this specific sender."""
        return [
            h
            for h in self.history_by_user.get(user_id, [])
            if h.get("sender_user_id") == sender_user_id and sender_user_id
        ]

    def business_history(self, user_id: str, business_id: str) -> list[dict]:
        """Past messages this user received from this specific business."""
        return [
            h
            for h in self.history_by_user.get(user_id, [])
            if h.get("business_id") == business_id and business_id
        ]

    def outcomes(self, messages: list[dict]) -> list[dict]:
        """message_events rows for the given history messages, in order."""
        out = []
        for h in messages:
            ev = self.events_by_id.get(h["message_id"])
            if ev is not None:
                out.append(ev)
        return out

    def daily_dismiss_rate(self, user_id: str) -> float:
        """Fraction of notifications this user dismissed, across all logged days."""
        rows = self.daily_load.get(user_id, [])
        sent = sum(to_int(r.get("notifications_sent")) for r in rows)
        dismissed = sum(to_int(r.get("notifications_dismissed")) for r in rows)
        return (dismissed / sent) if sent else 0.0


def load_dataset(dataset_dir: Path = DATASET_DIR) -> Dataset:
    ds = Dataset()

    ds.messages = _read_csv(dataset_dir / "messages.csv")
    ds.samples = _read_csv(dataset_dir / "sample_messages.csv")
    ds.output_ids = [r["message_id"] for r in _read_csv(dataset_dir / "output.csv")]

    ds.users = {r["user_id"]: r for r in _read_csv(dataset_dir / "users.csv")}
    ds.groups = {r["group_id"]: r for r in _read_csv(dataset_dir / "groups.csv")}
    ds.group_members = {
        (r["group_id"], r["user_id"]): r
        for r in _read_csv(dataset_dir / "group_members.csv")
    }
    ds.businesses = {
        r["business_id"]: r for r in _read_csv(dataset_dir / "business_accounts.csv")
    }
    ds.user_business = {
        (r["user_id"], r["business_id"]): r
        for r in _read_csv(dataset_dir / "user_business_history.csv")
    }

    ds.history = _read_csv(dataset_dir / "message_history.csv")
    ds.history_by_id = {r["message_id"]: r for r in ds.history}
    for row in ds.history:
        ds.history_by_user.setdefault(row["user_id"], []).append(row)

    ds.events_by_id = {
        r["message_id"]: r for r in _read_csv(dataset_dir / "message_events.csv")
    }

    ds.images = {r["image_id"]: r["file_path"] for r in _read_csv(dataset_dir / "images.csv")}
    ds.voice_notes = {
        r["voice_note_id"]: r["file_path"]
        for r in _read_csv(dataset_dir / "voice_notes.csv")
    }

    for row in _read_csv(dataset_dir / "daily_notification_summary.csv"):
        ds.daily_load.setdefault(row["user_id"], []).append(row)

    return ds
