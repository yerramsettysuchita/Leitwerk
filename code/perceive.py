"""S1 PERCEIVE (media) - OCR for image posters, ASR for voice notes.

21% of the scored rows are image or voice. Deciding those on metadata alone
leaves the largest single hole in the submission, so perception runs locally and
needs no API key or network:

    images  -> RapidOCR (ONNX runtime, CPU)
    voice   -> faster-whisper (CTranslate2, CPU, greedy decoding)

Determinism is a hard requirement of the contract, so ASR runs with beam_size=1,
temperature=0, and no cross-segment conditioning. Results are cached to
`media_cache.json` keyed by media id plus a hash of the file bytes, which makes
repeat runs both fast and byte-identical. Delete the cache to force re-extraction.

If an engine is unavailable the pipeline degrades honestly: the envelope keeps
`perception.source == "none"`, the belief layer applies its `unperceived_media`
humility term, and nothing is invented.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from config import REPO_ROOT
from envelope import MessageEnvelope, Perception, normalize_text

CACHE_PATH = REPO_ROOT / "media_cache.json"

# Whisper model size. `base` is the accuracy/latency knee for short voice notes;
# the dataset's longest clip is under a minute.
WHISPER_MODEL = "base"

_ocr_engine = None
_asr_model = None
_cache: dict[str, dict] | None = None


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def _load_cache() -> dict[str, dict]:
    global _cache
    if _cache is None:
        if CACHE_PATH.exists():
            _cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        else:
            _cache = {}
    return _cache


def save_cache() -> None:
    if _cache is not None:
        CACHE_PATH.write_text(
            json.dumps(_cache, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )


def _cache_key(media_id: str, path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    return f"{media_id}:{digest}"


def _cached_by_id(media_id: str) -> dict | None:
    """Cache entry for a media file that is not on disk. The raw images and
    voice notes are not redistributed, so the committed cache is looked up by
    media id alone; with the file present, the file-hash key is used as before."""
    hits = [v for k, v in _load_cache().items() if k.split(":")[0] == media_id]
    return hits[0] if len(hits) == 1 else None


# ---------------------------------------------------------------------------
# Engines, loaded lazily so a missing dependency never breaks a text-only run
# ---------------------------------------------------------------------------

def _get_ocr():
    global _ocr_engine
    if _ocr_engine is None:
        try:
            from rapidocr_onnxruntime import RapidOCR

            _ocr_engine = RapidOCR()
        except Exception as exc:  # pragma: no cover - environment dependent
            print(f"  [perceive] OCR unavailable: {exc}", file=sys.stderr)
            _ocr_engine = False
    return _ocr_engine or None


def _get_asr():
    global _asr_model
    if _asr_model is None:
        try:
            from faster_whisper import WhisperModel

            _asr_model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
        except Exception as exc:  # pragma: no cover - environment dependent
            print(f"  [perceive] ASR unavailable: {exc}", file=sys.stderr)
            _asr_model = False
    return _asr_model or None


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def _ocr_image(path: Path) -> dict:
    engine = _get_ocr()
    if engine is None:
        return {}
    result, _ = engine(str(path))
    if not result:
        return {"ocr_text": "", "source": "rapidocr"}
    # RapidOCR returns [box, text, score] in reading order.
    lines = [str(item[1]).strip() for item in result if len(item) > 1 and item[1]]
    return {"ocr_text": normalize_text(" ".join(lines)), "source": "rapidocr"}


def _asr_audio(path: Path) -> dict:
    model = _get_asr()
    if model is None:
        return {}
    segments, info = model.transcribe(
        str(path),
        beam_size=1,           # greedy: deterministic
        temperature=0.0,
        condition_on_previous_text=False,
        vad_filter=False,
    )
    text = " ".join(seg.text.strip() for seg in segments)
    return {
        "transcript": normalize_text(text),
        "duration_seconds": round(float(info.duration), 2),
        "language": info.language,
        "source": "faster-whisper",
    }


def perceive(env: MessageEnvelope, *, enabled: bool = True) -> Perception:
    """Fill the envelope's perception slot. Idempotent and cached."""
    if not enabled or not env.has_media or env.media_path is None:
        return Perception()
    if not env.media_path.exists():
        entry = _cached_by_id(env.media_id) or {}
    else:
        cache = _load_cache()
        key = _cache_key(env.media_id, env.media_path)

        if key not in cache:
            if env.media_type == "image":
                cache[key] = _ocr_image(env.media_path)
            elif env.media_type == "voice":
                cache[key] = _asr_audio(env.media_path)
            else:
                cache[key] = {}

        entry = cache.get(key) or {}
    if not entry.get("source"):
        return Perception()

    return Perception(
        ocr_text=entry.get("ocr_text", ""),
        transcript=entry.get("transcript", ""),
        caption=entry.get("caption", ""),
        source=entry["source"],
    )


def duration_seconds(env: MessageEnvelope) -> float | None:
    """Voice-note length. A real signal on its own: gold calls sample_msg_042
    'a short urgent request'. Available even when ASR is not."""
    if env.media_type != "voice" or env.media_path is None:
        return None
    cache = _load_cache()
    if not env.media_path.exists():
        entry = _cached_by_id(env.media_id) or {}
    else:
        entry = cache.get(_cache_key(env.media_id, env.media_path)) or {}
    return entry.get("duration_seconds")


def warm_cache(envelopes: list[MessageEnvelope]) -> dict[str, int]:
    """Extract every distinct media file once, up front, then persist the cache.
    Reports what actually got perceived so the run log never overstates it."""
    stats = {"image": 0, "voice": 0, "failed": 0, "cached": 0}
    seen: set[str] = set()

    for env in envelopes:
        if not env.has_media or env.media_path is None or not env.media_path.exists():
            continue
        if env.media_id in seen:
            continue
        seen.add(env.media_id)

        before = len(_load_cache())
        p = perceive(env)
        if len(_load_cache()) == before:
            stats["cached"] += 1
        if p.source == "none" or not p.text.strip():
            stats["failed"] += 1
        elif env.media_type in stats:
            stats[env.media_type] += 1

    save_cache()
    return stats
