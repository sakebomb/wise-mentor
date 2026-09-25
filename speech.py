"""Voice-note transcription for Wise Mentor.

Audio bytes go to an OpenAI-compatible POST /v1/audio/transcriptions
server and are then dropped. Only the transcript text is kept.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any

import httpx

log = logging.getLogger(__name__)

AUDIO_EXTENSIONS = {".ogg", ".opus", ".mp3", ".wav", ".m4a", ".webm", ".flac"}
DEFAULT_MAX_BYTES = 10_000_000
DEFAULT_MAX_CLIPS = 3
_HEARD_FIELD_LIMIT = 1024


class SpeechError(Exception):
    pass


class AudioTooLarge(SpeechError):
    pass


@dataclass(frozen=True)
class PendingClip:
    filename: str
    content_type: str | None
    size: int | None
    fetch: Callable[[], Awaitable[bytes]]


@dataclass
class VoiceNoteResult:
    transcript: str = ""
    clip_count: int = 0
    unconfigured: bool = False
    too_large: int = 0
    failed: int = 0
    empty: int = 0
    dropped_extra: int = 0

    @property
    def had_audio(self) -> bool:
        return self.clip_count > 0 or self.dropped_extra > 0

    def warnings(self) -> list[str]:
        notes = []
        if self.unconfigured:
            notes.append("⚠️ Voice notes aren't set up")
        if self.too_large:
            notes.append("⚠️ Voice note too long")
        if self.failed or (self.empty and not self.transcript):
            notes.append("⚠️ Couldn't make out that voice note")
        if self.dropped_extra:
            notes.append(f"⚠️ Only using the first {self.clip_count} voice notes")
        return notes

    def fallback_reply(self) -> str:
        if self.unconfigured:
            return "I can't listen to voice notes yet. Type it and I'll answer."
        if self.too_large and not self.failed and not self.empty:
            return "That voice note is too long. Send a shorter one, or type it."
        return "I couldn't make that out. Try again, or type it."


def is_audio_attachment(content_type: str | None, filename: str | None = None) -> bool:
    if content_type:
        media = content_type.split(";", 1)[0].strip().lower()
        if media.startswith("audio/"):
            return True
    suffix = Path(filename or "").suffix.lower()
    return suffix in AUDIO_EXTENSIONS


def merge_message_text(*parts: str) -> str:
    return "\n".join(part.strip() for part in parts if part and part.strip())


def heard_field(transcript: str, limit: int = _HEARD_FIELD_LIMIT) -> dict[str, Any]:
    value = " ".join(transcript.split())
    if len(value) > limit:
        value = value[: limit - 3] + "..."
    return {"name": "Heard", "value": value, "inline": False}


def format_heard_prefix(transcript: str) -> str:
    one_line = " ".join(transcript.split())
    return f"Heard: {one_line}\n\n"


def strip_heard_prefix(text: str) -> str:
    if not text.startswith("Heard: "):
        return text
    _head, sep, tail = text.partition("\n\n")
    return tail if sep else text


def should_answer_with_model(*, text: str, image_count: int, had_audio: bool) -> bool:
    if text.strip() or image_count:
        return True
    return not had_audio


def should_retry_voice(*, had_audio: bool, transcript: str, retryable: bool, has_other_text: bool, has_images: bool) -> bool:
    """Keep a failed audio-only note uncached so a later turn can transcribe it."""
    return had_audio and not transcript.strip() and retryable and not has_other_text and not has_images


def _transcriptions_url(base_url: str) -> str:
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        return f"{root}/audio/transcriptions"
    return f"{root}/v1/audio/transcriptions"


class SpeechToText:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        client: httpx.AsyncClient,
        api_key: str | None = None,
        max_bytes: int = DEFAULT_MAX_BYTES,
        timeout: float = 60.0,
    ) -> None:
        if not base_url or not model:
            raise ValueError("speech-to-text needs a base_url and a model")
        self._url = _transcriptions_url(base_url)
        self._model = model
        self._api_key = api_key or None
        self._client = client
        self.max_bytes = max_bytes
        self._timeout = timeout

    async def transcribe(self, *, filename: str, content_type: str | None, audio: bytes) -> str:
        if not audio:
            return ""
        if len(audio) > self.max_bytes:
            raise AudioTooLarge(f"audio is {len(audio)} bytes")
        headers = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        media = (content_type or "application/octet-stream").split(";", 1)[0].strip()
        response = await self._client.post(
            self._url,
            headers=headers,
            data={"model": self._model, "response_format": "json"},
            files={"file": (filename or "voice.ogg", audio, media)},
            timeout=self._timeout,
        )
        response.raise_for_status()
        body = response.json()
        if not isinstance(body, dict):
            return ""
        return str(body.get("text") or "").strip()


def from_config(config: dict[str, Any], client: httpx.AsyncClient) -> SpeechToText | None:
    stt = config.get("stt") or {}
    base_url = stt.get("base_url")
    model = stt.get("model")
    if not base_url or not model:
        return None
    return SpeechToText(
        base_url=str(base_url),
        model=str(model),
        api_key=stt.get("api_key") or None,
        client=client,
        max_bytes=int(stt.get("max_bytes") or DEFAULT_MAX_BYTES),
    )


async def _one_clip(stt: SpeechToText, clip: PendingClip, max_bytes: int) -> tuple[str, str]:
    if clip.size is not None and clip.size > max_bytes:
        return "too_large", ""
    audio = b""
    try:
        audio = await clip.fetch()
        if len(audio) > max_bytes:
            return "too_large", ""
        text = await stt.transcribe(filename=clip.filename, content_type=clip.content_type, audio=audio)
        return ("ok", text) if text else ("empty", "")
    except AudioTooLarge:
        return "too_large", ""
    except Exception as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        log.warning("voice transcription failed: %s status=%s", type(exc).__name__, status)
        return "failed", ""
    finally:
        audio = b""


async def resolve_voice_note(
    *,
    message_id: int,
    user_id: int,
    clips: list[PendingClip],
    stt: SpeechToText | None,
    load_saved: Callable[[int], str | None],
    store_text: Callable[[int, int, str], None],
    max_bytes: int,
    max_clips: int,
) -> VoiceNoteResult:
    if not clips:
        return VoiceNoteResult()
    saved = (load_saved(message_id) or "").strip()
    if saved:
        return VoiceNoteResult(transcript=saved, clip_count=len(clips))
    if stt is None:
        return VoiceNoteResult(clip_count=len(clips), unconfigured=True)
    chosen = clips[:max_clips]
    texts, counts = [], {"too_large": 0, "failed": 0, "empty": 0}
    for clip in chosen:
        status, text = await _one_clip(stt, clip, max_bytes)
        if status == "ok":
            texts.append(text)
        elif status in counts:
            counts[status] += 1
    transcript = "\n".join(texts)
    if transcript:
        store_text(message_id, user_id, transcript)
    return VoiceNoteResult(
        transcript=transcript,
        clip_count=len(chosen),
        too_large=counts["too_large"],
        failed=counts["failed"],
        empty=counts["empty"],
        dropped_extra=len(clips) - len(chosen),
    )
