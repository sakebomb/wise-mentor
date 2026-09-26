"""Pictures for Wise Mentor when someone asks for one.

The image bytes are attached to the Discord reply and then dropped.
Nothing is written to disk or sqlite.
"""

from __future__ import annotations

from base64 import b64decode
import logging
import re
from typing import Any

import httpx

log = logging.getLogger(__name__)

DEFAULT_IMAGE_MAX_BYTES = 8_000_000
DEFAULT_IMAGE_MODEL = "grok-imagine-image-2.0"

IMAGE_SYSTEM_NOTE = (
    "If he asks you to draw or show a picture of something ordinary, "
    "a picture is attached to your reply. Mention it in one sentence. "
    "If he asks for a sexual or nude picture, refuse plainly and offer a different picture. "
    "No picture is attached for a refusal. Don't claim you made a picture when he didn't ask."
)

_PREFIX = re.compile(r"^\s*<@!?\d+>:\s*")
_REQUEST = re.compile(
    r"^(?:(?:hey|hi|ok|okay|yo|so|um|please)[,!\s]+)*"
    r"(?:(?:can|could|would|will)\s+you\s+)?"
    r"(?:please[,!\s]+)*"
    r"(?:"
    r"(?:draw|sketch|paint|illustrate)\s+(?P<draw>.+)"
    r"|(?:make|create|generate|show)\s+(?:me\s+)?(?:a|an)\s+"
    r"(?:picture|image|drawing|illustration|photo|pic)\s+(?:of\s+)?(?P<made>.+)"
    r"|(?:a\s+)?(?:picture|image|photo|drawing|illustration)\s+of\s+(?P<of>.+)"
    r"|show\s+me\s+what\s+(?P<look>.+?)\s+looks?\s+like.*"
    r")\s*$",
    re.IGNORECASE | re.DOTALL,
)
_REFUSED = re.compile(
    r"\b(?:nude|naked|nudity|nsfw|porn|porno|hentai|erotic|erotica|sex|sexual|"
    r"boobs?|tits|penis|vagina|dick|cock|nipples?|loli|underage|sexy)\b",
    re.IGNORECASE,
)
_REFUSED_PHRASES = (
    "no clothes",
    "without clothes",
    "without any clothes",
    "undressed",
    "hot girl",
    "hot woman",
    "hot chick",
)
_EMPTY_PROMPTS = {"me", "something", "anything", "it", "that"}


def image_prompt_from_text(text: str) -> str | None:
    """The picture he asked for, or None when the message is not an image ask."""
    body = _PREFIX.sub("", text or "").strip()
    if not body:
        return None
    match = _REQUEST.match(body)
    if not match:
        return None
    raw = next(group for group in match.groups() if group)
    prompt = _normalize(raw)
    return prompt or None


def refuses_image(prompt: str) -> bool:
    """Sexual and nude asks are not sent to the image service."""
    text = (prompt or "").lower().replace("naked mole rat", "mole rat")
    if any(phrase in text for phrase in _REFUSED_PHRASES):
        return True
    return bool(_REFUSED.search(text))


def image_filename(image: bytes) -> str:
    if image.startswith(b"\x89PNG"):
        return "wise-mentor.png"
    if image.startswith(b"\xff\xd8"):
        return "wise-mentor.jpg"
    if len(image) >= 12 and image.startswith(b"RIFF") and image[8:12] == b"WEBP":
        return "wise-mentor.webp"
    return "wise-mentor.png"


def _normalize(prompt: str) -> str:
    text = " ".join(prompt.split()).strip(" \t\"'`“”")
    text = re.sub(r"[.!?]+$", "", text).strip()
    text = re.sub(r"\s+please$", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"^me\s+(?=(?:a|an|the|some)\b)", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(
        r"^(?:a\s+)?(?:picture|image|drawing|illustration|photo|pic)\s+of\s+",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()
    if len(text) < 3 or len(text) > 1000 or text.lower() in _EMPTY_PROMPTS:
        return ""
    return text


def _images_url(base_url: str) -> str:
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        return f"{root}/images/generations"
    return f"{root}/v1/images/generations"


class ImageGenerator:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        client: httpx.AsyncClient,
        api_key: str | None = None,
        max_bytes: int = DEFAULT_IMAGE_MAX_BYTES,
        timeout: float = 90.0,
    ) -> None:
        if not base_url or not model:
            raise ValueError("image generation needs a base_url and a model")
        self._url = _images_url(base_url)
        self._model = model
        self._api_key = api_key or None
        self._client = client
        self.max_bytes = max_bytes
        self._timeout = timeout

    async def generate(self, prompt: str) -> bytes:
        """One image for an ordinary ask. Bytes are not stored. Refusals return empty."""
        cleaned = (prompt or "").strip()
        if not cleaned or refuses_image(cleaned):
            return b""
        headers = {"Accept": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        try:
            response = await self._client.post(
                self._url,
                headers=headers,
                json={
                    "model": self._model,
                    "prompt": cleaned,
                    "n": 1,
                    "response_format": "b64_json",
                },
                timeout=self._timeout,
            )
            response.raise_for_status()
            return await self._bytes_from_body(response.json())
        except Exception as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            log.warning("picture failed: %s status=%s", type(exc).__name__, status)
            return b""

    async def _bytes_from_body(self, body: Any) -> bytes:
        if not isinstance(body, dict):
            return b""
        rows = body.get("data") or []
        if not rows or not isinstance(rows[0], dict):
            return b""
        row = rows[0]
        if row.get("b64_json"):
            raw = b64decode(row["b64_json"])
        elif row.get("url"):
            fetched = await self._client.get(row["url"], timeout=self._timeout)
            fetched.raise_for_status()
            raw = fetched.content
        else:
            return b""
        if not raw or len(raw) > self.max_bytes:
            return b""
        return raw


def images_from_config(config: dict[str, Any], client: httpx.AsyncClient) -> ImageGenerator | None:
    images = config.get("images") or {}
    base_url = images.get("base_url")
    model = images.get("model")
    api_key = images.get("api_key") or None
    if not base_url or not model or not api_key:
        return None
    return ImageGenerator(
        base_url=str(base_url),
        model=str(model),
        api_key=str(api_key),
        client=client,
        max_bytes=int(images.get("max_bytes") or DEFAULT_IMAGE_MAX_BYTES),
    )
