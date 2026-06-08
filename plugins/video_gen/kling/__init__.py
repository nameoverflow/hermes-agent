"""Kling direct API video generation backend.

This provider intentionally calls Kling's official API directly instead of
routing through FAL.ai.  The unified Hermes ``video_generate`` surface is
mapped to Kling's Omni video endpoint using ``model_name=kling-v3-omni`` and
``mode=pro`` by default.

Supported through the generic tool surface:
- text-to-video: ``prompt`` only
- image-to-video / image-guided: ``image_url`` and optional
  ``reference_image_urls`` become ``image_list`` entries

Credentials:
- ``KLING_TOKEN``: session Bearer token (may include or omit ``Bearer ``)
- or ``KLING_ACCESS_KEY_ID`` + ``KLING_SECRET_ACCESS_KEY``: signed as the
  Kling HS256 JWT expected by the OpenAPI
- or the ClawHub-compatible ``~/.config/kling/.credentials`` INI file
"""

from __future__ import annotations

import configparser
import datetime as _dt
import hmac
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import base64

import httpx

from agent.video_gen_provider import VideoGenProvider, error_response, success_response

logger = logging.getLogger(__name__)

DEFAULT_API_BASE = "https://api-beijing.klingai.com"
GLOBAL_API_BASE = "https://api-singapore.klingai.com"
API_OMNI_VIDEO = "/v1/videos/omni-video"
DEFAULT_MODEL = "kling-v3-omni"
DEFAULT_MODE = "pro"
DEFAULT_DURATION = 5
DEFAULT_ASPECT_RATIO = "16:9"
DEFAULT_TIMEOUT_SECONDS = 360
DEFAULT_POLL_INTERVAL_SECONDS = 10

VALID_ASPECT_RATIOS = {"16:9", "9:16", "1:1"}
VALID_RESOLUTIONS = {"720p", "1080p"}
MAX_DURATION = 15
MIN_DURATION = 3
MAX_REFERENCE_IMAGES = 7

_MODELS: Dict[str, Dict[str, Any]] = {
    DEFAULT_MODEL: {
        "display": "Kling V3 Omni Pro",
        "speed": "~1-5 min",
        "strengths": "Direct Kling API; Omni text-to-video and image-guided video; pro mode (1080P).",
        "price": "Kling account quota",
        "modalities": ["text", "image"],
    },
}


def _storage_root() -> Path:
    raw = (os.getenv("KLING_STORAGE_ROOT") or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return Path.home() / ".config" / "kling"


def _read_credentials_file() -> Tuple[str, str]:
    path = _storage_root() / ".credentials"
    parser = configparser.ConfigParser()
    try:
        parser.read(path, encoding="utf-8")
    except Exception:
        return "", ""
    if not parser.has_section("default"):
        return "", ""
    section = parser["default"]
    ak = (section.get("access_key_id") or section.get("access_key") or "").strip()
    sk = (section.get("secret_access_key") or section.get("secret_key") or "").strip()
    return ak, sk


def _base64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _make_kling_jwt(access_key_id: str, secret_access_key: str) -> str:
    now = int(time.time())
    header = _base64url(json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode())
    payload = _base64url(json.dumps({"iss": access_key_id, "exp": now + 1800, "nbf": now - 5}, separators=(",", ":")).encode())
    signing_input = f"{header}.{payload}".encode("ascii")
    sig = _base64url(hmac.new(secret_access_key.encode("utf-8"), signing_input, hashlib.sha256).digest())
    return f"{header}.{payload}.{sig}"


def _resolve_token() -> str:
    token = (os.getenv("KLING_TOKEN") or "").strip()
    if token:
        if token.lower().startswith("bearer "):
            token = token[7:].strip()
        return token

    ak = (os.getenv("KLING_ACCESS_KEY_ID") or "").strip()
    sk = (os.getenv("KLING_SECRET_ACCESS_KEY") or "").strip()
    if not (ak and sk):
        ak, sk = _read_credentials_file()
    if ak and sk:
        return _make_kling_jwt(ak, sk)
    return ""


def _api_base() -> str:
    raw = (os.getenv("KLING_API_BASE") or "").strip().rstrip("/")
    if raw:
        return raw
    region = (os.getenv("KLING_REGION") or "").strip().lower()
    if region in {"global", "sg", "singapore", "intl", "international"}:
        return GLOBAL_API_BASE
    return DEFAULT_API_BASE


def _headers(token: str, *, content_type: bool = True) -> Dict[str, str]:
    headers = {
        "Authorization": f"Bearer {token}",
        "User-Agent": "hermes-agent/video_gen/kling",
    }
    if content_type:
        headers["Content-Type"] = "application/json"
    return headers


def _parse_response(response: httpx.Response) -> Dict[str, Any]:
    response.raise_for_status()
    body = response.json()
    code = body.get("code")
    if code not in (0, 200, None):
        raise RuntimeError(f"Kling API error code={code}: {body.get('message') or body.get('msg') or 'Unknown error'}")
    data = body.get("data", body)
    if not isinstance(data, dict):
        raise RuntimeError("Kling API returned non-object data")
    return data


def _clamp_duration(duration: Optional[int]) -> int:
    value = int(duration or DEFAULT_DURATION)
    return max(MIN_DURATION, min(MAX_DURATION, value))


def _normalize_aspect_ratio(value: Optional[str]) -> str:
    ratio = (value or DEFAULT_ASPECT_RATIO).strip()
    return ratio if ratio in VALID_ASPECT_RATIOS else DEFAULT_ASPECT_RATIO


def _normalize_model(model: Optional[str]) -> str:
    value = (model or DEFAULT_MODEL).strip().lower()
    # User-facing aliases are accepted defensively but never sent as aliases.
    aliases = {
        "o3": DEFAULT_MODEL,
        "omni3": DEFAULT_MODEL,
        "omni-3": DEFAULT_MODEL,
        "omni-v3": DEFAULT_MODEL,
        "v3-omni": DEFAULT_MODEL,
        "kling-o3": DEFAULT_MODEL,
        "kling-video-o3": DEFAULT_MODEL,
    }
    value = aliases.get(value, value)
    if value != DEFAULT_MODEL:
        raise ValueError(f"Kling provider only supports {DEFAULT_MODEL} through Hermes video_generate, got {model!r}")
    return DEFAULT_MODEL


def _extract_video_url(data: Dict[str, Any]) -> str:
    result = data.get("task_result") or data.get("output") or {}
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        for key in ("url", "video_url"):
            if result.get(key):
                return str(result[key])
        videos = result.get("videos")
        if isinstance(videos, list) and videos:
            first = videos[0] or {}
            if isinstance(first, dict):
                for key in ("url", "video_url"):
                    if first.get(key):
                        return str(first[key])
    return ""


class KlingVideoGenProvider(VideoGenProvider):
    """Kling V3 Omni Pro provider using the official Kling API directly."""

    @property
    def name(self) -> str:
        return "kling"

    @property
    def display_name(self) -> str:
        return "Kling"

    def is_available(self) -> bool:
        return bool(_resolve_token())

    def list_models(self) -> List[Dict[str, Any]]:
        return [{"id": mid, **meta} for mid, meta in _MODELS.items()]

    def default_model(self) -> Optional[str]:
        return DEFAULT_MODEL

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Kling Direct API",
            "badge": "paid · direct",
            "tag": "Calls Kling API directly (no FAL.ai): kling-v3-omni + pro mode",
            # Kling supports two mutually exclusive credential styles: a
            # session Bearer token, or AK/SK credentials that Hermes signs into
            # a short-lived JWT. The generic env_vars flow requires every listed
            # variable, so use a post_setup hook to present an either/or picker.
            "env_vars": [],
            "post_setup": "kling",
        }

    def capabilities(self) -> Dict[str, Any]:
        return {
            "modalities": ["text", "image"],
            "aspect_ratios": sorted(VALID_ASPECT_RATIOS),
            "resolutions": sorted(VALID_RESOLUTIONS),
            "max_duration": MAX_DURATION,
            "min_duration": MIN_DURATION,
            "supports_audio": True,
            "supports_negative_prompt": False,
            "max_reference_images": MAX_REFERENCE_IMAGES,
        }

    def generate(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        image_url: Optional[str] = None,
        reference_image_urls: Optional[List[str]] = None,
        duration: Optional[int] = None,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        resolution: str = "1080p",
        negative_prompt: Optional[str] = None,
        audio: Optional[bool] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        normalized_model = DEFAULT_MODEL
        try:
            normalized_model = _normalize_model(model)
            token = _resolve_token()
            if not token:
                return error_response(
                    error=(
                        "No Kling credentials found. Set KLING_TOKEN, or set "
                        "KLING_ACCESS_KEY_ID + KLING_SECRET_ACCESS_KEY, or import them into "
                        "~/.config/kling/.credentials."
                    ),
                    error_type="auth_required",
                    provider=self.name,
                    model=normalized_model,
                    prompt=prompt,
                    aspect_ratio=aspect_ratio,
                )

            prompt = (prompt or "").strip()
            if not prompt:
                return error_response(
                    error="prompt is required for Kling video generation",
                    error_type="missing_prompt",
                    provider=self.name,
                    model=normalized_model,
                    prompt=prompt,
                    aspect_ratio=aspect_ratio,
                )

            refs = []
            first = (image_url or "").strip()
            if first:
                refs.append(first)
            refs.extend([u.strip() for u in (reference_image_urls or []) if (u or "").strip()])
            if len(refs) > MAX_REFERENCE_IMAGES:
                return error_response(
                    error=f"Kling Omni video supports at most {MAX_REFERENCE_IMAGES} image references through video_generate",
                    error_type="too_many_references",
                    provider=self.name,
                    model=normalized_model,
                    prompt=prompt,
                    aspect_ratio=aspect_ratio,
                )

            clamped_duration = _clamp_duration(duration)
            normalized_aspect = _normalize_aspect_ratio(aspect_ratio)
            sound = "on" if audio is True else "off"
            payload: Dict[str, Any] = {
                "model_name": normalized_model,
                "duration": str(clamped_duration),
                "mode": DEFAULT_MODE,
                "sound": sound,
                "callback_url": "",
                "multi_shot": False,
                "prompt": prompt,
                "aspect_ratio": normalized_aspect,
            }
            if refs:
                payload["image_list"] = [{"image_url": ref} for ref in refs]

            data = self._submit_and_poll(token, payload)
            video_url = _extract_video_url(data)
            if not video_url:
                return error_response(
                    error="Kling task succeeded but no video URL was returned",
                    error_type="missing_output",
                    provider=self.name,
                    model=normalized_model,
                    prompt=prompt,
                    aspect_ratio=normalized_aspect,
                )
            return success_response(
                video=video_url,
                model=normalized_model,
                prompt=prompt,
                modality="image" if refs else "text",
                aspect_ratio=normalized_aspect,
                duration=clamped_duration,
                provider=self.name,
                extra={"mode": DEFAULT_MODE, "task_id": data.get("task_id", "")},
            )
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code if exc.response is not None else "unknown"
            body = exc.response.text[:500] if exc.response is not None else ""
            logger.warning("Kling HTTP error: %s %s", status, body)
            return error_response(
                error=f"Kling API HTTP error {status}: {body}",
                error_type="api_error",
                provider=self.name,
                model=normalized_model,
                prompt=prompt,
                aspect_ratio=aspect_ratio,
            )
        except Exception as exc:
            logger.warning("Kling video generation failed: %s", exc, exc_info=True)
            return error_response(
                error=f"Kling video generation failed: {exc}",
                error_type="api_error",
                provider=self.name,
                model=normalized_model,
                prompt=prompt,
                aspect_ratio=aspect_ratio,
            )

    def _submit_and_poll(self, token: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        base = _api_base()
        with httpx.Client(timeout=60) as client:
            submit = client.post(f"{base}{API_OMNI_VIDEO}", headers=_headers(token), json=payload)
            data = _parse_response(submit)
            task_id = data.get("task_id")
            if not task_id:
                raise RuntimeError("Kling API did not return task_id")

            deadline = time.monotonic() + DEFAULT_TIMEOUT_SECONDS
            last_status = str(data.get("task_status") or "submitted")
            while time.monotonic() < deadline:
                if last_status == "succeed":
                    return data
                if last_status == "failed":
                    raise RuntimeError(data.get("task_status_msg") or "Kling task failed")
                time.sleep(DEFAULT_POLL_INTERVAL_SECONDS)
                poll = client.get(f"{base}{API_OMNI_VIDEO}/{task_id}", headers=_headers(token, content_type=False))
                data = _parse_response(poll)
                last_status = str(data.get("task_status") or "")
            raise TimeoutError(f"Kling task timed out after {DEFAULT_TIMEOUT_SECONDS}s (last status: {last_status})")


def register(ctx):
    ctx.register_video_gen_provider(KlingVideoGenProvider())
