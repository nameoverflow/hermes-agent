"""Tests for the direct Kling video generation plugin."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import pytest

from agent import video_gen_registry


@pytest.fixture(autouse=True)
def _reset_registry():
    video_gen_registry._reset_for_tests()
    yield
    video_gen_registry._reset_for_tests()


def test_kling_provider_registers():
    from plugins.video_gen.kling import DEFAULT_MODEL, KlingVideoGenProvider

    provider = KlingVideoGenProvider()
    video_gen_registry.register_provider(provider)

    assert video_gen_registry.get_provider("kling") is provider
    assert provider.display_name == "Kling"
    assert provider.default_model() == DEFAULT_MODEL


def test_kling_capabilities_direct_omni():
    from plugins.video_gen.kling import KlingVideoGenProvider

    caps = KlingVideoGenProvider().capabilities()
    assert caps["modalities"] == ["text", "image"]
    assert caps["supports_audio"] is True
    assert caps["max_duration"] == 15
    assert caps["min_duration"] == 3
    assert set(caps["aspect_ratios"]) == {"16:9", "9:16", "1:1"}


def test_kling_setup_schema_uses_cli_post_setup_for_alternative_credentials():
    from plugins.video_gen.kling import KlingVideoGenProvider

    schema = KlingVideoGenProvider().get_setup_schema()
    assert schema["env_vars"] == []
    assert schema["post_setup"] == "kling"


def test_kling_unavailable_without_credentials(monkeypatch, tmp_path):
    from plugins.video_gen.kling import KlingVideoGenProvider

    monkeypatch.delenv("KLING_TOKEN", raising=False)
    monkeypatch.delenv("KLING_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("KLING_SECRET_ACCESS_KEY", raising=False)
    monkeypatch.setenv("KLING_STORAGE_ROOT", str(tmp_path))

    assert KlingVideoGenProvider().is_available() is False
    result = KlingVideoGenProvider().generate("a cat")
    assert result["success"] is False
    assert result["error_type"] == "auth_required"


def test_kling_accepts_ak_sk_env(monkeypatch, tmp_path):
    from plugins.video_gen.kling import KlingVideoGenProvider, _resolve_token

    monkeypatch.delenv("KLING_TOKEN", raising=False)
    monkeypatch.setenv("KLING_ACCESS_KEY_ID", "test-ak")
    monkeypatch.setenv("KLING_SECRET_ACCESS_KEY", "test-sk")
    monkeypatch.setenv("KLING_STORAGE_ROOT", str(tmp_path))

    assert KlingVideoGenProvider().is_available() is True
    token = _resolve_token()
    assert token.count(".") == 2
    assert "test-sk" not in token


class _FakeResponse:
    def __init__(self, status: int = 200, payload: Optional[Dict[str, Any]] = None):
        self.status_code = status
        self._payload = payload or {}
        self.text = json.dumps(self._payload)

    def raise_for_status(self):
        if self.status_code >= 400:
            import httpx
            raise httpx.HTTPStatusError("err", request=None, response=self)  # type: ignore[arg-type]

    def json(self):
        return self._payload


class _FakeClient:
    def __init__(self):
        self.posts: List[Dict[str, Any]] = []
        self.gets: List[Dict[str, Any]] = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def post(self, url, headers=None, json=None):
        self.posts.append({"url": url, "headers": headers, "json": json})
        return _FakeResponse(200, {"code": 0, "data": {"task_id": "task-123", "task_status": "submitted"}})

    def get(self, url, headers=None):
        self.gets.append({"url": url, "headers": headers})
        return _FakeResponse(200, {
            "code": 0,
            "data": {
                "task_id": "task-123",
                "task_status": "succeed",
                "task_result": {"videos": [{"url": "https://kling.example/out.mp4"}]},
            },
        })


@pytest.fixture
def kling_provider(monkeypatch):
    monkeypatch.setenv("KLING_TOKEN", "test-token")
    monkeypatch.delenv("KLING_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("KLING_SECRET_ACCESS_KEY", raising=False)

    import plugins.video_gen.kling as kling_plugin

    captured: Dict[str, _FakeClient] = {}

    def _client_factory(*args, **kwargs):
        captured["client"] = _FakeClient()
        return captured["client"]

    monkeypatch.setattr(kling_plugin.httpx, "Client", _client_factory)
    monkeypatch.setattr(kling_plugin.time, "sleep", lambda *_: None)
    provider = kling_plugin.KlingVideoGenProvider()
    return provider, captured


def test_text_to_video_hits_omni_endpoint(kling_provider):
    provider, captured = kling_provider
    result = provider.generate("a cat running", duration=5, aspect_ratio="9:16")

    assert result["success"] is True
    post = captured["client"].posts[-1]
    assert post["url"].endswith("/v1/videos/omni-video")
    assert post["json"]["model_name"] == "kling-v3-omni"
    assert post["json"]["mode"] == "pro"
    assert post["json"]["prompt"] == "a cat running"
    assert post["json"]["aspect_ratio"] == "9:16"
    assert "image_list" not in post["json"]
    assert result["video"] == "https://kling.example/out.mp4"
    assert result["modality"] == "text"


def test_image_to_video_uses_image_list(kling_provider):
    provider, captured = kling_provider
    result = provider.generate(
        "animate this",
        image_url="https://example.com/cat.png",
        reference_image_urls=["https://example.com/style.png"],
        audio=True,
    )

    payload = captured["client"].posts[-1]["json"]
    assert payload["image_list"] == [
        {"image_url": "https://example.com/cat.png"},
        {"image_url": "https://example.com/style.png"},
    ]
    assert payload["sound"] == "on"
    assert result["modality"] == "image"


def test_kling_rejects_non_omni_model_without_api_call(kling_provider):
    provider, captured = kling_provider
    result = provider.generate("x", model="seedance-2.0-fast")

    assert result["success"] is False
    assert result["error_type"] == "api_error"
    assert "only supports kling-v3-omni" in result["error"]
    assert "client" not in captured
