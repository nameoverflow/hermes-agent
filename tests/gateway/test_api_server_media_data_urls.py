"""API response text is never interpreted as an attachment channel."""

import pytest

pytest.importorskip("aiohttp")

from gateway.platforms.api_server import _resolve_media_to_data_urls  # noqa: E402


def test_response_text_is_left_unchanged():
    marker = "MEDIA" + ":/tmp/example.png"
    text = f"Here you go: {marker}"
    assert _resolve_media_to_data_urls(text) == text


def test_ordinary_absolute_path_is_left_unchanged():
    text = "Saved at /tmp/example.png"
    assert _resolve_media_to_data_urls(text) == text
