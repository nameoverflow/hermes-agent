"""Tests for tools/send_attachment_tool.py."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from gateway.config import Platform
from gateway.session_context import clear_session_vars, set_session_vars
from tools.send_attachment_tool import send_attachment_tool


def _run_async_immediately(coro):
    return asyncio.run(coro)


def _make_config(platform=Platform.DISCORD):
    pconfig = SimpleNamespace(enabled=True, token="***", extra={})
    return SimpleNamespace(
        platforms={platform: pconfig},
        get_home_channel=lambda _platform: None,
    ), pconfig


class TestSendAttachmentTool:
    def test_sends_local_file_to_current_gateway_chat(self, tmp_path):
        image_path = tmp_path / "chart.png"
        image_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
        config, pconfig = _make_config(Platform.DISCORD)
        tokens = set_session_vars(
            platform="discord",
            chat_id="12345",
            thread_id="67890",
            user_id="user-1",
            session_key="discord:12345:67890",
        )
        try:
            with patch("gateway.config.load_gateway_config", return_value=config), \
                 patch("tools.interrupt.is_interrupted", return_value=False), \
                 patch("model_tools._run_async", side_effect=_run_async_immediately), \
                 patch("tools.send_message_tool._send_to_platform", new=AsyncMock(return_value={"success": True, "message_id": "m1"})) as send_mock, \
                 patch("gateway.mirror.mirror_to_session", return_value=True) as mirror_mock:
                result = json.loads(send_attachment_tool({"path": str(image_path), "caption": "Here is the chart"}))
        finally:
            clear_session_vars(tokens)

        assert result["success"] is True
        assert result["attachment"] == str(image_path)
        assert result["target"] == "discord:12345:67890"
        assert result["mirrored"] is True
        send_mock.assert_awaited_once_with(
            Platform.DISCORD,
            pconfig,
            "12345",
            "Here is the chart",
            thread_id="67890",
            media_files=[(str(image_path), False)],
            force_document=False,
        )
        mirror_mock.assert_called_once_with(
            "discord",
            "12345",
            "Here is the chart",
            source_label="discord",
            thread_id="67890",
            user_id="user-1",
        )

    def test_voice_mode_sets_voice_flag(self, tmp_path):
        voice_path = tmp_path / "voice.ogg"
        voice_path.write_bytes(b"OggS" + b"\x00" * 32)
        config, pconfig = _make_config(Platform.TELEGRAM)

        with patch("gateway.config.load_gateway_config", return_value=config), \
             patch("tools.interrupt.is_interrupted", return_value=False), \
             patch("model_tools._run_async", side_effect=_run_async_immediately), \
             patch("tools.send_message_tool._send_to_platform", new=AsyncMock(return_value={"success": True})) as send_mock, \
             patch("gateway.mirror.mirror_to_session", return_value=False):
            result = json.loads(send_attachment_tool({
                "path": str(voice_path),
                "target": "telegram:-1001:55",
                "mode": "voice",
            }))

        assert result["success"] is True
        send_mock.assert_awaited_once_with(
            Platform.TELEGRAM,
            pconfig,
            "-1001",
            "",
            thread_id="55",
            media_files=[(str(voice_path), True)],
            force_document=False,
        )

    def test_document_mode_forces_document_delivery(self, tmp_path):
        image_path = tmp_path / "infographic.jpg"
        image_path.write_bytes(b"\xff\xd8\xff" + b"\x00" * 32)
        config, pconfig = _make_config(Platform.FEISHU)

        with patch("gateway.config.load_gateway_config", return_value=config), \
             patch("tools.interrupt.is_interrupted", return_value=False), \
             patch("model_tools._run_async", side_effect=_run_async_immediately), \
             patch("tools.send_message_tool._send_to_platform", new=AsyncMock(return_value={"success": True})) as send_mock, \
             patch("gateway.mirror.mirror_to_session", return_value=False):
            result = json.loads(send_attachment_tool({
                "path": str(image_path),
                "target": "feishu:oc_abc:thread_xyz",
                "mode": "document",
            }))

        assert result["success"] is True
        send_mock.assert_awaited_once_with(
            Platform.FEISHU,
            pconfig,
            "oc_abc",
            "",
            thread_id="thread_xyz",
            media_files=[(str(image_path), False)],
            force_document=True,
        )

    def test_requires_current_chat_or_explicit_target(self, tmp_path):
        doc_path = tmp_path / "report.txt"
        doc_path.write_text("report")

        result = json.loads(send_attachment_tool({"path": str(doc_path)}))

        assert "error" in result
        assert "No current gateway conversation" in result["error"]

    def test_rejects_root_and_missing_paths(self):
        root_result = json.loads(send_attachment_tool({"path": "/"}))
        missing_result = json.loads(send_attachment_tool({"path": "/tmp/does-not-exist-hermes-attachment"}))

        assert root_result["success"] is False
        assert "non-root" in root_result["error"]
        assert missing_result["success"] is False
        assert "not found" in missing_result["error"]
