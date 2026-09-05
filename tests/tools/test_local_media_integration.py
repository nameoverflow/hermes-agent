"""Exercise local media contracts through the stable release's dispatchers."""

import json
from pathlib import Path


def test_attachment_uses_real_discord_dispatch_and_caption_routing(monkeypatch, tmp_path):
    from gateway.platform_registry import platform_registry
    from gateway.session_context import clear_session_vars, set_session_vars
    from hermes_cli.plugins import discover_plugins
    from tools.send_attachment_tool import send_attachment_tool

    # Real config loading, target resolution, coroutine bridge and dispatcher;
    # replace only the transport boundary to keep the test offline.
    (tmp_path / "config.yaml").write_text(
        "platforms:\n  discord:\n    enabled: true\n    token: test-token\n"
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    discover_plugins()
    entry = platform_registry.get("discord")
    assert entry is not None
    calls = []

    async def transport(config, chat_id, message, **kwargs):
        calls.append((chat_id, message, kwargs))
        for path, _voice in kwargs.get("media_files", []):
            assert Path(path).read_bytes() == b"attachment-payload"
        return {"success": True, "message_id": "sent-file"}

    monkeypatch.setattr(entry, "standalone_sender_fn", transport)
    monkeypatch.setattr("gateway.mirror.mirror_to_session", lambda *a, **kw: False)
    path = tmp_path / "report.png"
    path.write_bytes(b"attachment-payload")
    tokens = set_session_vars(
        platform="discord", chat_id="12345", thread_id="67890",
        user_id="test-user", session_key="discord:12345:67890",
    )
    try:
        result = json.loads(send_attachment_tool({"path": str(path), "caption": "Report"}))
    finally:
        clear_session_vars(tokens)

    assert result["success"] is True
    assert result["target"] == "discord:12345:67890"
    assert calls == [("12345", "", {
        "thread_id": "67890", "media_files": [(str(path), False)], "caption": "Report",
    })]
