"""Explicit attachment delivery tool for gateway conversations.

This replaces the old model-facing pattern of embedding magic ``MEDIA:``
strings in final responses.  The legacy parser remains for backwards
compatibility and for older tool outputs, but new agent-facing behavior should
send attachments through this structured tool.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from agent.redact import redact_sensitive_text
from tools.registry import registry, tool_error


SEND_ATTACHMENT_SCHEMA = {
    "name": "send_attachment",
    "description": (
        "Send a local file as a native attachment on the current messaging "
        "conversation, or to an explicit messaging target. Use this when you "
        "have generated or found a file that the user should receive. Do NOT "
        "put MEDIA: markers in your final response for new attachment delivery; "
        "call this tool with the file path instead, then mention briefly that it "
        "was sent."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute local path, or ~/path, to the file to attach.",
            },
            "caption": {
                "type": "string",
                "description": "Optional short text sent with/before the attachment.",
            },
            "mode": {
                "type": "string",
                "enum": ["auto", "document", "voice"],
                "description": (
                    "Delivery hint. 'auto' sends images/videos/audio using the platform's native media path when possible; "
                    "'document' forces image files to be sent as files/documents; 'voice' sends compatible OGG/Opus audio as a voice bubble on supported platforms."
                ),
            },
            "target": {
                "type": "string",
                "description": (
                    "Optional explicit target using send_message syntax, e.g. 'discord:channel_id:thread_id', "
                    "'telegram:-100123:45', or 'feishu:chat_id:thread_id'. Omit this in a gateway conversation to send to the current chat/thread."
                ),
            },
        },
        "required": ["path"],
    },
}


def _error(message: str) -> dict[str, str]:
    return {"error": redact_sensitive_text(str(message))}


def _normalize_file_path(raw_path: Any) -> str | None:
    path = str(raw_path or "").strip().strip("`\"'")
    if not path:
        return None
    expanded = os.path.abspath(os.path.expanduser(path))
    if expanded == os.path.sep:
        return None
    return expanded


def _current_gateway_target() -> tuple[str, str, str | None] | tuple[None, None, None]:
    from gateway.session_context import get_session_env

    platform = (get_session_env("HERMES_SESSION_PLATFORM", "") or "").strip().lower()
    chat_id = (get_session_env("HERMES_SESSION_CHAT_ID", "") or "").strip()
    thread_id = (get_session_env("HERMES_SESSION_THREAD_ID", "") or "").strip() or None
    if not platform or platform == "local" or not chat_id:
        return None, None, None
    return platform, chat_id, thread_id


def _resolve_target(args: dict[str, Any]) -> tuple[Any, Any, str, str | None, bool] | dict[str, str]:
    """Resolve tool args into platform enum, platform config, chat/thread IDs.

    Returns ``(platform, pconfig, chat_id, thread_id, used_home_channel)`` or an
    error dict.
    """
    try:
        from gateway.config import HomeChannel, Platform, PlatformConfig, load_gateway_config
        from tools.send_message_tool import _parse_target_ref
    except Exception as exc:
        return _error(f"Failed to load gateway messaging support: {exc}")

    target = str(args.get("target") or "").strip()
    used_home_channel = False

    if target:
        parts = target.split(":", 1)
        platform_name = parts[0].strip().lower()
        target_ref = parts[1].strip() if len(parts) > 1 else None
        chat_id = None
        thread_id = None
        is_explicit = False
        if target_ref:
            chat_id, thread_id, is_explicit = _parse_target_ref(platform_name, target_ref)
        if target_ref and not is_explicit:
            try:
                from gateway.channel_directory import resolve_channel_name

                resolved = resolve_channel_name(platform_name, target_ref)
                if resolved:
                    chat_id, thread_id, _ = _parse_target_ref(platform_name, resolved)
                else:
                    return _error(
                        f"Could not resolve '{target_ref}' on {platform_name}. "
                        "Use send_message(action='list') to see available targets."
                    )
            except Exception:
                return _error(
                    f"Could not resolve '{target_ref}' on {platform_name}. "
                    "Try using a numeric channel ID instead."
                )
    else:
        platform_name, chat_id, thread_id = _current_gateway_target()
        if not platform_name or not chat_id:
            return _error(
                "No current gateway conversation is available. Provide an explicit 'target' "
                "such as 'discord:CHANNEL_ID' or run this from a messaging gateway chat."
            )

    try:
        platform = Platform(platform_name)
    except (ValueError, KeyError):
        return _error(f"Unknown platform: {platform_name}")

    try:
        config = load_gateway_config()
    except Exception as exc:
        return _error(f"Failed to load gateway config: {exc}")

    pconfig = config.platforms.get(platform)
    if not pconfig or not pconfig.enabled:
        # Mirror send_message's env-only Weixin fallback.
        if platform_name == "weixin":
            wx_token = os.getenv("WEIXIN_TOKEN", "").strip()
            wx_account = os.getenv("WEIXIN_ACCOUNT_ID", "").strip()
            if wx_token and wx_account:
                pconfig = PlatformConfig(
                    enabled=True,
                    token=wx_token,
                    extra={
                        "account_id": wx_account,
                        "base_url": os.getenv("WEIXIN_BASE_URL", "").strip(),
                        "cdn_base_url": os.getenv("WEIXIN_CDN_BASE_URL", "").strip(),
                    },
                )
            else:
                return _error(f"Platform '{platform_name}' is not configured.")
        else:
            return _error(f"Platform '{platform_name}' is not configured.")

    if not chat_id:
        home = config.get_home_channel(platform)
        if not home and platform_name == "weixin":
            wx_home = os.getenv("WEIXIN_HOME_CHANNEL", "").strip()
            if wx_home:
                home = HomeChannel(platform=platform, chat_id=wx_home, name="Weixin Home")
        if home:
            chat_id = home.chat_id
            used_home_channel = True
        else:
            return _error(
                f"No chat target available for {platform_name}. Provide an explicit target or configure a home channel."
            )

    return platform, pconfig, str(chat_id), thread_id, used_home_channel


def send_attachment_tool(args: dict[str, Any], **kw) -> str:
    """Send a local file as a native attachment."""
    file_path = _normalize_file_path(args.get("path"))
    if not file_path:
        return tool_error("A non-root local file path is required", success=False)
    if not os.path.isfile(file_path):
        return tool_error(f"Attachment file not found or not a regular file: {file_path}", success=False)

    mode = str(args.get("mode") or "auto").strip().lower()
    if mode not in {"auto", "document", "voice"}:
        return tool_error("mode must be one of: auto, document, voice", success=False)

    resolved = _resolve_target(args)
    if isinstance(resolved, dict):
        return json.dumps(resolved, ensure_ascii=False)
    platform, pconfig, chat_id, thread_id, used_home_channel = resolved

    from tools.interrupt import is_interrupted

    if is_interrupted():
        return tool_error("Interrupted", success=False)

    caption = str(args.get("caption") or "")
    force_document = mode == "document"
    is_voice = mode == "voice"

    try:
        from model_tools import _run_async
        from tools.send_message_tool import _describe_media_for_mirror, _send_to_platform

        result = _run_async(
            _send_to_platform(
                platform,
                pconfig,
                chat_id,
                caption,
                thread_id=thread_id,
                media_files=[(file_path, is_voice)],
                force_document=force_document,
            )
        )
        if isinstance(result, dict) and result.get("success"):
            result.setdefault("attachment", file_path)
            result.setdefault("target", f"{platform.value}:{chat_id}" + (f":{thread_id}" if thread_id else ""))
            if used_home_channel:
                result["note"] = f"Sent to {platform.value} home channel (chat_id: {chat_id})"
            try:
                from gateway.mirror import mirror_to_session
                from gateway.session_context import get_session_env

                mirror_text = caption.strip() or _describe_media_for_mirror([(file_path, is_voice)])
                if mirror_text and mirror_to_session(
                    platform.value,
                    chat_id,
                    mirror_text,
                    source_label=get_session_env("HERMES_SESSION_PLATFORM", "cli"),
                    thread_id=thread_id,
                    user_id=get_session_env("HERMES_SESSION_USER_ID", "") or None,
                ):
                    result["mirrored"] = True
            except Exception:
                pass
        elif isinstance(result, dict) and "error" in result:
            result["error"] = redact_sensitive_text(str(result["error"]))
        return json.dumps(result, ensure_ascii=False)
    except Exception as exc:
        return json.dumps(_error(f"Attachment send failed: {exc}"), ensure_ascii=False)


# Use the same availability gate as send_message: available in gateway sessions,
# kanban workers, or when a gateway service is running for explicit targets.
def _check_send_attachment() -> bool:
    try:
        from tools.send_message_tool import _check_send_message

        return bool(_check_send_message())
    except Exception:
        return False


registry.register(
    name="send_attachment",
    toolset="messaging",
    schema=SEND_ATTACHMENT_SCHEMA,
    handler=send_attachment_tool,
    check_fn=_check_send_attachment,
    emoji="📎",
)
