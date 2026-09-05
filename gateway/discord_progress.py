"""Compact Discord progress: ten recent calls and one subdued commentary."""

import re
from collections import deque


def _compact(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    # Discord counts UTF-16 units; emoji must not overflow the message budget.
    if len(text.encode("utf-16-le")) // 2 <= limit:
        return text
    return text.encode("utf-16-le")[: max(0, limit - 1) * 2].decode(
        "utf-16-le", errors="ignore"
    ).rstrip() + "…"


def format_tool_call(tool_name, args, preview, adapter) -> str:
    from agent.display import (
        get_tool_emoji, get_tool_verb, prepare_tool_preview,
        tool_verb_connector, verb_drops_preview,
    )

    verb = get_tool_verb(tool_name)
    if not verb:
        # Plugin/MCP names still get a readable label, without dumping args.
        name = re.sub(r"^(?:mcp__|mcp_)", "", tool_name or "tool")
        name = re.sub(r"[_./:-]+", " ", name).strip()
        verb = f"Using {_compact(name, 60)}"
        preview = None
    detail = ""
    if preview and not verb_drops_preview(tool_name):
        prepared = prepare_tool_preview(
            tool_name, args, fallback=preview, max_len=40,
        )
        detail = adapter.format_tool_preview(prepared) if adapter else prepared.text
        # Keep short links intact; very long destinations become plain labels.
        if len(detail.encode("utf-16-le")) // 2 > 120:
            detail = prepared.text
        detail = tool_verb_connector(tool_name) + detail
    return _compact(f"{get_tool_emoji(tool_name, default='⚙️')} {verb}{detail}", 160)


class DiscordProgressWindow:
    """Retain calls independently of commentary and content segment breaks."""

    def __init__(self):
        self.calls = deque(maxlen=10)
        self.commentary = ""
        self.status = ""

    def append(self, event) -> None:
        if isinstance(event, tuple):
            if event[0] == "__tool__":
                self.calls.append(_compact(event[1], 160))
            elif event[0] == "__commentary__":
                # Escape code/format markers so model prose stays in subtext.
                text = _compact(event[1], 280)
                self.commentary = re.sub(r"([\\`*_~|<>])", r"\\\1", text)
            return
        self.status = _compact(event, 160)

    def render(self, limit: int) -> str:
        lines = list(self.calls)
        footer = self.commentary or self.status
        if footer:
            remaining = limit - len("\n".join(lines).encode("utf-16-le")) // 2 - 4
            if remaining > 0:
                lines.append("-# " + _compact(footer, remaining))
        return _compact_lines(lines, limit)


def _compact_lines(lines: list[str], limit: int) -> str:
    text = "\n".join(lines)
    if len(text.encode("utf-16-le")) // 2 <= limit:
        return text
    # Defensive fallback for adapters with a smaller-than-Discord limit.
    per_line = max(1, (limit - len(lines) + 1) // len(lines))
    return "\n".join(_compact(line, per_line) for line in lines)[:limit]
