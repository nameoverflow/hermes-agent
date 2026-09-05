"""Behavior of the bounded Discord progress surface."""

from gateway.discord_progress import DiscordProgressWindow, format_tool_call


def test_window_keeps_last_ten_calls_across_commentary_and_segments():
    window = DiscordProgressWindow()
    for i in range(25):
        window.append(("__tool__", f"Reading file {i}"))
        window.append(("__commentary__", f"Checking result {i}"))
        window.append(("__reset__",))
    lines = window.render(1936).splitlines()
    assert lines[:-1] == [f"Reading file {i}" for i in range(15, 25)]
    assert lines[-1] == "-# Checking result 24"


def test_duplicate_calls_remain_individual_entries():
    window = DiscordProgressWindow()
    for _ in range(12):
        window.append(("__tool__", "Searching the web"))
    assert window.render(1936).splitlines() == ["Searching the web"] * 10


def test_multiline_commentary_and_emoji_stay_within_discord_budget():
    window = DiscordProgressWindow()
    for _ in range(10):
        window.append(("__tool__", "🔎" * 500))
    window.append(("__commentary__", "```\n**Hello**\n" + "🌍" * 500))
    text = window.render(1936)
    assert len(text.encode("utf-16-le")) // 2 <= 1936
    assert len(text.splitlines()) == 11
    assert text.splitlines()[-1].startswith("-# ")
    assert "```" not in text


def test_tool_labels_are_short_prose_even_for_verbose_or_custom_tools():
    terminal = format_tool_call("terminal", {"command": "pwd\n" + "x" * 1000}, "pwd", None)
    custom = format_tool_call("mcp__my_service__fetch_items", {"secret": "hidden"}, '{"secret":"hidden"}', None)
    assert "Running" in terminal
    assert "```" not in terminal
    assert "Using my service fetch items" in custom
    assert "hidden" not in custom
    assert "_" not in custom
    assert len(terminal) <= 160


def test_builtin_tool_without_preview_still_uses_friendly_label():
    assert "Searching the web" in format_tool_call("web_search", {}, None, None)
