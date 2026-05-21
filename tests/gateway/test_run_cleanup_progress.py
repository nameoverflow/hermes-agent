"""Tests for opt-in cleanup of temporary progress bubbles.

When ``display.platforms.<plat>.cleanup_progress: true`` is set for a
platform whose adapter supports message deletion (e.g. Telegram), the
tool-progress bubble, "⏳ Still working..." notices, and status-callback
messages sent during a run are deleted after the final response is
delivered.

Failed runs skip cleanup so the bubbles remain as breadcrumbs.
Adapters without ``delete_message`` silently no-op.
"""

import asyncio
import importlib
import sys
import time
import types
from types import SimpleNamespace

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.session import SessionSource


# ---------------------------------------------------------------------------
# Test fakes — mirror those in test_run_progress_topics.py but add a
# delete_message implementation that records ids instead of hitting a bot.
# ---------------------------------------------------------------------------


class CleanupCaptureAdapter(BasePlatformAdapter):
    """Adapter that records every delete_message call for inspection."""

    _next_mid = 100

    def __init__(self, platform=Platform.TELEGRAM):
        super().__init__(PlatformConfig(enabled=True, token="***"), platform)
        self.sent = []
        self.edits = []
        self.deleted = []

    async def connect(self) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    def _mint_id(self) -> str:
        CleanupCaptureAdapter._next_mid += 1
        return str(CleanupCaptureAdapter._next_mid)

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        mid = self._mint_id()
        self.sent.append(
            {"chat_id": chat_id, "content": content, "message_id": mid, "metadata": metadata}
        )
        return SendResult(success=True, message_id=mid)

    async def edit_message(self, chat_id, message_id, content, *, finalize: bool = False) -> SendResult:
        self.edits.append({
            "chat_id": chat_id,
            "message_id": message_id,
            "content": content,
            "finalize": finalize,
        })
        return SendResult(success=True, message_id=message_id)

    async def delete_message(self, chat_id, message_id) -> bool:
        self.deleted.append({"chat_id": chat_id, "message_id": str(message_id)})
        return True

    async def send_typing(self, chat_id, metadata=None) -> None:
        return None

    async def stop_typing(self, chat_id) -> None:
        return None

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}


class NoDeleteAdapter(CleanupCaptureAdapter):
    """Adapter that inherits the base no-op delete_message (used to prove
    the cleanup path skips adapters without deletion support)."""

    async def delete_message(self, chat_id, message_id) -> bool:  # type: ignore[override]
        # Pretend to be an adapter whose platform doesn't support deletion:
        # match the base class behavior exactly. gateway/run.py checks
        # ``type(adapter).delete_message is BasePlatformAdapter.delete_message``
        # to detect this, so we re-assign at class body level below.
        raise AssertionError("should not be called — cleanup must skip this adapter")


# Re-bind so the class's delete_message identity equals the base's.
NoDeleteAdapter.delete_message = BasePlatformAdapter.delete_message


class FailingEditFeishuAdapter(CleanupCaptureAdapter):
    """Feishu fake whose progress/final edits fail after the first send."""

    async def edit_message(self, chat_id, message_id, content, *, finalize: bool = False) -> SendResult:
        self.edits.append({
            "chat_id": chat_id,
            "message_id": message_id,
            "content": content,
            "finalize": finalize,
        })
        return SendResult(success=False, message_id=message_id, error="simulated update failed")


class ProgressAgent:
    """Emits two tool-progress events and returns a normal final response."""

    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        cb = self.tool_progress_callback
        if cb is not None:
            cb("tool.started", "terminal", "pwd", {})
            time.sleep(0.25)
            cb("tool.started", "terminal", "ls", {})
            time.sleep(0.25)
        return {"final_response": "done", "messages": [], "api_calls": 1}


class FailingAgent:
    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        cb = self.tool_progress_callback
        if cb is not None:
            cb("tool.started", "terminal", "pwd", {})
            time.sleep(0.25)
        # Empty final_response + failed=True is the shape the gateway
        # actually returns on provider errors (see gateway/run.py where
        # failed keys are only propagated when final_response is empty).
        return {
            "final_response": "",
            "messages": [],
            "api_calls": 1,
            "failed": True,
            "error": "simulated provider failure",
        }


class BackgroundReviewAgent:
    """Queues a deferred status/background-review message, then succeeds."""

    def __init__(self, **kwargs):
        self.tools = []
        self.background_review_callback = None

    def run_conversation(self, message, conversation_history=None, task_id=None):
        cb = self.background_review_callback
        if cb is not None:
            cb("💾 Memory updated")
            time.sleep(0.05)
        return {"final_response": "done", "messages": [], "api_calls": 1}


class SharedBackgroundReviewProgressAgent:
    """Emits tool progress and a background-review note during one gateway turn."""

    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.tools = []
        self.background_review_callback = None

    def run_conversation(self, message, conversation_history=None, task_id=None):
        tool_cb = self.tool_progress_callback
        if tool_cb is not None:
            tool_cb("tool.started", "terminal", "pwd", {})
        time.sleep(0.45)
        bg_cb = self.background_review_callback
        if bg_cb is not None:
            bg_cb("💾 Memory updated")
        time.sleep(0.45)
        return {"final_response": "done", "messages": [], "api_calls": 1}


class SlowProgressAgent:
    """Runs long enough for a still-working notice after one progress bubble."""

    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        cb = self.tool_progress_callback
        if cb is not None:
            cb("tool.started", "terminal", "pwd", {})
        # The progress worker polls every ~0.3s; keep the fake run alive long
        # enough for the first progress bubble to be sent before final reuse.
        time.sleep(0.45)
        return {"final_response": "done", "messages": [], "api_calls": 1}


class SharedProgressAcrossContentAgent:
    """Emits progress, then visible content, then another progress event."""

    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.stream_delta_callback = kwargs.get("stream_delta_callback")
        self.interim_assistant_callback = kwargs.get("interim_assistant_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        cb = self.tool_progress_callback
        if cb is not None:
            cb("tool.started", "terminal", "pwd", {})
        time.sleep(0.45)
        stream_cb = getattr(self, "stream_delta_callback", None)
        if stream_cb is not None:
            stream_cb("Interim content before another tool.")
        # Mark the streamed content segment as closed so the gateway's
        # on_new_message hook fires. Feishu should keep using the original
        # progress bubble instead of starting a second one after this marker.
        interim_cb = getattr(self, "interim_assistant_callback", None)
        if interim_cb is not None:
            interim_cb("", already_streamed=True)
        time.sleep(0.45)
        if cb is not None:
            cb("tool.started", "terminal", "ls", {})
        time.sleep(0.45)
        return {"final_response": "done", "messages": [], "api_calls": 1}


def _make_runner(adapter):
    gateway_run = importlib.import_module("gateway.run")
    GatewayRunner = gateway_run.GatewayRunner
    runner = object.__new__(GatewayRunner)
    runner.adapters = {adapter.platform: adapter}
    runner._voice_mode = {}
    runner._prefill_messages = []
    runner._ephemeral_system_prompt = ""
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._session_db = None
    runner._running_agents = {}
    runner._session_run_generation = {}
    runner.hooks = SimpleNamespace(loaded_hooks=False)
    runner.config = SimpleNamespace(
        thread_sessions_per_user=False,
        group_sessions_per_user=False,
        stt_enabled=False,
    )
    return runner


def _install_fakes(monkeypatch, agent_cls, *, cleanup_on: bool, platform_key: str = "telegram"):
    """Wire up the module stubs every _run_agent test needs."""
    monkeypatch.setenv("HERMES_TOOL_PROGRESS_MODE", "all")

    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = agent_cls
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)
    import tools.terminal_tool  # noqa: F401 — register tool emoji

    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "fake"})

    # Wire the per-platform cleanup_progress flag via the config loader the
    # gateway actually reads (``_load_gateway_config`` returns user config).
    cfg = {
        "display": {
            "platforms": {
                platform_key: {"cleanup_progress": True},
            }
        }
    } if cleanup_on else {}
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: cfg)
    return gateway_run


async def _fire_post_delivery_callback(adapter, session_key, *, require: bool = False) -> None:
    cb = adapter.pop_post_delivery_callback(session_key)
    if require:
        assert callable(cb)
    if cb is None:
        return
    result = cb()
    if hasattr(result, "__await__"):
        await result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cleanup_off_by_default_leaves_bubbles(monkeypatch, tmp_path):
    """Without ``cleanup_progress: true``, firing whatever callback is
    registered never reaches delete_message."""
    adapter = CleanupCaptureAdapter()
    runner = _make_runner(adapter)
    gateway_run = _install_fakes(monkeypatch, ProgressAgent, cleanup_on=False)
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    source = SessionSource(platform=Platform.TELEGRAM, chat_id="-1001")
    session_key = "agent:main:telegram:group:-1001"

    result = await runner._run_agent(
        message="hello",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-1",
        session_key=session_key,
    )

    assert result["final_response"] == "done"
    # Even if an unrelated callback got registered (background-review
    # release lives in the same slot) firing it should never cause any
    # delete_message calls when cleanup is off.

    await _fire_post_delivery_callback(adapter, session_key)
    for _ in range(10):
        await asyncio.sleep(0.01)
    assert adapter.deleted == []


@pytest.mark.asyncio
async def test_cleanup_registers_callback_and_deletes_on_success(monkeypatch, tmp_path):
    """With the flag on, the cleanup callback deletes the progress bubble."""
    adapter = CleanupCaptureAdapter()
    runner = _make_runner(adapter)
    gateway_run = _install_fakes(monkeypatch, ProgressAgent, cleanup_on=True)
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    source = SessionSource(platform=Platform.TELEGRAM, chat_id="-1001")
    session_key = "agent:main:telegram:group:-1001"

    result = await runner._run_agent(
        message="hello",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-1",
        session_key=session_key,
    )

    assert result["final_response"] == "done"
    # The cleanup callback should be registered for this session.

    # Fire it (base.py does this in _process_message_background's finally)
    # and let the awaited cleanup coroutine run to completion.
    await _fire_post_delivery_callback(adapter, session_key, require=True)
    # The awaited cleanup usually deletes synchronously; keep a tiny drain
    # loop for chained/background sends that may resolve on later ticks.
    for _ in range(20):
        await asyncio.sleep(0.01)
        if adapter.deleted:
            break

    # At least the first tool-progress bubble should have been deleted.
    assert len(adapter.deleted) >= 1, f"deleted={adapter.deleted} sent={adapter.sent}"
    for entry in adapter.deleted:
        assert entry["chat_id"] == "-1001"


@pytest.mark.asyncio
async def test_cleanup_skipped_on_failed_run(monkeypatch, tmp_path):
    """Failed runs skip cleanup registration — breadcrumbs stay."""
    adapter = CleanupCaptureAdapter()
    runner = _make_runner(adapter)
    gateway_run = _install_fakes(monkeypatch, FailingAgent, cleanup_on=True)
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    source = SessionSource(platform=Platform.TELEGRAM, chat_id="-1001")
    session_key = "agent:main:telegram:group:-1001"

    result = await runner._run_agent(
        message="hello",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-1",
        session_key=session_key,
    )

    assert result.get("failed") is True
    # Whatever callback is registered should not trigger any deletion —
    # the cleanup callback is skipped on failed runs.

    await _fire_post_delivery_callback(adapter, session_key)
    for _ in range(10):
        await asyncio.sleep(0.01)
    assert adapter.deleted == []


@pytest.mark.asyncio
async def test_cleanup_noop_on_adapter_without_delete_support(monkeypatch, tmp_path):
    """Adapters that inherit the base-class delete_message no-op are
    detected up front — the cleanup path never registers its callback so
    a stray bg-review callback (if present) can fire harmlessly."""
    adapter = NoDeleteAdapter()
    runner = _make_runner(adapter)
    gateway_run = _install_fakes(monkeypatch, ProgressAgent, cleanup_on=True)
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    source = SessionSource(platform=Platform.TELEGRAM, chat_id="-1001")
    session_key = "agent:main:telegram:group:-1001"

    result = await runner._run_agent(
        message="hello",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-1",
        session_key=session_key,
    )

    assert result["final_response"] == "done"
    # No deletion attempts on an adapter without delete_message support.
    # (The NoDeleteAdapter.delete_message would raise AssertionError if
    # the cleanup closure had somehow captured a reference to it.)
    assert adapter.deleted == []


@pytest.mark.asyncio
async def test_cleanup_chains_with_existing_callback(monkeypatch, tmp_path):
    """When a bg-review-style callback is already registered, the cleanup
    callback chains with it — both fire, neither clobbers the other."""
    adapter = CleanupCaptureAdapter()
    runner = _make_runner(adapter)
    gateway_run = _install_fakes(monkeypatch, ProgressAgent, cleanup_on=True)
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    source = SessionSource(platform=Platform.TELEGRAM, chat_id="-1001")
    session_key = "agent:main:telegram:group:-1001"

    pre_existing_fired = []

    def _preexisting_callback() -> None:
        pre_existing_fired.append(True)

    # Pre-register a callback with the same generation the run will use
    # (run_generation=None in this test path — matches the default slot).
    adapter.register_post_delivery_callback(session_key, _preexisting_callback)

    result = await runner._run_agent(
        message="hello",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-1",
        session_key=session_key,
    )

    assert result["final_response"] == "done"

    await _fire_post_delivery_callback(adapter, session_key, require=True)
    for _ in range(20):
        await asyncio.sleep(0.01)
        if adapter.deleted:
            break

    # Both effects land: the pre-existing callback fires AND the cleanup
    # deletes at least one progress bubble.
    assert pre_existing_fired == [True]
    assert len(adapter.deleted) >= 1


@pytest.mark.asyncio
async def test_discord_cleanup_enabled_by_default(monkeypatch, tmp_path):
    """Discord opts into transient progress cleanup without user config."""
    adapter = CleanupCaptureAdapter(platform=Platform.DISCORD)
    runner = _make_runner(adapter)
    gateway_run = _install_fakes(monkeypatch, ProgressAgent, cleanup_on=False)
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    source = SessionSource(platform=Platform.DISCORD, chat_id="1234")
    session_key = "agent:main:discord:channel:1234"

    result = await runner._run_agent(
        message="hello",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-1",
        session_key=session_key,
    )

    assert result["final_response"] == "done"
    assert result.get("already_sent") is True
    all_progress_text = "\n".join(call["content"] for call in adapter.sent)
    all_progress_text += "\n".join(call["content"] for call in adapter.edits)
    assert "pwd" not in all_progress_text
    assert "ls" not in all_progress_text
    assert 'terminal: "' not in all_progress_text

    # Discord now follows the single-bubble path too: one temporary progress
    # message is edited into the final answer, so cleanup has nothing to delete.
    assert len(adapter.sent) == 1, adapter.sent
    progress_mid = adapter.sent[0]["message_id"]
    assert adapter.edits[-1]["message_id"] == progress_mid
    assert adapter.edits[-1]["content"] == "done"
    assert adapter.edits[-1]["finalize"] is True

    await _fire_post_delivery_callback(adapter, session_key, require=True)
    for _ in range(10):
        await asyncio.sleep(0.01)
    assert adapter.deleted == []


@pytest.mark.asyncio
async def test_discord_thread_progress_edits_and_cleanup_target_thread(monkeypatch, tmp_path):
    """Discord thread progress is sent via metadata, so edit/delete must use the thread id.

    The parent channel cannot fetch messages that live inside a Discord thread.
    If the gateway edits/deletes with the parent chat_id, real Discord rejects
    the lookup; edits fall back to one permanent message per update and cleanup
    never removes them.
    """
    adapter = CleanupCaptureAdapter(platform=Platform.DISCORD)
    runner = _make_runner(adapter)
    gateway_run = _install_fakes(monkeypatch, ProgressAgent, cleanup_on=False)
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="parent-channel",
        thread_id="thread-123",
    )
    session_key = "agent:main:discord:channel:thread-123"

    result = await runner._run_agent(
        message="hello",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-1",
        session_key=session_key,
    )

    assert result["final_response"] == "done"
    assert result.get("already_sent") is True
    # The first progress bubble is still sent via the parent + metadata so the
    # adapter can route it to the thread in the normal Discord send path.
    assert adapter.sent
    assert len(adapter.sent) == 1, adapter.sent
    progress_mid = adapter.sent[0]["message_id"]
    assert adapter.sent[0]["chat_id"] == "parent-channel"
    assert adapter.sent[0]["metadata"] == {"thread_id": "thread-123"}
    # Follow-up progress edits and final reuse must target the actual Discord
    # thread channel. The final answer remains in the reused progress message,
    # so there should be no post-delivery delete.
    assert adapter.edits
    assert {entry["chat_id"] for entry in adapter.edits} == {"thread-123"}
    assert adapter.edits[-1]["message_id"] == progress_mid
    assert adapter.edits[-1]["content"] == "done"
    assert adapter.edits[-1]["finalize"] is True

    await _fire_post_delivery_callback(adapter, session_key, require=True)
    for _ in range(10):
        await asyncio.sleep(0.01)
    assert adapter.deleted == []


@pytest.mark.asyncio
async def test_discord_still_working_merges_into_progress_bubble(monkeypatch, tmp_path):
    """Discord should not send a separate Still working bubble either."""
    adapter = CleanupCaptureAdapter(platform=Platform.DISCORD)
    runner = _make_runner(adapter)
    gateway_run = _install_fakes(monkeypatch, SlowProgressAgent, cleanup_on=False)
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setenv("HERMES_AGENT_NOTIFY_INTERVAL", "0.05")

    source = SessionSource(platform=Platform.DISCORD, chat_id="1234")
    session_key = "agent:main:discord:channel:1234"

    result = await runner._run_agent(
        message="hello",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-discord-still-working",
        session_key=session_key,
    )

    assert result["final_response"] == "done"
    assert result.get("already_sent") is True
    assert len(adapter.sent) == 1, adapter.sent
    progress_mid = adapter.sent[0]["message_id"]
    progress_edits = [entry for entry in adapter.edits if entry["message_id"] == progress_mid]
    assert any("Still working" in entry["content"] for entry in progress_edits), adapter.edits
    assert adapter.edits[-1]["message_id"] == progress_mid
    assert adapter.edits[-1]["content"] == "done"
    assert adapter.edits[-1]["finalize"] is True

    await _fire_post_delivery_callback(adapter, session_key, require=True)
    for _ in range(10):
        await asyncio.sleep(0.01)
    assert adapter.deleted == []


@pytest.mark.asyncio
async def test_discord_progress_stays_single_bubble_across_content_segments(monkeypatch, tmp_path):
    """Discord keeps one editable progress bubble across content resets."""
    adapter = CleanupCaptureAdapter(platform=Platform.DISCORD)
    runner = _make_runner(adapter)
    gateway_run = _install_fakes(monkeypatch, SharedProgressAcrossContentAgent, cleanup_on=False)
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    source = SessionSource(platform=Platform.DISCORD, chat_id="1234")
    session_key = "agent:main:discord:channel:1234"

    result = await runner._run_agent(
        message="hello",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-discord-content",
        session_key=session_key,
    )

    assert result["final_response"] == "done"
    assert result.get("already_sent") is True
    progress_sends = [entry for entry in adapter.sent if entry["content"].startswith("💻 terminal")]
    assert len(progress_sends) == 1, adapter.sent
    progress_mid = progress_sends[0]["message_id"]
    progress_edits = [entry for entry in adapter.edits if entry["message_id"] == progress_mid]
    assert progress_edits, adapter.edits
    # Discord's default compact progress hides raw previews, but the second
    # tool should still update the same message rather than sending a new one.
    assert any("×2" in entry["content"] for entry in progress_edits), adapter.edits
    assert adapter.edits[-1]["message_id"] == progress_mid
    assert adapter.edits[-1]["content"] == "done"
    assert adapter.edits[-1]["finalize"] is True

    await _fire_post_delivery_callback(adapter, session_key, require=True)
    for _ in range(10):
        await asyncio.sleep(0.01)
    assert adapter.deleted == []


@pytest.mark.asyncio
async def test_feishu_still_working_merges_into_progress_bubble(monkeypatch, tmp_path):
    """Feishu should not send a separate Still working bubble that later recalls.

    With tool progress enabled, the long-running notice is queued into the same
    editable progress bubble.  The single Feishu progress message can then be
    reused for the final answer instead of being deleted.
    """
    adapter = CleanupCaptureAdapter(platform=Platform.FEISHU)
    runner = _make_runner(adapter)
    gateway_run = _install_fakes(
        monkeypatch,
        SlowProgressAgent,
        cleanup_on=True,
        platform_key="feishu",
    )
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setenv("HERMES_AGENT_NOTIFY_INTERVAL", "0.05")

    source = SessionSource(platform=Platform.FEISHU, chat_id="oc_chat", chat_type="dm")
    session_key = "agent:main:feishu:dm:oc_chat"

    result = await runner._run_agent(
        message="hello",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-feishu",
        session_key=session_key,
    )

    assert result["final_response"] == "done"
    assert result.get("already_sent") is True
    # Only the original progress bubble is sent. The still-working notice and
    # final answer both arrive as edits to that same message.
    assert len(adapter.sent) == 1, adapter.sent
    assert adapter.sent[0]["content"].startswith("💻 terminal")
    assert any("Still working" in entry["content"] for entry in adapter.edits), adapter.edits
    assert adapter.edits[-1]["content"] == "done"
    assert adapter.edits[-1]["finalize"] is True
    assert adapter.deleted == []

    await _fire_post_delivery_callback(adapter, session_key)
    for _ in range(10):
        await asyncio.sleep(0.01)
    assert adapter.deleted == []


@pytest.mark.asyncio
async def test_feishu_progress_stays_single_bubble_across_content_segments(monkeypatch, tmp_path):
    """Feishu keeps one editable tool-progress bubble across content resets."""
    adapter = CleanupCaptureAdapter(platform=Platform.FEISHU)
    runner = _make_runner(adapter)
    gateway_run = _install_fakes(
        monkeypatch,
        SharedProgressAcrossContentAgent,
        cleanup_on=True,
        platform_key="feishu",
    )
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    source = SessionSource(platform=Platform.FEISHU, chat_id="oc_chat", chat_type="dm")
    session_key = "agent:main:feishu:dm:oc_chat"

    result = await runner._run_agent(
        message="hello",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-feishu-content",
        session_key=session_key,
    )

    assert result["final_response"] == "done"
    assert result.get("already_sent") is True
    progress_sends = [entry for entry in adapter.sent if entry["content"].startswith("💻 terminal")]
    assert len(progress_sends) == 1, adapter.sent
    progress_mid = progress_sends[0]["message_id"]
    progress_edits = [entry for entry in adapter.edits if entry["message_id"] == progress_mid]
    assert progress_edits, adapter.edits
    assert any('terminal: "ls"' in entry["content"] for entry in progress_edits), adapter.edits
    assert adapter.edits[-1]["message_id"] == progress_mid
    assert adapter.edits[-1]["content"] == "done"
    assert adapter.edits[-1]["finalize"] is True
    assert adapter.deleted == []


@pytest.mark.asyncio
async def test_feishu_background_review_merges_into_progress_bubble(monkeypatch, tmp_path):
    """Feishu background-review/status notes should share the progress bubble."""
    adapter = CleanupCaptureAdapter(platform=Platform.FEISHU)
    runner = _make_runner(adapter)
    gateway_run = _install_fakes(
        monkeypatch,
        SharedBackgroundReviewProgressAgent,
        cleanup_on=True,
        platform_key="feishu",
    )
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    source = SessionSource(platform=Platform.FEISHU, chat_id="oc_chat", chat_type="dm")
    session_key = "agent:main:feishu:dm:oc_chat"

    result = await runner._run_agent(
        message="hello",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-feishu-bg-review",
        session_key=session_key,
    )

    assert result["final_response"] == "done"
    assert result.get("already_sent") is True
    # One temporary Feishu bubble is created, then both the background-review
    # note and final answer edit that same message.  No post-delivery send means
    # no extra recalled/deleted marker later.
    assert len(adapter.sent) == 1, adapter.sent
    progress_mid = adapter.sent[0]["message_id"]
    progress_edits = [entry for entry in adapter.edits if entry["message_id"] == progress_mid]
    assert any("💾 Memory updated" in entry["content"] for entry in progress_edits), adapter.edits
    assert adapter.edits[-1]["message_id"] == progress_mid
    assert adapter.edits[-1]["content"] == "done"
    assert adapter.edits[-1]["finalize"] is True
    assert adapter.deleted == []
    await _fire_post_delivery_callback(adapter, session_key)
    assert adapter.deleted == []


@pytest.mark.asyncio
async def test_feishu_progress_edit_failure_does_not_spawn_more_tool_bubbles(monkeypatch, tmp_path):
    """If Feishu message.update rejects an edit, don't fall back to tool sends."""
    adapter = FailingEditFeishuAdapter(platform=Platform.FEISHU)
    runner = _make_runner(adapter)
    gateway_run = _install_fakes(
        monkeypatch,
        SharedProgressAcrossContentAgent,
        cleanup_on=True,
        platform_key="feishu",
    )
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    source = SessionSource(platform=Platform.FEISHU, chat_id="oc_chat", chat_type="dm")
    session_key = "agent:main:feishu:dm:oc_chat"

    result = await runner._run_agent(
        message="hello",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-feishu-edit-fail",
        session_key=session_key,
    )

    assert result["final_response"] == "done"
    assert not result.get("already_sent")
    progress_sends = [entry for entry in adapter.sent if entry["content"].startswith("💻 terminal")]
    assert len(progress_sends) == 1, adapter.sent
    assert all('terminal: "ls"' not in entry["content"] for entry in adapter.sent), adapter.sent
    assert adapter.edits, "edit failures should be attempted, not replaced by sends"
    assert adapter.deleted == []


@pytest.mark.asyncio
async def test_discord_background_review_merges_into_progress_bubble(monkeypatch, tmp_path):
    """Discord background-review/status notes should share the progress bubble."""
    adapter = CleanupCaptureAdapter(platform=Platform.DISCORD)
    runner = _make_runner(adapter)
    gateway_run = _install_fakes(monkeypatch, SharedBackgroundReviewProgressAgent, cleanup_on=False)
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    source = SessionSource(platform=Platform.DISCORD, chat_id="1234")
    session_key = "agent:main:discord:channel:1234"

    result = await runner._run_agent(
        message="hello",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-1",
        session_key=session_key,
    )

    assert result["final_response"] == "done"
    assert result.get("already_sent") is True
    assert len(adapter.sent) == 1, adapter.sent
    progress_mid = adapter.sent[0]["message_id"]
    progress_edits = [entry for entry in adapter.edits if entry["message_id"] == progress_mid]
    assert any("💾 Memory updated" in entry["content"] for entry in progress_edits), adapter.edits
    assert adapter.edits[-1]["message_id"] == progress_mid
    assert adapter.edits[-1]["content"] == "done"
    assert adapter.edits[-1]["finalize"] is True

    await _fire_post_delivery_callback(adapter, session_key, require=True)
    for _ in range(10):
        await asyncio.sleep(0.01)
    assert adapter.deleted == []
