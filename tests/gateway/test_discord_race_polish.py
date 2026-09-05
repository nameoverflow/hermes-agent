"""Discord adapter race polish: concurrent join_voice_channel must not
double-invoke channel.connect() on the same guild."""

import asyncio
import os
import time
from unittest.mock import MagicMock, patch

import pytest

from gateway.config import Platform, PlatformConfig


def _make_adapter():
    from plugins.platforms.discord.adapter import DiscordAdapter

    adapter = object.__new__(DiscordAdapter)
    adapter._platform = Platform.DISCORD
    adapter.config = PlatformConfig(enabled=True, token="t")
    adapter._ready_event = asyncio.Event()
    adapter._allowed_user_ids = set()
    adapter._allowed_role_ids = set()
    adapter._voice_clients = {}
    adapter._voice_locks = {}
    adapter._voice_receivers = {}
    adapter._voice_listen_tasks = {}
    adapter._voice_timeout_tasks = {}
    adapter._voice_text_channels = {}
    adapter._voice_sources = {}
    adapter._client = MagicMock()
    return adapter


@pytest.mark.asyncio
async def test_concurrent_joins_do_not_double_connect():
    """Two concurrent join_voice_channel calls on the same guild must
    serialize through the per-guild lock — only ONE channel.connect()
    actually fires; the second sees the _voice_clients entry the first
    just installed."""
    adapter = _make_adapter()

    connect_count = [0]
    release = asyncio.Event()

    class FakeVC:
        def __init__(self, channel):
            self.channel = channel

        def is_connected(self):
            return True

        async def move_to(self, _channel):
            return None

    async def slow_connect(self):
        connect_count[0] += 1
        await release.wait()
        return FakeVC(self)

    channel = MagicMock()
    channel.id = 111
    channel.guild.id = 42
    channel.connect = lambda: slow_connect(channel)

    from plugins.platforms.discord import adapter as discord_mod
    with patch.object(discord_mod, "VoiceReceiver",
                      MagicMock(return_value=MagicMock(start=lambda: None))):
        with patch.object(discord_mod.asyncio, "ensure_future",
                          lambda _c: asyncio.create_task(asyncio.sleep(0))):
            t1 = asyncio.create_task(adapter.join_voice_channel(channel))
            t2 = asyncio.create_task(adapter.join_voice_channel(channel))
            await asyncio.sleep(0.05)
            release.set()
            r1, r2 = await asyncio.gather(t1, t2)

    assert connect_count[0] == 1, (
        f"expected 1 channel.connect() call, got {connect_count[0]} — "
        "per-guild lock is not serializing join_voice_channel"
    )
    assert r1 is True and r2 is True
    assert 42 in adapter._voice_clients


def test_discord_inbound_claim_is_cross_instance_atomic(tmp_path):
    """A Discord message ID can be claimed only once across adapter instances.

    This guards the auto-thread path: if an old and new discord.py client both
    receive the same channel message during a reconnect/restart overlap, only
    one may continue to side effects such as message.create_thread().
    """
    from plugins.platforms.discord.adapter import _claim_discord_inbound_message_once

    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
        assert _claim_discord_inbound_message_once("msg-123") is True
        assert _claim_discord_inbound_message_once("msg-123") is False
        assert _claim_discord_inbound_message_once("msg-456") is True


def test_discord_inbound_claim_prunes_expired_markers(tmp_path):
    from plugins.platforms.discord.adapter import (
        _DISCORD_INBOUND_DEDUP_TTL_SECONDS,
        _claim_discord_inbound_message_once,
        _prune_discord_inbound_dedup_markers,
    )

    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
        assert _claim_discord_inbound_message_once("old-msg") is True
        marker_dir = tmp_path / "gateway" / "discord_inbound_dedup"
        [marker] = list(marker_dir.glob("*.seen"))
        old_time = time.time() - _DISCORD_INBOUND_DEDUP_TTL_SECONDS - 10
        os.utime(marker, (old_time, old_time))

        _prune_discord_inbound_dedup_markers()

        assert not marker.exists()
        assert _claim_discord_inbound_message_once("old-msg") is True
