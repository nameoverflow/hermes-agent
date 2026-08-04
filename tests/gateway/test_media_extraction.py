"""Tests for the explicit attachment-delivery contract."""

from gateway.platforms.base import BasePlatformAdapter


def test_legacy_marker_like_text_is_not_extracted():
    marker = "MEDIA" + ":/tmp/audio.ogg"
    content = '{"success": true, "file_path": "/tmp/audio.ogg", "media_tag": "' + marker + '"}'

    media, cleaned = BasePlatformAdapter.extract_media(content)

    assert media == []
    assert cleaned == content


def test_voice_and_document_directives_are_not_control_tags():
    marker = "MEDIA" + ":/tmp/audio.ogg"
    content = "[[audio_as_voice]]\n[[as_document]]\n" + marker

    media, cleaned = BasePlatformAdapter.extract_media(content)

    assert media == []
    assert cleaned == content


def test_bare_local_paths_are_not_inferred_as_attachments():
    content = "Saved report: /tmp/report.pdf"

    files, cleaned = BasePlatformAdapter.extract_local_files(content)

    assert files == []
    assert cleaned == content
