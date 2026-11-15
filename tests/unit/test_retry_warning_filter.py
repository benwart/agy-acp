"""Unit tests for transient `agy acp` retry warning filtering (batch and streaming)."""

from __future__ import annotations

import pytest

from agy_acp.logic.retry_warning_filter import (
    RetryFilterStep,
    RetryWarningStreamFilter,
    strip_agy_retry_warnings,
)


def test_strips_real_world_state_db_samples() -> None:
    """Real-world `state.db` samples strip retry banners without leaving stray newlines."""
    sample_1 = (
        "\n⚠ API error (attempt 1): request failed (code 502): "
        'HTML response "Error 502 (Server Error)!!1"\n'
        "Saved [check_upgrades.py]..."
    )
    assert strip_agy_retry_warnings(sample_1) == "Saved [check_upgrades.py]..."

    sample_2 = (
        "\n⚠ API error (attempt 1): request failed: Post "
        '"https://daily-cloudcode-pa.googleapis.com/v1internal:streamGenerateContent?alt=sse": '
        "read tcp 192.168.116.201:55499->172.217.119.4:443: read: connection reset by peer\n"
        "\n⚠ API error (attempt 2): request failed: Post "
        '"https://daily-cloudcode-pa.googleapis.com/v1internal:streamGenerateContent?alt=sse": '
        "read tcp 192.168.116.201:55612->172.217.115.4:443: read: connection reset by peer\n"
        "<!-- nothing_to_do -->"
    )
    assert strip_agy_retry_warnings(sample_2) == "<!-- nothing_to_do -->"

    sample_3 = (
        "\n⚠ API error (attempt 1): request failed: Post "
        '"https://daily-cloudcode-pa.googleapis.com/v1internal:streamGenerateContent?alt=sse": '
        "read tcp 192.168.116.201:56287->172.217.112.4:443: read: operation timed out\n"
        "<!-- nothing_to_do -->"
    )
    assert strip_agy_retry_warnings(sample_3) == "<!-- nothing_to_do -->"


def test_supports_variation_selector_crlf_and_optional_surrounding_newlines() -> None:
    """Supports `⚠️` (`\\ufe0f`), multi-digit attempts, CRLF, and missing leading/trailing `\\n`."""
    assert strip_agy_retry_warnings("⚠️ API error (attempt 12): transient 503\nDone.") == "Done."
    assert strip_agy_retry_warnings("\n⚠️ API error (attempt 3): transient 503") == ""
    assert strip_agy_retry_warnings("⚠ API error (attempt 1): eof without newline") == ""
    assert (
        strip_agy_retry_warnings("Line 1\r\n\r\n⚠ API error (attempt 1): reset\r\nLine 2\r\n")
        == "Line 1\r\nLine 2\r\n"
    )
    assert (
        strip_agy_retry_warnings("Line 1\n⚠ API error (attempt 1): reset\nLine 2")
        == "Line 1\nLine 2"
    )


def test_preserves_ordinary_prose_mentioning_api_error_or_warning_symbols() -> None:
    """Ordinary assistant prose without exact `⚠ API error (attempt <digits>):` prefix stays."""
    cases = [
        "The upstream returned an API error (attempt 1): connection reset.",
        "⚠ Disk space is low on /tmp.\nNext line.",
        "⚠️ API error: missing attempt counter.\nNext line.",
        "⚠ API error (attempt X): non-numeric attempt.\nNext line.",
        "⚠ API error (attempt 1) missing colon.\nNext line.",
        "`⚠ API error (attempt 1): inline code stays intact`",
        "- ⚠ API error (attempt 1): markdown list item stays intact",
        "Para 1.\n\nPara 2.\n",
    ]
    for prose in cases:
        assert strip_agy_retry_warnings(prose) == prose


def test_stream_filter_emits_ordinary_prose_immediately_and_extracts_warnings() -> None:
    """Ordinary prose streams immediately; retry warnings return structured metadata."""
    stream = RetryWarningStreamFilter()
    assert stream.feed("Initial chunk. ") == RetryFilterStep(text="Initial chunk. ", warnings=())
    assert stream.feed("More words.") == RetryFilterStep(text="More words.", warnings=())
    assert stream.flush() == RetryFilterStep(text="", warnings=())


def test_stream_filter_handles_split_chunks_divergent_prefixes_and_tool_boundaries() -> None:
    """Split chunks buffer only candidate prefixes and flush cleanly across tool boundaries."""
    stream = RetryWarningStreamFilter()
    assert stream.feed("\n⚠") == RetryFilterStep(text="", warnings=())
    assert stream.feed(" API error (attempt ") == RetryFilterStep(text="", warnings=())
    assert stream.feed("1): 502 Bad Gateway") == RetryFilterStep(text="", warnings=())
    step = stream.feed("\nSaved [check_upgrades.py]")
    assert step == RetryFilterStep(
        text="Saved [check_upgrades.py]",
        warnings=("⚠ API error (attempt 1): 502 Bad Gateway",),
    )
    assert stream.flush() == RetryFilterStep(text="", warnings=())

    # Divergent prefix flushes buffered prefix immediately without waiting for newline
    divergent = RetryWarningStreamFilter()
    assert divergent.feed("\n⚠ API ") == RetryFilterStep(text="", warnings=())
    div_step = divergent.feed("warning: quota at 80%")
    assert div_step == RetryFilterStep(text="\n⚠ API warning: quota at 80%", warnings=())

    # Tool-call boundary flushes incomplete non-banner prefix, and resets line-start so
    # a post-tool retry banner strips its leading `\n` cleanly.
    boundary = RetryWarningStreamFilter()
    assert boundary.feed("Checking file. ") == RetryFilterStep(text="Checking file. ", warnings=())
    assert boundary.flush_before_boundary() == RetryFilterStep(text="", warnings=())
    post_tool = boundary.feed("\n⚠ API error (attempt 1): 502\nAll green.")
    assert post_tool == RetryFilterStep(
        text="All green.",
        warnings=("⚠ API error (attempt 1): 502",),
    )

    partial_before_tool = RetryWarningStreamFilter()
    assert partial_before_tool.feed("\n⚠ Hold") == RetryFilterStep(text="\n⚠ Hold", warnings=())
    partial_candidate = RetryWarningStreamFilter()
    assert partial_candidate.feed("\n⚠ API ") == RetryFilterStep(text="", warnings=())
    assert partial_candidate.flush_before_boundary() == RetryFilterStep(
        text="\n⚠ API ",
        warnings=(),
    )


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 5, 11, 29])
def test_stream_filter_matches_batch_across_arbitrary_chunk_sizes(chunk_size: int) -> None:
    """Arbitrary fixed-size token chunking produces identical text and warnings to batch."""
    doc = (
        "\n⚠ API error (attempt 1): first transient failure\n"
        "\n⚠️ API error (attempt 2): second transient failure\n"
        "Intro paragraph mentioning API error (attempt 1) in prose.\n\n"
        "⚠ Notice: ordinary warning line.\r\n\r\n"
        "⚠ API error (attempt 3): crlf transient failure\r\n"
        "<!-- nothing_to_do -->"
    )
    stream = RetryWarningStreamFilter()
    parts: list[str] = []
    warnings: list[str] = []
    for idx in range(0, len(doc), chunk_size):
        step = stream.feed(doc[idx : idx + chunk_size])
        parts.append(step.text)
        warnings.extend(step.warnings)
    tail = stream.flush()
    parts.append(tail.text)
    warnings.extend(tail.warnings)

    assert "".join(parts) == strip_agy_retry_warnings(doc)
    assert warnings == [
        "⚠ API error (attempt 1): first transient failure",
        "⚠️ API error (attempt 2): second transient failure",
        "⚠ API error (attempt 3): crlf transient failure",
    ]
