"""Unit tests for Antigravity carousel Markdown normalization (batch and streaming)."""

from __future__ import annotations

import pytest

from agy_acp.logic.carousel_normalizer import (
    CarouselStreamNormalizer,
    normalize_carousel_markdown,
)


def test_unwraps_four_backtick_carousel_with_images_and_following_diff() -> None:
    """Reproduce user screenshot: 4-backtick carousel unwraps into GFM image paragraphs."""
    raw = (
        "Changes pushed to `origin/main`.\n\n"
        "````carousel\n"
        "![Synced Screen 2 full-screen To-Do snapshot]"
        "(/Users/benwart/.gemini/antigravity-cli/brain/ee2ecabe/screen-2.png)\n"
        "<!-- slide -->\n"
        "![Synced Screen 2B left-swipe Remove reveal snapshot]"
        "(/Users/benwart/.gemini/antigravity-cli/brain/ee2ecabe/screen-2b.png)\n"
        "<!-- slide -->\n"
        "![Synced Screen 2C Maps-style Add Stop snapshot]"
        "(/Users/benwart/.gemini/antigravity-cli/brain/ee2ecabe/screen-2c.png)\n"
        "````\n\n"
        "```diff\n"
        "diff --git a/app.swift b/app.swift\n"
        "+let ready = true\n"
        "```\n"
    )
    expected = (
        "Changes pushed to `origin/main`.\n\n"
        "![Synced Screen 2 full-screen To-Do snapshot]"
        "(/Users/benwart/.gemini/antigravity-cli/brain/ee2ecabe/screen-2.png)\n\n"
        "![Synced Screen 2B left-swipe Remove reveal snapshot]"
        "(/Users/benwart/.gemini/antigravity-cli/brain/ee2ecabe/screen-2b.png)\n\n"
        "![Synced Screen 2C Maps-style Add Stop snapshot]"
        "(/Users/benwart/.gemini/antigravity-cli/brain/ee2ecabe/screen-2c.png)\n\n"
        "```diff\n"
        "diff --git a/app.swift b/app.swift\n"
        "+let ready = true\n"
        "```\n"
    )
    assert normalize_carousel_markdown(raw) == expected


def test_handles_tight_fences_tilde_fences_and_indentation() -> None:
    """3-backtick, tilde, and indented carousels unwrap with clean block spacing."""
    tight = (
        "Intro:\n"
        "```Carousel  \n"
        "![One](/tmp/one.png)\n"
        "<!--slide-->\n"
        "![Two](/tmp/two.png)\n"
        "```\n"
        "Outro."
    )
    assert normalize_carousel_markdown(tight) == (
        "Intro:\n\n![One](/tmp/one.png)\n\n![Two](/tmp/two.png)\n\nOutro."
    )

    tilde_indented = (
        "- Gallery:\n"
        "  ~~~~carousel\n"
        "  ![Alpha](/tmp/alpha.png)\n"
        "  <!--  SLIDE  -->\n"
        "  ![Beta](/tmp/beta.png)\n"
        "  ~~~~\n"
    )
    assert normalize_carousel_markdown(tilde_indented) == (
        "- Gallery:\n\n  ![Alpha](/tmp/alpha.png)\n\n  ![Beta](/tmp/beta.png)\n"
    )


def test_preserves_nested_code_blocks_inside_carousel_slides() -> None:
    """Inner code fences inside a 4-backtick carousel remain fenced and untouched."""
    raw = (
        "````carousel\n"
        "![Diagram](/tmp/arch.png)\n"
        "<!-- slide -->\n"
        "```python\n"
        "# <!-- slide --> inside code comment must stay intact\n"
        'example = "![ignored](file:///do/not/touch.png)"\n'
        "```\n"
        "````\n"
    )
    expected = (
        "![Diagram](/tmp/arch.png)\n\n"
        "```python\n"
        "# <!-- slide --> inside code comment must stay intact\n"
        'example = "![ignored](file:///do/not/touch.png)"\n'
        "```\n"
    )
    assert normalize_carousel_markdown(raw) == expected


def test_does_not_unwrap_carousel_inside_outer_markdown_code_fence() -> None:
    """A carousel example inside an outer markdown fence must be preserved verbatim."""
    raw = (
        "`````markdown\n"
        "````carousel\n"
        "![Example](file:///tmp/example.png)\n"
        "<!-- slide -->\n"
        "![Example 2](/tmp/example2.png)\n"
        "````\n"
        "`````\n"
    )
    assert normalize_carousel_markdown(raw) == raw


def test_normalizes_file_uri_images_while_preserving_links_and_inline_code() -> None:
    """Markdown image file:// URIs rewrite to POSIX paths; links and inline code stay intact."""
    raw = (
        "See [source.py](file:///Users/benwart/src/source.py) and "
        "`![literal](file:///Users/benwart/raw.png)`.\n\n"
        "````carousel\n"
        "![Plain](file:///Users/benwart/plain.png)\n"
        "<!-- slide -->\n"
        "![Localhost](file://localhost/Users/benwart/local.png)\n"
        "<!-- slide -->\n"
        "![Spaced](file:///Users/benwart/My%20Screens/shot%20(1).png)\n"
        "````\n"
    )
    expected = (
        "See [source.py](file:///Users/benwart/src/source.py) and "
        "`![literal](file:///Users/benwart/raw.png)`.\n\n"
        "![Plain](/Users/benwart/plain.png)\n\n"
        "![Localhost](/Users/benwart/local.png)\n\n"
        "![Spaced](</Users/benwart/My Screens/shot (1).png>)\n"
    )
    assert normalize_carousel_markdown(raw) == expected


def test_inline_slide_markers_and_surrounding_blank_lines_collapse_cleanly() -> None:
    """Inline <!-- slide --> and extra blank lines around slide markers do not over-pad."""
    raw = (
        "````carousel\n\n"
        "![First](/tmp/1.png)\n\n"
        "<!-- slide -->\n\n"
        "![Second](/tmp/2.png) <!-- slide --> ![Third](/tmp/3.png)\n\n"
        "````\n"
    )
    assert normalize_carousel_markdown(raw) == (
        "![First](/tmp/1.png)\n\n![Second](/tmp/2.png)\n\n![Third](/tmp/3.png)\n"
    )


def test_unclosed_carousel_at_eof_flushes_slides_without_fence_header() -> None:
    """Truncated/cancelled stream inside carousel still renders buffered slides."""
    raw = "````carousel\n![Partial](/tmp/partial.png)\n<!-- slide -->\n![Last](/tmp/last.png)"
    assert normalize_carousel_markdown(raw) == (
        "![Partial](/tmp/partial.png)\n\n![Last](/tmp/last.png)"
    )


def test_stream_normalizer_handles_exact_acp_log_token_splits() -> None:
    """Verify exact split-fence token boundaries observed in live acp.log."""
    chunks = [
        "0` are pushed to `origin/main`.\n\n```",
        "`carousel\n![Synced Scree",
        "n 2](/Users/benwart/.gemini/antigravity-cli/brain/id/screen-2.png)\n<!-- ",
        "slide -->\n![Synced Screen 2C](/Users/benwart/.gemini/antigravity-cli/brain/id/",
        "screen-2c-add-stop-sheet.png)\n````\n\n",
        "```diff\ndiff --git a/ios/Trip.swift b/ios/Trip.swift\n```\n",
    ]
    stream = CarouselStreamNormalizer()
    out_parts = [stream.feed(c) for c in chunks]
    tail = stream.flush()
    combined = "".join(out_parts) + tail

    assert "````carousel" not in combined
    assert "<!-- slide -->" not in combined
    assert combined == normalize_carousel_markdown("".join(chunks))
    # First chunk emits prose before the buffered ` ``` ` fence candidate immediately.
    assert out_parts[0] == "0` are pushed to `origin/main`.\n\n"


def test_stream_normalizer_emits_ordinary_prose_immediately_without_newline() -> None:
    """Non-fence prose without trailing newline streams immediately for low TTFT and steer."""
    stream = CarouselStreamNormalizer()
    assert stream.feed("Initial chunk. ") == "Initial chunk. "
    assert stream.feed("More words.") == "More words."
    assert stream.flush() == ""


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 5, 11])
def test_stream_normalizer_matches_batch_across_arbitrary_chunk_sizes(chunk_size: int) -> None:
    """Arbitrary fixed-size token chunking produces identical output to batch normalization."""
    doc = (
        "Header with `inline` code and [link](file:///tmp/a.py).\r\n\r\n"
        "````carousel\r\n"
        "![A](file:///tmp/a.png)\r\n"
        "<!-- slide -->\r\n"
        "```ts\r\n"
        "const x = `![noop](file:///tmp/noop.png)`;\r\n"
        "```\r\n"
        "<!-- slide -->\r\n"
        "![B](/tmp/b.png)\r\n"
        "````\r\n\r\n"
        "Footer ![C](file:///tmp/c.png) done!"
    )
    stream = CarouselStreamNormalizer()
    pieces = [stream.feed(doc[idx : idx + chunk_size]) for idx in range(0, len(doc), chunk_size)]
    pieces.append(stream.flush())
    result = "".join(pieces)

    expected = normalize_carousel_markdown(doc)
    assert result == expected
    assert normalize_carousel_markdown(result) == expected


def test_flush_prose_before_boundary_emits_inline_code_line_and_keeps_midline_state() -> None:
    """Flushing before a tool boundary emits backtick prose without treating suffix as col 0."""
    stream = CarouselStreamNormalizer()
    assert stream.feed("Checking `config.py`:") == ""
    assert stream.flush_prose_before_boundary() == "Checking `config.py`:"
    # Appending ```carousel on the SAME line (no newline yet) must NOT open a carousel fence!
    out = stream.feed(" ```carousel\n````carousel\n![Img](/tmp/x.png)\n````\n") + stream.flush()
    assert out == " ```carousel\n\n![Img](/tmp/x.png)\n"
