"""Normalize Antigravity ``carousel`` Markdown blocks into standard GFM Markdown.

Antigravity (`agy`) models emit ```` ````carousel ```` fenced blocks separated by
`<!-- slide -->` comments containing Markdown image embeds (`![alt](/abs/path.png)`).
Standard CommonMark/GFM renderers (such as KiroCrew's `MarkdownRenderer`) treat
`carousel` as an unknown code-fence language and display raw Markdown source lines
instead of rendering images.

This module provides both a stateful streaming normalizer (`CarouselStreamNormalizer`)
for live token-delta chunks and a pure batch helper (`normalize_carousel_markdown`)
for session history replay.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import unquote

_OPENING_FENCE_RE = re.compile(r"^[ ]{0,3}(`{3,}|~{3,})(.*)$")
_FENCE_PREFIX_CANDIDATE_RE = re.compile(
    r"^[ ]{0,3}(?:`+|~+|`{3,}[^`]*|~{3,}.*)?$",
    re.DOTALL,
)
_SLIDE_SPLIT_RE = re.compile(r"[ \t]*<!--\s*slide\s*-->[ \t]*", re.IGNORECASE)
_IMAGE_FILE_URI_RE = re.compile(
    r"!\[([^\]]*)\]\("
    r"(?:<(file://[^>\r\n]+)>|(file://(?:[^\s()\r\n]+|\([^\s()\r\n]*\))+))"
    r"\)",
    re.IGNORECASE,
)
_WINDOWS_DRIVE_URI_PATH_RE = re.compile(r"^/[A-Za-z]:[\\/]")
_TRAILING_BLANK_LINES_RE = re.compile(r"(?:[ \t]*\r?\n)+$")
_NEEDS_ANGLE_BRACKETS_RE = re.compile(r"[\s()<>]")


@dataclass(frozen=True, slots=True)
class _FenceSpec:
    """CommonMark fenced code block delimiter specification."""

    char: str
    length: int


def _parse_opening_fence(line_body: str) -> tuple[_FenceSpec, str] | None:
    """Parse a CommonMark opening code fence (`0..3` spaces + `>=3` backticks/tildes)."""
    match = _OPENING_FENCE_RE.match(line_body)
    if match is None:
        return None
    fence_run, info_raw = match.group(1), match.group(2)
    fence_char = fence_run[0]
    if fence_char == "`" and "`" in info_raw:
        return None
    return _FenceSpec(char=fence_char, length=len(fence_run)), info_raw.strip()


def _is_closing_fence(line_body: str, spec: _FenceSpec) -> bool:
    """Return True when `line_body` closes `spec` per CommonMark §4.5."""
    stripped_leading = line_body.lstrip(" ")
    if len(line_body) - len(stripped_leading) > 3:
        return False
    run_len = len(stripped_leading) - len(stripped_leading.lstrip(spec.char))
    if run_len < spec.length:
        return False
    remainder = stripped_leading[run_len:]
    return remainder.strip(" \t") == ""


def _is_carousel_info(info: str) -> bool:
    """Return True when a code-fence info string designates a `carousel` block."""
    tokens = info.split(None, 1)
    return bool(tokens) and tokens[0].lower() == "carousel"


def _is_fence_prefix_candidate(line_prefix: str) -> bool:
    """Return True if `line_prefix` (without `\\n`) could still form an opening fence."""
    trimmed = line_prefix[:-1] if line_prefix.endswith("\r") else line_prefix
    return _FENCE_PREFIX_CANDIDATE_RE.match(trimmed) is not None


def _file_uri_to_local_path(raw_uri: str) -> str | None:
    """Convert a `file://` URI into a decoded local filesystem path."""
    if not raw_uri.lower().startswith("file://"):
        return None
    rest = raw_uri[len("file://") :]
    if rest.lower().startswith("localhost/"):
        rest = rest[len("localhost") :]
    if not rest.startswith("/"):
        return None
    decoded = unquote(rest)
    if _WINDOWS_DRIVE_URI_PATH_RE.match(decoded):
        decoded = decoded[1:]
    return decoded or None


def _rewrite_file_uri_images_in_segment(segment: str) -> str:
    """Rewrite `![alt](file:///...)` embeds within a non-code Markdown segment."""

    def _replace(match: re.Match[str]) -> str:
        alt = match.group(1)
        raw_uri = match.group(2) or match.group(3) or ""
        local_path = _file_uri_to_local_path(raw_uri)
        if local_path is None:
            return match.group(0)
        if _NEEDS_ANGLE_BRACKETS_RE.search(local_path):
            return f"![{alt}](<{local_path}>)"
        return f"![{alt}]({local_path})"

    return _IMAGE_FILE_URI_RE.sub(_replace, segment)


def _normalize_markdown_image_file_uris(line_body: str) -> str:
    """Rewrite `file://` image embeds outside inline backtick code spans."""
    if "![" not in line_body or "file://" not in line_body.lower():
        return line_body
    if "`" not in line_body:
        return _rewrite_file_uri_images_in_segment(line_body)

    out: list[str] = []
    plain_start = 0
    idx = 0
    length = len(line_body)
    while idx < length:
        if line_body[idx] != "`":
            idx += 1
            continue
        tick_end = idx
        while tick_end < length and line_body[tick_end] == "`":
            tick_end += 1
        tick_len = tick_end - idx
        close_idx = _find_matching_backtick_run(line_body, tick_end, tick_len)
        if close_idx < 0:
            idx = tick_end
            continue
        if idx > plain_start:
            out.append(_rewrite_file_uri_images_in_segment(line_body[plain_start:idx]))
        out.append(line_body[idx : close_idx + tick_len])
        idx = close_idx + tick_len
        plain_start = idx

    if plain_start < length:
        out.append(_rewrite_file_uri_images_in_segment(line_body[plain_start:]))
    return "".join(out)


def _find_matching_backtick_run(text: str, start: int, target_len: int) -> int:
    """Find the index of the next exact `target_len` backtick run in `text[start:]`."""
    idx = start
    length = len(text)
    while idx < length:
        if text[idx] != "`":
            idx += 1
            continue
        run_end = idx
        while run_end < length and text[run_end] == "`":
            run_end += 1
        if run_end - idx == target_len:
            return idx
        idx = run_end
    return -1


def _split_line_ending(raw_line: str) -> tuple[str, str]:
    """Split `raw_line` into `(body, ending)` preserving `\\r\\n` or `\\n`."""
    if raw_line.endswith("\r\n"):
        return raw_line[:-2], "\r\n"
    if raw_line.endswith("\n"):
        return raw_line[:-1], "\n"
    return raw_line, ""


def _updated_trailing_newlines(previous: int, piece: str) -> int:
    """Compute consecutive trailing line-break count after emitting `piece`."""
    match = _TRAILING_BLANK_LINES_RE.search(piece)
    if match is None:
        return 0
    line_breaks = match.group(0).count("\n")
    if match.start() == 0:
        return previous + line_breaks
    return line_breaks


class CarouselStreamNormalizer:
    """Stateful streaming normalizer that unwraps ``carousel`` Markdown fences."""

    def __init__(self) -> None:
        self._outer_code_fence: _FenceSpec | None = None
        self._carousel_fence: _FenceSpec | None = None
        self._nested_code_fence: _FenceSpec | None = None
        self._line_buf: str = ""
        self._at_line_start: bool = True
        self._emitted_any: bool = False
        self._trailing_newlines: int = 0
        self._need_blank_line: bool = False
        self._newline_seq: str = "\n"

    def feed(self, chunk: str) -> str:
        """Consume a streamed text delta and return normalized Markdown ready to emit."""
        if not chunk:
            return ""
        self._line_buf += chunk
        emitted_parts: list[str] = []
        while "\n" in self._line_buf:
            nl_pos = self._line_buf.index("\n") + 1
            raw_line = self._line_buf[:nl_pos]
            self._line_buf = self._line_buf[nl_pos:]
            piece = self._process_complete_line(raw_line)
            if piece:
                emitted_parts.append(piece)

        partial = self._flush_safe_outside_prefix()
        if partial:
            emitted_parts.append(partial)
        return "".join(emitted_parts)

    def flush(self) -> str:
        """Flush any buffered partial line at end-of-stream or turn boundary."""
        if not self._line_buf:
            return ""
        trailing_line = self._line_buf
        self._line_buf = ""
        return self._process_complete_line(trailing_line)

    def flush_prose_before_boundary(self) -> str:
        """Flush buffered `OUTSIDE` non-fence prose before a tool-call or steer boundary."""
        if self._outer_code_fence is not None or self._carousel_fence is not None:
            return ""
        if not self._line_buf:
            return ""
        if self._at_line_start and _is_fence_prefix_candidate(self._line_buf):
            return ""
        if self._at_line_start and self._need_blank_line and self._line_buf.strip(" \t\r") == "":
            return ""
        trailing_line = self._line_buf
        self._line_buf = ""
        return self._process_complete_line(trailing_line)

    def _emit_raw(self, text: str) -> str:
        """Record emission metadata and return `text` unchanged."""
        if not text:
            return ""
        self._emitted_any = True
        self._trailing_newlines = _updated_trailing_newlines(self._trailing_newlines, text)
        return text

    def _emit_content(self, text: str) -> str:
        """Emit non-blank content, inserting blank-line separation when pending."""
        if not text:
            return ""
        prefix = ""
        if self._need_blank_line:
            if self._emitted_any and self._trailing_newlines < 2:
                missing = 2 - self._trailing_newlines
                prefix = self._newline_seq * missing
            self._need_blank_line = False
        return self._emit_raw(prefix + text)

    def _emit_outside_blank_line(self, raw_line: str) -> str:
        """Emit a blank line outside fences while coalescing post-carousel breaks."""
        if self._need_blank_line and not self._emitted_any:
            return ""
        out = self._emit_raw(raw_line)
        if self._need_blank_line and self._trailing_newlines >= 2:
            self._need_blank_line = False
        return out

    def _process_complete_line(self, raw_line: str) -> str:
        """Process one complete line (or final EOF remainder)."""
        line_body, line_ending = _split_line_ending(raw_line)
        if line_ending:
            self._newline_seq = line_ending

        was_partial = not self._at_line_start
        self._at_line_start = bool(line_ending)
        if was_partial:
            normalized = _normalize_markdown_image_file_uris(line_body)
            return self._emit_raw(normalized + line_ending)

        if self._outer_code_fence is not None:
            if _is_closing_fence(line_body, self._outer_code_fence):
                self._outer_code_fence = None
            return self._emit_raw(raw_line)

        if self._carousel_fence is not None:
            return self._process_carousel_line(raw_line, line_body, line_ending)

        opening = _parse_opening_fence(line_body)
        if opening is not None:
            spec, info = opening
            if _is_carousel_info(info):
                self._carousel_fence = spec
                self._nested_code_fence = None
                self._need_blank_line = True
                return ""
            self._outer_code_fence = spec
            return self._emit_content(raw_line)

        if line_body.strip(" \t") == "":
            return self._emit_outside_blank_line(raw_line)

        normalized = _normalize_markdown_image_file_uris(line_body)
        return self._emit_content(normalized + line_ending)

    def _process_carousel_line(
        self,
        raw_line: str,
        line_body: str,
        line_ending: str,
    ) -> str:
        """Process a line inside an active ``carousel`` block."""
        assert self._carousel_fence is not None
        if self._nested_code_fence is not None:
            if _is_closing_fence(line_body, self._nested_code_fence):
                self._nested_code_fence = None
            return self._emit_raw(raw_line)

        if _is_closing_fence(line_body, self._carousel_fence):
            self._carousel_fence = None
            self._need_blank_line = True
            return ""

        opening = _parse_opening_fence(line_body)
        if opening is not None:
            self._nested_code_fence = opening[0]
            return self._emit_content(raw_line)

        parts = _SLIDE_SPLIT_RE.split(line_body)
        if len(parts) == 1:
            if line_body.strip(" \t") == "":
                self._need_blank_line = True
                return ""
            normalized = _normalize_markdown_image_file_uris(line_body)
            return self._emit_content(normalized + line_ending)

        emitted: list[str] = []
        last_idx = len(parts) - 1
        for idx, part in enumerate(parts):
            if idx > 0:
                self._need_blank_line = True
            if part.strip(" \t") == "":
                self._need_blank_line = True
                continue
            suffix = self._newline_seq if idx < last_idx else line_ending
            normalized = _normalize_markdown_image_file_uris(part)
            emitted.append(self._emit_content(normalized + suffix))
        return "".join(emitted)

    def _flush_safe_outside_prefix(self) -> str:
        """Immediately emit plain non-fence `OUTSIDE` prose before `\\n` arrives."""
        if self._outer_code_fence is not None or self._carousel_fence is not None:
            return ""
        if not self._line_buf:
            return ""
        if self._at_line_start:
            if _is_fence_prefix_candidate(self._line_buf):
                return ""
            if self._need_blank_line and self._line_buf.strip(" \t\r") == "":
                return ""
        if "`" in self._line_buf or "![" in self._line_buf or "\r" in self._line_buf:
            return ""

        if self._line_buf.endswith("!"):
            safe = self._line_buf[:-1]
            self._line_buf = "!"
        else:
            safe = self._line_buf
            self._line_buf = ""

        if not safe:
            return ""
        if self._at_line_start:
            self._at_line_start = False
            return self._emit_content(safe)
        return self._emit_raw(safe)


def normalize_carousel_markdown(text: str) -> str:
    """Normalize ``carousel`` fenced blocks and `file://` image URIs in `text`."""
    if not text:
        return ""
    normalizer = CarouselStreamNormalizer()
    return normalizer.feed(text) + normalizer.flush()
