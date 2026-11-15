"""Filter transient `agy acp` upstream API retry banners from assistant text streams.

When `agy acp` hits a transient HTTP/transport error talking to Cloud Code before
retrying inside the same `session/prompt` turn, the Go binary formats
`"\\n⚠ API error (attempt %d): %v\\n"` and emits it as an `agent_message_chunk`.

This module provides a pure stateful stream filter (`RetryWarningStreamFilter`)
for live token chunks and a pure batch helper (`strip_agy_retry_warnings`) for
persisted/replayed turn sanitization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

_WARNING_CHAR = "\u26a0"
_VARIATION_SELECTOR = "\ufe0f"
_BANNER_HEAD = " API error (attempt "
_BANNER_CLOSE = "):"

type _PrefixVerdict = Literal["need_more", "matched_banner_prefix", "not_banner"]


@dataclass(frozen=True, slots=True)
class RetryFilterStep:
    """Result of feeding or flushing `RetryWarningStreamFilter`."""

    text: str = ""
    warnings: tuple[str, ...] = ()


_EMPTY_STEP = RetryFilterStep()


def _classify_banner_prefix(line_prefix: str) -> _PrefixVerdict:
    """Classify a column-0 `line_prefix` (without `\\n`) against the `agy` retry banner."""
    if not line_prefix:
        return "need_more"
    if line_prefix[0] != _WARNING_CHAR:
        return "not_banner"
    if line_prefix in (_WARNING_CHAR, _WARNING_CHAR + _VARIATION_SELECTOR):
        return "need_more"

    rest = (
        line_prefix[2:]
        if line_prefix.startswith(_WARNING_CHAR + _VARIATION_SELECTOR)
        else line_prefix[1:]
    )
    if len(rest) <= len(_BANNER_HEAD):
        return "need_more" if _BANNER_HEAD.startswith(rest) else "not_banner"
    if not rest.startswith(_BANNER_HEAD):
        return "not_banner"

    after_head = rest[len(_BANNER_HEAD) :]
    digits_len = 0
    for ch in after_head:
        if "0" <= ch <= "9":
            digits_len += 1
        else:
            break
    if digits_len == 0:
        return "not_banner"

    after_digits = after_head[digits_len:]
    if not after_digits:
        return "need_more"
    if len(after_digits) < len(_BANNER_CLOSE):
        return "need_more" if _BANNER_CLOSE.startswith(after_digits) else "not_banner"
    return "matched_banner_prefix" if after_digits.startswith(_BANNER_CLOSE) else "not_banner"


class RetryWarningStreamFilter:
    """Stateful stream filter stripping `⚠ API error (attempt <N>): ...` lines."""

    def __init__(self) -> None:
        self._at_line_start: bool = True
        self._pending_blank_nl: str = ""
        self._buf: str = ""

    def feed(self, chunk: str) -> RetryFilterStep:
        """Consume a streamed text delta and return sanitized text plus stripped warnings."""
        if not chunk:
            return _EMPTY_STEP

        emitted: list[str] = []
        warnings: list[str] = []
        pos = 0
        length = len(chunk)

        while pos < length:
            if not self._at_line_start:
                nl_idx = chunk.find("\n", pos)
                if nl_idx == -1:
                    emitted.append(chunk[pos:])
                    break
                emitted.append(chunk[pos : nl_idx + 1])
                self._at_line_start = True
                pos = nl_idx + 1
                continue

            ch = chunk[pos]
            pos += 1
            self._consume_line_start_char(ch, emitted, warnings)

        if not emitted and not warnings:
            return _EMPTY_STEP
        return RetryFilterStep(text="".join(emitted), warnings=tuple(warnings))

    def flush_before_boundary(self) -> RetryFilterStep:
        """Flush buffered text before a tool-call or steer boundary and reset line-start."""
        step = self.flush()
        self._at_line_start = True
        return step

    def flush(self) -> RetryFilterStep:
        """Flush any buffered line prefix or blank newline at end-of-stream."""
        if not self._at_line_start:
            return _EMPTY_STEP

        if not self._buf:
            if not self._pending_blank_nl:
                return _EMPTY_STEP
            out = self._pending_blank_nl
            self._pending_blank_nl = ""
            return RetryFilterStep(text=out)

        line_body = self._buf[:-1] if self._buf.endswith("\r") else self._buf
        if _classify_banner_prefix(line_body) == "matched_banner_prefix":
            self._pending_blank_nl = ""
            self._buf = ""
            return RetryFilterStep(warnings=(line_body,))

        out = self._pending_blank_nl + self._buf
        self._pending_blank_nl = ""
        self._buf = ""
        self._at_line_start = False
        return RetryFilterStep(text=out)

    def _consume_line_start_char(
        self,
        ch: str,
        emitted: list[str],
        warnings: list[str],
    ) -> None:
        if not self._buf:
            if ch == "\n":
                self._stash_blank_newline("\n", emitted)
                return
            if ch == "\r":
                self._buf = "\r"
                return

        elif self._buf == "\r":
            if ch == "\n":
                self._buf = ""
                self._stash_blank_newline("\r\n", emitted)
                return
            emitted.append(self._pending_blank_nl + "\r" + ch)
            self._pending_blank_nl = ""
            self._buf = ""
            self._at_line_start = False
            return

        if ch == "\n":
            line_body = self._buf[:-1] if self._buf.endswith("\r") else self._buf
            if _classify_banner_prefix(line_body) == "matched_banner_prefix":
                warnings.append(line_body)
            else:
                emitted.append(self._pending_blank_nl + self._buf + "\n")
            self._pending_blank_nl = ""
            self._buf = ""
            return

        self._buf += ch
        if _classify_banner_prefix(self._buf) == "not_banner":
            emitted.append(self._pending_blank_nl + self._buf)
            self._pending_blank_nl = ""
            self._buf = ""
            self._at_line_start = False

    def _stash_blank_newline(self, nl: str, emitted: list[str]) -> None:
        if self._pending_blank_nl:
            emitted.append(self._pending_blank_nl)
        self._pending_blank_nl = nl


def strip_agy_retry_warnings(text: str) -> str:
    """Strip transient `⚠ API error (attempt <N>): ...` banners from `text`."""
    if not text or _WARNING_CHAR not in text:
        return text
    stream = RetryWarningStreamFilter()
    return stream.feed(text).text + stream.flush().text
