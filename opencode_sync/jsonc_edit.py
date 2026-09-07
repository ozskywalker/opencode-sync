"""Position-preserving JSONC scanning and text splicing.

Pure text -> text.  No file I/O, no opencode domain knowledge, stdlib only.

The central trick: comments are *masked to spaces* rather than deleted, so every
offset in the masked text maps 1:1 onto the original text.  A plain JSON scanner
can then locate the span of any value, and callers splice replacement text into
the *original*, leaving every byte outside the span untouched.

A pleasant consequence of masking rather than stripping: comments are whitespace
by the time the scanner runs, so the scanner needs no comment awareness at all.

Offsets are ``str`` indices (code points), never bytes.  Masking replaces one code
point with one space, so the mapping survives non-ASCII text.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple


class JsoncEditError(Exception):
    """Raised when text cannot be scanned or an edit cannot be applied safely."""


@dataclass(frozen=True)
class Span:
    """A half-open [start, end) range of str indices."""

    start: int
    end: int

    def of(self, text: str) -> str:
        return text[self.start : self.end]

    def is_empty(self) -> bool:
        return self.start >= self.end


@dataclass(frozen=True)
class Member:
    """One ``"key": value`` pair inside an object, with its attached comments."""

    key: str
    key_span: Span      # includes the surrounding quotes
    value_span: Span
    leading: Span       # own-line comment lines above the key
    trailing: Span      # same-line comment after this member
    comma: Optional[int]  # index of the separator comma, if present


@dataclass(frozen=True)
class ObjectBody:
    span: Span            # the whole {...}
    members: List[Member]
    block_leading: Span   # between '{' and the first member's line
    body_trailing: Span   # between the last member's trailing zone and the closing '}'


@dataclass(frozen=True)
class RenderEntry:
    leading: str    # own-line comment lines, newline-terminated, or ""
    key: str        # JSON-encoded key, including quotes
    value: str      # raw value text
    trailing: str   # same-line comment, or ""


_WS = " \t\r\n"
_DELIMS = ",}]"


# ---------------------------------------------------------------------------
# Comment and trailing-comma masking
# ---------------------------------------------------------------------------

def find_comment_spans(text: str) -> List[Span]:
    """Locate // and /* */ comment spans, ignoring anything inside string literals.

    A hand-rolled state machine, not a regex: only tracking string context can tell
    a comment from the // in a URL like "http://host:8080/v1".
    """
    spans: List[Span] = []
    i = 0
    n = len(text)
    in_string = False

    while i < n:
        c = text[i]
        if in_string:
            if c == "\\" and i + 1 < n:
                i += 2
            elif c == '"':
                in_string = False
                i += 1
            else:
                i += 1
        else:
            if c == '"':
                in_string = True
                i += 1
            elif c == "/" and i + 1 < n and text[i + 1] == "/":
                start = i
                while i < n and text[i] != "\n":
                    i += 1
                spans.append(Span(start, i))  # the newline itself is not part of the comment
            elif c == "/" and i + 1 < n and text[i + 1] == "*":
                start = i
                i += 2
                while i < n - 1:
                    if text[i] == "*" and text[i + 1] == "/":
                        i += 2
                        break
                    i += 1
                else:
                    i = n  # unterminated block comment — consume to EOF
                spans.append(Span(start, i))
            else:
                i += 1

    return spans


def mask_comments(text: str) -> str:
    """Replace comment characters with spaces, preserving length and newlines.

    Newlines inside block comments are kept so that JSONDecodeError line numbers
    still point at the right line of the user's file.
    """
    if not text:
        return text
    chars = list(text)
    for span in find_comment_spans(text):
        for i in range(span.start, span.end):
            if chars[i] != "\n":
                chars[i] = " "
    return "".join(chars)


def mask_trailing_commas(masked: str) -> str:
    """Blank out commas that precede a closing brace/bracket, preserving length.

    Must run on comment-masked text: comments are spaces by then, so the lookahead
    skips them as ordinary whitespace.  String-aware, because a literal ", }" inside
    a string value is data, not syntax.
    """
    chars = list(masked)
    i = 0
    n = len(masked)
    in_string = False

    while i < n:
        c = masked[i]
        if in_string:
            if c == "\\" and i + 1 < n:
                i += 2
                continue
            if c == '"':
                in_string = False
            i += 1
            continue

        if c == '"':
            in_string = True
            i += 1
            continue

        if c == ",":
            j = _skip_ws(masked, i + 1)
            if j < n and masked[j] in "}]":
                chars[i] = " "
        i += 1

    return "".join(chars)


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------

def _skip_ws(s: str, i: int) -> int:
    n = len(s)
    while i < n and s[i] in _WS:
        i += 1
    return i


def _scan_string(s: str, i: int) -> int:
    """Return the index just past the closing quote of the string starting at i."""
    n = len(s)
    if i >= n or s[i] != '"':
        raise JsoncEditError(f"expected '\"' at offset {i}")
    i += 1
    while i < n:
        c = s[i]
        if c == "\\":
            i += 2
            continue
        if c == '"':
            return i + 1
        i += 1
    raise JsoncEditError("unterminated string literal")


def _scan_container(s: str, i: int, close: str) -> int:
    """Scan an object or array body, tolerating trailing commas."""
    n = len(s)
    i += 1  # past the opening brace/bracket
    while True:
        i = _skip_ws(s, i)
        if i >= n:
            raise JsoncEditError(f"unterminated container, expected '{close}'")
        c = s[i]
        if c == close:
            return i + 1
        if c == ",":
            i += 1
            continue
        i = _scan_value(s, i)


def _scan_value(s: str, i: int) -> int:
    """Return the index just past the value starting at i."""
    n = len(s)
    if i >= n:
        raise JsoncEditError("expected a value, found end of input")
    c = s[i]
    if c == '"':
        return _scan_string(s, i)
    if c == "{":
        return _scan_container(s, i, "}")
    if c == "[":
        return _scan_container(s, i, "]")
    if c == ":":
        # An object member's separator: skip it and scan the value that follows.
        return _scan_value(s, _skip_ws(s, i + 1))
    j = i
    while j < n and s[j] not in _DELIMS and s[j] not in _WS:
        j += 1
    if j == i:
        raise JsoncEditError(f"unexpected character {c!r} at offset {i}")
    return j


def _line_start(s: str, i: int) -> int:
    return s.rfind("\n", 0, i) + 1


def find_object_members(masked: str, obj_span: Span) -> ObjectBody:
    """Decompose an object into members, attributing comments to each."""
    s = masked
    if s[obj_span.start] != "{":
        raise JsoncEditError(f"expected an object at offset {obj_span.start}")

    raw: List[Tuple[int, int, int, int, Optional[int]]] = []
    i = obj_span.start + 1
    end = obj_span.end - 1  # index of the closing '}'

    while True:
        i = _skip_ws(s, i)
        if i >= end:
            break
        if s[i] == ",":
            i += 1
            continue
        key_start = i
        key_end = _scan_string(s, i)
        i = _skip_ws(s, key_end)
        if i >= end or s[i] != ":":
            raise JsoncEditError(f"expected ':' after key at offset {key_end}")
        i = _skip_ws(s, i + 1)
        value_start = i
        value_end = _scan_value(s, i)
        j = _skip_ws(s, value_end)
        comma = j if j < end and s[j] == "," else None
        raw.append((key_start, key_end, value_start, value_end, comma))
        i = comma + 1 if comma is not None else value_end

    members: List[Member] = []
    for idx, (ks, ke, vs, ve, comma) in enumerate(raw):
        if idx == 0:
            leading = Span(ks, ks)  # block_leading covers everything above member 0
        else:
            _, _, _, prev_ve, prev_comma = raw[idx - 1]
            gap = prev_comma + 1 if prev_comma is not None else prev_ve
            first_nl = s.find("\n", gap, ks)
            if first_nl == -1:
                leading = Span(ks, ks)
            else:
                last_nl = s.rfind("\n", first_nl, ks)
                leading = Span(first_nl + 1, last_nl + 1)

        gap = comma + 1 if comma is not None else ve
        limit = raw[idx + 1][0] if idx + 1 < len(raw) else end
        first_nl = s.find("\n", gap, limit)
        trailing = Span(gap, first_nl if first_nl != -1 else limit)

        members.append(
            Member(
                key=json.loads(s[ks:ke]),
                key_span=Span(ks, ke),
                value_span=Span(vs, ve),
                leading=leading,
                trailing=trailing,
                comma=comma,
            )
        )

    if raw:
        line_start = _line_start(s, raw[0][0])
        body_start = obj_span.start + 1
        block_leading = (
            Span(body_start, line_start) if line_start > body_start else Span(body_start, body_start)
        )
        # Comments between the last member's trailing zone and the closing '}'
        # belong to no member: they describe the object itself ("bump these
        # limits when VRAM allows").  The last member's trailing span stops at
        # the first newline after its value, so without a dedicated span this
        # zone would be silently dropped on rebuild.  The block runs from that
        # newline up to the closing brace's line start: the brace's own
        # indentation is re-emitted by render_object (as ``base``), so it must
        # not be part of the block.
        last_comma = raw[-1][4]
        trailing_end = last_comma + 1 if last_comma is not None else raw[-1][3]
        after_last_nl = s.find("\n", trailing_end, end)
        if after_last_nl == -1:
            body_trailing = Span(end, end)
        else:
            body_trailing = Span(after_last_nl, _line_start(s, end))
    else:
        block_leading = Span(obj_span.start + 1, obj_span.start + 1)
        body_trailing = Span(obj_span.start + 1, obj_span.start + 1)

    return ObjectBody(
        span=obj_span, members=members, block_leading=block_leading, body_trailing=body_trailing
    )


def find_root_span(masked: str) -> Span:
    i = _skip_ws(masked, 0)
    if i >= len(masked) or masked[i] != "{":
        raise JsoncEditError("config root is not a JSON object")
    return Span(i, _scan_value(masked, i))


def find_value_span(masked: str, path: Sequence[str]) -> Optional[Span]:
    """Return the span of the value at ``path``, or None if any key is absent."""
    span = find_root_span(masked)
    for key in path:
        if masked[span.start] != "{":
            return None
        body = find_object_members(masked, span)
        match = None
        for member in body.members:
            if member.key == key:
                match = member  # last wins, matching json.loads on duplicate keys
        if match is None:
            return None
        span = match.value_span
    return span


# ---------------------------------------------------------------------------
# Rendering and splicing
# ---------------------------------------------------------------------------

def detect_base_indent(text: str, key_start: int) -> str:
    """Indentation of the line holding the key at ``key_start``."""
    start = _line_start(text, key_start)
    candidate = text[start:key_start]
    return candidate if candidate.strip() == "" else ""


def detect_unit(text: str, base: str, member_start: Optional[int]) -> str:
    """Indent step, inferred from a member's line indent minus ``base``."""
    if member_start is None:
        return "  "
    start = _line_start(text, member_start)
    indent = text[start:member_start]
    if indent.strip() != "" or not indent.startswith(base) or len(indent) <= len(base):
        return "  "
    return indent[len(base) :]


def render_object(
    base: str,
    unit: str,
    block_leading: str,
    entries: List[RenderEntry],
    body_trailing: str = "",
) -> str:
    """Rebuild an object body, re-emitting comments verbatim.

    ``body_trailing`` carries the comments (and whitespace) that sat between the
    last member and the closing brace; it is re-emitted verbatim so a rebuild
    cannot silently drop them.
    """
    if not entries:
        return "{}"
    out = ["{", block_leading]
    last = len(entries) - 1
    for idx, entry in enumerate(entries):
        out.append(entry.leading)
        out.append(base + unit + entry.key + ": " + entry.value)
        if idx != last:
            out.append(",")
        out.append(entry.trailing)
        # A comment-bearing body_trailing starts with the newline that ends
        # the last member's line; let it own that newline instead of doubling it.
        if idx == last and body_trailing and body_trailing.strip():
            continue
        out.append("\n")
    # Comments (and whitespace) that sat between the last member and the
    # closing brace.  Pure-whitespace blocks are dropped: the entry loop above
    # already emits the separating newline, and re-emitting an empty block's
    # newline would open a blank line ahead of the brace.  Comment-bearing
    # blocks are re-emitted verbatim, followed by a fresh newline, so the
    # brace lands on its own properly indented line.
    if body_trailing and body_trailing.strip():
        if not body_trailing.startswith("\n"):
            out.append("\n")
        out.append(body_trailing)
        if not body_trailing.endswith("\n"):
            out.append("\n")
    out.append(base + "}")
    return "".join(out)


def apply_edits(text: str, edits: List[Tuple[Span, str]]) -> str:
    """Splice replacements into ``text``.  Spans must not overlap."""
    ordered = sorted(edits, key=lambda e: e[0].start)
    for (a, _), (b, _) in zip(ordered, ordered[1:]):
        if a.end > b.start:
            raise JsoncEditError(
                f"overlapping edits: [{a.start}, {a.end}) and [{b.start}, {b.end})"
            )
    result = text
    for span, replacement in reversed(ordered):
        result = result[: span.start] + replacement + result[span.end :]
    return result
