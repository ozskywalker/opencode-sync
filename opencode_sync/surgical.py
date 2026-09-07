"""Surgical editing: rewrite only the spans a plan actually needs to touch.

Everything here is text -> text.  Given the original config source and the
plans to apply, splices replacement text into the original so comments,
indentation, and every unrelated byte survive exactly.  Two independent
guards (the fixpoint loop and _verify_edit) stop a bad edit from reaching
disk, and neither ever falls back to re-serializing: silently destroying a
user's comments is worse than failing loudly.
"""

from __future__ import annotations

import json
from typing import Dict, List, Sequence

from .io import parse_jsonc
from .jsonc_edit import (
    JsoncEditError,
    RenderEntry,
    Span,
    apply_edits,
    detect_base_indent,
    detect_unit,
    find_object_members,
    find_value_span,
    mask_comments,
    mask_trailing_commas,
    render_object,
)
from .planning import ProviderPlan, apply_plan, generate_display_name


# ---------------------------------------------------------------------------
# Surgical editing
#
# Rewrites only the spans a plan actually needs to touch, so comments,
# indentation, and every unrelated byte survive exactly.
# ---------------------------------------------------------------------------

def _member_at(masked: str, path: Sequence[str]):
    """Find the key+value member at ``path``, or None if any key is absent."""
    if not path:
        return None
    parent = find_value_span(masked, path[:-1])
    if parent is None or masked[parent.start] != "{":
        return None
    match = None
    for member in find_object_members(masked, parent).members:
        if member.key == path[-1]:
            match = member  # last wins, matching json.loads
    return match


def _render_value(value, base: str, unit: str) -> str:
    """Serialize a new value, indented to sit at ``base + unit`` depth."""
    text = json.dumps(value, indent=unit, ensure_ascii=False)
    lines = text.split("\n")
    return lines[0] + "".join("\n" + base + unit + line for line in lines[1:])


def _rebuild_object_edit(text: str, masked: str, member, entries: List[RenderEntry]):
    """An edit that replaces a member's object value with ``entries``."""
    base = detect_base_indent(text, member.key_span.start)
    body = find_object_members(masked, member.value_span)
    first = body.members[0].key_span.start if body.members else None
    unit = detect_unit(text, base, first)
    block_leading = body.block_leading.of(text) or "\n"
    body_trailing = body.body_trailing.of(text)
    return member.value_span, render_object(base, unit, block_leading, entries, body_trailing)


def _existing_entries(text: str, masked: str, obj_span: Span) -> Dict[str, RenderEntry]:
    """Existing members as render entries carrying their original source text."""
    body = find_object_members(masked, obj_span)
    return {
        m.key: RenderEntry(
            leading=m.leading.of(text),
            key=m.key_span.of(text),
            value=m.value_span.of(text),
            trailing=m.trailing.of(text),
        )
        for m in body.members
    }


def _models_edit(text: str, masked: str, plan: ProviderPlan):
    member = _member_at(masked, ["provider", plan.provider_id, "models"])
    if member is None:
        return None

    base = detect_base_indent(text, member.key_span.start)
    body = find_object_members(masked, member.value_span)
    unit = detect_unit(text, base, body.members[0].key_span.start if body.members else None)

    existing = _existing_entries(text, masked, member.value_span)
    source = {new: old for old, new in plan.renames.items()}  # new ID -> the entry it came from

    entries: List[RenderEntry] = []
    for mid in plan.model_ids:
        origin = existing.get(source.get(mid, mid))
        if origin is not None:
            value = origin.value
            if mid in plan.renames.values():
                # Renamed entry: the body rides along (settings, comments), but the
                # display name described the old ID, so swap in a fresh one.
                value = _renamed_entry_text(origin, mid)
            # Re-splice the original source text, so comments *inside* the entry ride along.
            entries.append(
                RenderEntry(origin.leading, json.dumps(mid), value, origin.trailing)
            )
        else:
            entries.append(
                RenderEntry(
                    "",
                    json.dumps(mid),
                    _render_value({"name": generate_display_name(mid)}, base, unit),
                    "",
                )
            )
    return _rebuild_object_edit(text, masked, member, entries)


def _renamed_entry_text(origin: RenderEntry, new_id: str) -> str:
    """The original entry's source text with only its "name" member rewritten.

    Splicing the original body keeps comments and every other key byte-identical;
    the name member is replaced with a generated display name for the new ID.
    An entry that had no explicit ``name`` key keeps having none, and a body that
    cannot be scanned falls back to the same re-render the dict path applies.
    """
    entry_text = origin.value
    entry_masked = mask_trailing_commas(mask_comments(entry_text))
    if not entry_masked.lstrip().startswith("{"):
        # Not an object body; the dict path replaced it wholesale, mirror that.
        return _render_value({"name": generate_display_name(new_id)}, "", "  ")
    try:
        obj = find_object_members(
            entry_masked, Span(entry_masked.index("{"), entry_masked.rindex("}") + 1)
        )
    except JsoncEditError:
        return _render_value({"name": generate_display_name(new_id)}, "", "  ")
    name_member = None
    for m in obj.members:
        if m.key == "name":
            name_member = m  # last wins, matching json.loads
    if name_member is None:
        return entry_text  # no explicit "name" key — nothing stale to fix
    fresh = _render_value(generate_display_name(new_id), "", "  ")
    return entry_text[: name_member.value_span.start] + fresh + entry_text[name_member.value_span.end :]


def _append_member_edit(text: str, masked: str, parent_path: Sequence[str], key: str, value):
    """An edit that appends ``key: value`` to the object at ``parent_path``.

    Rebuilds the parent object, but every pre-existing member is re-emitted from its
    own source text, so only the new member is genuinely new.
    """
    if parent_path:
        member = _member_at(masked, parent_path)
        if member is None:
            return None
        obj_span, key_start = member.value_span, member.key_span.start
    else:
        obj_span = find_value_span(masked, [])
        key_start = obj_span.start

    if masked[obj_span.start] != "{":
        return None

    base = detect_base_indent(text, key_start)
    body = find_object_members(masked, obj_span)
    unit = detect_unit(text, base, body.members[0].key_span.start if body.members else None)

    entries = list(_existing_entries(text, masked, obj_span).values())
    entries.append(RenderEntry("", json.dumps(key), _render_value(value, base, unit), ""))
    block_leading = body.block_leading.of(text) or "\n"
    body_trailing = body.body_trailing.of(text)
    return obj_span, render_object(base, unit, block_leading, entries, body_trailing)


def _ordered(value) -> str:
    """Order-sensitive rendering, for comparing "is this bit already right?"."""
    return json.dumps(value, ensure_ascii=False)


def _next_edit(text: str, masked: str, current: dict, desired: dict, plan: ProviderPlan):
    """Return the next single edit needed to move ``text`` toward ``desired``, or None.

    One edit at a time, re-scanned each round, because edits can nest: inserting an
    absent ``options`` key rebuilds the provider object, which *contains* the models
    span we may also need to rewrite.  Splicing both against the same offsets would
    have the outer rebuild silently swallow the inner one.
    """
    pid = plan.provider_id
    want_provider = desired.get("provider", {}).get(pid, {})

    if _member_at(masked, ["provider", pid]) is None:
        edit = _append_member_edit(text, masked, ["provider"], pid, want_provider)
        if edit is None:
            edit = _append_member_edit(text, masked, [], "provider", {pid: want_provider})
        if edit is None:
            raise JsoncEditError(f"cannot add provider {pid!r} to this config")
        return edit

    have_provider = current.get("provider", {}).get(pid, {})
    want_models = want_provider.get("models", {})

    if _member_at(masked, ["provider", pid, "models"]) is None:
        return _append_member_edit(text, masked, ["provider", pid], "models", want_models)

    if _ordered(have_provider.get("models", {})) != _ordered(want_models):
        edit = _models_edit(text, masked, plan)
        if edit is None:
            raise JsoncEditError(f"cannot locate models for provider {pid!r}")
        return edit

    want_url = want_provider.get("options", {}).get("baseURL")
    if want_url is not None and have_provider.get("options", {}).get("baseURL") != want_url:
        member = _member_at(masked, ["provider", pid, "options", "baseURL"])
        if member is not None:
            return member.value_span, json.dumps(want_url)
        edit = _append_member_edit(text, masked, ["provider", pid, "options"], "baseURL", want_url)
        if edit is None:
            edit = _append_member_edit(text, masked, ["provider", pid], "options",
                                       {"baseURL": want_url})
        if edit is None:
            raise JsoncEditError(f"cannot set baseURL for provider {pid!r}")
        return edit

    for key, value in plan.model_key_updates.items():
        if current.get(key) == value:
            continue
        member = _member_at(masked, [key])
        if member is not None:
            return member.value_span, json.dumps(value)
        # Planner modes "auto"/"explicit" may invent a pointer the file never had.
        # _append_member_edit re-emits every existing member from its own source
        # text, so adding the key costs nothing else in the object.
        edit = _append_member_edit(text, masked, [], key, value)
        if edit is None:
            raise JsoncEditError(f"cannot add top-level {key!r} to this config")
        return edit

    return None


_MAX_EDIT_ROUNDS = 8


def _dominant_eol(text: str) -> str:
    """The file's line-ending convention, judging by the EOLs actually present.

    CRLF wins on a majority (or a tie with any CRLF present): mixed files that
    are mostly CRLF were authored on Windows, and the reverse holds too.  The
    scanner treats both as whitespace, so this only ever shapes *inserted* text.
    """
    crlf = text.count("\r\n")
    lf = text.count("\n") - crlf
    return "\r\n" if crlf > 0 and crlf >= lf else "\n"


def _to_file_eol(text: str, replacement: str) -> str:
    """Render replacement text with the file's line endings.

    Producers (render_object, _render_value) join with "\n", but replacement
    text is not purely produced: surviving entries are re-spliced from their
    *source* text, so an edit can already contain the file's CRLFs.  Translating
    blindly would turn every source "\r\n" into "\r\r\n".  So only newlines that
    are not already CRLF-terminated are converted, and a lone "\r" is left
    alone (it belongs to the following "\n").  On an LF-dominant file nothing
    is touched at all.
    """
    eol = _dominant_eol(text)
    if eol == "\n" or "\n" not in replacement:
        return replacement
    out: List[str] = []
    lines = replacement.split("\n")
    for i, line in enumerate(lines[:-1]):
        if line.endswith("\r"):
            out.append(line + "\n")  # already CRLF-terminated; keep as-is
        else:
            out.append(line + eol)
    out.append(lines[-1])
    return "".join(out)


def _apply_one_plan(text: str, plan: ProviderPlan) -> str:
    current = parse_jsonc(text)
    desired = apply_plan(current, plan)

    for _ in range(_MAX_EDIT_ROUNDS):
        masked = mask_comments(text)
        edit = _next_edit(text, masked, current, desired, plan)
        if edit is None:
            return text
        span, replacement = edit
        text = apply_edits(text, [(span, _to_file_eol(text, replacement))])
        current = parse_jsonc(text)

    raise JsoncEditError(f"edits for provider {plan.provider_id!r} did not converge")


def apply_plans_to_text(text: str, config: dict, plans: Sequence[ProviderPlan]) -> str:
    """Apply every plan to the config source text, rewriting only what changed."""
    for plan in plans:
        text = _apply_one_plan(text, plan)
    _verify_edit(text, config, plans)
    return text


def _verify_edit(new_text: str, config: dict, plans: Sequence[ProviderPlan]) -> None:
    """Prove the surgical result means exactly what the plain dict path would have meant.

    json.dumps rather than == so the comparison is order-sensitive: model order is
    meaningful in opencode, and dict equality ignores it.  On mismatch we raise and
    write nothing — never fall back to re-serializing, since silently destroying the
    user's comments is worse than failing loudly.
    """
    expected = config
    for plan in plans:
        expected = apply_plan(expected, plan)

    try:
        actual = parse_jsonc(new_text)
    except ValueError as e:
        raise JsoncEditError(f"edit produced invalid JSONC: {e}") from e

    if json.dumps(actual, ensure_ascii=False) != json.dumps(expected, ensure_ascii=False):
        raise JsoncEditError(
            "surgical edit changed the config's meaning — refusing to write. "
            "This is a bug in opencode-sync; please report it."
        )
