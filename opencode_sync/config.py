"""Read, parse, update, and write opencode JSONC config files."""

from __future__ import annotations

import copy
import json
import os
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .jsonc_edit import (
    JsoncEditError,
    RenderEntry,
    Span,
    apply_edits,
    detect_base_indent,
    detect_unit,
    find_comment_spans,
    find_object_members,
    find_value_span,
    mask_comments,
    mask_trailing_commas,
    render_object,
)


# ---------------------------------------------------------------------------
# JSONC parsing
# ---------------------------------------------------------------------------

def _strip_jsonc_comments(text: str) -> str:
    """Remove // and /* */ comments from JSONC text, leaving string literals intact.

    Deletes the spans that jsonc_edit locates.  The surgical writer masks the same
    spans to spaces instead; both share one state machine so they can never drift.
    """
    spans = find_comment_spans(text)
    if not spans:
        return text
    out: List[str] = []
    prev = 0
    for span in spans:
        out.append(text[prev : span.start])
        prev = span.end
    out.append(text[prev:])
    return "".join(out)


def parse_jsonc(text: str) -> dict:
    """Parse a JSONC string (JSON with comments and trailing commas) into a dict."""
    masked = mask_trailing_commas(mask_comments(text))
    return json.loads(masked)


# ---------------------------------------------------------------------------
# Config file paths (platform-aware)
# ---------------------------------------------------------------------------

def _config_candidates() -> List[Path]:
    """Return candidate config paths in priority order for the current platform.

    opencode uses xdg-basedir on all platforms, so the search order is the same
    everywhere: XDG_CONFIG_HOME (if set) then ~/.config.  APPDATA and
    ~/Library/Application Support are NOT part of the documented load order.
    """
    candidates: List[Path] = []

    xdg_config = os.environ.get("XDG_CONFIG_HOME")
    if xdg_config:
        candidates.append(Path(xdg_config) / "opencode" / "opencode.jsonc")

    candidates.append(Path.home() / ".config" / "opencode" / "opencode.jsonc")

    return candidates


def find_config_path() -> Optional[Path]:
    """Return the first existing opencode config path, or the most-likely path if none found."""
    candidates = _config_candidates()
    for path in candidates:
        if path.exists():
            return path
    return candidates[0] if candidates else None


# ---------------------------------------------------------------------------
# Config I/O
# ---------------------------------------------------------------------------

def load_config_text(path: Path) -> str:
    """Read a config file verbatim.

    newline="" disables universal-newline translation, so a CRLF file stays CRLF
    instead of being silently converted to LF on the way back out.
    """
    with path.open("r", encoding="utf-8", newline="") as fh:
        return fh.read()


def load_config(path: Path) -> dict:
    """Load and parse an opencode JSONC config file."""
    return parse_jsonc(load_config_text(path))


def _atomic_write_text(path: Path, text: str, backup: bool = False) -> None:
    """Write text via a temp file + os.replace, so readers never see a torn file.

    The temp file must live in the target directory: os.replace is only atomic
    within a filesystem.  mkstemp creates 0600, so the original's mode has to be
    copied across explicitly or we would silently tighten permissions.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    if backup and path.exists():
        backup_path = path.with_suffix(path.suffix + ".bak")
        with path.open("r", encoding="utf-8", newline="") as src:
            existing = src.read()
        with backup_path.open("w", encoding="utf-8", newline="") as dst:
            dst.write(existing)

    mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else None

    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".opencode-sync-", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        if mode is not None:
            os.chmod(tmp_name, mode)
        os.replace(tmp_name, str(path))
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def save_config_text(path: Path, text: str, backup: bool = False) -> None:
    """Write already-rendered config text to disk."""
    _atomic_write_text(path, text, backup=backup)


def save_config(path: Path, config: dict) -> None:
    """Write a config dict to disk as formatted JSON.

    This is the greenfield path, for configs that do not exist on disk yet and so
    have no comments or formatting to preserve.  Editing an existing file goes
    through apply_plans_to_text instead.
    """
    content = json.dumps(config, indent=2, ensure_ascii=False) + "\n"
    _atomic_write_text(path, content)


# ---------------------------------------------------------------------------
# Config manipulation
# ---------------------------------------------------------------------------

def generate_display_name(model_id: str) -> str:
    """Derive a human-readable display name from a model ID."""
    # Strip org prefix (e.g., "Qwen/Qwen3-27B" -> "Qwen3-27B")
    return model_id.split("/")[-1] if "/" in model_id else model_id


def find_provider_by_url(config: dict, base_url: str) -> Optional[str]:
    """Return the provider ID whose baseURL matches, or None."""
    target = base_url.rstrip("/")
    for provider_id, provider in config.get("provider", {}).items():
        existing = provider.get("options", {}).get("baseURL", "").rstrip("/")
        if existing == target:
            return provider_id
    return None


@dataclass
class ProviderPlan:
    """What a sync intends to do to one provider.  Computed before anything is written."""

    provider_id: str
    model_ids: List[str]
    added: List[str]
    removed: List[str]
    renames: Dict[str, str]              # old model ID -> new model ID
    base_url: Optional[str]              # None means "don't touch the stored URL"
    model_key_updates: Dict[str, str]    # top-level "model"/"small_model" -> new value

    def is_noop(self) -> bool:
        return not (self.added or self.removed or self.renames or self.base_url
                    or self.model_key_updates)


def _infer_renames(
    existing_models: Dict,
    added: List[str],
    removed: List[str],
    explicit: Dict[str, str],
) -> Dict[str, str]:
    """Pair up a removed model with an added one when it is unambiguously a rename.

    Only fires on an exact 1:1 swap, and only for a removed entry that carries
    hand-tuned keys beyond "name" — those are the entries worth rescuing.  This is
    a heuristic: a genuine model swap would inherit the old limits and sampling, so
    callers are expected to report every inference they act on.
    """
    renames = dict(explicit)
    remaining_removed = [m for m in removed if m not in renames]
    remaining_added = [m for m in added if m not in renames.values()]

    if len(remaining_removed) != 1 or len(remaining_added) != 1:
        return renames

    old = remaining_removed[0]
    entry = existing_models.get(old)
    if isinstance(entry, dict) and set(entry) - {"name"}:
        renames[old] = remaining_added[0]
    return renames


def _plan_model_key_updates(
    config: dict,
    provider_id: str,
    model_ids: List[str],
    renames: Dict[str, str],
    normalize_bare_ids: bool,
) -> Dict[str, str]:
    """Decide how top-level model/small_model should change, if at all.

    The rule that matters: a pointer belonging to *another* provider is never
    touched.  Model IDs may themselves contain slashes ("org/model-a"), so
    "contains a slash" does not mean "provider-qualified" — the only reliable test
    is whether the first segment names a provider that actually exists.
    """
    known_providers = set(config.get("provider", {}))
    served = set(model_ids)
    updates: Dict[str, str] = {}
    if not model_ids:
        return updates

    first_qualified = f"{provider_id}/{model_ids[0]}"

    for key in ("model", "small_model"):
        current = config.get(key)
        if not current:
            continue  # never invent a pointer that wasn't there

        head, sep, rest = current.partition("/")
        owned_by_someone_else = sep and head in known_providers and head != provider_id
        if owned_by_someone_else:
            continue  # another provider's pointer — not ours to touch

        if sep and head == provider_id:
            target = renames.get(rest, rest)
            if target not in served:
                updates[key] = first_qualified
            elif target != rest:
                updates[key] = f"{provider_id}/{target}"
            continue

        # Bare model ID, i.e. written by an older version of this tool.
        if normalize_bare_ids and current in served:
            updates[key] = f"{provider_id}/{current}"

    return updates


def plan_provider_update(
    config: dict,
    provider_id: str,
    model_ids: List[str],
    base_url: Optional[str] = None,
    update_active_model: bool = True,
    renames: Optional[Dict[str, str]] = None,
    infer_renames: bool = True,
    normalize_bare_ids: bool = True,
) -> ProviderPlan:
    """Work out what syncing ``provider_id`` against ``model_ids`` would change."""
    existing_models: Dict = (
        config.get("provider", {}).get(provider_id, {}).get("models", {}) or {}
    )
    existing_ids = set(existing_models)
    new_ids = set(model_ids)

    explicit = {k: v for k, v in (renames or {}).items() if k in existing_ids}
    added = sorted(new_ids - existing_ids)
    removed = sorted(existing_ids - new_ids)

    resolved = (
        _infer_renames(existing_models, added, removed, explicit) if infer_renames else explicit
    )
    # A renamed model is a move, not an add plus a remove.
    added = [m for m in added if m not in resolved.values()]
    removed = [m for m in removed if m not in resolved]

    model_key_updates = (
        _plan_model_key_updates(config, provider_id, model_ids, resolved, normalize_bare_ids)
        if update_active_model
        else {}
    )

    existing_url = (
        config.get("provider", {}).get(provider_id, {}).get("options", {}).get("baseURL")
    )
    return ProviderPlan(
        provider_id=provider_id,
        model_ids=list(model_ids),
        added=added,
        removed=removed,
        renames=resolved,
        base_url=base_url if base_url != existing_url else None,
        model_key_updates=model_key_updates,
    )


def apply_plan(config: dict, plan: ProviderPlan) -> dict:
    """Apply a plan to a config dict, returning a new dict.  The input is not mutated."""
    config = copy.deepcopy(config)

    providers: Dict = config.setdefault("provider", {})
    if plan.provider_id not in providers:
        providers[plan.provider_id] = {
            "npm": "@ai-sdk/openai-compatible",
            "name": plan.provider_id,
            "options": {},
            "models": {},
        }

    provider = providers[plan.provider_id]

    if plan.base_url is not None:
        provider.setdefault("options", {})["baseURL"] = plan.base_url

    existing_models: Dict = provider.get("models", {})
    carried = {new: existing_models[old] for old, new in plan.renames.items()
               if old in existing_models}

    # Rebuild in server order: surviving and renamed entries keep their bodies verbatim,
    # so customised display names and hand-tuned settings ride through the sync.
    new_models: Dict = {}
    for mid in plan.model_ids:
        if mid in carried:
            new_models[mid] = carried[mid]
        elif mid in existing_models:
            new_models[mid] = existing_models[mid]
        else:
            new_models[mid] = {"name": generate_display_name(mid)}
    provider["models"] = new_models

    for key, value in plan.model_key_updates.items():
        config[key] = value

    return config


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
    return member.value_span, render_object(base, unit, block_leading, entries)


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
            # Re-splice the original source text, so comments *inside* the entry ride along.
            entries.append(
                RenderEntry(origin.leading, json.dumps(mid), origin.value, origin.trailing)
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
    return obj_span, render_object(base, unit, block_leading, entries)


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
        if member is None:
            # The planner only ever rewrites pointers that already exist.
            raise JsoncEditError(f"cannot locate top-level {key!r}")
        return member.value_span, json.dumps(value)

    return None


_MAX_EDIT_ROUNDS = 8


def _apply_one_plan(text: str, plan: ProviderPlan) -> str:
    current = parse_jsonc(text)
    desired = apply_plan(current, plan)

    for _ in range(_MAX_EDIT_ROUNDS):
        masked = mask_comments(text)
        edit = _next_edit(text, masked, current, desired, plan)
        if edit is None:
            return text
        text = apply_edits(text, [edit])
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


def update_provider_models(
    config: dict,
    provider_id: str,
    model_ids: List[str],
    base_url: Optional[str] = None,
    update_active_model: bool = True,
) -> Tuple[dict, List[str], List[str]]:
    """
    Update the models list for a provider in the config dict.

    Args:
        config:              The existing config dict (not mutated — a deep copy is returned).
        provider_id:         Which provider entry to update.
        model_ids:           The full list of model IDs now served by the server.
        base_url:            If set, also update the provider's options.baseURL.
        update_active_model: If True, update config.model / config.small_model when the
                             previously selected model is no longer available.

    Returns:
        (updated_config, added_ids, removed_ids)
    """
    plan = plan_provider_update(
        config=config,
        provider_id=provider_id,
        model_ids=model_ids,
        base_url=base_url,
        update_active_model=update_active_model,
        infer_renames=False,
    )
    # base_url is squashed to None by the planner when it already matches; apply_plan
    # only needs it when it differs, so the dict result is the same either way.
    return apply_plan(config, plan), plan.added, plan.removed
