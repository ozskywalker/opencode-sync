"""Pure planning logic: decide what a sync would change, before anything is written.

Dict-in, dict-out.  No file I/O, no text splicing, no opencode file knowledge
beyond the config's shape.  ``ProviderPlan`` lives here because both the CLI
and the surgical editor consume it, and it must never depend on either.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# Modes for how top-level model/small_model pointers are managed.  Exported so the
# CLI can validate before touching disk.
DEFAULT_MODEL_MODES = ("first", "none", "auto")


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


def _resolve_pointer_mode(
    mode: str,
    explicit: Optional[str],
    provider_id: str,
) -> Tuple[Optional[str], Optional[str]]:
    """Turn a --default-model style mode into (mode_for_planner, explicit_value).

    An explicit ``provider/model`` value wins over any keyword mode.  A bare model
    ID is rejected here rather than guessed at: only the user knows which provider
    an unqualified ID belongs to when several are configured.
    """
    if explicit is not None or (mode not in DEFAULT_MODEL_MODES and "/" in mode):
        # The CLI passes an explicit ID via the mode slot when only one flag form
        # is used; accept either spelling.
        value = explicit if explicit is not None else mode
        if "/" not in value:
            raise ValueError(
                f"explicit default model {value!r} must be provider-qualified "
                f"(provider/model-id)"
            )
        head = value.partition("/")[0]
        if head != provider_id:
            # Another provider's model: the caller plans that provider separately.
            # Pointing at it from this plan would couple unrelated syncs.
            raise ValueError(
                f"explicit default model {value!r} names provider {head!r}, "
                f"but this sync is for {provider_id!r}"
            )
        return "explicit", value

    if mode not in DEFAULT_MODEL_MODES:
        raise ValueError(
            f"unknown default-model mode {mode!r}; expected one of "
            f"{', '.join(DEFAULT_MODEL_MODES)} or an explicit provider/model ID"
        )
    return mode, None


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
    model_mode: str = "first",
    small_model_mode: str = "first",
    explicit_model: Optional[str] = None,
    explicit_small_model: Optional[str] = None,
    model_set_changed: bool = False,
) -> Dict[str, str]:
    """Decide how top-level model/small_model should change, if at all.

    Two orthogonal rules:

    *Which pointers may we touch?*  A pointer belonging to *another* provider is
    never touched.  Model IDs may themselves contain slashes ("org/model-a"), so
    "contains a slash" does not mean "provider-qualified" — the only reliable test
    is whether the first segment names a provider that actually exists.

    *When do we touch them?*  The modes:
      "first"  — legacy behavior.  Never invent a pointer; only repair one that
                 already points at this provider and has gone stale (removed model,
                 rename, or bare-ID normalization).  This is why a fresh sync of a
                 renamed server does not make opencode default to the new model when
                 no pointer existed.
      "none"   — never touch the key at all, not even to repair it.
      "auto"   — after a change to this provider's model set, point the key at the
                 first served model, so opencode's own defaultModel() (which prefers
                 config.model outright, then the recent[] state file, then the first
                 config model) resolves to something this server actually serves.
                 Only fires when the model set actually changed, so routine no-op
                 syncs don't rewrite the pointer.
      "explicit" — an explicit provider/model value was given; point the key there
                 whenever it isn't already there (regardless of whether the sync
                 changed anything else).

    "auto"/"explicit" deliberately override the "never invent a pointer" rule: that
    conservatism exists so a bare wrapper invocation can't surprise anyone, and the
    user has now explicitly asked to be surprised.
    """
    known_providers = set(config.get("provider", {}))
    served = set(model_ids)
    updates: Dict[str, str] = {}
    if not model_ids:
        return updates

    first_qualified = f"{provider_id}/{model_ids[0]}"

    modes = {"model": model_mode, "small_model": small_model_mode}
    explicit = {"model": explicit_model, "small_model": explicit_small_model}

    for key in ("model", "small_model"):
        mode = modes[key]
        current = config.get(key)

        if mode == "none":
            continue

        if mode == "explicit":
            target = explicit[key]
            if target is None:  # explicit value failed to resolve; treat as "first"
                mode = "first"
            elif current != target:
                updates[key] = target
                continue
            else:
                continue

        if mode == "auto":
            # Repair/repoint when this sync actually changed the provider's model
            # set (the call site threads that via model_set_changed→renames signal
            # and the explicit mode handles "always").  A pure no-op sync must not
            # churn the pointer: the wrapper runs on every opencode launch.
            if current is None:
                # Invent a pointer only if the model set actually changed.
                if not model_set_changed:
                    continue
                updates[key] = first_qualified
                continue
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
                # else: pointer still valid and nothing to repair — leave it
                continue

            # Bare model ID, i.e. written by an older version of this tool.
            if normalize_bare_ids and current in served:
                updates[key] = f"{provider_id}/{current}"
            elif model_set_changed:
                # Stale bare ID (or unresolvable) and the set moved: repoint.
                updates[key] = first_qualified
            continue

        # mode == "first": legacy repair-only behavior.
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
    model_mode: str = "first",
    small_model_mode: str = "first",
    explicit_model: Optional[str] = None,
    explicit_small_model: Optional[str] = None,
) -> ProviderPlan:
    """Work out what syncing ``provider_id`` against ``model_ids`` would change.

    ``model_mode``/``small_model_mode`` are "first", "none" or "auto";
    ``explicit_model``/``explicit_small_model`` override them with a literal
    "provider/model" pointer when given (validated against ``provider_id``).
    """
    existing_models: Dict = (
        config.get("provider", {}).get(provider_id, {}).get("models", {}) or {}
    )
    existing_ids = set(existing_models)
    new_ids = set(model_ids)

    explicit_renames = {k: v for k, v in (renames or {}).items() if k in existing_ids}
    added = sorted(new_ids - existing_ids)
    removed = sorted(existing_ids - new_ids)

    resolved = (
        _infer_renames(existing_models, added, removed, explicit_renames) if infer_renames else explicit_renames
    )
    # A renamed model is a move, not an add plus a remove.
    added = [m for m in added if m not in resolved.values()]
    removed = [m for m in removed if m not in resolved]

    model_set_changed = bool(added or removed or resolved)

    model_key_updates = {}
    if update_active_model:
        model_mode_eff, explicit_model_eff = _resolve_pointer_mode(
            model_mode, explicit_model, provider_id)
        small_mode_eff, explicit_small_eff = _resolve_pointer_mode(
            small_model_mode, explicit_small_model, provider_id)
        model_key_updates = _plan_model_key_updates(
            config, provider_id, model_ids, resolved, normalize_bare_ids,
            model_mode=model_mode_eff,
            small_model_mode=small_mode_eff,
            explicit_model=explicit_model_eff,
            explicit_small_model=explicit_small_eff,
            model_set_changed=model_set_changed,
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
    # A renamed entry keeps its tuned settings but gets a fresh display name:
    # the old name described the old model ID, and opencode surfaces it in the
    # picker, so leaving it stale makes the sync look like it failed.  An entry
    # that had no explicit name keeps having none, matching the text path.
    for new, entry in carried.items():
        if isinstance(entry, dict) and "name" in entry:
            entry = dict(entry)  # shallow copy: this entry is ours to change
            entry["name"] = generate_display_name(new)
            carried[new] = entry

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
