"""opencode state file (~/.local/state/opencode/model.json) maintenance.

opencode's TUI records recently used models there, and its defaultModel()
prefers that recent[] list (skipping dead entries) before falling back to
"first model in config".  After a server-side rename, stale recent entries
are harmless (opencode skips them) but they shadow the config order, so
pruning them lets the synced model list actually drive the default.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .io import _atomic_write_text


def state_model_json_path() -> Path:
    """Path of opencode's model.json state file, honoring XDG_STATE_HOME."""
    state_home = os.environ.get("XDG_STATE_HOME")
    if state_home:
        return Path(state_home) / "opencode" / "model.json"
    return Path.home() / ".local" / "state" / "opencode" / "model.json"


def prune_recent_models(
    path: Optional[Path] = None,
    live_providers: Optional[Dict[str, object]] = None,
    model_key_updates: Optional[Dict[str, str]] = None,
    dry_run: bool = False,
) -> List[Tuple[str, str]]:
    """Drop recent[] entries from opencode's state file that no longer resolve.

    An entry is kept when its provider still exists *and* its model ID is present
    in that provider's models (``live_providers`` maps provider ID -> models dict,
    i.e. the freshly synced config).  Entries for unknown providers are kept:
    this tool only knows about providers it syncs.

    If model.json's first recent entry is being dropped and ``model_key_updates``
    names a replacement, the replacement is prepended so opencode's defaultModel()
    recent[] lookup lands on the synced model instead of falling through.

    Returns the list of (providerID, modelID) pairs that were removed.
    The file is only rewritten when something changed and ``dry_run`` is False.
    """
    path = path or state_model_json_path()
    model_key_updates = model_key_updates or {}
    if not path.exists():
        return []

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return []  # unreadable/absent state: opencode will cope, so do nothing
    recent = data.get("recent")
    if not isinstance(recent, list):
        return []

    def _is_live(entry: object) -> bool:
        if not isinstance(entry, dict):
            return False
        pid = entry.get("providerID")
        mid = entry.get("modelID")
        if not isinstance(pid, str) or not isinstance(mid, str):
            return False
        provider = (live_providers or {}).get(pid)
        if provider is None:
            return True  # not ours to judge
        if not isinstance(provider, dict):
            return False
        models = provider.get("models")
        if not isinstance(models, dict):
            return True  # provider known but model list absent: not ours to judge
        return mid in models

    removed: List[Tuple[str, str]] = []
    kept: List[dict] = []
    for entry in recent:
        if _is_live(entry):
            kept.append(entry)
        elif isinstance(entry, dict) and isinstance(entry.get("providerID"), str) \
                and isinstance(entry.get("modelID"), str):
            removed.append((entry["providerID"], entry["modelID"]))
        # malformed entries are dropped silently — they can't resolve anyway

    if not removed:
        return []

    for key, value in model_key_updates.items():
        head, _, rest = value.partition("/")
        pair = {"providerID": head, "modelID": rest}
        if all(p != pair for p in kept):
            kept.insert(0, pair)
        break  # only seed "model"; small_model is not a recent[] concern

    if not dry_run:
        data["recent"] = kept
        _atomic_write_text(path, json.dumps(data) + "\n")
    return removed
