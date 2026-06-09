"""Read, parse, update, and write opencode JSONC config files."""

from __future__ import annotations

import copy
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# JSONC parsing
# ---------------------------------------------------------------------------

def _strip_jsonc_comments(text: str) -> str:
    """Remove // and /* */ comments from JSONC text, leaving string literals intact."""
    result: List[str] = []
    i = 0
    n = len(text)
    in_string = False

    while i < n:
        c = text[i]
        if in_string:
            if c == "\\" and i + 1 < n:
                result.append(c)
                result.append(text[i + 1])
                i += 2
            elif c == '"':
                in_string = False
                result.append(c)
                i += 1
            else:
                result.append(c)
                i += 1
        else:
            if c == '"':
                in_string = True
                result.append(c)
                i += 1
            elif c == "/" and i + 1 < n and text[i + 1] == "/":
                # Single-line comment: skip to end of line
                while i < n and text[i] != "\n":
                    i += 1
            elif c == "/" and i + 1 < n and text[i + 1] == "*":
                # Block comment: skip to */
                i += 2
                while i < n - 1:
                    if text[i] == "*" and text[i + 1] == "/":
                        i += 2
                        break
                    i += 1
                else:
                    i = n  # unterminated block comment — consume to EOF
            else:
                result.append(c)
                i += 1

    return "".join(result)


def _strip_trailing_commas(text: str) -> str:
    """Remove trailing commas before } or ] (valid JSONC, invalid JSON)."""
    return re.sub(r",(\s*[}\]])", r"\1", text)


def parse_jsonc(text: str) -> dict:
    """Parse a JSONC string (JSON with comments and trailing commas) into a dict."""
    cleaned = _strip_jsonc_comments(text)
    cleaned = _strip_trailing_commas(cleaned)
    return json.loads(cleaned)


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

def load_config(path: Path) -> dict:
    """Load and parse an opencode JSONC config file."""
    text = path.read_text(encoding="utf-8")
    return parse_jsonc(text)


def save_config(path: Path, config: dict) -> None:
    """Write config dict to disk as formatted JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(config, indent=2, ensure_ascii=False) + "\n"
    path.write_text(content, encoding="utf-8")


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
    config = copy.deepcopy(config)

    providers: Dict = config.setdefault("provider", {})

    if provider_id not in providers:
        providers[provider_id] = {
            "npm": "@ai-sdk/openai-compatible",
            "name": provider_id,
            "options": {},
            "models": {},
        }

    provider = providers[provider_id]

    if base_url is not None:
        provider.setdefault("options", {})["baseURL"] = base_url

    existing_models: Dict = provider.get("models", {})
    existing_ids = set(existing_models)
    new_ids = set(model_ids)
    added = sorted(new_ids - existing_ids)
    removed = sorted(existing_ids - new_ids)

    # Rebuild models dict: preserve existing entries, add new ones with generated names.
    new_models: Dict = {}
    for mid in model_ids:
        if mid in existing_models:
            new_models[mid] = existing_models[mid]
        else:
            new_models[mid] = {"name": generate_display_name(mid)}
    provider["models"] = new_models

    if update_active_model and model_ids:
        qualified_ids = {f"{provider_id}/{mid}" for mid in model_ids}
        first_qualified = f"{provider_id}/{model_ids[0]}"
        for key in ("model", "small_model"):
            current = config.get(key)
            if not current:
                continue
            if current in new_ids:
                # Bare model ID (written by a previous sync) — normalize to qualified form
                config[key] = f"{provider_id}/{current}"
            elif current not in qualified_ids:
                # Model no longer available — point to first available
                config[key] = first_qualified
            # Already provider-qualified and still available: leave as-is

    return config, added, removed
