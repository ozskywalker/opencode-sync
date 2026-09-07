"""Config file discovery, reading, atomic writing, and JSONC parsing.

Everything here is byte-exactness plumbing: reads never translate newlines,
writes go through a temp file + os.replace in the target directory, and a
rolling .bak is written only when content actually changes.  The greenfield
``save_config`` is the one json.dumps path — by design, because a config that
doesn't exist yet has no comments or formatting to preserve.
"""

from __future__ import annotations

import json
import os
import stat
import tempfile
from pathlib import Path
from typing import List, Optional

from .jsonc_edit import mask_comments, mask_trailing_commas


# ---------------------------------------------------------------------------
# JSONC parsing
# ---------------------------------------------------------------------------

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
