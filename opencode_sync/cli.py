"""CLI entry point for opencode-sync."""

from __future__ import annotations

import argparse
import os
import stat
import sys
from pathlib import Path
from typing import Optional

from .config import (
    find_config_path,
    find_provider_by_url,
    load_config,
    save_config,
    update_provider_models,
)
from .vllm_client import DEFAULT_BASE_URL, VLLMClient, VLLMClientError

DEFAULT_PORT = 8080
LLAMA_HOST_ENV = "LLAMA_ARG_HOST"
LLAMA_PORT_ENV = "LLAMA_ARG_PORT"

_WRAPPER_TEMPLATE = """\
#!/bin/sh
# Written by: opencode-sync install
# Syncs vLLM models into the opencode config before each session.
opencode-sync 2>/dev/null || true
exec {opencode_bin} "$@"
"""

_DEFAULT_WRAPPER = Path.home() / ".local" / "bin" / "opencode"
_DEFAULT_OPENCODE_BIN = Path.home() / ".opencode" / "bin" / "opencode"


def _find_opencode_bin(wrapper_path: Path) -> Optional[Path]:
    """Walk PATH, skip any entry that resolves to wrapper_path, return first hit."""
    wrapper_resolved = wrapper_path.resolve() if wrapper_path.exists() else wrapper_path
    for dir_str in os.environ.get("PATH", "").split(os.pathsep):
        candidate = Path(dir_str) / "opencode"
        if not candidate.exists():
            continue
        if candidate.resolve() == wrapper_resolved:
            continue
        return candidate
    return None


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="opencode-sync",
        description=(
            "Sync opencode config with models served by a vLLM/llama.cpp server.\n\n"
            "Default (no --host/--port/LLAMA_ARG_HOST/LLAMA_ARG_PORT): reads\n"
            "baseURL from the existing config and queries that server without\n"
            "changing the URL in the config.\n\n"
            "With CLI or llama.cpp env host/port: queries the specified target\n"
            "and updates the provider's baseURL in the config "
            "(use --no-url-update to suppress)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="subcommand")

    # ---- install subcommand ----
    install_p = sub.add_parser(
        "install",
        help="Write a shell wrapper that auto-syncs models each time opencode starts.",
        description=(
            "Write a shell wrapper script that runs 'opencode-sync' before launching\n"
            "the real opencode binary.  Install it somewhere earlier in your PATH than\n"
            "the real opencode binary (default: ~/.local/bin/opencode)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    install_p.add_argument(
        "--wrapper",
        default=None,
        type=Path,
        metavar="PATH",
        help=f"Where to write the wrapper script (default: {_DEFAULT_WRAPPER})",
    )
    install_p.add_argument(
        "--opencode-bin",
        default=None,
        type=Path,
        metavar="PATH",
        help="Path to the real opencode binary (default: auto-detect from PATH)",
    )
    install_p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing wrapper script",
    )
    install_p.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be written without creating the file",
    )

    # ---- sync flags (top-level, default subcommand) ----
    p.add_argument(
        "--host",
        default=None,
        metavar="HOST",
        help="vLLM server hostname (default: read from config, else localhost)",
    )
    p.add_argument(
        "--port",
        type=int,
        default=None,
        metavar="PORT",
        help=f"vLLM server port (default: read from config, else {DEFAULT_PORT})",
    )
    p.add_argument(
        "--provider",
        default=None,
        dest="provider_id",
        metavar="ID",
        help="Provider ID in opencode config to update (default: auto-detect)",
    )
    p.add_argument(
        "--config",
        default=None,
        type=Path,
        metavar="PATH",
        help="Path to opencode.jsonc (default: auto-detect per platform)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned changes without writing the config",
    )
    p.add_argument(
        "--no-url-update",
        action="store_true",
        help="Don't update baseURL in the provider config even when --host/--port is given",
    )
    p.add_argument(
        "--no-model-update",
        action="store_true",
        help="Don't update model/small_model even if the active model is removed",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=10,
        metavar="SECONDS",
        help="HTTP request timeout in seconds (default: 10)",
    )
    return p


def _cmd_install(args) -> int:
    wrapper_path: Path = args.wrapper or _DEFAULT_WRAPPER

    # Resolve the real opencode binary
    if args.opencode_bin is not None:
        opencode_bin = args.opencode_bin
    else:
        opencode_bin = _find_opencode_bin(wrapper_path) or _DEFAULT_OPENCODE_BIN

    content = _WRAPPER_TEMPLATE.format(opencode_bin=opencode_bin)

    if args.dry_run:
        print(f"[dry-run] Would write wrapper to: {wrapper_path}")
        print(f"[dry-run] Real opencode binary:    {opencode_bin}")
        print(f"[dry-run] Wrapper content:\n{content}")
        return 0

    if wrapper_path.exists() and not args.force:
        print(
            f"ERROR: {wrapper_path} already exists. Use --force to overwrite.",
            file=sys.stderr,
        )
        return 1

    wrapper_path.parent.mkdir(parents=True, exist_ok=True)
    wrapper_path.write_text(content, encoding="utf-8")
    wrapper_path.chmod(wrapper_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    print(f"Wrapper written: {wrapper_path}")
    print(f"  → runs opencode-sync, then exec {opencode_bin}")

    # Warn if the wrapper directory isn't on PATH before the real binary
    path_dirs = os.environ.get("PATH", "").split(os.pathsep)
    wrapper_dir = str(wrapper_path.parent)
    bin_dir = str(opencode_bin.parent) if isinstance(opencode_bin, Path) else ""
    wrapper_idx = next((i for i, d in enumerate(path_dirs) if d == wrapper_dir), None)
    bin_idx = next((i for i, d in enumerate(path_dirs) if d == bin_dir), None)
    if wrapper_idx is None:
        print(
            f"\nWARNING: {wrapper_path.parent} is not in your PATH.\n"
            f"  Add this to your shell rc file:\n"
            f"    export PATH=\"{wrapper_path.parent}:$PATH\"",
            file=sys.stderr,
        )
    elif bin_idx is not None and bin_idx < wrapper_idx:
        print(
            f"\nWARNING: {opencode_bin.parent} appears before {wrapper_path.parent} in PATH.\n"
            f"  The wrapper won't be used. Move {wrapper_path.parent} earlier in PATH.",
            file=sys.stderr,
        )

    return 0


def _env_value(name: str) -> Optional[str]:
    value = os.environ.get(name)
    return value if value else None


def _env_port() -> Optional[int]:
    value = _env_value(LLAMA_PORT_ENV)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        _die(f"{LLAMA_PORT_ENV} must be an integer, got {value!r}.")


def _resolve_config_path(explicit: Optional[Path]) -> Path:
    if explicit is not None:
        return explicit
    detected = find_config_path()
    if detected is None:
        _die("Could not locate opencode config. Use --config to specify the path.")
    return detected  # type: ignore[return-value]


def _die(msg: str, code: int = 1) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(code)


def main(argv=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.subcommand == "install":
        return _cmd_install(args)

    config_path = _resolve_config_path(args.config)
    env_host = _env_value(LLAMA_HOST_ENV)
    env_port = _env_port()
    target_specified = (
        args.host is not None
        or args.port is not None
        or env_host is not None
        or env_port is not None
    )
    target_host = args.host or env_host or "localhost"
    target_port = (
        args.port
        if args.port is not None
        else (env_port if env_port is not None else DEFAULT_PORT)
    )

    # ------------------------------------------------------------------ #
    # Load existing config (or start fresh)
    # ------------------------------------------------------------------ #
    if config_path.exists():
        print(f"Loading config: {config_path}")
        try:
            config = load_config(config_path)
        except Exception as e:
            _die(f"Failed to parse config: {e}")
    else:
        print(f"Config not found — will create: {config_path}")
        config = {}

    # ------------------------------------------------------------------ #
    # Resolve provider ID
    # ------------------------------------------------------------------ #
    provider_id: Optional[str] = args.provider_id

    if provider_id is None:
        providers = config.get("provider", {})
        if len(providers) == 1:
            provider_id = next(iter(providers))
        elif len(providers) == 0:
            # New config — default to "vllm"
            provider_id = "vllm"
        else:
            # Try to match by URL when we already know the target
            if target_specified:
                candidate_url = f"http://{target_host}:{target_port}/v1"
                provider_id = find_provider_by_url(config, candidate_url)
            if provider_id is None:
                _die(
                    f"Multiple providers found ({', '.join(providers)}). "
                    "Use --provider to specify which one to update."
                )

    # ------------------------------------------------------------------ #
    # Determine query URL and whether to update baseURL in config
    # ------------------------------------------------------------------ #
    existing_base_url: str = (
        config.get("provider", {})
        .get(provider_id, {})
        .get("options", {})
        .get("baseURL", "")
    )

    if target_specified:
        query_base_url = f"http://{target_host}:{target_port}/v1"
        new_config_base_url = None if args.no_url_update else query_base_url
    else:
        query_base_url = existing_base_url or DEFAULT_BASE_URL
        new_config_base_url = None  # Don't touch the stored URL

    # ------------------------------------------------------------------ #
    # Query the server
    # ------------------------------------------------------------------ #
    print(f"Querying {query_base_url}/models ...")
    client = VLLMClient(base_url=query_base_url, timeout=args.timeout)
    try:
        model_ids = client.get_model_ids()
    except VLLMClientError as e:
        _die(str(e))

    if not model_ids:
        print("WARNING: Server returned no models. Config will not be updated.", file=sys.stderr)
        return 0

    print(f"Server reports {len(model_ids)} model(s): {', '.join(model_ids)}")

    # ------------------------------------------------------------------ #
    # Compute and display changes
    # ------------------------------------------------------------------ #
    print(f"Updating provider '{provider_id}'...")

    updated_config, added, removed = update_provider_models(
        config=config,
        provider_id=provider_id,
        model_ids=model_ids,
        base_url=new_config_base_url,
        update_active_model=not args.no_model_update,
    )

    if added:
        print(f"  + Added:   {', '.join(added)}")
    if removed:
        print(f"  - Removed: {', '.join(removed)}")
    if not added and not removed:
        print("  Model list unchanged.")

    if new_config_base_url and new_config_base_url != existing_base_url:
        print(f"  baseURL: {existing_base_url!r} -> {new_config_base_url!r}")

    for key in ("model", "small_model"):
        old_val = config.get(key)
        new_val = updated_config.get(key)
        if old_val != new_val:
            print(f"  {key}: {old_val!r} -> {new_val!r} (previous selection no longer served)")

    # ------------------------------------------------------------------ #
    # Write (unless --dry-run)
    # ------------------------------------------------------------------ #
    if args.dry_run:
        print("\n[dry-run] Config not written.")
        return 0

    try:
        save_config(config_path, updated_config)
    except Exception as e:
        _die(f"Failed to write config: {e}")

    print(f"\nConfig saved: {config_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
