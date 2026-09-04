"""CLI entry point for opencode-sync."""

from __future__ import annotations

import argparse
import difflib
import json
import os
import stat
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from .config import (
    DEFAULT_MODEL_MODES,
    ProviderPlan,
    apply_plan,
    apply_plans_to_text,
    find_config_path,
    find_provider_by_url,
    load_config,
    load_config_text,
    plan_provider_update,
    prune_recent_models,
    save_config,
    save_config_text,
)
from .jsonc_edit import JsoncEditError
from .vllm_client import DEFAULT_BASE_URL, VLLMClient, VLLMClientError

DEFAULT_PORT = 8080
LLAMA_HOST_ENV = "LLAMA_ARG_HOST"
LLAMA_PORT_ENV = "LLAMA_ARG_PORT"

PYPI_PACKAGE_NAME = "opencode-sync"
PYPI_JSON_URL = f"https://pypi.org/pypi/{PYPI_PACKAGE_NAME}/json"
PYPI_PROJECT_URL = f"https://pypi.org/project/{PYPI_PACKAGE_NAME}/"
UPDATE_CHECK_TIMEOUT = 1  # seconds; must never delay a real run noticeably

_WRAPPER_TEMPLATE = """\
#!/bin/sh
# Written by: opencode-sync install
# Syncs vLLM models into the opencode config before each session.
opencode-sync 2>/dev/null || true
exec {opencode_bin} "$@"
"""

_DEFAULT_WRAPPER = Path.home() / ".local" / "bin" / "opencode"
_DEFAULT_OPENCODE_BIN = Path.home() / ".opencode" / "bin" / "opencode"


def _get_version() -> str:
    """Resolve the running version: installed metadata first, package fallback.

    importlib.metadata is authoritative for installed copies (including
    setuptools-scm dev versions like 0.5.0.dev1+g4b58801); __version__ covers
    running from a bare checkout with no install.
    """
    try:
        import importlib.metadata

        return importlib.metadata.version(PYPI_PACKAGE_NAME)
    except Exception:
        from . import __version__

        return __version__


def _print_banner() -> None:
    print(f"opencode-sync v{_get_version()}")


def _release_segments(version: str) -> Optional[Tuple[int, ...]]:
    """Extract the leading numeric release segments from a version string.

    Returns None for anything that isn't a plain release: pre-releases
    (0.6.0rc1), dev builds (0.5.0.dev1+g4b58801), or garbage.  A trailing
    local segment (+local) is stripped first and does not disqualify.
    Callers treat None as "don't compare".
    """
    head = version.split("+", 1)[0].strip()
    parts = head.split(".")
    segments = []
    for part in parts:
        # str.isdigit() accepts non-ASCII digit glyphs (superscripts etc.) that
        # int() rejects, so require plain ASCII 0-9 explicitly.
        if not (part.isascii() and part.isdigit()):
            return None
        segments.append(int(part))
    return tuple(segments) if segments else None


def _is_newer_release(pypi_version: str, local_version: str) -> bool:
    """True only when pypi_version is a strictly newer plain release than local.

    Dev/pre-release local versions (0.5.0.dev1+g...) never trigger an update
    nag: a dev checkout sits between releases and comparing it to one is noise.
    Malformed versions fail safe (no nag) rather than false-positive.
    """
    local = _release_segments(local_version)
    remote = _release_segments(pypi_version)
    if local is None or remote is None:
        return False
    return remote > local


def _pypi_http_get_json(url: str, timeout: int) -> dict:
    """Fetch a JSON document with stdlib urllib. Raises on any failure."""
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _check_pypi_update(
    _http_get_json: Callable[[str, int], dict] = _pypi_http_get_json,
) -> Optional[str]:
    """Return a one-line update notice when PyPI has a newer release, else None.

    Strictly best-effort: every failure mode (network, HTTP, JSON, timeout)
    is swallowed — this must never disturb the run that just completed.
    """
    version = _get_version()
    try:
        if _release_segments(version) is None:
            return None  # dev/pre-release build: never nag
        data = _http_get_json(PYPI_JSON_URL, UPDATE_CHECK_TIMEOUT)
        latest = data["info"]["version"]
        if not isinstance(latest, str) or not _is_newer_release(latest, version):
            return None
    except Exception:
        # Fetched payload, parsing, and comparison are all inside the swallow
        # guard: a hostile/garbage PyPI response must never crash the caller.
        return None
    return (
        f"Update available: {PYPI_PACKAGE_NAME} {latest} "
        f"(you have {version}) — {PYPI_PROJECT_URL}"
    )


def _report_update_check(enabled: bool) -> None:
    if not enabled:
        return
    notice = _check_pypi_update()
    if notice:
        print("\n---")
        print(notice)


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
    p.add_argument(
        "--version",
        action="version",
        version=f"opencode-sync v{_get_version()}",
    )
    p.add_argument(
        "--no-update-check",
        action="store_true",
        help="Skip the PyPI check for a newer opencode-sync release",
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
        "--default-model",
        default=None,
        metavar="MODE",
        help=(
            "How the top-level 'model' pointer is managed: 'first' (default: only "
            "repair an existing pointer), 'none' (never touch it), 'auto' (point it "
            "at the first served model whenever this provider's model set changes), "
            "or an explicit 'provider/model-id'"
        ),
    )
    p.add_argument(
        "--default-small-model",
        default=None,
        metavar="MODE",
        help=(
            "How 'small_model' is managed; same values as --default-model "
            "(default: 'first')"
        ),
    )
    p.add_argument(
        "--prune-recent",
        action="store_true",
        help=(
            "Drop dead models from opencode's recent[] state file "
            "(~/.local/state/opencode/model.json) so they can't shadow the default"
        ),
    )
    p.add_argument(
        "--rename",
        action="append",
        default=None,
        metavar="OLD=NEW",
        help=(
            "Move an existing model entry to a new ID, keeping its settings and "
            "comments (repeatable). Use when a server's model ID changes."
        ),
    )
    p.add_argument(
        "--no-infer-renames",
        action="store_true",
        help=(
            "Don't treat a 1-removed/1-added sync as a rename; drop the old entry "
            "and its settings instead"
        ),
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=10,
        metavar="SECONDS",
        help="HTTP request timeout in seconds (default: 10)",
    )
    return p


def _parse_renames(values: Optional[List[str]]) -> Dict[str, str]:
    renames: Dict[str, str] = {}
    for item in values or []:
        old, sep, new = item.partition("=")
        if not sep or not old or not new:
            _die(f"--rename expects OLD=NEW, got {item!r}.")
        renames[old] = new
    return renames


def _resolve_pointer_flags(args) -> Tuple[str, str, Optional[str], Optional[str]]:
    """Normalize --default-model/--default-small-model into planner arguments.

    An explicit 'provider/model-id' sets the mode to a marker that the planner
    validates per-provider (it dies if the named provider isn't the one syncing).
    Values are validated here so typos fail before any server is queried.
    """
    model_mode = (args.default_model or "first").strip()
    small_mode = (args.default_small_model or "first").strip()
    explicit_model = explicit_small = None

    def _split(value: str):
        if value in DEFAULT_MODEL_MODES:
            return value, None
        if "/" in value:
            return "explicit", value
        _die(
            f"--default-model expects 'first', 'none', 'auto', or 'provider/model-id', "
            f"got {value!r}."
        )

    model_mode, explicit_model = _split(model_mode)
    small_mode, explicit_small = _split(small_mode)
    return model_mode, small_mode, explicit_model, explicit_small


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
        _report_update_check(not args.no_update_check)
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

    _report_update_check(not args.no_update_check)
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


def _die(msg: str, code: int = 1, update_check: bool = False) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    _report_update_check(update_check)
    sys.exit(code)


@dataclass
class Target:
    """One provider to sync, and the URL to query for it."""

    provider_id: str
    query_base_url: str
    new_base_url: Optional[str]  # None means "leave the stored URL alone"


def _resolve_targets(args, config: dict, target_specified: bool, host: str, port: int) -> List[Target]:
    """Decide which providers to sync and which URL to ask each one about.

    Explicit target (--provider, or --host/--port naming one server) keeps the old
    single-provider semantics.  With neither, every provider that has a stored
    baseURL is synced against its own URL — which is what makes the bare wrapper
    invocation useful on a multi-provider config.
    """
    providers = config.get("provider", {})

    def query_url_for(provider_id: str) -> str:
        return providers.get(provider_id, {}).get("options", {}).get("baseURL", "")

    if target_specified:
        url = f"http://{host}:{port}/v1"
        provider_id = args.provider_id
        if provider_id is None:
            if len(providers) == 1:
                provider_id = next(iter(providers))
            elif not providers:
                provider_id = "vllm"
            else:
                provider_id = find_provider_by_url(config, url)
                if provider_id is None:
                    _die(
                        f"Multiple providers found ({', '.join(providers)}). "
                        "Use --provider to specify which one to update."
                    )
        return [Target(provider_id, url, None if args.no_url_update else url)]

    if args.provider_id is not None:
        url = query_url_for(args.provider_id) or DEFAULT_BASE_URL
        return [Target(args.provider_id, url, None)]

    if not providers:
        return [Target("vllm", DEFAULT_BASE_URL, None)]

    targets = []
    for provider_id in providers:
        url = query_url_for(provider_id)
        if not url:
            print(
                f"WARNING: provider '{provider_id}' has no options.baseURL — skipping.",
                file=sys.stderr,
            )
            continue
        targets.append(Target(provider_id, url, None))

    if not targets:
        _die("No providers with a baseURL to sync. Use --host/--port or --provider.")
    return targets


def _sync_one(args, config: dict, target: Target, renames: Dict[str, str], single: bool,
              model_mode: str = "first", small_model_mode: str = "first",
              explicit_model: Optional[str] = None, explicit_small_model: Optional[str] = None):
    """Query one server and plan its update.

    Returns (plan, failed).  A plan of None means "nothing to do"; failed says
    whether that was because the server was unreachable, which is the only case
    that should colour the exit code.
    """
    print(f"Querying {target.query_base_url}/models for '{target.provider_id}' ...")
    client = VLLMClient(base_url=target.query_base_url, timeout=args.timeout)
    try:
        model_ids = client.get_model_ids()
    except VLLMClientError as e:
        # A server we were explicitly pointed at is a hard error; one we merely
        # discovered from the config is a warning, so one box being off doesn't
        # block the others.
        if single:
            _die(str(e), update_check=not args.no_update_check)
        print(f"WARNING: {target.provider_id}: {e} — skipping.", file=sys.stderr)
        return None, True

    if not model_ids:
        # The server is up, it just has nothing loaded. Never wipe the models list.
        print(
            f"WARNING: {target.provider_id}: server returned no models — skipping.",
            file=sys.stderr,
        )
        return None, False

    print(f"  Server reports {len(model_ids)} model(s): {', '.join(model_ids)}")

    plan = plan_provider_update(
        config=config,
        provider_id=target.provider_id,
        model_ids=model_ids,
        base_url=target.new_base_url,
        update_active_model=not args.no_model_update,
        renames=renames,
        infer_renames=not args.no_infer_renames,
        normalize_bare_ids=single,
        model_mode=model_mode,
        small_model_mode=small_model_mode,
        explicit_model=explicit_model,
        explicit_small_model=explicit_small_model,
    )
    _report(plan, config)
    return plan, False


def _report(plan: ProviderPlan, config: dict) -> None:
    existing = config.get("provider", {}).get(plan.provider_id, {}).get("models", {})
    for old, new in plan.renames.items():
        kept = sorted(set(existing.get(old, {})) - {"name"})
        detail = f" (kept {', '.join(kept)})" if kept else ""
        print(f"  ~ Renamed: {old} -> {new}{detail}")
    if plan.added:
        print(f"  + Added:   {', '.join(plan.added)}")
    if plan.removed:
        print(f"  - Removed: {', '.join(plan.removed)}")
    if not (plan.added or plan.removed or plan.renames):
        print("  Model list unchanged.")
    if plan.base_url is not None:
        print(f"  baseURL -> {plan.base_url!r}")
    for key, value in plan.model_key_updates.items():
        print(f"  {key}: {config.get(key)!r} -> {value!r}")


def main(argv=None) -> int:
    _print_banner()
    parser = _build_parser()
    args = parser.parse_args(argv)
    update_check_enabled = not args.no_update_check

    if args.subcommand == "install":
        return _cmd_install(args)

    config_path = _resolve_config_path(args.config)
    renames = _parse_renames(args.rename)
    model_mode, small_model_mode, explicit_model, explicit_small_model = (
        _resolve_pointer_flags(args))
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
    text: Optional[str] = None
    if config_path.exists():
        print(f"Loading config: {config_path}")
        try:
            text = load_config_text(config_path)
            config = load_config(config_path)
        except Exception as e:
            _die(f"Failed to parse config: {e}")
    else:
        print(f"Config not found — will create: {config_path}")
        config = {}

    targets = _resolve_targets(args, config, target_specified, target_host, target_port)
    single = len(targets) == 1

    if renames and not single:
        _die("--rename needs a single provider. Use --provider to pick one.")

    # ------------------------------------------------------------------ #
    # Query each server and plan its changes
    # ------------------------------------------------------------------ #
    plans = []
    failures = 0
    for target in targets:
        plan, failed = _sync_one(
            args, config, target, renames, single,
            model_mode=model_mode,
            small_model_mode=small_model_mode,
            explicit_model=explicit_model,
            explicit_small_model=explicit_small_model,
        )
        failures += bool(failed)
        if plan is not None:
            plans.append(plan)

    if not plans:
        if failures:
            print("Nothing synced.", file=sys.stderr)
            _report_update_check(update_check_enabled)
            return 1
        _report_update_check(update_check_enabled)
        return 0  # servers were reachable, there was just nothing to apply

    if all(plan.is_noop() for plan in plans):
        print("\nConfig already up to date.")
        _report_update_check(update_check_enabled)
        return 0

    # ------------------------------------------------------------------ #
    # Render the new config text
    # ------------------------------------------------------------------ #
    if text is None:
        updated = config
        for plan in plans:
            updated = apply_plan(updated, plan)
        new_text = None  # greenfield: save_config renders it
    else:
        try:
            new_text = apply_plans_to_text(text, config, plans)
        except JsoncEditError as e:
            _die(f"Refusing to write: {e}")

    # ------------------------------------------------------------------ #
    # Write (unless --dry-run)
    # ------------------------------------------------------------------ #
    if args.dry_run:
        if new_text is not None:
            diff = difflib.unified_diff(
                text.splitlines(keepends=True),
                new_text.splitlines(keepends=True),
                fromfile=str(config_path),
                tofile=f"{config_path} (planned)",
            )
            sys.stdout.writelines(diff)
        print("\n[dry-run] Config not written.")
        _report_update_check(update_check_enabled)
        return 0

    try:
        if new_text is None:
            save_config(config_path, updated)
        elif new_text == text:
            print("\nConfig already byte-identical — not rewritten.")
            _report_update_check(update_check_enabled)
            return 0
        else:
            save_config_text(config_path, new_text, backup=True)
    except Exception as e:
        _die(f"Failed to write config: {e}")

    print(f"\nConfig saved: {config_path}")

    if args.prune_recent:
        # The freshly written config defines what "live" means.
        final = config
        for plan in plans:
            final = apply_plan(final, plan)
        removed = prune_recent_models(
            live_providers=final.get("provider", {}),
            model_key_updates=final_model_key_updates(plans),
        )
        if removed:
            print(
                "Pruned from recent[]: "
                + ", ".join(f"{pid}/{mid}" for pid, mid in removed)
            )
        else:
            print("recent[]: nothing to prune.")
    _report_update_check(update_check_enabled)
    return 0


def final_model_key_updates(plans: List[ProviderPlan]) -> Dict[str, str]:
    """Merge every plan's model/small_model pointer updates, last plan wins."""
    merged: Dict[str, str] = {}
    for plan in plans:
        merged.update(plan.model_key_updates)
    return merged


if __name__ == "__main__":
    sys.exit(main())
