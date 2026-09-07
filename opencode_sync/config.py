"""Config domain layer for opencode-sync.

Historically this module held everything: JSONC parsing, file I/O, planning,
surgical editing, and state-file maintenance.  It is now a facade that
re-exports from focused modules so existing imports keep working:

- :mod:`opencode_sync.io`        — path discovery, reads, atomic writes, parse_jsonc
- :mod:`opencode_sync.planning`  — pure plan computation (``ProviderPlan`` lives there)
- :mod:`opencode_sync.surgical`  — span-level text editing that preserves comments
- :mod:`opencode_sync.state`     — opencode's model.json recent[] pruning

Keep the re-export list complete: the CLI and the test suite import the
domain API from here, and the surgical/planning modules must never depend
on this facade (that would make the split circular).
"""

from __future__ import annotations

from .io import (
    _atomic_write_text,
    _config_candidates,
    find_config_path,
    load_config,
    load_config_text,
    parse_jsonc,
    save_config,
    save_config_text,
)
from .planning import (
    DEFAULT_MODEL_MODES,
    ProviderPlan,
    apply_plan,
    find_provider_by_url,
    generate_display_name,
    plan_provider_update,
)
from .surgical import apply_plans_to_text
from .state import prune_recent_models, state_model_json_path

__all__ = [
    "DEFAULT_MODEL_MODES",
    "ProviderPlan",
    "_atomic_write_text",
    "_config_candidates",
    "apply_plan",
    "apply_plans_to_text",
    "find_config_path",
    "find_provider_by_url",
    "generate_display_name",
    "load_config",
    "load_config_text",
    "parse_jsonc",
    "plan_provider_update",
    "prune_recent_models",
    "save_config",
    "save_config_text",
    "state_model_json_path",
]
