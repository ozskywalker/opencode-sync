"""Unit tests for opencode_sync.config."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from opencode_sync.config import (
    _strip_jsonc_comments,
    apply_plan,
    find_config_path,
    find_provider_by_url,
    generate_display_name,
    load_config,
    parse_jsonc,
    plan_provider_update,
    prune_recent_models,
    save_config,
    state_model_json_path,
    update_provider_models,
)
from tests.conftest import SAMPLE_CONFIG, SAMPLE_JSONC


# ---------------------------------------------------------------------------
# JSONC comment stripping
# ---------------------------------------------------------------------------

class TestStripJsoncComments:
    def test_single_line_comment_removed(self):
        text = '{"a": 1} // this is a comment\n'
        result = _strip_jsonc_comments(text)
        assert "//" not in result
        assert "this is a comment" not in result

    def test_block_comment_removed(self):
        text = '{"a": /* block comment */ 1}'
        result = _strip_jsonc_comments(text)
        assert "block comment" not in result
        assert "/*" not in result

    def test_url_in_string_preserved(self):
        text = '{"url": "http://example.com:8000/v1"}'
        result = _strip_jsonc_comments(text)
        assert "http://example.com:8000/v1" in result

    def test_double_slash_in_string_preserved(self):
        text = '{"key": "value // not a comment"}'
        result = _strip_jsonc_comments(text)
        assert "// not a comment" in result

    def test_escaped_quote_in_string(self):
        text = r'{"key": "he said \"hello\""}'
        result = _strip_jsonc_comments(text)
        assert r'\"hello\"' in result

    def test_comment_on_own_line(self):
        text = '{\n  // standalone comment\n  "a": 1\n}'
        result = _strip_jsonc_comments(text)
        assert "standalone comment" not in result
        assert '"a": 1' in result

    def test_multiple_comments(self):
        text = '{"a": 1} // c1\n// c2\n{"b": 2}'
        result = _strip_jsonc_comments(text)
        assert "c1" not in result and "c2" not in result

    def test_block_comment_spanning_lines(self):
        text = '{"a":\n/* line1\nline2 */\n1}'
        result = _strip_jsonc_comments(text)
        assert "line1" not in result
        assert "line2" not in result

    def test_empty_string(self):
        assert _strip_jsonc_comments("") == ""

    def test_no_comments_unchanged(self):
        text = '{"key": "value", "n": 42}'
        assert _strip_jsonc_comments(text) == text

    def test_unterminated_block_comment_does_not_crash(self):
        text = '{"a": 1 /* unterminated'
        # Should not raise; consumes to EOF
        _strip_jsonc_comments(text)


class TestTrailingCommas:
    """Trailing-comma handling, exercised through the public parse_jsonc API.

    The masker itself is covered in test_jsonc_edit.py.
    """

    def test_trailing_comma_before_brace(self):
        assert parse_jsonc('{"a": 1,}') == {"a": 1}

    def test_trailing_comma_before_bracket(self):
        assert parse_jsonc('{"a": [1, 2, 3,]}') == {"a": [1, 2, 3]}

    def test_trailing_comma_with_whitespace(self):
        assert parse_jsonc('{"a": 1,  \n}') == {"a": 1}

    def test_no_trailing_comma_unchanged(self):
        assert parse_jsonc('{"a": 1}') == {"a": 1}

    def test_comma_inside_a_string_value_survives(self):
        # Regression: the old regex stripper was not string-aware and silently
        # turned {"a": "x,  }"} into {"a": "x  }"}.
        assert parse_jsonc('{"a": "x,  }"}') == {"a": "x,  }"}


# ---------------------------------------------------------------------------
# parse_jsonc
# ---------------------------------------------------------------------------

class TestParseJsonc:
    def test_basic_json(self):
        assert parse_jsonc('{"a": 1}') == {"a": 1}

    def test_with_single_line_comments(self):
        text = '{\n  // comment\n  "a": 1\n}'
        assert parse_jsonc(text) == {"a": 1}

    def test_with_block_comments(self):
        text = '{"a": /* block */ 1}'
        assert parse_jsonc(text) == {"a": 1}

    def test_with_trailing_commas(self):
        text = '{"a": 1, "b": 2,}'
        assert parse_jsonc(text) == {"a": 1, "b": 2}

    def test_real_world_sample(self):
        result = parse_jsonc(SAMPLE_JSONC)
        assert result["provider"]["vllm"]["options"]["baseURL"] == "http://localhost:8080/v1"
        assert "org/model-a" in result["provider"]["vllm"]["models"]

    def test_invalid_json_raises(self):
        with pytest.raises(json.JSONDecodeError):
            parse_jsonc("{invalid}")

    def test_url_preserved_in_string(self):
        text = '{"url": "http://host:8000/v1"}'
        result = parse_jsonc(text)
        assert result["url"] == "http://host:8000/v1"


# ---------------------------------------------------------------------------
# find_config_path
# ---------------------------------------------------------------------------

class TestFindConfigPath:
    def test_existing_xdg_path(self, tmp_path):
        cfg = tmp_path / "opencode" / "opencode.jsonc"
        cfg.parent.mkdir(parents=True)
        cfg.write_text("{}")
        with patch.dict(os.environ, {"XDG_CONFIG_HOME": str(tmp_path)}):
            result = find_config_path()
        assert result == cfg

    def test_home_config_fallback(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_CONFIG_HOME", "")
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
        cfg = tmp_path / ".config" / "opencode" / "opencode.jsonc"
        cfg.parent.mkdir(parents=True)
        cfg.write_text("{}")
        result = find_config_path()
        assert result == cfg

    def test_returns_candidate_even_if_not_existing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_CONFIG_HOME", "")
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
        result = find_config_path()
        # Should return *something*, not None
        assert result is not None

    def test_windows_uses_home_config_not_appdata(self, tmp_path, monkeypatch):
        # opencode uses xdg-basedir on Windows; APPDATA is NOT in the load order.
        monkeypatch.setenv("XDG_CONFIG_HOME", "")
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
        appdata_cfg = tmp_path / "AppData" / "Roaming" / "opencode" / "opencode.jsonc"
        appdata_cfg.parent.mkdir(parents=True)
        appdata_cfg.write_text("{}")
        home_cfg = tmp_path / ".config" / "opencode" / "opencode.jsonc"
        home_cfg.parent.mkdir(parents=True)
        home_cfg.write_text("{}")
        with patch.dict(os.environ, {"APPDATA": str(tmp_path / "AppData" / "Roaming")}):
            result = find_config_path()
        assert result == home_cfg

    def test_windows_appdata_never_a_candidate(self, tmp_path, monkeypatch):
        # Even if APPDATA path exists and ~/.config doesn't, APPDATA is not returned.
        monkeypatch.setenv("XDG_CONFIG_HOME", "")
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
        appdata_cfg = tmp_path / "AppData" / "Roaming" / "opencode" / "opencode.jsonc"
        appdata_cfg.parent.mkdir(parents=True)
        appdata_cfg.write_text("{}")
        with patch.dict(os.environ, {"APPDATA": str(tmp_path / "AppData" / "Roaming")}):
            result = find_config_path()
        # Should fall back to ~/.config default, not APPDATA
        assert result == tmp_path / ".config" / "opencode" / "opencode.jsonc"

    def test_macos_uses_home_config_not_library(self, tmp_path, monkeypatch):
        # opencode uses xdg-basedir on macOS; Library/Application Support is NOT in the load order.
        monkeypatch.setenv("XDG_CONFIG_HOME", "")
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
        lib_cfg = tmp_path / "Library" / "Application Support" / "opencode" / "opencode.jsonc"
        lib_cfg.parent.mkdir(parents=True)
        lib_cfg.write_text("{}")
        home_cfg = tmp_path / ".config" / "opencode" / "opencode.jsonc"
        home_cfg.parent.mkdir(parents=True)
        home_cfg.write_text("{}")
        result = find_config_path()
        assert result == home_cfg

    def test_macos_library_never_a_candidate(self, tmp_path, monkeypatch):
        # Even if Library path exists and ~/.config doesn't, Library is not returned.
        monkeypatch.setenv("XDG_CONFIG_HOME", "")
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
        lib_cfg = tmp_path / "Library" / "Application Support" / "opencode" / "opencode.jsonc"
        lib_cfg.parent.mkdir(parents=True)
        lib_cfg.write_text("{}")
        result = find_config_path()
        assert result == tmp_path / ".config" / "opencode" / "opencode.jsonc"

    def test_fresh_windows_machine_fallback_is_home_config(self, tmp_path, monkeypatch):
        # On a fresh Windows machine with no config, the creation target must be ~/.config.
        monkeypatch.setenv("XDG_CONFIG_HOME", "")
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
        with patch.dict(os.environ, {"APPDATA": r"C:\Users\user\AppData\Roaming",
                                     "LOCALAPPDATA": r"C:\Users\user\AppData\Local"}):
            result = find_config_path()
        assert result == tmp_path / ".config" / "opencode" / "opencode.jsonc"


# ---------------------------------------------------------------------------
# load_config / save_config
# ---------------------------------------------------------------------------

class TestLoadSaveConfig:
    def test_load_jsonc(self, tmp_path):
        cfg = tmp_path / "opencode.jsonc"
        cfg.write_text(SAMPLE_JSONC, encoding="utf-8")
        result = load_config(cfg)
        assert result["model"] == "vllm/org/model-a"
        assert "vllm" in result["provider"]

    def test_save_and_reload(self, tmp_path):
        cfg = tmp_path / "opencode.jsonc"
        save_config(cfg, SAMPLE_CONFIG)
        reloaded = load_config(cfg)
        assert reloaded == SAMPLE_CONFIG

    def test_save_creates_parent_dirs(self, tmp_path):
        cfg = tmp_path / "a" / "b" / "opencode.jsonc"
        save_config(cfg, {"key": "val"})
        assert cfg.exists()

    def test_save_writes_valid_json(self, tmp_path):
        cfg = tmp_path / "opencode.jsonc"
        save_config(cfg, {"x": [1, 2, 3]})
        parsed = json.loads(cfg.read_text())
        assert parsed == {"x": [1, 2, 3]}

    def test_save_ends_with_newline(self, tmp_path):
        cfg = tmp_path / "opencode.jsonc"
        save_config(cfg, {})
        assert cfg.read_text().endswith("\n")

    def test_save_uses_indent_2(self, tmp_path):
        cfg = tmp_path / "opencode.jsonc"
        save_config(cfg, {"a": {"b": 1}})
        text = cfg.read_text()
        assert '  "b"' in text  # two-space indent


# ---------------------------------------------------------------------------
# generate_display_name
# ---------------------------------------------------------------------------

class TestGenerateDisplayName:
    def test_strips_org_prefix(self):
        assert generate_display_name("Qwen/Qwen3-27B") == "Qwen3-27B"

    def test_no_org_prefix(self):
        assert generate_display_name("llama-3-8b") == "llama-3-8b"

    def test_multiple_slashes_uses_last_segment(self):
        assert generate_display_name("a/b/c") == "c"

    def test_empty_string(self):
        assert generate_display_name("") == ""


# ---------------------------------------------------------------------------
# find_provider_by_url
# ---------------------------------------------------------------------------

class TestFindProviderByUrl:
    def test_finds_matching_provider(self):
        config = {
            "provider": {
                "vllm": {"options": {"baseURL": "http://localhost:8080/v1"}},
            }
        }
        assert find_provider_by_url(config, "http://localhost:8080/v1") == "vllm"

    def test_trailing_slash_ignored(self):
        config = {
            "provider": {
                "vllm": {"options": {"baseURL": "http://localhost:8080/v1/"}},
            }
        }
        assert find_provider_by_url(config, "http://localhost:8080/v1") == "vllm"

    def test_no_match_returns_none(self):
        config = {
            "provider": {
                "vllm": {"options": {"baseURL": "http://other:9000/v1"}},
            }
        }
        assert find_provider_by_url(config, "http://localhost:8080/v1") is None

    def test_empty_config_returns_none(self):
        assert find_provider_by_url({}, "http://localhost:8080/v1") is None

    def test_no_options_key_returns_none(self):
        config = {"provider": {"vllm": {}}}
        assert find_provider_by_url(config, "http://localhost:8080/v1") is None


# ---------------------------------------------------------------------------
# update_provider_models
# ---------------------------------------------------------------------------

class TestUpdateProviderModels:
    def _base(self):
        return copy.deepcopy(SAMPLE_CONFIG)

    # -- added / removed reporting

    def test_added_reported(self):
        _, added, _ = update_provider_models(self._base(), "vllm", ["org/model-a", "org/model-c"])
        assert added == ["org/model-c"]

    def test_removed_reported(self):
        _, _, removed = update_provider_models(self._base(), "vllm", ["org/model-a"])
        assert removed == ["org/model-b"]

    def test_no_changes(self):
        _, added, removed = update_provider_models(
            self._base(), "vllm", ["org/model-a", "org/model-b"]
        )
        assert added == [] and removed == []

    # -- model list updated

    def test_models_dict_updated(self):
        updated, _, _ = update_provider_models(self._base(), "vllm", ["org/model-c"])
        models = updated["provider"]["vllm"]["models"]
        assert list(models.keys()) == ["org/model-c"]

    def test_model_order_preserved(self):
        ids = ["z-model", "a-model", "m-model"]
        updated, _, _ = update_provider_models(self._base(), "vllm", ids)
        assert list(updated["provider"]["vllm"]["models"].keys()) == ids

    # -- display name handling

    def test_existing_display_name_preserved(self):
        updated, _, _ = update_provider_models(
            self._base(), "vllm", ["org/model-a", "org/model-b"]
        )
        assert updated["provider"]["vllm"]["models"]["org/model-a"]["name"] == "Model A"
        assert updated["provider"]["vllm"]["models"]["org/model-b"]["name"] == "Model B"

    def test_new_model_gets_generated_name(self):
        updated, _, _ = update_provider_models(self._base(), "vllm", ["org/new-model"])
        assert updated["provider"]["vllm"]["models"]["org/new-model"]["name"] == "new-model"

    # -- baseURL update

    def test_base_url_updated_when_provided(self):
        updated, _, _ = update_provider_models(
            self._base(), "vllm", ["org/model-a"], base_url="http://remote:9000/v1"
        )
        assert updated["provider"]["vllm"]["options"]["baseURL"] == "http://remote:9000/v1"

    def test_base_url_not_touched_when_none(self):
        updated, _, _ = update_provider_models(
            self._base(), "vllm", ["org/model-a"], base_url=None
        )
        assert updated["provider"]["vllm"]["options"]["baseURL"] == "http://localhost:8080/v1"

    # -- active model handling

    def test_active_model_unchanged_when_still_valid(self):
        # Already in provider-qualified form — left as-is
        updated, _, _ = update_provider_models(
            self._base(), "vllm", ["org/model-a", "org/model-b"]
        )
        assert updated["model"] == "vllm/org/model-a"
        assert updated["small_model"] == "vllm/org/model-a"

    def test_bare_model_id_normalized_to_qualified(self):
        # Bare model ID (written by a previous sync) is normalized to provider-qualified form
        config = copy.deepcopy(SAMPLE_CONFIG)
        config["model"] = "org/model-a"
        config["small_model"] = "org/model-a"
        updated, _, _ = update_provider_models(config, "vllm", ["org/model-a", "org/model-b"])
        assert updated["model"] == "vllm/org/model-a"
        assert updated["small_model"] == "vllm/org/model-a"

    def test_active_model_updated_when_removed(self):
        updated, _, _ = update_provider_models(self._base(), "vllm", ["org/model-b"])
        assert updated["model"] == "vllm/org/model-b"
        assert updated["small_model"] == "vllm/org/model-b"

    def test_active_model_not_updated_when_flag_false(self):
        updated, _, _ = update_provider_models(
            self._base(), "vllm", ["org/model-b"], update_active_model=False
        )
        assert updated["model"] == "vllm/org/model-a"  # unchanged even though removed

    # -- provider creation

    def test_creates_new_provider_if_missing(self):
        config = {}
        updated, _, _ = update_provider_models(config, "my-llm", ["some/model"])
        assert "my-llm" in updated["provider"]
        assert updated["provider"]["my-llm"]["npm"] == "@ai-sdk/openai-compatible"

    def test_creates_provider_with_base_url(self):
        updated, _, _ = update_provider_models(
            {}, "my-llm", ["some/model"], base_url="http://host/v1"
        )
        assert updated["provider"]["my-llm"]["options"]["baseURL"] == "http://host/v1"

    # -- original config not mutated

    def test_original_config_not_mutated(self):
        config = self._base()
        original = copy.deepcopy(config)
        update_provider_models(config, "vllm", ["org/model-c"])
        assert config == original


# ---------------------------------------------------------------------------
# Default-model pointer modes (--default-model / --default-small-model)
# ---------------------------------------------------------------------------

NO_POINTER_CONFIG = {
    "$schema": "https://opencode.ai/config.json",
    "provider": {
        "vllm": {
            "npm": "@ai-sdk/openai-compatible",
            "name": "vLLM (local)",
            "options": {"baseURL": "http://localhost:8080/v1"},
            "models": {"org/model-a": {"name": "Model A"}},
        },
        "other": {
            "npm": "@ai-sdk/openai-compatible",
            "name": "Other",
            "options": {"baseURL": "http://elsewhere:8080/v1"},
            "models": {"x/model-z": {"name": "Z"}},
        },
    },
}


class TestDefaultModelModes:
    """plan_provider_update with model_mode / small_model_mode."""

    @staticmethod
    def _plan(config, model_ids, **kw):
        kw.setdefault("model_mode", "first")
        kw.setdefault("small_model_mode", "first")
        return plan_provider_update(
            config=config, provider_id="vllm", model_ids=model_ids, **kw
        ).model_key_updates

    # -- "first" (legacy default): never invent, only repair -------------------

    def test_first_never_invents_pointer(self):
        config = copy.deepcopy(NO_POINTER_CONFIG)
        del config["provider"]["other"]  # single provider, still no pointer
        assert self._plan(config, ["org/new"]) == {}

    def test_first_repairs_stale_pointer(self):
        config = copy.deepcopy(NO_POINTER_CONFIG)
        del config["provider"]["other"]
        config["model"] = "vllm/org/dead"
        assert self._plan(config, ["org/new"]) == {"model": "vllm/org/new"}

    def test_first_ignores_other_providers_pointer(self):
        config = copy.deepcopy(NO_POINTER_CONFIG)
        config["model"] = "other/x/model-z"
        assert self._plan(config, ["org/new"]) == {}

    # -- "none": never touch, not even repair ----------------------------------

    def test_none_leaves_stale_pointer_alone(self):
        config = copy.deepcopy(NO_POINTER_CONFIG)
        del config["provider"]["other"]
        config["model"] = "vllm/org/dead"
        config["small_model"] = "vllm/org/dead"
        assert self._plan(
            config, ["org/new"], model_mode="none", small_model_mode="none"
        ) == {}

    # -- "auto": point at first served model when the set changes --------------

    def test_auto_invents_pointer_on_change(self):
        # The headline feature: fresh config, no pointer, server model changed.
        config = copy.deepcopy(NO_POINTER_CONFIG)
        del config["provider"]["other"]
        config["provider"]["vllm"]["models"] = {"org/old": {"name": "Old"}}
        assert self._plan(
            config, ["org/new"], model_mode="auto", small_model_mode="auto"
        ) == {"model": "vllm/org/new", "small_model": "vllm/org/new"}

    def test_auto_noop_when_set_unchanged_and_no_pointer(self):
        # Routine no-op sync must not invent a pointer (wrapper runs every launch).
        config = copy.deepcopy(NO_POINTER_CONFIG)
        del config["provider"]["other"]
        assert self._plan(
            config, ["org/model-a"], model_mode="auto", small_model_mode="auto"
        ) == {}

    def test_auto_keeps_pointer_when_unchanged(self):
        config = copy.deepcopy(NO_POINTER_CONFIG)
        del config["provider"]["other"]
        config["model"] = "vllm/org/model-a"
        assert self._plan(config, ["org/model-a"], model_mode="auto") == {}

    def test_auto_repairs_stale_pointer(self):
        config = copy.deepcopy(NO_POINTER_CONFIG)
        del config["provider"]["other"]
        config["model"] = "vllm/org/dead"
        assert self._plan(config, ["org/new"], model_mode="auto") == {
            "model": "vllm/org/new"
        }

    def test_auto_respects_other_providers_pointer(self):
        config = copy.deepcopy(NO_POINTER_CONFIG)
        config["model"] = "other/x/model-z"
        config["provider"]["vllm"]["models"] = {"org/old": {"name": "Old"}}
        # Set changed for vllm but the pointer belongs to 'other' — not ours.
        assert self._plan(config, ["org/new"], model_mode="auto") == {}

    def test_auto_noop_when_nothing_changed_with_existing_valid_pointer(self):
        config = copy.deepcopy(NO_POINTER_CONFIG)
        del config["provider"]["other"]
        config["model"] = "vllm/org/model-a"
        config["small_model"] = "vllm/org/model-a"
        assert self._plan(
            config, ["org/model-a"], model_mode="auto", small_model_mode="auto"
        ) == {}

    # -- explicit provider/model ------------------------------------------------

    def test_explicit_sets_pointer_even_without_change(self):
        config = copy.deepcopy(NO_POINTER_CONFIG)
        del config["provider"]["other"]
        assert self._plan(
            config, ["org/model-a"], model_mode="explicit", explicit_model="vllm/org/model-a"
        ) == {"model": "vllm/org/model-a"}

    def test_explicit_noop_when_already_correct(self):
        config = copy.deepcopy(NO_POINTER_CONFIG)
        del config["provider"]["other"]
        config["model"] = "vllm/org/model-a"
        assert self._plan(
            config, ["org/model-a"], model_mode="explicit", explicit_model="vllm/org/model-a"
        ) == {}

    def test_explicit_wrong_provider_rejected(self):
        with pytest.raises(ValueError, match="names provider"):
            plan_provider_update(
                config=NO_POINTER_CONFIG,
                provider_id="vllm",
                model_ids=["org/model-a"],
                model_mode="explicit",
                explicit_model="other/x/model-z",
            )

    def test_explicit_bare_id_rejected(self):
        # "org/model-a" looks qualified but 'org' isn't the provider being synced;
        # either way it must be rejected, never silently guessed at.
        with pytest.raises(ValueError, match="(provider-qualified|names provider)"):
            plan_provider_update(
                config=NO_POINTER_CONFIG,
                provider_id="vllm",
                model_ids=["org/model-a"],
                model_mode="explicit",
                explicit_model="org/model-a",
            )

    def test_unknown_mode_rejected(self):
        with pytest.raises(ValueError, match="unknown default-model mode"):
            plan_provider_update(
                config=NO_POINTER_CONFIG,
                provider_id="vllm",
                model_ids=["org/model-a"],
                model_mode="yolo",
            )

    # -- small_model independence -----------------------------------------------

    def test_small_model_mode_independent_of_model_mode(self):
        config = copy.deepcopy(NO_POINTER_CONFIG)
        del config["provider"]["other"]
        config["provider"]["vllm"]["models"] = {"org/old": {"name": "Old"}}
        updates = self._plan(
            config, ["org/new"], model_mode="none", small_model_mode="auto"
        )
        assert updates == {"small_model": "vllm/org/new"}

    def test_no_model_update_flag_still_wins(self):
        # update_active_model=False disables all pointer work, as before.
        config = copy.deepcopy(NO_POINTER_CONFIG)
        del config["provider"]["other"]
        config["provider"]["vllm"]["models"] = {"org/old": {"name": "Old"}}
        plan = plan_provider_update(
            config=config,
            provider_id="vllm",
            model_ids=["org/new"],
            update_active_model=False,
            model_mode="auto",
            small_model_mode="auto",
        )
        assert plan.model_key_updates == {}

    # -- noop plan accounting ----------------------------------------------------

    def test_explicit_pointer_change_makes_plan_not_noop(self):
        config = copy.deepcopy(NO_POINTER_CONFIG)
        del config["provider"]["other"]
        plan = plan_provider_update(
            config=config,
            provider_id="vllm",
            model_ids=["org/model-a"],
            model_mode="explicit",
            explicit_model="vllm/org/model-a",
        )
        assert not plan.is_noop()

    def test_first_mode_plan_is_noop_when_only_pointer_would_invent(self):
        config = copy.deepcopy(NO_POINTER_CONFIG)
        del config["provider"]["other"]
        plan = self._plan(config, ["org/model-a"])
        # (indirect: _plan returns {} so plan must be a no-op)
        plan2 = plan_provider_update(config=config, provider_id="vllm", model_ids=["org/model-a"])
        assert plan2.is_noop()
        assert plan == {}


# ---------------------------------------------------------------------------
# --prune-recent: opencode state file pruning
# ---------------------------------------------------------------------------

class TestPruneRecentModels:
    def _state_file(self, tmp_path, recent):
        path = tmp_path / "model.json"
        path.write_text(json.dumps({"recent": recent, "favorite": []}), encoding="utf-8")
        return path

    @staticmethod
    def _live():
        return {
            "vllm": {"models": {"org/model-a": {}, "org/model-b": {}}},
            "other": {"models": {"x/model-z": {}}},
        }

    def test_missing_file_is_noop(self, tmp_path):
        removed = prune_recent_models(path=tmp_path / "nope.json", live_providers=self._live())
        assert removed == []

    def test_stale_entry_removed(self, tmp_path):
        path = self._state_file(tmp_path, [
            {"providerID": "vllm", "modelID": "org/dead"},
            {"providerID": "vllm", "modelID": "org/model-a"},
        ])
        removed = prune_recent_models(path=path, live_providers=self._live())
        assert removed == [("vllm", "org/dead")]
        data = json.loads(path.read_text(encoding="utf-8"))
        assert [e["modelID"] for e in data["recent"]] == ["org/model-a"]
        assert data["favorite"] == []  # untouched

    def test_unknown_provider_entries_kept(self, tmp_path):
        path = self._state_file(tmp_path, [
            {"providerID": "mystery", "modelID": "whatever"},
        ])
        removed = prune_recent_models(path=path, live_providers=self._live())
        assert removed == []
        data = json.loads(path.read_text(encoding="utf-8"))
        assert len(data["recent"]) == 1

    def test_provider_without_models_dict_kept(self, tmp_path):
        path = self._state_file(tmp_path, [
            {"providerID": "mystery", "modelID": "whatever"},
        ])
        removed = prune_recent_models(path=path, live_providers={"mystery": {}})
        assert removed == []

    def test_no_removals_no_rewrite(self, tmp_path):
        path = self._state_file(tmp_path, [{"providerID": "vllm", "modelID": "org/model-a"}])
        before = path.read_text(encoding="utf-8")
        prune_recent_models(path=path, live_providers=self._live())
        assert path.read_text(encoding="utf-8") == before

    def test_model_key_update_seeds_recent_head(self, tmp_path):
        path = self._state_file(tmp_path, [
            {"providerID": "vllm", "modelID": "org/dead"},
        ])
        prune_recent_models(
            path=path,
            live_providers=self._live(),
            model_key_updates={"model": "vllm/org/model-b"},
        )
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["recent"][0] == {"providerID": "vllm", "modelID": "org/model-b"}

    def test_dry_run_does_not_write(self, tmp_path):
        path = self._state_file(tmp_path, [{"providerID": "vllm", "modelID": "org/dead"}])
        before = path.read_text(encoding="utf-8")
        removed = prune_recent_models(
            path=path, live_providers=self._live(), dry_run=True
        )
        assert removed == [("vllm", "org/dead")]
        assert path.read_text(encoding="utf-8") == before

    def test_corrupt_file_is_ignored(self, tmp_path):
        path = tmp_path / "model.json"
        path.write_text("{not json", encoding="utf-8")
        assert prune_recent_models(path=path, live_providers=self._live()) == []

    def test_non_list_recent_ignored(self, tmp_path):
        path = tmp_path / "model.json"
        path.write_text(json.dumps({"recent": "bogus"}), encoding="utf-8")
        assert prune_recent_models(path=path, live_providers=self._live()) == []

    def test_state_path_honors_xdg_state_home(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
        assert state_model_json_path() == tmp_path / "opencode" / "model.json"

    def test_state_path_default(self, monkeypatch):
        monkeypatch.delenv("XDG_STATE_HOME", raising=False)
        expected = Path.home() / ".local" / "state" / "opencode" / "model.json"
        assert state_model_json_path() == expected


# ---------------------------------------------------------------------------
# Renames refresh the display name
#
# Regression: moving an entry to a new model ID carried the old display name
# with it, so opencode kept showing e.g. "DeepSeek-V4-Flash-0731" after the
# server had started serving glm-5.3-flash-exl3-v2.
# ---------------------------------------------------------------------------

RENAME_CONFIG = {
    "provider": {
        "spark-2dd4": {
            "npm": "@ai-sdk/openai-compatible",
            "name": "spark-2dd4",
            "options": {"baseURL": "http://spark-2dd4:8000/v1"},
            "models": {
                "deepseek-v4-flash": {
                    "name": "DeepSeek-V4-Flash-0731",
                    "limit": {"context": 262144, "output": 32768},
                },
            },
        },
    },
}


class TestRenameRefreshesDisplayName:
    def _config(self):
        return copy.deepcopy(RENAME_CONFIG)

    def test_dict_path_refreshes_name(self):
        plan = plan_provider_update(self._config(), "spark-2dd4", ["glm-5.3-flash-exl3-v2"])
        assert plan.renames == {"deepseek-v4-flash": "glm-5.3-flash-exl3-v2"}
        updated = apply_plan(self._config(), plan)
        entry = updated["provider"]["spark-2dd4"]["models"]["glm-5.3-flash-exl3-v2"]
        assert entry["name"] == "glm-5.3-flash-exl3-v2"
        # The rest of the entry still rides along.
        assert entry["limit"] == {"context": 262144, "output": 32768}

    def test_dict_path_refreshes_name_for_explicit_renames(self):
        config = self._config()
        config["provider"]["spark-2dd4"]["models"]["extra"] = {"name": "Extra"}
        plan = plan_provider_update(
            self._config(), "spark-2dd4",
            ["glm-5.3-flash-exl3-v2", "extra"],
            renames={"deepseek-v4-flash": "glm-5.3-flash-exl3-v2"},
        )
        updated = apply_plan(config, plan)
        entry = updated["provider"]["spark-2dd4"]["models"]["glm-5.3-flash-exl3-v2"]
        assert entry["name"] == "glm-5.3-flash-exl3-v2"
        # A surviving (not-renamed) entry keeps its display name.
        assert updated["provider"]["spark-2dd4"]["models"]["extra"]["name"] == "Extra"

    def test_dict_path_keeps_untouched_entry_names(self):
        # A plain sync that adds/removes without any rename never rewrites names.
        config = self._config()
        config["provider"]["spark-2dd4"]["models"]["other"] = {"name": "Custom"}
        plan = plan_provider_update(config, "spark-2dd4", ["deepseek-v4-flash", "added"])
        assert plan.renames == {}
        updated = apply_plan(config, plan)
        models = updated["provider"]["spark-2dd4"]["models"]
        assert models["deepseek-v4-flash"]["name"] == "DeepSeek-V4-Flash-0731"
        assert models["added"]["name"] == "added"

    def test_dict_path_no_name_key_stays_nameless(self):
        # An entry that deliberately has no "name" key must not gain one.
        config = self._config()
        config["provider"]["spark-2dd4"]["models"] = {
            "old": {"limit": {"context": 1}},
        }
        plan = plan_provider_update(config, "spark-2dd4", ["new"])
        updated = apply_plan(config, plan)
        entry = updated["provider"]["spark-2dd4"]["models"]["new"]
        assert "name" not in entry
        assert entry["limit"] == {"context": 1}

    def test_dict_path_original_not_mutated_by_rename(self):
        config = self._config()
        original = copy.deepcopy(config)
        plan = plan_provider_update(config, "spark-2dd4", ["glm-5.3-flash-exl3-v2"])
        apply_plan(config, plan)
        assert config == original

    def test_rename_onto_surviving_id_refreshes_name(self):
        # Server order lists both IDs; only the renamed one is rewritten.
        config = self._config()
        config["provider"]["spark-2dd4"]["models"]["keep"] = {"name": "Keep Me"}
        plan = plan_provider_update(
            config, "spark-2dd4", ["keep", "glm-5.3-flash-exl3-v2"]
        )
        updated = apply_plan(config, plan)
        models = updated["provider"]["spark-2dd4"]["models"]
        assert models["keep"]["name"] == "Keep Me"
        assert models["glm-5.3-flash-exl3-v2"]["name"] == "glm-5.3-flash-exl3-v2"
