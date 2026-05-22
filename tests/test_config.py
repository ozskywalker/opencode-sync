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
    _strip_trailing_commas,
    find_config_path,
    find_provider_by_url,
    generate_display_name,
    load_config,
    parse_jsonc,
    save_config,
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


class TestStripTrailingCommas:
    def test_trailing_comma_before_brace(self):
        result = _strip_trailing_commas('{"a": 1,}')
        assert result == '{"a": 1}'

    def test_trailing_comma_before_bracket(self):
        result = _strip_trailing_commas("[1, 2, 3,]")
        assert result == "[1, 2, 3]"

    def test_trailing_comma_with_whitespace(self):
        result = _strip_trailing_commas('{"a": 1,  \n}')
        assert result == '{"a": 1,  \n}'.replace(",  \n}", "  \n}")

    def test_no_trailing_comma_unchanged(self):
        text = '{"a": 1}'
        assert _strip_trailing_commas(text) == text


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
        assert result["provider"]["vllm"]["options"]["baseURL"] == "http://localhost:8000/v1"
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
        with patch("opencode_sync.config.platform.system", return_value="Linux"):
            result = find_config_path()
        assert result == cfg

    def test_returns_candidate_even_if_not_existing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_CONFIG_HOME", "")
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
        with patch("opencode_sync.config.platform.system", return_value="Linux"):
            result = find_config_path()
        # Should return *something*, not None
        assert result is not None

    def test_windows_appdata_candidate(self, tmp_path):
        cfg = tmp_path / "opencode" / "opencode.jsonc"
        cfg.parent.mkdir(parents=True)
        cfg.write_text("{}")
        with patch("opencode_sync.config.platform.system", return_value="Windows"), \
             patch.dict(os.environ, {"APPDATA": str(tmp_path), "XDG_CONFIG_HOME": ""}):
            result = find_config_path()
        assert result == cfg

    def test_macos_library_candidate(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_CONFIG_HOME", "")
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
        lib_cfg = tmp_path / "Library" / "Application Support" / "opencode" / "opencode.jsonc"
        lib_cfg.parent.mkdir(parents=True)
        lib_cfg.write_text("{}")
        with patch("opencode_sync.config.platform.system", return_value="Darwin"):
            result = find_config_path()
        assert result == lib_cfg


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
                "vllm": {"options": {"baseURL": "http://localhost:8000/v1"}},
            }
        }
        assert find_provider_by_url(config, "http://localhost:8000/v1") == "vllm"

    def test_trailing_slash_ignored(self):
        config = {
            "provider": {
                "vllm": {"options": {"baseURL": "http://localhost:8000/v1/"}},
            }
        }
        assert find_provider_by_url(config, "http://localhost:8000/v1") == "vllm"

    def test_no_match_returns_none(self):
        config = {
            "provider": {
                "vllm": {"options": {"baseURL": "http://other:9000/v1"}},
            }
        }
        assert find_provider_by_url(config, "http://localhost:8000/v1") is None

    def test_empty_config_returns_none(self):
        assert find_provider_by_url({}, "http://localhost:8000/v1") is None

    def test_no_options_key_returns_none(self):
        config = {"provider": {"vllm": {}}}
        assert find_provider_by_url(config, "http://localhost:8000/v1") is None


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
        assert updated["provider"]["vllm"]["options"]["baseURL"] == "http://localhost:8000/v1"

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
