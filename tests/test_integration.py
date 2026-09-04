"""Integration tests: real HTTP server + real config file I/O."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from opencode_sync.cli import main
from opencode_sync.config import load_config
from tests.conftest import SAMPLE_CONFIG, SAMPLE_JSONC


def write_sample_config(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(SAMPLE_JSONC, encoding="utf-8")


# ---------------------------------------------------------------------------
# Full sync workflow
# ---------------------------------------------------------------------------

class TestFullSync:
    def test_adds_new_model(self, tmp_path, mock_server):
        cfg = tmp_path / "opencode.jsonc"
        write_sample_config(cfg)

        srv = mock_server(["org/model-a", "org/model-b", "org/model-c"])
        rc = main(["--config", str(cfg), "--host", "127.0.0.1", "--port", str(srv.port)])
        assert rc == 0

        updated = load_config(cfg)
        assert "org/model-c" in updated["provider"]["vllm"]["models"]

    def test_removes_old_model(self, tmp_path, mock_server):
        cfg = tmp_path / "opencode.jsonc"
        write_sample_config(cfg)

        srv = mock_server(["org/model-a"])  # model-b is gone
        rc = main(["--config", str(cfg), "--host", "127.0.0.1", "--port", str(srv.port)])
        assert rc == 0

        updated = load_config(cfg)
        assert "org/model-b" not in updated["provider"]["vllm"]["models"]
        assert "org/model-a" in updated["provider"]["vllm"]["models"]

    def test_preserves_custom_display_names(self, tmp_path, mock_server):
        cfg = tmp_path / "opencode.jsonc"
        write_sample_config(cfg)

        srv = mock_server(["org/model-a", "org/model-b"])
        main(["--config", str(cfg), "--host", "127.0.0.1", "--port", str(srv.port)])

        updated = load_config(cfg)
        assert updated["provider"]["vllm"]["models"]["org/model-a"]["name"] == "Model A"
        assert updated["provider"]["vllm"]["models"]["org/model-b"]["name"] == "Model B"

    def test_new_model_gets_generated_name(self, tmp_path, mock_server):
        cfg = tmp_path / "opencode.jsonc"
        write_sample_config(cfg)

        srv = mock_server(["org/brand-new-model"])
        main(["--config", str(cfg), "--host", "127.0.0.1", "--port", str(srv.port)])

        updated = load_config(cfg)
        assert updated["provider"]["vllm"]["models"]["org/brand-new-model"]["name"] == "brand-new-model"

    def test_server_rename_refreshes_display_name(self, tmp_path, mock_server, capsys):
        # The reported scenario: one model ID swapped for another, the old entry
        # carrying a display name that no longer matches.  The sync infers the
        # rename, keeps the tuned settings, and gives the entry a fresh name.
        cfg = tmp_path / "opencode.jsonc"
        config = json.dumps({
            "provider": {
                "spark-2dd4": {
                    "npm": "@ai-sdk/openai-compatible",
                    "name": "spark-2dd4",
                    "options": {"baseURL": "http://localhost:8000/v1"},
                    "models": {
                        "deepseek-v4-flash": {
                            "name": "DeepSeek-V4-Flash-0731",
                            "limit": {"context": 262144, "output": 32768},
                        },
                    },
                },
            },
            "model": "spark-2dd4/deepseek-v4-flash",
        }, indent=2)
        cfg.write_text(config, encoding="utf-8")

        srv = mock_server(["glm-5.3-flash-exl3-v2"])
        rc = main([
            "--config", str(cfg),
            "--host", "127.0.0.1", "--port", str(srv.port),
        ])
        assert rc == 0

        updated = load_config(cfg)
        entry = updated["provider"]["spark-2dd4"]["models"]["glm-5.3-flash-exl3-v2"]
        assert "deepseek-v4-flash" not in updated["provider"]["spark-2dd4"]["models"]
        assert entry["name"] == "glm-5.3-flash-exl3-v2", (
            "display name must follow the renamed model ID"
        )
        assert entry["limit"] == {"context": 262144, "output": 32768}
        assert updated["model"] == "spark-2dd4/glm-5.3-flash-exl3-v2"
        # The inference is reported, including which keys were kept.
        out = capsys.readouterr().out
        assert "~ Renamed: deepseek-v4-flash -> glm-5.3-flash-exl3-v2" in out
        assert "kept limit" in out

    def test_server_rename_via_flag_refreshes_display_name(self, tmp_path, mock_server):
        cfg = tmp_path / "opencode.jsonc"
        write_sample_config(cfg)

        srv = mock_server(["org/renamed-a"])
        rc = main([
            "--config", str(cfg),
            "--host", "127.0.0.1", "--port", str(srv.port),
            "--rename", "org/model-a=org/renamed-a",
        ])
        assert rc == 0

        updated = load_config(cfg)
        entry = updated["provider"]["vllm"]["models"]["org/renamed-a"]
        assert entry["name"] == "renamed-a"  # was "Model A"
        assert "org/model-a" not in updated["provider"]["vllm"]["models"]

    def test_updates_base_url_when_host_specified(self, tmp_path, mock_server):
        cfg = tmp_path / "opencode.jsonc"
        write_sample_config(cfg)

        srv = mock_server(["org/model-a"])
        main(["--config", str(cfg), "--host", "127.0.0.1", "--port", str(srv.port)])

        updated = load_config(cfg)
        expected_url = f"http://127.0.0.1:{srv.port}/v1"
        assert updated["provider"]["vllm"]["options"]["baseURL"] == expected_url

    def test_no_url_update_flag_preserves_base_url(self, tmp_path, mock_server):
        cfg = tmp_path / "opencode.jsonc"
        write_sample_config(cfg)
        original_url = SAMPLE_CONFIG["provider"]["vllm"]["options"]["baseURL"]

        srv = mock_server(["org/model-a"])
        main([
            "--config", str(cfg),
            "--host", "127.0.0.1",
            "--port", str(srv.port),
            "--no-url-update",
        ])

        updated = load_config(cfg)
        assert updated["provider"]["vllm"]["options"]["baseURL"] == original_url

    def test_active_model_updated_when_removed(self, tmp_path, mock_server):
        cfg = tmp_path / "opencode.jsonc"
        write_sample_config(cfg)
        # Config has model="org/model-a"; serve only model-b
        srv = mock_server(["org/model-b"])
        main(["--config", str(cfg), "--host", "127.0.0.1", "--port", str(srv.port)])

        updated = load_config(cfg)
        assert updated["model"] == "vllm/org/model-b"
        assert updated["small_model"] == "vllm/org/model-b"

    def test_no_model_update_flag(self, tmp_path, mock_server):
        cfg = tmp_path / "opencode.jsonc"
        write_sample_config(cfg)

        srv = mock_server(["org/model-b"])
        main([
            "--config", str(cfg),
            "--host", "127.0.0.1", "--port", str(srv.port),
            "--no-model-update",
        ])

        updated = load_config(cfg)
        # model-a is gone but --no-model-update keeps the stale reference
        assert updated["model"] == "vllm/org/model-a"


# ---------------------------------------------------------------------------
# --default-model / --default-small-model / --prune-recent end to end
# ---------------------------------------------------------------------------

NO_POINTER_JSONC = """\
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "vllm": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "vLLM (local)",
      "options": {
        "baseURL": "http://localhost:8080/v1"
      },
      "models": {
        "org/model-a": {
          "name": "Model A",
          "limit": { "context": 8192, "output": 4096 }
        }
      }
    }
  }
}
"""


def write_no_pointer_config(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(NO_POINTER_JSONC, encoding="utf-8")


class TestDefaultModelFlags:
    def test_auto_mode_adds_pointer_on_model_change(self, tmp_path, mock_server):
        cfg = tmp_path / "opencode.jsonc"
        write_no_pointer_config(cfg)

        # model-a is gone; server now serves model-b. No 'model' key existed.
        srv = mock_server(["org/model-b"])
        rc = main([
            "--config", str(cfg),
            "--host", "127.0.0.1", "--port", str(srv.port),
            "--default-model", "auto",
            "--default-small-model", "auto",
        ])
        assert rc == 0

        updated = load_config(cfg)
        assert updated["model"] == "vllm/org/model-b"
        assert updated["small_model"] == "vllm/org/model-b"

    def test_default_mode_does_not_add_pointer(self, tmp_path, mock_server):
        cfg = tmp_path / "opencode.jsonc"
        write_no_pointer_config(cfg)

        srv = mock_server(["org/model-b"])
        main(["--config", str(cfg), "--host", "127.0.0.1", "--port", str(srv.port)])

        updated = load_config(cfg)
        assert "model" not in updated  # legacy behavior preserved

    def test_none_mode_keeps_stale_pointer_untouched(self, tmp_path, mock_server):
        cfg = tmp_path / "opencode.jsonc"
        write_sample_config(cfg)  # has model=vllm/org/model-a

        srv = mock_server(["org/model-b"])
        main([
            "--config", str(cfg),
            "--host", "127.0.0.1", "--port", str(srv.port),
            "--default-model", "none",
            "--default-small-model", "none",
        ])

        updated = load_config(cfg)
        assert updated["model"] == "vllm/org/model-a"  # stale, but untouched

    def test_explicit_pointer_written_and_preserved(self, tmp_path, mock_server):
        cfg = tmp_path / "opencode.jsonc"
        write_no_pointer_config(cfg)

        srv = mock_server(["org/model-a"])
        rc = main([
            "--config", str(cfg),
            "--host", "127.0.0.1", "--port", str(srv.port),
            "--default-model", "vllm/org/model-a",
            "--no-url-update",  # keep baseURL stable so run 2 is byte-identical
        ])
        assert rc == 0

        updated = load_config(cfg)
        assert updated["model"] == "vllm/org/model-a"

        # Idempotent: second run must not rewrite (byte-identical shortcut).
        text_after_first = cfg.read_text()
        srv2 = mock_server(["org/model-a"])
        rc2 = main([
            "--config", str(cfg),
            "--host", "127.0.0.1", "--port", str(srv2.port),
            "--default-model", "vllm/org/model-a",
            "--no-url-update",
        ])
        assert rc2 == 0
        assert cfg.read_text() == text_after_first

    def test_invalid_mode_dies(self, tmp_path, mock_server):
        cfg = tmp_path / "opencode.jsonc"
        write_no_pointer_config(cfg)

        srv = mock_server(["org/model-a"])
        with pytest.raises(SystemExit):
            main([
                "--config", str(cfg),
                "--host", "127.0.0.1", "--port", str(srv.port),
                "--default-model", "yolo",
            ])

    def test_auto_noop_sync_does_not_invent_pointer(self, tmp_path, mock_server):
        cfg = tmp_path / "opencode.jsonc"
        write_no_pointer_config(cfg)

        # Model set unchanged → nothing happens, pointer NOT invented.
        srv = mock_server(["org/model-a"])
        rc = main([
            "--config", str(cfg),
            "--host", "127.0.0.1", "--port", str(srv.port),
            "--default-model", "auto",
        ])
        assert rc == 0

        updated = load_config(cfg)
        assert "model" not in updated


class TestPruneRecentFlag:
    def test_prunes_stale_entries_after_sync(self, tmp_path, mock_server, monkeypatch):
        cfg = tmp_path / "opencode.jsonc"
        write_no_pointer_config(cfg)
        monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
        state = tmp_path / "opencode" / "model.json"
        state.parent.mkdir(parents=True, exist_ok=True)
        state.write_text(json.dumps({
            "recent": [
                {"providerID": "vllm", "modelID": "org/dead"},
                {"providerID": "vllm", "modelID": "org/model-a"},
            ],
            "favorite": [],
        }), encoding="utf-8")

        srv = mock_server(["org/model-a", "org/model-b"])  # org/dead is gone
        rc = main([
            "--config", str(cfg),
            "--host", "127.0.0.1", "--port", str(srv.port),
            "--prune-recent",
        ])
        assert rc == 0

        data = json.loads(state.read_text(encoding="utf-8"))
        assert [e["modelID"] for e in data["recent"]] == ["org/model-a"]

    def test_prune_skipped_on_dry_run(self, tmp_path, mock_server, monkeypatch):
        cfg = tmp_path / "opencode.jsonc"
        write_no_pointer_config(cfg)
        monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
        state = tmp_path / "opencode" / "model.json"
        state.parent.mkdir(parents=True, exist_ok=True)
        state.write_text(json.dumps({
            "recent": [{"providerID": "vllm", "modelID": "org/dead"}],
        }), encoding="utf-8")

        srv = mock_server(["org/model-a"])
        main([
            "--config", str(cfg),
            "--host", "127.0.0.1", "--port", str(srv.port),
            "--prune-recent",
            "--dry-run",
        ])
        # dry-run returns before writing; state file untouched
        data = json.loads(state.read_text(encoding="utf-8"))
        assert data["recent"] == [{"providerID": "vllm", "modelID": "org/dead"}]


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------

class TestDryRun:
    def test_dry_run_does_not_write_config(self, tmp_path, mock_server):
        cfg = tmp_path / "opencode.jsonc"
        write_sample_config(cfg)
        original_text = cfg.read_text()

        srv = mock_server(["org/new-model"])
        rc = main([
            "--config", str(cfg),
            "--host", "127.0.0.1", "--port", str(srv.port),
            "--dry-run",
        ])

        assert rc == 0
        assert cfg.read_text() == original_text

    def test_dry_run_returns_zero(self, tmp_path, mock_server):
        cfg = tmp_path / "opencode.jsonc"
        write_sample_config(cfg)

        srv = mock_server(["org/model-a"])
        rc = main(["--config", str(cfg), "--host", "127.0.0.1", "--port", str(srv.port), "--dry-run"])
        assert rc == 0


# ---------------------------------------------------------------------------
# Connection failure
# ---------------------------------------------------------------------------

class TestConnectionFailure:
    def test_exits_nonzero_when_server_down(self, tmp_path):
        cfg = tmp_path / "opencode.jsonc"
        write_sample_config(cfg)

        with pytest.raises(SystemExit) as exc_info:
            main(["--config", str(cfg), "--host", "127.0.0.1", "--port", "1"])
        assert exc_info.value.code != 0

    def test_http_error_from_server(self, tmp_path, mock_server):
        cfg = tmp_path / "opencode.jsonc"
        write_sample_config(cfg)

        srv = mock_server([], fail_with=500)
        with pytest.raises(SystemExit) as exc_info:
            main(["--config", str(cfg), "--host", "127.0.0.1", "--port", str(srv.port)])
        assert exc_info.value.code != 0


# ---------------------------------------------------------------------------
# Uses existing baseURL from config (no --host/--port)
# ---------------------------------------------------------------------------

class TestUsesExistingUrl:
    def test_queries_existing_base_url(self, tmp_path, mock_server):
        srv = mock_server(["org/model-a", "org/model-b", "org/model-x"])
        cfg = tmp_path / "opencode.jsonc"

        # Config points at the mock server's actual address
        config = {
            "$schema": "https://opencode.ai/config.json",
            "provider": {
                "vllm": {
                    "npm": "@ai-sdk/openai-compatible",
                    "name": "vLLM (local)",
                    "options": {"baseURL": srv.base_url},
                    "models": {
                        "org/model-a": {"name": "Model A"},
                        "org/model-b": {"name": "Model B"},
                    },
                }
            },
            "model": "org/model-a",
            "small_model": "org/model-a",
        }
        cfg.write_text(json.dumps(config, indent=2), encoding="utf-8")

        rc = main(["--config", str(cfg)])  # no --host / --port
        assert rc == 0

        updated = load_config(cfg)
        assert "org/model-x" in updated["provider"]["vllm"]["models"]
        # baseURL should NOT have changed (no explicit host/port given)
        assert updated["provider"]["vllm"]["options"]["baseURL"] == srv.base_url

    def test_base_url_unchanged_when_no_host_port(self, tmp_path, mock_server):
        srv = mock_server(["org/model-a"])
        cfg = tmp_path / "opencode.jsonc"
        original_url = srv.base_url

        config = {
            "provider": {
                "vllm": {
                    "npm": "@ai-sdk/openai-compatible",
                    "name": "vLLM",
                    "options": {"baseURL": original_url},
                    "models": {"org/model-a": {"name": "Model A"}},
                }
            },
            "model": "org/model-a",
        }
        cfg.write_text(json.dumps(config, indent=2), encoding="utf-8")

        main(["--config", str(cfg)])
        updated = load_config(cfg)
        assert updated["provider"]["vllm"]["options"]["baseURL"] == original_url


# ---------------------------------------------------------------------------
# Environment target fallback
# ---------------------------------------------------------------------------

class TestEnvTarget:
    def test_env_host_port_used_when_no_cli_target(self, tmp_path, mock_server, monkeypatch):
        cfg = tmp_path / "opencode.jsonc"
        write_sample_config(cfg)

        srv = mock_server(["org/env-model"])
        monkeypatch.setenv("LLAMA_ARG_HOST", "127.0.0.1")
        monkeypatch.setenv("LLAMA_ARG_PORT", str(srv.port))

        rc = main(["--config", str(cfg)])
        assert rc == 0

        updated = load_config(cfg)
        assert "org/env-model" in updated["provider"]["vllm"]["models"]
        assert (
            updated["provider"]["vllm"]["options"]["baseURL"]
            == f"http://127.0.0.1:{srv.port}/v1"
        )

    def test_cli_target_overrides_env_target(self, tmp_path, mock_server, monkeypatch):
        cfg = tmp_path / "opencode.jsonc"
        write_sample_config(cfg)

        srv = mock_server(["org/cli-model"])
        monkeypatch.setenv("LLAMA_ARG_HOST", "127.0.0.1")
        monkeypatch.setenv("LLAMA_ARG_PORT", "1")

        rc = main([
            "--config", str(cfg),
            "--host", "127.0.0.1", "--port", str(srv.port),
        ])
        assert rc == 0

        updated = load_config(cfg)
        assert "org/cli-model" in updated["provider"]["vllm"]["models"]
        assert (
            updated["provider"]["vllm"]["options"]["baseURL"]
            == f"http://127.0.0.1:{srv.port}/v1"
        )

    def test_invalid_env_port_exits_nonzero(self, tmp_path, monkeypatch):
        cfg = tmp_path / "opencode.jsonc"
        write_sample_config(cfg)
        monkeypatch.setenv("LLAMA_ARG_PORT", "not-a-port")

        with pytest.raises(SystemExit) as exc_info:
            main(["--config", str(cfg)])
        assert exc_info.value.code != 0


# ---------------------------------------------------------------------------
# New config creation
# ---------------------------------------------------------------------------

class TestNewConfig:
    def test_creates_config_file_if_missing(self, tmp_path, mock_server):
        cfg = tmp_path / "new" / "opencode.jsonc"
        assert not cfg.exists()

        srv = mock_server(["org/model-a"])
        rc = main([
            "--config", str(cfg),
            "--host", "127.0.0.1", "--port", str(srv.port),
            "--provider", "vllm",
        ])
        assert rc == 0
        assert cfg.exists()

    def test_new_config_contains_models(self, tmp_path, mock_server):
        cfg = tmp_path / "opencode.jsonc"
        srv = mock_server(["org/model-a", "org/model-b"])
        main([
            "--config", str(cfg),
            "--host", "127.0.0.1", "--port", str(srv.port),
            "--provider", "vllm",
        ])

        updated = load_config(cfg)
        assert "org/model-a" in updated["provider"]["vllm"]["models"]
        assert "org/model-b" in updated["provider"]["vllm"]["models"]


# ---------------------------------------------------------------------------
# Provider selection
# ---------------------------------------------------------------------------

class TestProviderSelection:
    def test_explicit_provider_flag(self, tmp_path, mock_server):
        cfg = tmp_path / "opencode.jsonc"
        write_sample_config(cfg)

        srv = mock_server(["org/model-z"])
        main([
            "--config", str(cfg),
            "--host", "127.0.0.1", "--port", str(srv.port),
            "--provider", "vllm",
        ])

        updated = load_config(cfg)
        assert "org/model-z" in updated["provider"]["vllm"]["models"]

    def test_multiple_providers_requires_flag(self, tmp_path, mock_server):
        cfg = tmp_path / "opencode.jsonc"
        config = {
            "provider": {
                "provider-a": {"options": {"baseURL": "http://a/v1"}, "models": {}},
                "provider-b": {"options": {"baseURL": "http://b/v1"}, "models": {}},
            }
        }
        cfg.write_text(json.dumps(config, indent=2), encoding="utf-8")

        srv = mock_server(["org/model-a"])
        with pytest.raises(SystemExit) as exc_info:
            main(["--config", str(cfg), "--host", "127.0.0.1", "--port", str(srv.port)])
        assert exc_info.value.code != 0

    def test_single_provider_auto_detected(self, tmp_path, mock_server):
        cfg = tmp_path / "opencode.jsonc"
        write_sample_config(cfg)

        srv = mock_server(["org/model-a"])
        rc = main(["--config", str(cfg), "--host", "127.0.0.1", "--port", str(srv.port)])
        assert rc == 0  # should succeed without --provider

    def test_empty_model_list_no_write(self, tmp_path, mock_server, capsys):
        cfg = tmp_path / "opencode.jsonc"
        write_sample_config(cfg)
        original_text = cfg.read_text()

        srv = mock_server([])  # no models
        rc = main(["--config", str(cfg), "--host", "127.0.0.1", "--port", str(srv.port)])
        assert rc == 0
        assert cfg.read_text() == original_text
