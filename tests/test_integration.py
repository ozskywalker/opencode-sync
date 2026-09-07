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
        # Exit code is EXIT_NOTHING_TO_DO (2): the pointer is already right.
        text_after_first = cfg.read_text()
        srv2 = mock_server(["org/model-a"])
        rc2 = main([
            "--config", str(cfg),
            "--host", "127.0.0.1", "--port", str(srv2.port),
            "--default-model", "vllm/org/model-a",
            "--no-url-update",
        ])
        assert rc2 == 2
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

    def test_prune_skipped_on_dry_run(self, tmp_path, mock_server, monkeypatch, capsys):
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

    def test_dry_run_reports_what_would_be_pruned(self, tmp_path, mock_server, monkeypatch, capsys):
        # S3: dry-run must say what prune WOULD do, not stay silent.
        cfg = tmp_path / "opencode.jsonc"
        write_no_pointer_config(cfg)
        monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
        state = tmp_path / "opencode" / "model.json"
        state.parent.mkdir(parents=True, exist_ok=True)
        state.write_text(json.dumps({
            "recent": [{"providerID": "vllm", "modelID": "org/dead"}],
        }), encoding="utf-8")

        srv = mock_server(["org/model-a"])
        rc = main([
            "--config", str(cfg),
            "--host", "127.0.0.1", "--port", str(srv.port),
            "--prune-recent",
            "--dry-run",
        ])
        assert rc == 0
        out = capsys.readouterr().out
        assert "[dry-run] Would prune from recent[]: vllm/org/dead" in out

    def test_dry_run_prune_does_not_write_on_a_noop_sync(
        self, tmp_path, mock_server, monkeypatch, capsys
    ):
        # Regression: the no-op branch used to run the prune with dry_run=False
        # before the --dry-run short-circuit was ever reached, so a dry run
        # actually rewrote the state file.
        cfg = tmp_path / "opencode.jsonc"
        write_no_pointer_config(cfg)
        monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
        state = tmp_path / "opencode" / "model.json"
        state.parent.mkdir(parents=True, exist_ok=True)
        state.write_text(json.dumps({
            "recent": [{"providerID": "vllm", "modelID": "org/dead"}],
        }), encoding="utf-8")

        srv = mock_server(["org/model-a"])  # model set unchanged -> no-op sync
        rc = main([
            "--config", str(cfg),
            "--host", "127.0.0.1", "--port", str(srv.port),
            "--prune-recent",
            "--dry-run",
        ])
        assert rc == 0
        assert "[dry-run] Would prune from recent[]: vllm/org/dead" in capsys.readouterr().out
        data = json.loads(state.read_text(encoding="utf-8"))
        assert data["recent"] == [{"providerID": "vllm", "modelID": "org/dead"}]


    def test_prune_runs_on_noop_sync(self, tmp_path, mock_server, monkeypatch, capsys):
        # S2: stale recent[] entries shadow the default on every launch, so a
        # config that needs no write must not skip the prune.
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
        }), encoding="utf-8")

        srv = mock_server(["org/model-a"])  # model set unchanged -> no-op sync
        rc = main([
            "--config", str(cfg),
            "--host", "127.0.0.1", "--port", str(srv.port),
            "--prune-recent",
        ])
        assert rc == 0
        assert "Pruned from recent[]: vllm/org/dead" in capsys.readouterr().out
        data = json.loads(state.read_text(encoding="utf-8"))
        assert [e["modelID"] for e in data["recent"]] == ["org/model-a"]

    def test_prune_still_skipped_when_every_server_failed(self, tmp_path, monkeypatch, capsys):
        # All servers down -> nothing synced -> recent[] isn't judged against
        # a config that may not reflect reality.  Skipping is deliberate.
        cfg = tmp_path / "opencode.jsonc"
        write_no_pointer_config(cfg)
        monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
        state = tmp_path / "opencode" / "model.json"
        state.parent.mkdir(parents=True, exist_ok=True)
        state.write_text(json.dumps({
            "recent": [{"providerID": "vllm", "modelID": "org/dead"}],
        }), encoding="utf-8")

        with pytest.raises(SystemExit) as exc_info:
            main([
                "--config", str(cfg),
                "--host", "127.0.0.1", "--port", "1",
                "--prune-recent",
            ])
        assert exc_info.value.code != 0
        assert "Querying" in capsys.readouterr().out  # ran, but no plan survived
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
        # Server up but empty: nothing to apply -> EXIT_NOTHING_TO_DO.
        assert rc == 2
        assert cfg.read_text() == original_text


class TestProviderValidation:
    """Naming a provider is an explicit request: typos and missing URLs die loudly."""

    def test_nonexistent_provider_dies_and_writes_nothing(self, tmp_path, mock_server):
        cfg = tmp_path / "opencode.jsonc"
        write_sample_config(cfg)
        original_text = cfg.read_text(encoding="utf-8")

        srv = mock_server(["org/model-a"])
        with pytest.raises(SystemExit) as exc_info:
            main([
                "--config", str(cfg),
                "--host", "127.0.0.1", "--port", str(srv.port),
                "--provider", "typo-vllm",
            ])
        assert exc_info.value.code != 0
        assert cfg.read_text(encoding="utf-8") == original_text

    def test_nonexistent_provider_message_lists_known_providers(
        self, tmp_path, mock_server, capsys
    ):
        cfg = tmp_path / "opencode.jsonc"
        write_sample_config(cfg)
        srv = mock_server(["org/model-a"])
        with pytest.raises(SystemExit):
            main([
                "--config", str(cfg),
                "--host", "127.0.0.1", "--port", str(srv.port),
                "--provider", "typo-vllm",
            ])
        err = capsys.readouterr().err
        assert "typo-vllm" in err
        assert "vllm" in err

    def test_nonexistent_provider_without_host_dies(self, tmp_path, monkeypatch, capsys):
        cfg = tmp_path / "opencode.jsonc"
        write_sample_config(cfg)
        with pytest.raises(SystemExit) as exc_info:
            main(["--config", str(cfg), "--provider", "typo-vllm"])
        assert exc_info.value.code != 0
        assert "not in the config" in capsys.readouterr().err

    def test_greenfield_still_creates_provider(self, tmp_path, mock_server):
        # No config, no providers: naming one is how you bootstrap.
        cfg = tmp_path / "new" / "opencode.jsonc"
        srv = mock_server(["org/model-a"])
        rc = main([
            "--config", str(cfg),
            "--host", "127.0.0.1", "--port", str(srv.port),
            "--provider", "vllm",
        ])
        assert rc == 0
        assert "vllm" in load_config(cfg)["provider"]

    def test_provider_without_base_url_dies(self, tmp_path, capsys):
        # Explicitly named provider with no options.baseURL: nothing to query,
        # and silently falling back to a default server would be wrong.
        cfg = tmp_path / "opencode.jsonc"
        cfg.write_text(json.dumps(
            {"provider": {"myprov": {"npm": "@ai-sdk/openai-compatible", "models": {}}}},
            indent=2,
        ), encoding="utf-8")

        with pytest.raises(SystemExit) as exc_info:
            main(["--config", str(cfg), "--provider", "myprov"])
        assert exc_info.value.code != 0
        assert "no options.baseURL" in capsys.readouterr().err

    def test_provider_without_base_url_no_localhost_query(self, tmp_path, capsys):
        # Regression: this used to silently query http://localhost:8080/v1.
        cfg = tmp_path / "opencode.jsonc"
        cfg.write_text(json.dumps(
            {"provider": {"myprov": {"npm": "@ai-sdk/openai-compatible", "models": {}}}},
            indent=2,
        ), encoding="utf-8")

        with pytest.raises(SystemExit):
            main(["--config", str(cfg), "--provider", "myprov"])
        out = capsys.readouterr()
        assert "localhost:8080" not in out.out + out.err


class TestExplicitPointerValidation:
    """An explicit pointer naming a provider we aren't syncing dies before any HTTP."""

    def test_wrong_provider_pointer_dies_cleanly(self, tmp_path, mock_server, capsys):
        cfg = tmp_path / "opencode.jsonc"
        write_sample_config(cfg)
        srv = mock_server(["org/model-a"])

        with pytest.raises(SystemExit) as exc_info:
            main([
                "--config", str(cfg),
                "--host", "127.0.0.1", "--port", str(srv.port),
                "--default-model", "other/m",
            ])
        assert exc_info.value.code == 1
        err = capsys.readouterr().err
        assert "ERROR:" in err
        assert "other" in err
        assert not any(line.startswith("Traceback") for line in err.splitlines())

    def test_wrong_provider_pointer_dies_before_any_http(self, tmp_path, capsys, monkeypatch):
        # No server up at all: validation must fire before the query would.
        cfg = tmp_path / "opencode.jsonc"
        write_sample_config(cfg)

        # Guard: if the HTTP layer is reached, fail the test loudly instead of
        # hanging on connection timeouts.
        from opencode_sync import cli as cli_mod

        def boom(self, base_url, timeout=10):
            raise AssertionError("HTTP client must not be constructed before validation")

        monkeypatch.setattr(cli_mod.VLLMClient, "__init__", boom)
        with pytest.raises(SystemExit) as exc_info:
            main([
                "--config", str(cfg),
                "--host", "127.0.0.1", "--port", "1",
                "--default-model", "other/m",
            ])
        assert exc_info.value.code == 1
        assert "Querying" not in capsys.readouterr().out

    def test_wrong_small_model_pointer_dies_cleanly(self, tmp_path, mock_server, capsys):
        cfg = tmp_path / "opencode.jsonc"
        write_sample_config(cfg)
        srv = mock_server(["org/model-a"])

        with pytest.raises(SystemExit) as exc_info:
            main([
                "--config", str(cfg),
                "--host", "127.0.0.1", "--port", str(srv.port),
                "--default-small-model", "other/m",
            ])
        assert exc_info.value.code == 1
        assert "--default-small-model" in capsys.readouterr().err

    def test_pointer_at_synced_provider_is_accepted(self, tmp_path, mock_server):
        cfg = tmp_path / "opencode.jsonc"
        write_no_pointer_config(cfg)
        srv = mock_server(["org/model-a"])

        rc = main([
            "--config", str(cfg),
            "--host", "127.0.0.1", "--port", str(srv.port),
            "--default-model", "vllm/org/model-a",
        ])
        assert rc == 0
        assert load_config(cfg)["model"] == "vllm/org/model-a"

    def test_pointer_at_one_of_several_synced_providers_dies_before_http(
        self, tmp_path, capsys
    ):
        # Regression: sync-all + a pointer at ONE of the synced providers used
        # to pass the any()-precheck and then blow up with a raw ValueError
        # traceback when the planner reached a differently-named provider.
        cfg = tmp_path / "opencode.jsonc"
        cfg.write_text(json.dumps({
            "provider": {
                "a": {"npm": "@ai-sdk/openai-compatible",
                      "options": {"baseURL": "http://127.0.0.1:1/v1"}, "models": {}},
                "b": {"npm": "@ai-sdk/openai-compatible",
                      "options": {"baseURL": "http://127.0.0.1:2/v1"}, "models": {}},
            }
        }), encoding="utf-8")

        # Bare sync-all: every provider becomes a target, each against its own
        # (dead) URL. The pointer pre-check must fire before any query.
        with pytest.raises(SystemExit) as exc_info:
            main(["--config", str(cfg), "--default-model", "a/m"])
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "ERROR:" in captured.err
        assert "also covers" in captured.err
        assert "Querying" not in captured.out
        assert not any(line.startswith("Traceback") for line in captured.err.splitlines())



# ---------------------------------------------------------------------------
# C1/C2: --json output mode, --quiet/--verbose, exit-code contract
# ---------------------------------------------------------------------------

def _write_one_provider(cfg, base_url):
    cfg.write_text(json.dumps({
        "provider": {
            "vllm": {
                "npm": "@ai-sdk/openai-compatible",
                "options": {"baseURL": base_url},
                "models": {},
            }
        }
    }), encoding="utf-8")


class TestJsonOutputMode:
    def test_sync_emits_single_json_object(self, tmp_path, mock_server, capsys):
        cfg = tmp_path / "opencode.jsonc"
        srv = mock_server(["org/model-a"])
        _write_one_provider(cfg, srv.base_url)

        rc = main(["--config", str(cfg), "--json"])
        assert rc == 0
        out = capsys.readouterr().out
        payload = json.loads(out)  # parses => stdout is pure JSON (no banner)
        assert payload["result"] == "synced"
        assert payload["exit_code"] == 0
        assert payload["providers"][0]["provider"] == "vllm"
        assert "org/model-a" in payload["providers"][0]["added"]

    def test_noop_reports_nothing_to_do_with_exit_2(self, tmp_path, mock_server, capsys):
        cfg = tmp_path / "opencode.jsonc"
        srv = mock_server(["org/model-a"])
        _write_one_provider(cfg, srv.base_url)
        main(["--config", str(cfg), "--json"])

        capsys.readouterr()
        rc = main(["--config", str(cfg), "--json"])
        assert rc == 2
        payload = json.loads(capsys.readouterr().out)
        assert payload["result"] == "nothing-to-do"
        assert payload["exit_code"] == 2

    def test_all_servers_down_reports_error(self, tmp_path, capsys):
        cfg = tmp_path / "opencode.jsonc"
        _write_one_provider(cfg, "http://127.0.0.1:1/v1")

        # A server we named being down is a hard error (TestConnectionFailure
        # semantics); --json routes the failure through _die -> SystemExit(1).
        with pytest.raises(SystemExit) as exc_info:
            main(["--config", str(cfg), "--json"])
        assert exc_info.value.code == 1


    def test_dry_run_flagged_in_payload(self, tmp_path, mock_server, capsys):
        cfg = tmp_path / "opencode.jsonc"
        srv = mock_server(["org/model-a"])
        _write_one_provider(cfg, srv.base_url)

        rc = main(["--config", str(cfg), "--json", "--dry-run"])
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["dry_run"] is True
        assert payload["result"] == "synced"
        assert not payload["config"].endswith("written")  # path, not a write claim

    def test_pruned_entries_in_payload(self, tmp_path, mock_server, monkeypatch, capsys):
        cfg = tmp_path / "opencode.jsonc"
        srv = mock_server(["org/model-a"])
        _write_one_provider(cfg, srv.base_url)
        cfg.write_text(
            json.dumps({
                "provider": {"vllm": {
                    "npm": "@ai-sdk/openai-compatible",
                    "options": {"baseURL": srv.base_url},
                    "models": {"org/model-a": {"name": "a"}},
                }}
            }),
            encoding="utf-8",
        )
        monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
        state = tmp_path / "opencode" / "model.json"
        state.parent.mkdir(parents=True, exist_ok=True)
        state.write_text(json.dumps({
            "recent": [{"providerID": "vllm", "modelID": "org/dead"}],
        }), encoding="utf-8")

        rc = main(["--config", str(cfg), "--json", "--prune-recent"])
        # The model set is unchanged, but the prune still ran: nothing-to-do.
        assert rc == 2
        payload = json.loads(capsys.readouterr().out)
        assert payload["pruned_recent"] == [["vllm", "org/dead"]]



class TestExitCodeContract:
    def test_write_run_exits_zero(self, tmp_path, mock_server):
        cfg = tmp_path / "opencode.jsonc"
        srv = mock_server(["org/model-a"])
        _write_one_provider(cfg, srv.base_url)
        assert main(["--config", str(cfg)]) == 0

    def test_noop_run_exits_two(self, tmp_path, mock_server):
        cfg = tmp_path / "opencode.jsonc"
        srv = mock_server(["org/model-a"])
        _write_one_provider(cfg, srv.base_url)
        main(["--config", str(cfg)])
        assert main(["--config", str(cfg)]) == 2

    def test_dead_server_exits_one(self, tmp_path):
        cfg = tmp_path / "opencode.jsonc"
        _write_one_provider(cfg, "http://127.0.0.1:1/v1")
        with pytest.raises(SystemExit) as exc_info:
            main(["--config", str(cfg)])
        assert exc_info.value.code == 1


class TestQuietMode:
    def test_quiet_suppresses_progress_but_not_errors(self, tmp_path, mock_server, capsys):
        cfg = tmp_path / "opencode.jsonc"
        srv = mock_server(["org/model-a"])
        _write_one_provider(cfg, srv.base_url)

        rc = main(["--config", str(cfg), "--quiet"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Querying" not in out
        assert "opencode-sync v" not in out  # banner suppressed too

    def test_quict_still_writes_config(self, tmp_path, mock_server):
        cfg = tmp_path / "opencode.jsonc"
        srv = mock_server(["org/model-a"])
        _write_one_provider(cfg, srv.base_url)
        main(["--config", str(cfg), "--quiet"])
        assert "org/model-a" in load_config(cfg)["provider"]["vllm"]["models"]

    def test_error_printed_under_quiet(self, tmp_path, capsys):
        cfg = tmp_path / "opencode.jsonc"
        _write_one_provider(cfg, "http://127.0.0.1:1/v1")
        with pytest.raises(SystemExit):
            main(["--config", str(cfg), "--quiet"])
        assert "ERROR:" in capsys.readouterr().err
