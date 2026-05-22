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
