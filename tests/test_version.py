"""Tests for version reporting: banner, --version flag, PyPI update check."""

from __future__ import annotations

import json
import urllib.error

import pytest

import opencode_sync
from opencode_sync import cli
from opencode_sync.cli import (
    PYPI_JSON_URL,
    _check_pypi_update,
    _get_version,
    _is_newer_release,
    _release_segments,
    main,
)

# ---------------------------------------------------------------------------
# _get_version
# ---------------------------------------------------------------------------


class TestGetVersion:
    def test_real_metadata_matches_package_version(self):
        # On a clean install these agree; if they drift, the banner lies.
        assert _get_version() == opencode_sync.__version__

    def test_metadata_used_when_available(self, monkeypatch):
        monkeypatch.setattr(
            "importlib.metadata.version", lambda name: "9.9.9", raising=True
        )
        assert _get_version() == "9.9.9"

    def test_falls_back_to_dunder_version(self, monkeypatch):
        def raise_not_found(name):
            raise Exception("PackageNotFoundError stand-in")

        monkeypatch.setattr(
            "importlib.metadata.version", raise_not_found, raising=True
        )
        assert _get_version() == opencode_sync.__version__


# ---------------------------------------------------------------------------
# --version flag
# ---------------------------------------------------------------------------


class TestVersionFlag:
    def test_version_flag_prints_and_exits_zero(self, capsys):
        with pytest.raises(SystemExit) as excinfo:
            main(["--version"])
        assert excinfo.value.code == 0
        out = capsys.readouterr().out
        assert _banner() in out

    def test_version_flag_comes_before_any_config_or_server_work(self, capsys, tmp_path):
        # No config exists at this path and no server is up; --version must
        # still exit 0 without touching the filesystem.
        cfg = tmp_path / "opencode.jsonc"
        with pytest.raises(SystemExit) as excinfo:
            main(["--version", "--config", str(cfg)])
        assert excinfo.value.code == 0
        assert not cfg.exists()


# ---------------------------------------------------------------------------
# Version banner
# ---------------------------------------------------------------------------


def _first_line(text: str) -> str:
    return text.splitlines()[0] if text else ""


def _banner() -> str:
    return f"opencode-sync v{_get_version()}"


def _write_single_provider_config(cfg, base_url="http://127.0.0.1:1/v1"):
    """Minimal one-provider config for banner/check tests.

    base_url defaults to a closed port: the sync fails fast and the exit path
    (error exit) is what these tests exercise, never a real server.
    """
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(
        json.dumps(
            {
                "provider": {
                    "vllm": {
                        "npm": "@ai-sdk/openai-compatible",
                        "options": {"baseURL": base_url},
                        "models": {},
                    }
                }
            }
        ),
        encoding="utf-8",
    )


class TestBanner:
    def test_banner_on_sync_run(self, tmp_path, mock_server, monkeypatch, capsys):
        cfg = tmp_path / "opencode.jsonc"
        _write_single_provider_config(cfg, base_url=mock_server(["org/model-a"]).base_url)
        monkeypatch.setattr(cli, "_check_pypi_update", lambda *a, **k: None)
        rc = main(["--config", str(cfg)])
        assert rc == 0
        assert _first_line(capsys.readouterr().out) == _banner()

    def test_banner_on_install_run(self, tmp_path, monkeypatch, capsys):
        real_bin = tmp_path / "real" / "opencode"
        real_bin.parent.mkdir(parents=True)
        real_bin.write_text("#!/bin/sh\n")
        wrapper = tmp_path / "wrap" / "opencode"
        monkeypatch.setattr(cli, "_check_pypi_update", lambda *a, **k: None)
        rc = main(["install", "--wrapper", str(wrapper), "--opencode-bin", str(real_bin)])
        assert rc == 0
        assert _first_line(capsys.readouterr().out) == _banner()

    def test_banner_precedes_help_output(self, capsys):
        # Banner prints before parse_args, so -h output is preceded by it.
        with pytest.raises(SystemExit) as excinfo:
            main(["--help"])
        assert excinfo.value.code == 0
        out = capsys.readouterr().out
        assert _first_line(out) == _banner()
        assert "usage:" in out


# ---------------------------------------------------------------------------
# Release-segment parsing and comparison
# ---------------------------------------------------------------------------


class TestReleaseSegments:
    @pytest.mark.parametrize(
        "version,expected",
        [
            ("0.5.0", (0, 5, 0)),
            ("1.2", (1, 2)),
            ("10.0.3", (10, 0, 3)),
            ("0.5.0.dev1+g4b58801", None),  # scm dev build
            ("0.5.0.dev1", None),  # dev segment, no local
            ("0.6.0rc1", None),  # pre-release
            ("0.6.0a1", None),
            ("0.6.0b2", None),
            ("1.0.0+local", (1, 0, 0)),  # local segment stripped, still a release
            ("", None),
            ("garbage", None),
            ("0.5.x", None),
        ],
    )
    def test_parsing(self, version, expected):
        assert _release_segments(version) == expected


class TestIsNewerRelease:
    @pytest.mark.parametrize(
        "pypi,local,expected",
        [
            ("0.6.0", "0.5.0", True),
            ("0.5.1", "0.5.0", True),
            ("0.5.0", "0.5.0", False),
            ("0.5.0", "0.6.0", False),
            ("0.5.0", "1.0.0", False),
            # numeric (not lexicographic) segment comparison
            ("0.10.0", "0.9.0", True),
            ("0.9.0", "0.10.0", False),
            # dev/pre-release locals never nag
            ("0.6.0", "0.5.0.dev1+g4b58801", False),
            ("0.6.0", "0.5.0.dev1", False),
            # local segment on its own is still a release
            ("0.6.0", "0.5.0+local", True),
            # malformed PyPI payloads fail safe
            ("garbage", "0.5.0", False),
            ("", "0.5.0", False),
        ],
    )
    def test_comparison(self, pypi, local, expected):
        assert _is_newer_release(pypi, local) is expected


# ---------------------------------------------------------------------------
# PyPI update check
# ---------------------------------------------------------------------------


def _pypi_payload(version):
    return {"info": {"version": version}}


def _unwrap_check(module):
    """Return the real _check_pypi_update before any monkeypatching.

    Used by wiring tests that must call the genuine implementation with a
    stubbed fetch layer (its signature takes the fetch as an argument).
    Call it before monkeypatch.setattr replaces the module attribute.
    """
    return module._check_pypi_update


class TestCheckPypiUpdate:
    def test_returns_notice_when_newer_release_on_pypi(self):
        notice = _check_pypi_update(
            _http_get_json=lambda url, timeout: _pypi_payload("9.9.9")
        )
        assert notice is not None
        assert "9.9.9" in notice
        assert _get_version() in notice

    def test_queries_pypi_json_url(self):
        seen = {}

        def fake_get(url, timeout):
            seen["url"] = url
            seen["timeout"] = timeout
            return _pypi_payload("0.5.0")

        _check_pypi_update(_http_get_json=fake_get)
        assert seen["url"] == PYPI_JSON_URL
        assert seen["timeout"] == cli.UPDATE_CHECK_TIMEOUT

    def test_no_notice_when_pypi_equal(self):
        assert _check_pypi_update(
            _http_get_json=lambda url, timeout: _pypi_payload(_get_version())
        ) is None

    def test_no_notice_when_pypi_older(self):
        assert _check_pypi_update(
            _http_get_json=lambda url, timeout: _pypi_payload("0.0.1")
        ) is None

    @pytest.mark.parametrize(
        "fetch",
        [
            lambda url, timeout: (_ for _ in ()).throw(urllib.error.URLError("no net")),
            lambda url, timeout: (_ for _ in ()).throw(
                urllib.error.HTTPError(url, 500, "boom", None, None)  # noqa: F821
            ),
            lambda url, timeout: (_ for _ in ()).throw(TimeoutError("slow")),
            lambda url, timeout: (_ for _ in ()).throw(Exception("kaboom")),
            lambda url, timeout: "not json at all",  # bad shape, not a dict
            lambda url, timeout: {},  # missing info key
            lambda url, timeout: {"info": {}},  # missing version key
            lambda url, timeout: {"info": {"version": None}},  # null version
            lambda url, timeout: {"info": {"version": "garbage"}},  # unparseable
        ],
    )
    def test_silent_on_any_failure(self, fetch, capsys):
        assert _check_pypi_update(_http_get_json=fetch) is None
        assert capsys.readouterr().out == ""

    def test_skipped_entirely_for_dev_version(self, monkeypatch, capsys):
        monkeypatch.setattr(cli, "_get_version", lambda: "0.5.0.dev1+g4b58801")

        def fail(url, timeout):
            raise AssertionError("fetch must not be called for dev builds")

        assert _check_pypi_update(_http_get_json=fail) is None

    def test_default_fetch_is_used_when_not_injected(self):
        # Sanity: the real default fetcher exists and is wired in.
        assert cli._check_pypi_update.__defaults__[0] is cli._pypi_http_get_json


# ---------------------------------------------------------------------------
# --no-update-check wiring
# ---------------------------------------------------------------------------


class TestNoUpdateCheckFlag:
    def _sync_argv(self, cfg):
        # The provider's baseURL points at a closed port, so the sync itself
        # fails fast; these tests assert only what the PyPI check does.
        return ["--config", str(cfg)]

    def _write_config(self, tmp_path):
        cfg = tmp_path / "dead-url.jsonc"
        _write_single_provider_config(cfg)
        return cfg

    def test_flag_present_means_no_pypi_call(self, tmp_path, monkeypatch, capsys):
        cfg = self._write_config(tmp_path)
        calls = []

        def spy(*a, **k):
            calls.append(1)
            return None

        monkeypatch.setattr(cli, "_check_pypi_update", spy)
        with pytest.raises(SystemExit) as excinfo:
            main(self._sync_argv(cfg) + ["--no-update-check"])
        assert excinfo.value.code == 1  # unreachable server, as designed
        assert not calls  # the check itself was never invoked
        assert "Update available:" not in capsys.readouterr().out

    def test_flag_absent_attempts_pypi_call(self, tmp_path, monkeypatch, capsys):
        cfg = self._write_config(tmp_path)
        calls = []

        def fake_check(_http_get_json=None):
            calls.append(1)
            return None

        monkeypatch.setattr(cli, "_check_pypi_update", fake_check)
        with pytest.raises(SystemExit) as excinfo:
            main(self._sync_argv(cfg))
        assert excinfo.value.code == 1
        assert calls  # the check ran on the error exit path

    def test_pypi_outage_does_not_change_exit_code(self, tmp_path, monkeypatch, capsys):
        cfg = self._write_config(tmp_path)

        # Simulate the outage at the fetch layer, but keep the real check
        # function so the URLError is exercised through its swallow guard.
        def failing_fetch(url, timeout):
            raise urllib.error.URLError("pypi down")

        real_check = _unwrap_check(cli)
        monkeypatch.setattr(
            cli,
            "_check_pypi_update",
            lambda: real_check(failing_fetch),
        )
        with pytest.raises(SystemExit) as excinfo:
            main(self._sync_argv(cfg))
        assert excinfo.value.code == 1  # unchanged by the PyPI failure
        err = capsys.readouterr().err
        assert "ERROR:" in err  # the real error is still reported

    def test_update_notice_appears_after_sync_output(self, tmp_path, monkeypatch, capsys):
        cfg = self._write_config(tmp_path)
        monkeypatch.setattr(
            cli, "_check_pypi_update", lambda *a, **k: "Update available: x 9.9.9"
        )
        with pytest.raises(SystemExit) as excinfo:
            main(self._sync_argv(cfg))
        assert excinfo.value.code == 1
        out = capsys.readouterr().out
        assert _first_line(out) == _banner()
        lines = out.rstrip().splitlines()
        # Notice must be the LAST stdout line, after the sync's own output,
        # set off by a dash separator.
        assert lines[-1].startswith("Update available:")
        assert lines[-2] == "---"
        assert "9.9.9" in out

    def test_nothing_synced_path_also_reports_check(self, tmp_path, monkeypatch, capsys):
        # Two providers, both unreachable: "Nothing synced." rc-1 exit also
        # runs the update check (coverage for that return site).
        cfg = tmp_path / "two-dead.jsonc"
        _write_single_provider_config(cfg)
        cfg.write_text(
            json.dumps(
                {
                    "provider": {
                        "a": {
                            "npm": "@ai-sdk/openai-compatible",
                            "options": {"baseURL": "http://127.0.0.1:1/v1"},
                            "models": {},
                        },
                        "b": {
                            "npm": "@ai-sdk/openai-compatible",
                            "options": {"baseURL": "http://127.0.0.1:2/v1"},
                            "models": {},
                        },
                    }
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(
            cli, "_check_pypi_update", lambda *a, **k: "Update available: x"
        )
        rc = main(["--config", str(cfg)])
        assert rc == 1  # every provider failed
        out = capsys.readouterr().out
        assert out.rstrip().splitlines()[-1].startswith("Update available:")


# ---------------------------------------------------------------------------
# Integration: banner + check coexistence on a real sync
# ---------------------------------------------------------------------------


class TestSyncWithUpdateCheck:
    def test_full_sync_banner_then_notice(self, tmp_path, mock_server, monkeypatch, capsys):
        cfg = tmp_path / "opencode.jsonc"
        _write_single_provider_config(cfg, base_url=mock_server(["org/model-a"]).base_url)
        monkeypatch.setattr(
            cli,
            "_check_pypi_update",
            lambda *a, **k: f"Update available: opencode-sync 9.9.9 (you have {_get_version()})",
        )
        rc = main(["--config", str(cfg)])
        assert rc == 0
        out = capsys.readouterr().out
        lines = out.splitlines()
        assert lines[0] == _banner()
        assert "Update available:" in lines[-1]

    def test_pypi_failure_leaves_sync_output_intact(
        self, tmp_path, mock_server, monkeypatch, capsys
    ):
        cfg = tmp_path / "opencode.jsonc"
        _write_single_provider_config(cfg, base_url=mock_server(["org/model-a"]).base_url)

        # Simulate the outage through the real check function's swallow guard.
        def failing_fetch(url, timeout):
            raise urllib.error.URLError("offline")

        real_check = _unwrap_check(cli)
        monkeypatch.setattr(
            cli,
            "_check_pypi_update",
            lambda: real_check(failing_fetch),
        )
        rc = main(["--config", str(cfg)])
        assert rc == 0
        out = capsys.readouterr().out
        assert _first_line(out) == _banner()
        assert "Update available:" not in out


class TestByteIdenticalPath:
    def test_byte_identical_exit_still_runs_check(self, tmp_path, mock_server, monkeypatch, capsys):
        # The byte-identical early return fires when a NON-noop plan renders
        # to exactly the input text. Reach it deterministically: a real
        # change (model added) whose rendered text is forced back to the
        # original via a stubbed apply_plans_to_text.
        cfg = tmp_path / "opencode.jsonc"
        base_url = mock_server(["org/model-a", "org/model-b"]).base_url
        _write_single_provider_config(cfg, base_url=base_url)
        original = cfg.read_text(encoding="utf-8")
        monkeypatch.setattr(
            cli, "_check_pypi_update", lambda *a, **k: "Update available: x"
        )
        monkeypatch.setattr(cli, "apply_plans_to_text", lambda text, config, plans: text)
        rc = main(["--config", str(cfg)])
        assert rc == 0
        assert cfg.read_text(encoding="utf-8") == original  # not rewritten
        out = capsys.readouterr().out
        assert "byte-identical" in out
        assert out.rstrip().splitlines()[-1].startswith("Update available:")
