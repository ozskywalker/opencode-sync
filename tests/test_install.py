"""Tests for opencode-sync install subcommand."""

from __future__ import annotations

import os
import stat
from pathlib import Path
from unittest.mock import patch

import pytest

from opencode_sync.cli import _find_opencode_bin, main


# ---------------------------------------------------------------------------
# _find_opencode_bin
# ---------------------------------------------------------------------------

class TestFindOpencodeBin:
    def test_returns_first_hit_on_path(self, tmp_path):
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        real = bin_dir / "opencode"
        real.write_text("#!/bin/sh\nexec real\n")
        real.chmod(0o755)

        with patch.dict(os.environ, {"PATH": str(bin_dir)}):
            result = _find_opencode_bin(tmp_path / "local" / "bin" / "opencode")
        assert result == real

    def test_skips_wrapper_path(self, tmp_path):
        wrapper_dir = tmp_path / "local" / "bin"
        wrapper_dir.mkdir(parents=True)
        wrapper = wrapper_dir / "opencode"
        wrapper.write_text("#!/bin/sh\n# wrapper\n")

        real_dir = tmp_path / "real" / "bin"
        real_dir.mkdir(parents=True)
        real = real_dir / "opencode"
        real.write_text("#!/bin/sh\nexec real\n")

        path = f"{wrapper_dir}:{real_dir}"
        with patch.dict(os.environ, {"PATH": path}):
            result = _find_opencode_bin(wrapper)
        assert result == real

    def test_returns_none_when_not_found(self, tmp_path):
        with patch.dict(os.environ, {"PATH": str(tmp_path)}):
            result = _find_opencode_bin(tmp_path / "local" / "bin" / "opencode")
        assert result is None


# ---------------------------------------------------------------------------
# install subcommand
# ---------------------------------------------------------------------------

class TestInstall:
    def _run(self, tmp_path, extra_args=None, real_bin=None, env_path=None):
        wrapper = tmp_path / "local" / "bin" / "opencode"
        if real_bin is None:
            real_dir = tmp_path / "real" / "bin"
            real_dir.mkdir(parents=True)
            real_bin = real_dir / "opencode"
            real_bin.write_text("#!/bin/sh\nexec real\n")

        argv = ["install", "--wrapper", str(wrapper), "--opencode-bin", str(real_bin)]
        if extra_args:
            argv.extend(extra_args)

        path = env_path or os.environ.get("PATH", "")
        with patch.dict(os.environ, {"PATH": path}):
            rc = main(argv)

        return rc, wrapper, real_bin

    def test_writes_wrapper_file(self, tmp_path):
        rc, wrapper, _ = self._run(tmp_path)
        assert rc == 0
        assert wrapper.exists()

    def test_wrapper_is_executable(self, tmp_path):
        _, wrapper, _ = self._run(tmp_path)
        mode = wrapper.stat().st_mode
        assert mode & stat.S_IXUSR

    def test_wrapper_has_shebang(self, tmp_path):
        _, wrapper, _ = self._run(tmp_path)
        assert wrapper.read_text().startswith("#!/bin/sh")

    def test_wrapper_contains_opencode_sync(self, tmp_path):
        _, wrapper, _ = self._run(tmp_path)
        assert "opencode-sync" in wrapper.read_text()

    def test_wrapper_exec_line_contains_real_bin(self, tmp_path):
        _, wrapper, real_bin = self._run(tmp_path)
        assert f"exec {real_bin}" in wrapper.read_text()

    def test_wrapper_creates_parent_dirs(self, tmp_path):
        rc, wrapper, _ = self._run(tmp_path)
        assert wrapper.parent.is_dir()

    def test_existing_wrapper_refused_without_force(self, tmp_path):
        _, wrapper, real_bin = self._run(tmp_path)
        rc = main(["install", "--wrapper", str(wrapper), "--opencode-bin", str(real_bin)])
        assert rc != 0

    def test_force_overwrites_existing_wrapper(self, tmp_path):
        _, wrapper, real_bin = self._run(tmp_path)
        wrapper.write_text("old content")
        rc = main(["install", "--wrapper", str(wrapper), "--opencode-bin", str(real_bin), "--force"])
        assert rc == 0
        assert "opencode-sync" in wrapper.read_text()

    def test_dry_run_does_not_write(self, tmp_path):
        rc, wrapper, _ = self._run(tmp_path, extra_args=["--dry-run"])
        assert rc == 0
        assert not wrapper.exists()

    def test_dry_run_prints_wrapper_path(self, tmp_path, capsys):
        self._run(tmp_path, extra_args=["--dry-run"])
        out = capsys.readouterr().out
        assert "dry-run" in out
        assert "opencode" in out

    def test_path_warning_when_wrapper_dir_not_in_path(self, tmp_path, capsys):
        self._run(tmp_path, env_path="/usr/bin:/usr/local/bin")
        err = capsys.readouterr().err
        assert "not in your PATH" in err or "WARNING" in err

    def test_no_warning_when_wrapper_dir_first_in_path(self, tmp_path, capsys):
        wrapper = tmp_path / "local" / "bin" / "opencode"
        real_dir = tmp_path / "real" / "bin"
        real_dir.mkdir(parents=True)
        real_bin = real_dir / "opencode"
        real_bin.write_text("#!/bin/sh\n")

        path = f"{wrapper.parent}:{real_dir}"
        with patch.dict(os.environ, {"PATH": path}):
            main(["install", "--wrapper", str(wrapper), "--opencode-bin", str(real_bin)])

        err = capsys.readouterr().err
        assert "WARNING" not in err

    def test_wrapper_dir_after_real_bin_warns(self, tmp_path, capsys):
        wrapper = tmp_path / "local" / "bin" / "opencode"
        real_dir = tmp_path / "real" / "bin"
        real_dir.mkdir(parents=True)
        real_bin = real_dir / "opencode"
        real_bin.write_text("#!/bin/sh\n")

        # real bin dir comes BEFORE wrapper dir in PATH
        path = f"{real_dir}:{wrapper.parent}"
        with patch.dict(os.environ, {"PATH": path}):
            main(["install", "--wrapper", str(wrapper), "--opencode-bin", str(real_bin)])

        err = capsys.readouterr().err
        assert "WARNING" in err

    def test_auto_detect_bin_from_path(self, tmp_path):
        real_dir = tmp_path / "real" / "bin"
        real_dir.mkdir(parents=True)
        real_bin = real_dir / "opencode"
        real_bin.write_text("#!/bin/sh\n")

        wrapper = tmp_path / "local" / "bin" / "opencode"
        path = str(real_dir)
        with patch.dict(os.environ, {"PATH": path}):
            rc = main(["install", "--wrapper", str(wrapper)])
        assert rc == 0
        assert str(real_bin) in wrapper.read_text()
