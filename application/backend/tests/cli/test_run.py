import importlib
import os
import subprocess
from pathlib import Path

import click
import pytest

run_module = importlib.import_module("cli.run")


class TestLoadEnvFile:
    def test_missing_file_is_a_noop(self, tmp_path: Path) -> None:
        run_module.load_env_file(tmp_path / "missing.env")

    def test_sets_unset_variables(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("FOO", raising=False)
        env_file = tmp_path / ".env"
        env_file.write_text("FOO=bar\n")

        run_module.load_env_file(env_file)

        assert os.environ["FOO"] == "bar"

    def test_does_not_override_existing_non_empty_value(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("FOO", "existing")
        env_file = tmp_path / ".env"
        env_file.write_text("FOO=from_file\n")

        run_module.load_env_file(env_file)

        assert os.environ["FOO"] == "existing"

    def test_skips_blank_lines_and_comments(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("FOO", raising=False)
        env_file = tmp_path / ".env"
        env_file.write_text("\n# a comment\nFOO=bar\n")

        run_module.load_env_file(env_file)

        assert os.environ["FOO"] == "bar"

    def test_strips_export_prefix(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("FOO", raising=False)
        env_file = tmp_path / ".env"
        env_file.write_text("export FOO=bar\n")

        run_module.load_env_file(env_file)

        assert os.environ["FOO"] == "bar"

    def test_strips_export_prefix_with_multiple_spaces(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("FOO", raising=False)
        env_file = tmp_path / ".env"
        env_file.write_text("export   FOO=bar\n")

        run_module.load_env_file(env_file)

        assert os.environ["FOO"] == "bar"

    def test_strips_surrounding_quotes(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("FOO", raising=False)
        env_file = tmp_path / ".env"
        env_file.write_text('FOO="bar baz"\n')

        run_module.load_env_file(env_file)

        assert os.environ["FOO"] == "bar baz"

    def test_skips_lines_with_invalid_keys(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("1FOO", raising=False)
        env_file = tmp_path / ".env"
        env_file.write_text("1FOO=bar\n")

        run_module.load_env_file(env_file)

        assert "1FOO" not in os.environ

    def test_skips_lines_without_equals(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text("not_a_valid_line\n")

        # Should not raise.
        run_module.load_env_file(env_file)

    def test_warns_when_file_is_group_or_other_readable(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text("FOO=bar\n")
        env_file.chmod(0o644)

        warnings: list[str] = []
        monkeypatch.setattr(
            run_module.click,
            "echo",
            lambda msg, err=False: warnings.append(msg) if err else None,
        )

        run_module.load_env_file(env_file)

        assert any("readable by group/other" in msg for msg in warnings)


class TestResolveDevice:
    def test_defaults_to_cpu_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DEVICE", raising=False)

        resolved, explicit = run_module._resolve_device(None)

        assert resolved == "cpu"
        assert explicit is False

    def test_uses_explicit_device_argument(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DEVICE", raising=False)

        resolved, explicit = run_module._resolve_device("cuda")

        assert resolved == "cuda"
        assert explicit is True

    def test_falls_back_to_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DEVICE", "xpu")

        resolved, explicit = run_module._resolve_device(None)

        assert resolved == "xpu"
        assert explicit is True

    def test_argument_takes_precedence_over_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DEVICE", "xpu")

        resolved, explicit = run_module._resolve_device("cuda")

        assert resolved == "cuda"
        assert explicit is True

    def test_is_case_insensitive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DEVICE", raising=False)

        resolved, _ = run_module._resolve_device("CUDA")

        assert resolved == "cuda"

    def test_raises_on_invalid_device(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DEVICE", "tpu")

        with pytest.raises(click.ClickException):
            run_module._resolve_device(None)


class TestShouldSync:
    def test_flag_true_overrides_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SYNC", "false")

        assert run_module._should_sync(True) is True

    def test_flag_false_overrides_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SYNC", "true")

        assert run_module._should_sync(False) is False

    def test_defaults_to_true_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SYNC", raising=False)

        assert run_module._should_sync(None) is True

    def test_env_false_disables_sync(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SYNC", "false")

        assert run_module._should_sync(None) is False

    def test_env_is_case_insensitive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SYNC", "FALSE")

        assert run_module._should_sync(None) is False


class TestMaybeSync:
    def test_skips_when_sync_disabled(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        called = False

        def fake_run(*args, **kwargs):
            nonlocal called
            called = True

        monkeypatch.setattr(run_module.subprocess, "run", fake_run)

        run_module.maybe_sync(tmp_path, "cpu", sync=False)

        assert not called

    def test_invokes_uv_sync_with_device_extra(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        captured_args = None
        captured_cwd = None

        def fake_run(args, cwd, check):
            nonlocal captured_args, captured_cwd
            captured_args = args
            captured_cwd = cwd
            assert check is True

        monkeypatch.setattr(run_module.subprocess, "run", fake_run)

        run_module.maybe_sync(tmp_path, "cuda", sync=True)

        assert captured_args == ["uv", "sync", "--extra", "cuda"]
        assert captured_cwd == tmp_path

    def test_passes_through_extra_extras(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        captured_args = None

        def fake_run(args, cwd, check):
            nonlocal captured_args
            captured_args = args

        monkeypatch.setattr(run_module.subprocess, "run", fake_run)

        run_module.maybe_sync(tmp_path, "cpu", "extra1", "extra2", sync=True)

        assert captured_args == ["uv", "sync", "--extra", "cpu", "--extra", "extra1", "--extra", "extra2"]

    def test_wraps_subprocess_failure_in_click_exception(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def fake_run(args, cwd, check):
            raise subprocess.CalledProcessError(returncode=1, cmd=args)

        monkeypatch.setattr(run_module.subprocess, "run", fake_run)

        with pytest.raises(click.ClickException):
            run_module.maybe_sync(tmp_path, "cpu", sync=True)

    def test_wraps_os_error_in_click_exception(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_run(args, cwd, check):
            raise OSError("uv not found")

        monkeypatch.setattr(run_module.subprocess, "run", fake_run)

        with pytest.raises(click.ClickException):
            run_module.maybe_sync(tmp_path, "cpu", sync=True)
