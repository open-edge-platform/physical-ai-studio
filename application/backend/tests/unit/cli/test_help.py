"""Customer-facing CLI discovery must be useful without running commands."""

from importlib import import_module

import pytest
from click.testing import CliRunner

from cli import cli
from cli.hf import inspection


@pytest.mark.parametrize("help_flag", ["--help", "-h"])
@pytest.mark.parametrize(
    "command",
    [
        (),
        ("serve",),
        ("gen-api",),
        ("db",),
        ("db", "check"),
        ("db", "init"),
        ("db", "migrate"),
        ("db", "clean"),
        ("models",),
        ("models", "import-dir"),
        ("datasets",),
        ("datasets", "import-dir"),
        ("hf",),
        ("hf", "model"),
        ("hf", "model", "inspect"),
        ("hf", "model", "import"),
        ("hf", "model", "export"),
        ("hf", "dataset"),
        ("hf", "dataset", "inspect"),
        ("hf", "dataset", "import"),
        ("hf", "dataset", "export"),
    ],
)
def test_help_is_available_without_side_effects(command, help_flag, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("STORAGE_DIR", str(tmp_path / "storage"))
    monkeypatch.setenv("SETTINGS_FILE", str(tmp_path / "settings.json"))

    def unexpected_call(*args, **kwargs):
        pytest.fail("Help must not start a server, migrate storage, or contact the Hub")

    monkeypatch.setattr(import_module("cli.serve"), "start_server", unexpected_call)
    monkeypatch.setattr(import_module("cli.database"), "_run_migrations", unexpected_call)
    monkeypatch.setattr(inspection, "hub_api", unexpected_call)
    result = CliRunner().invoke(cli, [*command, help_flag])
    assert result.exit_code == 0, result.output
    assert "Usage:" in result.stdout
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ((), "STORAGE_DIR"),
        ((), "studio_cli.md"),
        (("serve",), "Startup can modify"),
        (("db", "check"), "2 = migrations needed"),
        (("db", "clean"), "DESTRUCTIVE"),
        (("db", "clean"), "without a confirmation"),
        (("models", "import-dir"), "copy keeps it intact"),
        (("datasets", "import-dir"), "existing project and environment"),
        (("hf", "model", "inspect"), "not necessarily importable=true"),
        (("hf", "dataset", "import"), "temporary empty environment"),
        (("hf", "model", "import"), "omit REPO_ID"),
        (("hf", "dataset", "export"), "Stop dataset recording"),
        (("hf", "model", "export"), "existing visibility is unchanged"),
    ],
)
def test_help_explains_decision_points(command, expected):
    result = CliRunner().invoke(cli, [*command, "--help"])
    assert result.exit_code == 0, result.output
    assert expected in " ".join(result.stdout.split())


def test_guide_url_is_not_hard_wrapped():
    result = CliRunner().invoke(cli, ["--help"])
    assert (
        "https://github.com/open-edge-platform/physical-ai-studio/blob/main/application/backend/docs/studio_cli.md"
        in result.stdout
    )
