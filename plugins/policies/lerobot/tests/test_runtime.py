# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Runtime adapter trust-boundary tests."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from physicalai_lerobot_policy_plugin.runtime import LeRobotRuntimeAdapter


def test_groot_package_is_rejected_before_loading_remote_policy_code(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Reject unreviewed Groot before the policy factory can load its model class."""
    from lerobot.configs.policies import PreTrainedConfig  # ruff: ignore[import-outside-top-level]
    from lerobot.policies import factory  # ruff: ignore[import-outside-top-level]
    from lerobot.utils import import_utils  # ruff: ignore[import-outside-top-level]

    artifact = tmp_path / "groot-model"
    artifact.mkdir()
    (artifact / "config.json").write_text('{"type":"groot"}', encoding="utf-8")

    def fake_config_load(_cls: type, *_args: object, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(type="groot")

    monkeypatch.setattr(PreTrainedConfig, "from_pretrained", classmethod(fake_config_load))
    monkeypatch.setattr(import_utils, "register_third_party_plugins", lambda: None)
    get_policy_class = Mock(side_effect=AssertionError("Groot model factory must not run"))
    monkeypatch.setattr(factory, "get_policy_class", get_policy_class)

    with pytest.raises(ValueError, match=r"Groot.*remote code.*reviewed and pinned"):
        LeRobotRuntimeAdapter().load(artifact)

    get_policy_class.assert_not_called()
