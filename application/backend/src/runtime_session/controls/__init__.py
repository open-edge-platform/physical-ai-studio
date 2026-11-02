"""The controls a session can run, and building one from its config. See ``base.py``."""

from __future__ import annotations

from typing import TYPE_CHECKING

from runtime_session.controls.base import Control
from runtime_session.controls.model import ModelControl
from runtime_session.controls.teleop import TeleopControl
from schemas.runtime_session import ModelControlConfig

if TYPE_CHECKING:
    from physicalai.robot import Robot

    from runtime_session.bus import Bus
    from runtime_session.model import ProcessModel
    from runtime_session.store import Store
    from schemas.runtime_session import ControlConfig

__all__ = ["Control", "ModelControl", "TeleopControl", "build_control"]


def build_control(
    config: ControlConfig,
    store: Store,
    leader: Robot | None,
    model: ProcessModel | None = None,
    bus: Bus | None = None,
) -> Control:
    """Raises ``ValueError`` when the session cannot run ``config``.

    A model control needs the loaded ``model``. A control serves its own commands on ``bus``.
    """
    if isinstance(config, ModelControlConfig):
        if model is None:
            raise ValueError("Load a model first")
        return ModelControl(store, model, config)
    if leader is None:
        raise ValueError("Teleoperation needs a connected leader arm; connect one and reload the environment")
    return TeleopControl(store, leader, config, bus)
