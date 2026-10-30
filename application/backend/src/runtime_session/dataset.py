"""Recording a dataset: what it needs from a session, and the recording itself.

A dataset stores the follower's joints packed into ``observation.state`` and
``action`` with ``<joint>.pos`` names, and each camera as
``observation.images.<key>``. The store uses the same names and keys, so a
dataset's spec compares to a session's directly. A dataset with nothing
recorded yet has no shape of its own and takes the session's.

A :class:`Recording` then records into it, from the session's store, through a
``DatasetWorker``. It has its own connection to the session's bus, where it
reports as ``dataset``; the worker serves the episode commands itself.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import AsyncExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

from runtime_session.bus import Bus, publish_status, until_serving
from runtime_session.store import FeatureSpec, ImageSpec
from schemas.runtime_session import SessionDataset
from workers import dataset_worker as dataset_rpc
from workers.dataset_worker import DatasetWorker, RecordingStatus

if TYPE_CHECKING:
    from runtime_session.store import Store
    from schemas.dataset import Dataset

COMPONENT = "dataset"
"""What a recording reports as."""

_READY_S = 10.0
"""How long a started dataset worker has to answer on the bus."""

DEFAULT_RECORDING_FPS = 30
"""The rate a new dataset is recorded at; an existing one keeps its own."""

_IMAGE_PREFIX = "observation.images."


def _info(dataset: Dataset) -> dict[str, Any] | None:
    info = Path(dataset.path) / "meta" / "info.json"
    return json.loads(info.read_text()) if info.is_file() else None


def dataset_fps(dataset: Dataset, requested: int | None = None) -> int:
    """The rate the dataset was recorded at; a new one is recorded at ``requested``, or the default."""
    info = _info(dataset)
    if info is not None:
        return int(info["fps"])
    return requested or DEFAULT_RECORDING_FPS


def dataset_spec(dataset: Dataset) -> FeatureSpec | None:
    """The joints and cameras the dataset was recorded with, or ``None`` when nothing is recorded yet.

    Raises:
        ValueError: The dataset's state and action name different joints.
    """
    info = _info(dataset)
    if info is None:
        return None
    features: dict[str, Any] = info["features"]
    joints = tuple(features["observation.state"]["names"])
    if tuple(features["action"]["names"]) != joints:
        raise ValueError(f"Dataset {dataset.name!r} records an action for other joints than its state")
    cameras = {
        key.removeprefix(_IMAGE_PREFIX): ImageSpec(*(int(dim) for dim in value["shape"]))
        for key, value in features.items()
        if key.startswith(_IMAGE_PREFIX)
    }
    return FeatureSpec(joints, cameras)


def lerobot_features(spec: FeatureSpec) -> dict[str, Any]:
    """The LeRobot feature metadata a new dataset is created with, from the session's spec."""
    # Imported here: lerobot is heavy, and only the dataset worker's process needs it.
    from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
    from lerobot.processor import make_default_processors
    from lerobot.utils.feature_utils import combine_feature_dicts

    teleop_action_processor, _robot_action_processor, robot_observation_processor = make_default_processors()
    action: dict[str, Any] = dict.fromkeys(spec.joint_names, float)
    observation: dict[str, Any] = dict.fromkeys(spec.joint_names, float)
    observation.update({key: image.shape for key, image in spec.observations.items()})
    return combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=action),
            use_videos=True,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=observation),
            use_videos=True,
        ),
    )


class Recording:
    """A dataset being recorded into from the session's store.

    ``async with`` starts its ``DatasetWorker`` and reports ``running``; leaving
    stops the worker, which copies what was recorded back into the dataset, and
    reports ``stopped``. A worker that dies is reported ``failed``; what it saved
    but did not copy back is lost.
    """

    def __init__(
        self, dataset: Dataset, session_id: str, store: Store, robot_type: str, fps: int | None = None
    ) -> None:
        self.dataset = dataset
        self.fps = dataset_fps(dataset, fps)
        self.fps_fixed = _info(dataset) is not None
        self.status = RecordingStatus(is_recording=False, episodes_recorded=0)
        """What the worker last announced on ``evt/dataset``; the worker owns it, the session keeps this up to date."""
        self._worker = DatasetWorker(
            dataset, session_id=session_id, store_name=store.name, spec=store.spec, fps=self.fps, robot_type=robot_type
        )
        self._session_id = session_id
        self._store = store
        self._stack: AsyncExitStack | None = None
        self._started = False

    def state(self) -> SessionDataset:
        return SessionDataset(
            dataset_id=self.dataset.id,
            name=self.dataset.name,
            fps=self.fps,
            fps_fixed=self.fps_fixed,
            is_recording=self.status.is_recording,
            episodes_recorded=self.status.episodes_recorded,
        )

    async def __aenter__(self) -> Self:
        """Start the worker and wait until it answers.

        Raises:
            FeatureSpecMismatchError: The dataset was recorded with joints or cameras the store lacks.
            WorkerStartupError: The dataset could not be opened for recording.
        """
        async with AsyncExitStack() as stack:
            bus = stack.enter_context(Bus.join(self._session_id))
            publish_status(bus, COMPONENT, "starting")
            # Runs after the worker has stopped, while the bus is still open.
            stack.callback(self._report_stopped, bus)
            try:
                if (recorded := dataset_spec(self.dataset)) is not None:
                    self._store.spec.require_compatible(recorded)
                await stack.enter_async_context(self._worker)
                self.status = await until_serving(
                    bus, dataset_rpc.STATUS, dataset_rpc.Empty(), RecordingStatus, within=_READY_S
                )
            except Exception as error:
                publish_status(bus, COMPONENT, "failed", str(error))
                raise
            watch = asyncio.create_task(self._report_death(bus))
            stack.callback(watch.cancel)
            publish_status(bus, COMPONENT, "running")
            self._started = True
            self._stack = stack.pop_all()
        return self

    async def __aexit__(self, *_exc: object) -> None:
        stack, self._stack = self._stack, None
        if stack is not None:
            await stack.aclose()

    async def _report_death(self, bus: Bus) -> None:
        await self._worker.wait_stopped()
        publish_status(bus, COMPONENT, "failed", self._worker.failure or "stopped unexpectedly")

    def _report_stopped(self, bus: Bus) -> None:
        # A worker that failed stays reported failed.
        if self._started and self._worker.failure is None:
            publish_status(bus, COMPONENT, "stopped")
