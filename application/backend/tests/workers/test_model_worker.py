from __future__ import annotations

import asyncio
import os
from multiprocessing import shared_memory
from types import SimpleNamespace
from unittest.mock import Mock, patch
from uuid import uuid4

import numpy as np
import pytest
from physicalai.inference.manifest import ObservationHistorySpec
from physicalai.inference.preprocessors.observation_history import ObservationWindow, StackObservationHistory

from runtime_session.bus import Bus
from runtime_session.model import ModelInputs, ProcessModel
from runtime_session.store import FeatureSpec, ImageSpec, Store
from schemas.model import Model
from workers.model_worker import ActionChunk, Inference, InferRequest, history_image_layout


@pytest.fixture
def store():
    value = Store.create(FeatureSpec(("joint",), {"cam": ImageSpec(2, 3), "other": ImageSpec(1, 2)}), uuid4().hex)
    try:
        yield value
    finally:
        value.unlink()


@pytest.mark.parametrize(
    "images", [{}, {"images.cam": "cam"}, {"images": "cam"}, {"images.cam": "cam", "images.other": "other"}]
)
@pytest.mark.parametrize("steps", [1, 3])
def test_proxy_transports_fixed_windows_and_worker_owns_copies(store, images, steps):
    history = ObservationHistorySpec(n_obs_steps=steps)
    proxy = ProcessModel.__new__(ProcessModel)
    proxy.inputs = ModelInputs(1, images, observation_history=history)
    proxy._spec = store.spec
    proxy._history_prefix = f"pai-{os.getpid()}-{uuid4().hex[:8]}-"
    inference = Inference(Mock(), store, images, history, proxy._history_prefix)
    observation = {"state": ObservationWindow(np.array([[tick]], np.float32) for tick in range(steps)), "task": ["now"]}
    for camera in images.values():
        key = "images" if len(images) == 1 else f"images.{camera}"
        observation[key] = ObservationWindow(
            np.full((1, *store.spec.observations[camera].shape), tick + 10, np.uint8) for tick in range(steps)
        )
    descriptors = []
    owned = []

    def request(_topic, request, _response, **_kwargs):
        descriptors.append(request.history_images)
        assert "base64" not in request.model_dump_json()
        with patch.object(store, "read_observation", side_effect=AssertionError("Temporal request reread Store")):
            owned.append(inference._history_input(request))
        if request.history_images:
            block = shared_memory.SharedMemory(request.history_images, track=False)
            try:
                block.buf[:] = bytes(block.size)
            finally:
                block.close()
        return ActionChunk(actions=[[7]])

    proxy._bus = SimpleNamespace(request_sync=request)
    np.testing.assert_array_equal(proxy.predict_action_chunk(observation), [[7]])
    result = StackObservationHistory()(owned[0])
    assert result["state"].shape == (1, steps, 1)
    assert result["task"] == ["now"]
    for name, camera in images.items():
        assert result[name].shape == (1, steps, *store.spec.observations[camera].shape)
        np.testing.assert_array_equal(result[name][0, :, 0, 0, 0], np.arange(steps) + 10)
        assert not owned[0][name][0].flags.writeable
    if images:
        with pytest.raises(FileNotFoundError):
            shared_memory.SharedMemory(descriptors[0], track=False)
    else:
        assert descriptors == [None]


@pytest.mark.parametrize("failure", [RuntimeError("worker error"), TimeoutError("timeout")])
def test_creator_unlinks_on_request_failure(store, failure):
    proxy = ProcessModel.__new__(ProcessModel)
    proxy.inputs = ModelInputs(1, {"images.cam": "cam"}, observation_history=ObservationHistorySpec(n_obs_steps=1))
    proxy._spec = store.spec
    proxy._history_prefix = f"pai-{os.getpid()}-{uuid4().hex[:8]}-"
    names = []

    def request(_topic, request, _response, **_kwargs):
        names.append(request.history_images)
        raise failure

    proxy._bus = SimpleNamespace(request_sync=request)
    observation = {"state": (np.zeros((1, 1)),), "images": (np.zeros((1, 2, 3, 3), np.uint8),)}
    with pytest.raises(type(failure)):
        proxy.predict_action_chunk(observation)
    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(names[0], track=False)


def test_creator_unlinks_on_malformed_image_selection(store):
    proxy = ProcessModel.__new__(ProcessModel)
    proxy.inputs = ModelInputs(1, {"images.cam": "cam"}, observation_history=ObservationHistorySpec(n_obs_steps=1))
    proxy._spec = store.spec
    proxy._history_prefix = f"pai-{os.getpid()}-{uuid4().hex[:8]}-"
    proxy._bus = Mock()
    names = []
    create = shared_memory.SharedMemory

    def record(**kwargs):
        block = create(**kwargs)
        names.append(block.name)
        return block

    with (
        patch("runtime_session.model.shared_memory.SharedMemory", side_effect=record),
        pytest.raises(ValueError, match="image selection"),
    ):
        proxy.predict_action_chunk({"state": (np.zeros((1, 1)),)})
    assert len(names) == 1
    with pytest.raises(FileNotFoundError):
        create(names[0], track=False)
    proxy._bus.request_sync.assert_not_called()


def test_worker_rejects_malformed_descriptors_before_inference(store):
    prefix = f"pai-{os.getpid()}-{uuid4().hex[:8]}-"
    images = {"images.cam": "cam"}
    inference = Inference(Mock(), store, images, ObservationHistorySpec(n_obs_steps=1), prefix)
    for name in [None, "another-session", prefix + "../bad!!", prefix + "f" * 9]:
        with pytest.raises(ValueError, match="namespace"):
            inference.infer(InferRequest(state=[[0]], history_images=name))
    _, size = history_image_layout(store.spec, images, 1)
    block = shared_memory.SharedMemory(prefix + "12345678", create=True, size=size + 1)
    try:
        with pytest.raises(ValueError, match="size"):
            inference.infer(InferRequest(state=[[0]], history_images=block.name))
        with pytest.raises(ValueError, match="state shape"):
            inference.infer(InferRequest(state=[0], history_images=block.name))
    finally:
        block.close()
        block.unlink()
    inference.model.predict_action_chunk.assert_not_called()


def test_no_history_reads_latest_store_frame_without_temporal_axes(store):
    store.write_observation("cam", np.full((2, 3, 3), 42, np.uint8), timestamp=1)
    model = Mock()
    model.predict_action_chunk.return_value = np.zeros((1, 1))
    inference = Inference(model, store, {"images.cam": "cam"})
    inference.infer(InferRequest(state=[5], task="pick"))
    observation = model.predict_action_chunk.call_args.args[0]
    assert observation["state"].shape == (1, 1)
    assert observation["images.cam"].shape == (1, 2, 3, 3)
    assert observation["task"] == ["pick"]
    np.testing.assert_array_equal(observation["images.cam"], 42)


@pytest.mark.integration
async def test_real_temporal_export_worker_and_runtime_parity(store, tmp_path):
    import torch
    from physicalai.export.backends import ExportBackend, OpenVINOExportParameters
    from physicalai.export.mixin_policy import ExportablePolicyMixin
    from physicalai.inference import InferenceModel
    from physicalai.inference.data import InferenceFeature, InferenceFeatureDtype, InferenceFeatureType
    from physicalai.inference.manifest import ComponentSpec, Manifest
    from physicalai.runtime import ChunkedActionQueue, PolicySource, SyncExecution
    from physicalai.runtime._callback_bus import _CallbackBus

    history = ObservationHistorySpec(n_obs_steps=3, stride=2)

    class TemporalModel(torch.nn.Module):
        def forward(self, batch):
            state = batch["state"]
            return (state[:, 0] + 2 * state[:, 1] + 3 * state[:, 2]).unsqueeze(1)

    class TemporalPolicy(ExportablePolicyMixin):
        def __init__(self):
            self.model = TemporalModel().eval()
            self._preprocessor = torch.nn.Identity()
            self.device = torch.device("cpu")

        @property
        def inputs_schema(self):
            return [InferenceFeature(InferenceFeatureType.STATE, (1,), "state", InferenceFeatureDtype.FLOAT32)]

        @property
        def extra_export_args(self):
            return {
                "openvino": OpenVINOExportParameters(
                    inputs=["state"],
                    observation_history=history,
                    preprocessors_specs=[ComponentSpec(type="stack_observation_history")],
                )
            }

        @staticmethod
        def get_supported_export_backends():
            return [ExportBackend.OPENVINO]

    policy = TemporalPolicy()
    export_dir = tmp_path / "exports" / "openvino"
    policy.export(export_dir, backend="openvino", input_sample={"state": torch.zeros(1, 3, 1)})
    manifest = Manifest.load(export_dir)
    assert manifest.observation_history == history
    assert [spec.type for spec in manifest.model.preprocessors] == ["stack_observation_history"]
    assert (export_dir / "temporalpolicy.xml").is_file()
    assert (export_dir / "temporalpolicy.bin").is_file()
    model = Model(id=uuid4(), name="Temporal", path=str(tmp_path), policy="temporal", properties={}, project_id=uuid4())
    session_id = uuid4().hex

    with Bus.host(session_id), InferenceModel(export_dir, backend="openvino", device="CPU") as runtime_model:
        async with ProcessModel(model, session_id, store, backend="openvino", device="CPU") as proxy:
            assert proxy.inputs.observation_history == history
            assert proxy._worker.pid != os.getpid()
            assert proxy._worker.is_alive()
            source = PolicySource(
                proxy,
                execution=SyncExecution(request_threshold=0.0),
                action_queue=ChunkedActionQueue(),
                observation_history=proxy.inputs.observation_history,
            )
            source.connect(bus=_CallbackBus([]), session_id=session_id)
            try:
                for values in ([1, 2, 3, 4, 5], [9, 10]):
                    for tick, value in enumerate(values):
                        observation = SimpleNamespace(state=np.array([value], np.float32), images=None)
                        selected = source.to_model_input(observation, {})
                        # An independent oracle catches lost history, wrong order/stride, and stale reset padding.
                        states = np.array(
                            [values[max(0, tick - 4)], values[max(0, tick - 2)], value], np.float32
                        ).reshape(1, 3, 1)
                        expected = policy.model({"state": torch.from_numpy(states)}).detach().numpy()[0]
                        direct = runtime_model.predict_action_chunk(selected)
                        transported = await asyncio.to_thread(proxy.predict_action_chunk, selected)
                        assert direct.shape == transported.shape == (1, 1)
                        np.testing.assert_allclose(direct, expected, rtol=1e-6, atol=1e-6)
                        np.testing.assert_allclose(transported, expected, rtol=1e-6, atol=1e-6)
                        action = await asyncio.to_thread(source.update, observation, {}, tick)
                        assert action.shape == (1,)
                        np.testing.assert_allclose(action, expected[0], rtol=1e-6, atol=1e-6)
                    await asyncio.to_thread(source.reset)
            finally:
                source.disconnect()
        assert not proxy._worker.is_alive()
        assert proxy._worker.failure is None
