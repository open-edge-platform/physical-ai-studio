from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

from workers.model_worker import RTC_INPUTS, Inference, InferRequest, RtcInputs

if TYPE_CHECKING:
    from pathlib import Path


class _Recorder:
    """Keeps what the worker hands the model."""

    def __init__(self) -> None:
        self.observation: dict[str, Any] = {}

    def predict_action_chunk(self, observation: dict[str, Any]) -> np.ndarray:
        self.observation = observation
        return np.zeros((4, 2), dtype=np.float32)

    def reset(self) -> None:
        pass


def _inference(rtc_chunk: tuple[int, int] | None) -> tuple[Inference, _Recorder]:
    model = _Recorder()
    return Inference(model, cast("Any", None), {}, rtc_chunk), model


def _request(rtc: RtcInputs | None = None) -> InferRequest:
    return InferRequest(state=[0.1, 0.2], task="pick", rtc=rtc)


def test_an_rtc_export_gets_the_rtc_inputs_it_is_sent() -> None:
    inference, model = _inference((4, 2))
    left_over = [[1.0, 2.0], [3.0, 4.0], [0.0, 0.0], [0.0, 0.0]]
    rtc = RtcInputs(prev_chunk_left_over=left_over, inference_delay=1, max_guidance_weight=5, execution_horizon=2)
    inference.infer(_request(rtc))
    sent = model.observation
    assert sent["prev_chunk_left_over"].shape == (1, 4, 2)
    np.testing.assert_array_equal(sent["prev_chunk_left_over"][0], left_over)
    assert (int(sent["inference_delay"]), float(sent["max_guidance_weight"]), int(sent["execution_horizon"])) == (
        1,
        5.0,
        2,
    )


def test_an_rtc_export_run_without_rtc_gets_the_neutral_inputs() -> None:
    inference, model = _inference((4, 2))
    inference.infer(_request())
    sent = model.observation
    assert set(RTC_INPUTS) <= set(sent)
    assert not sent["prev_chunk_left_over"].any()
    assert sent["prev_chunk_left_over"].shape == (1, 4, 2)
    assert float(sent["max_guidance_weight"]) == pytest.approx(1e-8)
    assert (int(sent["inference_delay"]), int(sent["execution_horizon"])) == (0, 1)


def test_a_plain_export_refuses_rtc_inputs() -> None:
    inference, model = _inference(None)
    inference.infer(_request())
    assert not set(RTC_INPUTS) & set(model.observation)
    rtc = RtcInputs(prev_chunk_left_over=[[0.0, 0.0]], inference_delay=0, max_guidance_weight=5, execution_horizon=1)
    with pytest.raises(ValueError, match="no RTC inputs"):
        inference.infer(_request(rtc))


def test_a_left_over_of_another_shape_is_refused() -> None:
    inference, _ = _inference((4, 2))
    rtc = RtcInputs(prev_chunk_left_over=[[0.0, 0.0]], inference_delay=0, max_guidance_weight=5, execution_horizon=1)
    with pytest.raises(ValueError, match=r"takes \(4, 2\)"):
        inference.infer(_request(rtc))


def _tiny_rtc_onnx(path: Path, *, rtc: bool = True) -> Path:
    """A 10 x 2 chunk ONNX graph: state, plus weight * tail when it takes RTC inputs (delay and horizon times 0)."""
    from onnx import TensorProto, helper, save

    inputs = [helper.make_tensor_value_info("state", TensorProto.FLOAT, [1, 2])]
    nodes = [helper.make_node("Unsqueeze", ["state", "axis"], ["base"])]
    initializers = [helper.make_tensor("axis", TensorProto.INT64, [1], [1])]
    out = "base"
    if rtc:
        inputs += [
            helper.make_tensor_value_info("prev_chunk_left_over", TensorProto.FLOAT, [1, 10, 2]),
            helper.make_tensor_value_info("inference_delay", TensorProto.INT64, []),
            helper.make_tensor_value_info("max_guidance_weight", TensorProto.FLOAT, []),
            helper.make_tensor_value_info("execution_horizon", TensorProto.INT64, []),
        ]
        nodes += [
            helper.make_node("Mul", ["prev_chunk_left_over", "max_guidance_weight"], ["guided"]),
            helper.make_node("Add", ["inference_delay", "execution_horizon"], ["ticks"]),
            helper.make_node("Cast", ["ticks"], ["ticks_f"], to=TensorProto.FLOAT),
            helper.make_node("Mul", ["ticks_f", "zero"], ["unused"]),
            helper.make_node("Add", ["base", "guided"], ["sum"]),
            helper.make_node("Add", ["sum", "unused"], ["action"]),
        ]
        initializers.append(helper.make_tensor("zero", TensorProto.FLOAT, [], [0.0]))
        out = "action"
    else:
        nodes.append(helper.make_node("Expand", ["base", "chunk"], ["action"]))
        initializers.append(helper.make_tensor("chunk", TensorProto.INT64, [3], [1, 10, 2]))
        out = "action"
    graph = helper.make_graph(
        nodes, "tiny", inputs, [helper.make_tensor_value_info(out, TensorProto.FLOAT, None)], initializers
    )
    save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)]), path)
    return path


@pytest.mark.parametrize("guided", [False, True], ids=["neutral", "guided"])
def test_onnx_runtime_takes_the_rtc_inputs_the_worker_sends(tmp_path: Path, guided: bool) -> None:
    from physicalai.inference.adapters.onnx import ONNXAdapter

    adapter = ONNXAdapter()
    adapter.load(_tiny_rtc_onnx(tmp_path / "tiny.onnx"))
    inference, model = _inference((10, 2))
    tail = [[1.0, 1.0]] * 3 + [[0.0, 0.0]] * 7
    rtc = RtcInputs(prev_chunk_left_over=tail, inference_delay=1, max_guidance_weight=2, execution_horizon=3)
    inference.infer(_request(rtc if guided else None))
    sent = {name: value for name, value in model.observation.items() if name != "task"}

    action = adapter.predict(sent)["action"]

    assert action.shape == (1, 10, 2)
    np.testing.assert_allclose(action[0, 0], [2.1, 2.2] if guided else [0.1, 0.2], rtol=1e-5)


def _tiny_rtc_openvino(path: Path) -> Path:
    """The same graph as ``_tiny_rtc_onnx``, built with OpenVINO."""
    import openvino as ov
    import openvino.opset13 as ops

    state = ops.parameter([1, 2], np.float32, name="state")
    tail = ops.parameter([1, 10, 2], np.float32, name="prev_chunk_left_over")
    delay = ops.parameter([], np.int64, name="inference_delay")
    weight = ops.parameter([], np.float32, name="max_guidance_weight")
    horizon = ops.parameter([], np.int64, name="execution_horizon")
    for parameter in (state, tail, delay, weight, horizon):
        parameter.output(0).set_names({parameter.get_friendly_name()})
    unused = ops.multiply(ops.convert(ops.add(delay, horizon), np.float32), ops.constant(np.float32(0)))
    base = ops.unsqueeze(state, ops.constant(np.array([1], np.int64)))
    action = ops.add(ops.add(base, ops.multiply(tail, weight)), unused)
    result = ops.result(action)
    result.output(0).set_names({"action"})
    ov.save_model(ov.Model([result], [state, tail, delay, weight, horizon], "tiny"), path)
    return path


@pytest.mark.parametrize("guided", [False, True], ids=["neutral", "guided"])
def test_openvino_takes_the_rtc_inputs_the_worker_sends(tmp_path: Path, guided: bool) -> None:
    from physicalai.inference.adapters.openvino import OpenVINOAdapter

    adapter = OpenVINOAdapter()
    adapter.load(_tiny_rtc_openvino(tmp_path / "tiny.xml"))
    inference, model = _inference((10, 2))
    tail = [[1.0, 1.0]] * 3 + [[0.0, 0.0]] * 7
    rtc = RtcInputs(prev_chunk_left_over=tail, inference_delay=1, max_guidance_weight=2, execution_horizon=3)
    inference.infer(_request(rtc if guided else None))
    sent = {name: value for name, value in model.observation.items() if name != "task"}

    action = next(iter(adapter.predict(sent).values()))

    np.testing.assert_allclose(np.asarray(action)[0, 0], [2.1, 2.2] if guided else [0.1, 0.2], rtol=1e-5)


def test_the_worker_sends_rank_zero_arrays_of_the_declared_dtypes() -> None:
    model = _Recorder()
    dtypes = {
        "prev_chunk_left_over": "float16",
        "inference_delay": "int32",
        "max_guidance_weight": "float16",
        "execution_horizon": "int32",
    }
    inference = Inference(model, cast("Any", None), {}, (4, 2), dtypes)
    inference.infer(_request())
    for name, dtype in dtypes.items():
        sent = model.observation[name]
        assert isinstance(sent, np.ndarray)
        assert sent.dtype == np.dtype(dtype)
        assert sent.ndim == (3 if name == "prev_chunk_left_over" else 0)
