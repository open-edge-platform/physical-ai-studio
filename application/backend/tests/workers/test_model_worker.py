from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from workers.model_worker import RTC_INPUTS, Inference, InferRequest, RtcInputs


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
