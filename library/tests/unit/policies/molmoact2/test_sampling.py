# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Gaussian initialization is the default, with explicit overrides preserved."""

from types import SimpleNamespace

import pytest
import torch

from physicalai.policies.molmoact2.components.backbone import MolmoAct2Backbone


@pytest.mark.parametrize("zero_noise", [False, True])
def test_sampling_defaults_to_gaussian_noise_with_explicit_override(zero_noise: bool) -> None:
    # A zero velocity field leaves exactly the sampled initialization visible.
    expert = SimpleNamespace(
        action_embed=torch.nn.Linear(4, 8),
        forward_with_context=lambda trajectory, *_args, **_kwargs: torch.zeros_like(trajectory),
    )
    backbone = SimpleNamespace(
        _require_action_expert=lambda: expert,
        _encode_action_context=lambda **_: None,
        _mask_action_dims=lambda value, _: value,
        flow_matching_num_steps=2,
        max_action_dim=4,
    )
    kwargs = {"sample_noise": False} if zero_noise else {}
    generator = torch.Generator().manual_seed(123)
    actual = MolmoAct2Backbone.generate_actions_from_inputs(
        backbone,
        input_ids=torch.ones(2, 1, dtype=torch.long),
        action_horizon=3,
        generator=generator,
        **kwargs,
    )
    expected = (
        torch.zeros(2, 3, 4) if zero_noise else torch.randn(2, 3, 4, generator=torch.Generator().manual_seed(123))
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_sampling_advances_generator_between_predictions() -> None:
    expert = SimpleNamespace(
        action_embed=torch.nn.Linear(4, 8),
        forward_with_context=lambda trajectory, *_args, **_kwargs: torch.zeros_like(trajectory),
    )
    backbone = SimpleNamespace(
        _require_action_expert=lambda: expert,
        _encode_action_context=lambda **_: None,
        _mask_action_dims=lambda value, _: value,
        flow_matching_num_steps=1,
        max_action_dim=4,
    )
    generator = torch.Generator().manual_seed(123)
    kwargs = {"input_ids": torch.ones(1, 1, dtype=torch.long), "action_horizon": 3, "generator": generator}
    first = MolmoAct2Backbone.generate_actions_from_inputs(backbone, **kwargs)
    second = MolmoAct2Backbone.generate_actions_from_inputs(backbone, **kwargs)
    assert not torch.equal(first, second)


def test_openvino_export_keeps_default_noise_stochastic() -> None:
    ov = pytest.importorskip("openvino")

    class Sampling(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.action_embed = torch.nn.Linear(4, 8)

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            expert = SimpleNamespace(
                action_embed=self.action_embed,
                forward_with_context=lambda trajectory, *_args, **_kwargs: torch.zeros_like(trajectory),
            )
            backbone = SimpleNamespace(
                _require_action_expert=lambda: expert,
                _encode_action_context=lambda **_: None,
                _mask_action_dims=lambda value, _: value,
                flow_matching_num_steps=1,
                max_action_dim=4,
            )
            return MolmoAct2Backbone.generate_actions_from_inputs(backbone, input_ids=input_ids, action_horizon=3)

    ids = torch.ones(1, 1, dtype=torch.long)
    converted = ov.convert_model(Sampling().eval(), example_input=ids)
    compiled = ov.Core().compile_model(converted, "CPU")
    # A zero velocity field makes the sampled trajectory directly observable;
    # its unused token input may be removed by OpenVINO's optimizer.
    inputs = [ids.numpy()] if compiled.inputs else []
    first = torch.from_numpy(compiled(inputs)[0].copy())
    second = torch.from_numpy(compiled(inputs)[0].copy())

    assert torch.isfinite(first).all()
    assert torch.isfinite(second).all()
    assert not torch.equal(first, second)
