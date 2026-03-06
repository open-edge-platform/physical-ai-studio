# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import copy
import tempfile
from unittest.mock import patch
import pytest
import torch
import numpy as np
import lightning

from physicalai.data import Observation
from physicalai.data.observation import IMAGES
from physicalai.policies import ACT
from physicalai.policies.act.model import ACT as ACTModel
from physicalai.policies.act.preprocessor import ACTPreprocessor


class TestACTolicy:
    """Tests for ACTPolicy and ACTModel."""

    @pytest.fixture
    def policy(self):
        policy = ACT(dataset_stats={"image": {"mean": [0.0]*3, "std": [1.0]*3, "type": "VISUAL", "name": "image", "shape": (3, 64, 64)},
                                    "state": {"mean": [0.0]*3, "std": [1.0]*3, "type": "STATE", "name": "state", "shape": (3,)},
                                    "action": {"mean": [0.0]*3, "std": [1.0]*3, "type": "ACTION", "name": "action", "shape": (3,)}},)
        return policy

    @pytest.fixture
    def batch(self):
        bs = 2
        return Observation(
            images=torch.randn(bs, 3, 64, 64),
            state=torch.randn(bs, 3),
            action=torch.randn(bs, 100, 3),  # 'bs' samples, 3 features, 100 action steps
            extra={"action_is_pad": torch.zeros(bs, 100, dtype=torch.bool)}
        )

    def test_initialization(self, policy):
        """Check model and action shape."""
        assert isinstance(policy.model, ACTModel)
        assert policy.model._input_normalizer is not None
        assert policy.model._output_denormalizer is not None

    def test_forward_training_and_eval(self, policy, batch):
        """Forward pass works in training and eval modes."""
        # Training
        policy.model.train()
        loss, loss_dict = policy.model(copy.deepcopy(batch).to_dict())
        assert isinstance(loss, torch.Tensor)
        assert loss >= 0
        assert loss_dict["kld_loss"] >= 0

        # Evaluation
        policy.model.eval()
        actions = policy.model(batch.to_dict())
        assert isinstance(actions, torch.Tensor)
        assert actions.shape == batch.action.shape

    def test_training_step(self, policy, batch):
        policy.model.train()
        loss = policy.training_step(batch, 0)

        assert "loss" in loss
        assert loss["loss"] >= 0

    def test_training_step_call(self, policy, batch):
        policy.model.train()

        with patch.object(policy, "forward", wraps=policy.forward) as mock_forward:
            policy.training_step(batch, 0)
            mock_forward.assert_called_once()

    def test_predict_action_chunk_with_explain(self, policy, batch):
        """Test predict_action_chunk_with_explain method."""
        policy.model.eval()
        actions, explain = policy.model.predict_action_chunk_with_explain(batch.to_dict())

        assert isinstance(actions, torch.Tensor)
        assert actions.shape == batch.action.shape
        assert isinstance(explain, torch.Tensor)
        assert explain.shape[0] == batch.action.shape[0]
        assert explain.shape[1] == 1
        assert explain.shape[2] > 1
        assert explain.shape[3] > 1

    def test_select_action(self, policy, batch):
        """Test select_action returns a single action (uses action queue)."""
        policy.eval()
        actions = policy.select_action(batch)

        assert isinstance(actions, torch.Tensor)
        assert actions.shape[0] == batch.images.shape[0]
        # select_action returns a single action, not a chunk
        assert actions.shape[1] == batch.action.shape[2]

    def test_predict_action_chunk(self, policy, batch):
        """Test predict_action_chunk returns the full action chunk."""
        policy.eval()
        actions = policy.predict_action_chunk(batch)

        assert isinstance(actions, torch.Tensor)
        assert actions.shape[0] == batch.images.shape[0]
        assert actions.shape[1] == policy.model._config.chunk_size
        assert actions.shape[2] == batch.action.shape[2]

    def test_sample_input(self, policy):
        """Test sample_input generation."""
        sample_input = policy.model.sample_input

        assert isinstance(sample_input, dict)
        assert "state" in sample_input
        assert "images" in sample_input

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_dtype_change(self, policy, batch, dtype):
        """Test model behavior with different input dtypes."""
        eval_policy = copy.deepcopy(policy)
        eval_policy = eval_policy.to(dtype).eval()

        input_batch = copy.deepcopy(batch).to_dict()
        input_batch["images"] = input_batch["images"].to(dtype)
        input_batch["state"] = input_batch["state"].to(dtype)

        actions = eval_policy.model(input_batch)
        assert isinstance(actions, torch.Tensor)
        assert actions.dtype == dtype

    def test_load_from_checkpoint(self, policy):
        """Test checkpoint save and load preserves model config and weights."""
        # Save checkpoint manually (simulating Lightning checkpoint)
        with tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False) as f:
            checkpoint_path = f.name

        try:
            checkpoint = {"state_dict": policy.state_dict()}
            checkpoint["epoch"] = 0
            checkpoint["global_step"] = 0
            checkpoint["pytorch-lightning_version"] = lightning.__version__
            checkpoint["loops"] = {}
            checkpoint["hparams_name"] = "kwargs"
            checkpoint["hyper_parameters"] = dict(policy.hparams)

            # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch
            torch.save(checkpoint, checkpoint_path)

            # Load from checkpoint
            loaded_policy = ACT.load_from_checkpoint(checkpoint_path)

            # Verify model type
            assert isinstance(loaded_policy.model, ACTModel)

            # Verify config is preserved
            assert list(loaded_policy.model.config.input_features.keys()) == list(
                policy.model.config.input_features.keys()
            )
            assert list(loaded_policy.model.config.output_features.keys()) == list(
                policy.model.config.output_features.keys()
            )
            assert loaded_policy.model.config.chunk_size == policy.model.config.chunk_size

            # Verify weights are loaded correctly
            orig_params = list(policy.model.parameters())
            loaded_params = list(loaded_policy.model.parameters())
            assert len(orig_params) == len(loaded_params)
            for orig, loaded in zip(orig_params, loaded_params, strict=True):
                assert torch.allclose(orig, loaded), "Weights should match after loading"

        finally:
            import os
            os.unlink(checkpoint_path)

    def test_load_from_exported_checkpoint(self, policy):
        """Test loading from an exported checkpoint."""
        # Export the model
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            export_path = f.name

        try:
            policy.to_torch(export_path)

            # Load from exported checkpoint
            loaded_policy = ACT.load_from_checkpoint(export_path)

            # Verify model type
            assert isinstance(loaded_policy.model, ACTModel)

            # Verify config is preserved
            assert list(loaded_policy.model.config.input_features.keys()) == list(
                policy.model.config.input_features.keys()
            )
            assert list(loaded_policy.model.config.output_features.keys()) == list(
                policy.model.config.output_features.keys()
            )
            assert loaded_policy.model.config.chunk_size == policy.model.config.chunk_size

        finally:
            import os
            os.unlink(export_path)


class TestACTPreprocessor:
    """Tests for ACTPreprocessor."""

    @staticmethod
    def _img(b: int = 2, c: int = 3, h: int = 480, w: int = 640) -> torch.Tensor:
        return torch.rand(b, c, h, w)

    def test_default_resolution(self):
        pre = ACTPreprocessor()
        assert pre.image_resolution == (512, 512)

    def test_custom_resolution(self):
        pre = ACTPreprocessor(image_resolution=(224, 224))
        assert pre.image_resolution == (224, 224)

    def test_resize_max_no_op_when_within_bounds(self):
        """Images already within bounds are returned unchanged."""
        img = self._img(h=64, w=64)
        out = ACTPreprocessor._resize_max(img, max_width=128, max_height=128)
        assert out.shape == img.shape
        assert out is img  # no copy expected

    def test_resize_max_reduces_oversized_image(self):
        """Oversized images are downscaled to fit within the max bounds."""
        img = self._img(h=480, w=640)
        out = ACTPreprocessor._resize_max(img, max_width=320, max_height=240)
        assert out.shape[2] <= 240
        assert out.shape[3] <= 320

    def test_resize_max_preserves_aspect_ratio(self):
        """Aspect ratio is maintained after resizing."""
        img = self._img(h=400, w=200)  # 2:1 height:width
        out = ACTPreprocessor._resize_max(img, max_width=100, max_height=100)
        out_h, out_w = out.shape[2], out.shape[3]
        assert abs(out_h / out_w - 2.0) < 0.05

    def test_resize_max_fits_exactly(self):
        """An image exactly at the limit is not resized."""
        img = self._img(h=128, w=128)
        out = ACTPreprocessor._resize_max(img, max_width=128, max_height=128)
        assert out.shape[2] == 128
        assert out.shape[3] == 128

    def test_resize_max_raises_on_wrong_ndim(self):
        """Non-4D tensors raise ValueError."""
        img_3d = torch.rand(3, 64, 64)
        with pytest.raises(ValueError, match="b,c,h,w"):
            ACTPreprocessor._resize_max(img_3d, max_width=64, max_height=64)

    # ------------------------------------------------------------------
    # forward – flat batch with images as a direct tensor
    # ------------------------------------------------------------------

    def test_forward_flat_tensor_images_resized(self):
        """Images stored as a flat tensor under the IMAGES key are resized."""
        pre = ACTPreprocessor(image_resolution=(64, 64))
        batch = {IMAGES: self._img(h=128, w=128), "state": torch.rand(2, 3)}
        out = pre(batch)
        assert out[IMAGES].shape[2] <= 64
        assert out[IMAGES].shape[3] <= 64

    def test_forward_flat_tensor_images_no_resize_when_small(self):
        """Small images are not resized."""
        pre = ACTPreprocessor(image_resolution=(256, 256))
        img = self._img(h=64, w=64)
        batch = {IMAGES: img}
        out = pre(batch)
        assert out[IMAGES].shape == img.shape

    def test_forward_does_not_mutate_input(self):
        """forward() operates on a copy and does not mutate the original batch."""
        pre = ACTPreprocessor(image_resolution=(64, 64))
        img = self._img(h=256, w=256)
        batch = {IMAGES: img}
        original_shape = img.shape
        pre(batch)
        assert batch[IMAGES].shape == original_shape

    # ------------------------------------------------------------------
    # forward – nested dict images (multiple cameras)
    # ------------------------------------------------------------------

    def test_forward_dict_images_all_cameras_resized(self):
        """All camera images inside a dict are resized."""
        pre = ACTPreprocessor(image_resolution=(64, 64))
        batch = {
            IMAGES: {
                "top": self._img(h=256, w=256),
                "wrist": self._img(h=480, w=640),
            }
        }
        out = pre(batch)
        for key in ("top", "wrist"):
            assert out[IMAGES][key].shape[2] <= 64
            assert out[IMAGES][key].shape[3] <= 64

    def test_forward_dict_images_preserves_other_keys(self):
        """Non-image keys in the batch are preserved unchanged."""
        pre = ACTPreprocessor(image_resolution=(64, 64))
        state = torch.rand(2, 8)
        batch = {
            IMAGES: {"top": self._img(h=128, w=128)},
            "state": state,
        }
        out = pre(batch)
        assert torch.equal(out["state"], state)

    # ------------------------------------------------------------------
    # forward – flattened observation dict (images.camera keys)
    # ------------------------------------------------------------------

    def test_forward_flat_keys_images_resized(self):
        """Images stored under flat 'images.camera' keys are resized."""
        pre = ACTPreprocessor(image_resolution=(64, 64))
        batch = {
            "images.top": self._img(h=256, w=256),
            "images.wrist": self._img(h=128, w=128),
            "state": torch.rand(2, 3),
        }
        out = pre(batch)
        assert out["images.top"].shape[2] <= 64
        assert out["images.top"].shape[3] <= 64
        assert out["images.wrist"].shape[2] <= 64
        assert out["images.wrist"].shape[3] <= 64

    def test_forward_flat_keys_is_pad_skipped(self):
        """Keys containing 'is_pad' are not treated as images and are left untouched."""
        pre = ACTPreprocessor(image_resolution=(64, 64))
        pad_tensor = torch.zeros(2, 100, dtype=torch.bool)
        batch = {
            "images.top": self._img(h=128, w=128),
            "images.top_is_pad": pad_tensor,
        }
        out = pre(batch)
        # is_pad tensor should be identical (not passed through _resize_max)
        assert torch.equal(out["images.top_is_pad"], pad_tensor)
