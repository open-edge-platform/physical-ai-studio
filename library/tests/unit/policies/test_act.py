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
from physicalai.inference.preprocessors.resize import ResizePreprocessor
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
        sample_input = policy.sample_input

        assert isinstance(sample_input, dict)
        assert "state" in sample_input
        assert "images" in sample_input

    def test_save_hyperparameters_ignores_compile_model(self):
        """Test compile_model is excluded from saved hyperparameters."""
        policy = ACT(compile_model=True)
        assert "compile_model" not in policy.hparams

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

    def test_configure_optimizers_param_groups(self, policy):
        """Test optimizer parameter groups separation and reference defaults."""
        assert policy.config.optimizer_lr == 1e-5
        assert policy.config.optimizer_lr_backbone == 1e-5
        assert policy.config.optimizer_grad_clip_norm == 10.0

        opt = policy.configure_optimizers()["optimizer"]
        assert len(opt.param_groups) == 2

        # Verify parameter identity split
        backbone_param_ids = {id(p) for n, p in policy.named_parameters() if ".backbone." in n and p.requires_grad}
        other_param_ids = {id(p) for n, p in policy.named_parameters() if ".backbone." not in n and p.requires_grad}

        group0_ids = {id(p) for p in opt.param_groups[0]["params"]}
        group1_ids = {id(p) for p in opt.param_groups[1]["params"]}

        assert group0_ids == other_param_ids
        assert group1_ids == backbone_param_ids
        assert len(group1_ids) > 0
        assert len(group0_ids) > 0

        # Group 0: non-backbone parameters
        assert opt.param_groups[0]["lr"] == 1e-5
        # Group 1: backbone parameters
        assert opt.param_groups[1]["lr"] == 1e-5

        # Verify distinct custom learning rates propagate to each group
        custom_policy = ACT(
            dataset_stats=policy.hparams["dataset_stats"],
            optimizer_lr=2e-5,
            optimizer_lr_backbone=5e-6,
        )
        custom_opt = custom_policy.configure_optimizers()["optimizer"]
        assert len(custom_opt.param_groups) == 2
        assert custom_opt.param_groups[0]["lr"] == 2e-5
        assert custom_opt.param_groups[1]["lr"] == 5e-6

    def test_single_camera_input_normalization(self):
        """Single-camera observations with flattened 'images.<name>' keys are normalized."""
        from physicalai.data import FeatureType

        dataset_stats = {
            "observation.images.wrist": {
                "name": "observation.images.wrist",
                "type": FeatureType.VISUAL,
                "shape": (3, 64, 64),
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
            "observation.state": {
                "name": "observation.state",
                "type": FeatureType.STATE,
                "shape": (3,),
                "mean": [0.0] * 3,
                "std": [1.0] * 3,
            },
            "action": {
                "name": "action",
                "type": FeatureType.ACTION,
                "shape": (3,),
                "mean": [0.0] * 3,
                "std": [1.0] * 3,
            },
        }
        policy = ACT(dataset_stats=dataset_stats)
        batch = {
            "images.wrist": torch.ones(1, 3, 64, 64),
            "state": torch.zeros(1, 3),
            "action": torch.zeros(1, 100, 3),
            "extra.action_is_pad": torch.zeros(1, 100, dtype=torch.bool),
        }
        orig_wrist = batch["images.wrist"].clone()
        normed = policy.model._input_normalizer(dict(batch))
        assert not torch.equal(normed["images.wrist"], orig_wrist)
        expected_ch0 = (1.0 - 0.485) / 0.229
        assert torch.isclose(normed["images.wrist"][0, 0, 0, 0], torch.tensor(expected_ch0), atol=1e-4)

    def test_multi_camera_input_normalization(self):
        """Multi-camera observations with flattened 'images.<name>' keys are normalized."""
        from physicalai.data import FeatureType

        dataset_stats = {
            "observation.images.wrist": {
                "name": "observation.images.wrist",
                "type": FeatureType.VISUAL,
                "shape": (3, 64, 64),
                "mean": [0.5, 0.5, 0.5],
                "std": [0.25, 0.25, 0.25],
            },
            "observation.images.overhead": {
                "name": "observation.images.overhead",
                "type": FeatureType.VISUAL,
                "shape": (3, 64, 64),
                "mean": [0.2, 0.2, 0.2],
                "std": [0.1, 0.1, 0.1],
            },
            "observation.state": {
                "name": "observation.state",
                "type": FeatureType.STATE,
                "shape": (3,),
                "mean": [0.0] * 3,
                "std": [1.0] * 3,
            },
            "action": {
                "name": "action",
                "type": FeatureType.ACTION,
                "shape": (3,),
                "mean": [0.0] * 3,
                "std": [1.0] * 3,
            },
        }
        policy = ACT(dataset_stats=dataset_stats, use_imagenet_stats=False)
        batch = {
            "images.wrist": torch.ones(1, 3, 64, 64),
            "images.overhead": torch.ones(1, 3, 64, 64),
            "state": torch.zeros(1, 3),
            "action": torch.zeros(1, 100, 3),
            "extra.action_is_pad": torch.zeros(1, 100, dtype=torch.bool),
        }
        normed = policy.model._input_normalizer(dict(batch))
        expected_wrist = (1.0 - 0.5) / 0.25
        expected_overhead = (1.0 - 0.2) / 0.1
        assert torch.isclose(normed["images.wrist"][0, 0, 0, 0], torch.tensor(expected_wrist), atol=1e-4)
        assert torch.isclose(normed["images.overhead"][0, 0, 0, 0], torch.tensor(expected_overhead), atol=1e-4)

    def test_imagenet_stats_normalization(self):
        """Visual features default to ImageNet mean and std when use_imagenet_stats is True."""
        from physicalai.data import FeatureType
        from physicalai.policies.act.config import IMAGENET_MEAN, IMAGENET_STD

        dataset_stats = {
            "observation.images.top": {
                "name": "observation.images.top",
                "type": FeatureType.VISUAL,
                "shape": (3, 64, 64),
                "mean": [0.1, 0.2, 0.3],
                "std": [0.01, 0.02, 0.03],
            },
            "observation.state": {
                "name": "observation.state",
                "type": FeatureType.STATE,
                "shape": (3,),
                "mean": [1.0, 2.0, 3.0],
                "std": [0.5, 0.5, 0.5],
            },
            "action": {
                "name": "action",
                "type": FeatureType.ACTION,
                "shape": (3,),
                "mean": [0.0] * 3,
                "std": [1.0] * 3,
            },
        }
        policy = ACT(dataset_stats=dataset_stats)
        assert policy.config.use_imagenet_stats is True
        # Visual stats should be overridden to ImageNet values
        assert policy._dataset_stats["observation.images.top"]["mean"] == IMAGENET_MEAN
        assert policy._dataset_stats["observation.images.top"]["std"] == IMAGENET_STD
        # State stats should remain intact
        assert policy._dataset_stats["observation.state"]["mean"] == [1.0, 2.0, 3.0]

        batch = {
            "images.top": torch.ones(1, 3, 64, 64),
            "state": torch.tensor([[1.0, 2.0, 3.0]]),
            "action": torch.zeros(1, 100, 3),
            "extra.action_is_pad": torch.zeros(1, 100, dtype=torch.bool),
        }
        normed = policy.model._input_normalizer(dict(batch))
        expected_top = (1.0 - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
        assert torch.isclose(normed["images.top"][0, 0, 0, 0], torch.tensor(expected_top), atol=1e-4)
        assert torch.allclose(normed["state"], torch.zeros(1, 3), atol=1e-4)

        # When use_imagenet_stats is False, empirical stats are preserved
        policy_empirical = ACT(dataset_stats=dataset_stats, use_imagenet_stats=False)
        assert policy_empirical.config.use_imagenet_stats is False
        assert policy_empirical._dataset_stats["observation.images.top"]["mean"] == [0.1, 0.2, 0.3]
        normed_empirical = policy_empirical.model._input_normalizer(dict(batch))
        expected_top_emp = (1.0 - 0.1) / 0.01
        assert torch.isclose(normed_empirical["images.top"][0, 0, 0, 0], torch.tensor(expected_top_emp), atol=1e-4)


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

    def test_resize_max_pads_small_image_to_target(self):
        """Images smaller than the target are scaled up and padded to the exact target size."""
        img = self._img(h=64, w=64)
        out = ACTPreprocessor._resize_with_ar_pad(img, target_width=128, target_height=128)
        assert out.shape == (img.shape[0], img.shape[1], 128, 128)

    def test_resize_max_reduces_oversized_image(self):
        """Oversized images are downscaled and padded to the exact target size."""
        img = self._img(h=480, w=640)
        out = ACTPreprocessor._resize_with_ar_pad(img, target_width=320, target_height=240)
        assert out.shape[2] == 240
        assert out.shape[3] == 320

    def test_resize_max_preserves_aspect_ratio(self):
        """Aspect ratio of the image content is maintained, with the remainder zero-padded."""
        img = self._img(h=400, w=200)  # 2:1 height:width
        out = ACTPreprocessor._resize_with_ar_pad(img, target_width=100, target_height=100)
        # Output is exactly the target size.
        assert out.shape[2] == 100
        assert out.shape[3] == 100
        # Content scaled to 100x50 (2:1) and centred, so side columns are zero padding.
        assert torch.all(out[:, :, :, :25] == 0)
        assert torch.all(out[:, :, :, 75:] == 0)

    def test_resize_max_fits_exactly(self):
        """An image exactly at the target size is returned at the target size."""
        img = self._img(h=128, w=128)
        out = ACTPreprocessor._resize_with_ar_pad(img, target_width=128, target_height=128)
        assert out.shape[2] == 128
        assert out.shape[3] == 128

    def test_resize_max_raises_on_wrong_ndim(self):
        """Non-4D tensors raise ValueError."""
        img_3d = torch.rand(3, 64, 64)
        with pytest.raises(ValueError, match="b,c,h,w"):
            ACTPreprocessor._resize_with_ar_pad(img_3d, target_width=64, target_height=64)

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

    def test_forward_flat_tensor_images_small_padded_to_target(self):
        """Small images are scaled up and padded to the target resolution."""
        pre = ACTPreprocessor(image_resolution=(256, 256))
        img = self._img(h=64, w=64)
        batch = {IMAGES: img}
        out = pre(batch)
        assert out[IMAGES].shape == (img.shape[0], img.shape[1], 256, 256)

    def test_forward_does_not_mutate_input(self):
        """forward() operates on a copy and does not mutate the original batch."""
        pre = ACTPreprocessor(image_resolution=(64, 64))
        img = self._img(h=256, w=256)
        batch = {IMAGES: img}
        original_shape = img.shape
        pre(batch)
        assert batch[IMAGES].shape == original_shape

    def test_forward_non_square_resolution_preserves_height_width_order(self):
        """A non-square image_resolution=(height, width) is not swapped through forward()."""
        pre = ACTPreprocessor(image_resolution=(240, 320))
        batch = {IMAGES: self._img(h=480, w=640)}
        out = pre(batch)
        assert out[IMAGES].shape[2] == 240
        assert out[IMAGES].shape[3] == 320

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

    # ------------------------------------------------------------------
    # cross-implementation consistency – torch vs numpy resize
    # ------------------------------------------------------------------

    def test_resize_matches_numpy_resize_preprocessor(self):
        """Torch ACTPreprocessor resize matches the numpy ResizePreprocessor.

        Both implementations resize while preserving aspect ratio and zero-pad to
        the target resolution. The torch path uses ``F.interpolate`` (bilinear)
        and the numpy path uses ``cv2.resize`` (bilinear), so a small tolerance
        accounts for differences between the two interpolation backends.
        """
        resolution = (256, 256)
        img = self._img(b=1, h=480, w=640)

        torch_pre = ACTPreprocessor(image_resolution=resolution)
        numpy_pre = ResizePreprocessor(image_resolution=resolution)

        torch_out = torch_pre({IMAGES: img})[IMAGES]
        numpy_out = numpy_pre({IMAGES: img.numpy()})[IMAGES]

        assert torch_out.shape == numpy_out.shape
        assert np.mean(np.abs(torch_out.numpy() - numpy_out)) < 1e-2
