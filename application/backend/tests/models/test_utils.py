from pathlib import Path
from unittest.mock import patch

from models.utils import load_inference_model
from schemas import InferenceBackend, InferenceDevice


def test_load_inference_model_uses_selected_torch_device(test_model) -> None:
    inference_device = InferenceDevice(backend=InferenceBackend.TORCH, device="cuda:0")

    with patch("models.utils.InferenceModel") as mock_inference_model:
        load_inference_model(test_model, inference_device=inference_device)

    mock_inference_model.assert_called_once_with(
        export_dir=Path(test_model.path) / "exports" / "torch",
        policy_name=test_model.policy,
        backend="torch",
        device="cuda:0",
    )


def test_load_inference_model_uses_selected_openvino_device(test_model) -> None:
    inference_device = InferenceDevice(backend=InferenceBackend.OPENVINO, device="GPU")

    with patch("models.utils.InferenceModel") as mock_inference_model:
        load_inference_model(test_model, inference_device=inference_device)

    mock_inference_model.assert_called_once_with(
        export_dir=Path(test_model.path) / "exports" / "openvino",
        policy_name=test_model.policy,
        backend="openvino",
        device="GPU",
    )
