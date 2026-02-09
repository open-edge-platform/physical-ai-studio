from pathlib import Path

from getiaction.data import DataModule
from getiaction.export import Export
from getiaction.inference import InferenceModel
from getiaction.policies import ACT, ACTModel, Pi0, SmolVLA
from getiaction.policies.base import Policy
from loguru import logger

from schemas import Model


def load_policy(model: Model) -> Export | Policy:
    """Load existing model."""
    model_path = str(Path(model.path) / "model.ckpt")
    if model.policy == "act":
        return ACT.load_from_checkpoint(model_path)
    if model.policy == "pi0":
        return Pi0.load_from_checkpoint(model_path, weights_only=True)
    if model.policy == "smolvla":
        return SmolVLA.load_from_checkpoint(model_path)
    raise ValueError(f"Policy {model.policy} not implemented.")


def load_inference_model(model: Model, backend: str) -> InferenceModel:
    """Loads inference model if available, otherwise generates it."""
    export_dir = Path(model.path) / "exports" / backend

    if not export_dir.is_dir():
        logger.info(f"Export not available yet. Loading model and exporting {backend}")
        policy = load_policy(model)
        policy.export(export_dir, backend=backend)

    return InferenceModel(export_dir=export_dir, policy_name=model.policy, backend=backend)


def setup_policy(model: Model, l_dm: DataModule) -> Policy:
    """Setup policy for Model training."""
    if model.policy == "act":
        lib_model = ACTModel(
            input_features=l_dm.train_dataset.observation_features,
            output_features=l_dm.train_dataset.action_features,
        )

        return ACT(model=lib_model)
    if model.policy == "pi0":
        return Pi0(
            variant="pi0",
            chunk_size=50,
            learning_rate=2.5e-5,
        )
    if model.policy == "smolvla":
        return SmolVLA()

    raise ValueError(f"Policy not implemented yet: {model.policy}")
