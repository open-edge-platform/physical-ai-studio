"""Studio training defaults are static, documented, and valid training payloads."""

import subprocess
import sys
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from pydantic import Field, ValidationError

from main import app
from schemas.job import LocalTrainJobPayload
from training.policy_catalog import POLICY_TRAINING_CONFIGURATIONS
from training.policy_training_configuration import (
    BatchSize,
    LoraDefaults,
    PolicyTrainingConfiguration,
    PolicyTrainingDefaults,
)


def _default_configuration(policy: str) -> PolicyTrainingConfiguration:
    response = TestClient(app).get(f"/api/policies/{policy}/default-training-config")
    assert response.status_code == 200
    return PolicyTrainingConfiguration.model_validate(response.json())


def test_no_policy_has_a_recipe_yet() -> None:
    assert POLICY_TRAINING_CONFIGURATIONS == {}


@pytest.mark.parametrize("policy", ["act", "pi05", "unknown-policy"])
def test_every_policy_uses_the_shared_defaults(policy: str) -> None:
    configuration = _default_configuration(policy)
    assert configuration.defaults == PolicyTrainingDefaults()
    assert configuration.help == {}


def test_defaults_are_valid_persistable_payloads() -> None:
    defaults = PolicyTrainingDefaults()
    values = defaults.model_dump(exclude={"validation", "lora", "snapflow", "precision"})
    lora = defaults.lora or LoraDefaults()
    payload = LocalTrainJobPayload(
        project_id=uuid4(),
        dataset_id=uuid4(),
        policy="act",
        model_name="act",
        **values,
        **({"precision": defaults.precision} if defaults.precision is not None else {}),
        val_split=defaults.validation.split,
        lora_enabled=defaults.lora is not None,
        lora_rank=lora.rank,
        lora_alpha=lora.alpha,
        lora_dropout=lora.dropout,
        lora_use_dora=lora.use_dora,
        snapflow_enabled=defaults.snapflow is not None,
        snapflow_distill_epochs=defaults.snapflow.distill_epochs if defaults.snapflow else 3,
    )
    assert LocalTrainJobPayload.model_validate_json(payload.model_dump_json()) == payload


def test_lookup_does_not_load_policy_dependencies() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import asyncio, sys; from api.policies import get_default_training_configuration; "
                "assert asyncio.run(get_default_training_configuration('pi05')).defaults.batch_size == 8; "
                "assert 'physicalai.policies' not in sys.modules"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_help_is_derived_from_overridden_field_descriptions() -> None:
    class LoraDefaultsWithHelp(LoraDefaults):
        rank: int = Field(default=16, ge=1, le=256, description="Start with a smaller adapter.")

    class DefaultsWithHelp(PolicyTrainingDefaults):
        batch_size: BatchSize = Field(default=16, description="This policy prefers a larger batch.")
        lora: LoraDefaultsWithHelp | None = Field(
            default_factory=LoraDefaultsWithHelp, description="This policy uses adapters."
        )

    configuration = PolicyTrainingConfiguration.from_defaults(DefaultsWithHelp())
    assert configuration.help == {
        "batch_size": "This policy prefers a larger batch.",
        "lora": "This policy uses adapters.",
        "lora.rank": "Start with a smaller adapter.",
    }


@pytest.mark.parametrize("model", ["PolicyTrainingDefaults", "LoraDefaults", "SnapflowDefaults", "ValidationDefaults"])
def test_openapi_documents_defaults(model: str) -> None:
    schema = app.openapi()["components"]["schemas"][model]
    assert all(field.get("description") for field in schema["properties"].values())
    assert set(schema["required"]) == set(schema["properties"])


def test_optional_methods_are_disabled_by_null_not_an_enabled_flag() -> None:
    defaults = PolicyTrainingDefaults()
    assert defaults.lora is None
    assert defaults.snapflow is None
    with pytest.raises(ValidationError):
        LoraDefaults(enabled=False)
