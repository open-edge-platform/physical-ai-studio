from uuid import uuid4

import pytest
from pydantic import ValidationError

from schemas.hardware import DeviceType
from schemas.remote_server import (
    LastCheckSummary,
    RemoteServer,
    RemoteServerCreate,
    RemoteServerInternal,
    RemoteServerUpdate,
    SSHAuthType,
)

_SECRET_FIELDS = {"ssh_secret_encrypted", "ssh_key_passphrase_encrypted", "host_key"}


def _internal_record(**overrides) -> RemoteServerInternal:
    defaults = {
        "id": uuid4(),
        "name": "gpu-box",
        "host": "10.0.0.5",
        "port": 22,
        "username": "trainer",
        "auth_type": SSHAuthType.KEY,
        "device_type": DeviceType.CUDA,
        "last_check": LastCheckSummary(),
        "ssh_secret_encrypted": "ciphertext-secret",
        "ssh_key_passphrase_encrypted": "ciphertext-passphrase",
        "host_key": "ssh-ed25519 AAAA...",
    }
    defaults.update(overrides)
    return RemoteServerInternal(**defaults)


def test_create_trims_whitespace() -> None:
    config = RemoteServerCreate(
        name="  gpu-box  ",
        host="  10.0.0.5  ",
        username="  trainer  ",
        auth_type=SSHAuthType.PASSWORD,
        device_type=DeviceType.CUDA,
        ssh_secret="hunter2",
    )

    assert config.name == "gpu-box"
    assert config.host == "10.0.0.5"
    assert config.username == "trainer"


def test_create_rejects_passphrase_with_password_auth() -> None:
    with pytest.raises(ValidationError, match="ssh_key_passphrase is only valid"):
        RemoteServerCreate(
            name="gpu-box",
            host="10.0.0.5",
            username="trainer",
            auth_type=SSHAuthType.PASSWORD,
            device_type=DeviceType.CUDA,
            ssh_secret="hunter2",
            ssh_key_passphrase="unexpected",
        )


def test_create_allows_passphrase_with_key_auth() -> None:
    config = RemoteServerCreate(
        name="gpu-box",
        host="10.0.0.5",
        username="trainer",
        auth_type=SSHAuthType.KEY,
        device_type=DeviceType.CUDA,
        ssh_secret="test-key-material",
        ssh_key_passphrase="unlock-me",
    )

    assert config.ssh_key_passphrase == "unlock-me"


def test_create_rejects_cpu_and_npu_device_types() -> None:
    for device_type in (DeviceType.CPU, DeviceType.NPU):
        with pytest.raises(ValidationError):
            RemoteServerCreate(
                name="gpu-box",
                host="10.0.0.5",
                username="trainer",
                auth_type=SSHAuthType.PASSWORD,
                device_type=device_type,
                ssh_secret="hunter2",
            )


def test_create_rejects_out_of_range_port() -> None:
    with pytest.raises(ValidationError):
        RemoteServerCreate(
            name="gpu-box",
            host="10.0.0.5",
            port=70000,
            username="trainer",
            auth_type=SSHAuthType.PASSWORD,
            device_type=DeviceType.CUDA,
            ssh_secret="hunter2",
        )


def test_update_all_fields_optional() -> None:
    update = RemoteServerUpdate()

    assert update.model_dump(exclude_unset=True) == {}


def test_public_schema_has_no_secret_fields() -> None:
    assert not _SECRET_FIELDS & set(RemoteServer.model_fields)


def test_internal_to_public_strips_confidential_fields() -> None:
    record = _internal_record()

    public = record.to_public()

    assert not _SECRET_FIELDS & set(public.model_dump())
    dumped_json = public.model_dump(mode="json")
    serialized = str(dumped_json)
    assert "ciphertext-secret" not in serialized
    assert "ciphertext-passphrase" not in serialized
    assert "ssh-ed25519" not in serialized
    assert dumped_json["id"] == str(record.id)
    assert dumped_json["name"] == record.name


def test_internal_record_still_serializes_secret_fields_for_repository_round_trip() -> None:
    # The internal record is what the repository/mapper persists; it must retain
    # ciphertext so it can round-trip through the database, unlike the public schema.
    record = _internal_record()

    dumped = record.model_dump()

    assert dumped["ssh_secret_encrypted"] == "ciphertext-secret"
    assert dumped["ssh_key_passphrase_encrypted"] == "ciphertext-passphrase"
    assert dumped["host_key"] == "ssh-ed25519 AAAA..."
