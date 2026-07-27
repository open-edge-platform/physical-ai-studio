from uuid import uuid4

from db.schema import RemoteServerDB
from repositories.mappers.remote_server_mapper import RemoteServerMapper
from schemas.hardware import DeviceType
from schemas.remote_server import LastCheckSummary, RemoteServerInternal, SSHAuthType


def _internal_record() -> RemoteServerInternal:
    return RemoteServerInternal(
        id=uuid4(),
        name="gpu-box",
        host="10.0.0.5",
        port=2222,
        username="trainer",
        auth_type=SSHAuthType.KEY,
        device_type=DeviceType.XPU,
        last_check=LastCheckSummary(status="healthy", latency_ms=42, reason_code=None),
        ssh_secret_encrypted="ciphertext-secret",
        ssh_key_passphrase_encrypted="ciphertext-passphrase",
        host_key="ssh-ed25519 AAAA...",
    )


def test_to_schema_persists_confidential_fields_as_provided() -> None:
    record = _internal_record()

    db_row = RemoteServerMapper.to_schema(record)

    assert db_row.id == str(record.id)
    assert db_row.host == "10.0.0.5"
    assert db_row.port == 2222
    assert db_row.auth_type == SSHAuthType.KEY
    assert db_row.device_type == DeviceType.XPU
    assert db_row.ssh_secret_encrypted == "ciphertext-secret"
    assert db_row.ssh_key_passphrase_encrypted == "ciphertext-passphrase"
    assert db_row.host_key == "ssh-ed25519 AAAA..."
    assert db_row.last_check_status == "healthy"
    assert db_row.last_check_latency_ms == 42


def test_round_trip_preserves_all_fields() -> None:
    record = _internal_record()

    db_row = RemoteServerMapper.to_schema(record)
    roundtripped = RemoteServerMapper.from_schema(db_row)

    assert roundtripped.id == record.id
    assert roundtripped.ssh_secret_encrypted == record.ssh_secret_encrypted
    assert roundtripped.ssh_key_passphrase_encrypted == record.ssh_key_passphrase_encrypted
    assert roundtripped.host_key == record.host_key
    assert roundtripped.last_check.status == "healthy"
    assert roundtripped.last_check.latency_ms == 42


def test_from_schema_handles_unset_last_check() -> None:
    db_row = RemoteServerDB(
        id=str(uuid4()),
        name="gpu-box",
        host="10.0.0.5",
        port=22,
        username="trainer",
        auth_type=SSHAuthType.PASSWORD,
        device_type=DeviceType.CUDA,
        ssh_secret_encrypted="ciphertext",
        ssh_key_passphrase_encrypted=None,
        host_key=None,
        last_check_status=None,
        last_check_at=None,
        last_check_latency_ms=None,
        last_check_reason_code=None,
    )

    record = RemoteServerMapper.from_schema(db_row)

    assert record.last_check.status is None
    assert record.host_key is None
    assert record.ssh_key_passphrase_encrypted is None
