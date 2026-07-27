import pytest
from cryptography.fernet import Fernet

from core import secret_encryption
from exceptions import RemoteServerSecretKeyMissingError, SecretDecryptionError
from settings import Settings


def _settings_with_key(key: str | None) -> Settings:
    return Settings(REMOTE_SERVER_SECRET_KEY=key)


@pytest.fixture(autouse=True)
def _clear_cipher_cache():
    """Avoid leaking a cached Fernet instance across tests using different keys."""
    secret_encryption._build_cipher.cache_clear()
    yield
    secret_encryption._build_cipher.cache_clear()


def test_encrypt_then_decrypt_round_trips() -> None:
    settings = _settings_with_key(Fernet.generate_key().decode())

    ciphertext = secret_encryption.encrypt_secret("hunter2", settings=settings)

    assert ciphertext != "hunter2"
    assert secret_encryption.decrypt_secret(ciphertext, settings=settings) == "hunter2"


def test_encrypt_without_configured_key_raises() -> None:
    settings = _settings_with_key(None)

    with pytest.raises(RemoteServerSecretKeyMissingError):
        secret_encryption.encrypt_secret("hunter2", settings=settings)


def test_encrypt_with_malformed_key_raises() -> None:
    settings = _settings_with_key("not-a-valid-fernet-key")

    with pytest.raises(RemoteServerSecretKeyMissingError):
        secret_encryption.encrypt_secret("hunter2", settings=settings)


def test_decrypt_with_wrong_key_raises_decryption_error() -> None:
    encrypt_settings = _settings_with_key(Fernet.generate_key().decode())
    ciphertext = secret_encryption.encrypt_secret("hunter2", settings=encrypt_settings)

    decrypt_settings = _settings_with_key(Fernet.generate_key().decode())
    with pytest.raises(SecretDecryptionError):
        secret_encryption.decrypt_secret(ciphertext, settings=decrypt_settings)


def test_decrypt_tampered_ciphertext_raises_decryption_error() -> None:
    settings = _settings_with_key(Fernet.generate_key().decode())
    ciphertext = secret_encryption.encrypt_secret("hunter2", settings=settings)

    with pytest.raises(SecretDecryptionError):
        secret_encryption.decrypt_secret(ciphertext[:-2] + "xx", settings=settings)


def test_uses_cached_settings_when_none_provided(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings_with_key(Fernet.generate_key().decode())
    monkeypatch.setattr(secret_encryption, "get_settings", lambda: settings)

    ciphertext = secret_encryption.encrypt_secret("hunter2")

    assert secret_encryption.decrypt_secret(ciphertext) == "hunter2"
