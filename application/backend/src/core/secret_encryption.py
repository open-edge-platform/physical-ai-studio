"""Fernet-based encryption for confidential remote-server SSH secrets.

Secrets (`ssh_secret`, `ssh_key_passphrase`) are encrypted before persistence
and must only be decrypted inside the SSH provisioning boundary — never on a
path that can reach an API response. The encryption key comes solely from the
`REMOTE_SERVER_SECRET_KEY` environment variable (see `Settings.remote_server_secret_key`)
it never lives in the database. Rotating or losing that key makes stored secrets
undecryptable, so every registered server's secret must be re-entered afterward.
"""

from functools import lru_cache

from cryptography.fernet import Fernet, InvalidToken

from exceptions import RemoteServerSecretKeyMissingError, SecretDecryptionError
from settings import Settings, get_settings


@lru_cache
def _build_cipher(secret_key: str) -> Fernet:
    """Build a Fernet cipher for a given key, failing closed on an invalid key."""
    try:
        return Fernet(secret_key.encode("utf-8"))
    except (ValueError, TypeError) as error:
        raise RemoteServerSecretKeyMissingError(
            "REMOTE_SERVER_SECRET_KEY is not a valid Fernet key. Generate one with: "
            'python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"'
        ) from error


def _require_cipher(settings: Settings | None = None) -> Fernet:
    """Return the configured cipher or raise a clear, fail-closed error."""
    active_settings = settings or get_settings()
    if not active_settings.remote_server_secret_key:
        raise RemoteServerSecretKeyMissingError
    return _build_cipher(active_settings.remote_server_secret_key)


def encrypt_secret(plaintext: str, settings: Settings | None = None) -> str:
    """Encrypt a confidential SSH secret before it is persisted.

    :param plaintext: Secret value (private key contents, password, or passphrase).
    :param settings: Optional settings override, defaults to the cached application settings.
    :return: Fernet ciphertext, safe to store at rest.
    """
    cipher = _require_cipher(settings)
    return cipher.encrypt(plaintext.encode("utf-8")).decode("utf-8")


def decrypt_secret(ciphertext: str, settings: Settings | None = None) -> str:
    """Decrypt a confidential SSH secret.

    Callers must stay inside the SSH provisioning boundary: never return this
    value from an API response or log it.

    :param ciphertext: Fernet ciphertext previously produced by :func:`encrypt_secret`.
    :param settings: Optional settings override, defaults to the cached application settings.
    :return: The original plaintext secret.
    """
    cipher = _require_cipher(settings)
    try:
        return cipher.decrypt(ciphertext.encode("utf-8")).decode("utf-8")
    except InvalidToken as error:
        raise SecretDecryptionError from error
