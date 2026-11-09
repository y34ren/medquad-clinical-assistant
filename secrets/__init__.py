"""NIH MedQuAD Secret Management Package (with Python stdlib `secrets` compatibility)."""

from __future__ import annotations

import hmac
import os
from random import SystemRandom

# Standard library `secrets` compatibility exports so `import secrets` inside
# `tempfile`, `uuid`, or third-party modules works seamlessly alongside
# `from secrets.secret_manager import SecretManagerClient`.
_sysrand = SystemRandom()
randbits = _sysrand.getrandbits
choice = _sysrand.choice
compare_digest = hmac.compare_digest
DEFAULT_ENTROPY = 32


def randbelow(exclusive_upper_bound: int) -> int:
    if exclusive_upper_bound <= 0:
        raise ValueError("Upper bound must be positive.")
    return _sysrand._randbelow(exclusive_upper_bound)


def token_bytes(nbytes: int | None = None) -> bytes:
    if nbytes is None:
        nbytes = DEFAULT_ENTROPY
    return os.urandom(nbytes)


def token_hex(nbytes: int | None = None) -> str:
    return token_bytes(nbytes).hex()


def token_urlsafe(nbytes: int | None = None) -> str:
    import base64
    tok = token_bytes(nbytes)
    return base64.urlsafe_b64encode(tok).rstrip(b"=").decode("ascii")


from secrets.secret_manager import SecretManagerClient, default_secret_client

__all__ = [
    "SecretManagerClient",
    "default_secret_client",
    "SystemRandom",
    "randbits",
    "randbelow",
    "choice",
    "compare_digest",
    "token_bytes",
    "token_hex",
    "token_urlsafe",
    "DEFAULT_ENTROPY",
]
