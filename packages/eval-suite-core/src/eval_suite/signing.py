"""Manifest signing — Ed25519 submitter attestation.

Uses Ed25519 (curve25519-edwards) from `cryptography` because it's:
- Built into a widely-trusted library (no rolling our own crypto).
- Small (32-byte keys, 64-byte signatures, well under JSON size limits).
- Fast enough that signing/verifying a manifest is microseconds.
- Standard in the Sigstore world we want to grow into for v1.0.

API:
- `generate_keypair()` → (private_bytes, public_bytes) for development /
  testing. In v1.0 keys come from Sigstore or a customer's HSM.
- `sign(payload: str, private_bytes: bytes) → str` — produces a hex
  Ed25519 signature over the canonical-JSON bytes the manifest hashes.
- `verify(payload: str, signature_hex: str, public_bytes: bytes) → bool` —
  returns True iff the signature is valid for the payload + public key.

Manifest integration is in `eval_suite/manifest.py`:
- `Manifest.submitter_signature: str | None` — hex signature.
- `Manifest.submitter_public_key: str | None` — hex public key for verification.
- `Manifest.verify()` returns True iff (a) the content hash matches AND
  (b) any submitter_signature present is valid for the public key.
"""

from __future__ import annotations

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    PublicFormat,
)


def generate_keypair() -> tuple[bytes, bytes]:
    """Returns (private_bytes, public_bytes) as raw Ed25519 byte sequences.

    Use only for development / contract tests. Production keys come from
    Sigstore / customer HSMs.
    """
    private = Ed25519PrivateKey.generate()
    private_bytes = private.private_bytes(
        encoding=Encoding.Raw,
        format=PrivateFormat.Raw,
        encryption_algorithm=NoEncryption(),
    )
    public_bytes = private.public_key().public_bytes(
        encoding=Encoding.Raw,
        format=PublicFormat.Raw,
    )
    return private_bytes, public_bytes


def sign(payload: str, private_bytes: bytes) -> str:
    """Ed25519-sign `payload` (UTF-8) with the raw private key bytes.

    Returns the signature as lowercase hex. Caller is responsible for
    canonicalizing the payload before passing — Manifest.sign() calls
    this with the same canonical-JSON bytes used for the content hash,
    so a verifier never has to guess about whitespace.
    """
    pk = Ed25519PrivateKey.from_private_bytes(private_bytes)
    sig = pk.sign(payload.encode("utf-8"))
    return sig.hex()


def verify(payload: str, signature_hex: str, public_bytes: bytes) -> bool:
    """True iff the Ed25519 signature is valid for the payload + public key.

    Returns False on any failure (invalid signature, malformed inputs).
    Never raises; this is the API surface for `Manifest.verify()` which
    must remain exception-clean.
    """
    try:
        pub = Ed25519PublicKey.from_public_bytes(public_bytes)
        sig = bytes.fromhex(signature_hex)
        pub.verify(sig, payload.encode("utf-8"))
        return True
    except (InvalidSignature, ValueError):
        return False


def to_hex(b: bytes) -> str:
    return b.hex()


def from_hex(s: str) -> bytes:
    return bytes.fromhex(s)
