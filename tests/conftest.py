from __future__ import annotations

import datetime as dt
import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from retrieval_lab.auth.jwt_validator import LocalJwtValidator
from retrieval_lab.config import JwtSettings, Settings

TEST_ISSUER = "https://lab.example.test"
TEST_AUDIENCE = "retrieval-lab"


@pytest.fixture(scope="session")
def jwt_keypair():
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_pem = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return private_pem.decode("utf-8"), public_pem.decode("utf-8")


@pytest.fixture(scope="session")
def jwt_settings() -> JwtSettings:
    return JwtSettings(
        issuer=TEST_ISSUER,
        audience=TEST_AUDIENCE,
        jwks_url="https://lab.example.test/jwks.json",
    )


@pytest.fixture(scope="session")
def jwt_validator(jwt_settings, jwt_keypair):
    _private, public_pem = jwt_keypair
    return LocalJwtValidator(jwt_settings, public_pem)


def mint_access_token(
    private_pem: str,
    *,
    subject: str,
    issuer: str = TEST_ISSUER,
    audience: str = TEST_AUDIENCE,
    expired: bool = False,
) -> str:
    now = dt.datetime.now(dt.timezone.utc)
    exp = now - dt.timedelta(minutes=5) if expired else now + dt.timedelta(hours=1)
    payload = {
        "iss": issuer,
        "sub": subject,
        "aud": audience,
        "exp": exp,
        "iat": now,
    }
    return jwt.encode(payload, private_pem, algorithm="RS256")


@pytest.fixture
def mint_token(jwt_keypair):
    private_pem, _public = jwt_keypair

    def _factory(**kwargs):
        return mint_access_token(private_pem, **kwargs)

    return _factory


@pytest.fixture
def test_settings(jwt_settings) -> Settings:
    return Settings(
        openai_api_key="test-key",
        supabase_url="https://example.supabase.co",
        supabase_key="test-supabase-key",
        jwt=jwt_settings,
    )


@pytest.fixture
def auth_header(mint_token):
    def _header(subject: str = "admin-all", **kwargs) -> dict[str, str]:
        token = mint_token(subject=subject, **kwargs)
        return {"Authorization": f"Bearer {token}"}

    return _header


@pytest.fixture
def full_access_store():
    from tests.fakes import MemoryStore

    store = MemoryStore()
    store.set_access(
        TEST_ISSUER,
        "admin-all",
        permissions=["documents:read", "documents:write"],
        read_collections=["legacy", "finance", "legal", "engineering"],
        write_collections=["legacy", "finance", "legal", "engineering"],
    )
    return store
