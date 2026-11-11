import logging
from typing import Any

import jwt
from jwt import PyJWKClient

from retrieval_lab.auth.models import Principal
from retrieval_lab.config import JwtSettings
from retrieval_lab.errors import AuthenticationError

logger = logging.getLogger(__name__)


class JwtValidator:
    def __init__(self, settings: JwtSettings) -> None:
        self._settings = settings
        self._jwks_client = PyJWKClient(settings.jwks_url, cache_keys=True)

    def validate(self, token: str) -> Principal:
        try:
            signing_key = self._jwks_client.get_signing_key_from_jwt(token)
            payload = jwt.decode(
                token,
                signing_key.key,
                algorithms=list(self._settings.algorithms),
                audience=self._settings.audience,
                issuer=self._settings.issuer,
                leeway=self._settings.leeway_seconds,
                options={"require": ["exp", "sub", "iss"]},
            )
        except jwt.PyJWTError as exc:
            logger.info("jwt validation failed: %s", exc.__class__.__name__)
            raise AuthenticationError() from exc
        return _principal_from_payload(payload)


class LocalJwtValidator:
    """Validates RS256 tokens against a fixed PEM public key (tests / local dev)."""

    def __init__(self, settings: JwtSettings, public_key_pem: str) -> None:
        self._settings = settings
        self._public_key_pem = public_key_pem

    def validate(self, token: str) -> Principal:
        try:
            payload = jwt.decode(
                token,
                self._public_key_pem,
                algorithms=list(self._settings.algorithms),
                audience=self._settings.audience,
                issuer=self._settings.issuer,
                leeway=self._settings.leeway_seconds,
                options={"require": ["exp", "sub", "iss"]},
            )
        except jwt.PyJWTError as exc:
            logger.info("jwt validation failed: %s", exc.__class__.__name__)
            raise AuthenticationError() from exc
        return _principal_from_payload(payload)


def _principal_from_payload(payload: dict[str, Any]) -> Principal:
    issuer = payload.get("iss")
    subject = payload.get("sub")
    if not isinstance(issuer, str) or not issuer.strip():
        raise AuthenticationError("Token issuer is missing.")
    if not isinstance(subject, str) or not subject.strip():
        raise AuthenticationError("Token subject is missing.")
    return Principal(issuer=issuer, subject=subject)
