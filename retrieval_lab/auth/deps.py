from typing import Annotated, Protocol

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from retrieval_lab.auth.models import Principal
from retrieval_lab.auth.service import resolve_access
from retrieval_lab.db import DocumentStore
from retrieval_lab.errors import AuthenticationError

bearer_scheme = HTTPBearer(auto_error=False)


class JwtValidatorProtocol(Protocol):
    def validate(self, token: str) -> Principal: ...


def get_jwt_validator(request: Request) -> JwtValidatorProtocol:
    validator = getattr(request.app.state, "jwt_validator", None)
    if validator is not None:
        return validator
    factory = getattr(request.app.state, "_jwt_validator_factory", None)
    if factory is not None:
        validator = factory()
        request.app.state.jwt_validator = validator
        return validator
    raise RuntimeError("JWT validator is not configured.")


def get_document_store(request: Request) -> DocumentStore:
    store = getattr(request.app.state, "store", None)

    if store is None:
        factory = getattr(request.app.state, "_store_factory", None)
        if factory is None:
            raise RuntimeError("Document store is not configured.")

        store = factory()
        request.app.state.store = store

    return store


def get_current_principal(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
    validator: Annotated[JwtValidatorProtocol, Depends(get_jwt_validator)],
) -> Principal:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise AuthenticationError()
    if not credentials.credentials:
        raise AuthenticationError()
    return validator.validate(credentials.credentials)


def get_access_context(
    principal: Annotated[Principal, Depends(get_current_principal)],
    store: Annotated[DocumentStore, Depends(get_document_store)],
):
    return resolve_access(store, principal)
