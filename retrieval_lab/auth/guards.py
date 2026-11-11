from retrieval_lab.auth.models import AccessContext, validate_collection_name
from retrieval_lab.errors import AuthorizationError, RetrievalError

READ_PERMISSION = "documents:read"
WRITE_PERMISSION = "documents:write"


def require_read_permission(access: AccessContext) -> None:
    if not access.has_permission(READ_PERMISSION):
        raise AuthorizationError("documents:read permission is required.")


def require_write_permission(access: AccessContext) -> None:
    if not access.has_permission(WRITE_PERMISSION):
        raise AuthorizationError("documents:write permission is required.")


def require_write_collection(access: AccessContext, collection: str) -> str:
    require_write_permission(access)
    try:
        normalized = validate_collection_name(collection)
    except ValueError as exc:
        raise RetrievalError(422, str(exc)) from exc
    if normalized not in access.write_collections:
        raise AuthorizationError(f"Write access to collection '{normalized}' is denied.")
    return normalized


def resolve_query_collections(
    access: AccessContext,
    requested: list[str] | None,
) -> list[str]:
    require_read_permission(access)
    allowed = sorted(access.read_collections)
    if not allowed:
        return []
    if not requested:
        return allowed
    normalized: list[str] = []
    for raw in requested:
        try:
            name = validate_collection_name(raw)
        except ValueError as exc:
            raise RetrievalError(422, str(exc)) from exc
        if name not in access.read_collections:
            raise AuthorizationError(f"Read access to collection '{name}' is denied.")
        normalized.append(name)
    return sorted(set(normalized))
