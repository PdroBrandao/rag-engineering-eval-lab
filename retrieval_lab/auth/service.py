from retrieval_lab.auth.models import AccessContext, Principal
from retrieval_lab.db import DocumentStore


def resolve_access(store: DocumentStore, principal: Principal) -> AccessContext:
    raw = store.resolve_lab_access(principal.issuer, principal.subject)
    permissions = frozenset(str(item) for item in raw.get("permissions", []))
    read_collections = frozenset(str(item) for item in raw.get("read_collections", []))
    write_collections = frozenset(str(item) for item in raw.get("write_collections", []))
    return AccessContext(
        principal=principal,
        permissions=permissions,
        read_collections=read_collections,
        write_collections=write_collections,
    )
