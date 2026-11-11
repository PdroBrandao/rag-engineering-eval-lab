import re
from dataclasses import dataclass

COLLECTION_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")


def validate_collection_name(collection: str) -> str:
    cleaned = collection.strip()
    if not COLLECTION_NAME_RE.fullmatch(cleaned):
        raise ValueError(
            "collection must match ^[a-z][a-z0-9_]{0,31}$"
        )
    return cleaned


@dataclass(frozen=True)
class Principal:
    issuer: str
    subject: str


@dataclass(frozen=True)
class AccessContext:
    principal: Principal
    permissions: frozenset[str]
    read_collections: frozenset[str]
    write_collections: frozenset[str]

    def has_permission(self, permission: str) -> bool:
        return permission in self.permissions
