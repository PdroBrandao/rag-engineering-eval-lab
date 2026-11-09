from pathlib import Path


def _sql_text(name: str) -> str:
    return Path("sql") / name


def test_migration_002_revokes_unscoped_rpcs_and_protects_auth_tables():
    sql = _sql_text("002_auth_rbac.sql").read_text(encoding="utf-8").lower()
    assert "drop function if exists public.match_documents(public.vector(1536), integer)" in sql
    assert "drop function if exists public.search_documents_fts(text, integer)" in sql
    assert "filter_collections text[]" in sql
    assert "d.collection = any (filter_collections)" in sql
    assert "revoke all on table public.lab_identities from public, anon, authenticated" in sql
    assert "enable row level security" in sql
    assert "grant execute on function public.match_documents(public.vector(1536), integer, text[])" in sql
    assert "to service_role" in sql
