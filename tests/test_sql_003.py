from pathlib import Path


def test_migration_003_scopes_index_sets_and_replaces_three_arg_rpcs():
    sql = Path("sql/003_experiment_index_sets.sql").read_text(encoding="utf-8").lower()
    assert "add column if not exists index_set_id" in sql
    assert "documents_collection_index_sha_chunk_uidx" in sql
    assert "drop function if exists public.match_documents(public.vector(1536), integer, text[])" in sql
    assert "filter_index_set_id text default null" in sql
    assert "d.index_set_id = filter_index_set_id" in sql
    assert "d.collection = any (filter_collections)" in sql
    assert "to service_role" in sql
