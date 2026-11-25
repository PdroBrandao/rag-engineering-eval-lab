from pathlib import Path


def test_migration_004_adds_portuguese_or_without_replacing_the_english_rpc():
    sql = Path("sql/004_lexical_pt_or.sql").read_text(encoding="utf-8").lower()
    previous = Path("sql/003_experiment_index_sets.sql").read_text(encoding="utf-8").lower()
    assert "fts_pt" in sql
    assert "to_tsvector('portuguese'" in sql
    assert "plainto_tsquery('portuguese'" in sql
    assert "ts_debug('portuguese'::regconfig" in sql
    assert "alias text" not in sql
    assert "lexemes text[]" not in sql
    assert "portuguese_or_tsquery" in sql
    assert "search_documents_fts_pt_or" in sql
    assert "ts_rank" in sql
    assert "order by matched.lexical_score desc, matched.id" in sql
    assert "documents_fts_pt_gin" in sql
    assert "using gin (fts_pt)" in sql
    assert "d.collection = any (filter_collections)" in sql
    assert "d.index_set_id = filter_index_set_id" in sql
    assert "grant execute" in sql
    assert "to service_role" in sql
    assert "execute format" not in sql
    assert "execute '" not in sql
    assert "drop function" not in sql
    assert "search_documents_fts(" in previous
    assert "websearch_to_tsquery('english'" in previous
