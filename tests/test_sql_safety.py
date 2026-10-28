from pathlib import Path


def _executable_sql() -> str:
    lines = []
    for line in Path("sql/001_retrieval.sql").read_text(encoding="utf-8").splitlines():
        if "--" in line:
            line = line[: line.index("--")]
        lines.append(line)
    return "\n".join(lines).lower()


def test_migration_does_not_destroy_data_or_weaken_rls():
    sql = _executable_sql()
    assert "delete from" not in sql
    assert "truncate" not in sql
    assert "drop table" not in sql
    assert "drop index" not in sql
    assert "drop policy" not in sql
    assert "disable row level security" not in sql
    assert "grant execute on function public.match_documents" in sql
    assert "to service_role" in sql
    assert "from public, anon, authenticated" in sql
    assert "opc.opcname = 'vector_cosine_ops'" in sql
