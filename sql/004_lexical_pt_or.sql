-- Retrieval Lab migration 004 — Portuguese OR lexical search.
--
-- Adds documents.fts_pt and search_documents_fts_pt_or.
-- Does not drop or replace search_documents_fts. The English AND query
-- remains available for the original experiment.
--
-- Apply manually in the Supabase SQL editor after 003.
-- Adding the generated column rewrites existing rows. On a large table
-- this can take a few minutes.

set search_path to public, extensions;

alter table public.documents
  add column if not exists fts_pt tsvector
  generated always as (
    pg_catalog.to_tsvector('portuguese', coalesce(content, ''))
  ) stored;

create index if not exists documents_fts_pt_gin
  on public.documents using gin (fts_pt);

create or replace function public.portuguese_or_tsquery(query_text text)
returns tsquery
language plpgsql
immutable
as $$
declare
  acc tsquery := null;
  lex text;
begin
  -- to_tsvector('portuguese') drops Portuguese stopwords and stems the rest.
  -- The lexemes are OR-ed. No dynamic SQL is built from the user text.
  for lex in
    select stemmed
    from ts_debug('portuguese'::regconfig, coalesce(query_text, '')) as parsed
    cross join lateral unnest(parsed.lexemes) as stemmed
    where stemmed is not null
      and btrim(stemmed) <> ''
    group by stemmed
    order by stemmed
  loop
    if acc is null then
      acc := pg_catalog.plainto_tsquery('portuguese', lex);
    else
      acc := acc || pg_catalog.plainto_tsquery('portuguese', lex);
    end if;
  end loop;
  return acc;
end;
$$;

create or replace function public.search_documents_fts_pt_or(
  query_text text,
  match_count integer,
  filter_collections text[],
  filter_index_set_id text default null
)
returns table (
  id bigint,
  content text,
  document_name text,
  page integer,
  metadata jsonb,
  collection text,
  lexical_score double precision,
  result_rank integer
)
language sql
stable
security invoker
set search_path = public, extensions
as $$
  with q as (
    select public.portuguese_or_tsquery(query_text) as tsq
  ),
  matched as (
    select
      d.id,
      d.content,
      d.document_name,
      d.page,
      d.metadata,
      d.collection,
      pg_catalog.ts_rank(d.fts_pt, q.tsq)::double precision as lexical_score
    from public.documents d
    cross join q
    where q.tsq is not null
      and coalesce(match_count, 0) >= 1
      and coalesce(cardinality(filter_collections), 0) > 0
      and d.collection = any (filter_collections)
      and (
        filter_index_set_id is null
        or btrim(filter_index_set_id) = ''
        or d.index_set_id = filter_index_set_id
      )
      and d.fts_pt @@ q.tsq
  )
  select
    matched.id,
    matched.content,
    matched.document_name,
    matched.page,
    matched.metadata,
    matched.collection,
    matched.lexical_score,
    row_number() over (
      order by matched.lexical_score desc, matched.id
    )::integer as result_rank
  from matched
  order by matched.lexical_score desc, matched.id
  limit least(coalesce(match_count, 0), 200);
$$;

revoke all on function public.portuguese_or_tsquery(text)
  from public, anon, authenticated;
revoke all on function public.search_documents_fts_pt_or(text, integer, text[], text)
  from public, anon, authenticated;

grant execute on function public.portuguese_or_tsquery(text) to service_role;
grant execute on function public.search_documents_fts_pt_or(text, integer, text[], text)
  to service_role;

notify pgrst, 'reload schema';
