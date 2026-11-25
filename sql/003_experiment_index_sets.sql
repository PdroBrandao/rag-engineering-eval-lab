-- Retrieval Lab migration 003 — index sets for chunk-size experiments.
-- Apply manually after 002. Does not delete document rows.
--
-- The 3-argument RPCs are replaced by 4-argument functions. The new argument
-- defaults to null, so callers that omit it keep the previous collection scope.

set search_path to public, extensions;

alter table public.documents
  add column if not exists index_set_id text;

update public.documents
set index_set_id = 'unscoped'
where index_set_id is null;

alter table public.documents
  alter column index_set_id set default 'unscoped';

alter table public.documents
  alter column index_set_id set not null;

create index if not exists documents_collection_index_set_idx
  on public.documents (collection, index_set_id);

drop index if exists public.documents_collection_sha256_chunk_uidx;

create unique index if not exists documents_collection_index_sha_chunk_uidx
  on public.documents (collection, index_set_id, (metadata->>'content_sha256'), chunk_index)
  where metadata->>'content_sha256' is not null;

drop function if exists public.match_documents(public.vector(1536), integer, text[]);
drop function if exists public.search_documents_fts(text, integer, text[]);

create or replace function public.match_documents(
  query_embedding public.vector(1536),
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
  similarity double precision,
  result_rank integer
)
language sql
stable
security invoker
set search_path = public, extensions
as $$
  select
    nearest.id,
    nearest.content,
    nearest.document_name,
    nearest.page,
    nearest.metadata,
    nearest.collection,
    nearest.similarity,
    row_number() over (
      order by nearest.distance, nearest.id
    )::integer as result_rank
  from (
    select
      d.id,
      d.content,
      d.document_name,
      d.page,
      d.metadata,
      d.collection,
      (d.embedding <=> query_embedding) as distance,
      (1 - (d.embedding <=> query_embedding))::double precision as similarity
    from public.documents d
    where d.embedding is not null
      and coalesce(match_count, 0) >= 1
      and coalesce(cardinality(filter_collections), 0) > 0
      and d.collection = any (filter_collections)
      and (
        filter_index_set_id is null
        or btrim(filter_index_set_id) = ''
        or d.index_set_id = filter_index_set_id
      )
    order by d.embedding <=> query_embedding, d.id
    limit least(coalesce(match_count, 0), 200)
  ) as nearest
  order by nearest.distance, nearest.id;
$$;

create or replace function public.search_documents_fts(
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
    select pg_catalog.websearch_to_tsquery('english', coalesce(query_text, '')) as tsq
  ),
  matched as (
    select
      d.id,
      d.content,
      d.document_name,
      d.page,
      d.metadata,
      d.collection,
      pg_catalog.ts_rank(d.fts, q.tsq)::double precision as lexical_score
    from public.documents d
    cross join q
    where coalesce(match_count, 0) >= 1
      and coalesce(cardinality(filter_collections), 0) > 0
      and d.collection = any (filter_collections)
      and (
        filter_index_set_id is null
        or btrim(filter_index_set_id) = ''
        or d.index_set_id = filter_index_set_id
      )
      and d.fts @@ q.tsq
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

revoke all on function public.match_documents(public.vector(1536), integer, text[], text)
  from public, anon, authenticated;
revoke all on function public.search_documents_fts(text, integer, text[], text)
  from public, anon, authenticated;

grant execute on function public.match_documents(public.vector(1536), integer, text[], text)
  to service_role;
grant execute on function public.search_documents_fts(text, integer, text[], text)
  to service_role;

notify pgrst, 'reload schema';
