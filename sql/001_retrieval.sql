-- Retrieval Lab migration 001
--
-- Read this before running it. The script does not DELETE, UPDATE, TRUNCATE,
-- or DROP existing rows, policies, or indexes. It does not disable RLS and it
-- does not grant access to anon or authenticated.
--
-- It may:
--   * add a partial unique index on (metadata content hash, chunk_index)
--   * add an English tsvector trigger only when fts is not a generated column
--   * add a GIN index only when no GIN index already mentions fts
--   * add an HNSW cosine index only when no HNSW index uses vector_cosine_ops
--   * create or replace match_documents and search_documents_fts
--   * revoke EXECUTE on those functions from PUBLIC, anon, and authenticated
--   * grant EXECUTE on those functions to service_role
--   * ask PostgREST to reload its schema cache
--
-- Live column type confirmed before writing this file: public.vector(1536).
-- Operators resolve through search_path public, extensions.

set search_path to public, extensions;

create unique index if not exists documents_content_sha256_chunk_uidx
  on public.documents ((metadata->>'content_sha256'), chunk_index)
  where metadata->>'content_sha256' is not null;

do $$
declare
  fts_generated "char";
  has_gin boolean;
  has_hnsw boolean;
  has_fts_trigger boolean;
begin
  select a.attgenerated
    into fts_generated
  from pg_attribute a
  join pg_class c on c.oid = a.attrelid
  join pg_namespace n on n.oid = c.relnamespace
  where n.nspname = 'public'
    and c.relname = 'documents'
    and a.attname = 'fts'
    and not a.attisdropped;

  if fts_generated is null then
    raise exception 'public.documents.fts does not exist';
  end if;

  if not exists (
    select 1
    from pg_type t
    join pg_namespace n on n.oid = t.typnamespace
    where n.nspname = 'public'
      and t.typname = 'vector'
  ) then
    raise exception 'public.vector does not exist';
  end if;

  if fts_generated = '' then
    execute $fn$
      create or replace function public.documents_fts_before_write()
      returns trigger
      language plpgsql
      security invoker
      set search_path = public, extensions
      as $body$
      begin
        new.fts := pg_catalog.to_tsvector('english', coalesce(new.content, ''));
        return new;
      end;
      $body$;
    $fn$;

    select exists (
      select 1
      from pg_trigger t
      where t.tgrelid = 'public.documents'::regclass
        and t.tgname = 'documents_fts_before_write'
        and not t.tgisinternal
    ) into has_fts_trigger;

    if not has_fts_trigger then
      execute $trg$
        create trigger documents_fts_before_write
        before insert or update of content on public.documents
        for each row
        execute function public.documents_fts_before_write()
      $trg$;
    end if;

    execute 'revoke all on function public.documents_fts_before_write() from PUBLIC, anon, authenticated';
  end if;

  select exists (
    select 1
    from pg_index i
    join pg_class t on t.oid = i.indrelid
    join pg_class ix on ix.oid = i.indexrelid
    join pg_namespace n on n.oid = t.relnamespace
    join pg_am am on am.oid = ix.relam
    where n.nspname = 'public'
      and t.relname = 'documents'
      and am.amname = 'gin'
      and pg_get_indexdef(i.indexrelid) ilike '%fts%'
  ) into has_gin;

  if not has_gin then
    execute 'create index documents_fts_gin on public.documents using gin (fts)';
  end if;

  select exists (
    select 1
    from pg_index i
    join pg_class t on t.oid = i.indrelid
    join pg_class ix on ix.oid = i.indexrelid
    join pg_namespace n on n.oid = t.relnamespace
    join pg_am am on am.oid = ix.relam
    join pg_opclass opc on opc.oid = any (i.indclass)
    where n.nspname = 'public'
      and t.relname = 'documents'
      and am.amname = 'hnsw'
      and opc.opcname = 'vector_cosine_ops'
  ) into has_hnsw;

  if not has_hnsw then
    execute 'create index documents_embedding_hnsw_cosine on public.documents using hnsw (embedding vector_cosine_ops)';
  end if;
end $$;

create or replace function public.match_documents(
  query_embedding public.vector(1536),
  match_count integer
)
returns table (
  id bigint,
  content text,
  document_name text,
  page integer,
  metadata jsonb,
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
      (d.embedding <=> query_embedding) as distance,
      (1 - (d.embedding <=> query_embedding))::double precision as similarity
    from public.documents d
    where d.embedding is not null
      and coalesce(match_count, 0) >= 1
    order by d.embedding <=> query_embedding, d.id
    limit least(coalesce(match_count, 0), 200)
  ) as nearest
  order by nearest.distance, nearest.id;
$$;

create or replace function public.search_documents_fts(
  query_text text,
  match_count integer
)
returns table (
  id bigint,
  content text,
  document_name text,
  page integer,
  metadata jsonb,
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
      pg_catalog.ts_rank(d.fts, q.tsq)::double precision as lexical_score
    from public.documents d
    cross join q
    where coalesce(match_count, 0) >= 1
      and d.fts @@ q.tsq
  )
  select
    matched.id,
    matched.content,
    matched.document_name,
    matched.page,
    matched.metadata,
    matched.lexical_score,
    row_number() over (
      order by matched.lexical_score desc, matched.id
    )::integer as result_rank
  from matched
  order by matched.lexical_score desc, matched.id
  limit least(coalesce(match_count, 0), 200);
$$;

revoke all on function public.match_documents(public.vector(1536), integer)
  from PUBLIC, anon, authenticated;
revoke all on function public.search_documents_fts(text, integer)
  from PUBLIC, anon, authenticated;

grant execute on function public.match_documents(public.vector(1536), integer)
  to service_role;
grant execute on function public.search_documents_fts(text, integer)
  to service_role;

notify pgrst, 'reload schema';
