-- Retrieval Lab migration 002 — JWT RBAC and collection-scoped retrieval
--
-- Non-destructive to document content: no DELETE/TRUNCATE on documents rows.
-- Drops only the old duplicate index name and old retrieval function signatures.
-- Backfills documents.collection = 'legacy' where needed.
--
-- Apply manually in Supabase SQL Editor. Do not commit secrets.

set search_path to public, extensions;

-- 1. Collection column on documents
alter table public.documents
  add column if not exists collection text;

update public.documents
set collection = 'legacy'
where collection is null;

alter table public.documents
  alter column collection set default 'legacy';

alter table public.documents
  alter column collection set not null;

create index if not exists documents_collection_idx on public.documents (collection);

-- 2. Per-collection deduplication (replaces global hash index from 001)
drop index if exists public.documents_content_sha256_chunk_uidx;

create unique index if not exists documents_collection_sha256_chunk_uidx
  on public.documents (collection, (metadata->>'content_sha256'), chunk_index)
  where metadata->>'content_sha256' is not null;

-- 3. Authorization tables (application trust store; not client-writable)
create table if not exists public.lab_identities (
  id bigint generated always as identity primary key,
  issuer text not null,
  subject text not null,
  created_at timestamptz not null default now(),
  constraint lab_identities_issuer_subject_uk unique (issuer, subject)
);

create table if not exists public.lab_permissions (
  identity_id bigint not null references public.lab_identities (id) on delete cascade,
  permission text not null,
  constraint lab_permissions_permission_chk
    check (permission in ('documents:read', 'documents:write')),
  primary key (identity_id, permission)
);

create table if not exists public.lab_collection_grants (
  identity_id bigint not null references public.lab_identities (id) on delete cascade,
  collection text not null,
  allow_read boolean not null default false,
  allow_write boolean not null default false,
  primary key (identity_id, collection),
  constraint lab_collection_grants_collection_chk
    check (collection ~ '^[a-z][a-z0-9_]{0,31}$'),
  constraint lab_collection_grants_access_chk
    check (allow_read or allow_write)
);

alter table public.lab_identities enable row level security;
alter table public.lab_permissions enable row level security;
alter table public.lab_collection_grants enable row level security;

revoke all on table public.lab_identities from public, anon, authenticated;
revoke all on table public.lab_permissions from public, anon, authenticated;
revoke all on table public.lab_collection_grants from public, anon, authenticated;

grant select, insert, update, delete on public.lab_identities to service_role;
grant select, insert, update, delete on public.lab_permissions to service_role;
grant select, insert, update, delete on public.lab_collection_grants to service_role;

-- 4. Resolve access for issuer + subject (server-side only)
create or replace function public.resolve_lab_access(
  p_issuer text,
  p_subject text
)
returns jsonb
language sql
stable
security invoker
set search_path = public, extensions
as $$
  select coalesce(
    (
      select jsonb_build_object(
        'permissions',
        coalesce(
          (
            select jsonb_agg(distinct lp.permission order by lp.permission)
            from public.lab_identities li
            join public.lab_permissions lp on lp.identity_id = li.id
            where li.issuer = p_issuer and li.subject = p_subject
          ),
          '[]'::jsonb
        ),
        'read_collections',
        coalesce(
          (
            select jsonb_agg(distinct lcg.collection order by lcg.collection)
            from public.lab_identities li
            join public.lab_collection_grants lcg on lcg.identity_id = li.id
            where li.issuer = p_issuer
              and li.subject = p_subject
              and lcg.allow_read
          ),
          '[]'::jsonb
        ),
        'write_collections',
        coalesce(
          (
            select jsonb_agg(distinct lcg.collection order by lcg.collection)
            from public.lab_identities li
            join public.lab_collection_grants lcg on lcg.identity_id = li.id
            where li.issuer = p_issuer
              and li.subject = p_subject
              and lcg.allow_write
          ),
          '[]'::jsonb
        )
      )
    ),
    jsonb_build_object(
      'permissions', '[]'::jsonb,
      'read_collections', '[]'::jsonb,
      'write_collections', '[]'::jsonb
    )
  );
$$;

revoke all on function public.resolve_lab_access(text, text)
  from public, anon, authenticated;
grant execute on function public.resolve_lab_access(text, text) to service_role;

-- 5. Remove unscoped retrieval RPCs (no collection filter = full corpus exposure)
drop function if exists public.match_documents(public.vector(1536), integer);
drop function if exists public.search_documents_fts(text, integer);

create or replace function public.match_documents(
  query_embedding public.vector(1536),
  match_count integer,
  filter_collections text[]
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
    order by d.embedding <=> query_embedding, d.id
    limit least(coalesce(match_count, 0), 200)
  ) as nearest
  order by nearest.distance, nearest.id;
$$;

create or replace function public.search_documents_fts(
  query_text text,
  match_count integer,
  filter_collections text[]
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

revoke all on function public.match_documents(public.vector(1536), integer, text[])
  from public, anon, authenticated;
revoke all on function public.search_documents_fts(text, integer, text[])
  from public, anon, authenticated;

grant execute on function public.match_documents(public.vector(1536), integer, text[])
  to service_role;
grant execute on function public.search_documents_fts(text, integer, text[])
  to service_role;

notify pgrst, 'reload schema';
