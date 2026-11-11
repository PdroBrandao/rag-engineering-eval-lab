-- 1. Cadastrar identidade
insert into public.lab_identities (issuer, subject)
values ('http://localhost:8000', 'pedro-local')
on conflict (issuer, subject) do nothing;

-- 2. Permitir leitura e escrita
insert into public.lab_permissions (identity_id, permission)
select id, permission
from public.lab_identities
cross join (values ('documents:read'), ('documents:write')) as p(permission)
where issuer = 'http://localhost:8000'
  and subject = 'pedro-local'
on conflict do nothing;

-- 3. Autorizar a coleção legacy
insert into public.lab_collection_grants
(identity_id, collection, allow_read, allow_write)
select id, 'legacy', true, true
from public.lab_identities
where issuer = 'http://localhost:8000'
  and subject = 'pedro-local'
on conflict (identity_id, collection)
do update set allow_read = true, allow_write = true;