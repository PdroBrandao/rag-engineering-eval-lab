-- 1. Register an identity
insert into public.lab_identities (issuer, subject)
values ('http://localhost:8000', 'local-dev')
on conflict (issuer, subject) do nothing;

-- 2. Grant read and write
insert into public.lab_permissions (identity_id, permission)
select id, permission
from public.lab_identities
cross join (values ('documents:read'), ('documents:write')) as p(permission)
where issuer = 'http://localhost:8000'
  and subject = 'local-dev'
on conflict do nothing;

-- 3. Authorize the legacy collection
insert into public.lab_collection_grants
(identity_id, collection, allow_read, allow_write)
select id, 'legacy', true, true
from public.lab_identities
where issuer = 'http://localhost:8000'
  and subject = 'local-dev'
on conflict (identity_id, collection)
do update set allow_read = true, allow_write = true;