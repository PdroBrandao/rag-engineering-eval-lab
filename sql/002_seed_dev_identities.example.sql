-- Example seed for local / dev authorization rows. Run manually after 002_auth_rbac.sql.
-- Adjust issuer/subject to match your test JWTs.

insert into public.lab_identities (issuer, subject)
values
  ('https://lab.example.test', 'reader-finance'),
  ('https://lab.example.test', 'writer-legal'),
  ('https://lab.example.test', 'admin-all')
on conflict (issuer, subject) do nothing;

insert into public.lab_permissions (identity_id, permission)
select li.id, perm.permission
from public.lab_identities li
cross join (values ('documents:read'), ('documents:write')) as perm(permission)
where li.subject = 'admin-all'
on conflict do nothing;

insert into public.lab_permissions (identity_id, permission)
select li.id, 'documents:read'
from public.lab_identities li
where li.subject = 'reader-finance'
on conflict do nothing;

insert into public.lab_permissions (identity_id, permission)
select li.id, 'documents:write'
from public.lab_identities li
where li.subject = 'writer-legal'
on conflict do nothing;

insert into public.lab_collection_grants (identity_id, collection, allow_read, allow_write)
select li.id, g.collection, g.allow_read, g.allow_write
from public.lab_identities li
join (
  values
    ('reader-finance', 'finance', true, false),
    ('reader-finance', 'legacy', true, false),
    ('writer-legal', 'legal', false, true),
    ('admin-all', 'finance', true, true),
    ('admin-all', 'legal', true, true),
    ('admin-all', 'engineering', true, true),
    ('admin-all', 'legacy', true, true)
) as g(subject, collection, allow_read, allow_write)
  on g.subject = li.subject
on conflict do nothing;
