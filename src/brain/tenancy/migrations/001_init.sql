-- One business per database, so no table carries a business id: tenant isolation is the
-- connection.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE items (
    id text PRIMARY KEY,
    source text NOT NULL,
    kind text NOT NULL,
    title text NOT NULL,
    location text NOT NULL,
    updated date NOT NULL,
    label text NOT NULL,
    tags text[] NOT NULL,
    -- label and sorted tags as one string: what the policy predicate matches on.
    facts_key text NOT NULL,
    -- The source ACL. Empty means visible to no one (&& with an empty array is false).
    acl_principals text[] NOT NULL
);
CREATE INDEX items_acl_idx ON items USING gin (acl_principals);
CREATE INDEX items_facts_idx ON items (facts_key);

CREATE TABLE chunks (
    -- "<item id>#<ordinal>": stable across reindexing, readable in the audit log.
    id text PRIMARY KEY,
    item_id text NOT NULL REFERENCES items (id) ON DELETE CASCADE,
    ord integer NOT NULL,
    location text NOT NULL,
    body text NOT NULL,
    title_len integer NOT NULL,
    body_len integer NOT NULL,
    embedding vector(1024) NOT NULL,
    UNIQUE (item_id, ord)
);
-- No ANN index on purpose: exact scan keeps the permission predicate exact at demo scale,
-- where an approximate index filtered afterwards can return fewer rows than asked for.

CREATE TABLE postings (
    term text NOT NULL,
    chunk_id text NOT NULL REFERENCES chunks (id) ON DELETE CASCADE,
    title_tf integer NOT NULL,
    body_tf integer NOT NULL,
    PRIMARY KEY (term, chunk_id)
);
CREATE INDEX postings_chunk_idx ON postings (chunk_id);

CREATE TABLE index_meta (
    key text PRIMARY KEY,
    value text NOT NULL
);
