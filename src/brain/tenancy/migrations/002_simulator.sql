-- Unit 04: people as sync state, document versions, and the change feed.

-- The people of the business. people.yaml seeds week 0; sync events change it. Only active
-- people resolve; an offboarded person keeps their row (history) with no memberships.
CREATE TABLE persons (
    id text PRIMARY KEY,
    name text NOT NULL,
    email text NOT NULL UNIQUE,
    role text NOT NULL,
    title text NOT NULL,
    status text NOT NULL CHECK (status IN ('active', 'offboarded')),
    groups text[] NOT NULL,
    slack_channels text[] NOT NULL,
    hubspot_teams integer[] NOT NULL
);

-- Items with the same series are versions of one document. Null: no known versions.
ALTER TABLE items ADD COLUMN series text;
CREATE INDEX items_series_idx ON items (series) WHERE series IS NOT NULL;

-- The change feed a source writes and the sync worker drains, in seq order.
CREATE TABLE source_events (
    seq bigserial PRIMARY KEY,
    source text NOT NULL,
    kind text NOT NULL,
    payload jsonb NOT NULL,
    -- The source's own date for the change (simulated time in the simulator).
    sim_date date NOT NULL,
    committed_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    applied_at timestamptz
);
CREATE INDEX source_events_pending_idx ON source_events (seq) WHERE applied_at IS NULL;
