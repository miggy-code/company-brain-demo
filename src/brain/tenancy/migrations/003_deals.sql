-- Unit 05: a HubSpot deal's structured fields, for the MCP `deals` tool. Written in the same
-- transaction as its item, and gone with it: who may see a deal is its item's ACL and policy.
CREATE TABLE deals (
    item_id text PRIMARY KEY REFERENCES items (id) ON DELETE CASCADE,
    pipeline text NOT NULL,
    stage text NOT NULL,
    amount numeric(16, 2) NOT NULL CHECK (amount >= 0),
    currency text NOT NULL,
    owner text NOT NULL,
    close_date date
);
