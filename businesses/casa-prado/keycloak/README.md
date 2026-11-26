# Casa Prado's Keycloak realm (synthetic, local only)

The realm the MCP server's HTTP transport trusts, imported by `docker compose --profile auth up`.
Every account is a synthetic persona of the fictional Casa Prado, and every password
(`demo-only-<username>`) is a local test value, not a secret.

- `brain-mcp`: the client an MCP client signs in with (authorization code, PKCE S256, loopback
  redirects).
- `brain-mcp-test`: direct grants, for tests and CI only.
- Access tokens (5 minutes) carry two audiences, added by mappers on access tokens only: `agentgate`
  (agentgate's run start) and `http://127.0.0.1:8765/mcp` (the MCP server). No client is named
  after either, because neither verifier checks a token's type.
- `carla-unverified` has Carla's address with `emailVerified: false`; `stranger` is a verified
  address of no one in the business. Both must be refused.
- Valeria exists in the realm from the start but is a person of the business only from week 5 of
  the simulator; before that she is refused like `stranger`.

**Local only. Never deploy this file.** Its passwords are public, its `brain-mcp-test` client lets
anyone with them get a token without a browser, and it runs without TLS (`sslRequired: none`). A
hosted realm (unit 06) needs its own accounts, no direct-grant client, TLS and a public issuer URL.
