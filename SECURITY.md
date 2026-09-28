# Security Policy

## Supported versions

| Version | Supported |
|---|---|
| 1.x (latest) | Yes |
| 0.6.x | No — upgrade to 1.x (no schema changes from 0.6.0; read the 1.0.0 upgrade notes in [CHANGELOG.md](CHANGELOG.md)) |
| 0.5.x | No — upgrade to 0.6.x or later (0.5.0 and earlier also carry the pre-0.5.1 data-loss bug — see [CHANGELOG.md](CHANGELOG.md)) |
| < 0.5.0 | No — please upgrade |

From 1.0 on, the public surface listed under "Compatibility policy" in
[README.md](README.md) is stable across 1.x. Security fixes ship as a patch
release on the latest 1.x minor line.

## Reporting a vulnerability

Please use GitHub's private vulnerability reporting for this repository
(Security tab → "Report a vulnerability") rather than opening a public
issue. This lets us assess impact and prepare a fix before public
disclosure.

If private reporting is unavailable to you, open a normal issue describing
the *category* of the problem without exploit details, and ask for a
private channel — we will follow up.

Please include:

- The version (`hermes-pgvector --version`) and how you installed it (pip,
  the install script, manual).
- Postgres version, pgvector version, and hermes-agent version if relevant.
- Minimal reproduction steps or a description of the failure mode.

We will acknowledge reports within a reasonable time and aim to ship a
fix (patch release, per [`docs/release/RELEASING.md`](docs/release/RELEASING.md))
before any public disclosure of details.

## Scope notes

This plugin is a **storage layer**, not a memory model or an access-control
system (see "Design philosophy" in [README.md](README.md)). A few things
that follow from that, worth knowing when assessing impact:

- **PII handling is bucketing, not encryption.** Direct-message,
  group/channel/thread, and benchmark traffic are isolated into dedicated
  themes (`whatsapp-dm`, `external-group`, `_bench`) at write time so they
  do not pollute or leak into ordinary theme recall — see "Multi-agent /
  per-minion themes" in README.md and `hermes_pgvector/identity.py`. This
  is *isolation*, not encryption at rest: row content in those tables is
  plain text in Postgres, same as every other row. If your embed endpoint
  or database is compromised, everything in it is exposed regardless of
  bucket.
- **No credentials are ever included in tool output.** Exception text
  returned to the model (and potentially persisted via conversation
  capture) has credential-looking fragments (`password=`, `sslkey=`, etc.)
  redacted before it is returned. Bearer tokens for the embed endpoint
  (`embed_api_key_env`) are read from the *named* environment variable at
  call time and are never logged, stored in config, or included in
  exception messages — only the variable's *name* lives in config.
- **Access control is Postgres roles, not a plugin-layer RBAC.** The
  runtime role gets DML only on plugin-owned tables (via migrations); this
  plugin does not implement per-theme authentication or authorization
  beyond the `agent_identity` scoping the hermes-agent gateway already
  establishes via `X-Hermes-Session-Key`. Anyone who can reach the gateway
  with a given theme header can read/write that theme's memory — the same
  trust boundary as the built-in `memory` tool it mirrors.
- **The maintenance CLI (`hermes-pgvector cleanup`/`prune`/`remap`) is a
  privileged operator tool**, not exposed to the agent. Destructive
  commands default to dry-run and require `--execute`; every mutating run
  is logged to `memory_maintenance_log`. Restrict who can run it the same
  way you would restrict `psql` access to this database.
- **Migrations require database superuser/owner privileges** (`CREATE
  EXTENSION vector`, `CREATE TABLE`) — see the admin/runtime split in
  README.md's "Design philosophy". Only the admin/deploy path needs that
  privilege; the running plugin never does.
