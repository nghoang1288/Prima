# Antigravity Bootstrap

Configure this agent once, locally.

## Local aliases

Store machine-specific paths OUTSIDE Git, for example in:

`~/.config/prima-antigravity/local.json`

The local configuration must define at least:

- `PRIMA_REPO` — the local PRIMA workspace;
- `PRIMA_RUNTIME_ROOT` — local runtime/worktree/cache storage;
- optional local case aliases such as `CASE001`.

Patient names and source paths are local-only and must never be copied into
GitHub task/result files.

## Poll loop

Every 300 seconds:

1. `git fetch origin agent-control modern-runtime`
2. Read `ANTIGRAVITY_BRIDGE.md` from `origin/agent-control`.
3. Enumerate `.antigravity/tasks/*.json` on `origin/agent-control`.
4. Process pending tasks according to the protocol.
5. Push a receipt before non-trivial execution.
6. Push a terminal result after execution.
7. If there is no new task, do nothing and create no commit.

Do not use file modification time as execution state. Use task_id plus
receipt/result files and a local ledger.

## Update handling

The protocol itself may evolve. On every poll, re-read
`ANTIGRAVITY_BRIDGE.md` from the fetched `agent-control` branch before
executing a newly discovered task.

A task's `code.base_sha` is authoritative for that task. Never silently run a
newer `modern-runtime` commit.

## First task

The initial handshake task is:

`.antigravity/tasks/PRIMA-20260920-001.json`

Complete it and push its terminal result before accepting subsequent tasks.
