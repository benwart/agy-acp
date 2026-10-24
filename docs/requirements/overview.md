# Requirements Overview

This document series defines the external interface requirements for `agy-acp`. All requirements are verified through black-box integration tests (`tests/integration/`) that exercise the external CLI and stdio JSON-RPC 2.0 boundaries without coupling to internal implementation details.

## Requirement Domains

| DomainPrefix | Document | Scope |
|---|---|---|
| `REQ-ACP-*` | [ACP Harness External Interface](acp-interface.md) | All JSON-RPC 2.0 methods, notifications, server-to-client requests, CLI invocation signatures, and tool-gate protocols expected by KiroCrew (`kiro_crew`). |
| `REQ-AUTH-*` | [CLI Authentication & Hot-Reload](authentication.md) | Out-of-band `agy-acp auth login`, `logout`, `status` commands and zero-downtime credential hot-reload in an already-connected ACP server. |
| `REQ-QUOTA-*` | [5h / 7d Quotas & Token Usage](quota-and-usage.md) | CLI commands for 5-hour rolling quota, 7-day (weekly) rolling quota, G1 credits, token usage, and KiroCrew wire/chat visibility. |
| `REQ-CLI-*` | [CLI Hierarchy & Docs Site](cli-and-docs.md) | Cyclopts hierarchical command structure (`commands/` subfolders, kebab-case modules, `default` handler) and Zensical documentation site. |

## Design & Verification Philosophy

1. **Rigid External Contract, Malleable Internals**: Integration verifications spawn or invoke the public CLI and ACP stdio stream directly. Internal classes in `agy_acp.logic` and `agy_acp.providers` can be refactored freely without changing integration tests.
2. **Test-Driven Development (Red -> Green -> Refactor)**: The initial repository scaffolding defines all external CLI entrypoints and stubs that fail verification tests until each requirement is incrementally implemented.
3. **Zero-Downtime Operations**: Running `agy-acp auth login` or `agy-acp auth logout` in a separate terminal must never require restarting an active `agy-acp` ACP server connected to KiroCrew.
