# agy-acp: Google Antigravity ACP Harness for KiroCrew

`agy-acp` wraps the [Google Antigravity Python SDK](https://github.com/google-antigravity/antigravity-sdk-python) (`google-antigravity`) in an **Agent Client Protocol (ACP)** JSON-RPC 2.0 server and Cyclopts CLI so that **KiroCrew** (`kiro_crew`) can orchestrate Google Antigravity agents as a drop-in ACP backend harness.

```mermaid
flowchart LR
    subgraph KiroCrew["KiroCrew Workspace & Orchestrator"]
        KC_Client["AcpClient / AcpRuntime"]
        KC_Gate["Tool Gate & Permission Handler"]
        KC_Dash["Usage & Quota Dashboard"]
    end

    subgraph AgyAcp["agy-acp CLI & ACP Server"]
        ACP_Stdio["Stdio JSON-RPC 2.0 Transport"]
        SessionMgr["Session & Steer Manager"]
        AuthHotReload["Hot-Reload Auth Store Watcher"]
        QuotaBridge["5h / 7d Quota & Usage Tracker"]
    end

    subgraph AntigravitySDK["google.antigravity Python SDK"]
        SDK_Agent["Agent (LocalAgentConfig)"]
        SDK_Hooks["pre_tool_call_decide / post_tool_call"]
        SDK_Store["Persistent Trajectory save_dir"]
    end

    KC_Client <-->|"Newline-delimited JSON-RPC 2.0"| ACP_Stdio
    KC_Gate <-->|"session/request_permission"| ACP_Stdio
    ACP_Stdio -->|"usage_update and metadata"| KC_Dash
    ACP_Stdio <--> SessionMgr
    SessionMgr <--> SDK_Agent
    SessionMgr <--> SDK_Hooks
    SDK_Agent <--> SDK_Store
    AuthHotReload -->|"Zero-downtime resume"| SessionMgr
    QuotaBridge -->|"Slash commands & wire telemetry"| SessionMgr
```

## Documentation Sections

- **[Requirements](requirements/overview.md)**: Complete external interface specification expected by KiroCrew, standalone CLI authentication with zero-downtime hot-reload, and 5-hour / 7-day rolling quota requirements.
- **[Verification Traceability Matrix](requirements/verification-matrix.md)**: Mapping from every requirement ID (`REQ-*`) to executable black-box `pytest` integration tests and unit tests.
- **[Design](design/architecture.md)**: Internal component boundaries, protocol translation, session resumption, and TDD implementation plan.
- **[Guides](guides/getting-started.md)**: Developer setup (`asdf`, `uv`, `pre-commit`, Python 3.15) and configuring KiroCrew to launch `agy-acp`.
- **[Runbooks](runbooks/auth-and-sessions.md)**: Operational procedures for credential rotation, session recovery, and quota exhaustion handling.
