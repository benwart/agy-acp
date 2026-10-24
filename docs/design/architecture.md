# Architecture & Component Design

`agy-acp` separates its rigid external boundaries (`commands/` CLI and ACP JSON-RPC 2.0 stdio framing) from its internal orchestration (`logic/`) and concrete infrastructure adapters (`providers/`).

```mermaid
flowchart TD
    subgraph CLI["CLI Entrypoints (src/agy_acp/commands/)"]
        CMD_Serve["acp/serve.py and serve.py"]
        CMD_Auth["auth/ (login, logout, status)"]
        CMD_Quota["quota/ (status, five-hour, seven-day, credits)"]
        CMD_Docs["docs/ (build, serve)"]
    end

    subgraph Logic["Domain Logic (src/agy_acp/logic/)"]
        Dispatcher["AcpDispatcher (JSON-RPC 2.0 Router)"]
        SessionMgr["SessionManager (Lifecycle, Steer, Cancel)"]
        AuthMgr["AuthManager (Fingerprint & Hot-Reload)"]
        QuotaTracker["QuotaTracker (5h/7d Windows & Token Ledger)"]
    end

    subgraph Providers["Providers (src/agy_acp/providers/)"]
        SDK_Provider["AntigravitySdkProvider (google.antigravity)"]
        Cred_Store["CredentialStoreProvider (Atomic 0600 JSON)"]
        Quota_Provider["QuotaProvider (5h/7d Bucket & Credit Source)"]
    end

    CMD_Serve --> Dispatcher
    CMD_Auth --> AuthMgr
    CMD_Quota --> QuotaTracker
    Dispatcher --> SessionMgr
    Dispatcher --> QuotaTracker
    SessionMgr --> AuthMgr
    SessionMgr --> SDK_Provider
    AuthMgr --> Cred_Store
    QuotaTracker --> Quota_Provider
```

## SOLID & GoF Pattern Application

1. **Adapter Pattern (`AntigravitySdkProvider`)**: Adapts `google.antigravity.Agent`, `LocalAgentConfig`, `CapabilitiesConfig`, `policy`, and `hooks` into an abstract session port consumed by `SessionManager`.
2. **Mediator / Router (`AcpDispatcher`)**: Decouples stdio framing and JSON-RPC request/response correlation (including server-to-client `session/request_permission` futures) from session execution.
3. **Strategy & Proxy (`AuthManager` + `CredentialStoreProvider`)**: Checks credential generation fingerprints before each turn and transparently rebinds the underlying SDK session (`SessionContinuationMode.RESUME`) when credentials change on disk.
4. **Observer (`@hooks.pre_tool_call_decide` & `@hooks.post_tool_call`)**: Streams real-time `tool_call` and `tool_call_update` notifications and gates tool execution on KiroCrew's `session/request_permission` response.
