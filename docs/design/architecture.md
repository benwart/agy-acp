# Architecture & Component Design

`agy-acp` separates its external boundaries (`commands/` Cyclopts CLI and async stdio JSON-RPC 2.0 framing) from protocol/domain orchestration (`logic/`) and concrete infrastructure adapters (`providers/`).

```mermaid
flowchart TD
    subgraph CLI["CLI Entrypoints (src/agy_acp/commands/)"]
        CMD_Serve["serve.py & acp/ (serve, validate-gate, logs)"]
        CMD_Chat["chat.py, whoami.py, status.py, update.py"]
        CMD_Auth["auth/ (login, logout, status) & agent/validate.py"]
        CMD_Quota["quota/ (status, five-hour, seven-day, credits)"]
        CMD_Docs["docs/ (build, serve)"]
    end

    subgraph Logic["Domain Logic (src/agy_acp/logic/)"]
        Dispatcher["AcpDispatcher + wire/ (v2_kiro, v3_kas)"]
        SessionMgr["SessionManager (Lifecycle, Fork, Compact, Steer)"]
        Classifier["tool_classifier.py & custom_agents.py"]
        Prober["McpCatalogProber & CarouselNormalizer"]
        AuthMgr["AuthManager (Fingerprint & Hot-Reload)"]
        QuotaTracker["QuotaTracker (5h/7d Windows & Token Ledger)"]
    end

    subgraph Providers["Providers (src/agy_acp/providers/)"]
        Acp_Client["AgyAcpClientProvider (agy acp --stdio --process-per-session)"]
        State_Store["SqliteStateStoreProvider (Sync SQLite WAL agy_acp.db)"]
        Cred_Store["CredentialStoreProvider & OAuthProvider"]
        Quota_Provider["QuotaProvider (run_agy_slash_json)"]
    end

    CMD_Serve --> Dispatcher
    CMD_Chat --> Acp_Client
    CMD_Chat --> QuotaTracker
    CMD_Auth --> AuthMgr
    CMD_Quota --> QuotaTracker
    Dispatcher --> SessionMgr
    Dispatcher --> QuotaTracker
    SessionMgr --> AuthMgr
    SessionMgr --> Acp_Client
    SessionMgr --> State_Store
    SessionMgr --> Classifier
    SessionMgr --> Prober
    Acp_Client --> Classifier
    Acp_Client --> State_Store
    AuthMgr --> Cred_Store
    QuotaTracker --> Quota_Provider
    QuotaTracker --> State_Store
```

## SOLID & GoF Pattern Application

1. **Adapter & Actor Supervisor (`AgyAcpClientProvider` + `_AcpSubprocessConnection`)**: Supervises a persistent `agy acp --stdio --process-per-session` subprocess inside an `anyio` structured-concurrency `TaskGroup`, decoupling full-duplex JSON-RPC multiplexing, crash auto-reconnect, and shielded POSIX process-group teardown (`os.killpg`) from `SessionManager`.
2. **Strategy (`EngineWireAdapter` — `v2_kiro.py` & `v3_kas.py`)**: Translates canonical session descriptors, tool calls, config options, steering lifecycle updates, and metadata into either Kiro ACP (`v2`) or Kiro Agent Server (`v3` / KAS) wire shapes without branching business logic.
3. **Mediator / Router (`AcpDispatcher` + Off-Loop `serve.py`)**: Routes inbound JSON-RPC requests and notifications concurrently while serializing outbound `stdout` frame batches off the event loop via `anyio.to_thread.run_sync` under an `anyio.Lock`.
4. **Chain of Policy & Security Containment (`tool_classifier.py` + `custom_agents.py`)**: Classifies `agy acp` native and MCP tool invocations into canonical ACP kinds, blocks native `agy` subagent tools when `kirocrew-core` `spawn_run` routing is active, enforces `.kiro/agents` `allowedTools` and path/command rules across both `tool_call` updates and `session/request_permission` callbacks, and validates `cwd`-contained regular files (`O_NOFOLLOW | O_NONBLOCK` + byte caps).
5. **Repository (`SqliteStateStoreProvider`)**: Persists sessions, turn history (`session_turns`), 24-hour TTL `models_cache`, cumulative `token_ledger`, and key-value state synchronously in SQLite WAL mode (`agy_acp.db`).
