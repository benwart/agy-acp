# CLI Hierarchy & Documentation Site Requirements

---

## 1. Cyclopts Hierarchical CLI Architecture

### REQ-CLI-001: Hierarchical `commands/` Discovery with Kebab-Case Modules & Docstring Help
- **Framework**: Built using `cyclopts` (`cyclopts.App`).
- **Command Hierarchy (`src/agy_acp/commands/`)**:
  - All CLI commands reside under `src/agy_acp/commands/`.
  - **Sub-folders as Sub-commands**: Every non-underscore sub-directory inside `commands/` registers as a nested `cyclopts.App` command group whose name is the directory name (in kebab-case) and whose group `--help` text is extracted from the module docstring of the sub-directory's `__init__.py`.
  - **Kebab-Case Module Filenames**: Every non-underscore `.py` file inside `commands/` (or any nested sub-folder) registers as a command whose CLI name is the module filename stem in kebab-case (e.g., `five-hour.py` -> `five-hour`, `seven-day.py` -> `seven-day`, `validate-gate.py` -> `validate-gate`).
  - **`default` Handler & Help Text**: Each command module defines a callable named `default` whose docstring provides the CLI `--help` text.
- **Supporting Packages**:
  - `src/agy_acp/logic/`: ACP JSON-RPC routing, `v2`/`v3` wire adapters, session lifecycle, tool classification, `.kiro/agents` custom-agent policies, MCP catalog probing, credential hot-reload, and quota aggregation.
  - `src/agy_acp/providers/`: Full-duplex `agy acp` subprocess client (`AgyAcpClientProvider`), synchronous SQLite WAL state store (`SqliteStateStoreProvider`), quota inspection (`QuotaProvider`), OAuth/credential persistence, and structured ACP logging.

### Command Tree
```text
agy-acp
├── serve                  (commands/serve.py -> default)
├── chat                   (commands/chat.py -> default)
├── status                 (commands/status.py -> default)
├── whoami                 (commands/whoami.py -> default)
├── update                 (commands/update.py -> default)
├── acp                    (commands/acp/__init__.py docstring)
│   ├── serve              (commands/acp/serve.py -> default)
│   ├── validate-gate      (commands/acp/validate-gate.py -> default)
│   └── logs               (commands/acp/logs.py -> default)
├── agent                  (commands/agent/__init__.py docstring)
│   └── validate           (commands/agent/validate.py -> default)
├── auth                   (commands/auth/__init__.py docstring)
│   ├── login              (commands/auth/login.py -> default)
│   ├── logout             (commands/auth/logout.py -> default)
│   └── status             (commands/auth/status.py -> default)
├── quota                  (commands/quota/__init__.py docstring)
│   ├── status             (commands/quota/status.py -> default)
│   ├── five-hour          (commands/quota/five-hour.py -> default)
│   ├── seven-day          (commands/quota/seven-day.py -> default)
│   └── credits            (commands/quota/credits.py -> default)
└── docs                   (commands/docs/__init__.py docstring)
    ├── build              (commands/docs/build.py -> default)
    └── serve              (commands/docs/serve.py -> default)
```

---

## 2. Zensical Documentation Site

### REQ-CLI-002: Zensical Site Configuration & Content Organization
- **Configuration**: `zensical.toml` at the repository root configuring navigation (`nav`), light/dark palettes, code highlighting, admonitions, and Mermaid diagrams (`pymdownx.superfences`).
- **Information Architecture (`docs/`)**:
  - `docs/requirements/`: External interface requirements and verification traceability matrix.
  - `docs/design/`: Internal architecture, ACP-to-`agy acp` subprocess bridge design, and auth/SQLite/quota design.
  - `docs/guides/`: Developer quickstart and KiroCrew setup guide.
  - `docs/runbooks/`: Operational runbooks for authentication hot-reload, session troubleshooting, and quota management.
