# Verification Traceability Matrix

Every requirement in `docs/requirements/` is verified by executable `pytest` integration suites in `tests/integration/` (including live zero-token contract tests against `agy acp --stdio --process-per-session`) and isolated unit tests in `tests/unit/`.

## Requirements-to-Pytest Mapping

| Requirement / Contract | Summary | Integration Verification (`tests/integration/`) | Unit Verification (`tests/unit/`) |
|---|---|---|---|
| `REQ-ACP-001` | Stdio JSON-RPC 2.0 framing, off-loop stdout writes & stdout purity | `test_acp_harness_interface.py::test_req_acp_001_stdio_framing_and_stdout_purity` | `test_serve.py`, `test_acp_dispatcher.py` |
| `REQ-ACP-002` | KiroCrew backend CLI invocation compatibility (`v2` & `v3` KAS flags) | `test_acp_harness_interface.py::test_req_acp_002_kirocrew_cli_invocation_variants` | `test_cli_structure.py` |
| `REQ-ACP-003` | `initialize` handshake, `agentCapabilities`, `agentInfo` | `test_acp_harness_interface.py::test_req_acp_003_initialize_handshake_and_capabilities` | `test_acp_dispatcher.py`, `test_wire_adapters.py` |
| `REQ-ACP-004` | `session/new` workspace binding, per-session `mcpServers`, `modes`, `models`, `configOptions` | `test_acp_harness_interface.py`, `test_kas_v3_harness_interface.py` | `test_session_manager.py`, `test_agy_acp_client.py` |
| `REQ-ACP-005` | `session/load` SQLite + `agy acp` resumption & 24h `models_cache` refresh | `test_acp_harness_interface.py::test_req_acp_005_session_load_resumes_existing_session` | `test_session_manager.py`, `test_state_store.py` |
| `REQ-ACP-006` | Multiplexed multi-session routing over persistent `agy acp` subprocess | `test_acp_harness_interface.py::test_req_acp_006_multiplexed_concurrent_sessions` | `test_agy_acp_client.py`, `test_session_manager.py` |
| `REQ-ACP-007` | `session/set_mode`, `session/set_model`, `session/set_config_option` (`effortLevel`) | `test_acp_harness_interface.py`, `test_kas_v3_harness_interface.py` | `test_session_manager.py`, `test_custom_agents.py` |
| `REQ-ACP-008` | `session/prompt` streaming chunks, inline contained base64 images, and `stopReason` | `test_acp_harness_interface.py::test_req_acp_008_session_prompt_streaming_thoughts_and_text` | `test_acp_dispatcher.py`, `test_carousel_normalizer.py` |
| `REQ-ACP-009` | Real-time `session/request_permission` gating, `.kiro/agents` policy & native subagent block | `test_acp_harness_interface.py`, `test_kas_v3_harness_interface.py` | `test_agy_acp_client.py`, `test_tool_classifier.py`, `test_custom_agents.py` |
| `REQ-ACP-010` | `session/cancel` & mid-turn `_session/steer` / `_session/steering` replay + turn persistence | `test_acp_harness_interface.py`, `test_kas_v3_harness_interface.py` | `test_agy_acp_client.py`, `test_acp_dispatcher.py` |
| `REQ-ACP-011` | `session/fork`, `_kiro/checkpoint/revertMultiple`, `_kiro/session/compact`, and session close/delete | `test_acp_harness_interface.py`, `test_kas_v3_harness_interface.py` | `test_session_manager.py`, `test_acp_dispatcher.py` |
| `LIVE-AGY-ACP` | Live `agy acp` subprocess handshake, concurrent per-session MCP isolation (`mcp_config.json` untouched), crash recovery & process-group reap | `test_live_agy_acp_subprocess.py` | `test_agy_acp_client.py` |
| `REQ-AUTH-001..004` | Out-of-band `auth login | logout | status` & zero-downtime hot-reload | `test_auth_and_hot_reload.py` | `test_auth_manager.py` |
| `REQ-QUOTA-001..004` | `5h` / `7d` rolling quotas, credits, SQLite `token_ledger`, and `/quota` `/usage` `/credits` | `test_quota_and_usage.py` | `test_quota_tracker.py` |
| `REQ-CLI-001..002` | Cyclopts hierarchical `commands/` discovery & Zensical documentation site | `test_cli_and_docs.py` | `test_cli_structure.py` |
