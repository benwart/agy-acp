# Verification Traceability Matrix

Every requirement in `docs/requirements/` is verified by an executable `pytest` integration test in `tests/integration/` that treats `agy-acp` as a black box (invoking the CLI or communicating over the ACP stdio JSON-RPC stream), supplemented by unit tests in `tests/unit/` that verify internal logic and providers in isolation.

!!! important "TDD Red Phase Scaffolding"
    In the initial scaffolding phase, all requirement verification tests and unit behavioral tests are expected to **fail** (`Red` phase of TDD) because the scaffolded command handlers and logic classes raise `NotImplementedError` until each increment is implemented.

## Requirements-to-Pytest Mapping

| Requirement ID | Summary | Black-Box Integration Verification (`tests/integration/`) | Unit Verification (`tests/unit/`) |
|---|---|---|---|
| `REQ-ACP-001` | Stdio newline-delimited JSON-RPC 2.0 transport & stdout purity | `test_acp_harness_interface.py::test_req_acp_001_stdio_framing_and_stdout_purity` | `test_acp_dispatcher.py::test_jsonrpc_frame_serialization` |
| `REQ-ACP-002` | KiroCrew backend CLI invocation compatibility (`--profile acp`, `--with-builtin`, `--agent`, etc.) | `test_acp_harness_interface.py::test_req_acp_002_kirocrew_cli_invocation_variants` | `test_cli_structure.py::test_acp_serve_accepts_kirocrew_flags` |
| `REQ-ACP-003` | `initialize` handshake, `agentCapabilities`, `agentInfo` | `test_acp_harness_interface.py::test_req_acp_003_initialize_handshake_and_capabilities` | `test_acp_dispatcher.py::test_handle_initialize_capabilities` |
| `REQ-ACP-004` | `session/new` workspace binding, `mcpServers`, `modes`, `models`, `configOptions` | `test_acp_harness_interface.py::test_req_acp_004_session_new_mcp_and_config_options` | `test_session_manager.py::test_create_session_maps_mcp_and_cwd` |
| `REQ-ACP-005` | `session/load` native trajectory resumption from persistent `save_dir` | `test_acp_harness_interface.py::test_req_acp_005_session_load_resumes_existing_session` | `test_session_manager.py::test_load_session_resumes_trajectory` |
| `REQ-ACP-006` | Multiplexed multi-session routing over single stdio stream | `test_acp_harness_interface.py::test_req_acp_006_multiplexed_concurrent_sessions` | `test_session_manager.py::test_multiplexed_session_isolation` |
| `REQ-ACP-007` | `session/set_mode`, `session/set_model`, `session/set_config_option` | `test_acp_harness_interface.py::test_req_acp_007_set_mode_model_and_config_options` | `test_session_manager.py::test_update_session_mode_and_model` |
| `REQ-ACP-008` | `session/prompt` streaming `agent_thought_chunk`, `agent_message_chunk`, image blocks, and `stopReason` | `test_acp_harness_interface.py::test_req_acp_008_session_prompt_streaming_thoughts_and_text` | `test_acp_dispatcher.py::test_prompt_streams_chunks_and_usage` |
| `REQ-ACP-009` | Tool call notifications & `session/request_permission` round-trip gating | `test_acp_harness_interface.py::test_req_acp_009_tool_call_and_permission_gating` | `test_acp_dispatcher.py::test_tool_permission_request_allow_and_reject` |
| `REQ-ACP-010` | `session/cancel` notification & mid-turn `steer` overlapping `session/prompt` | `test_acp_harness_interface.py::test_req_acp_010_session_cancel_and_mid_turn_steer` | `test_session_manager.py::test_cancel_and_steer_active_turn` |
| `REQ-ACP-011` | `session/close`, `_kiro.dev/session/terminate`, `_kiro/session/delete` | `test_acp_harness_interface.py::test_req_acp_011_session_teardown_methods` | `test_session_manager.py::test_terminate_and_delete_session` |
| `REQ-AUTH-001` | `agy-acp auth login` atomic credential persistence | `test_auth_and_hot_reload.py::test_req_auth_001_cli_login_persists_credentials` | `test_auth_manager.py::test_login_writes_atomic_credentials` |
| `REQ-AUTH-002` | `agy-acp auth logout` atomic credential clearing | `test_auth_and_hot_reload.py::test_req_auth_002_cli_logout_clears_credentials` | `test_auth_manager.py::test_logout_increments_generation` |
| `REQ-AUTH-003` | `agy-acp auth status` JSON/text inspection & exit codes | `test_auth_and_hot_reload.py::test_req_auth_003_cli_auth_status_reporting` | `test_auth_manager.py::test_status_reflects_active_credential_state` |
| `REQ-AUTH-004` | Zero-downtime hot-reload of login/logout across running ACP server | `test_auth_and_hot_reload.py::test_req_auth_004_running_acp_server_hot_reloads_auth_without_restart` | `test_auth_manager.py::test_detects_external_credential_rotation` |
| `REQ-QUOTA-001` | `agy-acp quota five-hour` CLI command (`5h` window) | `test_quota_and_usage.py::test_req_quota_001_cli_five_hour_quota` | `test_quota_tracker.py::test_five_hour_bucket_calculation` |
| `REQ-QUOTA-002` | `agy-acp quota seven-day` CLI command (`7d` window) | `test_quota_and_usage.py::test_req_quota_002_cli_seven_day_quota` | `test_quota_tracker.py::test_seven_day_bucket_calculation` |
| `REQ-QUOTA-003` | `agy-acp quota credits` & `agy-acp quota status` CLI commands | `test_quota_and_usage.py::test_req_quota_003_cli_credits_and_combined_status` | `test_quota_tracker.py::test_combined_quota_and_token_ledger` |
| `REQ-QUOTA-004` | KiroCrew ACP `usage_update`, `_kiro.dev/metadata`, `/quota` slash command & rate limit notice | `test_quota_and_usage.py::test_req_quota_004_kirocrew_wire_usage_and_slash_commands` | `test_quota_tracker.py::test_format_kirocrew_usage_and_slash_response` |
| `REQ-CLI-001` | Cyclopts hierarchical `commands/` discovery, kebab-case filenames, `default` handler | `test_cli_and_docs.py::test_req_cli_001_hierarchical_cli_execution` | `test_cli_structure.py::test_commands_hierarchy_and_docstrings` |
| `REQ-CLI-002` | Zensical documentation site build & serve | `test_cli_and_docs.py::test_req_cli_002_zensical_docs_build` | `test_cli_structure.py::test_zensical_config_and_pages` |
