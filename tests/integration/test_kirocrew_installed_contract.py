"""Contract tests verifying `agy-acp` against installed `KiroCrew.app` (`kiro_crew` ACP package)."""

from __future__ import annotations

import ast
import asyncio
import json
import os
import re
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from agy_acp import KIRO_CLI_COMPAT_VERSION
from agy_acp.logic.acp_dispatcher import AcpDispatcher, _to_normalized_tool_event
from agy_acp.logic.session_manager import SessionManager
from agy_acp.logic.tool_classifier import classify_tool_call_step
from agy_acp.logic.wire import (
    EngineWireAdapter,
    V2KiroWireAdapter,
    V3KasWireAdapter,
    WireTurnTelemetry,
)
from agy_acp.providers.agy_acp_client import _build_tool_event_from_record
from agy_acp.providers.state_store import SqliteStateStoreProvider
from tests.integration.conftest import CliResult, HarnessFactory

DEFAULT_KIROCREW_PKG_DIR = Path(
    "/Applications/KiroCrew.app/Contents/Resources/backend-dist/"
    "kirocrew-backend-arm64/lib/python3.12/site-packages/kiro_crew"
)
SRC_ROOT = str(Path(__file__).resolve().parents[2] / "src")
_KIRO_VERSION_RE = re.compile(r"(\d{1,9})\.(\d{1,9})(?:\.(\d{1,9}))?(?!\d)")
_POSSIBLE_VALUES_RE = re.compile(r"\[possible values:\s*([^\]]*)\]")


def _resolve_kirocrew_pkg_dir() -> Path:
    override = os.environ.get("KIROCREW_PACKAGE_DIR")
    pkg_dir = Path(override) if override else DEFAULT_KIROCREW_PKG_DIR
    if not pkg_dir.is_dir():
        pytest.skip(f"Installed kiro_crew package not found at {pkg_dir}")
    return pkg_dir


def _extract_module_literals(
    py_file: Path,
    *,
    pkg_root: Path | None = None,
) -> dict[str, Any]:
    """Extract top-level and class-body literal assignments from `py_file` via `ast`."""
    tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
    values: dict[str, Any] = {}

    def _eval_node(node: ast.AST) -> Any:
        if isinstance(node, ast.Name) and node.id in values:
            return values[node.id]
        if isinstance(node, ast.Set):
            return {_eval_node(elt) for elt in node.elts}
        if isinstance(node, ast.Tuple):
            return tuple(_eval_node(elt) for elt in node.elts)
        if isinstance(node, ast.List):
            return [_eval_node(elt) for elt in node.elts]
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "frozenset" and len(node.args) == 1:
                return frozenset(_eval_node(node.args[0]))
        return ast.literal_eval(node)

    def _collect(body: list[ast.stmt], prefix: str = "") -> None:
        for stmt in body:
            if (
                isinstance(stmt, ast.ImportFrom)
                and pkg_root is not None
                and stmt.module
                and stmt.module.startswith("kiro_crew.")
            ):
                mod_path = pkg_root.joinpath(*stmt.module.split(".")[1:]).with_suffix(".py")
                if mod_path.is_file():
                    imported = _extract_module_literals(mod_path)
                    for alias in stmt.names:
                        if alias.name in imported:
                            values[alias.asname or alias.name] = imported[alias.name]
            elif isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                target = stmt.targets[0]
                if isinstance(target, ast.Name):
                    try:
                        values[f"{prefix}{target.id}"] = _eval_node(stmt.value)
                    except ValueError, TypeError, SyntaxError:
                        continue
            elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
                if isinstance(stmt.target, ast.Name):
                    try:
                        values[f"{prefix}{stmt.target.id}"] = _eval_node(stmt.value)
                    except ValueError, TypeError, SyntaxError:
                        continue
            elif isinstance(stmt, ast.ClassDef):
                _collect(stmt.body, prefix=f"{prefix}{stmt.name}.")

    _collect(tree.body)
    return values


@dataclass(frozen=True)
class InstalledKiroCrewContract:
    """Extracted version floors, CLI flags, and ACP/KAS wire constants from `kiro_crew`."""

    kirocrew_version: str
    kas_pin_version: str
    version_floors: dict[str, tuple[int, int, int]]
    kiro_v2_protocol_version: int | str
    kas_v3_protocol_version: int | str
    kas_relay_subcommand: str
    kas_relay_engine_flag: str
    kas_relay_engine: str
    kas_relay_auth_flag: str
    kas_auth_modes: tuple[str, ...]
    kas_refusal_words: str
    kas_capabilities: tuple[str, ...]
    kas_extension_methods: tuple[str, ...]
    kas_mcp_toggle_method: str
    client_to_agent_methods: dict[str, str]
    kas_wire_kinds: dict[str, str]
    kas_credit_unit: str
    v2_fallback_session_update: str
    kas_tool_family_by_kiro_name: dict[str, tuple[str, ...]]
    allowed_tools_builtins: tuple[str, ...]
    policy_alias_deny_only_ids: frozenset[str]
    crew_only_entry_keys: frozenset[str]


@pytest.fixture(scope="module")
def installed_contract() -> InstalledKiroCrewContract:
    """Load and parse the contract directly from installed `KiroCrew.app`."""
    pkg = _resolve_kirocrew_pkg_dir()
    init_vals = _extract_module_literals(pkg / "__init__.py")
    hot_reload_vals = _extract_module_literals(pkg / "mcp_hot_reload.py")
    kiro_cli_vals = _extract_module_literals(pkg / "kiro_cli.py")
    kas_pin_vals = _extract_module_literals(pkg / "acp" / "kas_pin.py")
    kas_feat_vals = _extract_module_literals(pkg / "acp" / "kas_engine_features.py")
    kas_refusal_vals = _extract_module_literals(pkg / "acp" / "kas_engine_refusal.py")
    kas_transport_vals = _extract_module_literals(pkg / "acp" / "kas_transport.py")
    kas_wire_vals = _extract_module_literals(pkg / "acp" / "kas_wire.py")
    kas_perm_vals = _extract_module_literals(pkg / "acp" / "kas_permissions.py", pkg_root=pkg)
    kiro_harness_vals = _extract_module_literals(pkg / "acp" / "harness" / "kiro.py")
    kas_harness_vals = _extract_module_literals(pkg / "acp" / "harness" / "kas.py")
    types_vals = _extract_module_literals(pkg / "acp" / "types.py")
    spawn_launch_vals = _extract_module_literals(pkg / "acp" / "runtime_spawn_launch.py")
    fallback_vals = _extract_module_literals(pkg / "acp" / "engine_fallback.py")
    tool_names_vals = _extract_module_literals(pkg / "platform" / "tool_names.py")

    floors: dict[str, tuple[int, int, int]] = {
        "MCP_HOT_RELOAD_MIN_KIRO_CLI_VERSION": tuple(
            hot_reload_vals["MCP_HOT_RELOAD_MIN_KIRO_CLI_VERSION"]
        ),
        "SPEC_PERMISSIONS_MIN_VERSION": tuple(kiro_cli_vals["SPEC_PERMISSIONS_MIN_VERSION"]),
        "MANDATORY_MCPS_DROP_MIN_VERSION": tuple(kiro_cli_vals["MANDATORY_MCPS_DROP_MIN_VERSION"]),
        "KAS_AGENT_PROMPT_MIN_KIRO_CLI": tuple(kas_harness_vals["KAS_AGENT_PROMPT_MIN_KIRO_CLI"]),
        "KAS_AUTH_CLIENT_MIN_KIRO_CLI": tuple(kiro_cli_vals["KAS_AUTH_CLIENT_MIN_KIRO_CLI"]),
        "STANDALONE_KIRO_CLI_MIN_VERSION": tuple(kiro_cli_vals["STANDALONE_KIRO_CLI_MIN_VERSION"]),
        "KAS_SPEC_PERMISSIONS_MIN_ENGINE_VERSION": tuple(
            kas_pin_vals["KAS_SPEC_PERMISSIONS_MIN_ENGINE_VERSION"]
        ),
        "OAUTH_MIN_ENGINE_VERSION": tuple(kas_feat_vals["OAUTH_MIN_ENGINE_VERSION"]),
    }

    kas_extension_methods = (
        types_vals["METHOD_KAS_SESSION_COMPACT"],
        types_vals["METHOD_KAS_SESSION_DELETE"],
        types_vals["METHOD_KAS_MCP_RESET_SERVER"],
        types_vals["METHOD_KAS_SESSION_CONTEXT"],
        types_vals["METHOD_KAS_KNOWLEDGE"],
        types_vals["METHOD_KAS_CODE_INTELLIGENCE"],
        types_vals["METHOD_KAS_CHECKPOINT_REVERT"],
    )

    client_methods = {
        key: str(val)
        for key, val in types_vals.items()
        if key.startswith("METHOD_")
        and key
        not in {
            "METHOD_SESSION_UPDATE",
            "METHOD_KIRO_SESSION_UPDATE",
            "METHOD_METADATA",
            "METHOD_COMPACTION_STATUS",
            "METHOD_CLEAR_STATUS",
            "METHOD_AGENT_SWITCHED",
            "METHOD_MCP_OAUTH_REQUEST",
            "METHOD_MCP_SERVER_INITIALIZED",
            "METHOD_MCP_SERVER_INIT_FAILURE",
            "METHOD_SUBAGENT_LIST_UPDATE",
            "METHOD_REQUEST_PERMISSION",
            "METHOD_ELICITATION_CREATE",
            "METHOD_KAS_MCP_ELICITATION",
            "METHOD_KAS_USER_INPUT",
            "METHOD_KAS_MCP_STATUS",
            "METHOD_KAS_TOOLS_CHANGED",
            "METHOD_KAS_OPEN_EXTERNAL_URL",
        }
    }

    kas_kinds = {
        "context_usage": kas_wire_vals["KIND_CONTEXT_USAGE"],
        "turn_completion": kas_wire_vals["KIND_TURN_COMPLETION"],
        "summarization_started": kas_wire_vals["KIND_SUMMARIZATION_STARTED"],
        "summarization_completed": kas_wire_vals["KIND_SUMMARIZATION_COMPLETED"],
        "summarization_failed": kas_wire_vals["KIND_SUMMARIZATION_FAILED"],
        "steering_queued": kas_wire_vals["KIND_STEERING_QUEUED"],
        "steering_injected": kas_wire_vals["KIND_STEERING_INJECTED"],
        "steering_cleared": kas_wire_vals["KIND_STEERING_CLEARED"],
        "agent_subtask": kas_wire_vals["KIND_AGENT_SUBTASK"],
        "user_message_id_assigned": kas_wire_vals["KIND_USER_MESSAGE_ID_ASSIGNED"],
        "model_fallback": fallback_vals["KAS_KIND_MODEL_FALLBACK"],
    }

    return InstalledKiroCrewContract(
        kirocrew_version=str(init_vals["__version__"]),
        kas_pin_version=str(kas_pin_vals["KAS_PIN_VERSION"]),
        version_floors=floors,
        kiro_v2_protocol_version=kiro_harness_vals["PROTOCOL_VERSION"],
        kas_v3_protocol_version=kas_harness_vals["PROTOCOL_VERSION_KAS"],
        kas_relay_subcommand=str(kas_transport_vals["KAS_RELAY_SUBCMD"]),
        kas_relay_engine_flag=str(kas_transport_vals["KAS_RELAY_ENGINE_FLAG"]),
        kas_relay_engine=str(kas_transport_vals["KAS_RELAY_ENGINE"]),
        kas_relay_auth_flag=str(kas_transport_vals["KAS_RELAY_AUTH_FLAG"]),
        kas_auth_modes=(
            str(kas_transport_vals["KAS_RELAY_AUTH_OWNER"]),
            str(kas_transport_vals["KAS_RELAY_AUTH_CLIENT"]),
        ),
        kas_refusal_words=str(kas_refusal_vals["_KAS_ENGINE_REFUSAL_WORDS"]),
        kas_capabilities=(
            str(kas_wire_vals["CAPABILITY_CHECKPOINTS"]),
            str(kas_feat_vals["MCP_SERVER_OAUTH"]),
            str(kas_feat_vals["MCP_REGISTRY_POINTERS"]),
        ),
        kas_extension_methods=kas_extension_methods,
        kas_mcp_toggle_method=str(spawn_launch_vals["KAS_MCP_TOGGLE_METHOD"]),
        client_to_agent_methods=client_methods,
        kas_wire_kinds=kas_kinds,
        kas_credit_unit=str(kas_wire_vals["UNIT_CREDIT"]),
        v2_fallback_session_update=str(fallback_vals["V2_SESSION_UPDATE_MODEL_FALLBACK"]),
        kas_tool_family_by_kiro_name=dict(tool_names_vals["KAS_TOOL_FAMILY_BY_KIRO_CLI_NAME"]),
        allowed_tools_builtins=tuple(tool_names_vals["ALLOWED_TOOLS_BUILTINS"]),
        policy_alias_deny_only_ids=frozenset(tool_names_vals["POLICY_ALIAS_DENY_ONLY_IDS"]),
        crew_only_entry_keys=frozenset(kas_perm_vals["CREW_ONLY_ENTRY_KEYS"]),
    )


def _parse_kiro_cli_version_token(version_output: str) -> tuple[int, int, int] | None:
    """Replicate `kiro_crew.mcp_hot_reload.parse_kiro_cli_version` last-token parsing."""
    tokens = (version_output or "").strip().split()
    if not tokens:
        return None
    match = _KIRO_VERSION_RE.match(tokens[-1])
    if match is None:
        return None
    return (int(match.group(1)), int(match.group(2)), int(match.group(3) or 0))


def test_installed_kirocrew_version_floors_and_cli_version_probe(
    installed_contract: InstalledKiroCrewContract,
    run_cli: Callable[[list[str]], CliResult],
) -> None:
    """Verify `agy-acp --version` reports kiro-cli 3.5.0 and satisfies all KiroCrew floors."""
    assert KIRO_CLI_COMPAT_VERSION == "3.5.0"
    res = run_cli(["--version"])
    assert res.exit_code == 0

    parsed = _parse_kiro_cli_version_token(res.stdout)
    assert parsed == (3, 5, 0), f"Unexpected version output: {res.stdout!r}"

    for floor_name, floor_tuple in installed_contract.version_floors.items():
        assert parsed >= floor_tuple, f"{parsed} < {floor_name} ({floor_tuple})"


async def test_installed_kirocrew_cli_help_and_kas_start_verdict_probes(
    installed_contract: InstalledKiroCrewContract,
    run_cli: Callable[[list[str]], CliResult],
    tmp_path: Path,
) -> None:
    """Verify `acp --help`, `chat --help`, and `kas_engine_start_verdict` subprocess probes."""
    for subcmd in ([installed_contract.kas_relay_subcommand, "--help"], ["chat", "--help"]):
        help_res = run_cli(subcmd)
        assert help_res.exit_code == 0
        assert installed_contract.kas_relay_engine_flag in help_res.stdout
        assert installed_contract.kas_relay_engine in help_res.stdout

        at = help_res.stdout.find(installed_contract.kas_relay_engine_flag)
        assert at >= 0
        match = _POSSIBLE_VALUES_RE.search(help_res.stdout[at:])
        assert match is not None
        engines = {v.strip() for v in match.group(1).split(",") if v.strip()}
        assert {"v2", "v3"}.issubset(engines)

    probe_argvs = [
        [
            installed_contract.kas_relay_subcommand,
            installed_contract.kas_relay_engine_flag,
            installed_contract.kas_relay_engine,
            "--save-dir",
            str(tmp_path / "probe-v3"),
        ],
        *[
            [
                installed_contract.kas_relay_subcommand,
                installed_contract.kas_relay_engine_flag,
                installed_contract.kas_relay_engine,
                installed_contract.kas_relay_auth_flag,
                auth_mode,
                "--save-dir",
                str(tmp_path / f"probe-{auth_mode}"),
            ]
            for auth_mode in installed_contract.kas_auth_modes
        ],
        [
            installed_contract.kas_relay_subcommand,
            "--agent-engine",
            "v1",
            "--effort",
            "high",
            "--save-dir",
            str(tmp_path / "probe-v1"),
        ],
    ]

    for argv in probe_argvs:
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "agy_acp",
            *argv,
            cwd=str(tmp_path),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
            env={
                **os.environ,
                "PYTHONPATH": SRC_ROOT,
                "KIRO_CLI_CLIENT_APPLICATION": "kirocrew",
            },
        )
        _, stderr_bytes = await asyncio.wait_for(proc.communicate(), timeout=10.0)
        stderr_text = stderr_bytes.decode("utf-8", errors="replace")
        assert proc.returncode == 0, f"Probe {argv} failed (rc={proc.returncode}): {stderr_text}"
        assert installed_contract.kas_refusal_words not in stderr_text.casefold()


async def test_installed_kirocrew_v2_and_v3_initialize_and_extension_methods(
    installed_contract: InstalledKiroCrewContract,
    acp_harness_factory: HarnessFactory,
    tmp_path: Path,
) -> None:
    """Verify `KiroHarness` (v2 date protocol) and `KasHarness` (v3 int protocol) handshakes."""
    v2_harness = await acp_harness_factory.spawn(
        args=[
            installed_contract.kas_relay_subcommand,
            installed_contract.kas_relay_engine_flag,
            "v2",
            "--agent",
            "kirocrew",
            "--model",
            "gemini-3.7-pro",
            "--effort",
            "high",
            "--save-dir",
            str(tmp_path / "v2-store"),
        ],
        env={"KIRO_CLI_CLIENT_APPLICATION": "kirocrew", "PYTHONPATH": SRC_ROOT},
    )
    v2_init = await v2_harness.send_request(
        "initialize",
        {
            "protocolVersion": installed_contract.kiro_v2_protocol_version,
            "clientInfo": {"name": "kirocrew", "version": installed_contract.kirocrew_version},
            "clientCapabilities": {},
        },
    )
    assert "result" in v2_init, f"KiroHarness v2 initialize rejected: {v2_init}"
    assert v2_init["result"]["protocolVersion"] == 1
    assert v2_init["result"]["agentInfo"]["version"] == "3.5.0"
    assert v2_init["result"]["agentCapabilities"]["loadSession"] is True
    assert v2_init["result"]["agentCapabilities"]["promptCapabilities"]["image"] is True

    v3_harness = await acp_harness_factory.spawn(
        args=[
            installed_contract.kas_relay_subcommand,
            installed_contract.kas_relay_engine_flag,
            installed_contract.kas_relay_engine,
            installed_contract.kas_relay_auth_flag,
            "client",
            "--save-dir",
            str(tmp_path / "v3-store"),
        ],
        env={"KIRO_CLI_CLIENT_APPLICATION": "kirocrew", "PYTHONPATH": SRC_ROOT},
    )
    v3_init = await v3_harness.send_request(
        "initialize",
        {
            "protocolVersion": installed_contract.kas_v3_protocol_version,
            "clientInfo": {"name": "kirocrew", "version": installed_contract.kirocrew_version},
            "clientCapabilities": {
                "_meta": {"kiro": {"openExternalUrl": True, "secretStorage": True}}
            },
        },
    )
    assert "result" in v3_init, f"KasHarness v3 initialize rejected: {v3_init}"
    assert v3_init["result"]["protocolVersion"] == 1
    assert v3_init["result"]["agentInfo"]["version"] == "3.5.0"
    kiro_meta = v3_init["result"]["agentCapabilities"]["_meta"]["kiro"]
    assert kiro_meta["strictSessionLoad"] is True
    for cap_key in installed_contract.kas_capabilities:
        assert kiro_meta.get(cap_key) is True, f"Missing KAS capability {cap_key!r}"
    assert set(installed_contract.kas_extension_methods).issubset(
        set(kiro_meta["extensionMethods"])
    )


async def test_installed_kirocrew_wire_methods_tool_mappings_and_kas_discriminants(
    installed_contract: InstalledKiroCrewContract,
    tmp_path: Path,
) -> None:
    """Verify every client->agent RPC verb, tool family mapping, and KAS `_meta.kiro` kind."""
    notifications: list[dict[str, Any]] = []

    async def _sink(frame: dict[str, Any]) -> None:
        notifications.append(frame)

    store = SqliteStateStoreProvider(tmp_path / "contract.db")
    v3_adapter = V3KasWireAdapter()
    mgr = SessionManager(state_store=store, wire_adapter=v3_adapter)
    dispatcher = AcpDispatcher(
        notification_sink=_sink,
        session_manager=mgr,
        wire_adapter=v3_adapter,
    )

    async def _rpc(method: str, params: dict[str, Any], req_id: int = 1) -> dict[str, Any]:
        resp = await dispatcher.handle_message(
            json.dumps({"jsonrpc": "2.0", "id": req_id, "method": method, "params": params})
        )
        assert resp is not None, f"No response for {method}"
        err = resp.get("error")
        assert err is None or err.get("code") != -32601, f"Method not found for {method}: {resp}"
        return resp

    methods = installed_contract.client_to_agent_methods
    init_resp = await _rpc(
        methods["METHOD_INITIALIZE"],
        {"protocolVersion": installed_contract.kas_v3_protocol_version, "clientCapabilities": {}},
    )
    assert "result" in init_resp

    # Verify `_kiro/mcp/toggle` sent as JSON-RPC notification (no `id`) per `runtime_spawn_launch`
    notif_resp = await dispatcher.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "method": installed_contract.kas_mcp_toggle_method,
                "params": {"enabled": False},
            }
        )
    )
    assert notif_resp is None
    assert mgr.mcp_enabled is False
    toggle_resp = await _rpc(installed_contract.kas_mcp_toggle_method, {"enabled": True})
    assert "result" in toggle_resp
    assert mgr.mcp_enabled is True

    new_resp = await _rpc(
        methods["METHOD_SESSION_NEW"],
        {"cwd": str(tmp_path), "mcpServers": []},
    )
    sid = new_resp["result"]["sessionId"]
    dispatcher.pop_post_response_notifications()

    assert "result" in await _rpc(
        methods["METHOD_SESSION_LOAD"],
        {"sessionId": sid, "cwd": str(tmp_path), "mcpServers": []},
    )
    dispatcher.pop_post_response_notifications()

    assert "result" in await _rpc(
        methods["METHOD_SESSION_RESUME"],
        {"sessionId": sid, "cwd": str(tmp_path), "mcpServers": []},
    )
    dispatcher.pop_post_response_notifications()

    fork_resp = await _rpc(
        methods["METHOD_SESSION_FORK"],
        {"sessionId": sid, "cwd": str(tmp_path), "mcpServers": []},
    )
    fork_sid = fork_resp["result"]["sessionId"]
    dispatcher.pop_post_response_notifications()

    assert "result" in await _rpc(
        methods["METHOD_SET_MODE"],
        {"sessionId": sid, "modeId": "auto"},
    )
    assert "result" in await _rpc(
        methods["METHOD_SET_MODEL"],
        {"sessionId": sid, "modelId": "gemini-3.8-flash"},
    )
    assert "result" in await _rpc(
        methods["METHOD_SET_CONFIG_OPTION"],
        {"sessionId": sid, "configId": "effortLevel", "value": "xhigh"},
    )
    assert "result" in await _rpc(
        methods["METHOD_KAS_SESSION_CONTEXT"],
        {"sessionId": sid, "subcommand": "show"},
    )
    assert "result" in await _rpc(
        methods["METHOD_KAS_KNOWLEDGE"],
        {"sessionId": sid, "subcommand": "show"},
    )
    assert "result" in await _rpc(
        methods["METHOD_KAS_CODE_INTELLIGENCE"],
        {"sessionId": sid, "subcommand": "status"},
    )
    assert "result" in await _rpc(
        methods["METHOD_KAS_CHECKPOINT_REVERT"],
        {"sessionId": sid, "messageId": "nonexistent"},
    )
    assert "result" in await _rpc(
        methods["METHOD_KAS_SESSION_COMPACT"],
        {"sessionId": sid},
    )
    assert "result" in await _rpc(
        methods["METHOD_KAS_MCP_RESET_SERVER"],
        {"sessionId": sid, "serverName": "kirocrew-core"},
    )
    assert "result" in await _rpc(
        methods["METHOD_SESSION_STEERING"],
        {"sessionId": sid, "prompt": [{"type": "text", "text": "steer"}]},
    )
    assert "result" in await _rpc(
        methods["METHOD_SESSION_STEER_CLEAR"],
        {"sessionId": sid},
    )
    assert "result" in await _rpc(
        methods["METHOD_CANCEL"],
        {"sessionId": sid},
    )
    assert "result" in await _rpc(methods["METHOD_SESSION_CLOSE"], {"sessionId": fork_sid})

    sid_del = (await _rpc(methods["METHOD_SESSION_NEW"], {"cwd": str(tmp_path), "mcpServers": []}))[
        "result"
    ]["sessionId"]
    dispatcher.pop_post_response_notifications()
    assert "result" in await _rpc(methods["METHOD_KAS_SESSION_DELETE"], {"sessionId": sid_del})
    await mgr.close()

    # Verify V2-only `_kiro.dev/*` verbs (`commands/execute`, `session/terminate`)
    v2_mgr = SessionManager(state_store=store, wire_adapter=V2KiroWireAdapter())
    v2_disp = AcpDispatcher(
        notification_sink=_sink,
        session_manager=v2_mgr,
        wire_adapter=V2KiroWireAdapter(),
    )
    v2_new_raw = await v2_disp.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 20,
                "method": methods["METHOD_SESSION_NEW"],
                "params": {"cwd": str(tmp_path), "mcpServers": []},
            }
        )
    )
    assert v2_new_raw is not None and "result" in v2_new_raw
    v2_sid = v2_new_raw["result"]["sessionId"]
    v2_cmd_raw = await v2_disp.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 21,
                "method": methods["METHOD_COMMANDS_EXECUTE"],
                "params": {"sessionId": v2_sid, "command": "/usage"},
            }
        )
    )
    assert v2_cmd_raw is not None and "result" in v2_cmd_raw
    v2_term_raw = await v2_disp.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 22,
                "method": methods["METHOD_SESSION_TERMINATE"],
                "params": {"sessionId": v2_sid},
            }
        )
    )
    assert v2_term_raw is not None and "result" in v2_term_raw
    await v2_mgr.close()

    # Verify KAS `_meta.kiro.kind` discriminants & credit unit against `kas_wire.py`
    kinds = installed_contract.kas_wire_kinds
    telemetry = WireTurnTelemetry(
        context_used=125_000,
        context_size=1_000_000,
        context_usage_percentage=12.5,
        delta_credits=0.4,
    )
    telemetry_frames = v3_adapter.build_turn_telemetry_notifications(sid, telemetry)
    emitted_meta_list = [
        f["params"]["update"]["_meta"]["kiro"]
        for f in telemetry_frames
        if "_meta" in f["params"]["update"]
    ]
    assert any(m["kind"] == kinds["context_usage"] for m in emitted_meta_list)
    turn_comp = next(m for m in emitted_meta_list if m["kind"] == kinds["turn_completion"])
    assert turn_comp["promptTurnSummaries"][0]["unit"] == installed_contract.kas_credit_unit

    for phase, expected_kind in (
        ("started", kinds["summarization_started"]),
        ("completed", kinds["summarization_completed"]),
        ("failed", kinds["summarization_failed"]),
    ):
        comp_frame = v3_adapter.build_compaction_notification(sid, phase, summary="summary")
        assert comp_frame["params"]["update"]["_meta"]["kiro"]["kind"] == expected_kind

    for steer_phase, expected_kind in (
        ("queued", kinds["steering_queued"]),
        ("injected", kinds["steering_injected"]),
        ("cleared", kinds["steering_cleared"]),
    ):
        steer_frame = v3_adapter.build_steering_notification(sid, steer_phase, "note")
        assert steer_frame["params"]["update"]["_meta"]["kiro"]["kind"] == expected_kind

    msg_id_frames = v3_adapter.build_turn_start_notifications(sid, "msg-1")
    assert (
        msg_id_frames[0]["params"]["update"]["_meta"]["kiro"]["kind"]
        == kinds["user_message_id_assigned"]
    )

    sub_frames = v3_adapter.build_subagent_notifications(
        sid, [{"sessionId": "sub-1", "role": "worker", "status": "running"}]
    )
    assert sub_frames[0]["params"]["update"]["_meta"]["kiro"]["kind"] == kinds["agent_subtask"]

    v3_fb = v3_adapter.build_model_fallback_notifications(
        sid,
        from_model="gemini-3.7-pro",
        to_model="gemini-3.8-flash",
        trigger="availability",
        scope="session",
    )
    assert v3_fb[0]["params"]["update"]["_meta"]["kiro"]["kind"] == kinds["model_fallback"]

    v2_fb = V2KiroWireAdapter().build_model_fallback_notifications(
        sid,
        from_model="gemini-3.7-pro",
        to_model="gemini-3.8-flash",
        trigger="availability",
        scope="session",
    )
    assert (
        v2_fb[0]["params"]["update"]["sessionUpdate"]
        == installed_contract.v2_fallback_session_update
    )

    # Verify tool family mappings match `kiro_crew.platform.tool_names`
    sample_inputs = [
        ("run_command", {"CommandLine": "ls"}),
        ("read_file", {"Path": "/repo/a.py"}),
        ("list_dir", {"DirectoryPath": "/repo"}),
        ("write_file", {"Path": "/repo/a.py"}),
        ("edit_file", {"Path": "/repo/a.py"}),
        ("delete_file", {"Path": "/repo/a.py"}),
        ("grep_search", {"Query": "foo"}),
        ("find_by_name", {"Pattern": "*.py"}),
        ("search_web", {"query": "acp"}),
        ("read_url_content", {"Url": "https://example.com"}),
        ("invoke_subagent", {"Prompt": "check"}),
    ]
    families = installed_contract.kas_tool_family_by_kiro_name
    kiro_name_by_kas_id = {
        **dict.fromkeys(installed_contract.policy_alias_deny_only_ids, "fs_write"),
        **{kas_id: kiro_name for kiro_name, fam in families.items() for kas_id in fam},
        "run_command": "execute_bash",
    }
    allowed_builtins = set(installed_contract.allowed_tools_builtins)
    for agy_name, raw_in in sample_inputs:
        cls = classify_tool_call_step("", agy_name, raw_in)
        assert cls.v2_tool_name in allowed_builtins, (
            f"Unrecognized V2 tool {cls.v2_tool_name!r} (not in {allowed_builtins})"
        )
        policy_name = kiro_name_by_kas_id.get(cls.kas_builtin_id, cls.kas_builtin_id)
        assert policy_name == cls.v2_tool_name, (
            f"KAS tool {cls.kas_builtin_id!r} resolves to policy {policy_name!r}, "
            f"expected {cls.v2_tool_name!r}"
        )


def _import_installed_kirocrew_modules() -> dict[str, Any]:
    """Import live permission, identity, tool_paths, and security modules from `KiroCrew.app`."""
    import importlib

    pkg_dir = _resolve_kirocrew_pkg_dir()
    site_packages = str(pkg_dir.parent)
    if site_packages not in sys.path:
        sys.path.append(site_packages)
    return {
        "dispatch": importlib.import_module("kiro_crew.acp._dispatch"),
        "types": importlib.import_module("kiro_crew.acp.types"),
        "tool_identity": importlib.import_module("kiro_crew.hook_runtime.tool_identity"),
        "tool_input_denials": importlib.import_module(
            "kiro_crew.llm_helpers_parts.tool_input_denials"
        ),
        "tool_paths": importlib.import_module("kiro_crew.platform.tool_paths"),
        "security": importlib.import_module("kiro_crew.security"),
    }


def _roundtrip_permission_event(
    kc_mods: Mapping[str, Any],
    adapter: EngineWireAdapter,
    session_id: str,
    record: Mapping[str, Any],
    *,
    req_id: int = 1,
) -> Any:
    """Run `_build_tool_event_from_record` -> wire frames -> `kiro_crew` permission parser."""
    tc_id = str(record.get("toolCallId") or "tc-contract-1")
    _, _, tc_event = _build_tool_event_from_record(
        tc_id,
        record,
        known_mcp_servers=(),
    )
    norm_event = _to_normalized_tool_event(tc_event)
    tc_frame = adapter.build_tool_call_update(session_id, norm_event)
    perm_params = adapter.build_permission_request_params(session_id, norm_event)

    is_kas = adapter.engine_id == "v3"
    tool_input_cache: dict[str, str] = {}
    shell_cache: dict[str, bool] = {}
    raw_params_cache: dict[str, dict[str, Any]] = {}
    mcp_server_name_cache: dict[str, str] = {}
    tool_name_cache: dict[str, str] = {}
    tool_input_redacted_cache: dict[str, bool] = {}
    diff_path_cache: dict[str, str] = {}
    harness_tool_name_cache: dict[str, str] = {}
    tool_kind_cache: dict[str, str] | None = {} if is_kas else None
    cache_overflow: set[str] | None = set() if is_kas else None
    identity_unreadable_cache: dict[str, bool] | None = {} if is_kas else None

    dispatch_mod = kc_mods["dispatch"]
    dispatch_mod.parse_session_update(
        tc_frame["params"]["update"],
        tool_input_cache=tool_input_cache,
        shell_cache=shell_cache,
        raw_params_cache=raw_params_cache,
        mcp_server_name_cache=mcp_server_name_cache,
        tool_name_cache=tool_name_cache,
        cache_scope=session_id,
        tool_input_redacted_cache=tool_input_redacted_cache,
        diff_path_cache=diff_path_cache,
        harness_tool_name_cache=harness_tool_name_cache,
        tool_kind_cache=tool_kind_cache,
        cache_overflow=cache_overflow,
        identity_unreadable_cache=identity_unreadable_cache,
    )
    msg = kc_mods["types"].JsonRpcMessage(
        id=req_id,
        method="session/request_permission",
        params=perm_params,
    )
    perm_event, _ = dispatch_mod.build_permission_event(
        msg,
        tool_input_cache=tool_input_cache,
        shell_cache=shell_cache,
        raw_params_cache=raw_params_cache,
        mcp_server_name_cache=mcp_server_name_cache,
        tool_name_cache=tool_name_cache,
        cache_scope=session_id,
        tool_input_redacted_cache=tool_input_redacted_cache,
        diff_path_cache=diff_path_cache,
        harness_tool_name_cache=harness_tool_name_cache,
        tool_kind_cache=tool_kind_cache,
        identity_unreadable_cache=identity_unreadable_cache,
    )
    assert perm_event is not None
    return perm_event


@pytest.mark.parametrize(
    "adapter",
    [V3KasWireAdapter(), V2KiroWireAdapter()],
    ids=["v3_kas", "v2_kiro"],
)
def test_installed_kirocrew_long_shell_permission_title_and_git_publish_gate(
    adapter: EngineWireAdapter,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify >200-char `run_command` titles untrunate cleanly without backtick/80-char block."""
    kc = _import_installed_kirocrew_modules()
    monkeypatch.setenv("KIROCREW_HOME", str(tmp_path / "kc_home"))
    monkeypatch.setattr(kc["security"], "_emit_deny_event", lambda *a, **kw: None)
    monkeypatch.setattr(kc["security"], "_schedule_push_allow_audit", lambda *a, **kw: None)

    long_worktree = (
        "/Users/developer/.kiro/crew/workspace-member-agy-acp-bot/"
        "worktrees/agy-acp-bot/deeply-nested-feature-worktree-" + ("a" * 95)
    )
    feature_push_cmd = (
        f"git -C {long_worktree} push --force-with-lease origin crew/agy-acp-bot/feat-sync"
    )
    assert len(feature_push_cmd) > 200

    feat_perm = _roundtrip_permission_event(
        kc,
        adapter,
        "sess-shell-1",
        {
            "toolCallId": "tc-push-feat",
            "name": "run_command",
            "kind": "execute",
            "raw_args": {
                "CommandLine": feature_push_cmd,
                "toolSummary": "Push feature branch",
            },
        },
    )
    assert feat_perm.is_shell is True
    assert feat_perm.shell_command == feature_push_cmd
    assert feat_perm.title.startswith("Running: ")
    assert feat_perm.title.endswith("...")
    assert "`" not in feat_perm.title

    judged_feat = kc["tool_identity"].untruncated_shell_title(
        feat_perm.title,
        feat_perm.shell_command,
    )
    assert judged_feat == f"Running: {feature_push_cmd}"
    assert (
        kc["tool_input_denials"]._is_exempt_command_text(
            judged_feat,
            feat_perm.shell_command,
        )
        is True
    )
    assert kc["security"].is_denied(judged_feat) is None

    # Protected-branch push >200 chars must still hit gated protected-branch deny (not unverifiable)
    main_push_cmd = f"git -C {long_worktree} push origin main"
    assert len(main_push_cmd) > 200
    main_perm = _roundtrip_permission_event(
        kc,
        adapter,
        "sess-shell-2",
        {
            "toolCallId": "tc-push-main",
            "name": "run_command",
            "kind": "execute",
            "raw_args": {"CommandLine": main_push_cmd},
        },
    )
    judged_main = kc["tool_identity"].untruncated_shell_title(
        main_perm.title,
        main_perm.shell_command,
    )
    assert judged_main == f"Running: {main_push_cmd}"
    main_denial = kc["security"].is_denied(judged_main)
    assert main_denial is not None
    assert "git-publish-target-unverifiable" not in main_denial


def test_installed_kirocrew_document_body_keys_and_edit_target_candidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify >25 KiB `write_file` and `edit_file` bodies skip `command_shaped_strings`."""
    kc = _import_installed_kirocrew_modules()
    monkeypatch.setenv("KIROCREW_HOME", str(tmp_path / "kc_home"))

    adapter = V3KasWireAdapter()
    target_doc = str(tmp_path / "GUIDE.md")
    large_prose_body = (
        "# Deployment Guide\n\nExample forbidden command in docs: `git push origin main`\n"
        + ("Documentation paragraph explaining ACP wire framing.\n" * 520)
    )
    assert len(large_prose_body) > 25 * 1024

    write_perm = _roundtrip_permission_event(
        kc,
        adapter,
        "sess-doc-1",
        {
            "toolCallId": "tc-write-large",
            "name": "write_file",
            "kind": "edit",
            "raw_args": {
                "Path": target_doc,
                "Content": large_prose_body,
            },
        },
    )
    assert write_perm.raw_params_trusted is True
    write_scanned = kc["tool_paths"].command_shaped_strings(write_perm.raw_tool_params)
    assert not write_scanned.truncated
    assert large_prose_body not in write_scanned
    assert all(len(s) < 1024 for s in write_scanned)
    assert kc["tool_input_denials"]._first_tool_input_denial(write_scanned, None) is None

    write_targets = kc["tool_paths"].edit_target_candidates(
        write_perm.raw_tool_params,
        write_perm.diff_path,
        tool_kind=write_perm.tool_kind,
    )
    assert not write_targets.truncated
    assert not write_targets.unanchored
    assert list(write_targets) == [target_doc]

    target_mod = str(tmp_path / "module.py")
    old_snippet = "os.system('git push origin main')"
    new_snippet = "os.system('git push origin crew/bot/feat')"
    chunk_old = "subprocess.run(['kirocrew', 'restart'])"
    chunk_new = "subprocess.run(['kirocrew', 'status'])"

    edit_perm = _roundtrip_permission_event(
        kc,
        adapter,
        "sess-doc-2",
        {
            "toolCallId": "tc-edit-chunks",
            "name": "edit_file",
            "kind": "edit",
            "raw_args": {
                "TargetFile": target_mod,
                "Target": old_snippet,
                "Replacement": new_snippet,
                "ReplacementChunks": [
                    {
                        "TargetContent": chunk_old,
                        "ReplacementContent": chunk_new,
                        "AllowMultiple": False,
                    }
                ],
            },
        },
    )
    assert edit_perm.raw_params_trusted is True
    edit_scanned = kc["tool_paths"].command_shaped_strings(edit_perm.raw_tool_params)
    assert not edit_scanned.truncated
    for body_str in (old_snippet, new_snippet, chunk_old, chunk_new):
        assert body_str not in edit_scanned
    assert kc["tool_input_denials"]._first_tool_input_denial(edit_scanned, None) is None

    edit_targets = kc["tool_paths"].edit_target_candidates(
        edit_perm.raw_tool_params,
        edit_perm.diff_path,
        tool_kind=edit_perm.tool_kind,
    )
    assert not edit_targets.truncated
    assert not edit_targets.unanchored
    assert list(edit_targets) == [target_mod]


async def test_installed_kirocrew_0_13_0_shared_runtime_and_rejection_reason_contract(
    installed_contract: InstalledKiroCrewContract,
    tmp_path: Path,
) -> None:
    """Verify `0.13.0` shared-runtime `session/close` and KAS rejection-reason contract."""
    store = SqliteStateStoreProvider(tmp_path / "shared_contract.db")
    v3_adapter = V3KasWireAdapter()
    mgr = SessionManager(state_store=store, wire_adapter=v3_adapter)
    dispatcher = AcpDispatcher(session_manager=mgr, wire_adapter=v3_adapter)

    init_resp = await dispatcher.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": installed_contract.kas_v3_protocol_version,
                    "clientCapabilities": {},
                },
            }
        )
    )
    assert init_resp is not None and "result" in init_resp
    # KiroCrew 0.13.0 KasHarness.release() requires `sessionCapabilities.close` to keep records
    assert isinstance(
        init_resp["result"]["agentCapabilities"]["sessionCapabilities"].get("close"), dict
    )

    # Create two concurrent sessions on one shared dispatcher, close B, and verify B resumes
    # cleanly via `session/load` while A remains active.
    s_a = await dispatcher.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "session/new",
                "params": {"cwd": str(tmp_path / "a"), "mcpServers": []},
            }
        )
    )
    s_b = await dispatcher.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "session/new",
                "params": {"cwd": str(tmp_path / "b"), "mcpServers": []},
            }
        )
    )
    assert s_a is not None and s_b is not None
    sid_a = s_a["result"]["sessionId"]
    sid_b = s_b["result"]["sessionId"]

    close_b = await dispatcher.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "session/close",
                "params": {"sessionId": sid_b},
            }
        )
    )
    assert close_b is not None and "result" in close_b
    assert mgr.get_session(sid_a)["cwd"] == str(tmp_path / "a")

    load_b = await dispatcher.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 5,
                "method": "session/load",
                "params": {"sessionId": sid_b, "cwd": str(tmp_path / "b"), "mcpServers": []},
            }
        )
    )
    assert load_b is not None and "result" in load_b
    assert load_b["result"]["sessionId"] == sid_b

    # Verify KAS v3 permission params satisfy KiroCrew 0.13.0 `takes_rejection_reason` contract
    _, _, tc_event = _build_tool_event_from_record(
        "tc-rej-contract",
        {
            "toolCallId": "tc-rej-contract",
            "name": "run_command",
            "kind": "execute",
            "raw_args": {"CommandLine": "pytest"},
        },
        known_mcp_servers=(),
    )
    perm_params = v3_adapter.build_permission_request_params(
        sid_a,
        _to_normalized_tool_event(tc_event),
    )
    assert isinstance(perm_params.get("_meta", {}).get("kiro"), dict)
    option_kinds = {opt.get("kind"): opt.get("optionId") for opt in perm_params["options"]}
    assert option_kinds.get("reject_once") == "reject_once"
    await mgr.close()


async def test_installed_kirocrew_mcp_dashboard_stdio_probe_succeeds(
    installed_contract: InstalledKiroCrewContract,
    tmp_path: Path,
) -> None:
    """Verify `probe_session_mcp_servers` connects to real `kirocrew mcp-dashboard`."""
    _ = installed_contract
    pkg_dir = _resolve_kirocrew_pkg_dir()
    kirocrew_bin = pkg_dir.parents[3] / "bin" / "kirocrew"
    if not kirocrew_bin.is_file():
        pytest.skip(f"Installed kirocrew binary not found at {kirocrew_bin}")

    v3_adapter = V3KasWireAdapter()
    store = SqliteStateStoreProvider(tmp_path / "dash_probe.db")
    mgr = SessionManager(state_store=store, wire_adapter=v3_adapter)
    try:
        created = await mgr.create_session(
            cwd=str(tmp_path),
            mcp_servers=[
                {
                    "name": "kirocrew-dashboard",
                    "type": "stdio",
                    "command": str(kirocrew_bin),
                    "args": ["mcp-dashboard"],
                    "env": [
                        {"name": "KIROCREW_BOUND_PORT", "value": "5476"},
                        {"name": "KIROCREW_SESSION_KEY", "value": "dashboard:test-slot"},
                    ],
                    "_meta": {"kiro": {"versionNegotiation": "legacy"}},
                }
            ],
        )
        sid = str(created["sessionId"])
        statuses = await mgr.probe_session_mcp_servers(sid)
        assert len(statuses) == 1
        assert statuses[0].name == "kirocrew-dashboard"
        assert statuses[0].status == "connected", f"Probe failed: {statuses[0].error}"
        tool_names = {t.name for t in statuses[0].tools}
        assert "chat_folder_tree" in tool_names
    finally:
        await mgr.close()


async def test_installed_kirocrew_crew_only_keys_and_bare_401_reset_server_oauth(
    installed_contract: InstalledKiroCrewContract,
    tmp_path: Path,
) -> None:
    """Verify `CREW_ONLY_ENTRY_KEYS` superset parity and bare-401 `_kiro/mcp/resetServer` OAuth."""
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from agy_acp.logic.custom_agents import CREW_ONLY_ENTRY_KEYS

    assert installed_contract.crew_only_entry_keys
    assert installed_contract.crew_only_entry_keys <= CREW_ONLY_ENTRY_KEYS

    expected_auth_url = "https://oauth.example.com/authorize-mcp"

    class _Bare401McpHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            _ = (format, args)

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 0:
                _ = self.rfile.read(length)
            self.send_response(401)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error": "unauthorized"}')

        def do_GET(self) -> None:
            if self.path == "/.well-known/oauth-protected-resource":
                assert isinstance(self.server, HTTPServer)
                origin = f"http://127.0.0.1:{self.server.server_port}"
                payload = json.dumps(
                    {
                        "resource": f"{origin}/mcp",
                        "authorization_servers": [expected_auth_url],
                    }
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(payload)
                return
            self.send_response(404)
            self.end_headers()

    httpd = HTTPServer(("127.0.0.1", 0), _Bare401McpHandler)
    port = httpd.server_port
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    notifications: list[dict[str, Any]] = []
    client_rpcs: list[tuple[str, dict[str, Any]]] = []

    async def _sink(frame: dict[str, Any]) -> None:
        notifications.append(frame)

    async def _client_rpc(method: str, params: dict[str, Any]) -> dict[str, Any]:
        client_rpcs.append((method, params))
        return {}

    store = SqliteStateStoreProvider(tmp_path / "bare401_oauth.db")
    v3_adapter = V3KasWireAdapter()
    mgr = SessionManager(state_store=store, wire_adapter=v3_adapter)
    dispatcher = AcpDispatcher(
        notification_sink=_sink,
        client_rpc_caller=_client_rpc,
        session_manager=mgr,
        wire_adapter=v3_adapter,
    )
    try:
        init_resp = await dispatcher.handle_message(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": installed_contract.kas_v3_protocol_version,
                        "clientCapabilities": {
                            "_meta": {"kiro": {"openExternalUrl": True}},
                        },
                    },
                }
            )
        )
        assert init_resp is not None and "result" in init_resp

        new_resp = await dispatcher.handle_message(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "session/new",
                    "params": {
                        "cwd": str(tmp_path),
                        "mcpServers": [
                            {
                                "name": "stripe-like",
                                "url": f"http://127.0.0.1:{port}/mcp",
                            }
                        ],
                    },
                }
            )
        )
        assert new_resp is not None and "result" in new_resp
        sid = new_resp["result"]["sessionId"]
        dispatcher.pop_post_response_notifications()
        notifications.clear()

        reset_resp = await dispatcher.handle_message(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "_kiro/mcp/resetServer",
                    "params": {
                        "sessionId": sid,
                        "serverName": "stripe-like",
                        "startOAuth": True,
                    },
                }
            )
        )
        assert reset_resp is not None and "result" in reset_resp

        mcp_status_frames = [f for f in notifications if f.get("method") == "_kiro/mcp/status"]
        assert mcp_status_frames
        servers = mcp_status_frames[-1]["params"]["servers"]
        target = next(s for s in servers if s["name"] == "stripe-like")
        assert target["status"] == "authRequired"
        assert target["authUrl"] == expected_auth_url

        assert ("_kiro/openExternalUrl", {"url": expected_auth_url}) in client_rpcs
    finally:
        await mgr.close()
        httpd.shutdown()
        httpd.server_close()
