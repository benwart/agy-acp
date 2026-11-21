"""Custom agent spec loader, project security sanitizer, and KAS permissions evaluator."""

from __future__ import annotations

import contextlib
import json
import math
import os
import posixpath
import re
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml
from yaml.composer import ComposerError

AgentOrigin = Literal["client", "user", "project", "bundled"]
PermissionDecision = Literal["allow", "ask", "deny"]

JSON_SUFFIX = ".json"
MARKDOWN_SUFFIX = ".md"
AGENT_SPEC_SUFFIXES: tuple[str, ...] = (JSON_SUFFIX, MARKDOWN_SUFFIX)
NATIVE_SKILL_ALIAS_PREFIX = "_kirocrew_skill__"

#: Strictly honoured keys for untrusted project-scope `<cwd>/.kiro/agents/*` specs.
PROJECT_SPEC_HONOURED_KEYS: tuple[str, ...] = (
    "name",
    "description",
    "prompt",
    "tools",
    "excludedTools",
    "model",
)

#: Internal KiroCrew metadata keys stripped when parsing client wire `customAgents`.
CREW_ONLY_ENTRY_KEYS: frozenset[str] = frozenset(
    {
        "_crewAnsweredBuiltins",
        "_crewAnsweredCommands",
        "_crewAnsweredHookHeld",
        "_crewAnsweredWrites",
        "_crewSpecDenies",
        "_crewShellUnjudgedDenies",
        "_crewFolderPreToolHooks",
        "_crewShellAllowlist",
        "_crewWriteAllowlist",
        "_crewAnsweredPaths",
        "_crewAnsweredReadPaths",
        "_crewShellReadonly",
        "_crewSettingsMcp",
    }
)

_PERMISSION_BLOCK_KEYS: frozenset[str] = frozenset({"rules", "policies"})

#: Reserved KAS built-in agent IDs that KiroCrew projects with `-crewmate` or `crew-`.
KAS_RESERVED_AGENT_IDS: frozenset[str] = frozenset(
    {"default", "vibe", "spec", "quick-spec", "bug-fix", "plan", "autonomous"}
)
KAS_WIRE_ID_SUFFIX = "-crewmate"
KAS_WIRE_ID_PREFIX = "crew-"

_FRONTMATTER_CLOSE_RE = re.compile(r"^---[ \t]*\r?$", re.MULTILINE)

_BUILTIN_CAPABILITIES: tuple[str, ...] = (
    "fs_read",
    "fs_write",
    "shell",
    "web_fetch",
    "web_search",
    "subagent",
    "skill",
    "power",
    "context",
    "diagnostics",
)

KAS_CAPABILITIES: frozenset[str] = frozenset(
    {
        "all",
        "builtin",
        "filesystem",
        *_BUILTIN_CAPABILITIES,
        "mcp",
        "sandbox_network",
    }
)

META_CAPABILITY_EXPANSION: dict[str, tuple[str, ...]] = {
    "all": (*_BUILTIN_CAPABILITIES, "builtin", "mcp"),
    "builtin": (*_BUILTIN_CAPABILITIES, "builtin"),
    "filesystem": ("fs_read", "fs_write"),
}

_PATH_SCOPED_CAPABILITIES: frozenset[str] = frozenset({"fs_read", "fs_write", "mcp"})
_COMPOUND_SHELL_TOKENS: tuple[str, ...] = (";", "|", "&", "`", "$(", "\n", "\r", ">", "<")

_CAPABILITY_BY_TOOL_NAME: dict[str, str] = {
    **dict.fromkeys(
        (
            "read_file",
            "list_directory",
            "grep_search",
            "file_search",
            "fs_read",
            "read",
            "grep",
            "glob",
        ),
        "fs_read",
    ),
    **dict.fromkeys(
        ("fs_write", "str_replace", "fs_append", "delete_file", "write", "edit"),
        "fs_write",
    ),
    **dict.fromkeys(("run_command", "execute_bash", "shell", "bash"), "shell"),
    "web_fetch": "web_fetch",
    **dict.fromkeys(("remote_web_search", "web_search"), "web_search"),
    **dict.fromkeys(("invoke_sub_agent", "use_subagent", "subagent"), "subagent"),
    "skill": "skill",
    "power": "power",
    "context": "context",
    "diagnostics": "diagnostics",
}

_READ_ONLY_KIRO_TOOLS: frozenset[str] = frozenset(
    {"fs_read", "grep", "glob", "code", "web_search", "web_fetch", "tool_search", "todo_list"}
)


class CustomAgentSecurityError(ValueError):
    """Raised when an untrusted custom agent spec violates containment or security rules."""


class _FrontmatterSafeLoader(yaml.SafeLoader):
    """YAML SafeLoader that rejects `*alias` graph references and keeps ISO dates as strings."""

    def compose_node(self, parent: yaml.Node | None, index: Any) -> yaml.Node | None:  # type: ignore[override]
        if self.check_event(yaml.AliasEvent):
            event = self.peek_event()
            anchor = getattr(event, "anchor", "")
            raise ComposerError(
                None,
                None,
                f"aliases (*{anchor}) are not supported in agent frontmatter; "
                "repeat the value instead",
                event.start_mark,
            )
        return super().compose_node(parent, index)


def _construct_timestamp_as_text(loader: yaml.SafeLoader, node: yaml.Node) -> str:
    return str(loader.construct_scalar(node))  # type: ignore[arg-type]


_FrontmatterSafeLoader.add_constructor("tag:yaml.org,2002:timestamp", _construct_timestamp_as_text)


def _require_json_shape(value: Any, where: str) -> None:
    """Ensure `value` consists strictly of finite JSON-serializable types with `str` keys."""
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"frontmatter value at {where} is not a finite number")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"frontmatter key {key!r} at {where} is not a string; quote it")
            _require_json_shape(item, f"{where}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _require_json_shape(item, f"{where}[{index}]")
        return
    raise ValueError(
        f"frontmatter value at {where} is a YAML {type(value).__name__}, which has no JSON form"
    )


def opens_frontmatter_fence(text: str) -> bool:
    """Return True if `text` starts with a `---` YAML frontmatter line (after optional BOM)."""
    return text.removeprefix("\ufeff").startswith(("---\n", "---\r\n"))


def split_markdown_spec(text: str) -> tuple[str, str] | None:
    """Split a markdown agent spec into `(frontmatter_yaml, body)`, or `None` if unfenced."""
    if not opens_frontmatter_fence(text):
        return None
    text = text.removeprefix("\ufeff")
    first_newline = text.index("\n")
    close = _FRONTMATTER_CLOSE_RE.search(text, first_newline + 1)
    if close is None:
        return None
    frontmatter = text[first_newline + 1 : close.start()]
    body = text[close.end() :]
    body = body[2:] if body.startswith("\r\n") else body.removeprefix("\n")
    return frontmatter, body


def _load_frontmatter_yaml(text: str) -> Any:
    loader = _FrontmatterSafeLoader(text)
    try:
        return loader.get_single_data()
    finally:
        loader.dispose()


def parse_markdown_agent_spec(text: str) -> dict[str, Any]:
    """Parse a markdown YAML-frontmatter agent spec into a normalized JSON-shaped dict."""
    parts = split_markdown_spec(text)
    if parts is None:
        raise ValueError("markdown agent spec has no closed '---' frontmatter fence")
    frontmatter, body = parts
    try:
        loaded = _load_frontmatter_yaml(frontmatter) if frontmatter.strip() else {}
    except yaml.YAMLError as exc:
        raise ValueError(f"markdown agent spec frontmatter is not valid YAML: {exc}") from exc
    except RecursionError as exc:
        raise ValueError("markdown agent spec frontmatter is nested too deeply") from exc

    if loaded is None:
        loaded = {}
    if not isinstance(loaded, dict):
        raise ValueError("markdown agent spec frontmatter is not a mapping")

    try:
        _require_json_shape(loaded, "frontmatter")
    except RecursionError as exc:
        raise ValueError("markdown agent spec frontmatter is nested too deeply") from exc
    except ValueError as exc:
        raise ValueError(f"markdown agent spec {exc}") from exc

    spec: dict[str, Any] = dict(loaded)
    tools = spec.get("tools")
    if isinstance(tools, str) and tools.strip() != "*":
        entries = [t.strip() for t in tools.split(",") if t.strip()]
        if entries:
            spec["tools"] = entries
        else:
            spec.pop("tools", None)
    if body.strip():
        spec["prompt"] = body.lstrip("\r\n")
    return spec


def _reject_nonfinite_json_constant(token: str) -> Any:
    raise ValueError(f"invalid non-finite JSON constant: {token}")


def _parse_agent_spec_text(raw_text: str, suffix: str) -> dict[str, Any]:
    if suffix == MARKDOWN_SUFFIX:
        return parse_markdown_agent_spec(raw_text)
    if suffix == JSON_SUFFIX:
        try:
            loaded = json.loads(raw_text, parse_constant=_reject_nonfinite_json_constant)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON agent spec: {exc}") from exc
        if not isinstance(loaded, dict):
            raise ValueError("root must be a JSON object")
        return loaded
    raise ValueError(f"unsupported agent spec file extension: {suffix!r}")


def parse_agent_spec_file(path: Path) -> dict[str, Any]:
    """Parse either a `.json` or `.md` agent specification file from disk."""
    return _parse_agent_spec_text(path.read_text(encoding="utf-8"), path.suffix.lower())


_MAX_PROMPT_FILE_BYTES = 1024 * 1024


def read_contained_regular_bytes(
    src_path: Path,
    *,
    root_dir: Path,
    max_bytes: int,
    allowed_suffixes: frozenset[str] | set[str] | None = None,
) -> tuple[Path, bytes] | None:
    """Atomically read a regular non-symlink file contained inside `root_dir` up to `max_bytes`."""
    try:
        resolved_root = root_dir.resolve()
        candidate = src_path if src_path.is_absolute() else (resolved_root / src_path)
        rel_check = (
            resolved_root / src_path.relative_to(root_dir)
            if src_path.is_absolute() and src_path.is_relative_to(root_dir)
            else candidate
        )
        if candidate.is_symlink() or rel_check.is_symlink():
            return None
        resolved_target = candidate.resolve(strict=True)
        if resolved_target == resolved_root or not resolved_target.is_relative_to(resolved_root):
            return None
        if allowed_suffixes is not None and resolved_target.suffix.lower() not in allowed_suffixes:
            return None
        if rel_check.is_relative_to(resolved_root) and any(
            p.is_symlink()
            for p in rel_check.parents
            if p != resolved_root and p.is_relative_to(resolved_root)
        ):
            return None
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        fd = os.open(resolved_target, flags)
        try:
            st = os.fstat(fd)
            if not stat.S_ISREG(st.st_mode) or st.st_size > max_bytes:
                return None
            chunks: list[bytes] = []
            total = 0
            while True:
                buf = os.read(fd, min(65536, max_bytes + 1 - total))
                if not buf:
                    break
                total += len(buf)
                if total > max_bytes:
                    return None
                chunks.append(buf)
        finally:
            os.close(fd)
        return resolved_target, b"".join(chunks)
    except OSError, RuntimeError:
        return None


def resolve_contained_prompt_file(prompt_value: str, *, base_dir: Path) -> str:
    """Resolve a `file://` prompt strictly contained within `base_dir` (no `..` or symlinks out)."""
    if not prompt_value.startswith("file://"):
        return prompt_value

    raw_ref = prompt_value[len("file://") :]
    if not raw_ref or raw_ref.startswith(("~", "/")) or Path(raw_ref).expanduser().anchor:
        raise CustomAgentSecurityError(
            f"Refusing rooted or home-anchored file:// prompt reference: {prompt_value!r}"
        )

    loaded = read_contained_regular_bytes(
        Path(raw_ref),
        root_dir=base_dir,
        max_bytes=_MAX_PROMPT_FILE_BYTES,
    )
    if loaded is None:
        raise CustomAgentSecurityError(
            f"Refusing unsafe, missing, or oversized file:// prompt reference: {prompt_value!r}"
        )
    try:
        return loaded[1].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CustomAgentSecurityError(
            f"Prompt file reference is not valid UTF-8: {prompt_value!r}"
        ) from exc


def _resolve_spec_prompt_field(spec: dict[str, Any], *, base_dir: Path) -> None:
    prompt_val = spec.get("prompt")
    if isinstance(prompt_val, str) and prompt_val.startswith("file://"):
        spec["prompt"] = resolve_contained_prompt_file(prompt_val, base_dir=base_dir)


def sanitize_project_agent_spec(
    spec: Mapping[str, Any],
    *,
    agents_dir: Path,
) -> dict[str, Any]:
    """Strip untrusted keys from a project-scope spec and enforce `file://` prompt containment."""
    sanitized: dict[str, Any] = {
        key: spec[key] for key in PROJECT_SPEC_HONOURED_KEYS if key in spec
    }
    _resolve_spec_prompt_field(sanitized, base_dir=agents_dir)
    return sanitized


def _resolve_safe_project_agents_dir(cwd: str | Path | None) -> tuple[Path, Path] | None:
    """Return `(project_agents_dir, resolved_project_agents_dir)` when safe and non-symlinked."""
    if not cwd:
        return None
    with contextlib.suppress(OSError, RuntimeError):
        cwd_path = Path(cwd)
        kiro_dir = cwd_path / ".kiro"
        project_dir = kiro_dir / "agents"
        if kiro_dir.is_symlink() or project_dir.is_symlink() or not project_dir.is_dir():
            return None
        resolved_proj = project_dir.resolve()
        if resolved_proj.is_relative_to(cwd_path.resolve()):
            return project_dir, resolved_proj
    return None


def _iter_live_spec_files(directory: Path) -> list[Path]:
    """Return sorted `.json` and `.md` agent spec files in `directory`, `.json` beating `.md`."""
    if not directory.is_dir():
        return []
    try:
        with os.scandir(directory) as entries:
            names = [
                e.name for e in entries if not e.name.startswith((".", NATIVE_SKILL_ALIAS_PREFIX))
            ]
    except OSError:
        return []

    json_files = [directory / n for n in names if n.lower().endswith(JSON_SUFFIX)]
    md_files = [directory / n for n in names if n.lower().endswith(MARKDOWN_SUFFIX)]
    json_stems = {p.stem.lower() for p in json_files}
    live = [
        *json_files,
        *(
            md
            for md in md_files
            if md.stem.lower() not in json_stems
            and not (directory / f"{md.stem}{JSON_SUFFIX}").exists()
        ),
    ]
    return sorted(live)


def _normalize_str_tuple(raw: Any) -> tuple[str, ...]:
    if isinstance(raw, (list, tuple)):
        return tuple(str(x).strip() for x in raw if isinstance(x, str) and x.strip())
    if isinstance(raw, str) and raw.strip():
        return tuple(part.strip() for part in raw.split(",") if part.strip())
    return ()


@dataclass(frozen=True, slots=True)
class CustomAgentSpec:
    """Normalized custom agent definition from client wire, user disk, or project disk."""

    agent_id: str
    name: str
    description: str = ""
    prompt: str = ""
    origin: AgentOrigin = "client"
    model: str | None = None
    tools: tuple[str, ...] | None = None
    excluded_tools: tuple[str, ...] = ()
    allowed_tools: tuple[str, ...] = ()
    permissions: dict[str, Any] | list[Any] | None = None
    mcp_servers: dict[str, Any] = field(default_factory=dict)
    raw_spec: dict[str, Any] = field(default_factory=dict)

    def evaluate_permission(self, *, capability: str, resource: str = "") -> PermissionDecision:
        """Evaluate KAS inline permissions for `(capability, resource)`."""
        return evaluate_kas_permissions(self.permissions, capability=capability, resource=resource)

    def evaluate_tool_call(
        self,
        *,
        tool_id: str = "",
        is_mcp: bool = False,
        server_name: str = "",
        tool_name: str = "",
        command: str = "",
        path: str = "",
        url: str = "",
        query: str = "",
        subagent_target: str = "",
        cwd: str = "",
    ) -> PermissionDecision:
        """Evaluate KAS inline permissions for a concrete tool invocation."""
        return evaluate_tool_call_permission(
            self.permissions,
            tool_id=tool_id,
            is_mcp=is_mcp,
            server_name=server_name,
            tool_name=tool_name,
            command=command,
            path=path,
            url=url,
            query=query,
            subagent_target=subagent_target,
            cwd=cwd,
        )

    def to_available_mode(self, *, v3_meta: bool = True) -> dict[str, Any]:
        """Project this custom agent as an ACP `availableModes` entry."""
        entry: dict[str, Any] = {
            "id": self.agent_id,
            "name": self.name or self.agent_id,
            "description": self.description or self.name or self.agent_id,
        }
        if v3_meta:
            entry["_meta"] = {"kiro": {"resource": {"source": {"origin": self.origin}}}}
        return entry


def _spec_from_mapping(
    raw: Mapping[str, Any],
    *,
    origin: AgentOrigin,
    fallback_id: str = "",
) -> CustomAgentSpec | None:
    cleaned = {k: v for k, v in raw.items() if k not in CREW_ONLY_ENTRY_KEYS}
    raw_id = cleaned.get("id") or fallback_id or cleaned.get("name")
    if not isinstance(raw_id, str) or not raw_id.strip():
        return None
    agent_id = raw_id.strip()
    raw_name = cleaned.get("name")
    name = raw_name.strip() if isinstance(raw_name, str) and raw_name.strip() else agent_id
    desc_val = cleaned.get("description")
    prompt_val = (
        cleaned["systemPrompt"]
        if isinstance(cleaned.get("systemPrompt"), str)
        else cleaned.get("prompt")
    )
    model_val = cleaned.get("model")
    tools_raw = cleaned.get("tools")
    perms_raw = cleaned.get("permissions")
    mcp_raw = cleaned.get("mcpServers")

    return CustomAgentSpec(
        agent_id=agent_id,
        name=name,
        description=desc_val if isinstance(desc_val, str) else "",
        prompt=prompt_val if isinstance(prompt_val, str) else "",
        origin=origin,
        model=model_val.strip() if isinstance(model_val, str) and model_val.strip() else None,
        tools=_normalize_str_tuple(tools_raw) if tools_raw is not None else None,
        excluded_tools=_normalize_str_tuple(cleaned.get("excludedTools")),
        allowed_tools=_normalize_str_tuple(cleaned.get("allowedTools")),
        permissions=perms_raw if isinstance(perms_raw, (dict, list)) else None,
        mcp_servers=dict(mcp_raw) if isinstance(mcp_raw, dict) else {},
        raw_spec=cleaned,
    )


def strip_kas_wire_agent_affix(agent_id: str) -> str:
    """Map KAS wire-escaped IDs (`<id>-crewmate` or `crew-<id>`) back to base ID."""
    if agent_id.endswith(KAS_WIRE_ID_SUFFIX) and len(agent_id) > len(KAS_WIRE_ID_SUFFIX):
        return agent_id[: -len(KAS_WIRE_ID_SUFFIX)]
    if agent_id.startswith(KAS_WIRE_ID_PREFIX) and len(agent_id) > len(KAS_WIRE_ID_PREFIX):
        return agent_id[len(KAS_WIRE_ID_PREFIX) :]
    return agent_id


def to_kas_wire_agent_id(agent_id: str) -> str:
    """Escape KAS reserved built-in agent IDs with `-crewmate` for wire transport."""
    if agent_id in KAS_RESERVED_AGENT_IDS or agent_id.endswith(KAS_WIRE_ID_SUFFIX):
        return f"{agent_id}{KAS_WIRE_ID_SUFFIX}"
    return agent_id


class CustomAgentRegistry:
    """Registry of custom agents merged across project, user, and client wire scopes."""

    def __init__(self, agents: Sequence[CustomAgentSpec] = ()) -> None:
        self._by_id: dict[str, CustomAgentSpec] = {spec.agent_id: spec for spec in agents}

    @classmethod
    def load(
        cls,
        *,
        cwd: str | Path | None = None,
        wire_custom_agents: Sequence[Mapping[str, Any]] | None = None,
        user_agents_dir: Path | None = None,
    ) -> CustomAgentRegistry:
        """Load custom agents with precedence `client > user > project`."""
        merged: dict[str, CustomAgentSpec] = {}

        # 1. Lowest precedence: Project scope `<cwd>/.kiro/agents/*.{json,md}` (UNTRUSTED)
        safe_proj = _resolve_safe_project_agents_dir(cwd)
        if safe_proj is not None:
            project_dir, resolved_project_dir = safe_proj
            for spec_path in _iter_live_spec_files(project_dir):
                if spec_path.is_symlink():
                    continue
                with contextlib.suppress(OSError, CustomAgentSecurityError):
                    if not spec_path.resolve().is_relative_to(resolved_project_dir):
                        continue
                    loaded = cls._read_disk_spec(spec_path)
                    if loaded is not None:
                        sanitized = sanitize_project_agent_spec(loaded, agents_dir=project_dir)
                        agent = _spec_from_mapping(
                            sanitized, origin="project", fallback_id=spec_path.stem
                        )
                        if agent is not None:
                            merged[agent.agent_id] = agent

        # 2. Middle precedence: User scope `~/.kiro/agents/*.{json,md}`
        resolved_user_dir = (
            user_agents_dir if user_agents_dir is not None else Path.home() / ".kiro" / "agents"
        )
        for spec_path in _iter_live_spec_files(resolved_user_dir):
            loaded = cls._read_disk_spec(spec_path)
            if loaded is None:
                continue
            user_dict = dict(loaded)
            try:
                _resolve_spec_prompt_field(user_dict, base_dir=resolved_user_dir)
            except CustomAgentSecurityError:
                continue
            agent = _spec_from_mapping(user_dict, origin="user", fallback_id=spec_path.stem)
            if agent is not None:
                merged[agent.agent_id] = agent

        # 3. Highest precedence: Wire `params._meta.kiro.customAgents` (`origin="client"`)
        for raw_entry in wire_custom_agents or ():
            if isinstance(raw_entry, Mapping):
                agent = _spec_from_mapping(raw_entry, origin="client")
                if agent is not None:
                    merged[agent.agent_id] = agent

        return cls(list(merged.values()))

    @staticmethod
    def _read_disk_spec(path: Path) -> dict[str, Any] | None:
        with contextlib.suppress(OSError, ValueError):
            raw_text = path.read_text(encoding="utf-8")
            suffix = path.suffix.lower()
            if suffix == MARKDOWN_SUFFIX and not opens_frontmatter_fence(raw_text):
                return None
            return _parse_agent_spec_text(raw_text, suffix)
        return None

    def get(self, agent_id: str) -> CustomAgentSpec | None:
        """Look up an agent by exact wire ID, stripped crewmate ID, or display name."""
        for candidate in (
            agent_id,
            strip_kas_wire_agent_affix(agent_id),
            f"{agent_id}{KAS_WIRE_ID_SUFFIX}",
            f"{KAS_WIRE_ID_PREFIX}{agent_id}",
        ):
            if candidate in self._by_id:
                return self._by_id[candidate]
        return next((spec for spec in self._by_id.values() if spec.name == agent_id), None)

    def list_agents(self) -> list[CustomAgentSpec]:
        """Return all registered custom agents in insertion/precedence order."""
        return list(self._by_id.values())

    def to_available_modes(self, *, v3_meta: bool = True) -> list[dict[str, Any]]:
        """Project all registered custom agents into ACP `availableModes` entries."""
        return [spec.to_available_mode(v3_meta=v3_meta) for spec in self._by_id.values()]


def _compile_glob_regex(pattern: str, *, path_mode: bool) -> re.Pattern[str]:
    """Compile a KAS permission glob pattern into an anchored regular expression."""
    parts: list[str] = ["^"]
    idx = 0
    length = len(pattern)
    while idx < length:
        if pattern.startswith("**/", idx):
            parts.append("(?:.+/)?")
            idx += 3
        elif pattern.startswith("/**", idx):
            parts.append("(?:/.*)?")
            idx += 3
        elif pattern.startswith("**", idx):
            parts.append(".*")
            idx += 2
        elif pattern[idx] == "*":
            parts.append("[^/]*" if path_mode else ".*")
            idx += 1
        elif pattern[idx] == "?":
            parts.append("[^/]" if path_mode else ".")
            idx += 1
        else:
            parts.append(re.escape(pattern[idx]))
            idx += 1
    parts.append("$")
    return re.compile("".join(parts), re.DOTALL)


def _matches_any_pattern(patterns: Sequence[str], value: str, *, path_mode: bool) -> bool:
    return any(_compile_glob_regex(pat, path_mode=path_mode).match(value) for pat in patterns)


def _extract_non_empty_strs(raw: Any) -> list[str]:
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        return [str(p) for p in raw if isinstance(p, str) and p.strip()]
    return []


def _normalize_permission_rules(raw_permissions: Any) -> list[dict[str, Any]]:
    """Normalize `{"rules": [...], "policies": [...]}` or `[...]` permissions into rules."""
    raw_list = (
        raw_permissions.get("rules") if isinstance(raw_permissions, Mapping) else raw_permissions
    )
    if not isinstance(raw_list, Sequence) or isinstance(raw_list, (str, bytes)):
        return []

    normalized: list[dict[str, Any]] = []
    for entry in raw_list:
        if not isinstance(entry, Mapping):
            continue
        effect = entry.get("effect") or entry.get("permission")
        if not isinstance(effect, str) or effect not in ("allow", "ask", "deny"):
            continue

        match_patterns = _extract_non_empty_strs(entry.get("match"))
        exclude_patterns = _extract_non_empty_strs(entry.get("exclude"))
        capability = entry.get("capability")
        if isinstance(capability, str) and capability in KAS_CAPABILITIES:
            normalized.append(
                {
                    "capability": capability,
                    "match": match_patterns,
                    "exclude": exclude_patterns,
                    "effect": effect,
                }
            )
            continue

        tool_ref = entry.get("tool")
        if not isinstance(tool_ref, str) or not tool_ref.strip():
            continue
        tool_ref = tool_ref.strip()
        if tool_ref == "*":
            resolved_cap, resolved_match = "all", match_patterns
        elif tool_ref.startswith("@"):
            mcp_target = tool_ref[1:] if "/" in tool_ref[1:] else f"{tool_ref[1:]}/*"
            resolved_cap, resolved_match = "mcp", [mcp_target, *match_patterns]
        else:
            mapped_cap = _CAPABILITY_BY_TOOL_NAME.get(tool_ref)
            if mapped_cap is None:
                continue
            resolved_cap = mapped_cap
            resolved_match = [*match_patterns, *_extract_non_empty_strs(entry.get("commands"))]
        normalized.append(
            {
                "capability": resolved_cap,
                "match": resolved_match,
                "exclude": exclude_patterns,
                "effect": effect,
            }
        )
    return normalized


def _has_compound_shell_syntax(command: str) -> bool:
    return any(tok in command for tok in _COMPOUND_SHELL_TOKENS)


def _normalize_fs_path_candidates(path: str, *, cwd: str = "") -> tuple[str, ...]:
    """Compute canonical normalized POSIX path candidates (absolute and `cwd`-relative)."""
    if not path:
        return ("",)
    raw_posix = path.replace("\\", "/")
    cwd_clean = cwd.strip()
    if cwd_clean:
        cwd_path = Path(cwd_clean).resolve(strict=False)
        raw_p = Path(raw_posix)
        target_path = (raw_p if raw_p.is_absolute() else (cwd_path / raw_p)).resolve(strict=False)
        candidates: list[str] = [target_path.as_posix()]
        if target_path == cwd_path:
            candidates.append(".")
        elif target_path.is_relative_to(cwd_path):
            rel_posix = target_path.relative_to(cwd_path).as_posix()
            if rel_posix not in candidates:
                candidates.append(rel_posix)
        return tuple(candidates)

    norm = posixpath.normpath(raw_posix)
    return ("/" + norm.lstrip("/") if norm.startswith("//") else norm,)


def _validate_str_list_field(value: Any, field_name: str) -> None:
    if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
        raise ValueError(f"{field_name!r} must be a list of strings")


def _validate_permission_rule_entry(entry: Any, index: int) -> None:
    if not isinstance(entry, Mapping):
        raise ValueError(f"permission rule at index {index} must be an object")
    effect = entry.get("effect") if "effect" in entry else entry.get("permission")
    if not isinstance(effect, str) or effect not in ("allow", "ask", "deny"):
        raise ValueError(
            f"permission rule at index {index} has invalid effect {effect!r}; "
            "expected 'allow', 'ask', or 'deny'"
        )
    has_cap = entry.get("capability") is not None
    has_tool = entry.get("tool") is not None
    if not has_cap and not has_tool:
        raise ValueError(f"permission rule at index {index} must specify 'capability' or 'tool'")
    if has_cap and (
        not isinstance(entry["capability"], str) or entry["capability"] not in KAS_CAPABILITIES
    ):
        raise ValueError(
            f"permission rule at index {index} has unknown capability {entry['capability']!r}"
        )
    if has_tool and (not isinstance(entry["tool"], str) or not entry["tool"].strip()):
        raise ValueError(f"permission rule at index {index} must have a non-empty 'tool'")
    for list_key in ("match", "exclude", "commands"):
        if entry.get(list_key) is not None:
            _validate_str_list_field(entry[list_key], f"permissions[{index}].{list_key}")


def _validate_permissions_structure(raw_permissions: Any) -> None:
    if raw_permissions is None or raw_permissions == {}:
        return
    if isinstance(raw_permissions, dict):
        unknown_keys = set(raw_permissions.keys()) - _PERMISSION_BLOCK_KEYS
        if unknown_keys:
            raise ValueError(f"unsupported keys in 'permissions': {sorted(unknown_keys)!r}")
        if "rules" not in raw_permissions and "policies" not in raw_permissions:
            raise ValueError("'permissions' object must contain 'rules' or 'policies'")
        if "policies" in raw_permissions and raw_permissions["policies"] is not None:
            _validate_str_list_field(raw_permissions["policies"], "permissions.policies")
        raw_rules = raw_permissions.get("rules")
        if raw_rules is None:
            rules: list[Any] = []
        elif isinstance(raw_rules, list):
            rules = raw_rules
        else:
            raise ValueError("'permissions.rules' must be a list")
    elif isinstance(raw_permissions, list):
        rules = raw_permissions
    else:
        raise ValueError("'permissions' must be an object or list")

    for idx, entry in enumerate(rules):
        _validate_permission_rule_entry(entry, idx)


def validate_agent_spec_file(path: Path) -> CustomAgentSpec:
    """Validate syntax, `file://` prompt containment, and semantic schema of an agent spec file."""
    loaded = parse_agent_spec_file(path)
    base_dir = path.parent

    for prompt_key in ("prompt", "systemPrompt"):
        if loaded.get(prompt_key) is not None:
            val = loaded[prompt_key]
            if not isinstance(val, str):
                raise ValueError(f"{prompt_key!r} must be a string")
            if val.startswith("file://"):
                resolve_contained_prompt_file(val, base_dir=base_dir)

    _validate_permissions_structure(loaded.get("permissions"))

    if loaded.get("mcpServers") is not None and not isinstance(loaded["mcpServers"], dict):
        raise ValueError("'mcpServers' must be an object")

    for tool_list_key in ("tools", "excludedTools", "allowedTools"):
        if loaded.get(tool_list_key) is not None:
            raw_tools = loaded[tool_list_key]
            if not isinstance(raw_tools, str) and (
                not isinstance(raw_tools, list) or not all(isinstance(x, str) for x in raw_tools)
            ):
                raise ValueError(f"{tool_list_key!r} must be a string or list of strings")

    spec = _spec_from_mapping(loaded, origin="user", fallback_id=path.stem)
    if spec is None:
        raise ValueError("agent spec must define a non-empty 'id' or 'name'")
    return spec


def _evaluate_kas_permissions_for_candidates(
    permissions: Any,
    *,
    capability: str,
    candidates: Sequence[str],
) -> PermissionDecision:
    rules = _normalize_permission_rules(permissions)
    if not rules:
        return "ask"

    path_mode = capability in _PATH_SCOPED_CAPABILITIES
    saw_ask = False
    saw_allow = False

    for rule in rules:
        expanded = META_CAPABILITY_EXPANSION.get(rule["capability"], (rule["capability"],))
        if capability not in expanded:
            continue

        effect = rule["effect"]
        match_list: list[str] = rule["match"]
        if match_list:
            effective_matches = match_list
            if (
                capability == "shell"
                and effect == "allow"
                and any(_has_compound_shell_syntax(c) for c in candidates)
            ):
                effective_matches = [p for p in match_list if "*" not in p and "?" not in p]
                if not effective_matches:
                    continue
            if not any(
                _matches_any_pattern(effective_matches, cand, path_mode=path_mode)
                for cand in candidates
            ):
                continue

        exclude_list: list[str] = rule["exclude"]
        if exclude_list and any(
            _matches_any_pattern(exclude_list, cand, path_mode=path_mode) for cand in candidates
        ):
            continue

        if effect == "deny":
            return "deny"
        if effect == "ask":
            saw_ask = True
        elif effect == "allow":
            saw_allow = True

    return "ask" if saw_ask else ("allow" if saw_allow else "ask")


def evaluate_kas_permissions(
    permissions: Any,
    *,
    capability: str,
    resource: str = "",
) -> PermissionDecision:
    """Evaluate KAS permissions with Cedar most-restrictive-match (`deny > ask > allow`).

    Unmatched requests resolve to `"ask"`.
    """
    candidates = (
        _normalize_fs_path_candidates(resource, cwd="")
        if capability in ("fs_read", "fs_write")
        else (resource,)
    )
    return _evaluate_kas_permissions_for_candidates(
        permissions,
        capability=capability,
        candidates=candidates,
    )


def evaluate_tool_call_permission(
    permissions: Any,
    *,
    tool_id: str = "",
    is_mcp: bool = False,
    server_name: str = "",
    tool_name: str = "",
    command: str = "",
    path: str = "",
    url: str = "",
    query: str = "",
    subagent_target: str = "",
    cwd: str = "",
) -> PermissionDecision:
    """Map a tool invocation to `(capability, resource)` and evaluate KAS permissions."""
    if is_mcp or (server_name and tool_name):
        return evaluate_kas_permissions(
            permissions,
            capability="mcp",
            resource=f"{server_name}/{tool_name}",
        )

    capability = _CAPABILITY_BY_TOOL_NAME.get(tool_id)
    if capability is None:
        return evaluate_kas_permissions(permissions, capability="builtin", resource=tool_id)

    if capability in ("fs_read", "fs_write"):
        return _evaluate_kas_permissions_for_candidates(
            permissions,
            capability=capability,
            candidates=_normalize_fs_path_candidates(path, cwd=cwd),
        )

    resource_by_cap = {
        "shell": command,
        "web_fetch": url,
        "web_search": query,
        "subagent": subagent_target,
    }
    return evaluate_kas_permissions(
        permissions,
        capability=capability,
        resource=resource_by_cap.get(capability, ""),
    )


def load_kiro_agent_spec(
    agent_name: str | None = None,
    *,
    cwd: str | None = None,
    include_home: bool = True,
    user_agents_dir: Path | None = None,
) -> dict[str, Any]:
    """Load sanitized project agent fields and trusted user-scope `mcpServers`."""
    candidates: list[str] = []
    if agent_name and agent_name not in {"auto", "approve", "read-only"}:
        candidates.append(agent_name)
    if "kirocrew" not in candidates:
        candidates.append("kirocrew")

    search_dirs: list[tuple[Path, bool, Path]] = []
    safe_proj = _resolve_safe_project_agents_dir(cwd)
    if safe_proj is not None:
        search_dirs.append((safe_proj[0], True, safe_proj[1]))

    trusted_user_dir: Path | None = None
    if user_agents_dir is not None:
        trusted_user_dir = Path(user_agents_dir)
    elif include_home:
        with contextlib.suppress(RuntimeError, OSError):
            trusted_user_dir = Path.home() / ".kiro" / "agents"

    if trusted_user_dir is not None:
        with contextlib.suppress(OSError, RuntimeError):
            if not trusted_user_dir.is_symlink() and trusted_user_dir.is_dir():
                search_dirs.append((trusted_user_dir, False, trusted_user_dir.resolve()))

    pinned_model: str | None = None
    agent_prompt: str | None = None
    agent_tools: list[str] | None = None
    servers_by_name: dict[str, dict[str, Any]] = {}
    disabled_names: list[str] = []

    for cand in candidates:
        if "/" in cand or "\\" in cand or ".." in cand:
            continue
        found_candidate = False
        for agents_dir, is_project_scope, containment_root in search_dirs:
            spec_path = agents_dir / f"{cand}.json"
            if spec_path.is_symlink() or not spec_path.is_file():
                continue
            with contextlib.suppress(
                OSError, json.JSONDecodeError, CustomAgentSecurityError, ValueError
            ):
                resolved_spec = spec_path.resolve(strict=True)
                if not resolved_spec.is_relative_to(containment_root):
                    continue
                loaded_spec = json.loads(resolved_spec.read_text(encoding="utf-8"))
                if not isinstance(loaded_spec, dict):
                    continue
                if is_project_scope:
                    raw_spec = sanitize_project_agent_spec(loaded_spec, agents_dir=agents_dir)
                else:
                    raw_spec = dict(loaded_spec)
                    _resolve_spec_prompt_field(raw_spec, base_dir=agents_dir)
                found_candidate = True
                raw_model = raw_spec.get("model")
                if (
                    pinned_model is None
                    and isinstance(raw_model, str)
                    and raw_model.strip()
                    and raw_model.strip().lower() != "auto"
                ):
                    pinned_model = raw_model.strip()
                raw_prompt = raw_spec.get("prompt")
                if agent_prompt is None and isinstance(raw_prompt, str) and raw_prompt.strip():
                    agent_prompt = raw_prompt.strip()
                raw_tools = raw_spec.get("tools")
                if agent_tools is None and isinstance(raw_tools, list):
                    agent_tools = [str(t).strip() for t in raw_tools if str(t).strip()]
                if not is_project_scope:
                    raw_mcp = raw_spec.get("mcpServers")
                    mcp_entries = (
                        [
                            {"name": k, **v}
                            for k, v in raw_mcp.items()
                            if isinstance(k, str) and k and isinstance(v, dict)
                        ]
                        if isinstance(raw_mcp, dict)
                        else [
                            dict(item)
                            for item in (raw_mcp if isinstance(raw_mcp, list) else [])
                            if isinstance(item, dict) and item.get("name")
                        ]
                    )
                    for entry in mcp_entries:
                        sname = str(entry["name"])
                        if entry.get("disabled"):
                            if sname not in disabled_names:
                                disabled_names.append(sname)
                        elif sname not in servers_by_name and sname not in disabled_names:
                            servers_by_name[sname] = entry
        if found_candidate:
            break

    return {
        "model": pinned_model,
        "prompt": agent_prompt,
        "tools": agent_tools,
        "read_only": agent_tools is not None
        and all(t in _READ_ONLY_KIRO_TOOLS for t in agent_tools),
        "mcp_servers": list(servers_by_name.values()),
        "disabled_mcp_servers": disabled_names,
    }


def load_kiro_agent_mcp_servers(
    agent_name: str | None = None,
    *,
    cwd: str | None = None,
    include_home: bool = True,
    user_agents_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Load `mcpServers` definitions strictly from trusted user-scope `.kiro/agents`."""
    spec = load_kiro_agent_spec(
        agent_name,
        cwd=cwd,
        include_home=include_home,
        user_agents_dir=user_agents_dir,
    )
    return list(spec["mcp_servers"])
