"""Custom agent spec loader, project security sanitizer, and KAS permissions evaluator."""

from __future__ import annotations

import json
import math
import os
import re
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
        "_crewShellAllowlist",
        "_crewWriteAllowlist",
    }
)

#: Reserved KAS built-in agent IDs that KiroCrew projects with `-crewmate` or `crew-`.
KAS_RESERVED_AGENT_IDS: frozenset[str] = frozenset(
    {
        "default",
        "vibe",
        "spec",
        "quick-spec",
        "bug-fix",
        "plan",
        "autonomous",
    }
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
        "fs_read",
        "fs_write",
        "shell",
        "web_fetch",
        "web_search",
        "mcp",
        "subagent",
        "skill",
        "power",
        "context",
        "diagnostics",
        "sandbox_network",
    }
)

META_CAPABILITY_EXPANSION: dict[str, tuple[str, ...]] = {
    "all": _BUILTIN_CAPABILITIES + ("mcp",),
    "builtin": _BUILTIN_CAPABILITIES,
    "filesystem": ("fs_read", "fs_write"),
}

_PATH_SCOPED_CAPABILITIES: frozenset[str] = frozenset({"fs_read", "fs_write", "mcp"})

_CAPABILITY_BY_TOOL_NAME: dict[str, str] = {
    # KAS v3 built-in tool IDs
    "read_file": "fs_read",
    "list_directory": "fs_read",
    "grep_search": "fs_read",
    "file_search": "fs_read",
    "fs_write": "fs_write",
    "str_replace": "fs_write",
    "fs_append": "fs_write",
    "delete_file": "fs_write",
    "run_command": "shell",
    "web_fetch": "web_fetch",
    "remote_web_search": "web_search",
    "invoke_sub_agent": "subagent",
    # Kiro v2 / Crew tool spellings
    "fs_read": "fs_read",
    "read": "fs_read",
    "grep": "fs_read",
    "glob": "fs_read",
    "write": "fs_write",
    "edit": "fs_write",
    "execute_bash": "shell",
    "shell": "shell",
    "bash": "shell",
    "web_search": "web_search",
    "use_subagent": "subagent",
    "subagent": "subagent",
    "skill": "skill",
    "power": "power",
    "context": "context",
    "diagnostics": "diagnostics",
}


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


_FrontmatterSafeLoader.add_constructor(
    "tag:yaml.org,2002:timestamp",
    _construct_timestamp_as_text,
)


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
    if text.startswith("\ufeff"):
        text = text[1:]
    return text.startswith(("---\n", "---\r\n"))


def split_markdown_spec(text: str) -> tuple[str, str] | None:
    """Split a markdown agent spec into `(frontmatter_yaml, body)`, or `None` if unfenced."""
    if not opens_frontmatter_fence(text):
        return None
    if text.startswith("\ufeff"):
        text = text[1:]
    first_newline = text.index("\n")
    close = _FRONTMATTER_CLOSE_RE.search(text, first_newline + 1)
    if close is None:
        return None
    frontmatter = text[first_newline + 1 : close.start()]
    body = text[close.end() :]
    if body.startswith("\r\n"):
        body = body[2:]
    elif body.startswith("\n"):
        body = body[1:]
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


def parse_agent_spec_file(path: Path) -> dict[str, Any]:
    """Parse either a `.json` or `.md` agent specification file from disk."""
    suffix = path.suffix.lower()
    raw_text = path.read_text(encoding="utf-8")
    if suffix == MARKDOWN_SUFFIX:
        return parse_markdown_agent_spec(raw_text)
    if suffix == JSON_SUFFIX:
        try:
            loaded = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON agent spec: {exc}") from exc
        if not isinstance(loaded, dict):
            raise ValueError("root must be a JSON object")
        return loaded
    raise ValueError(f"unsupported agent spec file extension: {path.suffix!r}")


def resolve_contained_prompt_file(prompt_value: str, *, base_dir: Path) -> str:
    """Resolve a `file://` prompt strictly contained within `base_dir` (no `..` or symlinks out)."""
    if not prompt_value.startswith("file://"):
        return prompt_value

    raw_ref = prompt_value[len("file://") :]
    if not raw_ref or raw_ref.startswith(("~", "/")) or Path(raw_ref).expanduser().anchor:
        raise CustomAgentSecurityError(
            f"Refusing rooted or home-anchored file:// prompt reference: {prompt_value!r}"
        )

    resolved_base = base_dir.resolve()
    candidate = resolved_base / raw_ref
    try:
        resolved_target = candidate.resolve(strict=True)
    except OSError as exc:
        raise CustomAgentSecurityError(
            f"Prompt file reference could not be resolved inside {resolved_base}: {raw_ref!r}"
        ) from exc

    if resolved_target == resolved_base or resolved_base not in resolved_target.parents:
        raise CustomAgentSecurityError(
            f"Refusing file:// prompt escaping {resolved_base}: {prompt_value!r}"
        )
    if not resolved_target.is_file():
        raise CustomAgentSecurityError(
            f"Prompt file reference is not a regular file: {prompt_value!r}"
        )
    return resolved_target.read_text(encoding="utf-8")


def sanitize_project_agent_spec(
    spec: Mapping[str, Any],
    *,
    agents_dir: Path,
) -> dict[str, Any]:
    """Strip untrusted keys from a project-scope spec and enforce `file://` prompt containment."""
    sanitized: dict[str, Any] = {
        key: spec[key] for key in PROJECT_SPEC_HONOURED_KEYS if key in spec
    }
    prompt_val = sanitized.get("prompt")
    if isinstance(prompt_val, str) and prompt_val.startswith("file://"):
        sanitized["prompt"] = resolve_contained_prompt_file(prompt_val, base_dir=agents_dir)
    return sanitized


def _iter_live_spec_files(directory: Path) -> list[Path]:
    """Return sorted `.json` and `.md` agent spec files in `directory`, `.json` beating `.md`."""
    if not directory.is_dir():
        return []
    try:
        with os.scandir(directory) as entries:
            names = [entry.name for entry in entries if not entry.name.startswith(".")]
    except OSError:
        return []

    filtered = [n for n in names if not n.startswith(NATIVE_SKILL_ALIAS_PREFIX)]
    json_files = [directory / n for n in filtered if n.lower().endswith(JSON_SUFFIX)]
    md_files = [directory / n for n in filtered if n.lower().endswith(MARKDOWN_SUFFIX)]
    json_stems = {p.stem.lower() for p in json_files}
    live = list(json_files)
    for md_path in md_files:
        if md_path.stem.lower() in json_stems:
            continue
        if (directory / f"{md_path.stem}{JSON_SUFFIX}").exists():
            continue
        live.append(md_path)
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

    def evaluate_permission(
        self,
        *,
        capability: str,
        resource: str = "",
    ) -> PermissionDecision:
        """Evaluate KAS inline permissions for `(capability, resource)`."""
        return evaluate_kas_permissions(
            self.permissions,
            capability=capability,
            resource=resource,
        )

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
        )

    def to_available_mode(self, *, v3_meta: bool = True) -> dict[str, Any]:
        """Project this custom agent as an ACP `availableModes` entry."""
        entry: dict[str, Any] = {
            "id": self.agent_id,
            "name": self.name or self.agent_id,
            "description": self.description or self.name or self.agent_id,
        }
        if v3_meta:
            entry["_meta"] = {
                "kiro": {
                    "resource": {
                        "source": {
                            "origin": self.origin,
                        }
                    }
                }
            }
        return entry


def _spec_from_mapping(
    raw: Mapping[str, Any],
    *,
    origin: AgentOrigin,
    fallback_id: str = "",
) -> CustomAgentSpec | None:
    cleaned = {k: v for k, v in raw.items() if k not in CREW_ONLY_ENTRY_KEYS}
    raw_id = cleaned.get("id") or cleaned.get("name") or fallback_id
    if not isinstance(raw_id, str) or not raw_id.strip():
        return None
    agent_id = raw_id.strip()
    raw_name = cleaned.get("name")
    name = raw_name.strip() if isinstance(raw_name, str) and raw_name.strip() else agent_id
    desc_val = cleaned.get("description")
    description = desc_val if isinstance(desc_val, str) else ""

    prompt_val = cleaned.get("systemPrompt")
    if not isinstance(prompt_val, str):
        prompt_val = cleaned.get("prompt")
    prompt = prompt_val if isinstance(prompt_val, str) else ""

    model_val = cleaned.get("model")
    model = model_val.strip() if isinstance(model_val, str) and model_val.strip() else None

    tools_raw = cleaned.get("tools")
    tools: tuple[str, ...] | None = (
        _normalize_str_tuple(tools_raw) if tools_raw is not None else None
    )
    excluded_tools = _normalize_str_tuple(cleaned.get("excludedTools"))
    allowed_tools = _normalize_str_tuple(cleaned.get("allowedTools"))

    perms_raw = cleaned.get("permissions")
    permissions: dict[str, Any] | list[Any] | None = (
        perms_raw if isinstance(perms_raw, (dict, list)) else None
    )

    mcp_raw = cleaned.get("mcpServers")
    mcp_servers: dict[str, Any] = dict(mcp_raw) if isinstance(mcp_raw, dict) else {}

    return CustomAgentSpec(
        agent_id=agent_id,
        name=name,
        description=description,
        prompt=prompt,
        origin=origin,
        model=model,
        tools=tools,
        excluded_tools=excluded_tools,
        allowed_tools=allowed_tools,
        permissions=permissions,
        mcp_servers=mcp_servers,
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
        self._by_id: dict[str, CustomAgentSpec] = {}
        for spec in agents:
            self._by_id[spec.agent_id] = spec

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
        if cwd is not None:
            project_dir = Path(cwd) / ".kiro" / "agents"
            for spec_path in _iter_live_spec_files(project_dir):
                loaded = cls._read_disk_spec(spec_path)
                if loaded is None:
                    continue
                try:
                    sanitized = sanitize_project_agent_spec(loaded, agents_dir=project_dir)
                except CustomAgentSecurityError:
                    continue
                agent = _spec_from_mapping(
                    sanitized,
                    origin="project",
                    fallback_id=spec_path.stem,
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
            prompt_val = user_dict.get("prompt")
            if isinstance(prompt_val, str) and prompt_val.startswith("file://"):
                try:
                    user_dict["prompt"] = resolve_contained_prompt_file(
                        prompt_val,
                        base_dir=resolved_user_dir,
                    )
                except CustomAgentSecurityError:
                    continue
            agent = _spec_from_mapping(
                user_dict,
                origin="user",
                fallback_id=spec_path.stem,
            )
            if agent is not None:
                merged[agent.agent_id] = agent

        # 3. Highest precedence: Wire `params._meta.kiro.customAgents` (`origin="client"`)
        if wire_custom_agents:
            for raw_entry in wire_custom_agents:
                if not isinstance(raw_entry, Mapping):
                    continue
                agent = _spec_from_mapping(raw_entry, origin="client")
                if agent is not None:
                    merged[agent.agent_id] = agent

        return cls(list(merged.values()))

    @staticmethod
    def _read_disk_spec(path: Path) -> dict[str, Any] | None:
        try:
            raw_text = path.read_text(encoding="utf-8")
        except OSError:
            return None
        if path.suffix.lower() == MARKDOWN_SUFFIX and not opens_frontmatter_fence(raw_text):
            return None
        try:
            if path.suffix.lower() == MARKDOWN_SUFFIX:
                return parse_markdown_agent_spec(raw_text)
            parsed = json.loads(raw_text)
            return parsed if isinstance(parsed, dict) else None
        except ValueError, json.JSONDecodeError:
            return None

    def get(self, agent_id: str) -> CustomAgentSpec | None:
        """Look up an agent by exact wire ID, stripped crewmate ID, or display name."""
        direct = self._by_id.get(agent_id)
        if direct is not None:
            return direct
        stripped = strip_kas_wire_agent_affix(agent_id)
        if stripped != agent_id and stripped in self._by_id:
            return self._by_id[stripped]
        suffixed = f"{agent_id}{KAS_WIRE_ID_SUFFIX}"
        if suffixed in self._by_id:
            return self._by_id[suffixed]
        prefixed = f"{KAS_WIRE_ID_PREFIX}{agent_id}"
        if prefixed in self._by_id:
            return self._by_id[prefixed]
        for spec in self._by_id.values():
            if spec.name == agent_id:
                return spec
        return None

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


def _expand_rule_capabilities(capability: str) -> tuple[str, ...]:
    if capability in META_CAPABILITY_EXPANSION:
        return META_CAPABILITY_EXPANSION[capability]
    return (capability,)


def _normalize_permission_rules(raw_permissions: Any) -> list[dict[str, Any]]:
    """Normalize `{"rules": [...]}` or `[...]` permission policies into canonical rules."""
    if raw_permissions is None:
        return []
    if isinstance(raw_permissions, Mapping):
        raw_list = raw_permissions.get("rules")
        if not isinstance(raw_list, Sequence) or isinstance(raw_list, (str, bytes)):
            return []
    elif isinstance(raw_permissions, Sequence) and not isinstance(raw_permissions, (str, bytes)):
        raw_list = raw_permissions
    else:
        return []

    normalized: list[dict[str, Any]] = []
    for entry in raw_list:
        if not isinstance(entry, Mapping):
            continue
        effect = entry.get("effect") or entry.get("permission")
        if not isinstance(effect, str) or effect not in ("allow", "ask", "deny"):
            continue

        capability = entry.get("capability")
        match_patterns: list[str] = []
        exclude_patterns: list[str] = []

        raw_match = entry.get("match")
        if isinstance(raw_match, Sequence) and not isinstance(raw_match, (str, bytes)):
            match_patterns.extend(str(p) for p in raw_match if isinstance(p, str) and p.strip())

        raw_exclude = entry.get("exclude")
        if isinstance(raw_exclude, Sequence) and not isinstance(raw_exclude, (str, bytes)):
            exclude_patterns.extend(str(p) for p in raw_exclude if isinstance(p, str) and p.strip())

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

        # Support legacy/list `{tool, permission, commands?}` entries
        tool_ref = entry.get("tool")
        if not isinstance(tool_ref, str) or not tool_ref.strip():
            continue
        tool_ref = tool_ref.strip()
        if tool_ref == "*":
            normalized.append(
                {
                    "capability": "all",
                    "match": match_patterns,
                    "exclude": exclude_patterns,
                    "effect": effect,
                }
            )
            continue
        if tool_ref.startswith("@"):
            mcp_target = tool_ref[1:]
            if "/" not in mcp_target:
                mcp_target = f"{mcp_target}/*"
            normalized.append(
                {
                    "capability": "mcp",
                    "match": [mcp_target, *match_patterns],
                    "exclude": exclude_patterns,
                    "effect": effect,
                }
            )
            continue

        mapped_cap = _CAPABILITY_BY_TOOL_NAME.get(tool_ref)
        if mapped_cap is None:
            continue
        raw_commands = entry.get("commands")
        if isinstance(raw_commands, Sequence) and not isinstance(raw_commands, (str, bytes)):
            match_patterns.extend(str(c) for c in raw_commands if isinstance(c, str) and c.strip())
        normalized.append(
            {
                "capability": mapped_cap,
                "match": match_patterns,
                "exclude": exclude_patterns,
                "effect": effect,
            }
        )

    return normalized


def evaluate_kas_permissions(
    permissions: Any,
    *,
    capability: str,
    resource: str = "",
) -> PermissionDecision:
    """Evaluate KAS permissions with Cedar most-restrictive-match (`deny > ask > allow`).

    Unmatched requests resolve to `"ask"`.
    """
    rules = _normalize_permission_rules(permissions)
    if not rules:
        return "ask"

    path_mode = capability in _PATH_SCOPED_CAPABILITIES
    saw_ask = False
    saw_allow = False

    for rule in rules:
        expanded = _expand_rule_capabilities(rule["capability"])
        if capability not in expanded:
            continue

        match_list: list[str] = rule["match"]
        if match_list and not _matches_any_pattern(match_list, resource, path_mode=path_mode):
            continue

        exclude_list: list[str] = rule["exclude"]
        if exclude_list and _matches_any_pattern(exclude_list, resource, path_mode=path_mode):
            continue

        effect = rule["effect"]
        if effect == "deny":
            return "deny"
        if effect == "ask":
            saw_ask = True
        elif effect == "allow":
            saw_allow = True

    if saw_ask:
        return "ask"
    if saw_allow:
        return "allow"
    return "ask"


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
        return "ask"

    if capability == "shell":
        resource = command
    elif capability in ("fs_read", "fs_write"):
        resource = path
    elif capability == "web_fetch":
        resource = url
    elif capability == "web_search":
        resource = query
    elif capability == "subagent":
        resource = subagent_target
    else:
        resource = ""

    return evaluate_kas_permissions(
        permissions,
        capability=capability,
        resource=resource,
    )
