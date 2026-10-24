"""Hierarchical Cyclopts command and subfolder group discovery for agy-acp."""

from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType
from typing import Any

import anyio
from cyclopts import App

__all__ = ["AnyioApp", "register_commands"]

COMMANDS_DIR = Path(__file__).resolve().parent
DEFAULT_PACKAGE = __name__


class AnyioApp(App):
    """Cyclopts App subclass that executes commands via anyio."""

    def __call__(
        self,
        tokens: None | str | Iterable[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Execute command via anyio.run, using a worker thread if an event loop is active."""

        async def _runner() -> Any:
            return await self.run_async(tokens, **kwargs)

        try:
            anyio.current_effective_deadline()
        except RuntimeError:
            return anyio.run(_runner)

        with ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(anyio.run, _runner).result()


def _is_ignored_entry(entry: Path) -> bool:
    """Return True for private/underscore-prefixed or hidden entries."""
    return entry.name.startswith(("_", "."))


def _read_group_docstring(group_dir: Path) -> str | None:
    """Extract the module docstring from a group directory's __init__.py if present."""
    init_file = group_dir / "__init__.py"
    if not init_file.is_file():
        return None
    tree = ast.parse(init_file.read_text(encoding="utf-8"), filename=str(init_file))
    return ast.get_docstring(tree)


def _load_module(file_path: Path, package_prefix: str | None) -> ModuleType:
    """Import a command module via package name or file location (supporting kebab-case files)."""
    if package_prefix is not None and file_path.stem.isidentifier():
        return importlib.import_module(f"{package_prefix}.{file_path.stem}")

    safe_stem = file_path.stem.replace("-", "_")
    module_name = f"_dynamic_cmd_{abs(hash(file_path.resolve()))}_{safe_stem}"
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        msg = f"Unable to load command module from {file_path}"
        raise ImportError(msg)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def register_commands(
    app: App,
    commands_dir: Path | None = None,
    *,
    package_prefix: str | None = None,
) -> App:
    """Recursively scan a directory to register kebab-case command files and subfolder groups."""
    target_dir = COMMANDS_DIR if commands_dir is None else commands_dir.resolve()
    active_prefix = (
        DEFAULT_PACKAGE if package_prefix is None and target_dir == COMMANDS_DIR else package_prefix
    )

    for entry in sorted(target_dir.iterdir(), key=lambda p: p.name):
        if _is_ignored_entry(entry):
            continue

        if entry.is_file() and entry.suffix == ".py":
            module = _load_module(entry, active_prefix)
            default_fn = getattr(module, "default", None)
            if callable(default_fn):
                app.command(
                    default_fn,
                    name=entry.stem,
                    help=inspect.getdoc(default_fn),
                )
        elif entry.is_dir():
            group_help = _read_group_docstring(entry)
            group_app = (
                AnyioApp(name=entry.name, help=group_help)
                if group_help is not None
                else AnyioApp(name=entry.name)
            )
            sub_prefix = (
                f"{active_prefix}.{entry.name}"
                if active_prefix is not None and entry.name.isidentifier()
                else None
            )
            init_file = entry / "__init__.py"
            if init_file.is_file():
                init_mod = _load_module(init_file, sub_prefix)
                group_default = getattr(init_mod, "default", None)
                if callable(group_default):
                    group_app.default(group_default)
            register_commands(group_app, entry, package_prefix=sub_prefix)
            app.command(group_app)

    return app
