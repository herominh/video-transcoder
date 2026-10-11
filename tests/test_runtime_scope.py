"""core.protocol runs only on the RunPod entry path: the Modal app and the v1 dev server never import it.

Neither image carries the contract tree (the Modal image carries no jsonschema either). The check reads
the source (every import statement, function bodies included, followed through the repository's own
modules) and runs nothing: an import made by name at run time (importlib, __import__) is not seen.
The v1 dev server (wrappers/docker_server.py) serves `core.api:app` by name, so core/api.py stands for it.
"""

import ast
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
PROTOCOL_PACKAGE = "core.protocol"
ENTRY_POINTS = ["core/api.py", "wrappers/modal_app.py"]


def _module_path(module: str) -> Path | None:
    """The repository file of a dotted module name; None when the module is not the repository's."""
    base = REPOSITORY_ROOT.joinpath(*module.split("."))
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def _module_name(path: Path) -> str:
    relative = path.relative_to(REPOSITORY_ROOT).with_suffix("")
    parts = relative.parts[:-1] if relative.name == "__init__" else relative.parts
    return ".".join(parts)


def _package_of(path: Path) -> list[str]:
    name = _module_name(path).split(".")
    return name if path.name == "__init__.py" else name[:-1]


def _imported_names(path: Path) -> set[str]:
    """Every dotted name an import statement of the file names, relative imports resolved."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                package = _package_of(path)
                anchor = package[: len(package) - node.level + 1]
                base = ".".join([*anchor, *([node.module] if node.module else [])])
            else:
                base = node.module or ""
            names.add(base)
            # `from core import protocol` imports the module core.protocol.
            names.update(f"{base}.{alias.name}" for alias in node.names)
    return names


def _reachable_modules(entry: Path) -> set[str]:
    """The repository modules that importing `entry` can run: its imports, theirs, and their parent packages."""
    seen: set[str] = set()
    pending = [entry]
    while pending:
        path = pending.pop()
        name = _module_name(path)
        if name in seen:
            continue
        seen.add(name)
        for imported in _imported_names(path):
            parts = imported.split(".")
            for end in range(1, len(parts) + 1):
                found = _module_path(".".join(parts[:end]))
                if found is not None:
                    pending.append(found)
    return seen


@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_entry_point_when_its_imports_are_followed_should_never_reach_core_protocol(entry):
    # Act
    reachable = _reachable_modules(REPOSITORY_ROOT / entry)

    # Assert
    assert sorted(name for name in reachable if name == PROTOCOL_PACKAGE or name.startswith(PROTOCOL_PACKAGE + ".")) == []


def test_reachable_modules_when_following_the_modal_app_should_find_what_it_imports_through_core_api():
    # Act: the Modal app imports core.api inside a function; only core.api imports core.storage and the others.
    reachable = _reachable_modules(REPOSITORY_ROOT / "wrappers/modal_app.py")

    # Assert
    assert {"core", "core.api", "core.callback", "core.storage", "core.transcoder"} <= reachable
