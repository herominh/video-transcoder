"""core.protocol runs only on the RunPod entry path: the Modal app and the v1 dev server never import it.

Neither image carries the contract tree (the Modal image carries no jsonschema either). The check reads
the source (every import statement, function bodies included, followed through the repository's own
modules) and runs nothing: an import made by name at run time (importlib, __import__) is not seen.
The v1 dev server (wrappers/docker_server.py) serves `core.api:app` by name, so core/api.py stands for it.

The RunPod entry point (wrappers/runpod_handler.py) is the other side of that boundary: it reaches one
start of the worker under contract v2 (core.protocol.handler) and the job runner (core.job.runner), and
none of the v1 worker's modules.

The import endpoint's entry point (wrappers/runpod_import_handler.py; its image ships core/drive_import/ and
core/__init__.py alone) reaches the import job (core.drive_import) and nothing of the transcode: no core.protocol,
no core.job, no v1 module. The transcode's RunPod entry point reaches nothing of the import.

The transcode image's build context leaves out what the shared one does and the import's files too: BuildKit reads
Dockerfile.runpod.dockerignore in place of .dockerignore for Dockerfile.runpod, so its patterns are the shared
file's, in their order, followed by the Drive import's exclusions.
"""

import ast
import re
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
PROTOCOL_PACKAGE = "core.protocol"
ENTRY_POINTS = ["core/api.py", "wrappers/modal_app.py", "wrappers/docker_server.py"]
RUNPOD_ENTRY_POINT = "wrappers/runpod_handler.py"
IMPORT_ENTRY_POINT = "wrappers/runpod_import_handler.py"
IMPORT_PACKAGE = "core.drive_import"
IMPORT_REQUIREMENTS = "requirements-import.lock"
# A requirement line's distribution name: what comes before a version specifier, an extra or a marker.
REQUIREMENT_NAME = re.compile(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)")
JOB_PACKAGE = "core.job"
V1_MODULES = [
    "core.api", "core.config", "core.transcoder", "core.storage", "core.callback", "core.signing",
    "core.v1_admission",
]
SHARED_DOCKERIGNORE = ".dockerignore"
RUNPOD_DOCKERIGNORE = "Dockerfile.runpod.dockerignore"
# What the transcode image's build context leaves out beyond the shared one: the files of the import's own image.
IMPORT_BUILD_EXCLUSIONS = ["core/drive_import/", IMPORT_ENTRY_POINT]


def _in_package(name: str, package: str) -> bool:
    return name == package or name.startswith(package + ".")


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


def test_reachable_modules_when_following_the_modal_app_should_find_admission_without_the_pipeline():
    # Arrange
    pipeline = {"core.api", "core.callback", "core.storage", "core.transcoder"}

    # Act
    reachable = _reachable_modules(REPOSITORY_ROOT / "wrappers/modal_app.py")

    # Assert
    assert {"core", "core.v1_admission"} <= reachable
    assert not pipeline & reachable


def test_runtime_modules_when_searched_should_have_no_v1_url_acquisition():
    # Arrange
    modules = [path for directory in ("core", "wrappers") for path in (REPOSITORY_ROOT / directory).rglob("*.py")]

    # Act
    sources = {path.relative_to(REPOSITORY_ROOT).as_posix(): path.read_text(encoding="utf-8") for path in modules}

    # Assert: the v2 job's separate exact-source downloader is retained.
    for name, source in sources.items():
        assert "source_url" not in source, name
        assert "upload_original" not in source, name
        if not name.startswith("core/job/"):
            assert "download_source" not in source, name


def test_v1_entry_points_when_traced_should_reach_only_the_guarded_pipeline():
    # Arrange
    modules = [path for directory in ("core", "wrappers") for path in (REPOSITORY_ROOT / directory).rglob("*.py")]

    # Act
    references = [path.relative_to(REPOSITORY_ROOT).as_posix() for path in modules if "_process_transcode" in path.read_text()]
    tree = ast.parse((REPOSITORY_ROOT / "core/api.py").read_text())
    functions = {node.name: node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}

    # Assert: every pipeline reference is in the guarded API; the dev launcher serves that API.
    assert references == ["core/api.py"]
    for name in ("transcode", "_process_transcode"):
        statements = functions[name].body
        if ast.get_docstring(functions[name]) is not None:
            statements = statements[1:]
        first = statements[0]
        if isinstance(first, ast.Try):
            first = first.body[0]
        assert isinstance(first, ast.Expr)
        assert isinstance(first.value, ast.Call)
        assert isinstance(first.value.func, ast.Name)
        assert first.value.func.id == "require_v1_execution"
    assert 'uvicorn.run("core.api:app"' in (REPOSITORY_ROOT / "wrappers/docker_server.py").read_text()


def test_runpod_entry_point_when_its_imports_are_followed_should_reach_the_v2_start_and_the_job_runner():
    # Act
    reachable = _reachable_modules(REPOSITORY_ROOT / RUNPOD_ENTRY_POINT)

    # Assert
    assert {"core.protocol.handler", "core.job.runner"} <= reachable


def test_runpod_entry_point_when_its_imports_are_followed_should_never_reach_a_v1_module():
    # Act
    reachable = _reachable_modules(REPOSITORY_ROOT / RUNPOD_ENTRY_POINT)

    # Assert
    assert sorted(name for name in reachable if name in V1_MODULES) == []


def test_runpod_entry_point_when_its_imports_are_followed_should_never_reach_the_import_job():
    # Act
    reachable = _reachable_modules(REPOSITORY_ROOT / RUNPOD_ENTRY_POINT)

    # Assert
    assert sorted(name for name in reachable if _in_package(name, IMPORT_PACKAGE)) == []


def test_import_entry_point_when_its_imports_are_followed_should_reach_the_import_job_and_its_contract():
    # Act
    reachable = _reachable_modules(REPOSITORY_ROOT / IMPORT_ENTRY_POINT)

    # Assert
    assert {"core.drive_import.job", "core.drive_import.contract", "core.drive_import.input_check"} <= reachable


def test_import_entry_point_when_its_imports_are_followed_should_never_reach_the_transcode_or_a_v1_module():
    # Act
    reachable = _reachable_modules(REPOSITORY_ROOT / IMPORT_ENTRY_POINT)

    # Assert
    transcode = [name for name in reachable if _in_package(name, PROTOCOL_PACKAGE) or _in_package(name, JOB_PACKAGE)]
    assert sorted(transcode) == []
    assert sorted(name for name in reachable if name in V1_MODULES) == []


def test_import_entry_point_when_its_imports_are_followed_should_reach_only_what_its_image_ships():
    # Act: the import image copies core/__init__.py, core/drive_import/ and the entry point, nothing else of core/
    reachable = _reachable_modules(REPOSITORY_ROOT / IMPORT_ENTRY_POINT)

    # Assert
    shipped = ("core", "wrappers.runpod_import_handler")
    outside = [name for name in reachable if name not in shipped and not _in_package(name, IMPORT_PACKAGE)]
    assert sorted(outside) == []


def _third_party_imports(entry: Path) -> set[str]:
    """The top-level names of what the repository modules that `entry` can run import from outside the repository
    and outside the standard library."""
    names: set[str] = set()
    for module in _reachable_modules(entry):
        path = _module_path(module)
        if path is None:
            continue
        for imported in _imported_names(path):
            top = imported.split(".")[0]
            if top and top != "__future__" and top not in sys.stdlib_module_names and _module_path(top) is None:
                names.add(top)
    return names


def _requirements(name: str) -> set[str]:
    names: set[str] = set()
    for line in (REPOSITORY_ROOT / name).read_text(encoding="utf-8").splitlines():
        match = REQUIREMENT_NAME.match(line.split("#")[0])
        if match is not None:
            names.add(match.group(1).lower().replace("-", "_"))
    return names


def test_import_entry_point_when_its_imports_are_followed_should_need_only_what_its_image_installs():
    # Act: every distribution of these happens to be named as its import is
    imported = _third_party_imports(REPOSITORY_ROOT / IMPORT_ENTRY_POINT)

    # Assert
    assert {"requests", "jsonschema", "referencing", "runpod"} <= imported
    assert sorted(imported - _requirements(IMPORT_REQUIREMENTS)) == []


def test_import_image_when_installing_requirements_should_use_only_the_hashed_lock():
    # Arrange
    dockerfile = REPOSITORY_ROOT / "Dockerfile.import"

    # Act
    instructions = dockerfile.read_text(encoding="utf-8").splitlines()

    # Assert
    assert f"COPY {IMPORT_REQUIREMENTS} ." in instructions
    assert f"RUN pip install --no-cache-dir --require-hashes --no-deps -r {IMPORT_REQUIREMENTS}" in instructions
    assert not any("requirements-import.txt" in line for line in instructions if not line.startswith("#"))


def _ignore_patterns(name: str) -> list[str]:
    """An ignore file's patterns in their order, as BuildKit reads them: a line starting with `#` is a comment, every
    other line is trimmed, and an empty one is skipped."""
    lines = (REPOSITORY_ROOT / name).read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if not line.startswith("#") and line.strip()]


def test_import_dockerignore_when_read_should_include_only_the_lock_and_the_files_the_image_copies():
    # Arrange
    expected = [
        "*",
        f"!{IMPORT_REQUIREMENTS}",
        "!core/__init__.py",
        "!core/drive_import/",
        f"!{IMPORT_ENTRY_POINT}",
        "!tests/contracts/import/v1/",
        "**/__pycache__",
        "**/*.pyc",
    ]

    # Act
    patterns = _ignore_patterns("Dockerfile.import.dockerignore")

    # Assert
    assert patterns == expected


def test_runpod_dockerignore_when_read_beside_the_shared_one_should_hold_its_patterns_and_the_import_exclusions_only():
    # Act
    shared = _ignore_patterns(SHARED_DOCKERIGNORE)
    runpod = _ignore_patterns(RUNPOD_DOCKERIGNORE)

    # Assert: a pattern added to one file alone shows here (their order matters, `!` patterns among them)
    assert runpod == shared + IMPORT_BUILD_EXCLUSIONS
