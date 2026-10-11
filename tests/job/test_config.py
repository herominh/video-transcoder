"""The job's own configuration: the image digest a result reports and the directory jobs make their
scratch directories in. Whole or refused; a refusal names the variable and the rule, never the value."""

from __future__ import annotations

import errno
import os
import stat
import tempfile
import traceback
from collections.abc import Iterator
from pathlib import Path

import pytest

from core.job.config import JobSettings
from core.protocol.settings import SettingsInvalid

DIGEST_VARIABLE = "VH_IMAGE_DIGEST"
SCRATCH_VARIABLE = "VH_SCRATCH_DIR"
# The variable an operator sets to give the system a temporary directory.
TEMPORARY_DIRECTORY_VARIABLE = "TMPDIR"
VALID_DIGEST = "sha256:" + "0123456789abcdef" * 4
HEX_DIGITS = 64
# A directory the system's own error names when it finds no temporary directory: no refusal may show it.
TRIED_DIRECTORY = "/CANARY/tried/for/a/temporary/directory"
CLOSED_DIRECTORY_NAME = "closed-scratch"
# The worker image runs the suite as root, which no permission bits keep out of a directory.
needs_permission_bits_to_bind = pytest.mark.skipif(
    os.name != "posix" or os.geteuid() == 0,
    reason="permission bits close no directory to root, nor to a process off POSIX",
)


def _env(tmp_path: Path, **changes: object) -> dict[str, object]:
    env: dict[str, object] = {DIGEST_VARIABLE: VALID_DIGEST, SCRATCH_VARIABLE: str(tmp_path)}
    env.update(changes)
    return env


def _refusal(env: object) -> SettingsInvalid:
    with pytest.raises(SettingsInvalid) as raised:
        JobSettings.from_env(env)  # type: ignore[arg-type]
    return raised.value


def _no_temporary_directory() -> str:
    """tempfile.gettempdir() on a system without a usable temporary directory: its error lists what it tried."""
    raise FileNotFoundError(errno.ENOENT, f"No usable temporary directory found in ['{TRIED_DIRECTORY}']")


@pytest.fixture
def closed_directory(tmp_path: Path) -> Iterator[Path]:
    """An existing directory for a test to take permissions from; they are given back, so it can be removed."""
    directory = tmp_path / CLOSED_DIRECTORY_NAME
    directory.mkdir()
    yield directory
    directory.chmod(stat.S_IRWXU)


def test_from_env_when_both_variables_are_valid_should_hold_the_digest_and_the_scratch_directory(tmp_path):
    # Arrange
    env = _env(tmp_path)

    # Act
    settings = JobSettings.from_env(env)

    # Assert
    assert settings.image_digest == VALID_DIGEST
    assert settings.scratch_root == tmp_path


def test_from_env_when_the_scratch_directory_is_not_set_should_use_the_systems_temporary_directory():
    # Arrange
    env = {DIGEST_VARIABLE: VALID_DIGEST}

    # Act
    settings = JobSettings.from_env(env)

    # Assert
    assert settings.scratch_root == Path(tempfile.gettempdir())
    assert settings.scratch_root.is_dir()


def test_from_env_when_the_image_digest_is_missing_should_refuse_it_as_required(tmp_path):
    # Arrange
    env = _env(tmp_path)
    del env[DIGEST_VARIABLE]

    # Act
    refusal = _refusal(env)

    # Assert
    assert refusal.variable == DIGEST_VARIABLE
    assert "required" in refusal.rule


@pytest.mark.parametrize(
    "digest",
    [
        pytest.param("0123456789abcdef" * 4, id="no-algorithm-prefix"),
        pytest.param("sha512:" + "0123456789abcdef" * 4, id="another-algorithm"),
        pytest.param("sha256:" + "0" * (HEX_DIGITS - 1), id="one-digit-short"),
        pytest.param("sha256:" + "0" * (HEX_DIGITS + 1), id="one-digit-long"),
        pytest.param("sha256:" + "0123456789ABCDEF" * 4, id="upper-case-hex"),
        pytest.param("SHA256:" + "0123456789abcdef" * 4, id="upper-case-prefix"),
        pytest.param("sha256:" + "g" * HEX_DIGITS, id="not-hex"),
        pytest.param(" " + VALID_DIGEST, id="leading-space"),
        pytest.param(VALID_DIGEST + "\n", id="trailing-line-break"),
        pytest.param("registry.example/worker@" + VALID_DIGEST, id="an-image-reference"),
        pytest.param(12345, id="not-a-string"),
    ],
)
def test_from_env_when_the_image_digest_is_malformed_should_refuse_it_without_showing_it(tmp_path, digest):
    # Arrange
    env = _env(tmp_path, **{DIGEST_VARIABLE: digest})

    # Act
    refusal = _refusal(env)

    # Assert
    assert refusal.variable == DIGEST_VARIABLE
    assert str(digest).strip() not in str(refusal)


def test_from_env_when_the_image_digest_is_empty_should_refuse_it(tmp_path):
    # Arrange
    env = _env(tmp_path, **{DIGEST_VARIABLE: ""})

    # Act
    refusal = _refusal(env)

    # Assert
    assert refusal.variable == DIGEST_VARIABLE


def test_from_env_when_the_scratch_directory_is_relative_should_refuse_it(tmp_path, monkeypatch):
    # Arrange: the relative path exists from the working directory, so only its being relative refuses it
    (tmp_path / "relative-scratch").mkdir()
    monkeypatch.chdir(tmp_path)

    # Act
    refusal = _refusal(_env(tmp_path, **{SCRATCH_VARIABLE: "relative-scratch"}))

    # Assert
    assert refusal.variable == SCRATCH_VARIABLE
    assert "absolute" in refusal.rule
    assert "relative-scratch" not in str(refusal)


def test_from_env_when_the_scratch_directory_does_not_exist_should_refuse_it(tmp_path):
    # Arrange
    missing = tmp_path / "never-made"

    # Act
    refusal = _refusal(_env(tmp_path, **{SCRATCH_VARIABLE: str(missing)}))

    # Assert
    assert refusal.variable == SCRATCH_VARIABLE
    assert "never-made" not in str(refusal)


def test_from_env_when_the_scratch_directory_is_a_file_should_refuse_it(tmp_path):
    # Arrange
    a_file = tmp_path / "a-file"
    a_file.write_bytes(b"not a directory")

    # Act
    refusal = _refusal(_env(tmp_path, **{SCRATCH_VARIABLE: str(a_file)}))

    # Assert
    assert refusal.variable == SCRATCH_VARIABLE
    assert "a-file" not in str(refusal)


@needs_permission_bits_to_bind
@pytest.mark.parametrize(
    "mode",
    [
        pytest.param(stat.S_IRUSR | stat.S_IXUSR, id="not-writable"),
        pytest.param(stat.S_IRUSR | stat.S_IWUSR, id="not-searchable"),
    ],
)
def test_from_env_when_the_process_cannot_make_a_directory_in_the_scratch_directory_should_refuse_it(
    tmp_path, closed_directory, mode
):
    # Arrange: the directory exists, and its permissions keep this process from making one in it
    closed_directory.chmod(mode)

    # Act
    refusal = _refusal(_env(tmp_path, **{SCRATCH_VARIABLE: str(closed_directory)}))

    # Assert: the variable and the rule, never the value
    assert refusal.variable == SCRATCH_VARIABLE
    assert "writable and searchable" in refusal.rule
    assert CLOSED_DIRECTORY_NAME not in str(refusal)


@pytest.mark.parametrize(
    "name",
    [
        pytest.param("with:colon", id="colon"),
        pytest.param("with%percent", id="percent"),
        pytest.param("with\nline-feed", id="line-feed"),
        pytest.param("with\rcarriage-return", id="carriage-return"),
    ],
)
def test_from_env_when_the_scratch_directory_holds_a_character_the_stages_refuse_should_refuse_it(tmp_path, name):
    # Arrange: the directory exists, so only the character refuses it
    directory = tmp_path / name
    directory.mkdir()

    # Act
    refusal = _refusal(_env(tmp_path, **{SCRATCH_VARIABLE: str(directory)}))

    # Assert
    assert refusal.variable == SCRATCH_VARIABLE
    assert name not in str(refusal)


def test_from_env_when_the_scratch_directory_holds_a_double_quote_should_refuse_it_by_that_rule(tmp_path):
    # Arrange: the directory exists, so only the character refuses it
    name = 'scratch-"quoted"'
    directory = tmp_path / name
    directory.mkdir()

    # Act
    refusal = _refusal(_env(tmp_path, **{SCRATCH_VARIABLE: str(directory)}))

    # Assert: the variable and the rule, never the value
    assert refusal.variable == SCRATCH_VARIABLE
    assert "double quote" in refusal.rule
    assert name not in str(refusal)


@pytest.mark.parametrize(
    "name",
    [
        pytest.param("scratch-été", id="an-accented-letter"),
        pytest.param("scratch-視頻", id="letters-of-another-script"),
        pytest.param("scratch- wide", id="a-no-break-space"),
        pytest.param("scratch-\twith-a-tab", id="a-tab"),
        pytest.param("scratch-\x01control", id="a-control-character"),
        pytest.param("scratch-\x7fdelete", id="the-delete-character"),
    ],
)
def test_from_env_when_the_scratch_directory_holds_anything_but_printable_ascii_should_refuse_it_by_that_rule(
    tmp_path, name
):
    # Arrange: the directory exists, so only the character refuses it
    directory = tmp_path / name
    directory.mkdir()

    # Act
    refusal = _refusal(_env(tmp_path, **{SCRATCH_VARIABLE: str(directory)}))

    # Assert: the variable and the rule, never the value
    assert refusal.variable == SCRATCH_VARIABLE
    assert "printable ASCII" in refusal.rule
    assert name not in str(refusal)
    assert str(directory) not in str(refusal)


@pytest.mark.parametrize(
    "name",
    [
        pytest.param("with a space", id="a-space"),
        pytest.param("with ~!#$&'()+,;=@[]^_`{}-.", id="printable-punctuation"),
    ],
)
def test_from_env_when_the_scratch_directory_holds_only_printable_ascii_should_accept_it(tmp_path, name):
    # Arrange
    directory = tmp_path / name
    directory.mkdir()

    # Act
    settings = JobSettings.from_env(_env(tmp_path, **{SCRATCH_VARIABLE: str(directory)}))

    # Assert
    assert settings.scratch_root == directory


def test_from_env_when_the_default_temporary_directory_holds_a_non_ascii_character_should_refuse_it(
    tmp_path, monkeypatch
):
    # Arrange: no VH_SCRATCH_DIR, and a system temporary directory with an accented name
    directory = tmp_path / "temporaire-é"
    directory.mkdir()
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(directory))

    # Act
    refusal = _refusal({DIGEST_VARIABLE: VALID_DIGEST})

    # Assert
    assert refusal.variable == SCRATCH_VARIABLE
    assert "printable ASCII" in refusal.rule
    assert str(directory) not in str(refusal)


@pytest.mark.parametrize(
    "scratch_directory_is_set",
    [pytest.param(True, id="scratch-directory-set"), pytest.param(False, id="scratch-directory-not-set")],
)
def test_from_env_when_the_system_has_no_temporary_directory_should_refuse_the_start_naming_tmpdir(
    tmp_path, monkeypatch, scratch_directory_is_set
):
    # Arrange: the version check and the preflight work in the system's temporary directory, whatever
    # the scratch directory is; this system has none
    monkeypatch.setattr(tempfile, "gettempdir", _no_temporary_directory)
    env = _env(tmp_path) if scratch_directory_is_set else {DIGEST_VARIABLE: VALID_DIGEST}

    # Act
    refusal = _refusal(env)

    # Assert: TMPDIR and the rule in the module's words; nothing of the system's error, which lists
    # directories: not in the text, not in the traceback a refused start prints, not riding along
    printed = "".join(traceback.format_exception(type(refusal), refusal, refusal.__traceback__))
    assert refusal.variable == TEMPORARY_DIRECTORY_VARIABLE
    assert "temporary directory" in refusal.rule
    assert TRIED_DIRECTORY not in str(refusal) + printed
    assert refusal.__cause__ is None and refusal.__context__ is None


@pytest.mark.parametrize("value", [pytest.param("", id="empty"), pytest.param(7, id="not-a-string")])
def test_from_env_when_the_scratch_directory_is_not_a_usable_text_should_refuse_it(tmp_path, value):
    # Arrange
    env = _env(tmp_path, **{SCRATCH_VARIABLE: value})

    # Act
    refusal = _refusal(env)

    # Assert
    assert refusal.variable == SCRATCH_VARIABLE


def test_from_env_when_the_process_environment_holds_the_variables_should_read_only_the_given_mapping(
    tmp_path, monkeypatch
):
    # Arrange: the process environment is complete and valid; the mapping given is empty
    other_directory = tmp_path / "from-the-process"
    other_directory.mkdir()
    monkeypatch.setenv(DIGEST_VARIABLE, VALID_DIGEST)
    monkeypatch.setenv(SCRATCH_VARIABLE, str(other_directory))

    # Act
    refusal = _refusal({})
    settings = JobSettings.from_env({DIGEST_VARIABLE: VALID_DIGEST})

    # Assert
    assert (refusal.variable, refusal.rule) == (DIGEST_VARIABLE, "is required")
    assert settings.scratch_root != other_directory


@pytest.mark.parametrize("env", [None, [("VH_IMAGE_DIGEST", VALID_DIGEST)], "VH_IMAGE_DIGEST"])
def test_from_env_when_the_environment_is_not_a_mapping_should_raise_type_error(env):
    # Act and Assert
    with pytest.raises(TypeError):
        JobSettings.from_env(env)


def test_job_settings_when_built_should_be_immutable(tmp_path):
    # Arrange
    settings = JobSettings.from_env(_env(tmp_path))

    # Act and Assert
    with pytest.raises(AttributeError):
        settings.image_digest = "sha256:" + "f" * HEX_DIGITS  # type: ignore[misc]
