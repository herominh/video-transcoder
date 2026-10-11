"""The job's own configuration, read from the environment beside the protocol's (core.protocol.settings).

    VH_IMAGE_DIGEST   required: `sha256:` and the 64 lowercase hex digits of the image this worker runs
                      from. A result reports it as `encoder.image_digest`, and a container cannot learn
                      its own digest, so the deployment states it.
    VH_SCRATCH_DIR    optional: the existing directory every job makes its scratch directory in;
                      default the system's temporary directory. Either way this process may write to
                      and search it, and its path holds printable ASCII only and none of the
                      characters the stages refuse in a path (`:`, `%`, a line break, a double quote).

Whatever the scratch directory is, the system must have a usable temporary directory (TMPDIR gives it
one): the ffmpeg version check and the preflight make their working directories there. A start on a
system without one is refused, naming TMPDIR.

A refusal is the protocol settings' SettingsInvalid: it names the variable and the rule, never the value.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from core.encode import KEY_PATH_FORBIDDEN
from core.encode_command import UNSAFE_PATH_CHARACTERS
from core.protocol import schema
from core.protocol.settings import SettingsInvalid

IMAGE_DIGEST_VARIABLE = "VH_IMAGE_DIGEST"
SCRATCH_DIR_VARIABLE = "VH_SCRATCH_DIR"
# Not read here: the variable an operator sets to give the system a temporary directory, which a
# start refused for want of one names.
TEMPORARY_DIR_VARIABLE = "TMPDIR"
# The characters a scratch path may hold: printable ASCII, the space included.
PRINTABLE_ASCII_FIRST = 0x20
PRINTABLE_ASCII_LAST = 0x7E
# What making a directory in the scratch directory takes of it.
SCRATCH_DIR_ACCESS = os.W_OK | os.X_OK
_IMAGE_DIGEST_DEFINITION = (
    schema.SCHEMA_BASE_URI + "schemas/common.schema.json#/$defs/encoder/properties/image_digest"
)


@dataclass(frozen=True)
class JobSettings:
    """The validated job configuration; build it with from_env()."""

    image_digest: str
    scratch_root: Path

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> JobSettings:
        """Read and check both variables from `env` alone, and require the system's temporary
        directory; the first problem raises SettingsInvalid."""
        if not isinstance(env, Mapping):
            raise TypeError("from_env() takes the environment as a mapping")
        image_digest = _image_digest(env)
        temporary_directory = _system_temporary_directory()
        return cls(image_digest=image_digest, scratch_root=_scratch_root(env, temporary_directory))


def _image_digest(env: Mapping[str, str]) -> str:
    digest = env.get(IMAGE_DIGEST_VARIABLE)
    if digest is None:
        raise SettingsInvalid(IMAGE_DIGEST_VARIABLE, "is required")
    if not isinstance(digest, str):
        raise SettingsInvalid(IMAGE_DIGEST_VARIABLE, "must be a string")
    if schema.error_pointers_for_schema({"$ref": _IMAGE_DIGEST_DEFINITION}, digest):
        raise SettingsInvalid(IMAGE_DIGEST_VARIABLE, "must be `sha256:` followed by 64 lowercase hex digits")
    return digest


def _scratch_root(env: Mapping[str, str], default: str) -> Path:
    """The directory jobs make their scratch directories in, `default` when the variable is not set,
    held to what the stages accept of a path and to what making a directory in it takes."""
    text = env.get(SCRATCH_DIR_VARIABLE, default)
    if not isinstance(text, str):
        raise SettingsInvalid(SCRATCH_DIR_VARIABLE, "must be a string")
    if text == "":
        raise SettingsInvalid(SCRATCH_DIR_VARIABLE, "must not be empty")
    # The stages refuse a path holding one of these: ffmpeg's command line would read it as something else.
    if any(character in text for character in UNSAFE_PATH_CHARACTERS):
        raise SettingsInvalid(SCRATCH_DIR_VARIABLE, "must not hold `:`, `%` or a line break")
    # The encode writes the playlist it checks a rendition through, which quotes this path, as ASCII:
    # with any other character in it every job would fail after its first rendition was encoded.
    if not all(PRINTABLE_ASCII_FIRST <= ord(character) <= PRINTABLE_ASCII_LAST for character in text):
        raise SettingsInvalid(SCRATCH_DIR_VARIABLE, "must hold printable ASCII characters only")
    # That playlist names an encrypted job's key file between double quotes, and on a host without a
    # memory file the key file lies under this path: with a quote in it such a job would be refused
    # after its first rendition was encoded.
    if KEY_PATH_FORBIDDEN in text:
        raise SettingsInvalid(SCRATCH_DIR_VARIABLE, "must not hold a double quote")
    scratch_root = Path(text)
    if not scratch_root.is_absolute():
        raise SettingsInvalid(SCRATCH_DIR_VARIABLE, "must be an absolute path")
    if not _is_directory(scratch_root):
        raise SettingsInvalid(SCRATCH_DIR_VARIABLE, "must be an existing directory")
    # A job makes its own directory in this one only after its preflight: where the process may not,
    # every job would fail there, so the process is refused at its start.
    if not os.access(scratch_root, SCRATCH_DIR_ACCESS):
        raise SettingsInvalid(SCRATCH_DIR_VARIABLE, "must be writable and searchable by this process")
    return scratch_root


def _system_temporary_directory() -> str:
    """The system's temporary directory, which a start must have whatever its scratch directory is: the
    ffmpeg version check and the preflight make their working directories there, so without one every
    job would fail. It is asked for at every start, and a start without one is refused."""
    try:
        directory: str | None = tempfile.gettempdir()
    except OSError:
        # The system's own error lists the directories it tried. The refusal is raised once this block
        # has ended: raised inside it, it would carry that error as its context, hidden but there.
        directory = None
    if directory is None:
        raise SettingsInvalid(TEMPORARY_DIR_VARIABLE, "must name a usable temporary directory: the system has none")
    return directory


def _is_directory(path: Path) -> bool:
    """False as well for a path the system refuses to look up (a name too long, a parent it may not enter)."""
    try:
        return path.is_dir()
    except OSError:
        return False
