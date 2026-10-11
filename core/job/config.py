"""The job's own configuration, read from the environment beside the protocol's (core.protocol.settings).

    VH_IMAGE_DIGEST   required: `sha256:` and the 64 lowercase hex digits of the image this worker runs
                      from. A result reports it as `encoder.image_digest`, and a container cannot learn
                      its own digest, so the deployment states it.
    VH_SCRATCH_DIR    optional: the existing directory every job makes its scratch directory in;
                      default the system's temporary directory. Either way its path holds printable
                      ASCII only, and none of the characters the stages refuse in a path.

A refusal is the protocol settings' SettingsInvalid: it names the variable and the rule, never the value.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from core.encode_command import UNSAFE_PATH_CHARACTERS
from core.protocol import schema
from core.protocol.settings import SettingsInvalid

IMAGE_DIGEST_VARIABLE = "VH_IMAGE_DIGEST"
SCRATCH_DIR_VARIABLE = "VH_SCRATCH_DIR"
# The characters a scratch path may hold: printable ASCII, the space included.
PRINTABLE_ASCII_FIRST = 0x20
PRINTABLE_ASCII_LAST = 0x7E
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
        """Read and check both variables from `env` alone; the first problem raises SettingsInvalid."""
        if not isinstance(env, Mapping):
            raise TypeError("from_env() takes the environment as a mapping")
        return cls(image_digest=_image_digest(env), scratch_root=_scratch_root(env))


def _image_digest(env: Mapping[str, str]) -> str:
    digest = env.get(IMAGE_DIGEST_VARIABLE)
    if digest is None:
        raise SettingsInvalid(IMAGE_DIGEST_VARIABLE, "is required")
    if not isinstance(digest, str):
        raise SettingsInvalid(IMAGE_DIGEST_VARIABLE, "must be a string")
    if schema.error_pointers_for_schema({"$ref": _IMAGE_DIGEST_DEFINITION}, digest):
        raise SettingsInvalid(IMAGE_DIGEST_VARIABLE, "must be `sha256:` followed by 64 lowercase hex digits")
    return digest


def _scratch_root(env: Mapping[str, str]) -> Path:
    """The directory jobs make their scratch directories in, held to what the stages accept of a path."""
    text = env.get(SCRATCH_DIR_VARIABLE, tempfile.gettempdir())
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
    scratch_root = Path(text)
    if not scratch_root.is_absolute():
        raise SettingsInvalid(SCRATCH_DIR_VARIABLE, "must be an absolute path")
    if not _is_directory(scratch_root):
        raise SettingsInvalid(SCRATCH_DIR_VARIABLE, "must be an existing directory")
    return scratch_root


def _is_directory(path: Path) -> bool:
    """False as well for a path the system refuses to look up (a name too long, a parent it may not enter)."""
    try:
        return path.is_dir()
    except OSError:
        return False
