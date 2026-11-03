"""Protocol configuration, clocks and transport shared by both runtime families."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Generic, Mapping, TypeVar, cast

from . import files
from .settings import ProtocolSettings
from .transport import HttpsTransport, Transport

SettingsT = TypeVar("SettingsT", bound=ProtocolSettings)


@dataclass(frozen=True)
class Runtime(Generic[SettingsT]):
    """What every start of this process shares: the configuration, the carrier, the clocks and the id source.

    `clock` is the aware wall-clock time the contract's timestamps are judged on; `monotonic` gives seconds
    that only move forward, for the worker's own waits.
    """

    settings: SettingsT
    transport: Transport
    clock: Callable[[], datetime]
    monotonic: Callable[[], float]
    sleep: Callable[[float], None]
    new_uuid: Callable[[], str]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _random_uuid() -> str:
    return str(uuid.uuid4())


def load_runtime(env: Mapping[str, str], settings_type: type[SettingsT]) -> Runtime[SettingsT]:
    """The process's runtime, or an error: ContractTreeInvalid for the tree, SettingsInvalid for the configuration."""
    files.verify_contract_tree()
    return Runtime(
        settings=cast(SettingsT, settings_type.from_env(env)),
        transport=HttpsTransport(),
        clock=_utc_now,
        monotonic=time.monotonic,
        sleep=time.sleep,
        new_uuid=_random_uuid,
    )


