"""The single outcome of validating one payload: accepted, or rejected at one layer for one reason."""

from __future__ import annotations

from dataclasses import dataclass

from .reasons import Layer, ReasonCode


@dataclass(frozen=True)
class Verdict:
    """`instance_path` is an RFC 6901 pointer ("" = root); None for accepted payloads and L0/L1 rejections."""

    accepted: bool
    layer: Layer | None
    reason: ReasonCode | None
    instance_path: str | None

    def __post_init__(self) -> None:
        if self.accepted and (self.layer, self.reason, self.instance_path) != (None, None, None):
            raise ValueError("an accepted verdict carries no layer, reason or instance path")
        if not self.accepted and (not isinstance(self.layer, Layer) or not isinstance(self.reason, ReasonCode)):
            raise ValueError("a rejection names its layer and reason")
        if self.instance_path is not None and not isinstance(self.instance_path, str):
            raise ValueError("instance_path is a JSON pointer string or None")

    @classmethod
    def accept(cls) -> Verdict:
        return cls(accepted=True, layer=None, reason=None, instance_path=None)

    @classmethod
    def reject(cls, layer: Layer, reason: ReasonCode, instance_path: str | None) -> Verdict:
        return cls(accepted=False, layer=layer, reason=reason, instance_path=instance_path)
