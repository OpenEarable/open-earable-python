from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StreamEvent:
    """A data event emitted by a subscription stream."""

    subscription_id: int
    stream: str
    device_id: str
    data: Any
    raw: dict[str, Any]
