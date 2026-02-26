from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


class IPCError(Exception):
    """Base exception for IPC client failures."""


@dataclass
class IPCRemoteError(IPCError):
    """Error returned by the IPC server."""

    message: str
    error_type: Optional[str] = None
    stack: Optional[str] = None
    payload: Optional[dict[str, Any]] = None

    def __str__(self) -> str:
        if self.error_type:
            return f"{self.error_type}: {self.message}"
        return self.message


class IPCProtocolError(IPCError):
    """Raised when messages violate the IPC protocol."""


class IPCClosedError(IPCError):
    """Raised when using a client that is not connected."""


class IPCStreamError(IPCError):
    """Raised when a stream subscription fails."""

    def __init__(self, message: str, event_payload: Optional[dict[str, Any]] = None):
        super().__init__(message)
        self.event_payload = event_payload
