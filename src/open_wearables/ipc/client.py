from __future__ import annotations

import asyncio
import inspect
import itertools
import json
from collections import defaultdict
from typing import Any, Awaitable, Callable, Dict, Optional, Set

from websockets import ConnectionClosed
from websockets.client import WebSocketClientProtocol, connect

from .errors import (
    IPCClosedError,
    IPCProtocolError,
    IPCRemoteError,
    IPCStreamError,
)
from .models import StreamEvent

EventCallback = Callable[[dict[str, Any]], Optional[Awaitable[None]]]


class _StreamEnd:
    pass


_STREAM_END = _StreamEnd()


class StreamSubscription:
    """Represents a live subscription and supports async iteration."""

    def __init__(
        self,
        client: "OpenEarableIPCClient",
        subscription_id: int,
        stream: str,
        device_id: str,
        queue: "asyncio.Queue[Any]",
    ) -> None:
        self._client = client
        self.subscription_id = subscription_id
        self.stream = stream
        self.device_id = device_id
        self._queue = queue
        self._closed = False

    def __aiter__(self) -> "StreamSubscription":
        return self

    async def __anext__(self) -> StreamEvent:
        item = await self._queue.get()
        if item is _STREAM_END:
            self._closed = True
            raise StopAsyncIteration
        if isinstance(item, Exception):
            self._closed = True
            raise item
        return item

    async def close(self) -> dict[str, Any]:
        """Unsubscribe and close the local iterator."""
        if self._closed:
            return {"subscription_id": self.subscription_id, "cancelled": True}

        self._closed = True
        return await self._client.unsubscribe(self.subscription_id)


class OpenEarableIPCClient:
    """Async client for OpenEarable WebSocket IPC daemon."""

    def __init__(
        self,
        uri: str = "ws://127.0.0.1:8765/ws",
        request_timeout: float = 10.0,
        subscription_queue_size: int = 0,
    ) -> None:
        self.uri = uri
        self.request_timeout = request_timeout
        self.subscription_queue_size = subscription_queue_size

        self._ws: Optional[WebSocketClientProtocol] = None
        self._receiver_task: Optional[asyncio.Task[None]] = None
        self._next_request_id = itertools.count(1)
        self._pending: Dict[int, "asyncio.Future[Any]"] = {}
        self._callbacks: Dict[str, Set[EventCallback]] = defaultdict(set)
        self._subscriptions: Dict[int, "asyncio.Queue[Any]"] = {}
        self._waiters: Set["asyncio.Future[dict[str, Any]]"] = set()

    async def __aenter__(self) -> "OpenEarableIPCClient":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    @property
    def is_connected(self) -> bool:
        return self._ws is not None

    async def connect(self, wait_for_ready: bool = True) -> Optional[dict[str, Any]]:
        """Open WebSocket connection and optionally wait for `ready` event."""
        if self._ws is not None:
            return None

        self._ws = await connect(self.uri)
        self._receiver_task = asyncio.create_task(self._receiver_loop())

        if wait_for_ready:
            return await self.wait_for_event("ready", timeout=self.request_timeout)
        return None

    async def close(self) -> None:
        """Close connection and tear down pending requests/subscriptions."""
        ws = self._ws
        self._ws = None

        if ws is not None:
            await ws.close()

        if self._receiver_task is not None:
            try:
                await self._receiver_task
            finally:
                self._receiver_task = None

        for future in list(self._pending.values()):
            if not future.done():
                future.set_exception(IPCClosedError("IPC client closed"))
        self._pending.clear()

        for queue in self._subscriptions.values():
            await queue.put(_STREAM_END)
        self._subscriptions.clear()

        for waiter in list(self._waiters):
            if not waiter.done():
                waiter.set_exception(IPCClosedError("IPC client closed"))
        self._waiters.clear()

    def on_event(self, event: str, callback: EventCallback) -> None:
        """Register callback for a server event.

        Use `event='*'` to receive all events.
        """
        self._callbacks[event].add(callback)

    def remove_event_listener(self, event: str, callback: EventCallback) -> None:
        """Remove a callback for an event name."""
        self._callbacks[event].discard(callback)

    async def wait_for_event(
        self,
        event: str,
        timeout: Optional[float] = None,
    ) -> dict[str, Any]:
        """Wait for the next event with the given name."""
        loop = asyncio.get_running_loop()
        waiter: "asyncio.Future[dict[str, Any]]" = loop.create_future()
        waiter._open_earable_event_name = event  # type: ignore[attr-defined]
        self._waiters.add(waiter)

        try:
            if timeout is None:
                return await waiter
            return await asyncio.wait_for(waiter, timeout)
        finally:
            self._waiters.discard(waiter)

    async def call(self, method: str, params: Optional[dict[str, Any]] = None) -> Any:
        """Invoke a method on the IPC server and return its `result`."""
        if self._ws is None:
            raise IPCClosedError("Call attempted before connect()")

        request_id = next(self._next_request_id)
        payload = {
            "id": request_id,
            "method": method,
            "params": params or {},
        }

        loop = asyncio.get_running_loop()
        future: "asyncio.Future[Any]" = loop.create_future()
        self._pending[request_id] = future

        await self._ws.send(json.dumps(payload))

        try:
            return await asyncio.wait_for(future, timeout=self.request_timeout)
        finally:
            self._pending.pop(request_id, None)

    async def ping(self) -> dict[str, Any]:
        return await self.call("ping")

    async def methods(self) -> list[str]:
        return await self.call("methods")

    async def has_permissions(self) -> bool:
        return await self.call("has_permissions")

    async def check_and_request_permissions(self) -> bool:
        return await self.call("check_and_request_permissions")

    async def start_scan(self, check_and_request_permissions: bool = True) -> dict[str, Any]:
        return await self.call(
            "start_scan",
            {"check_and_request_permissions": check_and_request_permissions},
        )

    async def get_discovered_devices(self) -> list[dict[str, Any]]:
        return await self.call("get_discovered_devices")

    async def connect_device(
        self,
        device_id: str,
        connected_via_system: bool = False,
    ) -> dict[str, Any]:
        return await self.call(
            "connect",
            {
                "device_id": device_id,
                "connected_via_system": connected_via_system,
            },
        )

    async def connect_system_devices(
        self,
        ignored_device_ids: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {}
        if ignored_device_ids is not None:
            params["ignored_device_ids"] = ignored_device_ids
        return await self.call("connect_system_devices", params)

    async def list_connected(self) -> list[dict[str, Any]]:
        return await self.call("list_connected")

    async def disconnect(self, device_id: str) -> dict[str, Any]:
        return await self.call("disconnect", {"device_id": device_id})

    async def set_auto_connect(self, device_ids: list[str]) -> dict[str, Any]:
        return await self.call("set_auto_connect", {"device_ids": device_ids})

    async def get_wearable(self, device_id: str) -> dict[str, Any]:
        return await self.call("get_wearable", {"device_id": device_id})

    async def get_actions(self, device_id: str) -> list[str]:
        return await self.call("get_actions", {"device_id": device_id})

    async def invoke_action(
        self,
        device_id: str,
        action: str,
        args: Optional[dict[str, Any]] = None,
    ) -> Any:
        params: dict[str, Any] = {
            "device_id": device_id,
            "action": action,
        }
        if args is not None:
            params["args"] = args
        return await self.call("invoke_action", params)

    async def subscribe(
        self,
        device_id: str,
        stream: str,
        args: Optional[dict[str, Any]] = None,
    ) -> StreamSubscription:
        params: dict[str, Any] = {
            "device_id": device_id,
            "stream": stream,
        }
        if args is not None:
            params["args"] = args

        result = await self.call("subscribe", params)
        subscription_id = int(result["subscription_id"])
        queue: "asyncio.Queue[Any]" = asyncio.Queue(maxsize=self.subscription_queue_size)
        self._subscriptions[subscription_id] = queue

        return StreamSubscription(
            client=self,
            subscription_id=subscription_id,
            stream=str(result.get("stream", stream)),
            device_id=str(result.get("device_id", device_id)),
            queue=queue,
        )

    async def unsubscribe(self, subscription_id: int) -> dict[str, Any]:
        result = await self.call("unsubscribe", {"subscription_id": subscription_id})
        queue = self._subscriptions.pop(subscription_id, None)
        if queue is not None:
            await queue.put(_STREAM_END)
        return result

    async def _receiver_loop(self) -> None:
        ws = self._ws
        if ws is None:
            return

        try:
            async for raw in ws:
                message = self._decode_message(raw)
                if "id" in message and ("result" in message or "error" in message):
                    self._handle_response(message)
                elif "event" in message:
                    await self._handle_event(message)
                else:
                    raise IPCProtocolError(f"Invalid IPC message: {message}")
        except ConnectionClosed:
            pass
        finally:
            for future in list(self._pending.values()):
                if not future.done():
                    future.set_exception(IPCClosedError("WebSocket connection closed"))

            for queue in self._subscriptions.values():
                await queue.put(_STREAM_END)
            self._subscriptions.clear()

            for waiter in list(self._waiters):
                if not waiter.done():
                    waiter.set_exception(IPCClosedError("WebSocket connection closed"))
            self._waiters.clear()

    def _decode_message(self, raw: Any) -> dict[str, Any]:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")

        if not isinstance(raw, str):
            raise IPCProtocolError(f"Unexpected websocket payload type: {type(raw)!r}")

        message = json.loads(raw)
        if not isinstance(message, dict):
            raise IPCProtocolError(f"Expected JSON object message, got: {type(message)!r}")
        return message

    def _handle_response(self, message: dict[str, Any]) -> None:
        request_id = message.get("id")
        if not isinstance(request_id, int):
            raise IPCProtocolError(f"Response has non-int id: {message}")

        future = self._pending.get(request_id)
        if future is None:
            return

        if "error" in message and message["error"] is not None:
            error_payload = message["error"]
            if not isinstance(error_payload, dict):
                future.set_exception(IPCProtocolError(f"Invalid error payload: {error_payload!r}"))
                return

            future.set_exception(
                IPCRemoteError(
                    message=str(error_payload.get("message", "Unknown IPC error")),
                    error_type=error_payload.get("type"),
                    stack=error_payload.get("stack"),
                    payload=error_payload,
                )
            )
            return

        future.set_result(message.get("result"))

    async def _handle_event(self, message: dict[str, Any]) -> None:
        event_name = message.get("event")
        if not isinstance(event_name, str):
            raise IPCProtocolError(f"Event has invalid name: {message}")

        for waiter in list(self._waiters):
            target = getattr(waiter, "_open_earable_event_name", None)
            if target == event_name and not waiter.done():
                waiter.set_result(message)

        for callback in list(self._callbacks.get(event_name, set())) + list(
            self._callbacks.get("*", set())
        ):
            try:
                maybe_awaitable = callback(message)
                if inspect.isawaitable(maybe_awaitable):
                    asyncio.create_task(maybe_awaitable)
            except Exception:
                # Event listener failures should not stop the receiver loop.
                continue

        if event_name == "stream":
            subscription_id = message.get("subscription_id")
            if isinstance(subscription_id, int):
                queue = self._subscriptions.get(subscription_id)
                if queue is not None:
                    stream_event = StreamEvent(
                        subscription_id=subscription_id,
                        stream=str(message.get("stream", "")),
                        device_id=str(message.get("device_id", "")),
                        data=message.get("data"),
                        raw=message,
                    )
                    await queue.put(stream_event)
            return

        if event_name == "stream_error":
            subscription_id = message.get("subscription_id")
            if isinstance(subscription_id, int):
                queue = self._subscriptions.pop(subscription_id, None)
                if queue is not None:
                    error_message = "Stream failed"
                    maybe_error = message.get("error")
                    if isinstance(maybe_error, dict):
                        error_message = str(maybe_error.get("message", error_message))
                    elif isinstance(maybe_error, str):
                        error_message = maybe_error
                    await queue.put(IPCStreamError(error_message, event_payload=message))
                    await queue.put(_STREAM_END)
            return

        if event_name == "stream_done":
            subscription_id = message.get("subscription_id")
            if isinstance(subscription_id, int):
                queue = self._subscriptions.pop(subscription_id, None)
                if queue is not None:
                    await queue.put(_STREAM_END)
