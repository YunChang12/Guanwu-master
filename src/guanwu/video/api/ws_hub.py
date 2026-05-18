from __future__ import annotations

import asyncio
from collections import defaultdict

from fastapi import WebSocket


VALID_TOPICS = {"object.updated", "relation.changed", "event.detected", "sim.collision"}


class ConnectionHub:
    def __init__(self) -> None:
        self._subs: dict[str, set[WebSocket]] = defaultdict(set)
        self._lock = asyncio.Lock()

    async def connect(self, topic: str, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self._subs[topic].add(ws)

    async def disconnect(self, topic: str, ws: WebSocket) -> None:
        async with self._lock:
            self._subs[topic].discard(ws)

    async def publish(self, topic: str, payload: dict) -> None:
        async with self._lock:
            sockets = list(self._subs[topic])
        dead: list[WebSocket] = []
        for ws in sockets:
            try:
                await ws.send_json(payload)
            except Exception:
                dead.append(ws)
        if dead:
            async with self._lock:
                for ws in dead:
                    self._subs[topic].discard(ws)


__all__ = ["ConnectionHub", "VALID_TOPICS"]
