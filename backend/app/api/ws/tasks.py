"""WebSocket endpoint for real-time task status updates.

Subscribes to a Redis pub/sub channel (task:<task_id>) and forwards
JSON messages to the connected browser. The Celery task publishes a
message each time the status changes.

If Redis is unavailable (e.g. tests), the connection is accepted and
immediately closed gracefully.
"""

from __future__ import annotations

import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter(tags=["ws"])


@router.websocket("/ws/tasks/{task_id}")
async def ws_task_status(websocket: WebSocket, task_id: str) -> None:
    await websocket.accept()

    try:
        from backend.app.core.config import get_settings

        import redis.asyncio as aioredis

        settings = get_settings()
        r = aioredis.Redis.from_url(settings.REDIS_URL)
        pubsub = r.pubsub()
        await pubsub.subscribe(f"task:{task_id}")

        try:
            async for message in pubsub.listen():
                if message["type"] == "message":
                    payload = message["data"]
                    text = payload.decode() if isinstance(payload, bytes) else payload
                    await websocket.send_text(text)
                    data = json.loads(text)
                    if data.get("status") in ("done", "failed"):
                        break
        except WebSocketDisconnect:
            pass
        finally:
            await pubsub.unsubscribe(f"task:{task_id}")
            await r.aclose()

    except Exception:
        # Redis not available — close cleanly
        pass

    try:
        await websocket.close()
    except Exception:
        pass
