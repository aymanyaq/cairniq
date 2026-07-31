import asyncio
import logging

from fastapi import WebSocket
from langchain_core.runnables import Runnable

from agent.logger import log_to_component

# ============================================================
# GLOBAL AGENT STATE
# ============================================================
# The compiled LangGraph agent is stored here during the FastAPI lifespan
_global_agent: Runnable | None = None

def get_agent() -> Runnable:
    """Dependency to inject the global agent instance into routes."""
    if _global_agent is None:
        raise RuntimeError("Agent not initialized. Ensure lifespan has completed.")
    return _global_agent

def set_agent(agent: Runnable):
    """Set the global agent instance during startup."""
    global _global_agent
    _global_agent = agent

# ============================================================
# WEBSOCKET MANAGER
# ============================================================
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        """Send a JSON payload to all connected clients."""
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                # Connection might be stale
                log_to_component("server", "WebSocket", f"Failed to broadcast to connection: {e}", level=logging.DEBUG)

manager = ConnectionManager()

def get_connection_manager() -> ConnectionManager:
    """Dependency to inject the connection manager into routes."""
    return manager

# ============================================================
# GRAPH MEMORY SYNC
# ============================================================
# The server's main event loop, captured at startup so graph-update broadcasts
# can be scheduled onto it even from synchronous tool code running in a worker
# thread (where there is no running loop of its own).
_main_loop: asyncio.AbstractEventLoop | None = None


def set_main_loop(loop: asyncio.AbstractEventLoop):
    """Record the server's event loop (called once during lifespan startup)."""
    global _main_loop
    _main_loop = loop


def broadcast_graph_update(data: dict):
    """Callback for graph memory updates — safe to call from any thread.

    The knowledge graph is often saved from synchronous tool code running in a
    worker thread, which has no running event loop. In that case we schedule the
    broadcast onto the captured main loop via run_coroutine_threadsafe; when
    already on the loop thread we just create_task.
    """
    msg = {"type": "graph_update", "data": data}
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(manager.broadcast(msg))
        return
    except RuntimeError:
        pass

    if _main_loop is not None and _main_loop.is_running():
        try:
            asyncio.run_coroutine_threadsafe(manager.broadcast(msg), _main_loop)
        except Exception as e:
            log_to_component("server", "WebSocket", f"graph_update schedule failed: {e}", level=logging.DEBUG)
