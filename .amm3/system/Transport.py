#!/usr/bin/env python3
"""
Transport.py - Communication layer for AMM3.

MCPTransport: JSON-RPC client for Toolbox.py subprocess (stdio).
NetworkTransport: TCP client for Network.py remote agents.

Both use request_id-based multiplexing to prevent cross-talk
in parallel mode.
"""

import asyncio
import json
import os
import subprocess
import sys
import uuid


class MCPTransport:
    """Multiplexed JSON-RPC client for Toolbox.py.

    Concurrent requests: multiple models can have in-flight tool calls simultaneously.
    A background reader task matches responses to pending futures by request ID.
    """

    def __init__(self):
        self._proc = None
        self._req_id = 0
        self._pending: dict[int, asyncio.Future] = {}
        self._reader_task = None
        self._write_lock = asyncio.Lock()

    async def start(self) -> bool:
        """Spawn Toolbox.py as an async subprocess and start the reader."""
        from system.Toolbox import tool_log

        toolbox_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'system', 'Toolbox.py')
        self._proc = await asyncio.create_subprocess_exec(
            sys.executable, toolbox_path,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        # Start background reader
        self._reader_task = asyncio.create_task(self._reader_loop())

        # Send initialize
        resp = await self._request("initialize", {})
        if resp and "result" in resp:
            tool_log("[AMM3]: MCP Toolbox connected (multiplexed)")
        return True

    async def _reader_loop(self):
        """Background task: reads responses from Toolbox stdout, resolves matching futures."""
        try:
            while True:
                line = await self._proc.stdout.readline()
                if not line:
                    break  # Toolbox exited

                try:
                    resp = json.loads(line.decode().strip())
                except json.JSONDecodeError:
                    continue

                req_id = resp.get("id")
                if req_id is not None and req_id in self._pending:
                    future = self._pending.pop(req_id)
                    if not future.done():
                        future.set_result(resp)
        except (asyncio.CancelledError, Exception):
            pass
        finally:
            # Resolve any remaining pending futures with errors
            for future in self._pending.values():
                if not future.done():
                    future.set_result({"error": "Toolbox connection lost"})
            self._pending.clear()

    async def _request(self, method: str, params: dict) -> dict:
        """Send a JSON-RPC request and await its response via the reader."""
        if not self._proc or self._proc.returncode is not None:
            return {"error": "Toolbox not running"}

        self._req_id += 1
        req_id = self._req_id
        request = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params,
        }

        # Create future before sending (reader might be fast)
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        self._pending[req_id] = future

        # Write with lock — multiple callers may send concurrently
        async with self._write_lock:
            line = json.dumps(request) + "\n"
            self._proc.stdin.write(line.encode())
            await self._proc.stdin.drain()

        # Await the response (matched by ID in reader_loop)
        # Toolbox timeout is 30s; give 5s margin for eval + overhead
        try:
            return await asyncio.wait_for(future, timeout=35)
        except asyncio.TimeoutError:
            self._pending.pop(req_id, None)
            return {"error": "Toolbox request timed out"}

    async def call_tool(self, name: str, arguments: dict, is_local: bool = False,
                        targets: list = None, model: str = "",
                        trust_tier: str = None, knowledge_count: int = 0) -> dict:
        """Call a tool on the MCP server. Returns full response dict with verdict.

        Response: {"verdict": {"action", "risk", "reasoning"}, "content": [...] or None, "cwd": "~"}
        If verdict is "allow", content is populated (already executed).
        If verdict is "confirm" or "block", content is None.
        """
        if trust_tier is None:
            trust_tier = "local" if is_local else "remote"
        params = {"name": name, "arguments": arguments, "is_local": is_local,
                  "model": model, "trust_tier": trust_tier}
        if targets:
            params["targets"] = targets
        if knowledge_count:
            params["knowledge_count"] = knowledge_count
        resp = await self._request("tools/call", params)

        if "error" in resp:
            return {
                "verdict": {"action": "block", "risk": "high", "reasoning": f"MCP Error: {resp['error']}"},
                "content": None,
                "cwd": "~"
            }

        return resp.get("result", {})

    async def execute_tool(self, name: str, arguments: dict, model: str = "") -> tuple[str, str]:
        """Post-confirmation execution. Returns (result_text, cwd)."""
        resp = await self._request("tools/execute", {"name": name, "arguments": arguments, "model": model})

        if "error" in resp:
            return f"[MCP Error: {resp['error']}]", "~"

        result = resp.get("result", {})
        cwd = result.get("cwd", "~")
        content = result.get("content", [])
        text = content[0].get("text", "[No output]") if content else "[No output]"
        return text, cwd

    async def stop(self):
        """Shut down the Toolbox subprocess and reader task."""
        # Send explicit shutdown command — Toolbox acks and exits cleanly
        if self._proc and self._proc.returncode is None:
            try:
                await asyncio.wait_for(self._request("shutdown", {}), timeout=3.0)
            except (asyncio.TimeoutError, Exception):
                pass

        if self._reader_task and not self._reader_task.done():
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass

        if self._proc and self._proc.returncode is None:
            self._proc.stdin.close()
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                self._proc.terminate()
                try:
                    await asyncio.wait_for(self._proc.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    self._proc.kill()
        self._proc = None


class NetworkTransport:
    """TCP client for Network.py remote agents.

    Each instance registers with a unique instance_id (identity hash) so that
    multiple AMM3 instances can share a single Network.py server.
    """

    NETWORK_SCRIPT = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'system', 'Network.py')

    def __init__(self, instance_id: str = "default",
                 interrupt_event=None, on_chunk=None):
        """
        Args:
            instance_id: Identity hash for multi-instance isolation.
            interrupt_event: threading.Event checked during streaming.
            on_chunk: callable(agent_name, status, data) for streaming display.
        """
        self.instance_id = instance_id
        self.reader = None
        self.writer = None
        self.connected = False
        self._read_lock = asyncio.Lock()
        self._interrupt_event = interrupt_event
        self._on_chunk = on_chunk

    async def _ensure_running(self):
        if self.connected:
            return True

        # Try connecting first — Network.py may already be running
        connected_existing = False
        for _ in range(3):
            try:
                self.reader, self.writer = await asyncio.open_connection('127.0.0.1', 8889)
                connected_existing = True
                break
            except (ConnectionRefusedError, OSError):
                await asyncio.sleep(0.2)

        if not connected_existing:
            # Launch Network.py in a new Terminal, then restore focus to AMM3 window
            cmd = f'{sys.executable} {self.NETWORK_SCRIPT}'
            applescript = f'''tell application "Terminal"
    set origWindow to front window
    do script "{cmd}"
    delay 0.3
    set frontmost of origWindow to true
end tell'''
            subprocess.run(['osascript', '-e', applescript], capture_output=True)

            for _ in range(30):
                try:
                    self.reader, self.writer = await asyncio.open_connection('127.0.0.1', 8889)
                    break
                except (ConnectionRefusedError, OSError):
                    await asyncio.sleep(0.2)
            else:
                return False

        # Register this instance
        await self._send({"action": "register", "instance_id": self.instance_id})
        resp = await self._read_response()
        if resp.get("status") == "ok":
            self.connected = True
            return True
        else:
            return False

    async def _send(self, msg: dict):
        line = json.dumps(msg) + '\n'
        self.writer.write(line.encode())
        await self.writer.drain()

    async def _read_response(self) -> dict:
        data = await self.reader.readline()
        if not data:
            raise ConnectionError("Network.py disconnected")
        return json.loads(data.decode().strip())

    async def start_agent(self, agent_name: str) -> tuple[bool, str]:
        if not await self._ensure_running():
            return False, "Failed to start network server"

        async with self._read_lock:
            await self._send({"action": "start_agent", "agent": agent_name})
            resp = await self._read_response()

        if resp.get("status") == "ok":
            info = resp.get("info", "")
            if info == "already running":
                return True, f"{agent_name} is already running"
            return True, f"{agent_name} has joined the chat"
        else:
            return False, resp.get("error", "unknown error")

    async def stop_agent(self, agent_name: str) -> tuple[bool, str]:
        if not self.connected:
            return False, "Network not connected"

        async with self._read_lock:
            await self._send({"action": "stop_agent", "agent": agent_name})
            resp = await self._read_response()

        if resp.get("status") == "ok":
            return True, f"Removed {agent_name}"
        else:
            return False, resp.get("error", "unknown error")

    async def send(self, agent_name: str, message: str) -> tuple[str, list | None]:
        """Send a message to a remote agent.

        Returns (response_text, native_calls_or_None).
        """
        if not self.connected:
            return f"[{agent_name} not connected]", None

        async with self._read_lock:
            await self._send({"action": "send", "agent": agent_name, "message": message})
            return await self._recv_response(agent_name)

    async def send_tool_results(self, agent_name: str, results: list) -> tuple[str, list | None]:
        """Send tool execution results back to a native-tool agent.

        Args:
            results: list of {"name": str, "output": str} dicts

        Returns (response_text, native_calls_or_None) — same as send().
        """
        if not self.connected:
            return f"[{agent_name} not connected]", None

        async with self._read_lock:
            await self._send({"action": "tool_result", "agent": agent_name, "results": results})
            return await self._recv_response(agent_name)

    async def send_tool_denied(self, agent_name: str, tool_names: list) -> None:
        """Notify Network.py that tool calls were denied (for tracing)."""
        if not self.connected:
            return
        async with self._read_lock:
            await self._send({"action": "tool_denied", "agent": agent_name, "tools": tool_names})
            try:
                await asyncio.wait_for(self._read_response(agent_name), timeout=2.0)
            except (asyncio.TimeoutError, Exception):
                pass

    async def _recv_response(self, agent_name: str) -> tuple[str, list | None]:
        """Read streaming response from Network.py. Returns (text, native_calls_or_None)."""
        while True:
            if self._interrupt_event and self._interrupt_event.is_set():
                await self._send({"action": "interrupt", "agent": agent_name})
                self._interrupt_event.clear()

            try:
                resp = await asyncio.wait_for(self._read_response(), timeout=0.5)
            except asyncio.TimeoutError:
                continue

            status = resp.get("status")

            if status == "thinking":
                if self._on_chunk:
                    self._on_chunk(agent_name, "thinking", None)

            elif status == "chunk":
                if self._on_chunk:
                    self._on_chunk(agent_name, "chunk", resp.get("text", ""))

            elif status == "done":
                response = resp.get("response", "")
                if self._on_chunk:
                    self._on_chunk(agent_name, "done", response)
                return response, None

            elif status == "tool_calls":
                # Native tool calls from Gemini
                text = resp.get("text", "")
                calls = resp.get("calls", [])
                if self._on_chunk:
                    self._on_chunk(agent_name, "tool_calls", text)
                return text, calls

            elif status == "interrupted":
                response = resp.get("response", "")
                if self._on_chunk:
                    self._on_chunk(agent_name, "interrupted", response)
                return response + " [interrupted]", None

            elif status == "error":
                error = resp.get("error", "unknown error")
                if self._on_chunk:
                    self._on_chunk(agent_name, "error", error)
                return f"[ERROR] {error}", None

    async def shutdown(self):
        """Disconnect this instance from Network.py. Server stays alive for other instances."""
        if not self.connected:
            return

        try:
            async with self._read_lock:
                await self._send({"action": "shutdown", "instance_id": self.instance_id})
                try:
                    await asyncio.wait_for(self._read_response(), timeout=5.0)
                except (asyncio.TimeoutError, ConnectionError):
                    pass
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

        if self.writer and not self.writer.is_closing():
            self.writer.close()
            try:
                await self.writer.wait_closed()
            except Exception:
                pass

        self.connected = False
        self.reader = None
        self.writer = None

    def has_agents(self) -> bool:
        return self.connected
