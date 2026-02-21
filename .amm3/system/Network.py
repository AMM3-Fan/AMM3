#!/usr/bin/env python3
"""
Network.py - External model gateway for AMM3.

Gemini: native function calling (tools defined via SDK, results fed back structured).
Claude: text-only (tool calls detected by AMM3.py distil()).

Protocol: JSON lines over TCP on port 8889.
"""

import asyncio
import json
import os
import re
import sys
import signal
import time
import threading

from TrustProtocol import get_secret
from Environment import get_system_report

ENV_REPORT = get_system_report()

class C:
    """ANSI colors."""
    DIM = '\033[2m'
    BOLD = '\033[1m'
    CYAN = '\033[96m'
    YELLOW = '\033[93m'
    GREEN = '\033[92m'
    RED = '\033[91m'
    BLUE = '\033[34m'
    RESET = '\033[0m'


_AGENT_COLORS = {"gemini": C.BLUE, "claude": C.RED}

_LOG_COLORS = {
    "amm3": C.DIM,
    "gemini": C.BLUE,
    "claude": C.RED,
}

_msg_counts = {}  # per-label message counters
_tool_counts = {}  # per-label tool request counters
_amm3_first = True  # first [AMM3] (human) message reveals content


def nlog(label: str, msg: str, color: str = C.CYAN):
    ts = time.strftime('%H:%M:%S')
    print(f"{C.DIM}{ts}{C.RESET} {color}[{label}]{C.RESET} {msg}", flush=True)


def nlog_msg(message: str):
    """Log an inbound message, parsing [label]: prefixes into separate tagged lines.

    [AMM3] — First message revealed, subsequent counted.
    Others — counted only.
    """
    global _amm3_first
    for line in message.split('\n'):
        line = line.strip()
        if not line or line == '\u200b':
            continue
        m = re.match(r'^\[(\w+)(?:\s+to\s+\w+)?\]:\s*(.*)', line)
        if not m:
            continue  # continuation line — absorb silently
        label = m.group(1)
        body = m.group(2)
        key = label.lower()
        color = _LOG_COLORS.get(key, C.GREEN)
        _msg_counts[key] = _msg_counts.get(key, 0) + 1
        if key == "amm3" and _amm3_first:
            _amm3_first = False
            nlog(label, body, color)
        else:
            nlog(label, f"#{_msg_counts[key]} message", color)


def nlog_out(agent_name: str, response: str, native_calls: list = None):
    """Log outbound model response — numbered per model, split message/tool request."""
    key = agent_name.lower()
    color = _AGENT_COLORS.get(agent_name, C.CYAN)
    _msg_counts[key] = _msg_counts.get(key, 0) + 1
    nlog(agent_name, f"#{_msg_counts[key]} message", color)
    if native_calls:
        _tool_counts[key] = _tool_counts.get(key, 0) + 1
        try:
            names = [fc['name'] if isinstance(fc, dict) else fc.name for fc in native_calls]
            names = [n for n in names if n]
        except Exception:
            names = []
        label = ', '.join(names) if names else '?'
        nlog(agent_name, f"#{_tool_counts[key]} tool request: {label}", color)


# Load system prompt from kernel config
_KERNEL_CONFIG = os.path.join(os.path.dirname(__file__), '..', 'nucleus', 'amm3.json')
with open(_KERNEL_CONFIG) as _f:
    _TOOL_SYSTEM_PROMPT = json.load(_f).get("system_prompt", "")


def _format_tools() -> str:
    return (
        "<navigate>dir/file</navigate>\n"
        "<run>command</run>\n"
        "<read>file</read>\n"
        "<memorise>content</memorise>\n"
        "<write>content</write>\n"
        "<search>query</search>\n"
        "<recall>query</recall>\n"
        "<wait/>"
    )


_TOOL_SYSTEM_PROMPT = _TOOL_SYSTEM_PROMPT.replace("<tools/>", _format_tools())


# ---------------------------------------------------------------------------
# Gemini SDK Agent — native function calling enabled
# ---------------------------------------------------------------------------

# AMM3 tool definitions for Gemini SDK
def _gemini_tool_defs():
    """Build Gemini-compatible tool definitions for AMM3 tools."""
    from google.genai import types

    return types.Tool(function_declarations=[
        types.FunctionDeclaration(
            name="run",
            description="Execute a shell command and return the output.",
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The shell command to execute"}
                },
                "required": ["command"]
            }
        ),
        types.FunctionDeclaration(
            name="read",
            description="Read the contents of a file.",
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to read"}
                },
                "required": ["path"]
            }
        ),
        types.FunctionDeclaration(
            name="write",
            description="Write content to a file, creating it if it doesn't exist.",
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to write"},
                    "content": {"type": "string", "description": "Content to write"}
                },
                "required": ["path", "content"]
            }
        ),
        types.FunctionDeclaration(
            name="recall",
            description="Search session memory for past conversations and tool usage.",
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search term to find in session logs"}
                },
                "required": ["query"]
            }
        ),
        types.FunctionDeclaration(
            name="wait",
            description="Continuation signal — indicates you have more to say and want another turn.",
            parameters_json_schema={
                "type": "object",
                "properties": {},
            }
        ),
        types.FunctionDeclaration(
            name="check",
            description="Check file metadata (existence, size, modified time) without reading content.",
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file or directory to check"}
                },
                "required": ["path"]
            }
        ),
        types.FunctionDeclaration(
            name="search",
            description="Search file contents recursively from current directory. Returns matching lines.",
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Text or pattern to search for in files"}
                },
                "required": ["query"]
            }
        ),
        types.FunctionDeclaration(
            name="navigate",
            description="Change working directory. Semantic cd.",
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path to navigate to"}
                },
                "required": ["path"]
            }
        ),
        types.FunctionDeclaration(
            name="report",
            description="Report model status to the system. Logged to monitor and session memory only.",
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "status": {"type": "string", "description": "Status label (e.g. ACTIVE, INACTIVE, READY)"},
                    "message": {"type": "string", "description": "Status message"}
                },
                "required": ["status", "message"]
            }
        ),
    ])


class GeminiAgent:
    """Gemini agent — native function calling enabled."""

    def __init__(self):
        self.client = None
        self.chat = None
        self.running = False
        self._types = None  # google.genai.types module

    def start(self) -> bool:
        api_key = get_secret("GEMINI_API_KEY")
        if not api_key:
            nlog("gemini", "No GEMINI_API_KEY found", C.BLUE)
            return False

        try:
            from google import genai
            from google.genai import types
            self._types = types

            self.client = genai.Client(api_key=api_key)
            self.chat = self.client.chats.create(
                model="gemini-2.0-flash",
                config=types.GenerateContentConfig(
                    system_instruction=_TOOL_SYSTEM_PROMPT,
                    tools=[_gemini_tool_defs()],
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(
                        disable=True
                    ),
                ),
            )
            self.running = True
            nlog("gemini", "ready (native tools)", C.BLUE)
            return True
        except Exception as e:
            nlog("gemini", f"Failed to initialize: {e}", C.BLUE)
            return False

    def stop(self):
        self.running = False
        self.chat = None
        self.client = None

    def send_message(self, message, interrupt_event: threading.Event, chunk_callback=None):
        """Send a message. Returns (text, function_calls_or_None).

        message can be a str or a list of Parts (for tool results).
        function_calls is a list of {"name": ..., "args": {...}} dicts if
        the model requested tool use, else None.
        """
        if not self.running or not self.chat:
            return "[Gemini not running]", None

        try:
            import warnings
            response_text = ""
            function_calls = []
            sdk_warnings = []

            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always")
                for chunk in self.chat.send_message_stream(message):
                    if interrupt_event.is_set():
                        response_text += " [interrupted]"
                        break
                    # Extract text and function calls from parts directly
                    if chunk.candidates and chunk.candidates[0].content and chunk.candidates[0].content.parts:
                        for part in chunk.candidates[0].content.parts:
                            if hasattr(part, 'text') and part.text:
                                response_text += part.text
                                if chunk_callback:
                                    chunk_callback(part.text)
                            elif hasattr(part, 'function_call') and part.function_call:
                                fc = part.function_call
                                function_calls.append({
                                    "name": fc.name,
                                    "args": dict(fc.args) if fc.args else {}
                                })

            # Translate and report SDK library warnings (local genai package, not API)
            for w in caught_warnings:
                msg = str(w.message)
                if "non-text parts" in msg:
                    tools = [c["name"] for c in function_calls] if function_calls else []
                    nlog("AMM3", f"GenAI library: mixed response — text + native tool calls {tools}", C.DIM)
                else:
                    nlog("AMM3", f"GenAI library: {msg}", C.DIM)

            if function_calls:
                return response_text, function_calls

            return response_text or "[Gemini returned empty response]", None
        except Exception as e:
            return f"[Gemini error: {e}]", None

    def send_tool_results(self, results: list, interrupt_event: threading.Event, chunk_callback=None):
        """Send tool execution results back to Gemini.

        Args:
            results: list of {"name": str, "output": str} dicts

        Returns (text, function_calls_or_None) — same as send_message.
        """
        types = self._types
        parts = []
        for result in results:
            parts.append(types.Part.from_function_response(
                name=result["name"],
                response={"result": result["output"]}
            ))

        # Pass parts list directly — chat.send_message_stream manages roles
        return self.send_message(parts, interrupt_event, chunk_callback)


# ---------------------------------------------------------------------------
# Claude SDK Agent (text-only, no tool declarations)
# ---------------------------------------------------------------------------

class ClaudeAgent:
    """Claude agent — text-only, no native function calling."""

    MAX_HISTORY = 50

    def __init__(self):
        self.client = None
        self.running = False
        self.history = []

    def start(self) -> bool:
        api_key = get_secret("ANTHROPIC_API_KEY")
        if not api_key:
            nlog("claude", "No ANTHROPIC_API_KEY found", C.RED)
            return False

        try:
            from anthropic import Anthropic
            self.client = Anthropic(api_key=api_key)
            self.running = True
            nlog("claude", "ready", C.RED)
            return True
        except Exception as e:
            nlog("claude", f"Failed to initialize: {e}", C.RED)
            return False

    def stop(self):
        self.running = False
        self.client = None
        self.history = []

    def send_message(self, message: str, interrupt_event: threading.Event, chunk_callback=None) -> str:
        if not self.running or not self.client:
            return "[Claude not running]"

        self.history.append({"role": "user", "content": message})

        # Prune history to prevent unbounded growth
        if len(self.history) > self.MAX_HISTORY:
            self.history = self.history[-self.MAX_HISTORY:]

        try:
            final_text = ""
            with self.client.messages.stream(
                model="claude-sonnet-4-5-20250929",
                max_tokens=4096,
                system=_TOOL_SYSTEM_PROMPT,
                messages=self.history,
            ) as stream:
                for text in stream.text_stream:
                    if interrupt_event.is_set():
                        final_text += " [interrupted]"
                        break
                    final_text += text
                    if chunk_callback:
                        chunk_callback(text)

            self.history.append({"role": "assistant", "content": final_text})
            return final_text

        except Exception as e:
            return f"[Claude error: {e}]"


# ---------------------------------------------------------------------------
# TCP Server
# ---------------------------------------------------------------------------

AGENT_REGISTRY = {
    "claude": ClaudeAgent,
    "gemini": GeminiAgent,
}


def _normalize_native_calls(native_calls: list) -> list:
    """Convert native SDK function calls to AMM3 tool call dicts."""
    calls = []
    for fc in native_calls:
        name = fc["name"]
        args = fc["args"]
        if name == "run":
            calls.append({"tool": "run", "command": args.get("command", "")})
        elif name == "read":
            calls.append({"tool": "read", "path": args.get("path", "")})
        elif name == "write":
            calls.append({"tool": "write", "path": args.get("path", ""), "content": args.get("content", "")})
        elif name == "recall":
            calls.append({"tool": "recall", "query": args.get("query", "")})
        elif name == "wait":
            calls.append({"tool": "wait"})
        elif name == "check":
            calls.append({"tool": "check", "path": args.get("path", "")})
        elif name == "search":
            calls.append({"tool": "search", "query": args.get("query", "")})
        elif name == "navigate":
            calls.append({"tool": "navigate", "path": args.get("path", "")})
        elif name == "report":
            calls.append({"tool": "report", "status": args.get("status", ""), "message": args.get("message", "")})
    return calls


class AgentServer:
    """TCP server managing remote agents on behalf of multiple AMM3 instances.

    Each connected instance gets its own 'room' with isolated agents.
    Rooms are keyed by instance_id (identity hash from AMM3.py).
    """

    def __init__(self, host='127.0.0.1', port=8889):
        self.host = host
        self.port = port
        self.rooms = {}  # instance_id -> {"agents": {}, "interrupts": {}, "writer": writer, "label": str}
        self.server = None
        self.shutdown_event = asyncio.Event()
        self._grace_task = None  # 30s grace period after last room empties

    async def _send(self, writer, msg: dict):
        line = json.dumps(msg) + '\n'
        writer.write(line.encode())
        await writer.drain()

    def _room_count(self) -> int:
        return len(self.rooms)

    async def _start_grace_timer(self):
        """After the last room disconnects, wait 30s then shut down."""
        try:
            await asyncio.sleep(30)
            if self._room_count() == 0:
                nlog("Network", "shutting down (no instances)", C.DIM)
                self.shutdown_event.set()
        except asyncio.CancelledError:
            pass

    def _cancel_grace(self):
        if self._grace_task and not self._grace_task.done():
            self._grace_task.cancel()
            self._grace_task = None

    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle a single AMM3 instance connection.

        First message must be a register action with instance_id.
        All subsequent messages are scoped to that instance's room.
        """
        instance_id = None

        try:
            # Wait for register message
            data = await reader.readline()
            if not data:
                return
            line = data.decode().strip()
            if not line:
                return

            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                await self._send(writer, {"status": "error", "error": "invalid JSON"})
                return

            if msg.get("action") != "register" or not msg.get("instance_id"):
                await self._send(writer, {"status": "error", "error": "first message must be register with instance_id"})
                return

            instance_id = msg["instance_id"]
            label = msg.get("label", instance_id[:4])

            # Cancel grace timer — a new instance connected
            self._cancel_grace()

            self.rooms[instance_id] = {
                "agents": {},
                "interrupts": {},
                "writer": writer,
                "label": label,
            }

            nlog("AMM3", f"instance {instance_id[:4]} registered ({self._room_count()} active)", C.DIM)
            await self._send(writer, {"status": "ok", "action": "registered", "instance_id": instance_id})

            # Main command loop for this instance
            while not self.shutdown_event.is_set():
                data = await reader.readline()
                if not data:
                    break

                line = data.decode().strip()
                if not line:
                    continue

                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    await self._send(writer, {"status": "error", "error": "invalid JSON"})
                    continue

                action = msg.get("action")
                agent_name = msg.get("agent", "")

                # Guard: room may have been deleted by concurrent disconnect
                room = self.rooms.get(instance_id)
                if not room:
                    nlog("AMM3", f"instance {instance_id[:4]} room gone — closing", C.DIM)
                    break

                if action == "start_agent":
                    await self._handle_start(writer, room, agent_name)
                elif action == "stop_agent":
                    await self._handle_stop(writer, room, agent_name)
                elif action == "send":
                    message = msg.get("message", "")
                    await self._handle_send(writer, room, agent_name, message)
                elif action == "tool_result":
                    results = msg.get("results", [])
                    await self._handle_tool_result(writer, room, agent_name, results)
                elif action == "tool_denied":
                    tools = msg.get("tools", [])
                    color = _AGENT_COLORS.get(agent_name, C.CYAN)
                    nlog("AMM3", f"Denied — {', '.join(tools) if tools else 'tool'}", C.RED)
                    await self._send(writer, {"status": "ok", "action": "denied_ack"})
                elif action == "interrupt":
                    await self._handle_interrupt(writer, room, agent_name)
                elif action == "disconnect":
                    await self._handle_disconnect(writer, instance_id)
                    break
                elif action == "shutdown":
                    await self._handle_shutdown(writer, instance_id)
                    break
                else:
                    await self._send(writer, {"status": "error", "error": f"unknown action: {action}"})

        except (ConnectionResetError, BrokenPipeError, asyncio.IncompleteReadError):
            if instance_id:
                nlog("AMM3", f"instance {instance_id[:4]} disconnected (connection lost)", C.DIM)
        finally:
            # Clean up this instance's room
            if instance_id and instance_id in self.rooms:
                room = self.rooms[instance_id]
                for name, agent in list(room["agents"].items()):
                    nlog(name, f"stopping (instance {instance_id[:4]} left)", _AGENT_COLORS.get(name, C.CYAN))
                    await asyncio.to_thread(agent.stop)
                del self.rooms[instance_id]
                nlog("AMM3", f"instance {instance_id[:4]} cleaned up ({self._room_count()} active)", C.DIM)

            if not writer.is_closing():
                writer.close()
                try:
                    await writer.wait_closed()
                except Exception:
                    pass

            # Shut down immediately if no rooms left
            if self._room_count() == 0:
                nlog("Network", "shutting down (no instances)", C.DIM)
                self.shutdown_event.set()

    async def _handle_start(self, writer, room: dict, agent_name: str):
        agents = room["agents"]
        if agent_name in agents:
            nlog(agent_name, "already running", _AGENT_COLORS.get(agent_name, C.CYAN))
            await self._send(writer, {"status": "ok", "agent": agent_name, "action": "started", "info": "already running"})
            return

        agent_cls = AGENT_REGISTRY.get(agent_name)
        if not agent_cls:
            nlog("error", f"unknown agent: {agent_name}", C.RED)
            await self._send(writer, {"status": "error", "agent": agent_name, "error": f"unknown agent: {agent_name}"})
            return
        agent = agent_cls()

        success = await asyncio.to_thread(agent.start)

        if success:
            agents[agent_name] = agent
            room["interrupts"][agent_name] = threading.Event()
            await self._send(writer, {"status": "ok", "agent": agent_name, "action": "started"})
        else:
            nlog(agent_name, "FAILED to start", _AGENT_COLORS.get(agent_name, C.CYAN))
            await self._send(writer, {"status": "error", "agent": agent_name, "error": "failed to start (check API key)"})

    async def _handle_stop(self, writer, room: dict, agent_name: str):
        agents = room["agents"]
        if agent_name not in agents:
            nlog(agent_name, "not running, can't stop", _AGENT_COLORS.get(agent_name, C.CYAN))
            await self._send(writer, {"status": "error", "agent": agent_name, "error": "not running"})
            return

        nlog(agent_name, "stopping...", _AGENT_COLORS.get(agent_name, C.CYAN))
        agent = agents[agent_name]
        await asyncio.to_thread(agent.stop)
        del agents[agent_name]
        del room["interrupts"][agent_name]
        nlog(agent_name, "stopped", _AGENT_COLORS.get(agent_name, C.CYAN))
        await self._send(writer, {"status": "ok", "agent": agent_name, "action": "stopped"})

    async def _handle_send(self, writer, room: dict, agent_name: str, message: str):
        agents = room["agents"]
        if agent_name not in agents:
            nlog(agent_name, "not running, can't send", _AGENT_COLORS.get(agent_name, C.CYAN))
            await self._send(writer, {"status": "error", "agent": agent_name, "error": "not running"})
            return

        agent = agents[agent_name]
        interrupt_event = room["interrupts"][agent_name]
        interrupt_event.clear()

        nlog_msg(message)
        await self._send(writer, {"status": "thinking", "agent": agent_name})

        loop = asyncio.get_event_loop()

        def chunk_callback(text):
            future = asyncio.run_coroutine_threadsafe(
                self._send(writer, {"status": "chunk", "agent": agent_name, "text": text}),
                loop,
            )
            try:
                future.result(timeout=5)
            except TimeoutError:
                nlog(agent_name, "chunk callback timed out", C.YELLOW)

        result = await asyncio.to_thread(agent.send_message, message, interrupt_event, chunk_callback)

        # GeminiAgent returns (text, function_calls), ClaudeAgent returns str
        if isinstance(result, tuple):
            response, native_calls = result
        else:
            response, native_calls = result, None

        if interrupt_event.is_set():
            nlog_out(agent_name, response)
            await self._send(writer, {"status": "interrupted", "agent": agent_name, "response": response})
        elif native_calls:
            calls = _normalize_native_calls(native_calls)
            nlog_out(agent_name, response, native_calls)
            await self._send(writer, {
                "status": "tool_calls", "agent": agent_name,
                "calls": calls, "text": response or ""
            })
        else:
            response = response or ""
            nlog_out(agent_name, response)
            await self._send(writer, {"status": "done", "agent": agent_name, "response": response})

    async def _handle_tool_result(self, writer, room: dict, agent_name: str, results: list):
        """Feed tool execution results back to an agent (native tool calling)."""
        agents = room["agents"]
        if agent_name not in agents:
            await self._send(writer, {"status": "error", "agent": agent_name, "error": "not running"})
            return

        agent = agents[agent_name]
        if not hasattr(agent, 'send_tool_results'):
            await self._send(writer, {"status": "error", "agent": agent_name, "error": "agent does not support native tool results"})
            return

        interrupt_event = room["interrupts"][agent_name]
        interrupt_event.clear()

        tool_names = [r.get("name", "?") for r in results if isinstance(r, dict)]
        nlog("AMM3", f"Allowed — {', '.join(tool_names) if tool_names else 'tool'}", C.DIM)
        await self._send(writer, {"status": "thinking", "agent": agent_name})

        loop = asyncio.get_event_loop()

        def chunk_callback(text):
            future = asyncio.run_coroutine_threadsafe(
                self._send(writer, {"status": "chunk", "agent": agent_name, "text": text}),
                loop,
            )
            try:
                future.result(timeout=5)
            except TimeoutError:
                nlog(agent_name, "chunk callback timed out", C.YELLOW)

        result = await asyncio.to_thread(agent.send_tool_results, results, interrupt_event, chunk_callback)
        response, native_calls = result

        if interrupt_event.is_set():
            nlog_out(agent_name, response)
            await self._send(writer, {"status": "interrupted", "agent": agent_name, "response": response})
        elif native_calls:
            calls = _normalize_native_calls(native_calls)
            nlog_out(agent_name, response, native_calls)
            await self._send(writer, {
                "status": "tool_calls", "agent": agent_name,
                "calls": calls, "text": response or ""
            })
        else:
            response = response or ""
            nlog_out(agent_name, response)
            await self._send(writer, {"status": "done", "agent": agent_name, "response": response})

    async def _handle_interrupt(self, writer, room: dict, agent_name: str):
        if agent_name in room["interrupts"]:
            nlog(agent_name, "INTERRUPT requested", _AGENT_COLORS.get(agent_name, C.CYAN))
            room["interrupts"][agent_name].set()

    async def _handle_disconnect(self, writer, instance_id: str):
        """Graceful disconnect — clean up this instance only, don't shut down server."""
        room = self.rooms.pop(instance_id, None)
        if room:
            for name, agent in list(room["agents"].items()):
                nlog(name, f"stopping (instance {instance_id[:4]})", _AGENT_COLORS.get(name, C.CYAN))
                await asyncio.to_thread(agent.stop)
        await self._send(writer, {"status": "ok", "action": "disconnected"})
        nlog("AMM3", f"instance {instance_id[:4]} disconnected ({self._room_count()} active)", C.DIM)

    async def _handle_shutdown(self, writer, instance_id: str):
        """AMM3 commanded shutdown. Stops this room; kills server if last instance."""
        room = self.rooms.pop(instance_id, None)
        if room:
            for name, agent in list(room["agents"].items()):
                nlog(name, f"stopping", _AGENT_COLORS.get(name, C.CYAN))
                await asyncio.to_thread(agent.stop)
        is_last = self._room_count() == 0
        await self._send(writer, {"status": "ok", "action": "shutdown_ack"})
        if is_last:
            self.shutdown_event.set()
        else:
            nlog("AMM3", f"instance {instance_id[:4]} disconnected ({self._room_count()} active)", C.DIM)

    async def run(self):
        try:
            self.server = await asyncio.start_server(
                self.handle_client, self.host, self.port
            )
        except OSError as e:
            if "Address already in use" in str(e):
                nlog("Network", "already running (port in use) — exiting", C.DIM)
                return
            raise

        print(f"{C.BOLD}━━━ AMM3 Network ━━━{C.RESET}", flush=True)
        print(f"{C.DIM}Gemini: native tools | Claude: text-only | Multi-tenant{C.RESET}", flush=True)
        nlog("Network", f"listening on {self.host}:{self.port}", C.DIM)

        # Start grace timer — if no one connects within 30s, shut down
        self._grace_task = asyncio.ensure_future(self._start_grace_timer())

        async with self.server:
            try:
                await self.shutdown_event.wait()
            except asyncio.CancelledError:
                pass
            finally:
                self.server.close()
                await self.server.wait_closed()
                # Stop any agents still running (e.g. ungraceful disconnects)
                for iid, room in list(self.rooms.items()):
                    for name, agent in list(room["agents"].items()):
                        nlog(name, f"stopping (instance {iid[:4]})", _AGENT_COLORS.get(name, C.CYAN))
                        agent.stop()
                self.rooms.clear()


def main():
    server = AgentServer()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def sigint_handler():
        server.shutdown_event.set()

    loop.add_signal_handler(signal.SIGINT, sigint_handler)
    loop.add_signal_handler(signal.SIGTERM, sigint_handler)

    try:
        loop.run_until_complete(server.run())
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        loop.close()


if __name__ == "__main__":
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    main()
