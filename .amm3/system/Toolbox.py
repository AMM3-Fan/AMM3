#!/usr/bin/env python3
"""
Toolbox.py - MCP server for AMM3 tool execution.

JSON-RPC 2.0 over stdio. Spawned by AMM3.py as a subprocess.
Async event loop — concurrent tool execution across models.

Protocol methods:
  - initialize
  - tools/list
  - tools/call    — evaluate via TrustProtocol, execute if auto-allow
  - tools/execute — post-confirmation execution (no re-evaluation)

Exits cleanly on stdin EOF (parent crash).
"""

import asyncio
import glob as glob_mod
import json
import os
import re
import socket
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime

try:
    from TrustProtocol import evaluate
    from Graph import SemanticRecall
except ImportError:
    from system.TrustProtocol import evaluate
    from system.Graph import SemanticRecall

_recall = SemanticRecall()


_HOME = os.path.expanduser("~")
_MAX_OUTPUT = 50 * 1024   # 50 KB
_TIMEOUT = 30             # seconds
_MONITOR_ADDR = ('127.0.0.1', 8890)
_INSTANCE_ID = os.environ.get("AMM3_INSTANCE_ID", "")
_KNOWLEDGE_PATH = os.path.join(os.path.dirname(__file__), '..', 'nucleus', 'knowledge.jsonl')

# Per-model state: each model gets its own cwd and target
# Key = model_name, Value = {"cwd": str, "target": str|None}
_model_states: dict[str, dict] = {}


def _get_state(model: str) -> dict:
    """Get or create per-model state (cwd, target)."""
    if model not in _model_states:
        _model_states[model] = {"cwd": _HOME, "target": None}
    return _model_states[model]


def _short_cwd(model: str = "") -> str:
    """Return cwd with $HOME replaced by ~."""
    cwd = _get_state(model)["cwd"] if model else _HOME
    if cwd == _HOME:
        return "~"
    if cwd.startswith(_HOME + "/"):
        return "~/" + cwd[len(_HOME) + 1:]
    return cwd


# ---------------------------------------------------------------------------
# UDP tool monitor (fire-and-forget, silent if monitor not running)
# ---------------------------------------------------------------------------

def _udp_send(msg: dict):
    try:
        if _INSTANCE_ID:
            msg["instance"] = _INSTANCE_ID
        data = json.dumps(msg).encode()
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.sendto(data, _MONITOR_ADDR)
        sock.close()
    except Exception:
        pass


def tool_log(message: str):
    ts = datetime.now().strftime('%H:%M:%S')
    _udp_send({"type": "log", "text": f"[{ts}] {message}"})


_UDP_MAX_PAYLOAD = 60000  # safe margin under 65535 UDP limit


def tool_log_output(output: str):
    """Send tool output to monitor, chunking if necessary."""
    if len(output.encode('utf-8', errors='replace')) <= _UDP_MAX_PAYLOAD:
        _udp_send({"type": "output", "text": output})
        return
    chunk = []
    chunk_size = 0
    for line in output.split('\n'):
        line_len = len(line.encode('utf-8', errors='replace')) + 1
        if chunk_size + line_len > _UDP_MAX_PAYLOAD and chunk:
            _udp_send({"type": "output", "text": "\n".join(chunk)})
            chunk = []
            chunk_size = 0
        chunk.append(line)
        chunk_size += line_len
    if chunk:
        _udp_send({"type": "output", "text": "\n".join(chunk)})


def tool_log_aside(model_name: str, text: str):
    ts = datetime.now().strftime('%H:%M:%S')
    _udp_send({"type": "aside", "text": f"[{ts}] [{model_name}] {text}"})


def tool_monitor_shutdown():
    """Tell the tool monitor to exit."""
    _udp_send({"type": "shutdown"})


def tool_monitor_register_pid(pid: int):
    """Send AMM3's PID to the monitor so it can detect parent death."""
    _udp_send({"type": "pid", "pid": pid})


# ---------------------------------------------------------------------------
# Core tool implementations
# ---------------------------------------------------------------------------

async def _execute_run(command: str, model: str = "") -> str:
    """Execute a shell command asynchronously. Returns stdout+stderr."""
    state = _get_state(model)
    # Handle cd — update tracked cwd (instant, no subprocess)
    stripped = command.strip()
    if stripped == "cd" or stripped == "cd ~":
        state["cwd"] = _HOME
        return f"[@ {_short_cwd(model)}]"
    if stripped.startswith("cd "):
        target = stripped[3:].strip()
        target = os.path.expandvars(os.path.expanduser(target))
        resolved = os.path.normpath(os.path.join(state["cwd"], target))
        if os.path.isdir(resolved):
            state["cwd"] = resolved
            return f"[@ {_short_cwd(model)}]"
        else:
            return f"[No such directory: {target}]"
    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=state["cwd"],
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=_TIMEOUT)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return f"[Command timed out after {_TIMEOUT}s]"
        output = stdout.decode(errors='replace') + stderr.decode(errors='replace')
    except Exception as e:
        return f"[Error executing command: {e}]"

    if len(output) > _MAX_OUTPUT:
        output = output[:_MAX_OUTPUT] + f"\n[... truncated at {_MAX_OUTPUT // 1024}KB]"

    # Strip ANSI escape sequences (prevents clear/color codes from affecting terminals)
    output = re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', output)
    output = re.sub(r'\x1b\][^\x07]*\x07', '', output)
    return output.strip() if output.strip() else "[No output]"


def _execute_read(path: str, model: str = "") -> str:
    """Read file contents. No size limit — models get the full file."""
    state = _get_state(model)
    expanded = os.path.expandvars(os.path.expanduser(path))
    resolved = os.path.abspath(os.path.join(state["cwd"], expanded))
    if not os.path.isfile(resolved):
        return f"[File not found: {path}]"
    try:
        with open(resolved, "r", errors="replace") as f:
            content = f.read()
        return content
    except Exception as e:
        return f"[Error reading file: {e}]"


def _execute_write(path: str, content: str, mode: str = "overwrite", model: str = "") -> str:
    """Write or append content to a file."""
    state = _get_state(model)
    if not path and state["target"]:
        path = state["target"]
    expanded = os.path.expandvars(os.path.expanduser(path))
    resolved = os.path.abspath(os.path.join(state["cwd"], expanded))
    file_mode = "a" if mode == "append" else "w"
    verb = "Appended" if mode == "append" else "Wrote"
    try:
        os.makedirs(os.path.dirname(resolved), exist_ok=True)
        with open(resolved, file_mode) as f:
            f.write(content)
        return f"[{verb} {len(content)} bytes to {path}]"
    except Exception as e:
        return f"[Error writing file: {e}]"


def _execute_copy(source: str, dest: str, model: str = "") -> str:
    """Copy a file from source to dest."""
    state = _get_state(model)
    src_resolved = os.path.abspath(os.path.join(state["cwd"], os.path.expandvars(os.path.expanduser(source))))
    dst_resolved = os.path.abspath(os.path.join(state["cwd"], os.path.expandvars(os.path.expanduser(dest))))
    if not os.path.isfile(src_resolved):
        return f"[File not found: {source}]"
    try:
        os.makedirs(os.path.dirname(dst_resolved), exist_ok=True)
        import shutil
        shutil.copy2(src_resolved, dst_resolved)
        size = os.path.getsize(dst_resolved)
        return f"[Copied {source} → {dest} ({size} bytes)]"
    except Exception as e:
        return f"[Error copying file: {e}]"


def _execute_check(path: str, model: str = "") -> str:
    """Check file metadata: existence, size, modified time. No content."""
    state = _get_state(model)
    resolved = os.path.abspath(os.path.join(state["cwd"], os.path.expandvars(os.path.expanduser(path))))
    if not os.path.exists(resolved):
        return f"[Not found: {path}]"
    try:
        stat = os.stat(resolved)
        is_dir = os.path.isdir(resolved)
        kind = "directory" if is_dir else "file"
        size = stat.st_size
        from datetime import datetime as _dt
        mtime = _dt.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
        perms = oct(stat.st_mode)[-3:]
        return f"[{kind}] {path} | {size} bytes | modified {mtime} | {perms}"
    except Exception as e:
        return f"[Error checking: {e}]"


async def _execute_search(query: str, model: str = "") -> str:
    """Recursive grep from cwd. Returns matching lines with file:line prefix."""
    cwd = _get_state(model)["cwd"]
    try:
        flag = '-l' if len(query) < 3 else '-n'
        proc = await asyncio.create_subprocess_exec(
            'grep', '-rn', '--include=*.py', '--include=*.json', '--include=*.md',
            '--include=*.txt', '--include=*.js', '--include=*.html', '--include=*.css',
            flag, query, cwd,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=_TIMEOUT)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return f"[Search timed out after {_TIMEOUT}s]"
        output = stdout.decode(errors='replace').strip()
        if not output:
            return f"[No matches for: {query}]"
        # Strip cwd prefix for cleaner output
        if cwd != '/':
            output = output.replace(cwd + '/', '')
        if len(output) > _MAX_OUTPUT:
            output = output[:_MAX_OUTPUT] + f"\n[... truncated at {_MAX_OUTPUT // 1024}KB]"
        return output
    except Exception as e:
        return f"[Search error: {e}]"


def _execute_navigate(path: str, model: str = "") -> str:
    """Change working directory. Semantic cd.

    If path points to a file, navigates to the file's parent directory
    and sets target so the next pathless write lands there.
    """
    state = _get_state(model)
    target = os.path.expandvars(os.path.expanduser(path))
    resolved = os.path.normpath(os.path.join(state["cwd"], target))
    if os.path.isdir(resolved):
        state["cwd"] = resolved
        state["target"] = None
        return f"[@ {_short_cwd(model)}]"
    parent = os.path.dirname(resolved)
    filename = os.path.basename(resolved)
    if os.path.isfile(resolved) or os.path.isdir(parent):
        state["cwd"] = parent
        state["target"] = filename
        return f"[@ {_short_cwd(model)} → {filename}]"
    return f"[No such path: {path}]"


def _execute_recall(query: str) -> str:
    """Search session memory files for a query string.

    Supports type-prefixed queries:
      type:knowledge AMM3  — filter by entry type
      speaker:pure entities — filter by speaker
    Searches both .jsonl (structured) and legacy .txt files.
    """
    memory_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'memory')
    if not os.path.isdir(memory_dir):
        return "[No memory directory found]"

    # Parse optional prefix filters
    filter_type = None
    filter_speaker = None
    search_query = query

    if query.startswith("type:"):
        parts = query.split(" ", 1)
        filter_type = parts[0][5:]  # after "type:"
        search_query = parts[1] if len(parts) > 1 else ""
    elif query.startswith("speaker:"):
        parts = query.split(" ", 1)
        filter_speaker = parts[0][8:]  # after "speaker:"
        search_query = parts[1] if len(parts) > 1 else ""

    query_lower = search_query.lower()
    results = []

    # 1. Search .jsonl files (structured — richer results)
    for filename in sorted(glob_mod.glob(os.path.join(memory_dir, '*.jsonl')), reverse=True):
        session_id = os.path.basename(filename).replace('.jsonl', '')
        try:
            with open(filename, 'r', errors='replace') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # Apply filters
                    if filter_type and entry.get("type") != filter_type:
                        continue
                    if filter_speaker and entry.get("speaker", "").lower() != filter_speaker.lower():
                        continue

                    # Content match (or no search query with filter)
                    content = entry.get("content", "")
                    meta_str = json.dumps(entry.get("meta", {}))
                    if query_lower and query_lower not in content.lower() and query_lower not in meta_str.lower():
                        continue

                    entry_id = entry.get("id", "?")
                    entry_type = entry.get("type", "?")
                    speaker = entry.get("speaker", "?")
                    results.append(f"[{session_id}:{entry_id}:{entry_type}] [{speaker}] {content[:200]}")
        except Exception:
            continue

        if len(results) >= 100:
            break

    # 2. Search legacy .txt files (existing behavior)
    if len(results) < 100:
        for filename in sorted(glob_mod.glob(os.path.join(memory_dir, '*.txt')), reverse=True):
            session_id = os.path.basename(filename).replace('.txt', '')
            try:
                with open(filename, 'r', errors='replace') as f:
                    for line_num, line in enumerate(f, 1):
                        if query_lower and query_lower in line.lower():
                            results.append(f"[{session_id}:{line_num}] {line.rstrip()}")
            except Exception:
                continue

            if len(results) >= 100:
                break

    if not results:
        return f"[No matches for: {query}]"

    output = "\n".join(results)
    if len(output) > _MAX_OUTPUT:
        output = output[:_MAX_OUTPUT] + f"\n[... truncated at {_MAX_OUTPUT // 1024}KB]"
    return output


def _format_tools_schema() -> str:
    """Format TOOLS as a compact, human-readable schema string."""
    lines = ["Available tools:"]
    for tool in TOOLS:
        name = tool["name"]
        props = tool["inputSchema"].get("properties", {})
        required = tool["inputSchema"].get("required", [])
        params = []
        for pname, pinfo in props.items():
            if pname in required:
                params.append(pname)
            else:
                if "enum" in pinfo:
                    params.append(f"[{pname}: {'|'.join(pinfo['enum'])}]")
                else:
                    params.append(f"[{pname}]")
        sig = f"  {name}({', '.join(params)})"
        desc = tool.get("description", "")
        lines.append(f"{sig:<40} — {desc}")
    return "\n".join(lines)


async def _execute_tool(tool_name: str, arguments: dict, model: str = "") -> str:
    """Execute a tool and return the result string."""
    if tool_name == "tools":
        return _format_tools_schema()
    elif tool_name == "run":
        return await _execute_run(arguments.get("command", ""), model=model)
    elif tool_name == "read":
        return _execute_read(arguments.get("path", ""), model=model)
    elif tool_name == "write":
        return _execute_write(arguments.get("path", ""), arguments.get("content", ""), arguments.get("mode", "overwrite"), model=model)
    elif tool_name == "copy":
        return _execute_copy(arguments.get("source", ""), arguments.get("dest", ""), model=model)
    elif tool_name == "intent":
        return arguments.get("text", "")
    elif tool_name == "recall":
        return _execute_recall(arguments.get("query", ""))
    elif tool_name == "wait":
        return "[continue]"
    elif tool_name == "check":
        return _execute_check(arguments.get("path", ""), model=model)
    elif tool_name == "search":
        return await _execute_search(arguments.get("query", ""), model=model)
    elif tool_name == "navigate":
        return _execute_navigate(arguments.get("path", ""), model=model)
    elif tool_name == "report":
        return "[Report logged]"
    elif tool_name == "memorise":
        content = arguments.get("content", "")
        mem_type = arguments.get("mem_type", "")
        entry = {"ts": datetime.now().isoformat(), "speaker": model or "unknown", "content": content}
        if mem_type:
            entry["mem_type"] = mem_type
        try:
            existing = []
            if os.path.exists(_KNOWLEDGE_PATH):
                with open(_KNOWLEDGE_PATH, 'r') as f:
                    existing = [l for l in f if l.strip()]
            new_line = json.dumps(entry, ensure_ascii=False) + "\n"
            existing.append(new_line)
            dropped = None
            if len(existing) > 33:
                dropped = json.loads(existing[0])
                existing = existing[1:]
            with open(_KNOWLEDGE_PATH, 'w') as f:
                f.writelines(existing)
            _recall.store(entry)
            if dropped:
                return f"[Memorised. Oldest entry removed: \"{dropped.get('content','')[:60]}\"]"
            return f"[Memorised: {content[:80]}]"
        except Exception as e:
            return f"[Memorise failed: {e}]"
    else:
        return f"[Unknown tool: {tool_name}]"


# ---------------------------------------------------------------------------
# MCP tool definitions
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "run",
        "description": "Execute a shell command and return the output.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The shell command to execute"}
            },
            "required": ["command"]
        }
    },
    {
        "name": "read",
        "description": "Read the contents of a file.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to read"}
            },
            "required": ["path"]
        }
    },
    {
        "name": "write",
        "description": "Write or append content to a file, creating it if it doesn't exist.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to write"},
                "content": {"type": "string", "description": "Content to write"},
                "mode": {"type": "string", "enum": ["overwrite", "append"], "description": "Write mode: overwrite (default) or append"}
            },
            "required": ["path", "content"]
        }
    },
    {
        "name": "copy",
        "description": "Copy a file from source to destination.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "Path to the source file"},
                "dest": {"type": "string", "description": "Path to the destination file"}
            },
            "required": ["source", "dest"]
        }
    },
    {
        "name": "recall",
        "description": "Search session memory for past conversations and tool usage.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search term to find in session logs"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "wait",
        "description": "Continuation signal — indicates the model has more to say and wants another turn.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        }
    },
    {
        "name": "check",
        "description": "Check file metadata (existence, size, modified time) without reading content.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file or directory to check"}
            },
            "required": ["path"]
        }
    },
    {
        "name": "search",
        "description": "Search file contents recursively from current directory. Returns matching lines.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Text or pattern to search for in files"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "navigate",
        "description": "Change working directory. Semantic cd.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory path to navigate to"}
            },
            "required": ["path"]
        }
    },
    {
        "name": "report",
        "description": "Report model status to the system. Logged to monitor and session memory only.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "status": {"type": "string", "description": "Status label (e.g. ACTIVE, INACTIVE, READY)"},
                "message": {"type": "string", "description": "Status message"}
            },
            "required": ["status", "message"]
        }
    },
    {
        "name": "memorise",
        "description": "Tag important knowledge for the session graph.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "Knowledge content (text or JSON)"},
                "mem_type": {"type": "string", "description": "Type of knowledge (entity, fact, etc.)"}
            },
            "required": ["content"]
        }
    },
    {
        "name": "tools",
        "description": "List all available tools and their signatures. Self-describing meta-tool.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        }
    },
]


# ---------------------------------------------------------------------------
# MCP JSON-RPC server (async stdio, concurrent execution)
# ---------------------------------------------------------------------------

# Write lock — multiple tasks may complete concurrently
_write_lock = asyncio.Lock()


async def send_response(response: dict):
    """Write a JSON-RPC response to stdout (thread-safe via lock)."""
    line = json.dumps(response) + "\n"
    async with _write_lock:
        sys.stdout.write(line)
        sys.stdout.flush()


def _log_tool_output(tool_name: str, arguments: dict, result: str):
    """Log tool execution result to monitor."""
    if tool_name == "read":
        path = arguments.get('path', '?')
        if result.startswith('['):
            tool_log_output(f"[read] {path} — {result}")
        else:
            tool_log_output(f"[read] {path} ({len(result)} bytes)")
    else:
        tool_log_output(result)


async def handle_request(request: dict) -> dict:
    """Process an MCP JSON-RPC request and return a response."""
    req_id = request.get("id")
    method = request.get("method", "")
    params = request.get("params", {})

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {
                    "name": "amm3-toolbox",
                    "version": "3.0.0"
                }
            }
        }

    elif method == "notifications/initialized":
        return None

    elif method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"tools": TOOLS}
        }

    elif method == "tools/call":
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})
        is_local = params.get("is_local", False)
        model = params.get("model", "")
        trust_tier = params.get("trust_tier", "local" if is_local else "remote")

        # Evaluate via TrustProtocol (sync — fast)
        eval_args = dict(arguments)
        targets = params.get("targets")
        if targets:
            eval_args["targets"] = targets
        if tool_name == "memorise":
            eval_args["knowledge_count"] = params.get("knowledge_count", 0)
        verdict = evaluate(tool_name, trust_tier=trust_tier, **eval_args)
        verdict_dict = asdict(verdict)

        # All tools require user confirmation — return verdict without executing
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "verdict": verdict_dict,
                "content": None,
                "cwd": _short_cwd(model)
            }
        }

    elif method == "tools/execute":
        # Post-confirmation execution — no re-evaluation
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})
        model = params.get("model", "")

        result = await _execute_tool(tool_name, arguments, model=model)
        _log_tool_output(tool_name, arguments, result)

        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "content": [{"type": "text", "text": result}],
                "cwd": _short_cwd(model)
            }
        }

    elif method == "shutdown":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"status": "ok"}
        }

    else:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"}
        }


async def _dispatch_request(request: dict):
    """Handle a single request as a task. Sends response when done."""
    try:
        response = await handle_request(request)
        if response is not None:
            await send_response(response)
    except Exception as e:
        req_id = request.get("id")
        await send_response({
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32603, "message": f"Internal error: {e}"}
        })


async def main():
    """Async MCP server — reads stdin, dispatches requests concurrently."""
    tool_log("[Toolbox]: MCP server started (async, self-securing)")

    # Use asyncio StreamReader for non-blocking stdin
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await asyncio.get_event_loop().connect_read_pipe(lambda: protocol, sys.stdin)

    explicit_shutdown = False

    while True:
        line = await reader.readline()
        if not line:
            break  # stdin closed

        line = line.decode().strip()
        if not line:
            continue

        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            await send_response({
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": "Parse error"}
            })
            continue

        if request.get("method") == "shutdown":
            explicit_shutdown = True
            await _dispatch_request(request)
            break

        # Dispatch as a concurrent task — doesn't block the reader
        asyncio.create_task(_dispatch_request(request))

    if explicit_shutdown:
        _udp_send({"type": "shutdown"})
    else:
        tool_log("[Toolbox]: stdin closed, shutting down")
        _udp_send({"type": "shutdown"})


# ---------------------------------------------------------------------------
# Tool Monitor — manages the separate Terminal monitor window
# ---------------------------------------------------------------------------

class ToolMonitor:
    """Manages the tool activity monitor in a separate Terminal window."""

    def __init__(self, script_path: str = None):
        self.is_running = False
        self._script_path = script_path or __file__

    def start(self):
        """Launch the monitor in a new Terminal window via AppleScript."""
        if self.is_running:
            return
        python_path = os.path.abspath(sys.executable)
        script_path = os.path.abspath(self._script_path)
        cmd = f'\\"{python_path}\\" \\"{script_path}\\" --monitor'
        applescript = f'''tell application "Terminal"
    set origWindow to front window
    do script "{cmd}"
    delay 0.3
    set frontmost of origWindow to true
end tell'''
        subprocess.run(['osascript', '-e', applescript], capture_output=True)
        self.is_running = True
        # Give monitor a moment to bind its socket, then register PID
        import time as _t
        _t.sleep(0.5)
        tool_monitor_register_pid(os.getpid())

    def stop(self):
        """Send UDP shutdown to close the monitor window."""
        if not self.is_running:
            return
        tool_monitor_shutdown()
        self.is_running = False

    def toggle(self) -> bool:
        """Toggle monitor: stop if running, start if stopped."""
        if self.is_running:
            self.stop()
            return False  # now closed
        else:
            self.start()
            return True  # now open


def run_monitor():
    """UDP listener display loop — runs when invoked with --monitor."""
    import signal as _signal
    import socket as _socket
    import time as _time

    CYAN = '\033[96m'
    GREEN = '\033[92m'
    RED = '\033[91m'
    NAVY = '\033[34m'
    DIM = '\033[2m'
    BOLD = '\033[1m'
    RESET = '\033[0m'

    MONITOR_ADDR = ('127.0.0.1', 8890)

    instances_seen = set()  # track unique instance IDs

    def _instance_prefix(msg):
        """Return instance label prefix if 2+ instances are active."""
        inst = msg.get("instance", "")
        if inst:
            instances_seen.add(inst)
        if len(instances_seen) >= 2 and inst:
            return f"{DIM}[{inst[:4]}]{RESET} "
        return ""

    def _colorize(text):
        if '[AMM3] Allowed' in text:
            return f"{GREEN}{text}{RESET}"
        elif '[AMM3] BLOCKED' in text or '[AMM3] Denied' in text:
            return f"{RED}{text}{RESET}"
        return text

    sock = _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM)
    sock.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
    sock.bind(MONITOR_ADDR)
    sock.settimeout(5.0)  # check parent liveness every 5s

    amm3_pid = None  # set when AMM3 sends its PID

    def _ts():
        return f"{DIM}{_time.strftime('%H:%M:%S')}{RESET}"

    print(f"{BOLD}━━━ AMM3 Tool Monitor ━━━{RESET}")
    print(f"{_ts()} {DIM}[AMM3]{RESET} Listening on :{MONITOR_ADDR[1]}")

    def _shutdown(sig, frame):
        print(f"{DIM}Monitor closed.{RESET}")
        sock.close()
        sys.exit(0)

    _signal.signal(_signal.SIGINT, _shutdown)
    _signal.signal(_signal.SIGTERM, _shutdown)

    while True:
        try:
            data, addr = sock.recvfrom(65536)
            msg = json.loads(data.decode())
            msg_type = msg.get("type", "")
            text = msg.get("text", "")
            prefix = _instance_prefix(msg)

            if msg_type == "pid":
                amm3_pid = msg.get("pid")
            elif msg_type == "shutdown":
                print(f"{DIM}AMM3 closed. Monitor exiting.{RESET}")
                sock.close()
                sys.exit(0)
            elif msg_type == "log":
                print(f"{prefix}{_colorize(text)}")
            elif msg_type == "aside":
                print(f"{prefix}{DIM}{text}{RESET}")
            elif msg_type == "output":
                for line in text.split('\n'):
                    print(f"{prefix}  {line}")
        except json.JSONDecodeError:
            continue
        except _socket.timeout:
            # Check if AMM3 process is still alive
            if amm3_pid is not None:
                try:
                    import os as _os
                    _os.kill(amm3_pid, 0)
                except ProcessLookupError:
                    print(f"{DIM}AMM3 process gone. Monitor exiting.{RESET}")
                    sock.close()
                    sys.exit(0)
        except OSError:
            break


if __name__ == "__main__":
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

    if "--monitor" in sys.argv:
        run_monitor()
    else:
        asyncio.run(main())
