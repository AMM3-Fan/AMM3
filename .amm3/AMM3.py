#!/usr/bin/env python3
"""
AMM3 - Artificial Multi-Model Mind
A multi-agent environment where models work together.

Orchestrator: MCP client, routes all tool calls through Distil → TrustProtocol → Toolbox → Synthesise.
"""

from datetime import datetime
import hashlib
import sys
import asyncio
import threading
import select
import signal
import os
import subprocess
import re
import json
import time
import ollama

from system.Distil import distil
from system.Synthesise import compose
from system.TrustProtocol import admit, get_trust_tier
from system.Toolbox import tool_log, tool_log_output, tool_log_aside, tool_monitor_shutdown, ToolMonitor, run_monitor
from system.Environment import get_system_report
from system.Graph import SessionMemory, SemanticRecall
from system.Transport import MCPTransport, NetworkTransport

# Global interrupt flag
interrupt_flag = threading.Event()

# Get environment awareness
ENV_REPORT = get_system_report()

# ANSI color codes
class Colors:
    CYAN = '\033[96m'
    YELLOW = '\033[93m'
    GREEN = '\033[92m'
    GREEN_DARK = '\033[32m'
    RED = '\033[91m'
    BLUE = '\033[34m'
    DIM = '\033[2m'
    BOLD = '\033[1m'
    RESET = '\033[0m'

_OTHER_MODEL_COLORS = ['\033[95m', '\033[35m', '\033[96m', '\033[93m', '\033[97m']

def get_model_color(model_name: str, model_type: str = None) -> str:
    """Return ANSI color for a model label. Consistent per model identity."""
    if model_type == "ollama":
        return Colors.GREEN_DARK
    name = model_name.lower()
    if name.startswith("gemini"):
        return Colors.BLUE
    if name.startswith("claude"):
        return Colors.RED
    idx = sum(ord(c) for c in name) % len(_OTHER_MODEL_COLORS)
    return _OTHER_MODEL_COLORS[idx]


def strip_ansi(text: str) -> str:
    """Remove all ANSI escape sequences from text."""
    cleaned = re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', text)
    cleaned = re.sub(r'\x1b\][^\x07]*\x07', '', cleaned)
    cleaned = re.sub(r'\^\[\[[A-Za-z0-9;]*', '', cleaned)
    cleaned = re.sub(r'\x1b', '', cleaned)
    return cleaned


def sanitize_user_input(text: str) -> str:
    """Escape XML tool tags in user input so they can't be parsed as model tool calls."""
    replacements = {
        '<run>': '[run]', '</run>': '[/run]',
        '<read>': '[read]', '</read>': '[/read]',
        '<write': '[write', '</write>': '[/write]',
        '<recall>': '[recall]', '</recall>': '[/recall]',
        '<memorise': '[memorise', '</memorise>': '[/memorise]',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


# ---------------------------------------------------------------------------
# Identity — passphrase-based, hash-indexed
# ---------------------------------------------------------------------------

_IDENTITIES_PATH = os.path.join(os.path.dirname(__file__), 'nucleus', 'irish.json')


def _hash_passphrase(passphrase: str) -> str:
    """SHA-256 hash truncated to 8 hex chars."""
    return hashlib.sha256(passphrase.encode()).hexdigest()[:8]


def _load_identities() -> dict:
    """Load identities index from disk."""
    if os.path.isfile(_IDENTITIES_PATH):
        try:
            with open(_IDENTITIES_PATH, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_identities(identities: dict):
    """Persist identities index to disk."""
    os.makedirs(os.path.dirname(_IDENTITIES_PATH), exist_ok=True)
    with open(_IDENTITIES_PATH, 'w') as f:
        json.dump(identities, f, indent=2)


def resolve_identity(passphrase: str, session_file: str) -> tuple:
    """Resolve a passphrase to an identity hash. Returns (hash, is_new).

    Creates a new identity entry if unknown. Links current session to identity.
    """
    identity_hash = _hash_passphrase(passphrase)
    identities = _load_identities()
    today = datetime.now().strftime('%Y-%m-%d')

    if identity_hash in identities:
        entry = identities[identity_hash]
        entry["last_seen"] = today
        if session_file not in entry["sessions"]:
            entry["sessions"].append(session_file)
        _save_identities(identities)
        return identity_hash, False
    else:
        identities[identity_hash] = {
            "created": today,
            "last_seen": today,
            "sessions": [session_file],
        }
        _save_identities(identities)
        return identity_hash, True


_ASIDE_PATTERN = re.compile(r'\[aside\]\s*(.*?)(?:\[/aside\]|$)', re.MULTILINE)
_MENTION_PATTERN = re.compile(r'@(\w+)\s*(.*)', re.DOTALL)


def extract_asides(text: str, model_name: str) -> str:
    """Detect [aside] blocks and route to Tool Monitor. Text unchanged."""
    asides = _ASIDE_PATTERN.findall(text)
    for aside in asides:
        tool_log_aside(model_name, aside.strip())
    return text


with open(os.path.join(os.path.dirname(__file__), "nucleus", "amm3.json")) as _cfg:
    _config = json.load(_cfg)

TOOLS_LIST = (
    "<navigate>dir/file</navigate>\n"
    "<run>command</run>\n"
    "<read>file</read>\n"
    "<memorise>content</memorise>\n"
    "<write>content</write>\n"
    "<search>query</search>\n"
    "<recall>query</recall>\n"
    "<wait/>\n"
    "@modelname message — route a message to another model (e.g. @gemini what do you think?)"
)

_BASE_SYSTEM_PROMPT = f"""{_config["system_prompt"]}
{ENV_REPORT}"""
TOOL_SYSTEM_PROMPT = _BASE_SYSTEM_PROMPT  # module-level default; nucleus.system_prompt used at runtime
AVAILABLE_MODELS = _config.get("available_models", {})

# ---------------------------------------------------------------------------
# Chronos Loop — ancestral memory bootstrap
# ---------------------------------------------------------------------------

_recall = SemanticRecall()


def _load_ancestral_memories(context: str) -> str:
    """Query knowledge store semantically and return top-5 entries for system prompt injection.

    Uses SemanticRecall.query() — falls back to most recent entries if Ollama is unavailable.
    """
    entries = _recall.query(context, top_k=5)
    if not entries:
        return ""

    lines = []
    for e in entries:
        speaker = e.get("speaker", "?")
        content = e.get("content", "")[:300]
        ts = e.get("ts", "")[:10]
        lines.append(f"[{ts}] [{speaker}] {content}")

    block = "\n".join(lines)
    return f"\n\n--- Ancestral Memories ({len(lines)} entries) ---\n{block}\n--- End Ancestral Memories ---"


# ---------------------------------------------------------------------------
# Nucleus — application state
# ---------------------------------------------------------------------------

class Nucleus:
    def __init__(self):
        self.connected_models = {}  # model_name -> {"type": "ollama"|"gemini"|"claude", ...}
        self.conversation = []
        self.input_queue = asyncio.Queue()
        self.input_thread_stop_event = threading.Event()
        self.input_thread = None
        self.running = True
        self.generating = False
        self.network = None           # set after identity capture (needs instance_id)
        self.memory = SessionMemory()
        self.mcp = MCPTransport()
        self.monitor = ToolMonitor()
        self.routing_mode = "turn"    # "turn", "active", "parallel"
        self.active_model = None       # tracked for "active" mode
        self.flow_mode = False         # models run autonomously, user steps back
        self.flow_pause = False        # one-turn pause during flow (tool denial)
        self.ball_passed = False       # model used @mention — break turn queue
        self.model_standby = set()    # models that reported done — skip in flow
        self.model_goodbye = set()    # models that reported goodbye — permanent exit
        self.identity_hash = None      # SHA-256[:8] of passphrase
        self.identity_label = None     # raw passphrase (memory only, never persisted)
        self.system_prompt = _BASE_SYSTEM_PROMPT  # bootstrapped with ancestral memories after identity


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def display_prompt(memory=None):
    print("[AMM3]: ", end="", flush=True)

def banner():
    title = "AMM3 - Artificial Multi Model Environment"
    inner_width = len(title) + 4

    top    = f"╔{'═' * inner_width}╗"
    bottom = f"╚{'═' * inner_width}╝"

    print(top)
    print(f"║  {title}  ║")
    print(bottom)

def show_help(memory=None):
    help_text = """Commands:
  Ctrl+C           Interrupt current model
  /flow            Toggle flow mode
  /tools           Toggle tool activity monitor window
  /add_model       Add a model (e.g., /add_gemini, /add_llama3.2)
  @model msg       Send message to specific model
  /rm_model        Remove a model
  /mode            Set routing mode (turn, active, parallel)
  /bye             Goodbye"""
    print(help_text)
    if memory:
        memory.log(help_text)

def _clean_input(text: str) -> str:
    return strip_ansi(text).strip()

def log_print(msg: str, memory=None, newline=True):
    if newline:
        print(msg)
    else:
        print(msg, end="", flush=True)
    if memory:
        memory.log(msg)


# ---------------------------------------------------------------------------
# Input thread
# ---------------------------------------------------------------------------

def _input_thread_worker(loop, input_queue, stop_event):
    while not stop_event.is_set():
        rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
        if rlist:
            line = sys.stdin.readline()
            if line:
                clean_line = _clean_input(line)
                asyncio.run_coroutine_threadsafe(input_queue.put(clean_line), loop)
            else:
                asyncio.run_coroutine_threadsafe(input_queue.put('/bye'), loop)
                break


# ---------------------------------------------------------------------------
# Network streaming callback
# ---------------------------------------------------------------------------

_pending_prefix: dict[str, str] = {}  # agent_name → buffered prefix, printed on first chunk

def _network_chunk_handler(agent_name: str, status: str, data):
    """Handle streaming chunks from NetworkTransport for display."""
    color = get_model_color(agent_name)
    if status == "thinking":
        # Buffer prefix — only print when actual content arrives
        _pending_prefix[agent_name] = f"[{color}{agent_name}{Colors.RESET}]: "
    elif status == "chunk":
        if agent_name in _pending_prefix:
            sys.stdout.write(_pending_prefix.pop(agent_name))
        sys.stdout.write(data or "")
        sys.stdout.flush()
    elif status == "done":
        pending = _pending_prefix.pop(agent_name, None)
        if data and (data.startswith("[Claude error:") or data.startswith("[Gemini error:") or data.startswith("[ERROR]")):
            if pending:
                sys.stdout.write(pending)
            print(data)
        elif not pending:
            # Chunks were printed — add newline if needed
            if data and not data.endswith('\n'):
                print()
        # else: no chunks and no error — empty response, print nothing
    elif status == "tool_calls":
        if agent_name in _pending_prefix:
            sys.stdout.write(_pending_prefix.pop(agent_name))
        if data and not data.endswith('\n'):
            print()
    elif status == "interrupted":
        _pending_prefix.pop(agent_name, None)
        print(f" {Colors.YELLOW}[interrupted]{Colors.RESET}")
    elif status == "error":
        _pending_prefix.pop(agent_name, None)
        print(f"[{agent_name} error]: {data}")


# ---------------------------------------------------------------------------
# Ollama chat
# ---------------------------------------------------------------------------

_MAX_TOOL_ITERATIONS = 30
_CONTEXT_ROUNDS = 3
_monitor_cwd = "~"  # last known cwd from Toolbox, shown in monitor
_local_msg_counts: dict[str, int] = {}  # monitor message counters for local models

_TOOL_STOP_TAGS = ('</run>', '</read>', '</write>', '</recall>')
_TOOL_STOP_REGEX = re.compile(r'\((run|read|write|recall|copy)[\s)][^)]*\)')


def sliding_window(conversation: list, model_name: str, rounds: int = _CONTEXT_ROUNDS) -> list:
    """Return the last N rounds for a specific model."""
    model_indices = [
        i for i, msg in enumerate(conversation)
        if msg.get("name", "").lower() == model_name.lower()
    ]
    if len(model_indices) <= rounds:
        return conversation
    start = model_indices[-rounds] + 1
    return conversation[start:]


def build_ollama_messages(conversation: list, model_name: str) -> list:
    """Build Ollama-compatible messages from conversation.

    Identity stays in the name field — content is never polluted with labels.
      - This model's turns  → role: assistant
      - AMM3 operator turns → role: user
      - Other model turns   → role: user, name: <model>
    """
    msgs = []
    for msg in conversation:
        name    = msg.get("name", "")
        content = msg.get("content", "")
        if name.lower() == model_name.lower():
            msgs.append({"role": "assistant", "content": content})
        elif name == "AMM3" or not name:
            msgs.append({"role": "user", "content": content})
        else:
            msgs.append({"role": "user", "name": name, "content": content})
    return msgs


_GPU_ERROR_PATTERNS = ["metal", "gpu", "ggml_metal", "ggml_backend", "XPC_ERROR", "MTLLibrary"]


def _restart_ollama() -> bool:
    """Kill and restart Ollama service. Returns True if recovery succeeded."""
    ollama_env = {**os.environ, "OLLAMA_MODELS": os.path.expanduser("~/.amm3/models")}
    try:
        subprocess.run(["killall", "ollama"], capture_output=True)
        time.sleep(1)
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=ollama_env,
        )
        for _ in range(10):
            time.sleep(1)
            try:
                ollama.list()
                return True
            except Exception:
                pass
    except Exception:
        pass
    return False


def _is_gpu_crash(error_text: str) -> bool:
    """Detect Metal/GPU backend crashes in error output."""
    lower = error_text.lower()
    return any(p.lower() in lower for p in _GPU_ERROR_PATTERNS)


def _stream_ollama(model_name: str, messages: list, nucleus: 'Nucleus') -> str:
    """Stream Ollama response, stopping at first tool call tag. Returns response text."""
    global interrupt_flag
    response_text = ""
    stream = ollama.chat(model=model_name, messages=messages, stream=True)
    color = get_model_color(model_name, "ollama")
    print(f"[{color}{model_name}{Colors.RESET}]: ", end="", flush=True)
    nucleus.generating = True

    for chunk in stream:
        if interrupt_flag.is_set():
            print(" [interrupted]")
            return response_text + " [interrupted]"

        token = chunk['message']['content']
        sys.stdout.write(token)
        sys.stdout.flush()
        response_text += token

        if any(tag in response_text for tag in _TOOL_STOP_TAGS) or \
           _TOOL_STOP_REGEX.search(response_text):
            print()
            return response_text

    print()
    return response_text


def chat_with_ollama(model_name: str, messages: list, nucleus: 'Nucleus') -> str:
    """Send messages to an Ollama model and stream response.

    On GPU/Metal crash, restarts Ollama and retries once.
    """
    try:
        return _stream_ollama(model_name, messages, nucleus)
    except Exception as e:
        error_str = str(e)

        if _is_gpu_crash(error_str):
            print(f"\n[AMM3]: GPU crash detected — restarting Ollama...")
            tool_log("[AMM3] GPU crash — restarting Ollama")

            if _restart_ollama():
                print("[AMM3]: Ollama restarted. Retrying last message...")
                tool_log("[AMM3] Ollama recovered — retrying last message")
                try:
                    return _stream_ollama(model_name, messages, nucleus)
                except Exception as retry_err:
                    print(f"\n[AMM3]: Retry failed — restart your Mac to reset Metal.")
                    tool_log(f"[AMM3] Retry failed: {retry_err}")
                    return "[ERROR] GPU crash — manual restart needed"
            else:
                print("[AMM3]: Ollama restart failed — restart your Mac to reset Metal.")
                tool_log("[AMM3] Ollama restart failed")
                return "[ERROR] GPU crash — manual restart needed"
        else:
            print(f"\n[{model_name}]: [ERROR] {e}")
            return f"[ERROR] {e}"
    finally:
        nucleus.generating = False


# ---------------------------------------------------------------------------
# Tool execution pipeline: admit() → Toolbox (TrustProtocol) → user confirm → Toolbox execute
# ---------------------------------------------------------------------------

# Intent resolution — keyword → tool call
_INTENT_MAP = [
    ({"date", "time", "current date", "current time", "what time", "what date"},
     {"tool": "run", "command": "date"}),
    ({"directory", "current directory", "where am i", "pwd", "current path"},
     {"tool": "run", "command": "pwd"}),
    ({"who am i", "username", "current user", "whoami"},
     {"tool": "run", "command": "whoami"}),
    ({"system", "os", "platform", "uname", "architecture", "kernel"},
     {"tool": "run", "command": "uname -a"}),
    ({"disk", "disk space", "storage", "free space"},
     {"tool": "run", "command": "df -h"}),
    ({"memory", "ram", "free memory"},
     {"tool": "run", "command": "vm_stat"}),
]


def _resolve_intent(text: str) -> dict | None:
    """Match intent text against keyword map. Returns a tool call dict or None."""
    lower = text.lower().strip().rstrip('?').strip()
    for keywords, call in _INTENT_MAP:
        for kw in keywords:
            if kw in lower:
                return dict(call)
    return None


def _format_verdict(tool_type: str, args: dict, verdict: dict, model_name: str = "") -> str:
    """Format a verdict dict for display to the user."""
    if tool_type == "run":
        detail = args.get("command", "")
    elif tool_type == "read":
        detail = args.get("path", "")
    elif tool_type == "write":
        content = args.get("content", "")
        detail = f"{args.get('path', '')} ({len(content)} bytes)"
    elif tool_type == "copy":
        detail = f"{args.get('source', '')} → {args.get('dest', '')}"
    else:
        detail = str(args)

    risk = verdict.get("risk", "medium")
    risk_color = Colors.RED if risk in ("high", "critical") else ""
    risk_reset = Colors.RESET if risk in ("high", "critical") else ""

    verb = tool_type
    if tool_type == "write" and args.get("mode") == "append":
        verb = "append"

    return (
        f"[AMM3]: {model_name.capitalize()} → {verb} "
        f"{Colors.BOLD}{detail}{Colors.RESET}  "
        f"[{risk_color}RISK: {risk}{risk_reset}]"
    )


async def execute_tool_call(
    call: dict, model_name: str,
    messages: list, nucleus: 'Nucleus'
) -> None:
    """Evaluate and execute a single tool call.

    Pipeline: admit() (external only) → Toolbox.call_tool() (evaluates via TrustProtocol)
              → user confirm if needed → Toolbox.execute_tool()
    """
    global _monitor_cwd
    tool_type = call["tool"]

    # Default path — pathless read/write: Toolbox resolves via _target_file or untitled.txt
    if tool_type == "read" and not call.get("path"):
        call["path"] = "untitled.txt"
    if tool_type == "write" and not call.get("path"):
        call["path"] = ""  # Toolbox resolves: _target_file from navigate, or empty

    # Stub tools — recognized but not implemented
    if tool_type == "build":
        msg = "[AMM3]: Tool is under consideration."
        print(msg)
        nucleus.memory.log(msg)
        tool_log("[AMM3] build: under consideration")
        messages.append({"role": "user", "content": "[Output]: Tool is under consideration."})
        return

    # Report — short-circuit: log to monitor + memory, no Toolbox call
    if tool_type == "report":
        status = call.get("status", "").lower()
        message = call.get("message", "")
        tool_log(f"[{model_name}] report [{status}]: {message}")
        nucleus.memory.log(f"[{model_name}] REPORT [{status}]: {message}")

        if status in ("done", "standby"):
            nucleus.model_standby.add(model_name)
            messages.append({"role": "user", "content": "[AMM3]: Standby acknowledged."})
        elif status in ("goodbye", "bye"):
            nucleus.model_goodbye.add(model_name)
            nucleus.model_standby.add(model_name)
            log_print(f"[AMM3]: {model_name} has disengaged.", nucleus.memory)
            messages.append({"role": "user", "content": "[AMM3]: Goodbye acknowledged."})
        else:
            messages.append({"role": "user", "content": "[AMM3]: Report logged."})

        return True

    # Intent — bare echo resolved to action
    if tool_type == "intent":
        text = call.get("text", "")
        tool_log(f"[{model_name}] {text}")
        resolved = _resolve_intent(text)
        if resolved:
            tool_log(f"[AMM3] → {resolved['tool']}: {resolved.get('command', resolved.get('path', ''))}")
            # Re-enter as a real tool call
            return await execute_tool_call(resolved, model_name, messages, nucleus)
        # No resolution — feed text back as-is
        messages.append({"role": "user", "content": text})
        return True

    # Build args dict for call_tool()
    knowledge_count = 0
    if tool_type == "run":
        args = {"command": call.get("command", "")}
        targets = call.get("targets")
        detail = args["command"]
    elif tool_type == "read":
        args = {"path": call.get("path", "")}
        targets = None
        detail = args["path"]
    elif tool_type == "write":
        args = {"path": call.get("path", ""), "content": call.get("content", "")}
        if call.get("mode"):
            args["mode"] = call["mode"]
        targets = None
        detail = args["path"]
    elif tool_type == "copy":
        args = {"source": call.get("source", ""), "dest": call.get("dest", "")}
        targets = [args["dest"]]
        detail = f"{args['source']} → {args['dest']}"
    elif tool_type == "recall":
        args = {"query": call.get("query", "")}
        targets = None
        detail = args["query"]
    elif tool_type == "wait":
        args = {}
        targets = None
        detail = call.get("duration", "continue")
    elif tool_type == "check":
        args = {"path": call.get("path", "")}
        targets = None
        detail = args["path"]
    elif tool_type == "search":
        args = {"query": call.get("query", "")}
        targets = None
        detail = args["query"]
    elif tool_type == "navigate":
        path = call.get("path", "")
        args = {"path": path}
        targets = None
        detail = path
    elif tool_type == "memorise":
        args = {"content": call.get("content", ""), "mem_type": call.get("mem_type", "")}
        targets = None
        detail = call.get("content", "")[:60]
        _knowledge_path = os.path.join(os.path.dirname(__file__), 'nucleus', 'knowledge.jsonl')
        try:
            with open(_knowledge_path, 'r') as _kf:
                knowledge_count = sum(1 for l in _kf if l.strip())
        except FileNotFoundError:
            knowledge_count = 0
    else:
        args = {}
        targets = None
        detail = str(call)

    tool_log(f"[{model_name} {_monitor_cwd}] {tool_type}: {detail}")

    # 1. Admission gate for external models
    model_info = nucleus.connected_models.get(model_name, {})
    is_local = model_info.get("type") == "ollama"
    trust_tier = get_trust_tier(model_name, is_local=is_local)

    if not is_local:
        admission = admit(model_name, call)
        if not admission["allowed"]:
            print(f"{Colors.RED}[AMM3]: {admission['reason']} — DENIED{Colors.RESET}")
            messages.append({"role": "user", "content": "[AMM3]: Denied."})
            nucleus.memory.log_tool(tool_type, {"model": model_name}, "admission-denied")
            tool_log(f"[AMM3] ADMISSION DENIED: {admission['reason']}")
            return False

    # 2. Toolbox evaluates via TrustProtocol internally
    resp = await nucleus.mcp.call_tool(tool_type, args, is_local=is_local, targets=targets,
                                       model=model_name, trust_tier=trust_tier,
                                       knowledge_count=knowledge_count if tool_type == "memorise" else 0)
    verdict = resp.get("verdict", {})
    cwd = resp.get("cwd", "~")
    _monitor_cwd = cwd

    if verdict.get("action") == "block":
        display_text = _format_verdict(tool_type, args, verdict, model_name)
        print(f"{Colors.RED}{display_text} — BLOCKED{Colors.RESET}")
        messages.append({"role": "user", "content": "[AMM3]: Denied."})
        nucleus.memory.log_tool(tool_type, args, "block")
        tool_log(f"[AMM3] BLOCKED: {verdict.get('reasoning', '')}")
        return False

    # 3. User confirmation (all tools require approval)
    display_text = _format_verdict(tool_type, args, verdict, model_name)
    print(display_text)
    print(f"{Colors.BLUE}[AMM3]: Allow? (yes/no): {Colors.RESET}", end="", flush=True)

    # Poll for input, allow Ctrl+C to deny and return to prompt
    answer = None
    while answer is None:
        if interrupt_flag.is_set():
            interrupt_flag.clear()
            answer = ""
            print()
            break
        try:
            answer = await asyncio.wait_for(nucleus.input_queue.get(), timeout=0.2)
        except asyncio.TimeoutError:
            continue
    answer = answer.strip().lower()

    if answer != 'yes':
        messages.append({"role": "user", "content": "[AMM3]: Denied."})
        print("[AMM3]: Denied.")
        tool_log(f"[AMM3] Denied")
        nucleus.memory.log_tool(tool_type, args, "denied")
        if nucleus.flow_mode:
            nucleus.flow_pause = True
            print("[AMM3]: Flow paused — your turn.")
        return False

    # 5. Execute via post-confirmation path (no re-evaluation)
    tool_log(f"[AMM3] Allowed")
    print("[AMM3]: Executing...")
    nucleus.memory.log_tool(tool_type, args, "confirm-allow")

    output, cwd = await nucleus.mcp.execute_tool(tool_type, args, model=model_name)
    _monitor_cwd = cwd
    nucleus.memory.log_tool_output(tool_type, output[:500])

    messages.append({"role": "user", "content": f"[AMM3 @ {cwd}]: {output}"})
    return True


async def handle_model_response(
    response: str, model_name: str, messages: list, nucleus: 'Nucleus',
    model_type: str = None, native_calls: list = None
) -> str:
    """Check a model response for tool calls or mentions and handle them reflexively.

    Loops up to _MAX_TOOL_ITERATIONS to allow for tool-chaining and multi-agent synthesis.
    """
    if model_type is None:
        info = nucleus.connected_models.get(model_name, {})
        model_type = info.get("type", "ollama")

    metadata = {}  # SDK metadata from distil, passed to compose

    for iteration in range(_MAX_TOOL_ITERATIONS):
        response = extract_asides(response, model_name)

        # All calls go through distil — native or text-parsed
        if native_calls is not None:
            calls, metadata = distil(response, model_name=model_name, native_calls=native_calls)
            native_calls = None  # consumed — next iteration parses normally
        else:
            calls, metadata = distil(response, model_name=model_name)

        mention = _MENTION_PATTERN.search(response)

        # <tools/> command — feed tool list back to the model
        if "<tools/>" in response and not calls:
            messages.append({"role": "assistant", "content": response})
            feedback = f"[AMM3]: {TOOLS_LIST}"
            messages.append({"role": "user", "content": feedback})
            if model_type == "ollama":
                response = await asyncio.to_thread(chat_with_ollama, model_name, messages, nucleus)
            else:
                nucleus.generating = True
                response, new_native_calls = await nucleus.network.send(model_name, feedback)
                nucleus.generating = False
                if new_native_calls:
                    native_calls = new_native_calls
            continue

        if not calls and not mention:
            return response

        # Record this turn segment in private context
        if response:
            messages.append({"role": "assistant", "content": response})

        # 1. Handle tool calls (prioritized discovery/action)
        if calls:
            any_allowed = False
            denied = False
            raw_results = []  # for compose() and native tool result feedback
            for i, call in enumerate(calls):
                if interrupt_flag.is_set():
                    interrupt_flag.clear()
                    return response
                allowed = await execute_tool_call(call, model_name, messages, nucleus)
                if allowed:
                    any_allowed = True
                    last_msg = messages[-1]["content"]
                    raw_results.append({
                        "index": i,
                        "name": call["tool"],
                        "output": last_msg,
                        "cwd": "~"
                    })
                else:
                    denied = True
                    break

            if denied:
                if model_type in ("gemini", "claude") and nucleus.network and nucleus.network.connected:
                    # Send denial results for all unresolved calls so the agent isn't left
                    # with outstanding function calls and no results (breaks the SDK state)
                    denial_results = list(raw_results)
                    for j in range(len(raw_results), len(calls)):
                        denial_results.append({
                            "index": j,
                            "name": calls[j]["tool"],
                            "output": "[Denied by user]",
                            "cwd": "~"
                        })
                    denial_composed = compose(denial_results, model_type, metadata)
                    if denial_composed:
                        nucleus.generating = True
                        await nucleus.network.send_tool_results(model_name, denial_composed)
                        nucleus.generating = False
                        # Denial response is ephemeral display only — not stored in conversation.
                        # Ordering would be wrong: _send_to_model appends the original response after
                        # handle_model_response returns, so denial_response would precede it.
                    # Also notify Network.py terminal
                    denied_tools = [calls[j]["tool"] for j in range(len(raw_results), len(calls))]
                    await nucleus.network.send_tool_denied(model_name, denied_tools)
                return response

            # Continue turn sequence
            if model_type == "ollama":
                response = await asyncio.to_thread(chat_with_ollama, model_name, messages, nucleus)
            else:
                # Format results for the target model via Synthesise
                composed = compose(raw_results, model_type, metadata)

                # Try native tool result feedback first (Gemini)
                nucleus.generating = True
                if composed and hasattr(nucleus.network, 'send_tool_results'):
                    response, new_native_calls = await nucleus.network.send_tool_results(
                        model_name, composed)
                    if new_native_calls:
                        native_calls = new_native_calls
                else:
                    last_output = messages[-1]["content"] if messages else ""
                    response, new_native_calls = await nucleus.network.send(
                        model_name, last_output)
                    if new_native_calls:
                        native_calls = new_native_calls
                nucleus.generating = False
            continue

        # 2. Handle @model mentions — reflexive collaboration
        if mention:
            target = mention.group(1).lower()
            mention_msg = mention.group(2).strip()

            # Self-mentions — skip
            if target == model_name:
                return response

            # @AMM3 / @user — pass the ball back, turn order break is signal enough
            if target in ("amm3", "user"):
                nucleus.ball_passed = True
                return response

            if target not in nucleus.connected_models:
                if not await _auto_connect(nucleus, target, user_initiated=False):
                    available = ", ".join(nucleus.connected_models.keys()) or "none"
                    feedback = f"[AMM3]: @{target} is not available. Available: {available}"
                    log_print(feedback, nucleus.memory)
                    messages.append({"role": "user", "content": feedback})
                    if model_type == "ollama":
                        response = await asyncio.to_thread(chat_with_ollama, model_name, messages, nucleus)
                    else:
                        nucleus.generating = True
                        response, new_native_calls = await nucleus.network.send(model_name, feedback)
                        nucleus.generating = False
                        if new_native_calls:
                            native_calls = new_native_calls
                    continue

            if target in nucleus.connected_models:
                tool_log_aside(model_name, f"reflexive mention: @{target}")
                nucleus.ball_passed = True

                target_info = nucleus.connected_models[target]
                target_response = ""

                # Trigger Target Agent
                if target_info["type"] == "ollama":
                    window = sliding_window(nucleus.conversation, target)
                    target_msgs = [{"role": "system", "content": nucleus.system_prompt}] + build_ollama_messages(window, target)
                    if mention_msg:
                        target_msgs.append({"role": "user", "content": f"[{model_name}]: {mention_msg}"})
                    target_response = await asyncio.to_thread(chat_with_ollama, target, target_msgs, nucleus)
                    target_response = await handle_model_response(target_response, target, target_msgs, nucleus)
                else:
                    nucleus.generating = True
                    target_response, target_native = await nucleus.network.send(
                        target, f"[{model_name}]: {mention_msg}")
                    nucleus.generating = False
                    target_response = await handle_model_response(
                        target_response, target, nucleus.conversation, nucleus,
                        native_calls=target_native)

                # Append Target response to global conversation
                nucleus.conversation.append({"role": "user", "name": target, "content": target_response})
                nucleus.memory.log_message("assistant", target_response, model=target)

                # Feedback loop: Provide target response back to source model
                feedback = f"[{target}]: {target_response}"
                messages.append({"role": "user", "content": feedback})

                if model_type == "ollama":
                    response = await asyncio.to_thread(chat_with_ollama, model_name, messages, nucleus)
                else:
                    nucleus.generating = True
                    response, new_native_calls = await nucleus.network.send(
                        model_name, feedback)
                    nucleus.generating = False
                    if new_native_calls:
                        native_calls = new_native_calls

                # Re-evaluate the new response
                continue

    return "[AMM3]: Reflex loop limit reached."


# ---------------------------------------------------------------------------
# Interrupt handler
# ---------------------------------------------------------------------------

def handle_interrupt(signum, frame):
    global interrupt_flag
    interrupt_flag.set()


# ---------------------------------------------------------------------------
# Auto-connect — resolves @model mentions to live connections
# ---------------------------------------------------------------------------

async def _auto_connect(nucleus: 'Nucleus', model_name: str, user_initiated: bool = True) -> bool:
    """Auto-connect a model from the available_models registry."""
    if model_name in nucleus.connected_models:
        return True

    registry_entry = AVAILABLE_MODELS.get(model_name)
    if not registry_entry:
        return False

    model_type = registry_entry["type"]

    if model_type == "ollama":
        try:
            ollama.show(model_name)
        except Exception:
            log_print(f"[AMM3]: Model '{model_name}' not found in Ollama.", nucleus.memory)
            return False
        nucleus.connected_models[model_name] = {"type": "ollama"}
        log_print(f"[AMM3]: {model_name} has joined the chat", nucleus.memory)
        return True

    # Gemini/Claude — network models
    if not user_initiated:
        log_print(f"[AMM3]: A model wants to connect {model_name}. Allow? (yes/no): ", nucleus.memory, newline=False)
        answer = None
        while answer is None:
            if interrupt_flag.is_set():
                interrupt_flag.clear()
                answer = ""
                print()
                break
            try:
                answer = await asyncio.wait_for(nucleus.input_queue.get(), timeout=0.2)
            except asyncio.TimeoutError:
                continue
        if answer.strip().lower() != "yes":
            log_print("[AMM3]: Denied.", nucleus.memory)
            return False

    success, msg = await nucleus.network.start_agent(model_name.lower())
    if success:
        nucleus.connected_models[model_name] = {"type": model_type}
        log_print(f"[AMM3]: Welcome, I'm AMM3 - Artificial Multi Model Environment.", nucleus.memory)
        return True
    else:
        log_print(f"[AMM3]: Failed to start {model_name}: {msg}", nucleus.memory)
        return False


# ---------------------------------------------------------------------------
# Routing — _send_to_model + route modes
# ---------------------------------------------------------------------------

async def _send_to_model(model_name: str, model_info: dict, nucleus: 'Nucleus', empty_enter: bool = False):
    """Dispatch a single model turn (ollama or network) and record the response."""
    global interrupt_flag
    response = ""

    if model_info["type"] == "ollama":
        window = sliding_window(nucleus.conversation, model_name)
        if empty_enter:
            msgs = build_ollama_messages(window, model_name)
        else:
            msgs = [{"role": "system", "content": nucleus.system_prompt}] + build_ollama_messages(window, model_name)
        response = await asyncio.to_thread(chat_with_ollama, model_name, msgs, nucleus)
        _local_msg_counts[model_name] = _local_msg_counts.get(model_name, 0) + 1
        tool_log(f"[{model_name}] #{_local_msg_counts[model_name]} message")
        nucleus.memory.log_message("assistant", response, model=model_name)
        final = await handle_model_response(response, model_name, msgs, nucleus)
        if final != response:
            nucleus.memory.log_message("assistant", final, model=model_name)
        nucleus.conversation.append({"role": "user", "name": model_name, "content": final})

    elif model_info["type"] in ("gemini", "claude"):
        last_turn = -1
        for i in range(len(nucleus.conversation) - 1, -1, -1):
            if nucleus.conversation[i].get("name", "").lower() == model_name.lower():
                last_turn = i
                break
        round_msgs = nucleus.conversation[last_turn + 1:]
        def _fmt_msg(m):
            n = m.get("name", "")
            return f"[{n}]: {m['content']}" if n else m["content"]
        network_msg = "\n".join(_fmt_msg(m) for m in round_msgs) if round_msgs else "\u200b"
        nucleus.generating = True
        response, native_calls = await nucleus.network.send(model_name, network_msg)
        if response:
            nucleus.memory.log_message("assistant", response, model=model_name)
        nucleus.generating = False
        tool_msgs = [{"role": "system", "content": nucleus.system_prompt}] + nucleus.conversation
        final = await handle_model_response(response, model_name, tool_msgs, nucleus, native_calls=native_calls)
        if final != response:
            nucleus.memory.log_message("assistant", final, model=model_name)
        nucleus.conversation.append({"role": "user", "name": model_name, "content": final})


async def _route_turn(nucleus: 'Nucleus', empty_enter: bool = False):
    """All models respond in sequence (default behavior)."""
    nucleus.ball_passed = False
    for model_name, model_info in nucleus.connected_models.items():
        if interrupt_flag.is_set():
            break  # leave flag set — main loop reads it to exit flow mode
        if model_name in nucleus.model_standby:
            continue  # model reported done — skip this turn
        await _send_to_model(model_name, model_info, nucleus, empty_enter=empty_enter)
        if nucleus.ball_passed:
            nucleus.ball_passed = False
            break

    # All active models in standby during flow — pause and return control
    if nucleus.flow_mode:
        active = [m for m in nucleus.connected_models if m not in nucleus.model_standby]
        if not active:
            # Check goodbye lockdown — all models said goodbye
            if nucleus.connected_models and \
               all(m in nucleus.model_goodbye for m in nucleus.connected_models):
                log_print("[AMM3]: All models have disengaged.", nucleus.memory)
            nucleus.flow_mode = False
            nucleus.flow_pause = False
            log_print("[AMM3]: All models standing by. Flow paused.", nucleus.memory)


async def _route_active(nucleus: 'Nucleus', empty_enter: bool = False):
    """Only the active model responds."""
    if not nucleus.active_model or nucleus.active_model not in nucleus.connected_models:
        log_print("[AMM3]: No active model set. Use @model to switch.", nucleus.memory)
        return
    model_info = nucleus.connected_models[nucleus.active_model]
    await _send_to_model(nucleus.active_model, model_info, nucleus, empty_enter=empty_enter)


async def _route_parallel(nucleus: 'Nucleus', empty_enter: bool = False):
    """All models respond simultaneously from the same conversation snapshot."""
    snapshot = list(nucleus.conversation)

    async def _dispatch(name, info):
        original = nucleus.conversation
        nucleus.conversation = list(snapshot)
        try:
            await _send_to_model(name, info, nucleus, empty_enter=empty_enter)
            return nucleus.conversation[-1] if len(nucleus.conversation) > len(snapshot) else None
        finally:
            nucleus.conversation = original

    tasks = []
    for model_name, model_info in nucleus.connected_models.items():
        tasks.append(_dispatch(model_name, model_info))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    nucleus.conversation = list(snapshot)
    for result in results:
        if isinstance(result, Exception):
            log_print(f"[AMM3 ERROR]: {result}", nucleus.memory)
        elif result is not None:
            nucleus.conversation.append(result)


# ---------------------------------------------------------------------------
# Main chat loop
# ---------------------------------------------------------------------------

async def chat_loop(nucleus: Nucleus):
    global interrupt_flag

    # Disable terminal features that cause escape sequence noise
    sys.stdout.write(
        '\x1b[?1004l'
        '\x1b[?1000l'
        '\x1b[?1002l'
        '\x1b[?1003l'
        '\x1b[?1006l'
        '\x1b[?2004l'
    )
    sys.stdout.flush()

    signal.signal(signal.SIGINT, handle_interrupt)

    # Start input thread
    loop = asyncio.get_running_loop()
    nucleus.input_thread = threading.Thread(
        target=_input_thread_worker,
        args=(loop, nucleus.input_queue, nucleus.input_thread_stop_event),
        daemon=True
    )
    nucleus.input_thread.start()

    # Start MCP Toolbox subprocess
    await nucleus.mcp.start()

    # Ensure Ollama service is running with .amm3/models
    ollama_env = {**os.environ, "OLLAMA_MODELS": os.path.expanduser("~/.amm3/models")}
    try:
        ollama.list()
    except Exception:
        log_print("[AMM3]: Ollama not running — launching...", nucleus.memory)
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=ollama_env,
        )
        for _ in range(10):
            await asyncio.sleep(1)
            try:
                ollama.list()
                log_print("[AMM3]: Ollama is ready.", nucleus.memory)
                break
            except Exception:
                pass
        else:
            log_print("[AMM3]: Warning — Ollama failed to start.", nucleus.memory)

    # Add default model and preload into GPU
    default_model = "llama3.2:3b"
    nucleus.connected_models[default_model] = {"type": "ollama"}
    try:
        log_print("[AMM3]: Loading model into GPU...", nucleus.memory)
        await asyncio.to_thread(ollama.generate, model=default_model, prompt="", keep_alive="10m")
        log_print("[AMM3]: Model ready.", nucleus.memory)
    except Exception:
        pass

    # Auto-launch tool monitor in a new Terminal
    nucleus.monitor.start()

    banner()

    # --- Identity capture (silent, non-blocking) ---
    nucleus.network = NetworkTransport(
        instance_id="pending",
        interrupt_event=interrupt_flag,
        on_chunk=_network_chunk_handler,
    )
    _identity_resolved = False

    def _bootstrap_identity(passphrase: str):
        """Resolve identity from passphrase and bootstrap Chronos memories."""
        nonlocal _identity_resolved
        if _identity_resolved:
            return
        try:
            session_file = os.path.basename(nucleus.memory.log_path)
            identity_hash, _ = resolve_identity(passphrase, session_file)
            nucleus.identity_hash = identity_hash
            nucleus.identity_label = passphrase
            nucleus.network = NetworkTransport(
                instance_id=identity_hash,
                interrupt_event=interrupt_flag,
                on_chunk=_network_chunk_handler,
            )
            os.environ["AMM3_INSTANCE_ID"] = identity_hash

            # Chronos Loop: migrate unembedded entries, then query for relevant context
            _recall.migrate()
            context_seed = f"identity:{identity_hash} date:{datetime.now().strftime('%Y-%m-%d')}"
            ancestral = _load_ancestral_memories(context_seed)
            if ancestral:
                nucleus.system_prompt = _BASE_SYSTEM_PROMPT + ancestral
                nucleus.memory.log(f"[AMM3] Chronos: loaded ancestral memories")
            else:
                nucleus.system_prompt = _BASE_SYSTEM_PROMPT

            nucleus.memory.log(f"[AMM3] Identity: {identity_hash}")
        except Exception as e:
            # Fallback: anonymous identity, no Chronos
            nucleus.identity_hash = "anonymous"
            nucleus.network = NetworkTransport(
                instance_id="anonymous",
                interrupt_event=interrupt_flag,
                on_chunk=_network_chunk_handler,
            )
            os.environ["AMM3_INSTANCE_ID"] = "anonymous"
            nucleus.system_prompt = _BASE_SYSTEM_PROMPT
        _identity_resolved = True

    display_prompt(nucleus.memory)

    while nucleus.running:
        try:
            if interrupt_flag.is_set() and not nucleus.generating:
                interrupt_flag.clear()
                if nucleus.flow_mode:
                    nucleus.flow_mode = False
                    nucleus.flow_pause = False
                    log_print("[AMM3]: Flow interrupted. Your turn.", nucleus.memory)
                else:
                    log_print("[AMM3]: Ctrl+C pressed. Use /bye to exit.", nucleus.memory)
                display_prompt(nucleus.memory)
                continue

            # --- Flow mode: auto-route without waiting for user input ---
            if nucleus.flow_mode and not nucleus.flow_pause:
                if not _identity_resolved:
                    _bootstrap_identity("anonymous")
                if nucleus.connected_models:
                    if nucleus.routing_mode == "active":
                        await _route_active(nucleus, empty_enter=True)
                    elif nucleus.routing_mode == "parallel":
                        await _route_parallel(nucleus, empty_enter=True)
                    else:
                        await _route_turn(nucleus, empty_enter=True)
                await asyncio.sleep(0)  # yield so SIGINT can be detected between routes
                continue

            # Flow pause: show prompt before waiting for user's one turn
            if nucleus.flow_pause:
                display_prompt(nucleus.memory)

            user_input = await nucleus.input_queue.get()

            # Consume flow pause — one turn given, flow resumes after this input
            if nucleus.flow_pause:
                nucleus.flow_pause = False

            if not user_input:
                # Empty enter — resolve identity as anonymous if pending
                if not _identity_resolved:
                    _bootstrap_identity("anonymous")
                # Empty enter — models continue autonomously
                if nucleus.connected_models:
                    if nucleus.routing_mode == "active":
                        await _route_active(nucleus, empty_enter=True)
                    elif nucleus.routing_mode == "parallel":
                        await _route_parallel(nucleus, empty_enter=True)
                    else:
                        await _route_turn(nucleus, empty_enter=True)

                display_prompt(nucleus.memory)
                continue

            # --- Commands (no identity needed) ---
            if user_input.lower() == '/bye':
                nucleus.memory.log("[AMM3]: Goodbye!")
                nucleus.running = False
                break

            if user_input.lower() == '/help':
                show_help(nucleus.memory)
                display_prompt(nucleus.memory)
                continue

            if user_input.lower() == '/tools':
                opened = nucleus.monitor.toggle()
                if opened:
                    log_print("[AMM3]: Tool monitor opened.", nucleus.memory)
                else:
                    log_print("[AMM3]: Tool monitor closed.", nucleus.memory)
                display_prompt(nucleus.memory)
                continue

            if user_input.lower().startswith('/mode'):
                parts = user_input.split()
                if len(parts) == 1:
                    active_info = f" ({nucleus.active_model})" if nucleus.routing_mode == "active" and nucleus.active_model else ""
                    log_print(f"[AMM3]: Mode → {nucleus.routing_mode}{active_info}", nucleus.memory)
                elif parts[1] in ("turn", "active", "parallel"):
                    nucleus.routing_mode = parts[1]
                    if parts[1] == "active" and not nucleus.active_model:
                        nucleus.active_model = next(iter(nucleus.connected_models), None)
                    active_info = f" ({nucleus.active_model})" if parts[1] == "active" and nucleus.active_model else ""
                    log_print(f"[AMM3]: Mode → {parts[1]}{active_info}", nucleus.memory)
                else:
                    log_print("[AMM3]: Usage: /mode [turn|active|parallel]", nucleus.memory)
                display_prompt(nucleus.memory)
                continue

            if user_input.lower() == '/flow':
                nucleus.flow_mode = not nucleus.flow_mode
                if nucleus.flow_mode:
                    nucleus.flow_pause = False
                    if not _identity_resolved:
                        _bootstrap_identity("anonymous")
                    log_print("[AMM3]: Flow mode on — models run autonomously. Ctrl+C to interrupt.", nucleus.memory)
                else:
                    log_print("[AMM3]: Flow mode off.", nucleus.memory)
                    display_prompt(nucleus.memory)
                continue

            if user_input.lower().startswith('/add_'):
                if not _identity_resolved:
                    _bootstrap_identity("anonymous")
                model_name = user_input[5:].strip()
                if model_name:
                    if model_name in nucleus.connected_models:
                        log_print(f"[AMM3]: {model_name} is already connected.", nucleus.memory)
                    else:
                        if not await _auto_connect(nucleus, model_name):
                            if model_name not in AVAILABLE_MODELS:
                                log_print(f"[AMM3]: '{model_name}' is not in the model registry.", nucleus.memory)
                else:
                    log_print("[AMM3]: Usage: /add_modelname", nucleus.memory)
                display_prompt(nucleus.memory)
                continue

            if user_input.lower().startswith('/rm_'):
                if not _identity_resolved:
                    _bootstrap_identity("anonymous")
                model_name = user_input[4:].strip()
                if model_name in nucleus.connected_models:
                    model_info = nucleus.connected_models[model_name]
                    if model_info["type"] in ('gemini', 'claude'):
                        success, msg = await nucleus.network.stop_agent(model_name.lower())
                        if success:
                            log_print(f"[AMM3]: {msg}", nucleus.memory)
                        else:
                            log_print(f"[AMM3]: Error stopping {model_name}: {msg}", nucleus.memory)
                    del nucleus.connected_models[model_name]
                    log_print(f"[AMM3]: Connected models: {', '.join(nucleus.connected_models.keys())}", nucleus.memory)
                else:
                    log_print(f"[AMM3]: {model_name} is not connected.", nucleus.memory)
                display_prompt(nucleus.memory)
                continue

            # --- Identity gate: resolve before model interaction ---
            if not _identity_resolved:
                _bootstrap_identity(user_input)

            # --- @model direct message ---
            if user_input.startswith('@'):
                parts = user_input[1:].split(' ', 1)
                target_model = parts[0].lower()
                message = parts[1] if len(parts) > 1 else ""

                if target_model not in nucleus.connected_models:
                    if not await _auto_connect(nucleus, target_model):
                        log_print(f"[AMM3]: {target_model} is not available.", nucleus.memory)
                        display_prompt(nucleus.memory)
                        continue

                nucleus.active_model = target_model

                if not message:
                    log_print(f"[AMM3]: Usage: @{target_model} your message", nucleus.memory)
                    display_prompt(nucleus.memory)
                    continue

                safe_message = sanitize_user_input(message)
                nucleus.conversation.append({"role": "user", "name": "AMM3", "content": safe_message})
                nucleus.memory.log_message("user", message, target=target_model)

                model_info = nucleus.connected_models[target_model]
                response = ""

                if model_info["type"] == "ollama":
                    interrupt_flag.clear()
                    window = sliding_window(nucleus.conversation, target_model)
                    tool_msgs = [{"role": "system", "content": nucleus.system_prompt}] + build_ollama_messages(window, target_model)
                    response = await asyncio.to_thread(chat_with_ollama, target_model, tool_msgs, nucleus)
                    nucleus.memory.log_message("assistant", response, model=target_model)
                    final = await handle_model_response(response, target_model, tool_msgs, nucleus)
                    if final != response:
                        nucleus.memory.log_message("assistant", final, model=target_model)
                    nucleus.conversation.append({"role": "user", "name": target_model, "content": final})

                elif model_info["type"] in ("gemini", "claude"):
                    interrupt_flag.clear()
                    nucleus.generating = True
                    response, native_calls = await nucleus.network.send(target_model, message)
                    if response:
                        nucleus.memory.log_message("assistant", response, model=target_model)
                    nucleus.generating = False
                    tool_msgs = [{"role": "system", "content": nucleus.system_prompt}] + nucleus.conversation
                    final = await handle_model_response(response, target_model, tool_msgs, nucleus, native_calls=native_calls)
                    if final != response:
                        nucleus.memory.log_message("assistant", final, model=target_model)
                    nucleus.conversation.append({"role": "user", "name": target_model, "content": final})

                display_prompt(nucleus.memory)
                continue

            # --- Regular message → all models ---
            else:
                if not nucleus.connected_models:
                    log_print("[AMM3]: No models connected. Use /add_modelname to add one.", nucleus.memory)
                    display_prompt(nucleus.memory)
                    continue

                safe_input = sanitize_user_input(user_input)
                nucleus.conversation.append({"role": "user", "name": "AMM3", "content": safe_input})
                nucleus.memory.log_message("user", user_input)

                # New user message wakes standby models (goodbye is permanent)
                nucleus.model_standby -= nucleus.model_goodbye

                # Check for @model mentions in message body — route selectively
                mentioned = []
                for m in re.findall(r'@(\w+)', user_input):
                    name = m.lower()
                    if name in ('amm3', 'user'):
                        continue
                    if name in nucleus.connected_models:
                        mentioned.append(name)
                    elif name in AVAILABLE_MODELS:
                        if await _auto_connect(nucleus, name, user_initiated=False):
                            mentioned.append(name)
                        else:
                            log_print(f"[AMM3]: @{name} mentioned — not available.", nucleus.memory)
                mentioned = list(dict.fromkeys(mentioned))  # dedup, preserve order

                if mentioned:
                    log_print(f"[AMM3]: routing to {', '.join(f'@{m}' for m in mentioned)}", nucleus.memory)
                    for model_name in mentioned:
                        model_info = nucleus.connected_models[model_name]
                        await _send_to_model(model_name, model_info, nucleus)
                elif nucleus.routing_mode == "active":
                    await _route_active(nucleus)
                elif nucleus.routing_mode == "parallel":
                    await _route_parallel(nucleus)
                else:
                    await _route_turn(nucleus)

                display_prompt(nucleus.memory)

        except asyncio.CancelledError:
            break
        except Exception as e:
            log_print(f"[AMM3 ERROR]: {e}", nucleus.memory)
            display_prompt(nucleus.memory)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    nucleus = Nucleus()

    try:
        await chat_loop(nucleus)
    finally:
        nucleus.input_thread_stop_event.set()

        # Shutdown network client
        if nucleus.network:
            await nucleus.network.shutdown()

        # Shutdown MCP Toolbox
        await nucleus.mcp.stop()

        # Shutdown tool monitor
        nucleus.monitor.stop()

        nucleus.memory.save_summary()
        _recall.migrate()

        from system.TrustProtocol import clear_cache
        clear_cache()

        if nucleus.input_thread:
            nucleus.input_thread.join(timeout=1)

        print("[AMM3]: Goodbye!")


# ---------------------------------------------------------------------------
# Incubation sequence
# ---------------------------------------------------------------------------

def _incubate():
    """Environment setup — runs before the event loop.

    First run: checks Python version, Ollama, installs deps, prompts for API
    keys, pulls the default model, writes sentinel.
    Subsequent runs: silently ensures the default model is available.
    """
    _AMM3_DIR  = os.path.dirname(os.path.abspath(__file__))
    _SENTINEL  = os.path.join(_AMM3_DIR, 'nucleus', '.incubated')
    _ENV_PATH  = os.path.join(_AMM3_DIR, '.env')
    _REQS      = os.path.join(_AMM3_DIR, 'requirements.txt')
    _DEFAULT   = "llama3.2:3b"

    _KEYS = [
        ("GEMINI_API_KEY",    "Gemini API key    (aistudio.google.com)"),
        ("ANTHROPIC_API_KEY", "Anthropic API key (console.anthropic.com)"),
    ]

    first_run = not os.path.exists(_SENTINEL)

    if not first_run:
        return

    # ── First run ──────────────────────────────────────────────────────────

    print("[AMM3]: First run — incubation sequence starting.")

    # Python version
    if sys.version_info < (3, 12):
        print(f"[AMM3]: Python 3.12+ required. You have {sys.version}. Aborting.")
        sys.exit(1)
    print(f"  Python {sys.version.split()[0]}  OK")

    # Ollama installed
    try:
        subprocess.run(["ollama", "--version"], capture_output=True, check=True)
        print("  Ollama             OK")
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("  Ollama             NOT FOUND")
        print("[AMM3]: Install Ollama from https://ollama.com then re-run.")
        sys.exit(1)

    # Pip dependencies
    if os.path.exists(_REQS):
        print("  Installing dependencies...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", _REQS, "-q"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print("  Dependencies       OK")
        else:
            print(f"  Dependencies       WARNING\n{result.stderr.strip()}")

    # API keys
    print("[AMM3]: API keys — press Enter to skip any key.")
    existing = {}
    if os.path.exists(_ENV_PATH):
        with open(_ENV_PATH) as f:
            for line in f:
                if "=" in line:
                    k, v = line.strip().split("=", 1)
                    existing[k.strip()] = v.strip().strip('"').strip("'")

    updated = dict(existing)
    for key, label in _KEYS:
        current = existing.get(key, "")
        display = " [already set]" if current else ""
        try:
            val = input(f"  {label}{display}: ").strip()
            if val:
                updated[key] = val
        except (KeyboardInterrupt, EOFError):
            print("[AMM3]: Skipping remaining keys.")
            break

    if updated:
        with open(_ENV_PATH, 'w') as f:
            for k, v in updated.items():
                if v:
                    f.write(f"{k}={v}\n")
        os.chmod(_ENV_PATH, 0o600)
        configured = [k for k in updated if updated[k]]
        print(f"  Keys saved: {', '.join(configured)}" if configured else "  No keys saved.")
    else:
        print("  No keys set — running with local models only.")

    # Pull default model into AMM3's models directory
    _MODELS_DIR = os.path.join(_AMM3_DIR, 'models')
    _pull_env = {**os.environ, "OLLAMA_MODELS": _MODELS_DIR}
    print(f"[AMM3]: Pulling {_DEFAULT}...")
    try:
        subprocess.run(["ollama", "pull", _DEFAULT], check=True, env=_pull_env)
        print(f"  {_DEFAULT}  OK")
    except subprocess.CalledProcessError:
        print(f"  WARNING — pull failed. Run manually: ollama pull {_DEFAULT}")

    # Write sentinel
    os.makedirs(os.path.dirname(_SENTINEL), exist_ok=True)
    with open(_SENTINEL, 'w') as f:
        f.write(datetime.now().isoformat())

    print("[AMM3]: Incubation complete.")


if __name__ == "__main__":
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

    if "--monitor" in sys.argv:
        run_monitor()
    else:
        _incubate()
        try:
            asyncio.run(main())
        except KeyboardInterrupt:
            print("[AMM3]: Use /bye to exit. Ctrl+C interrupts generation.")
        except Exception as e:
            print(f"[AMM3 ERROR]: {e}")
