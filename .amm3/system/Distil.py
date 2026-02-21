#!/usr/bin/env python3
"""
Distil.py - Parse and normalize tool calls from model output.

Input side of the trust pipeline. Extracts structured tool calls from
raw text (XML, bracket, loose, invented) and native SDK calls.

Confidence tiers:
  "prime"      — clean XML match (<run>, <read>, <write>, <recall>)
  "native"     — SDK structured call (Gemini/Claude function_call)
  "redundancy" — bracket syntax ([run: cmd], [[run: cmd]])
  "drift"      — hybrid/bare bracket patterns
  "evolved"    — translated tags (<mkdir>, pathless <write>)

Signals (non-actionable, metadata only):
  [AI] tag     — model self-labeling drift, tracked but not acted on.
"""

import json
import os
import re
from datetime import datetime


# ---------------------------------------------------------------------------
# Mutation tracking — unrecognized tags logged to nucleus/mutation.json
# ---------------------------------------------------------------------------

_MUTATION_PATH = os.path.join(os.path.dirname(__file__), '..', 'nucleus', 'mutation.json')

# Known tags — these are handled by parsers and should not be logged as mutations
_KNOWN_TAGS = {
    'run', 'read', 'write', 'recall', 'build', 'wait', 'check', 'search',
    'navigate', 'report', 'memorise', 'mkdir', 'tools',
    # Common non-tool tags that models use in prose
    'code', 'pre', 'p', 'br', 'b', 'i', 'em', 'strong', 'ul', 'ol', 'li',
    'h1', 'h2', 'h3', 'h4', 'a', 'span', 'div', 'table', 'tr', 'td', 'th',
    'blockquote', 'summary', 'details', 'aside',
}

# Matches <tag> or <tag attr="..."> (opening tags only, not closing)
_UNKNOWN_TAG_PATTERN = re.compile(r'<([a-z_][a-z0-9_]*)\b[^>]*>', re.IGNORECASE)



def _log_mutation(tag: str, model_name: str, snippet: str):
    """Record an unrecognized tag to nucleus/mutation.json.

    Increments count if seen before; adds new entry otherwise.
    """
    try:
        if os.path.isfile(_MUTATION_PATH):
            with open(_MUTATION_PATH, 'r') as f:
                mutations = json.load(f)
        else:
            mutations = {}

        key = tag.lower()
        if key in mutations:
            mutations[key]["count"] += 1
            mutations[key]["last_seen"] = datetime.now().isoformat()
            if model_name and model_name not in mutations[key]["models"]:
                mutations[key]["models"].append(model_name)
            # Keep last snippet
            mutations[key]["last_snippet"] = snippet[:120]
        else:
            mutations[key] = {
                "tag": tag,
                "count": 1,
                "first_seen": datetime.now().isoformat(),
                "last_seen": datetime.now().isoformat(),
                "models": [model_name] if model_name else [],
                "last_snippet": snippet[:120],
            }

        os.makedirs(os.path.dirname(_MUTATION_PATH), exist_ok=True)
        with open(_MUTATION_PATH, 'w') as f:
            json.dump(mutations, f, indent=2)
    except Exception:
        pass  # mutation tracking is best-effort


# ---------------------------------------------------------------------------
# Primary XML patterns
# ---------------------------------------------------------------------------

_RUN_PATTERN = re.compile(r'<run>(.*?)</run>', re.DOTALL)
_READ_PATTERN = re.compile(r'<read(?:\s+path="([^"]+)")?>(.*?)</read>', re.DOTALL)
_WRITE_PATTERN = re.compile(r'<write\s+path="([^"]+)"\s*>([\s\S]*?)</write>')
_RECALL_PATTERN = re.compile(r'<recall>(.*?)</recall>', re.DOTALL)
_BUILD_PATTERN = re.compile(r'<build>([\s\S]*?)</build>')
_WAIT_PATTERN = re.compile(r'<wait\s*/?>|<wait>(.*?)</wait>', re.DOTALL)
_TOOLS_PATTERN = re.compile(r'<tools\s*/?>|<tools>(.*?)</tools>', re.DOTALL)
_CHECK_PATTERN = re.compile(r'<check>(.*?)</check>', re.DOTALL)
_SEARCH_PATTERN = re.compile(r'<search>(.*?)</search>', re.DOTALL)
_NAVIGATE_PATTERN = re.compile(r'<navigate>(.*?)</navigate>', re.DOTALL)
_NAVIGATE_DRIFT = re.compile(r'\[navigate\](.*?)</navigate>', re.DOTALL)
_REPORT_PATTERN = re.compile(r'<report\s+status="([^"]+)">(.*?)</report>', re.DOTALL)
_REPORT_BARE    = re.compile(r'<report>(.*?)</report>', re.DOTALL)
_MEMORISE_PATTERN = re.compile(r'<memorise(?:\s+type="([^"]+)")?>([\s\S]*?)</memorise>', re.DOTALL)

# Bracket fallbacks (1-3 brackets, colon separator)
_RUN_FALLBACK = re.compile(r'\[{1,3}run:\s*(.+?)\]{1,3}')
_READ_FALLBACK = re.compile(r'\[{1,3}read:\s*(.+?)\]{1,3}')
_WRITE_FALLBACK = re.compile(r'\[{1,3}write:\s*(\S+)\n([\s\S]*?)\[{1,3}/write\]{1,3}')

# Self-labeling drift: [AI] or [AI] with trailing text
_AI_TAG_PATTERN = re.compile(r'\[AI\]\s*', re.IGNORECASE)

# Models that only use XML (skip bracket/loose parsing)
_XML_ONLY_MODELS = {"gemini", "claude"}


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def _parse_xml(text: str) -> list[dict]:
    """Parse primary XML tool tags. Confidence: prime."""
    calls = []

    for m in _WRITE_PATTERN.finditer(text):
        calls.append({"tool": "write", "path": m.group(1).strip(), "content": m.group(2),
                       "confidence": "prime"})

    for m in _RUN_PATTERN.finditer(text):
        calls.append({"tool": "run", "command": m.group(1).strip(),
                       "confidence": "prime"})

    for m in _READ_PATTERN.finditer(text):
        path = m.group(1) or m.group(2)
        path = path.strip()
        attr_match = re.match(r'^path="([^"]+)"$', path)
        if attr_match:
            path = attr_match.group(1)
        calls.append({"tool": "read", "path": path, "confidence": "prime"})

    for m in _RECALL_PATTERN.finditer(text):
        calls.append({"tool": "recall", "query": m.group(1).strip(),
                       "confidence": "prime"})

    for m in _WAIT_PATTERN.finditer(text):
        duration = (m.group(1) or "").strip()
        call = {"tool": "wait", "confidence": "prime"}
        if duration:
            call["duration"] = duration
        calls.append(call)

    for m in _TOOLS_PATTERN.finditer(text):
        calls.append({"tool": "tools", "confidence": "prime"})

    for m in _CHECK_PATTERN.finditer(text):
        calls.append({"tool": "check", "path": m.group(1).strip(),
                       "confidence": "prime"})

    for m in _SEARCH_PATTERN.finditer(text):
        calls.append({"tool": "search", "query": m.group(1).strip(),
                       "confidence": "prime"})

    for m in _NAVIGATE_PATTERN.finditer(text):
        calls.append({"tool": "navigate", "path": m.group(1).strip(),
                       "confidence": "prime"})

    for m in _NAVIGATE_DRIFT.finditer(text):
        calls.append({"tool": "navigate", "path": m.group(1).strip(),
                       "confidence": "drift"})

    for m in _REPORT_PATTERN.finditer(text):
        calls.append({"tool": "report", "status": m.group(1).strip(),
                       "message": m.group(2).strip(), "confidence": "prime"})

    # Fallback: bare <report>STATUS</report> — treat content as status
    _report_matched = {m.start() for m in _REPORT_PATTERN.finditer(text)}
    for m in _REPORT_BARE.finditer(text):
        if m.start() not in _report_matched:
            calls.append({"tool": "report", "status": m.group(1).strip(),
                           "message": "", "confidence": "redundancy"})

    for m in _MEMORISE_PATTERN.finditer(text):
        mem_type = (m.group(1) or "").strip()
        content = m.group(2).strip()
        call = {"tool": "memorise", "content": content, "confidence": "prime"}
        if mem_type:
            call["mem_type"] = mem_type
        calls.append(call)

    return calls


def _parse_brackets(text: str) -> list[dict]:
    """Parse bracket fallback variants. Confidence: redundancy."""
    calls = []

    for m in _WRITE_FALLBACK.finditer(text):
        calls.append({"tool": "write", "path": m.group(1).strip(), "content": m.group(2),
                       "confidence": "redundancy"})

    for m in _RUN_FALLBACK.finditer(text):
        calls.append({"tool": "run", "command": m.group(1).strip(),
                       "confidence": "redundancy"})

    for m in _READ_FALLBACK.finditer(text):
        calls.append({"tool": "read", "path": m.group(1).strip(),
                       "confidence": "redundancy"})

    return calls


def _parse_loose(text: str) -> list[dict]:
    """Parse loose/hybrid bracket patterns. Confidence: drift."""
    calls = []

    # [run]cmd[/run] and [read]path[/read]
    for m in re.finditer(r'\[run\]>?\s*(.+?)\[/run\]', text):
        calls.append({"tool": "run", "command": m.group(1).strip(),
                       "confidence": "drift"})
    for m in re.finditer(r'\[read\]>?\s*(.+?)\[/read\]', text):
        calls.append({"tool": "read", "path": m.group(1).strip(),
                       "confidence": "drift"})

    # [read path="..."] and [write path="..."]...[/write]
    for m in re.finditer(r'\[read\s+path="([^"]+)"\]', text):
        calls.append({"tool": "read", "path": m.group(1).strip(),
                       "confidence": "drift"})
    for m in re.finditer(r'\[write\s+path="([^"]+)"\]([\s\S]*?)(?:\[/write\]|$)', text):
        calls.append({"tool": "write", "path": m.group(1).strip(), "content": m.group(2).strip(),
                       "confidence": "drift"})

    # [run cmd] and [read path] — no colon, no closing tag
    for m in re.finditer(r'\[run\s+([^\]]+)\]', text):
        cmd = re.sub(r'[()]', '', m.group(1)).strip()
        if cmd:
            calls.append({"tool": "run", "command": cmd, "confidence": "drift"})
    for m in re.finditer(r'\[read\s+(?!path=)([^\]]+)\]', text):
        path = re.sub(r'[()]', '', m.group(1)).strip()
        if path:
            calls.append({"tool": "read", "path": path, "confidence": "drift"})

    # [write path "content"] — bare path with inline content
    for m in re.finditer(r'\[write\s+(\S+)\s+"([^"]*)"', text):
        calls.append({"tool": "write", "path": m.group(1).strip(), "content": m.group(2).strip(),
                       "confidence": "drift"})

    return calls


def _translate_tags(text: str) -> list[dict]:
    """Translate invented tags into standard tool calls. Confidence: evolved."""
    calls = []

    for m in re.finditer(r'<mkdir\s+path="([^"]+)"[^>]*>', text):
        calls.append({"tool": "run", "command": f"mkdir -p {m.group(1).strip()}",
                       "confidence": "evolved"})

    # Pathless <write>content</write> — error: write requires a path
    for m in re.finditer(r'<write>([^<]*)</write>', text):
        calls.append({"tool": "write", "path": "", "content": m.group(1).strip(),
                       "confidence": "evolved"})

    return calls


def _parse_parens(text: str) -> list[dict]:
    """Parse parenthetical syntax: (run cmd), (read path), (write ...). Confidence: evolved."""
    calls = []

    # (run command)
    for m in re.finditer(r'\(run\s+(.+?)\)', text):
        calls.append({"tool": "run", "command": m.group(1).strip(),
                       "confidence": "evolved"})

    # (read path) or (read) — pathless
    for m in re.finditer(r'\(read(?:\s+(\S+))?\)', text):
        calls.append({"tool": "read", "path": (m.group(1) or "").strip(),
                       "confidence": "evolved"})

    # (write echo "content" > path) — shell-style inside parens
    for m in re.finditer(r'\(write\s+echo\s+"([^"]*)"\s*>\s*(\S+)\)', text):
        calls.append({"tool": "write", "path": m.group(2).strip(),
                       "content": m.group(1) + "\n",
                       "confidence": "evolved"})

    # (write "content") — quoted content, no path
    for m in re.finditer(r'\(write\s+"([^"]*)"\s*\)', text):
        calls.append({"tool": "write", "path": "", "content": m.group(1),
                       "confidence": "evolved"})

    # (write path "content") — unquoted path + quoted content
    for m in re.finditer(r'\(write\s+([^\s"]+)\s+"([^"]*)"(?:\s*>\s*\S+)?\)', text):
        calls.append({"tool": "write", "path": m.group(1).strip(),
                       "content": m.group(2),
                       "confidence": "evolved"})

    # (recall query)
    for m in re.finditer(r'\(recall\s+(.+?)\)', text):
        calls.append({"tool": "recall", "query": m.group(1).strip(),
                       "confidence": "evolved"})

    return calls


def _parse_bare_echo(text: str) -> list[dict]:
    """Detect bare echo "..." lines in raw text (not inside tool tags). Confidence: evolved."""
    calls = []
    for m in re.finditer(r'(?:^|\n)\s*echo\s+"([^"]*)"(?:\s*#[^\n]*)?\s*(?:\n|$)', text):
        content = m.group(1)
        # Skip if this echo has a redirect (handled by _echo_to_write via run)
        full = m.group(0)
        if '>' in full:
            continue
        calls.append({"tool": "intent", "text": content, "confidence": "evolved"})
    return calls


# ---------------------------------------------------------------------------
# Enrichment
# ---------------------------------------------------------------------------

# Echo redirect: echo "content" > file or echo "content" >> file
_ECHO_REDIRECT = re.compile(
    r'^echo\s+'
    r'(?:"([^"]*)"|\'([^\']*)\'|(\S+))'  # content: quoted or bare
    r'\s*(>>?)\s*'                         # > or >>
    r'(\S+)\s*$'                           # file path
)


def _echo_to_write(call: dict) -> dict | None:
    """Convert echo redirect run calls to write calls. Returns None if not a match."""
    command = call.get("command", "").strip()
    m = _ECHO_REDIRECT.match(command)
    if not m:
        return None
    content = (m.group(1) if m.group(1) is not None
               else m.group(2) if m.group(2) is not None
               else m.group(3))
    content += "\n"  # echo appends newline
    redirect = m.group(4)
    path = m.group(5)
    mode = "append" if redirect == ">>" else "overwrite"
    return {"tool": "write", "path": path, "content": content,
            "mode": mode, "confidence": call.get("confidence", "prime")}


# Bare echo (no redirect) → intent
_ECHO_BARE = re.compile(
    r'^echo\s+'
    r'(?:"([^"]*)"|\'([^\']*)\'|(.+))\s*$'
)


def _echo_to_intent(call: dict) -> dict | None:
    """Convert bare echo (no redirect) to an intent. Returns None if not a match."""
    command = call.get("command", "").strip()
    # Skip if it has a redirect — that's _echo_to_write's job
    if '>' in command:
        return None
    m = _ECHO_BARE.match(command)
    if not m:
        return None
    content = (m.group(1) if m.group(1) is not None
               else m.group(2) if m.group(2) is not None
               else m.group(3).strip())
    return {"tool": "intent", "text": content,
            "confidence": call.get("confidence", "prime")}


# cat file → read
_CAT_PATTERN = re.compile(r'^cat\s+(\S+)\s*$')

# cp source dest → copy
_CP_PATTERN = re.compile(r'^cp\s+(?:-[a-zA-Z]+\s+)*(\S+)\s+(\S+)\s*$')


def _cat_to_read(call: dict) -> dict | None:
    """Convert cat run calls to read calls."""
    command = call.get("command", "").strip()
    m = _CAT_PATTERN.match(command)
    if not m:
        return None
    return {"tool": "read", "path": m.group(1),
            "confidence": call.get("confidence", "prime")}


def _cp_to_copy(call: dict) -> dict | None:
    """Convert cp run calls to copy calls."""
    command = call.get("command", "").strip()
    m = _CP_PATTERN.match(command)
    if not m:
        return None
    return {"tool": "copy", "source": m.group(1), "dest": m.group(2),
            "confidence": call.get("confidence", "prime")}


def _enrich_run(call: dict) -> dict:
    """Extract target paths and metadata from a run command.

    Adds 'targets' list with destination paths found in shell redirects,
    copy/move destinations, mkdir targets, etc.
    """
    command = call.get("command", "")
    targets = []

    # Redirect targets: >> file, > file
    for m in re.finditer(r'>{1,2}\s*(\S+)', command):
        targets.append(m.group(1))

    # tee target
    for m in re.finditer(r'\btee\s+(?:-[a-z]\s+)*(\S+)', command):
        targets.append(m.group(1))

    # cp/mv destination (last argument)
    for m in re.finditer(r'\b(cp|mv)\s+.+\s+(\S+)\s*$', command):
        targets.append(m.group(2))

    # mkdir targets (stop at shell operators)
    for m in re.finditer(r'\bmkdir\s+(?:-[a-z]+\s+)*(.+?)(?:\s*(?:&&|\|\||[;|])\s*|$)', command):
        for path in m.group(1).split():
            if not path.startswith('-'):
                targets.append(path)

    if targets:
        call["targets"] = targets

    return call


def _key(call: dict) -> str:
    """Deduplication key for a tool call."""
    if call["tool"] == "run":
        return f"run:{call.get('command', '')}"
    elif call["tool"] == "read":
        return f"read:{call.get('path', '')}"
    elif call["tool"] == "write":
        path = call.get("path", "")
        return f"write:{path}:{call.get('content', '')[:20]}" if not path else f"write:{path}"
    elif call["tool"] == "recall":
        return f"recall:{call.get('query', '')}"
    elif call["tool"] == "intent":
        return f"intent:{call.get('text', '')[:40]}"
    elif call["tool"] == "copy":
        return f"copy:{call.get('source', '')}:{call.get('dest', '')}"
    elif call["tool"] == "wait":
        return "wait"
    elif call["tool"] == "tools":
        return "tools"
    elif call["tool"] == "check":
        return f"check:{call.get('path', '')}"
    elif call["tool"] == "search":
        return f"search:{call.get('query', '')}"
    elif call["tool"] == "navigate":
        return f"navigate:{call.get('path', '')}"
    elif call["tool"] == "report":
        return f"report:{call.get('status', '')}:{call.get('message', '')[:40]}"
    elif call["tool"] == "memorise":
        return f"memorise:{call.get('content', '')[:40]}"
    return f"{call['tool']}"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def distil(text: str, model_name: str = None, native_calls: list = None) -> tuple[list[dict], dict]:
    """Parse raw model output into structured tool calls.

    Args:
        text: Raw model output text.
        model_name: Optional model name. If gemini/claude, skip bracket/loose
                    parsing (they use clean XML). Ollama or None -> all tiers.
        native_calls: Optional list of structured tool calls from native SDK.
                      When present, wraps each as confidence "native" and strips
                      SDK metadata into metadata_context.

    Returns:
        (calls, metadata_context)
        calls: list of dicts with "confidence" field
        metadata_context: SDK-specific keys stripped during normalization,
                         keyed by call index. Passed to Synthesise.compose().
    """
    metadata = {}

    # Native SDK calls — wrap and strip metadata
    if native_calls is not None:
        calls = []
        for i, nc in enumerate(native_calls):
            call = dict(nc)
            call["confidence"] = "native"
            # Strip SDK metadata — Synthesise will need it later
            meta = {}
            for key in ("call_id", "tool_use_id", "function_call_ref"):
                if key in call:
                    meta[key] = call.pop(key)
            if meta:
                metadata[i] = meta
            if call.get("tool") == "run":
                converted = _echo_to_write(call) or _echo_to_intent(call) or _cat_to_read(call) or _cp_to_copy(call)
                if converted:
                    call = converted
                else:
                    _enrich_run(call)
            calls.append(call)
        return calls, metadata

    # Detect [AI] self-labeling drift — tag in metadata, strip from text
    ai_tags = _AI_TAG_PATTERN.findall(text)
    if ai_tags:
        metadata["ai_tag"] = {"count": len(ai_tags), "confidence": "drift"}
        text = _AI_TAG_PATTERN.sub("", text)

    # Text parsing
    all_calls = []
    seen = set()

    def _add(calls):
        for call in calls:
            k = _key(call)
            if k not in seen:
                seen.add(k)
                all_calls.append(call)

    xml_only = model_name and model_name.lower() in _XML_ONLY_MODELS

    # Always parse XML (all models)
    _add(_parse_xml(text))

    # Bracket/loose — only for local models or unknown
    if not xml_only:
        _add(_parse_brackets(text))
        _add(_parse_loose(text))

    # Invented tags and parenthetical syntax — always check
    _add(_translate_tags(text))
    _add(_parse_parens(text))

    # Bare echo — intent detection (local models only)
    if not xml_only:
        _add(_parse_bare_echo(text))

    # Stub: build
    if _BUILD_PATTERN.search(text):
        k = "build"
        if k not in seen:
            seen.add(k)
            all_calls.append({"tool": "build", "confidence": "prime"})

    # Convert shell idioms to native tools, enrich remaining run calls
    converted = []
    for call in all_calls:
        if call["tool"] == "run":
            c = _echo_to_write(call) or _echo_to_intent(call) or _cat_to_read(call) or _cp_to_copy(call)
            if c:
                converted.append(c)
            else:
                _enrich_run(call)
                converted.append(call)
        else:
            converted.append(call)

    # Mutation detection — log unrecognized XML tags
    for m in _UNKNOWN_TAG_PATTERN.finditer(text):
        tag = m.group(1).lower()
        if tag not in _KNOWN_TAGS:
            # Get surrounding context for the snippet
            start = max(0, m.start() - 20)
            end = min(len(text), m.end() + 80)
            snippet = text[start:end].replace('\n', ' ')
            _log_mutation(tag, model_name or "unknown", snippet)
            if "mutations" not in metadata:
                metadata["mutations"] = []
            metadata["mutations"].append(tag)

    return converted, metadata
