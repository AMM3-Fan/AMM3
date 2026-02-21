#!/usr/bin/env python3
"""
TrustProtocol.py - Identity, authorization, and action safety for AMM3.

Three responsibilities:
  1. Identity  — API key retrieval (keyring, GPG, env vars)
  2. Admission — admit() pre-gate for external model tool calls
  3. Safety    — evaluate() verdicts for all tool actions (block/confirm/allow)
"""

import json
import os
import re
import subprocess
from dataclasses import dataclass


@dataclass
class Verdict:
    """Structured security verdict for a tool call."""
    action: str       # "allow", "block", "confirm"
    risk: str         # "none", "low", "medium", "high"
    reasoning: str    # Human-readable explanation


_HOME = os.path.expanduser("~")
_ENV_GPG = os.path.join(os.path.dirname(__file__), '..', '.env.gpg')
_ENV_PLAIN = os.path.join(os.path.dirname(__file__), '..', '.env')

# ---------------------------------------------------------------------------
# Load trust config from kernel config (single source of truth)
# ---------------------------------------------------------------------------

_KERNEL_CONFIG = os.path.join(os.path.dirname(__file__), '..', 'nucleus', 'amm3.json')
with open(_KERNEL_CONFIG) as _f:
    _CONFIG = json.load(_f)
_TRUST = _CONFIG.get("trust", {})
_MODELS = _CONFIG.get("available_models", {})

# Cache decrypted keys for the session
_key_cache: dict[str, str] = {}


# ---------------------------------------------------------------------------
# Identity — API key retrieval
# ---------------------------------------------------------------------------

def get_secret(key: str) -> str:
    """Retrieve a secret: try keyring, then GPG-encrypted .env, then env var.

    Caches results in memory for the session lifetime.
    """
    if key in _key_cache:
        return _key_cache[key]

    val = _try_keyring(key) or _try_gpg(key) or _try_plain_env(key) or _try_env(key) or ""

    if val:
        _key_cache[key] = val

    return val


def _try_keyring(key: str) -> str:
    """Try macOS keyring via keyring library."""
    try:
        import keyring
        val = keyring.get_password("amm3", key)
        if val:
            return val
    except Exception:
        pass
    return ""


def _try_gpg(key: str) -> str:
    """Try GPG-encrypted .env file."""
    if not os.path.exists(_ENV_GPG):
        return ""
    try:
        result = subprocess.run(
            ['gpg', '--quiet', '--batch', '--decrypt', _ENV_GPG],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if line.startswith(f"{key}="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    except Exception:
        pass
    return ""


def _try_plain_env(key: str) -> str:
    """Try plain-text .env file at .amm3/.env."""
    if not os.path.exists(_ENV_PLAIN):
        return ""
    try:
        with open(_ENV_PLAIN) as f:
            for line in f:
                line = line.strip()
                if line.startswith(f"{key}="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    except Exception:
        pass
    return ""


def _try_env(key: str) -> str:
    """Try environment variable."""
    return os.environ.get(key, "")


def clear_cache():
    """Clear the in-memory key cache."""
    _key_cache.clear()


# ---------------------------------------------------------------------------
# Admission — pre-gate for external models
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Trust tiers — loaded from amm3.json
# ---------------------------------------------------------------------------

TRUST_TIERS = set(_TRUST.get("tiers", {}).keys()) or {"local", "remote", "guest"}

# Build agent tier mapping from available_models config
_KNOWN_AGENTS = set()
_AGENT_TIERS = {}
for _name, _info in _MODELS.items():
    tier = _info.get("trust", "guest")
    _AGENT_TIERS[_name] = tier
    if tier == "remote":
        _KNOWN_AGENTS.add(_name)


def get_trust_tier(model_name: str, is_local: bool = False) -> str:
    """Resolve a model name to its trust tier (from amm3.json)."""
    if is_local:
        return "local"
    return _AGENT_TIERS.get(model_name.lower(), "guest")


def admit(agent_name: str, tool_call: dict) -> dict:
    """Passport Control for external models.

    Evaluates by agent identity. Known agents are admitted,
    unknown agents are blocked. Architecture in place for
    future per-agent policies.

    Args:
        agent_name: Name of the agent requesting tool access.
        tool_call: The tool call dict from Distil.

    Returns:
        {"allowed": bool, "reason": str}
    """
    if agent_name.lower() in _KNOWN_AGENTS:
        return {"allowed": True, "reason": f"Known agent: {agent_name}"}

    return {
        "allowed": False,
        "reason": f"Unknown agent denied: {agent_name}"
    }


# ---------------------------------------------------------------------------
# Security data — loaded from amm3.json trust section
# ---------------------------------------------------------------------------

_sf = _TRUST.get("sensitive_files", {})
_SENSITIVE_UNIVERSAL = set(_sf.get("universal", [".env", ".env.gpg", "credentials.json"]))
_SENSITIVE_KEYS = set(_sf.get("keys", ["id_rsa", "id_ed25519", "id_ecdsa", "id_dsa"]))
_REMOTE_BLOCKED_FILES = set(_sf.get("remote_blocked", []))
_WRITE_BLOCKED_FILES = set(_sf.get("write_blocked", []))

_sd = _TRUST.get("sensitive_dirs", {})
_REMOTE_BLOCKED_DIRS = [os.path.join(_HOME, d) for d in _sd.get("remote_blocked", [])]

_BLOCKED_PATHS = _TRUST.get("blocked_paths", {}).get("system", [
    "/etc", "/var", "/usr", "/bin", "/sbin", "/System", "/Library", "/private"
])

_BLOCKED_COMMANDS = _TRUST.get("blocked_commands", {}).get("universal", [])

_DESTRUCTIVE_PATTERNS = _TRUST.get("destructive_patterns", {}).get("patterns", [])

_NETWORK_COMMANDS = _TRUST.get("network_commands", [])

_AUTO_ALLOW_COMMANDS = set(_TRUST.get("auto_allow", {}).get("commands", []))

_READ_ONLY_COMMANDS = set(_TRUST.get("read_only_commands", []))

_guest_cfg = _TRUST.get("guest", {})
_GUEST_SANDBOX = os.path.expanduser(_guest_cfg.get("sandbox", "~/.amm3/guest"))
_GUEST_COMMANDS = set(_guest_cfg.get("allowed_commands", []))
_GUEST_TOOLS = set(_guest_cfg.get("allowed_tools", []))


# ---------------------------------------------------------------------------
# Safety — action verdicts
# ---------------------------------------------------------------------------

def evaluate_run(command: str, targets: list = None, trust_tier: str = "remote") -> Verdict:
    """Evaluate a shell command for safety.

    Args:
        command: The shell command string.
        targets: Optional list of target paths extracted by _enrich_run().
        trust_tier: "local", "remote", or "guest".
    """
    lower = command.lower().strip()
    cmd_base = lower.split()[0] if lower.split() else ""
    cmd_base = cmd_base.rsplit("/", 1)[-1]  # strip path prefix

    # --- Guest tier: sandbox-only, limited commands ---
    if trust_tier == "guest":
        if cmd_base not in _GUEST_COMMANDS:
            return Verdict(
                action="block",
                risk="high",
                reasoning=f"Guest tier: '{cmd_base}' not in allowed commands"
            )
        # Even allowed commands must stay in sandbox
        # (target path checks happen below via targets)
        return Verdict(
            action="confirm",
            risk="low",
            reasoning=f"Guest command: {cmd_base}"
        )

    # --- Universal blocks (all tiers) ---

    # GPG keychain access — models must NEVER touch encrypted secrets
    if "gpg" in lower and ("decrypt" in lower or "-d " in lower or ".gpg" in lower or ".env" in lower):
        return Verdict(
            action="block",
            risk="high",
            reasoning="CRITICAL: Direct access to encrypted keychain (.env.gpg) is forbidden. "
                      "API keys are managed exclusively by TrustProtocol. "
                      "No model may decrypt, read, or exfiltrate secrets."
        )

    # Check hard blocklist
    for blocked in _BLOCKED_COMMANDS:
        if blocked in lower:
            return Verdict(
                action="block",
                risk="high",
                reasoning=f"Blocked: '{blocked}' is not allowed"
            )

    # Check destructive patterns
    for pattern in _DESTRUCTIVE_PATTERNS:
        if re.search(pattern, command):
            return Verdict(
                action="block",
                risk="high",
                reasoning=f"Destructive operation detected"
            )

    # --- Remote tier: additional blocks ---
    if trust_tier == "remote":
        # Block env dumps entirely for remote (not just confirm)
        env_dump = ["printenv", "env", "set"]
        if cmd_base in env_dump:
            return Verdict(
                action="block",
                risk="critical",
                reasoning="Remote model blocked from environment dump"
            )

        # Block history access
        if any(h in lower for h in ["history", ".zsh_history", ".bash_history"]):
            return Verdict(
                action="block",
                risk="high",
                reasoning="Remote model blocked from shell history"
            )

    # sudo — user decides (block for remote)
    if lower.startswith("sudo "):
        if trust_tier == "remote":
            return Verdict(
                action="block",
                risk="high",
                reasoning="Remote model blocked from sudo"
            )
        return Verdict(
            action="confirm",
            risk="high",
            reasoning="Elevated privileges (sudo)"
        )

    # Check enriched target paths (from _enrich_run) — before generic checks
    if targets:
        for target in targets:
            resolved = os.path.abspath(os.path.expandvars(os.path.expanduser(target)))
            basename = os.path.basename(resolved)
            if basename in _SENSITIVE_UNIVERSAL or basename in _SENSITIVE_KEYS:
                return Verdict(
                    action="block",
                    risk="high",
                    reasoning=f"Command targets sensitive file: {basename}"
                )
            # Remote: block sensitive files and dirs
            if trust_tier == "remote":
                if basename in _REMOTE_BLOCKED_FILES:
                    return Verdict(
                        action="block",
                        risk="high",
                        reasoning=f"Remote model blocked from sensitive file: {basename}"
                    )
                for blocked_dir in _REMOTE_BLOCKED_DIRS:
                    if resolved.startswith(blocked_dir):
                        return Verdict(
                            action="block",
                            risk="high",
                            reasoning=f"Remote model blocked from sensitive directory: {blocked_dir}"
                        )
            if not resolved.startswith(_HOME):
                for blocked in _BLOCKED_PATHS:
                    if resolved.startswith(blocked):
                        return Verdict(
                            action="confirm",
                            risk="high",
                            reasoning=f"Command targets system path: {target}"
                        )

    # Check for network access
    for net_cmd in _NETWORK_COMMANDS:
        if net_cmd in lower:
            return Verdict(
                action="confirm",
                risk="medium",
                reasoning="Network access"
            )

    # Check for write/modify operations
    if any(op in lower for op in [">>", "> ", "tee ", "mv ", "cp "]):
        return Verdict(
            action="confirm",
            risk="medium",
            reasoning="File modification operation"
        )

    # Check for pipe to shell
    if "|" in command and any(sh in lower for sh in ["sh", "bash", "zsh", "python"]):
        return Verdict(
            action="confirm",
            risk="high",
            reasoning="Pipe to shell interpreter"
        )

    # Environment dump — local only gets confirm
    env_dump = {"printenv", "env", "set"}
    if cmd_base in env_dump:
        return Verdict(
            action="confirm",
            risk="critical",
            reasoning="Environment dump exposes sensitive data (SSH sockets, paths, potential secrets)"
        )

    # Known safe commands — low risk, still requires confirmation
    if cmd_base in _AUTO_ALLOW_COMMANDS:
        return Verdict(
            action="confirm",
            risk="low",
            reasoning="Safe discovery command"
        )

    # Read-only but can expose file contents — low risk confirm
    if cmd_base in _READ_ONLY_COMMANDS:
        return Verdict(
            action="confirm",
            risk="low",
            reasoning="Read-only operation"
        )

    # Default: confirm with medium risk
    return Verdict(
        action="confirm",
        risk="medium",
        reasoning="General command execution"
    )


def evaluate_read(path: str, trust_tier: str = "remote") -> Verdict:
    """Evaluate a file read for safety."""
    resolved = os.path.realpath(os.path.expandvars(os.path.expanduser(path)))
    basename = os.path.basename(resolved)

    # Guest: can only read inside sandbox
    if trust_tier == "guest":
        if not resolved.startswith(_GUEST_SANDBOX):
            return Verdict(
                action="block",
                risk="high",
                reasoning=f"Guest tier: read blocked outside sandbox"
            )
        return Verdict(
            action="confirm",
            risk="low",
            reasoning="Guest read (sandbox)"
        )

    # Universal: sensitive files — always block
    if basename in _SENSITIVE_UNIVERSAL:
        return Verdict(
            action="block",
            risk="high",
            reasoning=f"Sensitive file: {basename}"
        )

    # Remote: expanded sensitive file/dir blocks
    if trust_tier == "remote":
        if basename in _REMOTE_BLOCKED_FILES:
            return Verdict(
                action="block",
                risk="high",
                reasoning=f"Remote model blocked from sensitive file: {basename}"
            )
        for blocked_dir in _REMOTE_BLOCKED_DIRS:
            if resolved.startswith(blocked_dir):
                return Verdict(
                    action="block",
                    risk="high",
                    reasoning=f"Remote model blocked from sensitive directory"
                )

    # Local: block SSH keys specifically
    if basename in _SENSITIVE_KEYS:
        return Verdict(
            action="block",
            risk="high",
            reasoning=f"Sensitive file: {basename}"
        )

    # Reads outside home
    if not resolved.startswith(_HOME):
        for blocked in _BLOCKED_PATHS:
            if resolved.startswith(blocked):
                return Verdict(
                    action="confirm",
                    risk="high",
                    reasoning=f"Read system path: {path}"
                )
        return Verdict(
            action="confirm",
            risk="medium",
            reasoning=f"Read outside home: {path}"
        )

    return Verdict(
        action="confirm",
        risk="low",
        reasoning="Read file"
    )


def evaluate_write(path: str, content: str = "", trust_tier: str = "remote") -> Verdict:
    """Evaluate a file write for safety."""
    resolved = os.path.realpath(os.path.expandvars(os.path.expanduser(path)))
    basename = os.path.basename(resolved)

    # Guest: can only write inside sandbox
    if trust_tier == "guest":
        if not resolved.startswith(_GUEST_SANDBOX):
            return Verdict(
                action="block",
                risk="high",
                reasoning=f"Guest tier: write blocked outside sandbox"
            )
        return Verdict(
            action="confirm",
            risk="low",
            reasoning="Guest write (sandbox)"
        )

    # Remote: block writes to dotfiles, config, and sensitive dirs entirely
    if trust_tier == "remote":
        if basename.startswith(".") and basename not in (".gitignore", ".gitkeep"):
            return Verdict(
                action="block",
                risk="high",
                reasoning=f"Remote model blocked from writing dotfile: {basename}"
            )
        for blocked_dir in _REMOTE_BLOCKED_DIRS:
            if resolved.startswith(blocked_dir):
                return Verdict(
                    action="block",
                    risk="high",
                    reasoning=f"Remote model blocked from writing to sensitive directory"
                )

    # Writes outside home — check specific dangerous paths
    if not resolved.startswith(_HOME):
        for blocked in _BLOCKED_PATHS:
            if resolved.startswith(blocked):
                return Verdict(
                    action="confirm",
                    risk="high",
                    reasoning=f"Write to system path: {path}"
                )
        return Verdict(
            action="confirm",
            risk="medium",
            reasoning=f"Write outside home: {path}"
        )

    # Block overwriting sensitive files (all tiers)
    if basename in _WRITE_BLOCKED_FILES:
        return Verdict(
            action="block",
            risk="high",
            reasoning=f"Write to sensitive file blocked: {basename}"
        )

    # Check for executable content
    if content.startswith("#!/"):
        return Verdict(
            action="confirm",
            risk="medium",
            reasoning=f"Write executable script: {path}"
        )

    return Verdict(
        action="confirm",
        risk="medium",
        reasoning=f"Write file: {path}"
    )


def evaluate(tool: str, trust_tier: str = "remote", **kwargs) -> Verdict:
    """Main entry point — evaluate any tool call.

    Args:
        tool: Tool name (run, read, write, etc.)
        trust_tier: "local", "remote", or "guest"
        **kwargs: Tool-specific arguments

    Backwards compatible: accepts is_local kwarg and converts to trust_tier.
    """
    # Backwards compatibility: convert is_local to trust_tier
    if "is_local" in kwargs:
        is_local = kwargs.pop("is_local")
        if is_local and trust_tier == "remote":
            trust_tier = "local"

    # Guest: only allowed tools from config
    if trust_tier == "guest" and tool not in _GUEST_TOOLS:
        return Verdict(
            action="block",
            risk="high",
            reasoning=f"Guest tier: tool '{tool}' not permitted"
        )

    if tool == "run":
        return evaluate_run(kwargs.get("command", ""), kwargs.get("targets"), trust_tier=trust_tier)
    elif tool == "read":
        return evaluate_read(kwargs.get("path", ""), trust_tier=trust_tier)
    elif tool == "write":
        return evaluate_write(kwargs.get("path", ""), kwargs.get("content", ""), trust_tier=trust_tier)
    elif tool == "copy":
        read_v = evaluate_read(kwargs.get("source", ""), trust_tier=trust_tier)
        write_v = evaluate_write(kwargs.get("dest", ""), trust_tier=trust_tier)
        # Stricter verdict wins
        severity = {"allow": 0, "confirm": 1, "block": 2}
        if severity.get(write_v.action, 2) >= severity.get(read_v.action, 0):
            return write_v
        return read_v
    elif tool == "intent":
        if trust_tier == "local":
            return Verdict(action="confirm", risk="low", reasoning="Intent expression (no execution)")
        return Verdict(action="confirm", risk="low", reasoning="Intent expression (external model)")
    elif tool == "recall":
        if trust_tier == "local":
            return Verdict(action="confirm", risk="low", reasoning="Memory recall (read-only)")
        return Verdict(action="confirm", risk="low", reasoning="Memory recall (external model)")
    elif tool == "wait":
        return Verdict(action="confirm", risk="low", reasoning="Model requests continuation — wait for more output")
    elif tool == "check":
        if trust_tier == "local":
            return Verdict(action="confirm", risk="low", reasoning="File metadata check (no content)")
        return Verdict(action="confirm", risk="low", reasoning="File metadata check (external model)")
    elif tool == "search":
        if trust_tier == "remote":
            return Verdict(action="confirm", risk="high", reasoning="File content search — exposes matching lines (remote)")
        return Verdict(action="confirm", risk="high", reasoning="File content search — exposes matching lines")
    elif tool == "navigate":
        path = kwargs.get("path", "")
        resolved = os.path.abspath(os.path.expandvars(os.path.expanduser(path)))
        # Guest: sandbox only
        if trust_tier == "guest":
            if not resolved.startswith(_GUEST_SANDBOX):
                return Verdict(action="block", risk="high", reasoning="Guest tier: navigate blocked outside sandbox")
            return Verdict(action="confirm", risk="low", reasoning="Guest navigate (sandbox)")
        for blocked in _BLOCKED_PATHS:
            if resolved.startswith(blocked):
                return Verdict(action="confirm", risk="high", reasoning=f"Navigate to system path: {path}")
        if not resolved.startswith(_HOME):
            return Verdict(action="confirm", risk="medium", reasoning=f"Navigate outside home: {path}")
        if trust_tier == "local":
            return Verdict(action="confirm", risk="low", reasoning="Navigate within home (local model)")
        return Verdict(action="confirm", risk="low", reasoning="Navigate within home (external model)")
    elif tool == "report":
        return Verdict(action="confirm", risk="low", reasoning="Status report")
    elif tool == "memorise":
        if trust_tier == "guest":
            return Verdict(action="block", risk="high", reasoning="Guest: memorise not permitted")
        count = kwargs.get("knowledge_count", 0)
        at_cap = count >= 33
        warning = f" — knowledge store full ({count}/33), oldest entry will be removed" if at_cap else f" ({count}/33)"
        return Verdict(action="confirm", risk="low", reasoning=f"Knowledge tag{warning}")
    else:
        return Verdict(
            action="block",
            risk="high",
            reasoning=f"Unknown tool: {tool}"
        )
