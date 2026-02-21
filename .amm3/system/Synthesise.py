#!/usr/bin/env python3
"""
Synthesise.py - Format tool results for target models.

Output side of the trust pipeline. Re-attaches metadata stripped by Distil
and formats results for the target model type (ollama, gemini, claude).
"""


def compose(results: list[dict], model_type: str, metadata: dict = None) -> list[dict]:
    """Format tool results for the target model.

    Args:
        results: list of {"index": int, "name": str, "output": str, "cwd": str}
        model_type: "ollama", "gemini", or "claude"
        metadata: context from Distil.distil() — SDK keys stripped during normalization

    Returns:
        Formatted result list appropriate for the target model.
    """
    if metadata is None:
        metadata = {}

    if model_type == "ollama":
        return _compose_ollama(results)
    elif model_type == "gemini":
        return _compose_gemini(results, metadata)
    elif model_type == "claude":
        return _compose_claude(results, metadata)
    else:
        return _compose_ollama(results)


def _compose_ollama(results: list[dict]) -> list[dict]:
    """Format for Ollama — text messages with cwd prefix."""
    formatted = []
    for r in results:
        cwd = r.get("cwd", "~")
        output = r.get("output", "[No output]")
        formatted.append({
            "role": "user",
            "content": f"[AMM3 @ {cwd}]:\n{output}"
        })
    return formatted


def _compose_gemini(results: list[dict], metadata: dict) -> list[dict]:
    """Format for Gemini — tool result dicts with SDK metadata re-attached."""
    formatted = []
    for r in results:
        idx = r.get("index", 0)
        entry = {
            "name": r.get("name", "run"),
            "output": r.get("output", "[No output]"),
        }
        # Re-attach SDK metadata that Distil stripped
        if idx in metadata:
            entry.update(metadata[idx])
        formatted.append(entry)
    return formatted


def _compose_claude(results: list[dict], metadata: dict) -> list[dict]:
    """Format for Claude — tool_result dicts with SDK metadata re-attached."""
    formatted = []
    for r in results:
        idx = r.get("index", 0)
        entry = {
            "type": "tool_result",
            "name": r.get("name", "run"),
            "content": r.get("output", "[No output]"),
        }
        # Re-attach SDK metadata that Distil stripped
        if idx in metadata:
            entry.update(metadata[idx])
        formatted.append(entry)
    return formatted
