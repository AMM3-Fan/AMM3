#!/usr/bin/env python3
"""
Graph.py - Memory layer for AMM3.

SessionMemory: structured JSONL session transcript writer.
SemanticRecall: vector-based semantic retrieval over nucleus/knowledge.jsonl.

Vector store lives in .amm3/memory/knowledge_vectors.jsonl alongside session logs.
Embeddings via Ollama (nomic-embed-text). Falls back to recency if Ollama is unavailable.
"""

import hashlib
import json
import os
import uuid
from datetime import datetime


# ---------------------------------------------------------------------------
# SessionMemory
# ---------------------------------------------------------------------------

class SessionMemory:
    """Persists session transcript to .amm3/memory/ as structured JSONL."""

    def __init__(self, base_dir: str = None):
        if base_dir is None:
            base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'memory')
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

        self.session_id = datetime.now().strftime('%Y-%m-%d_%H-%M')
        self.started = datetime.now().strftime('%Y-%m-%d %H:%M')
        self.log_path = os.path.join(self.base_dir, f'{self.session_id}.jsonl')

        # Restrict session log to owner only (no other users can read it)
        open(self.log_path, 'a').close()
        os.chmod(self.log_path, 0o600)

        self._entry("AMM3", "system", f"Session started: {self.started}")

    def _entry(self, speaker: str, entry_type: str, content: str, meta: dict = None):
        """Build and write a single JSONL entry."""
        entry = {
            "id": uuid.uuid4().hex[:8],
            "ts": datetime.now().isoformat(),
            "speaker": speaker,
            "type": entry_type,
            "content": content,
            "meta": meta or {},
        }
        with open(self.log_path, 'a') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    def log(self, text: str):
        """Log a system-level message (backwards-compatible)."""
        self._entry("AMM3", "system", text)

    def log_agent_output(self, agent: str, output: str):
        """Log agent output, auto-detecting structured JSON content."""
        try:
            parsed = json.loads(output)
            if isinstance(parsed, dict):
                keys = list(parsed.keys())
                self._entry(agent, "structured", output, {"format": "json", "keys": keys})
                return
        except (json.JSONDecodeError, ValueError):
            pass
        self._entry(agent, "chat", output)

    def log_message(self, role: str, content: str, model: str = None, target: str = None):
        """Log a conversation message."""
        speaker = model or role
        meta = {"target": target} if target else {}
        self._entry(speaker, "chat", content, meta)

    def log_tool(self, tool: str, args, verdict: str):
        """Log a tool execution request with its verdict."""
        content = json.dumps(args, ensure_ascii=False) if isinstance(args, dict) else str(args)
        self._entry("AMM3", "tool_exec", content, {"tool": tool, "verdict": verdict})

    def log_tool_output(self, tool: str, output: str):
        """Log tool execution output."""
        self._entry("AMM3", "tool_output", output, {"tool": tool})

    def log_knowledge(self, speaker: str, content: str, mem_type: str = None):
        """Log explicitly tagged knowledge (from <memorise> tags)."""
        meta = {"tagged": True}
        if mem_type:
            meta["mem_type"] = mem_type
        self._entry(speaker, "knowledge", content, meta)

    def save_summary(self):
        ended = datetime.now().strftime('%Y-%m-%d %H:%M')
        self._entry("AMM3", "system", f"Session ended: {ended}")


# ---------------------------------------------------------------------------
# SemanticRecall
# ---------------------------------------------------------------------------

_MEMORY_DIR  = os.path.join(os.path.dirname(__file__), '..', 'memory')
_VECTORS_DIR = os.path.join(_MEMORY_DIR, 'vectors')
_META_PATH   = os.path.join(_VECTORS_DIR, 'knowledge.jsonl')  # human-readable, no vectors
_BIN_PATH    = os.path.join(_VECTORS_DIR, 'vectors.npy')      # compact float32 matrix
_EMBED_MODEL = "nomic-embed-text"
_OLLAMA_URL  = "http://localhost:11434/api/embed"

# Entry types from session logs worth indexing
_INDEX_TYPES = {"chat", "knowledge"}


def _embed(text: str) -> list | None:
    """Request an embedding from Ollama. Returns None on any failure."""
    try:
        import urllib.request
        payload = json.dumps({"model": _EMBED_MODEL, "input": text}).encode()
        req = urllib.request.Request(
            _OLLAMA_URL, data=payload,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        return data["embeddings"][0]
    except Exception:
        return None


def _load_store() -> tuple:
    """Load (meta_list, matrix).

    meta_list: ordered list of {id, content, speaker, ts}.
    matrix:    numpy float32 array of shape (N, D), or None if unavailable.

    Auto-converts old format (inline vectors in JSONL) on first load.
    """
    import numpy as np

    if not os.path.isfile(_META_PATH):
        return [], None

    meta = []
    try:
        with open(_META_PATH) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # Old format: inline vectors — auto-convert
                if "vector" in obj:
                    return _convert_legacy()
                meta.append(obj)
    except Exception:
        return [], None

    if not meta:
        return [], None

    if not os.path.isfile(_BIN_PATH):
        return meta, None

    try:
        matrix = np.load(_BIN_PATH)
        if matrix.shape[0] != len(meta):
            return meta, None  # Mismatch — treat as no vectors
        return meta, matrix
    except Exception:
        return meta, None


def _convert_legacy() -> tuple:
    """Convert old inline-vector JSONL to split format. Called once."""
    import numpy as np

    meta, vectors = [], []
    try:
        with open(_META_PATH) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if "vector" not in obj or "id" not in obj:
                        continue
                    meta.append({
                        "id":      obj["id"],
                        "content": obj.get("content", ""),
                        "speaker": obj.get("speaker", "?"),
                        "ts":      obj.get("ts", ""),
                    })
                    vectors.append(obj["vector"])
                except (json.JSONDecodeError, KeyError):
                    continue
    except Exception:
        return [], None

    if not meta:
        return [], None

    matrix = np.array(vectors, dtype=np.float32)
    _save_store(meta, matrix)
    return meta, matrix


def _save_store(meta: list, matrix):
    """Write metadata JSONL (no vectors) + numpy binary."""
    import numpy as np
    os.makedirs(_VECTORS_DIR, exist_ok=True)

    with open(_META_PATH, 'w') as f:
        for obj in meta:
            f.write(json.dumps({
                "id":      obj["id"],
                "content": obj["content"],
                "speaker": obj["speaker"],
                "ts":      obj["ts"],
            }, ensure_ascii=False) + '\n')

    np.save(_BIN_PATH, np.array(matrix, dtype=np.float32))


class SemanticRecall:
    """Semantic retrieval over session logs using Ollama embeddings.

    knowledge.jsonl — human-readable metadata (id, content, speaker, ts).
    vectors.npy     — compact float32 matrix, rows align with JSONL lines.
    """

    def store(self, entry: dict) -> bool:
        """Embed a single entry and append it to the store immediately.

        Uses entry["id"] if present, otherwise derives one from content hash.
        Returns True if embedding succeeded.
        """
        import numpy as np

        content = entry.get("content", "")
        if not content:
            return False
        eid = entry.get("id") or hashlib.sha256(content.encode()).hexdigest()[:8]

        vector = _embed(content)
        if vector is None:
            return False

        meta, matrix = _load_store()
        existing_ids = {e["id"] for e in meta}
        if eid in existing_ids:
            return True

        meta.append({
            "id":      eid,
            "content": content[:500],
            "speaker": entry.get("speaker", "?"),
            "ts":      entry.get("ts", "")[:10],
        })
        new_row = np.array([vector], dtype=np.float32)
        matrix = np.vstack([matrix, new_row]) if matrix is not None else new_row

        _save_store(meta, matrix)
        return True

    def migrate(self):
        """Index all unembedded session log entries.

        Scans memory/*.jsonl. Indexes chat and knowledge type entries.
        Stops silently if Ollama is unavailable, resumes next session.
        """
        import glob
        import numpy as np

        meta, matrix = _load_store()
        existing_ids = {e["id"] for e in meta}
        new_meta = list(meta)
        new_vecs = list(matrix) if matrix is not None else []
        changed = False

        log_files = sorted(glob.glob(os.path.join(_MEMORY_DIR, '*.jsonl')))
        for log_file in log_files:
            try:
                with open(log_file, 'r', errors='replace') as f:
                    lines = [l for l in f if l.strip()]
            except Exception:
                continue
            for line in lines:
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if entry.get("type") not in _INDEX_TYPES:
                    continue
                content = entry.get("content", "")
                if not content:
                    continue
                eid = entry.get("id") or hashlib.sha256(content.encode()).hexdigest()[:8]
                if eid in existing_ids:
                    continue

                vector = _embed(content)
                if vector is None:
                    if changed:
                        _save_store(new_meta, np.array(new_vecs, dtype=np.float32))
                    return  # Ollama unavailable — retry next session

                new_meta.append({
                    "id":      eid,
                    "content": content[:500],
                    "speaker": entry.get("speaker", "?"),
                    "ts":      entry.get("ts", "")[:10],
                })
                new_vecs.append(vector)
                existing_ids.add(eid)
                changed = True

        if changed:
            _save_store(new_meta, np.array(new_vecs, dtype=np.float32))

    def query(self, context: str, top_k: int = 5) -> list:
        """Return top_k most semantically relevant entries for context.

        Falls back to most-recent top_k if Ollama is unavailable.
        Returns list of {content, speaker, ts} dicts.
        """
        import numpy as np

        meta, matrix = _load_store()
        if not meta:
            return []

        if matrix is None:
            return [{"content": e["content"], "speaker": e["speaker"], "ts": e["ts"]}
                    for e in meta[-top_k:]]

        q_vec = _embed(context)
        if q_vec is None:
            return [{"content": e["content"], "speaker": e["speaker"], "ts": e["ts"]}
                    for e in meta[-top_k:]]

        q = np.array(q_vec, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return [{"content": e["content"], "speaker": e["speaker"], "ts": e["ts"]}
                    for e in meta[-top_k:]]
        q = q / q_norm

        # Vectorised cosine similarity over the full matrix
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1
        scores = (matrix / norms) @ q

        top_idx = np.argsort(scores)[::-1][:top_k]
        return [{"content": meta[i]["content"], "speaker": meta[i]["speaker"], "ts": meta[i]["ts"]}
                for i in top_idx]
