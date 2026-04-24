"""Ultra-lightweight vLLM conversation monitor.

Appends one JSON line per completed vLLM call to a JSONL file.
A separate SSE server tails this file and streams to browsers.

Impact: ~0.1ms file append per call. If anything fails, silently continues.
"""
import json, os, time, threading

MONITOR_FILE = "/tmp/vllm_conversations.jsonl"
_lock = threading.Lock()
# Auto-truncate at 50MB to prevent disk fill
MAX_FILE_SIZE = 50 * 1024 * 1024


def emit(script, prompt, response, latency_ms, status="ok", tokens=0, error=None):
    """Append a conversation event. Never raises."""
    try:
        event = {
            "ts": time.time(),
            "script": script,
            "prompt": prompt[:3000],
            "response": (response or "")[:5000],
            "latency_ms": round(latency_ms),
            "status": status,
            "tokens": tokens,
        }
        if error:
            event["error"] = str(error)[:500]
        line = json.dumps(event, ensure_ascii=False) + "\n"
        with _lock:
            try:
                sz = os.path.getsize(MONITOR_FILE)
                if sz > MAX_FILE_SIZE:
                    os.truncate(MONITOR_FILE, 0)
            except FileNotFoundError:
                pass
            with open(MONITOR_FILE, "a") as f:
                f.write(line)
    except Exception:
        pass
