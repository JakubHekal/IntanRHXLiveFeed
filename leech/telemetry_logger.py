from datetime import datetime
from pathlib import Path
import threading

_lock = threading.Lock()
_file_path = None


def set_telemetry_file(path: str):
    global _file_path
    with _lock:
        if not path:
            _file_path = None
            return
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        _file_path = p


def append_telemetry_line(line: str):
    if not line:
        return
    with _lock:
        if _file_path is None:
            return
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        try:
            with open(_file_path, "a", encoding="utf-8") as f:
                f.write(f"[{ts}] {line}\n")
        except Exception:
            # Telemetry must never break the realtime path.
            pass
