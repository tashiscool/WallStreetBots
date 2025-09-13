from __future__ import annotations
import time, os, json, uuid, pathlib


class ClockDriftError(Exception): ...


def assert_ntp_ok(max_drift_ms: int = 250) -> None:
    # Without NTP service call, use monotonic as a weak guard.
    t1 = time.time()
    t2 = time.perf_counter()
    time.sleep(0.01)
    drift_ms = abs((time.time() - t1) - (time.perf_counter() - t2)) * 1000
    if drift_ms > max_drift_ms:
        raise ClockDriftError(
            f"Clock drift detected: ~{drift_ms:.1f}ms > {max_drift_ms}ms"
        )


class Journal:
    """Append-only JSONL journal for decisions and a replay cursor."""

    def __init__(self, path: str = "./.state/journal.jsonl"):
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.cursor_path = self.path.with_suffix(".cursor")

    def append(self, record: dict) -> str:
        record = {"id": str(uuid.uuid4()), **record}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        return record["id"]

    def replay_from_cursor(self) -> list[dict]:
        pos = 0
        if self.cursor_path.exists():
            pos = int(self.cursor_path.read_text().strip() or "0")
        out = []
        with self.path.open("r", encoding="utf-8") as f:
            f.seek(pos)
            for line in f:
                out.append(json.loads(line))
            self.cursor_path.write_text(str(f.tell()))
        return out
