import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable

TRANSIENT_STATUS = {429, 500, 502, 503, 504}
PERMANENT_STATUS = {400, 401, 403, 404, 422}
CALL_TIMEOUT_SECONDS = 60.0
MAX_ATTEMPTS = 3


def classify_failure(exc: BaseException) -> tuple[str, float | None]:
    """Return 'transient' or 'permanent', plus Retry-After seconds when present."""
    status = getattr(exc, "status_code", None)
    response = getattr(exc, "response", None)
    if status is None and response is not None:
        status = getattr(response, "status_code", None)
    retry_after = _retry_after(exc)
    if _postgres_code(exc) == "42601":
        return "permanent", None
    if status in TRANSIENT_STATUS:
        return "transient", retry_after
    if status in PERMANENT_STATUS:
        return "permanent", None
    text = f"{type(exc).__name__} {exc}".lower()
    if any(token in text for token in ("timeout", "timed out", "connection", "connecterror", "temporarily")):
        return "transient", retry_after
    return "permanent", None


def _postgres_code(exc: BaseException) -> str | None:
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        code = getattr(current, "code", None)
        if str(code or "") == "42601" or "42601" in str(current):
            return "42601"
        current = current.__cause__
    return None


def _retry_after(exc: BaseException) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) or getattr(exc, "headers", None)
    if not headers:
        return None
    raw = headers.get("retry-after") or headers.get("Retry-After")
    if raw is None:
        return None
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return None


def call_with_retry(
    operation: str,
    fn: Callable[[], Any],
    *,
    attempts: int = MAX_ATTEMPTS,
    base_delay: float = 0.5,
    sleep: Callable[[float], None] = time.sleep,
    rng: random.Random | None = None,
    on_retry: Callable[[str, int, str, float], None] | None = None,
) -> Any:
    """Try an operation up to `attempts` times. Permanent errors are not retried."""
    if attempts < 1:
        raise ValueError("attempts must be at least 1.")
    generator = rng or random.Random()
    last: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:
            last = exc
            kind, retry_after = classify_failure(exc)
            if kind != "transient" or attempt == attempts:
                raise
            if retry_after is None:
                delay = min(30.0, base_delay * (2 ** (attempt - 1)))
                delay += generator.random() * 0.25
                delay = round(delay, 3)
            else:
                delay = round(retry_after, 3)
            if on_retry is not None:
                on_retry(operation, attempt, _reason(exc), delay)
            sleep(delay)
    assert last is not None
    raise last


def _reason(exc: BaseException) -> str:
    if _postgres_code(exc) == "42601":
        return "PostgreSQL 42601"
    status = getattr(exc, "status_code", None)
    if status is None:
        return type(exc).__name__
    return f"HTTP {status}"


class ProgressLog:
    def __init__(self, total: int, stream: Any = None) -> None:
        self.total = total
        self.stream = stream or sys.stdout
        self.started = time.monotonic()
        self.completed = 0
        self.calls = 0
        self.retries = 0
        self.failures = 0

    def line(self, message: str) -> None:
        stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.stream.write(f"{stamp} {message}\n")
        self.stream.flush()

    def status(self, *, question: str, stage: str, config_id: str = "") -> None:
        elapsed = time.monotonic() - self.started
        remaining = max(self.total - self.completed, 0)
        eta = 0.0 if self.completed == 0 else elapsed / self.completed * remaining
        config = f" config={config_id}" if config_id else ""
        self.line(
            f"{question} stage={stage}{config} done={self.completed}/{self.total} "
            f"elapsed={_fmt(elapsed)} eta={_fmt(eta)} "
            f"calls={self.calls} retries={self.retries} failures={self.failures}"
        )

    def retry(self, operation: str, attempt: int, reason: str, delay: float) -> None:
        self.retries += 1
        self.line(
            f"retry operation={operation} attempt={attempt}/{MAX_ATTEMPTS} "
            f"reason={reason} wait_s={delay}"
        )

    def question_summary(self, question: str, success: int, failed: int) -> None:
        self.line(f"{question} summary success={success} failed={failed}")

    def final_summary(self, success: int, failed: int) -> None:
        self.line(
            f"stage=export done={self.completed}/{self.total} "
            f"success={success} failed={failed} calls={self.calls} "
            f"retries={self.retries} failures={self.failures} elapsed={_fmt(time.monotonic() - self.started)}"
        )


def _fmt(seconds: float) -> str:
    whole = max(0, int(seconds))
    return f"{whole // 60}m{whole % 60:02d}s"


def append_jsonl(path: Path, row: dict) -> None:
    line = json.dumps(row, ensure_ascii=False) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def load_jsonl(path: Path) -> list[dict]:
    """Load records. A truncated final line is ignored and is not a completed operation."""
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    rows: list[dict] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def success_keys(rows: list[dict], stage: str) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for row in rows:
        if row.get("stage") not in (None, stage):
            continue
        if row.get("status", "success") != "success":
            continue
        question_id = row.get("question_id")
        config_id = row.get("config_id")
        if question_id and config_id:
            keys.add((question_id, config_id))
    return keys


def latest_success(rows: list[dict]) -> list[dict]:
    chosen: dict[tuple[str, str], dict] = {}
    for row in rows:
        if row.get("status", "success") != "success":
            continue
        question_id = row.get("question_id")
        config_id = row.get("config_id")
        if not question_id or not config_id:
            continue
        chosen[(question_id, config_id)] = row
    return list(chosen.values())


def fingerprint(manifest: dict, questions: list[dict], layers: set[str]) -> str:
    import hashlib

    payload = {
        "baseline_config_id": manifest["baseline_config_id"],
        "collections": manifest["collections"],
        "candidate_k": manifest["retrieval"]["candidate_k"],
        "rrf_k": manifest["retrieval"]["rrf_k"],
        "model": manifest["generate"]["model"],
        "judge_model": manifest["generate"]["judge_model"],
        "temperature": manifest["generate"]["temperature"],
        "max_output_tokens": manifest["generate"]["max_output_tokens"],
        "system_prompt": manifest["generate"]["system_prompt"],
        "user_template": manifest["generate"]["user_template"],
        "lexical_version": manifest["retrieval"].get("lexical_version", "english_and"),
        "metrics_contract": "v0.4",
        "layers": sorted(layers),
        "questions": [
            {
                "question_id": item["question_id"],
                "question": item["question"],
                "evidence_quote": item["evidence_quote"],
                "minimal_support_spans": item["minimal_support_spans"],
                "grading": item["grading"],
                "table_ambiguous": bool(item.get("table_ambiguous")),
            }
            for item in questions
        ],
    }
    strategies = manifest.get("strategies")
    if strategies is not None and set(strategies) != {"lexical", "vector", "hybrid"}:
        payload["strategies"] = sorted(strategies)
    if manifest.get("config_ids"):
        payload["config_ids"] = list(manifest["config_ids"])
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_or_check_checkpoint(
    path: Path,
    fingerprint_value: str,
    *,
    run_id: str,
    lexical_version: str = "english_and",
) -> None:
    if not path.exists():
        path.write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "lexical_version": lexical_version,
                    "fingerprint": fingerprint_value,
                    "call_timeout_seconds": CALL_TIMEOUT_SECONDS,
                    "max_attempts": MAX_ATTEMPTS,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return
    saved = json.loads(path.read_text(encoding="utf-8"))
    saved_version = saved.get("lexical_version")
    if (
        saved.get("run_id") != run_id
        or saved.get("fingerprint") != fingerprint_value
        or (saved_version is not None and saved_version != lexical_version)
    ):
        raise RuntimeError(
            "Checkpoint does not match this run-id, manifest, ground truth, lexical version, or model settings."
        )
