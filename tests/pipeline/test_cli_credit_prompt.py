import io
import threading
import time

import pytest

from router.common.config import ModelConfig
from router.pipeline import cli as cli_module
from router.pipeline.calibration.calibrate import CreditExhaustedError
from router.pipeline.calibration.grading.base import Task


class _FakeStdin(io.StringIO):
    def __init__(self, text: str, tty: bool):
        super().__init__(text)
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


@pytest.fixture(autouse=True)
def _no_desktop_notifications(monkeypatch):
    monkeypatch.setattr(cli_module, "_notify_desktop", lambda title, body: None)


def _error() -> CreditExhaustedError:
    task = Task(task_id="t1", source="bigcodebench", prompt="p", reference_solution="", row={})
    model = ModelConfig(
        model_id="gpt-5.6-sol", provider="openai", runner="pi",
        cost_input=0.004, cost_output=0.02, context_window=272000, max_tokens=128000,
    )
    return CreditExhaustedError(task, model, "429 You exceeded your current quota")


def test_pause_resumes_on_enter(monkeypatch, tmp_path):
    monkeypatch.setattr(cli_module.sys, "stdin", _FakeStdin("\n", tty=True))
    monkeypatch.setattr(cli_module.select, "select", lambda r, w, x, t: (r, [], []))
    handler = cli_module._make_credit_pause_handler(tmp_path / "PAUSED", auto_retry_seconds=0, poll_seconds=0.01)

    assert handler(_error()) is True
    assert not (tmp_path / "PAUSED").exists()


def test_pause_resumes_unattended_after_the_auto_retry_interval(monkeypatch, tmp_path):
    # Nobody at the machine (nohup / no terminal): must keep waiting, then re-probe — never abort.
    monkeypatch.setattr(cli_module.sys, "stdin", _FakeStdin("", tty=False))
    handler = cli_module._make_credit_pause_handler(tmp_path / "PAUSED", auto_retry_seconds=0.03, poll_seconds=0.01)

    assert handler(_error()) is True
    assert not (tmp_path / "PAUSED").exists()


def test_pause_resumes_when_someone_deletes_the_pause_file(monkeypatch, tmp_path):
    # e.g. a coworker over SSH: `rm .cache/CALIBRATION_PAUSED`. auto_retry_seconds=0 = never on its own.
    monkeypatch.setattr(cli_module.sys, "stdin", _FakeStdin("", tty=False))
    pause_file = tmp_path / "PAUSED"
    handler = cli_module._make_credit_pause_handler(pause_file, auto_retry_seconds=0, poll_seconds=0.01)
    result = {}
    worker = threading.Thread(target=lambda: result.setdefault("value", handler(_error())))
    worker.start()

    deadline = time.monotonic() + 5
    while not pause_file.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    contents = pause_file.read_text(encoding="utf-8")
    assert "gpt-5.6-sol" in contents and "t1" in contents and f"rm {pause_file}" in contents
    assert worker.is_alive()  # still paused

    pause_file.unlink()
    worker.join(timeout=5)
    assert result == {"value": True}


def test_pause_does_not_abort_on_end_of_input(monkeypatch, tmp_path):
    # Closed stdin stops listening for Enter but must keep pausing until another trigger fires.
    monkeypatch.setattr(cli_module.sys, "stdin", _FakeStdin("", tty=True))
    monkeypatch.setattr(cli_module.select, "select", lambda r, w, x, t: (r, [], []))
    handler = cli_module._make_credit_pause_handler(tmp_path / "PAUSED", auto_retry_seconds=0.03, poll_seconds=0.01)

    assert handler(_error()) is True
