import logging

from router.common.logging_config import configure_logging


def test_configure_logging_sets_level_and_a_single_handler():
    configure_logging(level="WARNING")
    logger = logging.getLogger("router")
    assert logger.level == logging.WARNING
    assert len(logger.handlers) == 1


def test_configure_logging_is_idempotent():
    configure_logging()
    configure_logging()
    logger = logging.getLogger("router")
    assert len(logger.handlers) == 1  # replaced, not stacked


def test_configure_logging_formats_a_plain_readable_line(capsys):
    configure_logging(level="INFO")
    logger = logging.getLogger("router.pipeline.corpus")
    logger.info("loaded 20 rows from swe-smith")

    err = capsys.readouterr().err
    assert "INFO" in err
    assert "loaded 20 rows from swe-smith" in err
    # No structured fields — the message itself is the whole line.
    assert "router.pipeline.corpus" not in err
    assert "event" not in err


def test_child_loggers_respect_the_configured_level(capsys):
    configure_logging(level="WARNING")
    logger = logging.getLogger("router.common.embedding")
    logger.info("this should be suppressed")
    logger.warning("this should show up")

    err = capsys.readouterr().err
    assert "this should be suppressed" not in err
    assert "this should show up" in err


def test_log_file_none_keeps_the_single_stderr_handler():
    configure_logging(level="INFO", log_file=None)
    logger = logging.getLogger("router")
    assert len(logger.handlers) == 1


def test_log_file_adds_a_second_handler_and_writes_to_disk(tmp_path):
    log_path = tmp_path / "run.log"
    configure_logging(level="INFO", log_file=log_path)
    logger = logging.getLogger("router")
    assert len(logger.handlers) == 2

    logging.getLogger("router.pipeline.calibration.calibrate").info("calibrating llama-3.1-8b-instant")
    for handler in logger.handlers:
        handler.flush()

    assert "calibrating llama-3.1-8b-instant" in log_path.read_text(encoding="utf-8")


def test_log_file_still_writes_to_stderr_too(tmp_path, capsys):
    configure_logging(level="INFO", log_file=tmp_path / "run.log")
    logging.getLogger("router.pipeline.corpus").info("both sinks should see this")

    assert "both sinks should see this" in capsys.readouterr().err


def test_log_file_creates_missing_parent_directories(tmp_path):
    log_path = tmp_path / "nested" / "dir" / "run.log"
    configure_logging(level="INFO", log_file=log_path)
    assert log_path.parent.is_dir()


def test_log_file_is_appended_to_not_overwritten_on_a_second_run(tmp_path):
    log_path = tmp_path / "run.log"
    configure_logging(level="INFO", log_file=log_path)
    logging.getLogger("router").info("first run")
    for handler in logging.getLogger("router").handlers:
        handler.flush()

    configure_logging(level="INFO", log_file=log_path)  # simulates a second CLI invocation
    logging.getLogger("router").info("second run")
    for handler in logging.getLogger("router").handlers:
        handler.flush()

    content = log_path.read_text(encoding="utf-8")
    assert "first run" in content
    assert "second run" in content
