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
