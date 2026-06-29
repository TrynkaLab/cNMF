"""Pytest session hooks for cNMF tests."""

from utils import (
    cleanup_reproducibility_fixtures_for_pytest,
    cleanup_runtime_cache_dirs,
    configure_test_cache_dirs,
    prepare_reproducibility_fixtures_for_pytest,
    write_pytest_log,
)


configure_test_cache_dirs()


def pytest_collection_modifyitems(config, items):
    """Prepare active-stack reproducibility fixtures only when that suite is collected."""
    prepare_reproducibility_fixtures_for_pytest(config, items)


def pytest_sessionfinish(session, exitstatus):
    """Clean generated reproducibility fixture artifacts after the test session."""
    cleanup_reproducibility_fixtures_for_pytest(session.config)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Write a compact test result log under tests/.cache."""
    write_pytest_log(terminalreporter, exitstatus)


def pytest_unconfigure(config):
    """Remove local runtime cache subdirectories after pytest has finished."""
    cleanup_runtime_cache_dirs()
