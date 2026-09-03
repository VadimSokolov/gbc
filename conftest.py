"""Pytest configuration for the gbc test suite.

Defines the ``slow`` marker and the ``--runslow`` opt-in.

A handful of tests have to train a network for long enough to show a real
training dynamic rather than a smoke-test forward pass. They are the ones that
demonstrate an architectural claim: that a body-only IQN saturates past the
quantile levels it was trained on, and that the spliced GPD head keeps
extrapolating. Neither can be established without actually fitting.

They are worth keeping and not worth paying for on every run, so they are
marked ``slow`` and deselected by default::

    python -m pytest                # fast suite
    python -m pytest --runslow      # everything, including the fits
    python -m pytest -m slow --runslow   # only the slow ones

CI should run with ``--runslow``. A bare ``pytest`` reports how many were
deselected, so a skipped guard is visible rather than silent.
"""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="run tests marked slow (network fits that take minutes)",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: trains a network long enough to matter; deselected unless --runslow",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="slow: pass --runslow to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
