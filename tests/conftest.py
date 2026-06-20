"""Pytest config: gate the network/torch smoke test behind --run-smoke."""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-smoke",
        action="store_true",
        default=False,
        help="run the end-to-end smoke test (needs network + torch)",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-smoke"):
        return
    skip = pytest.mark.skip(reason="needs --run-smoke (network + torch)")
    for item in items:
        if "smoke" in item.keywords:
            item.add_marker(skip)
