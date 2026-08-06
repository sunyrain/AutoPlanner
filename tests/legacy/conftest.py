from __future__ import annotations

from pathlib import Path

import pytest


LEGACY_TEST_ROOT = Path(__file__).resolve().parent


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if item.path.is_relative_to(LEGACY_TEST_ROOT):
            item.add_marker(pytest.mark.legacy)
