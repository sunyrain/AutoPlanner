from __future__ import annotations

import pytest

from scripts.legacy.serve_combined_web import build_parser


pytestmark = pytest.mark.legacy


def test_combined_web_is_available_only_through_explicit_legacy_launcher() -> None:
    args = build_parser().parse_args([])

    assert args.host == "127.0.0.1"
    assert args.port == 7860
    assert args.server == "auto"
