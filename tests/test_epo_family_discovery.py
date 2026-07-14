from __future__ import annotations

from typing import Any

from cascade_planner.interfaces.epo_family_discovery import (
    epo_family_pdf_candidates,
)


class _Response:
    status_code = 200

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload
        self.content = b"{}"

    def json(self) -> dict[str, Any]:
        return self._payload


def _binding(uri: str, date: str, title: str, root_title: str) -> dict[str, Any]:
    return {
        "object": {"value": uri},
        "date": {"value": date},
        "title": {"value": title},
        "rootTitle": {"value": root_title},
    }


def test_wo_family_resolves_to_official_ep_pdf() -> None:
    payload = {
        "results": {
            "bindings": [
                _binding(
                    "http://data.epo.org/linked-data/data/publication/EP/2552906/A1/-",
                    "2013-02-06",
                    "CGRP RECEPTOR ANTAGONIST",
                    "CGRP RECEPTOR ANTAGONIST",
                ),
                _binding(
                    "http://data.epo.org/linked-data/data/publication/EP/2552906/B1/-",
                    "2016-01-06",
                    "CGRP RECEPTOR ANTAGONIST",
                    "CGRP RECEPTOR ANTAGONIST",
                ),
            ]
        }
    }
    calls: list[dict[str, Any]] = []

    def requester(_url: str, **kwargs: Any) -> _Response:
        calls.append(kwargs)
        return _Response(payload)

    rows = epo_family_pdf_candidates(
        "WO-2011123232-A1",
        timeout_s=12,
        max_response_bytes=100_000,
        requester=requester,
    )

    assert [row["publication_number"] for row in rows] == [
        "EP2552906B1",
        "EP2552906A1",
    ]
    assert rows[0]["pdf_url"].endswith(
        "/20160106/patents/EP2552906NWB1/document.pdf"
    )
    assert rows[0]["family_id"] == "epo-family:WO-2011123232-A1"
    assert "WO/2011123232/A1/-" in calls[0]["params"]["query"]


def test_non_wo_publication_does_not_query_epo_family_graph() -> None:
    assert (
        epo_family_pdf_candidates(
            "US8481546B2",
            timeout_s=12,
            max_response_bytes=100_000,
            requester=lambda *_args, **_kwargs: None,
        )
        == []
    )
