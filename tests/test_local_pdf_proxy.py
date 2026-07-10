import json
import tempfile
import unittest
from pathlib import Path

from cascade_planner.harness.local_pdf_proxy import (
    build_pdf_request,
    download_pdf_requests,
    local_pdf_proxy_manifest_entry,
    requests_from_source_material_locator_pack,
    write_pdf_request_queue,
)


class LocalPdfProxyTest(unittest.TestCase):
    def test_manifest_entry_exposes_queue_without_credentials(self):
        with tempfile.TemporaryDirectory() as tmp:
            entry = local_pdf_proxy_manifest_entry(None, output_dir=tmp)

        self.assertEqual(entry["schema_version"], "local_pdf_proxy_manifest_entry.v1")
        self.assertEqual(entry["status"], "planned")
        self.assertIn("pdf_requests.jsonl", entry["request_queue_path"])
        self.assertFalse(entry["source_policy"]["credentials_stored"])
        self.assertFalse(entry["source_policy"]["cookies_stored"])

    def test_source_material_locator_pack_becomes_deduped_requests(self):
        pack = {
            "target": {"name": "ethanol"},
            "material_records": [
                {
                    "record_id": "doi:10.0000/example:publisher_landing",
                    "doi": "10.0000/example",
                    "title": "Synthesis of ethanol",
                    "url": "https://doi.org/10.0000/example",
                    "material_type": "publisher_landing",
                    "evidence_refs": ["ev_ethanol"],
                },
                {
                    "record_id": "doi:10.0000/example:publisher_landing",
                    "doi": "10.0000/example",
                    "title": "Synthesis of ethanol",
                    "url": "https://doi.org/10.0000/example",
                    "material_type": "publisher_landing",
                    "evidence_refs": ["ev_ethanol"],
                },
            ],
        }

        requests = requests_from_source_material_locator_pack(pack)

        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0]["case_id"], "ethanol")
        self.assertEqual(requests[0]["doi"], "10.0000/example")
        self.assertEqual(requests[0]["evidence_refs"], ["ev_ethanol"])
        self.assertTrue(requests[0]["source_policy"]["metadata_pointer_only"])

    def test_source_material_locator_requests_prioritize_target_terms(self):
        pack = {
            "target": {"name": "bufotalin_v0_fullflow_20260606"},
            "material_records": [
                {
                    "record_id": "doi:10.0000/other",
                    "doi": "10.0000/other",
                    "title": "Synthesis of a key intermediate for another target",
                    "url": "https://doi.org/10.0000/other",
                    "material_type": "publisher_landing",
                },
                {
                    "record_id": "doi:10.0000/bufotalin",
                    "doi": "10.0000/bufotalin",
                    "title": "Construction of advanced intermediate for the synthesis of bufotalin",
                    "url": "https://doi.org/10.0000/bufotalin",
                    "material_type": "publisher_landing",
                },
            ],
        }

        requests = requests_from_source_material_locator_pack(pack)

        self.assertEqual(requests[0]["doi"], "10.0000/bufotalin")

    def test_pdf_request_preserves_content_scope(self):
        request = build_pdf_request(
            {
                "doi": "10.0000/example",
                "url": "https://doi.org/10.0000/example",
                "content_scope": "si",
            },
            case_id="case",
        )

        self.assertEqual(request["content_scope"], "si")

    def test_download_worker_writes_pdf_and_manifest(self):
        def fake_fetch(url, headers, timeout_s, max_bytes):
            del headers, timeout_s, max_bytes
            return {
                "status": 200,
                "final_url": url + "/paper.pdf",
                "headers": {"content-type": "application/pdf"},
                "body": b"%PDF-1.4\nexample\n",
            }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            queue = root / "requests.jsonl"
            pdf_dir = root / "pdfs"
            manifest = root / "manifest.jsonl"
            request = build_pdf_request({"doi": "10.0000/example"}, case_id="case")
            write_pdf_request_queue([request], queue, append=False)

            result = download_pdf_requests(
                queue_path=queue,
                pdf_dir=pdf_dir,
                manifest_path=manifest,
                fetch_url=fake_fetch,
                delay_s=0,
            )
            rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
            pdf_exists = Path(rows[0]["pdf_path"]).exists()

        self.assertEqual(result["downloaded_count"], 1)
        self.assertEqual(rows[0]["status"], "downloaded")
        self.assertTrue(pdf_exists)
        self.assertFalse(rows[0]["source_policy"]["credentials_stored"])

    def test_html_login_response_is_manual_access_gap(self):
        def fake_fetch(url, headers, timeout_s, max_bytes):
            del url, headers, timeout_s, max_bytes
            return {
                "status": 200,
                "final_url": "https://publisher.example/login",
                "headers": {"content-type": "text/html"},
                "body": b"<html>institution login required</html>",
            }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            queue = root / "requests.jsonl"
            manifest = root / "manifest.jsonl"
            request = build_pdf_request({"url": "https://publisher.example/article"}, case_id="case")
            write_pdf_request_queue([request], queue, append=False)

            result = download_pdf_requests(
                queue_path=queue,
                pdf_dir=root / "pdfs",
                manifest_path=manifest,
                fetch_url=fake_fetch,
                delay_s=0,
            )
            row = json.loads(manifest.read_text(encoding="utf-8").strip())

        self.assertEqual(result["needs_manual_access_count"], 1)
        self.assertEqual(row["status"], "needs_manual_access")
        self.assertFalse(row["accepted"])

    def test_pdf_content_type_with_html_body_is_not_downloaded(self):
        def fake_fetch(url, headers, timeout_s, max_bytes):
            del url, headers, timeout_s, max_bytes
            return {
                "status": 200,
                "final_url": "https://publisher.example/viewer",
                "headers": {"content-type": "application/pdf"},
                "body": b"<!doctype html><html>PDF viewer shell</html>",
            }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            queue = root / "requests.jsonl"
            manifest = root / "manifest.jsonl"
            request = build_pdf_request({"url": "https://publisher.example/article"}, case_id="case")
            write_pdf_request_queue([request], queue, append=False)

            result = download_pdf_requests(
                queue_path=queue,
                pdf_dir=root / "pdfs",
                manifest_path=manifest,
                fetch_url=fake_fetch,
                delay_s=0,
            )
            row = json.loads(manifest.read_text(encoding="utf-8").strip())

        self.assertEqual(result["downloaded_count"], 0)
        self.assertEqual(row["status"], "failed")
        self.assertFalse(row["accepted"])


if __name__ == "__main__":
    unittest.main()
