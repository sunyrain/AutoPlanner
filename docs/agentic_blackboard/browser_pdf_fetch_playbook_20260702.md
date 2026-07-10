# Browser PDF Fetch Playbook, 2026-07-02

## What Worked

- Use a normal visible Chrome process and attach over CDP.
- Warm the publisher article page first so institutional access cookies and ACS session state are present.
- Fetch PDF URLs inside the article page with `fetch(url, { credentials: "include", cache: "no-store" })`.
- Save the returned bytes only when the payload starts with `%PDF`.
- Treat Chrome's built-in PDF viewer as success, not as a missing download event.

## What Failed

- Waiting for Playwright `download` events is unreliable for ACS, because Chrome opens article PDFs in the built-in PDF viewer.
- Browser contexts launched directly by automation are more likely to hit ACS Cloudflare verification on PDF/SI endpoints.
- Nature `https://www.nature.com/articles/367630a0.pdf` currently returns the article HTML page for the tested Taxol source, not a PDF payload.
- Links such as `#_suppInfo`, `#_i5`, and scite report pages are not PDF files and should be filtered before fetch attempts.

## Verified Downloads

The retry run saved real PDF payloads under:

`results/shared/large_lit_pdf_panel_20260702/pdf_retry_browser_fetch_20260702/pdfs`

Successful source coverage:

- Paclitaxel: Holton JACS article PDF and SI PDFs; Danishefsky JACS article PDF.
- Artemisinin: Schmid/Hofheinz JACS article PDF and SI; Zhu/Cook JACS article PDF and SI; Turconi OPRD article PDF.
- Erythromycin A: JACS total synthesis article PDF and SI.

## Rerun Strategy

- Pass the downloaded article PDFs as local literature cache entries with explicit DOI/source metadata.
- Keep local PDFs as evidence cache, not blind route proof.
- Use higher but bounded budgets: 5 rounds, 2 Codex research/scout runs, 3 visual/PDF extraction calls, and 10 action-planner tool calls.
- Emit blackboard steps for every run so planner decisions can be compared round by round.
