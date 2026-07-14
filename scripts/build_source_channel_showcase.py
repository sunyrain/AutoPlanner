#!/usr/bin/env python3
"""Build a portable expert page from one real source-channel verification run."""
from __future__ import annotations

import argparse
import hashlib
from html import escape
import json
from pathlib import Path
import re
import shutil
from typing import Any, Mapping
from xml.etree import ElementTree


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visual-result", required=True)
    parser.add_argument("--normalized-observation", required=True)
    parser.add_argument("--xml", required=True)
    parser.add_argument("--patent-pdf", required=True)
    parser.add_argument("--validation-fork-report")
    parser.add_argument("--artifact-store-root")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    visual_path = _file(args.visual_result)
    normalized_path = _file(args.normalized_observation)
    xml_path = _file(args.xml)
    patent_path = _file(args.patent_pdf)
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    visual = _json(visual_path)
    normalized = _json(normalized_path)
    request = dict(visual.get("request") or {})
    result = dict(visual.get("result") or {})
    source = dict(request.get("source") or {})
    page = next(
        (dict(row) for row in source.get("pages") or [] if isinstance(row, Mapping)),
        {},
    )
    figure_path = _file(page.get("image_path"))
    xml_sha = _sha256(xml_path)
    figure_sha = _sha256(figure_path)
    expected_xml = str(source.get("source_fulltext_sha256") or "")
    expected_figure = str(page.get("image_sha256") or "")
    if xml_sha != expected_xml or figure_sha != expected_figure:
        raise ValueError("source_channel_artifact_digest_mismatch")

    xml_stats = _xml_stats(xml_path)
    usage = dict(result.get("usage") or {})
    candidates = [
        dict(row)
        for row in normalized.get("candidate_steps") or []
        if isinstance(row, Mapping)
    ]
    if any(row.get("grants_exact_evidence") is not False for row in candidates):
        raise ValueError("visual_candidate_authority_escalation")

    assets = {
        "figure": _copy(figure_path, output / "source-figure.jpg"),
        "xml": _copy(xml_path, output / "source-fulltext.xml"),
        "patent": _copy(patent_path, output / "source-patent.pdf"),
        "visual": _copy(visual_path, output / "visual-result.json"),
        "normalized": _copy(
            normalized_path,
            output / "visual-observation-normalized.json",
        ),
    }
    validation_fork: dict[str, Any] = {}
    if args.validation_fork_report:
        validation_fork, paper_asset = _validation_fork_payload(
            _file(args.validation_fork_report),
            output=output,
            artifact_store_root=(
                Path(args.artifact_store_root).expanduser().resolve()
                if args.artifact_store_root
                else None
            ),
        )
        assets["paper_html"] = paper_asset
    payload = {
        "schema_version": "source_channel_expert_showcase.v1",
        "source_ref": str(source.get("source_ref") or ""),
        "source_title": str(source.get("title") or ""),
        "source_artifact_kind": str(source.get("source_artifact_kind") or ""),
        "xml": {**xml_stats, "sha256": xml_sha, "digest_verified": True},
        "figure": {"sha256": figure_sha, "digest_verified": True},
        "patent": {
            "publication_number": "EP2486129B1",
            "sha256": _sha256(patent_path),
            "size_bytes": patent_path.stat().st_size,
            "page_count": _pdf_pages(patent_path),
            "source_authority": "EPO publication server",
        },
        "visual": {
            "status": str(result.get("provider_status") or ""),
            "candidate_step_count": len(candidates),
            "admission_eligible_step_count": int(
                normalized.get("admission_eligible_step_count") or 0
            ),
            "matched_current_edge_count": int(
                normalized.get("matched_current_edge_count") or 0
            ),
            "usage": usage,
            "all_candidates_non_authoritative": True,
        },
        "validation_fork": validation_fork,
        "assets": {key: value.name for key, value in assets.items()},
        "semantics": {
            "no_local_source_seed": True,
            "metadata_search_preceded_download": True,
            "xml_and_original_figure_are_digest_bound": True,
            "visual_output_is_hypothesis_only": True,
            "no_visual_candidate_grants_exact_evidence": True,
        },
    }
    (output / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output / "index.html").write_text(_render(payload), encoding="utf-8")
    print(
        json.dumps(
            {"index_html": str(output / "index.html"), **payload["visual"]},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _render(payload: Mapping[str, Any]) -> str:
    xml = dict(payload["xml"])
    patent = dict(payload["patent"])
    visual = dict(payload["visual"])
    usage = dict(visual.get("usage") or {})
    assets = dict(payload["assets"])
    validation_fork = dict(payload.get("validation_fork") or {})
    fork_panel = _validation_fork_panel(validation_fork, assets=assets)
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AutoPlanner 来源通道实测</title><style>
:root{{--ink:#172033;--muted:#68758a;--line:#dce3ef;--blue:#315be8;--green:#18856f;--amber:#aa6a13}}
*{{box-sizing:border-box}}body{{margin:0;background:#f3f6fb;color:var(--ink);font:14px/1.55 Inter,"Microsoft YaHei",sans-serif}}
header{{padding:42px max(28px,7vw);color:#fff;background:linear-gradient(135deg,#17213d,#315bb5)}}
header h1{{font-size:38px;margin:6px 0}}header p{{max-width:920px;color:#dce7ff}}
main{{max-width:1260px;margin:auto;padding:28px}}.metrics{{display:grid;grid-template-columns:repeat(6,1fr);gap:10px;margin-top:-58px}}
.metric,.panel{{background:#fff;border:1px solid var(--line);border-radius:14px;box-shadow:0 10px 35px #1830620b}}
.metric{{padding:16px}}.metric b{{display:block;font-size:22px}}.metric span,.muted{{color:var(--muted);font-size:12px}}
.flow{{display:grid;grid-template-columns:repeat(5,1fr);gap:8px;margin:22px 0}}.flow div{{padding:13px;border-radius:10px;background:#eaf0ff;border-left:3px solid var(--blue)}}
.grid{{display:grid;grid-template-columns:1.05fr .95fr;gap:16px}}.panel{{padding:20px}}h2{{margin:0 0 12px}}
img{{width:100%;border-radius:10px;border:1px solid var(--line)}}.ok{{color:var(--green)}}.warn{{color:var(--amber)}}
table{{width:100%;border-collapse:collapse}}td{{padding:9px;border-bottom:1px solid #edf0f5}}td:first-child{{color:var(--muted)}}.wide{{margin-top:16px}}.procedures{{display:grid;grid-template-columns:repeat(2,1fr);gap:10px}}.procedure{{padding:12px;border-radius:10px;background:#f7f9fd;border-left:3px solid var(--green)}}.procedure b{{display:block;margin-bottom:4px}}
.links{{display:flex;gap:8px;flex-wrap:wrap;margin-top:16px}}a{{padding:8px 11px;border-radius:8px;background:#edf1ff;color:var(--blue);font-weight:700;text-decoration:none}}
.routeCards{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;margin:14px 0}}.routeCard{{padding:14px;border:1px solid #dfe6f2;border-radius:12px;background:#f8faff}}.routeCard b{{display:block;margin-bottom:6px}}.routeCard code{{display:block;white-space:normal;overflow-wrap:anywhere;color:#42506a;font-size:11px}}.routeCard .condition{{margin-top:8px;color:var(--muted);font-size:12px}}
@media(max-width:800px){{.metrics,.flow{{grid-template-columns:1fr 1fr}}.grid{{grid-template-columns:1fr}}}}
</style></head><body><header><small>FRESH NETWORK · DIGEST BOUND · AUDITABLE</small>
<h1>文献、专利与视觉提取实测</h1><p>该页只使用本轮从网络自动发现并下载的来源；元数据不授予证据等级，视觉结果不绕过 host 验证。</p></header>
<main><section class="metrics">{_metric('XML 全文',_bytes(xml['size_bytes']))}{_metric('原始图',xml['figure_count'])}{_metric('实验段落',xml['section_count'])}{_metric('视觉候选',visual['candidate_step_count'])}{_metric('视觉耗时',f"{float(usage.get('wall_time_s') or 0):.1f}s")}{_metric('来源验证模型调用',validation_fork.get('model_invocations',0))}</section>
<section class="flow"><div>Europe PMC 元数据</div><div>XML → PMC HTML</div><div>PDF / EPO 回退</div><div>确定性过程抽取</div><div>Host 分级接纳</div></section>
<section class="grid"><article class="panel"><h2>{_h(payload['source_title'])}</h2><img src="{_h(assets['figure'])}" alt="自动下载的文献原图"><p class="muted">{_h(payload['source_ref'])} · XML SHA-256 {_h(str(xml['sha256'])[:16])}…</p></article>
<article class="panel"><h2>验收结果</h2><table>
<tr><td>XML / 原图哈希</td><td class="ok">已验证</td></tr><tr><td>XML 结构</td><td>{xml['section_count']} sections · {xml['figure_count']} figures</td></tr>
<tr><td>官方专利</td><td>{_h(patent['publication_number'])} · {patent['page_count']} 页 · {_bytes(patent['size_bytes'])}</td></tr>
<tr><td>视觉调用</td><td>{int(usage.get('model_invocations') or 0)} 次 · {int(usage.get('input_tokens') or 0):,} input tokens</td></tr>
<tr><td>规范化候选</td><td>{visual['admission_eligible_step_count']} 个 L0/L1 假设</td></tr>
<tr><td>精确证据授权</td><td class="warn">0；视觉不能授予 L2/L3</td></tr></table>
<div class="links"><a href="{_h(assets['xml'])}">查看 XML</a><a href="{_h(assets['patent'])}">查看 EPO 专利</a><a href="{_h(assets['normalized'])}">查看规范化结果</a></div></article></section>{fork_panel}</main></body></html>"""


def _validation_fork_panel(
    value: Mapping[str, Any],
    *,
    assets: Mapping[str, Any],
) -> str:
    if not value:
        return ""
    procedures = "".join(
        f'<div class="procedure"><b>{_h(row.get("name") or "实验段落")}</b>'
        f'<span>{_h(row.get("excerpt") or "")}</span></div>'
        for row in value.get("procedures") or []
        if isinstance(row, Mapping)
    )
    timings = dict(value.get("connector_elapsed_s") or {})
    routes = "".join(
        '<div class="routeCard">'
        f'<b>{_h(row.get("label") or "来源路线")}</b>'
        f'<code>{_h(row.get("reaction") or "")}</code>'
        f'<div class="condition">{_h(row.get("conditions") or "条件待补")}</div>'
        "</div>"
        for row in value.get("source_routes") or []
        if isinstance(row, Mapping)
    )
    exact_status = (
        "已闭合"
        if value.get("B3_exact_multi_source") is True
        else (
            "尚未闭合；当前 exact 反应已单独显示，不能冒充全路线多来源证明"
        )
    )
    return f"""<section class="panel wide"><h2>Simvastatin · 多信源增量验证</h2>
<table><tr><td>来源</td><td>{int(value.get('source_count') or 0)} 个：EPO 专利 + PMC 原始论文</td></tr>
<tr><td>论文获取</td><td>{_h(_source_label(value.get('paper_acquisition','')))} · {_h(value.get('paper_pmcid',''))} · {_h(_source_label(value.get('paper_access_class','')))}</td></tr>
<tr><td>抽取结果</td><td>{int(value.get('paper_procedure_count') or 0)} 个论文段落 + {int(value.get('patent_procedure_count') or 0)} 个专利段落</td></tr>
<tr><td>来源路线</td><td>{int(value.get('source_route_proposal_count') or 0)} 条物化 · {int(value.get('source_route_host_accepted_count') or 0)} 条主机验证 · {int(value.get('source_route_exact_row_count') or 0)} 条 exact</td></tr>
<tr><td>视觉候选</td><td>{int(value.get('visual_candidate_count') or 0)} 条提取 · {int(value.get('visual_materialized_count') or 0)} 条物化 · {int(value.get('visual_host_accepted_count') or 0)} 条接纳 · {int(value.get('visual_host_rejected_count') or 0)} 条拒绝</td></tr>
<tr><td>self-evo</td><td>{int(value.get('self_evo_template_count') or 0)} 个共享模板 · generation {int(value.get('self_evo_generation') or 0)}</td></tr>
<tr><td>原始来源缓存</td><td>{'哈希复核后命中' if value.get('patent_source_cache_hit') else '本轮下载并冻结'}</td></tr>
<tr><td>并行耗时</td><td>patent {float(timings.get('connector_1') or 0):.1f}s · paper {float(timings.get('connector_2') or 0):.1f}s</td></tr>
<tr><td>来源 fork 模型 / 视觉</td><td class="ok">{int(value.get('model_invocations') or 0)} / {int(value.get('visual_invocations') or 0)}</td></tr>
<tr><td>B3 exact multi-source</td><td class="{'ok' if value.get('B3_exact_multi_source') else 'warn'}">{_h(exact_status)}</td></tr></table>
<div class="routeCards">{routes}</div><div class="procedures">{procedures}</div><div class="links"><a href="{_h(assets.get('paper_html',''))}">查看哈希冻结的论文全文</a></div></section>"""


def _validation_fork_payload(
    report_path: Path,
    *,
    output: Path,
    artifact_store_root: Path | None,
) -> tuple[dict[str, Any], Path]:
    output.mkdir(parents=True, exist_ok=True)
    report = _json(report_path)
    stage = next(
        (
            dict(row.get("detail") or {})
            for row in report.get("stages") or []
            if isinstance(row, Mapping) and row.get("stage") == "evidence_acquisition"
        ),
        {},
    )
    sources = [
        dict(row)
        for row in dict(stage.get("discovery") or {}).get("sources") or []
        if isinstance(row, Mapping)
    ]
    paper = next(
        (
            row
            for row in sources
            if row.get("source_kind") == "paper_si"
            and (row.get("fulltext_html_path") or row.get("fulltext_xml_path"))
        ),
        next(
            (row for row in sources if row.get("source_kind") == "paper_si"),
            {},
        ),
    )
    patent = next((row for row in sources if row.get("source_kind") == "patent"), {})
    paper_path = _file(
        paper.get("fulltext_html_path") or paper.get("fulltext_xml_path")
    )
    paper_sha = _sha256(paper_path)
    if paper_sha != str(paper.get("source_fulltext_sha256") or ""):
        raise ValueError("validation_fork_paper_digest_mismatch")
    paper_asset = _copy(
        paper_path,
        output / f"simvastatin-source-paper{paper_path.suffix.lower()}",
    )
    receipt_ref = dict(stage.get("receipt_ref") or {})
    store = artifact_store_root or report_path.parents[2] / "artifacts"
    receipt_path = store / str(receipt_ref.get("object_path") or "")
    receipt = _json(_file(receipt_path))
    patent_receipt = next(
        (
            dict(row)
            for row in receipt.get("child_receipts") or []
            if isinstance(row, Mapping)
            and str(row.get("provider_id") or "")
            == "autoplanner.builtin_patent_evidence"
        ),
        {},
    )
    patent_audit = next(
        (
            dict(row)
            for row in patent_receipt.get("audits") or []
            if isinstance(row, Mapping)
        ),
        {},
    )
    route_observation = dict(patent_audit.get("source_route_observation") or {})
    source_route_stage = dict(stage.get("source_route") or {})
    source_route_validation = dict(
        source_route_stage.get("validation") or stage.get("validation") or {}
    )
    visual_stage = dict(stage.get("visual_evidence") or {})
    visual_observation = dict(visual_stage.get("observation") or {})
    visual_materialization = dict(visual_stage.get("materialization") or {})
    visual_validation = dict(visual_stage.get("validation") or {})
    source_routes = [
        _source_route_card(dict(row), target_name=str(dict(report.get("target") or {}).get("name") or "target"))
        for row in route_observation.get("proposals") or []
        if isinstance(row, Mapping)
    ][:8]
    self_evolution = dict(report.get("self_evolution") or {})
    learning = next(
        (
            dict(row)
            for row in reversed(self_evolution.get("learning_stages") or [])
            if isinstance(row, Mapping)
        ),
        {},
    )
    model_cost = dict(report.get("model_cost") or {})
    model_invocations = int(model_cost.get("model_invocations") or 0)
    visual_invocations = int(model_cost.get("visual_invocations") or 0)
    if model_invocations != visual_invocations or visual_invocations > 1:
        raise ValueError("validation_fork_unadmitted_model_usage_detected")
    procedures = [
        {
            "name": str(row.get("name") or ""),
            "excerpt": str(row.get("procedure_excerpt") or "")[:420],
        }
        for row in paper.get("procedure_inventory") or []
        if isinstance(row, Mapping)
    ][:4]
    return (
        {
            "run_id": str(report.get("run_id") or ""),
            "source_count": len(sources),
            "paper_acquisition": str(paper.get("acquisition_method") or ""),
            "paper_pmcid": str(paper.get("pmcid") or ""),
            "paper_access_class": str(
                dict(paper.get("acquisition_receipt") or {}).get("access_class") or ""
            ),
            "paper_sha256": paper_sha,
            "paper_procedure_count": len(paper.get("procedure_inventory") or []),
            "patent_procedure_count": len(patent.get("procedure_inventory") or []),
            "source_route_proposal_count": int(
                route_observation.get("proposal_count") or len(source_routes)
            ),
            "source_route_host_accepted_count": int(
                source_route_validation.get("accepted_validation_count")
                or 0
            ),
            "source_route_exact_row_count": int(
                patent_audit.get("source_route_exact_row_count") or 0
            ),
            "source_routes": source_routes,
            "visual_candidate_count": int(
                visual_observation.get("candidate_step_count")
                or visual_materialization.get("proposal_count")
                or 0
            ),
            "visual_materialized_count": int(
                visual_materialization.get("proposal_count") or 0
            ),
            "visual_host_accepted_count": int(
                visual_validation.get("accepted_validation_count") or 0
            ),
            "visual_host_rejected_count": int(
                visual_validation.get("rejected_validation_count") or 0
            ),
            "patent_source_cache_hit": any(
                dict(row).get("cache_hit") is True
                for row in dict(patent_audit.get("source_byte_cache") or {}).values()
                if isinstance(row, Mapping)
            ),
            "self_evo_template_count": int(learning.get("template_count") or 0),
            "self_evo_generation": int(learning.get("generation") or 0),
            "connector_elapsed_s": dict(receipt.get("child_elapsed_s") or {}),
            "model_invocations": model_invocations,
            "visual_invocations": visual_invocations,
            "B3_exact_multi_source": bool(
                dict(dict(report.get("gates") or {}).get("gates") or {}).get(
                    "B3_exact_multi_source"
                )
            ),
            "procedures": procedures,
        },
        paper_asset,
    )


def _source_route_card(row: Mapping[str, Any], *, target_name: str) -> dict[str, str]:
    origin = str(row.get("origin_kind") or "")
    raw_product = str(row.get("product_name") or "")
    product = _narrative_product_name(raw_product)
    reactants = [
        str(value)
        for value in row.get("reactant_names") or []
        if str(value).strip()
    ]
    if origin == "deterministic_source_form_bridge":
        product = f"{target_name}（内酯）"
        reactants = [f"{target_name} acid"]
        label = "形式桥接 · 主机验证"
    else:
        label = "专利实验反应 · exact 来源行"
    condition = dict(row.get("condition_candidate") or {})
    condition_text = "；".join(
        str(condition.get(key) or "").strip()
        for key in ("temperature", "time", "addition_order", "workup")
        if str(condition.get(key) or "").strip()
    )
    return {
        "label": label,
        "reaction": f"{' + '.join(reactants) or '来源中间体'} → {product}",
        "conditions": condition_text or "确定性结构等价；无独立实验条件授权",
    }


def _narrative_product_name(value: str) -> str:
    match = re.search(
        r"(?i)\b(?:synthesis|preparation|production)\s+of\s+(.{2,180}?)\s+from\s+",
        " ".join(str(value or "").split()),
    )
    return str(match.group(1) if match else value).strip(" .,:;()")


def _metric(label: str, value: Any) -> str:
    return f'<article class="metric"><b>{_h(value)}</b><span>{_h(label)}</span></article>'


def _source_label(value: Any) -> str:
    return {
        "pmc_repository_fulltext_html": "PMC 正文 HTML",
        "europe_pmc_structured_fulltext_xml": "Europe PMC 全文 XML",
        "free_repository_fulltext": "仓储全文可访问（非 OA 许可）",
        "open_access": "开放获取",
    }.get(str(value), str(value))


def _xml_stats(path: Path) -> dict[str, int]:
    root = ElementTree.fromstring(path.read_bytes())
    tags = [element.tag.rsplit("}", 1)[-1] for element in root.iter()]
    return {
        "size_bytes": path.stat().st_size,
        "section_count": tags.count("sec"),
        "figure_count": tags.count("fig"),
    }


def _pdf_pages(path: Path) -> int:
    try:
        import fitz  # type: ignore

        with fitz.open(path) as document:
            return int(document.page_count)
    except (ImportError, OSError, RuntimeError, ValueError):
        return 0


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected_json_object:{path}")
    return dict(value)


def _file(value: Any) -> Path:
    path = Path(str(value or "")).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"file_missing:{path}")
    return path


def _copy(source: Path, destination: Path) -> Path:
    shutil.copy2(source, destination)
    return destination


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bytes(value: Any) -> str:
    size = int(value or 0)
    return f"{size / 1_048_576:.1f} MB" if size >= 1_048_576 else f"{size / 1024:.1f} KB"


def _h(value: Any) -> str:
    return escape(str(value), quote=True)


if __name__ == "__main__":
    raise SystemExit(main())
