"""Readable bilingual supplement to the native AutoPlanner graph export."""
import html


def details_page(original, localized, title):
    def prose(value):
        rows = value if isinstance(value, list) else [value]
        return ''.join(f'<p>{html.escape(str(row))}</p>' for row in rows if row)

    labels = {'query': '策略设计', 'critical_assumption': '关键假设', 'critic_checkpoint': '审查重点',
              'transformation_rationale': '这一步的设计依据', 'conditions': '反应条件与后处理',
              'catalyst': '催化剂', 'enzyme': '酶', 'builder_limitations': '构建者披露的限制',
              'critic_reasons': '审查意见', 'critic_condition_assessment': '条件评估',
              'critic_suggested_revision': '建议修订', 'route_overall_evaluation': '路线整体评估',
              'route_level_risks': '路线层面风险', 'decisive_risk': '主要风险'}

    def fields(zh, en, keys):
        return ''.join(f'<section class="text-field"><h4>{labels[key]}</h4>{prose(zh[key])}'
                       f'<details class="original"><summary>英文原文</summary>{prose(en.get(key))}</details></section>'
                       for key in keys if zh.get(key))

    strategies = {s['index']: s for s in localized['projection']['strategies']}
    original_strategies = {s['index']: s for s in original['projection']['strategies']}
    original_branches = {b['branch_index']: b for b in original['projection']['branches']}
    sections, links = [], []
    strategy_keys = ('query', 'critical_assumption', 'critic_checkpoint')
    step_keys = tuple(key for key in labels if key not in strategy_keys)
    for branch in localized['projection']['branches']:
        index = branch['branch_index']
        links.append(f'<a href="#strategy-{index}">策略 {index}</a>')
        strategy, en_strategy = strategies.get(index, {}), original_strategies.get(index, {})
        content = fields(strategy, en_strategy, strategy_keys)
        refreshes = strategy.get('strategy_refreshes', [])
        en_refreshes = en_strategy.get('strategy_refreshes', [])
        if refreshes:
            content += '<details class="revisions"><summary>展开后续策略调整与对应假设</summary>'
            for revision, en_revision in zip(refreshes, en_refreshes, strict=True):
                content += f'<h3>里程碑 {html.escape(str(revision.get("milestone_index", "")))}</h3>'
                content += fields(revision, en_revision, strategy_keys)
            content += '</details>'
        content += fields(branch, original_branches[index], step_keys)
        for number, (step, en_step) in enumerate(zip(branch['steps'], original_branches[index]['steps'], strict=True), 1):
            name = step.get('reaction_family') or step.get('transformation_rationale') or '未记录反应名称'
            verdict = step.get('critic_verdict') or '未记录'
            content += (f'<details class="step" id="s{index}-r{number}"><summary><span>记录 {number:02d}</span>'
                        f'{html.escape(name)}</summary><div class="step-body"><small>审查判定（原始值）：'
                        f'{html.escape(str(verdict))} · {html.escape(str(step.get("step_id", "")))}</small>'
                        f'<details class="original"><summary>英文反应名称</summary>{prose(en_step.get("reaction_family"))}</details>'
                        + fields(step, en_step, step_keys) + '</div></details>')
        sections.append(f'<article id="strategy-{index}"><header><small>STRATEGY {index:02d} · '
                        f'{len(branch["steps"])} 条反应记录</small><h2>策略 {index}</h2></header>{content}</article>')
    return ('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{html.escape(title)} · 策略与条件</title><style>'
            '*{box-sizing:border-box}body{margin:0;background:#f5f7f5;color:#19392b;font:17px/1.8 system-ui,"Microsoft YaHei",sans-serif}'
            'main{max-width:1120px;margin:auto;padding:32px 24px 64px}h1{font-size:30px;line-height:1.4}h2{font-size:26px;margin:4px 0 16px}'
            'h3{font-size:21px}h4{font-size:16px;color:#456a56;margin:0 0 8px}p{margin:0 0 12px;overflow-wrap:anywhere}'
            'a{color:#17634a;text-underline-offset:4px}nav{display:flex;flex-wrap:wrap;gap:12px 24px;margin:20px 0}'
            'article{background:white;border:1px solid #d9e4db;border-radius:16px;padding:28px;margin:24px 0;scroll-margin-top:20px}'
            'small{font-size:13px;color:#5d7064;overflow-wrap:anywhere}.text-field{margin:20px 0}.original{font-size:15px;color:#657569}'
            'summary{cursor:pointer;overflow-wrap:anywhere}.original summary{font-size:13px;color:#527662}.original p{margin-top:12px}'
            '.revisions{background:#f5f8f4;padding:16px 20px;border-radius:10px;margin-bottom:24px}'
            '.step{border-top:1px solid #dce5dc;padding:18px 0}.step>summary{font-size:18px;font-weight:600}'
            '.step>summary>span{font-size:13px;display:inline-block;margin-right:16px;color:#547563;min-width:64px}'
            '.step-body{padding:16px 0 0}.notice{color:#617466;font-size:15px;max-width:920px}'
            '@media(max-width:650px){main{padding:24px 14px}article{padding:20px 16px}h1{font-size:24px}}'
            '@media print{body{background:white}main{max-width:none}article{break-before:page;border:0}nav{display:none}}'
            '</style></head><body><main><small>AUTOPLANNER · 中文专家交流资料</small>'
            f'<h1>{html.escape(title)}</h1><p class="notice">展示译本，不更改分子结构、路线拓扑或审查判定。'
            '筛选假设仍是待验证假设；反应记录包括原始与修订备选，不等同于单一路径的线性步数。点击记录展开全部条件；英文原文可逐项对照。</p>'
            '<nav><a href="../../index.html">全部任务</a><a href="graph.zh.html">中文路线画布</a><a href="route.zh.html">结构与条件总览</a></nav>'
            '<nav>' + ''.join(links) + '</nav>' + ''.join(sections) + '</main></body></html>')
