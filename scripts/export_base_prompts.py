"""Offline documentation export. Never dispatch a worker or query a provider."""
from __future__ import annotations

import ast
from dataclasses import replace
import inspect
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cascade_planner.orchestration import sequential_strategy_director as d
from cascade_planner.orchestration import chemical_reasoning as cr
from cascade_planner.orchestration import reaction_granularity as rg
from cascade_planner.orchestration import repair_recovery as rr
from cascade_planner.application import material_boundary as mb
from cascade_planner.application import planning_evidence as pe
from cascade_planner.agent import codex_worker as w

OUT = ROOT / 'docs/architecture'
SOURCE = ROOT / 'cascade_planner/orchestration/sequential_strategy_director.py'
SOURCE_TEXT = SOURCE.read_text(encoding='utf-8')
TREE = ast.parse(SOURCE_TEXT)
CARD = {k: '<' + k + '>' for k in ('strategy_query', 'critical_assumption', 'critic_checkpoint')}
TARGET = 'CCO'
MARKERS = []
RENDERED = []


def link(obj):
    filename = Path(inspect.getsourcefile(obj))
    line = inspect.getsourcelines(obj)[1]
    relative = filename.relative_to(ROOT).as_posix()
    return f'[{relative}:{line}](../../{relative}#L{line})'


def block(value, language='text'):
    return f'```{language}\n{value.rstrip()}\n```\n'


def separate(prompt):
    lines = prompt.splitlines()
    context = json.loads(lines[-1])
    marker = lines[-2]
    return '\n'.join(lines[:-2]), marker, context


def add_prompt(parts, ident, title, prompt, obj, note, extra_keys=()):
    body, marker, context = separate(prompt)
    RENDERED.append((ident, body))
    MARKERS.append(marker)
    parts.extend([f'<a id="{ident}"></a>\n', f'## {title}\n', note + '\n',
                  '源码：' + link(obj) + '\n',
                  '以下为当前函数生成的英文指令原文；动态值不填入本基础模板。\n',
                  block(body + '\n' + marker + '\n<HOST_CONTEXT_JSON>'),
                  '动态输入根键（离线基础实例）：`' + '`, `'.join(context) + '`。\n'])
    if extra_keys:
        parts.append('按实际状态还会出现：`' + '`, `'.join(extra_keys) + '`。具体取值由 Host 生成，不是另一套固定提示词。\n')


def task(kind, artifact, **kwargs):
    return w.WorkerTask(task_id='documentation-only', case_id='documentation-only',
                        task_type=kind, required_artifact_type=artifact,
                        objective='<ROLE_PROMPT_AND_HOST_CONTEXT>', **kwargs)


def body_delta(parts, title, base, variant):
    import difflib
    a, _, _ = separate(base)
    b, _, _ = separate(variant)
    parts.extend([f'### {title}\n', '只列相对于上述基础版本发生变化的行；未列出的行沿用原文。`-` 为被替换行，`+` 为新行。\n'])
    lines = list(difflib.unified_diff(a.splitlines(), b.splitlines(), fromfile='base', tofile='variant', n=0, lineterm=''))
    parts.append(block('\n'.join(lines), 'diff') if lines else '正文不变，仅动态 context 改变。\n')


def string_expression(expr):
    return eval(compile(ast.Expression(expr), '<documentation-expression>', 'eval'), vars(d))


def builder_conditionals(parts):
    fn = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == '_node_prompt')
    records = []
    def visit(node, conditions=()):
        if isinstance(node, ast.If):
            cond = ast.unparse(node.test)
            for child in node.body:
                visit(child, conditions + (cond,))
            for child in node.orelse:
                visit(child, conditions + ('not (' + cond + ')',))
            return
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if ast.unparse(node.func) == 'context_guidance.append':
                records.append((node.lineno, conditions, string_expression(node.args[0])))
        for child in ast.iter_child_nodes(node):
            visit(child, conditions)
    visit(fn)
    parts.extend(['### Builder 的全部条件追加段\n',
                  '以下是 `_node_prompt` 中每一个 `context_guidance.append`。触发条件保留代码表达式，指令保留英文原文；不会在一轮调用里无条件全部加入。普通正文已包含的 continuation/clarification 三句不再列作条件追加。\n'])
    for lineno, conditions, value in records:
        # The first enclosing condition is the shared paper-matched Builder branch.
        actual = conditions[1:] if conditions and conditions[0].startswith('paper_matched') else conditions
        parts.extend([f'触发：`{" and ".join(actual)}`；源码行 {lineno}。\n', block(value)])
    return len(records)


def const_source(module, name):
    path = Path(module.__file__)
    tree = ast.parse(path.read_text(encoding='utf-8'))
    for n in tree.body:
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets):
            relative = path.relative_to(ROOT).as_posix()
            return f'[{relative}:{n.lineno}](../../{relative}#L{n.lineno})'
    raise ValueError(name)


base = dict(target=TARGET, branch_index=0, lens='<strategy_lens>', selected_product=TARGET,
            selected_product_mapped=d._mapped_smiles(TARGET), steps=[], open_leaves=[TARGET],
            prior_rejections=[], repair=False, strategy_card=CARD, forbidden_strategy_cards=[],
            host_failure_feedback={}, paper_matched=True)
builder = d._node_prompt(**base)
portfolio = d._paper_strategy_portfolio_prompt(target=TARGET, enhanced=True)
milestone_args = dict(campaign_target=TARGET, selected_product=TARGET,
                      selected_product_mapped=d._mapped_smiles(TARGET), branch_index=0,
                      milestone_index=2, strategy_mandate='<strategy_lens>',
                      completed_strategy_cards=[], route_steps=[])
milestone = d._milestone_strategy_prompt(**milestone_args)
critic_args = dict(target=TARGET, branch_index=0, strategy_card=CARD, steps=[], paper_matched=True)
key_critic = d._critic_prompt(**critic_args, audit_kind='key_event', focus_step_id='<focus_step_id>')
route_critic = d._critic_prompt(**critic_args)
editor_args = dict(target=TARGET, strategy_card=CARD, steps=[], critic_feedback={})
editor = d._path_repair_editor_prompt(**editor_args, repair_mode='cut_frontier')

parts = ['''# 当前基础提示词全册

整理日期：2026-09-16。内容依据当前工作区源码（包含尚未提交的现有改动），不是历史版本或未来设计。文档导出只做离线读取和模板渲染，没有调用模型、库存或外部检索。

同日更新：面向模型的任务/角色身份改为 AutoPlanner，去掉通用身份句中的 blind、paper-matched 和 paper-style 定位；四维化学比较保留，但不再称为某篇论文的维度。共享条件指令扩充到影响执行、选择性、相容性和任务目标的必要参数。内部 task_type、schema_version、context 标记及历史绑定字符串暂保留兼容名称；它们不再作为模型角色定位。没有新增条件字段，也没有启用新的需求模块。

最新清理：策略差异按任务瓶颈判断；酶交接的可解释设计与实验性能分开评价；不再制造空的酶结构化副本。根据用户澄清，停止使用共享 session，恢复每次 agent 调用独立会话。每次发送完整的适用规则与 Host 上下文，不依赖上一轮会话记忆；因此跨调用重复出现的必要规则不能直接删去。使用 `python scripts/export_base_prompts.py` 离线更新本册及两个附录。

本册的范围是当前 `self_correcting_sequential` / `synthex_matched` 逆合成链路：Strategy → Builder → Key-event Critic → Whole-route Critic → Editor，以及它们的包装、工具声明、条件分支和输出协议。同一 director 内的固定三策略、逐分支策略、旧式 RouteJSON Editor 和非 paper 分支放在兼容附录，不能误认为全部同时生效。其他独立研究工具、文献匹配评测、GlobalCampaignDirector 等入口不属于这条主线，其位置列在附录边界说明中。

**本册是现状清单，不是新 prompt 方案。** 英文代码块保留程序生成的指令；中文解释和目录不是发给模型的内容。正文中的 `<HOST_CONTEXT_JSON>` 仅替代分子、路线、反馈等动态数据。用于离线渲染的占位目标不是实际化学任务，不作为执行证据。条件追加段、兼容分支和 schema 分别标注，避免把它们拼成一个从未存在的超级 prompt。

用户任务文件里的 `Process-development objective derived from ...`、PPT 文件名、页码、具体工艺要求不属于基础提示词，不收录为固定模板。历史运行日志保留原样。本册仍原样列出现有代码中的共享约束前缀，因为它已经是当前运行行为；是否拆出、重写，以及可插拔模块放在哪里，留待后续讨论。前一版 [PROCESS_BRIEF_DESIGN.md](PROCESS_BRIEF_DESIGN.md) 是待讨论草案，不代表已确定新增字段或接入位置。

## 阅读导航

| 内容 | 入口 |
| --- | --- |
| 实际拼接顺序、动态数据边界 | [调用装配](#assembly) |
| Worker 外包装与工具限制 | [包装与工具](#wrappers) |
| 现有约束注入及证据装饰器 | [共享注入](#injection) |
| 共用化学推理段 | [共享规则](#shared) |
| 条件说明的适用范围和具体内容 | [条件说明口径](#condition-scope) |
| 初始策略生成、策略组合审查 | [初始 Strategy](#strategy-initial)、[Portfolio Critic](#strategy-review) |
| 上游策略生成、上游策略审查 | [Upstream Strategy](#strategy-upstream)、[Upstream Critic](#strategy-upstream-review) |
| 一步 Builder、失败反馈、两种修复 | [Builder](#builder)、[Builder 变体](#builder-variants) |
| 关键事件初审和复审 | [Key-event Critic](#key-critic) |
| 整路线评价和修复后重评 | [Whole-route Critic](#route-critic) |
| 当前 Editor 两种模式 | [Path Editor](#editor) |
| 固定三策略、旧 Editor、非 paper 分支 | [兼容原文附录](BASE_PROMPTS_COMPATIBILITY.md) |
| 完整模型输出 JSON Schema 和工具声明 | [协议附录](BASE_PROMPTS_SCHEMAS.md) |
| 为下一轮模块讨论保留的事实 | [讨论边界](#discussion) |

<a id="assembly"></a>

## 当前实际装配顺序

```text
独立的 worker developer 指令：允许工具范围（不是 objective 字符串的一部分）
独立的工具定义：启用时提供 inspect_mapped_smiles / query_planning_evidence

提交的 worker 正文：
  1. _codex_worker_prompt 的输出 JSON 要求、任务模式和思考/简洁要求
  2. Task objective:
     2a. Bounded planning evidence capability v1（启用时）
     2b. User task constraints ... + UserTaskConstraints JSON（存在非默认约束时）
     2c. 角色指令正文（含共享化学段和条件追加段）
     2d. Context 标记 + Host 动态 JSON
     2e. 定向不确定性复审提示或物料边界评审说明（仅对应调用入口）

独立的模型输出约束：_worker_model_output_json_schema(task)
```

调用代码先生成角色正文，再由 `_with_target_constraints` 前置约束，随后 `decorate_task` 再前置证据能力说明，因此最终文本中工具说明在约束之前。证据装饰器还会做四处字符串替换，并不是纯粹添加一段文字。最外层 worker 包装最后生成。模型输出 schema 通过独立参数提交，不等于角色正文中的“Return only”句子。

`model-io.jsonl` 的 `model_input.prompt` 记录的是装饰后的业务 objective；不能把它当成包括 Worker 包装、独立工具 schema、developer 指令和供应商系统上下文的全部输入。供应商或 CLI 自动附加且未在本仓库公开记录的上下文不在本册范围内。

角色正文的输入大小检查可发生在这些前缀最终加入之前；本册不改变该现状。约束变化会进入后续装饰后的 worker 输入及现有复用身份，但输入绑定不等于已有独立评价自动全部失效或重跑。

角色输出与后续使用概览：

| 角色 | 核心输出 | 下游使用 |
| --- | --- | --- |
| 初始 Strategy | `strategy_cards`，每卡三个策略句子 | Portfolio Critic 和 Host 分支建立 |
| Portfolio Critic | 原卡/修订卡、`review_decision`、`decisive_risk` | Host 处理保留、修订、替换或 discard |
| 上游 Strategy / Critic | 当前叶的策略三句；审查时加两个 review 字段 | 为新的局部 horizon 提供方向 |
| Builder | `checkpoint_relation`、`reaction_intent`、`reaction_operations`、`conditions`、`continuation_hint` | Host 重放、MCTS、关键事件审查 |
| 修复 Builder | 上述字段加 `recovery` | 修复内展开、回退或请求扩大 Editor 范围 |
| Key-event Critic | checkpoint 匹配、化学 verdict、不确定原因、修复边界与理由 | Host 决定接受、澄清、局部修复或换 horizon |
| Whole-route Critic | 逐步评价、整体评语、风险、修复建议、依赖关系 | Host 汇总化学判断并组织修复 |
| Path Editor | 要改的步骤、修复目标和必要限制 | Host 计算范围，普通 Builder 实施重建 |

这张表不是精确 schema；完整字段、枚举、长度和数组限制见协议附录。库存判定、搜索停止和 solved 仍由 Host 拥有。

当前 Builder 每次返回一个反应，不输出路线质量分数。Host 的 `ranked_candidate_cost` 在多候选兼容路径中将返回顺序换成 `1/(index+1)` 的搜索 prior；单候选固定为 `1`。AiZynthFinder MCTS 使用该权重管理搜索；`-log(prior)` 仅供 ChemEnzy 适配路径使用。这些数值不是化学或工艺质量评价。

`connected_path_reactions` 保留紧凑历史，仅最后一个直接下游步骤附带已有条件和催化体系，用于连续步骤的进料与催化剂去除安排。`continuation_hint` 只说明当前前体的下一步机会。条件中的筛选依赖说明一次，不反复输出通用免责声明或无关比较。
''']

parts.extend(['<a id="wrappers"></a>\n', '## Worker 外包装与工具限制\n', '源码：' + link(w._codex_worker_prompt) + '\n'])
basic_task = task('paper_matched_route_step', 'RetrosynthesisProposalReport')
parts.extend(['### 无证据查询工具的普通化学 worker\n', block(w._codex_worker_prompt(basic_task))])
for title, worker in [
    ('仅本地结构/库存查询', replace(basic_task, allowed_tools=['query_planning_evidence'], host_context={'planning_evidence_transport':{'operations':['stock','list']}})),
    ('允许受限外部证据查询', replace(basic_task, allowed_tools=['query_planning_evidence'], host_context={'planning_evidence_transport':{'operations':['stock','compound','search','read','list']}})),
    ('无证据查询工具的 Path Editor', task('path_repair_editor','RetrosynthesisProposalReport')),
]:
    parts.extend([f'### {title}\n', block(w._codex_worker_prompt(worker))])
parts.extend(['### 独立的工具权限指令\n', '源码：' + link(w._configure_strict_chemistry_worker_environment) + '\n',
              '下面是实际构建表达式，三种尾句按配置择一；不是让模型自己选择权限。\n'])
wtree = ast.parse(Path(w.__file__).read_text(encoding='utf-8'))
tool_assign = next(n for n in ast.walk(wtree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='tool_instructions' for t in n.targets))
parts.append(block(ast.get_source_segment(Path(w.__file__).read_text(encoding='utf-8'), tool_assign), 'python'))
parts.append('允许工具及输入 schema 原文见 [协议附录的工具部分](BASE_PROMPTS_SCHEMAS.md#tools)。该指令由 strict chemistry worker 的运行环境配置提供，与普通业务正文区分。\n')

parts.extend(['<a id="injection"></a>\n','## 当前共享注入段\n',
              '### 现有 UserTaskConstraints 前缀\n',
              '源码：' + link(d.SequentialStrategyDirectorRunner._with_target_constraints) + '\n',
              '存在至少一个非默认约束才加入；只传默认值时不加入。这里只展示通用原文和 JSON 占位，不装入任何实验的具体工艺要求。\n'])
runner = SimpleNamespace(target_constraints={'safety_limits':{'example':'<TASK_VALUE>'}})
decorated = d.SequentialStrategyDirectorRunner._with_target_constraints(runner, basic_task)
prefix = decorated.objective.split('UserTaskConstraints:\n',1)[0]
parts.append(block(prefix+'UserTaskConstraints:\n<TARGET_CONSTRAINTS_JSON>\n\n<ROLE_PROMPT_AND_HOST_CONTEXT>'))
parts.extend(['### 证据工具前缀：仅本地查询配置示例\n',
              '源码：' + link(pe.BoundedPlanningEvidence.decorate_task) + '\n',
              '以下额度是本地配置示例，不是不可修改的基础预算；列表和额度由当前能力配置渲染。没有创建查询会话或执行查询。\n'])
def evidence_manager(external):
    manager = object.__new__(pe.BoundedPlanningEvidence)
    manager.policy = pe.PlanningEvidencePolicy(limits={'stock':24,'compound':4 if external else 0,'search':8 if external else 0,'read':4 if external else 0}, calls_per_worker=6)
    manager.stock_lookup = lambda *_: None
    manager.providers = {k:lambda *_:None for k in ('compound','search','read')} if external else {}
    return manager
local = pe.BoundedPlanningEvidence.decorate_task(evidence_manager(False), basic_task).objective
external = pe.BoundedPlanningEvidence.decorate_task(evidence_manager(True), basic_task).objective
parts.append(block(local))
parts.extend(['### 证据工具前缀：有外部查询能力的配置示例\n',block(external),
              '### 启用证据工具时对正文的四处替换\n'])
efn=ast.parse(Path(pe.__file__).read_text(encoding='utf-8'))
for n in sorted((n for n in ast.walk(efn) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='replace' and len(n.args)==2 and all(isinstance(a,ast.Constant) and isinstance(a.value,str) for a in n.args)),key=lambda x:(x.end_lineno,x.end_col_offset)):
    parts.extend(['原文：\n',block(n.args[0].value),'替换为：\n',block(n.args[1].value)])
parts.append('装饰器还将 `query_planning_evidence` 和 `inspect_mapped_smiles` 加入 allowed_tools。外部操作具体是否可用，取决于 providers 和当前配置，不由工艺任务文字决定。\n')

parts.append('''<a id="condition-scope"></a>

## 当前条件说明口径

条件应足以判断反应的正向实现、所需选择性和步骤衔接。它是路线规划层的实现假设，不是已经优化的实验规程；数值有依据时保留依据，没有依据时不能填写假精度。

| 信息 | 在哪些情况下关键 |
| --- | --- |
| 试剂、催化剂/配体或酶体系，必要的当量和负载 | 决定转化、氧化还原能力、位点选择性或绝对构型；不能只写“手性催化剂” |
| 溶剂或反应介质，必要的含水状态、浓度和 pH | 决定相容性、溶解/相态、分子内外竞争、催化活性或中间体稳定性 |
| 温度程序，必要的时间、停留时间或终点 | 活化、主反应和后处理可能需要不同条件；副反应和不稳定中间体可能由停留时间控制 |
| 加料顺序及必要的速度、预活化、分批操作 | 避免把顺序使用的酸碱或其他不相容体系写成同时混合 |
| 气氛、气体身份和必要的压力 | 区分惰性保护、氢气参与、氧气参与；压力可影响反应与设备可行性 |
| 方法特有的驱动参数 | 光化学的光源/波长或敏化剂，电化学的电极和电位/电流，酶催化的辅因子和再生体系等 |
| 淬灭、后处理、分离和溶剂置换 | 交付的游离体/盐型等应与 Host 结构一致；残余试剂和溶剂不能与下一步冲突 |
| 规模相关的混合、传热或供料问题 | 只有给定规模、设备或明确放大任务时，才要求相应评价；不从实验室候选条件推定工业可行性 |

这些不是每步必须填满的八栏。试剂体系和介质说明所提出的实现；其余参数按化学和任务相关性展开。关键值未知时，在现有 conditions 或评语中指出它为何影响判断。Critic 继续使用既有 uncertainty_source：缺少可明确的实现细节属于 proposal_underspecified，已经明确的可行设想缺少适用性证据属于 evidence_missing；具体相容性矛盾才支持 reject。普通缺项不自动成为化学失败。

字段继续使用 reaction_intent、conditions、condition_assessment 和现有风险/修改建议。共享指令放在 STAGE_GUIDANCE，Builder、两类 Critic 和 Editor 沿现有组合引用。全局约束前缀只简短指向该规则；不在每个角色重复维护一份参数清单。

''')

parts.extend(['<a id="shared"></a>\n','## 共享化学规则原文\n',
              '这些段落在完整基础模板中内嵌于角色正文。共享模式将转化粒度、条件及催化规划规则移至会话 developer 指令，其余角色规则仍在正文；不是将本节额外叠加一次。\n'])
for module,name,usage in [
    (cr,'STRATEGY_PRIORS','初始和上游策略推理；不是指定反应清单。'),
    (cr,'STRATEGY_DIVERSITY_GUIDANCE','按骨架构建或用户给定的工艺/选择性瓶颈判断策略差异，不要求固定数量。'),
    (cr,'CHEMICAL_REVIEW_SCOPE','关键事件、整路线审查及 Path Editor 的化学/库存边界。'),
    (cr,'DEPENDENCY_CRITIC_GUIDANCE','整路线 Critic 的化学前提依赖。'),
    (cr,'EDITOR_INTENT_GUIDANCE','Path Editor 的因果修改范围。'),
    (rg,'TRANSFORMATION_GUIDANCE','每条 reaction edge 的化学转化粒度。'),
    (rg,'STAGE_GUIDANCE','条件假设内的活化、主反应和后处理顺序。'),
    (rg,'BUILDER_GRANULARITY_GUIDANCE','前两段加 Builder 专用尾句，角色正文中作为一个段落展开。'),
    (rg,'CRITIC_GRANULARITY_GUIDANCE','前两段加 Critic 专用尾句。'),
    (mb,'MATERIAL_BOUNDARY_GUIDANCE','仅允许物料边界请求时加入上游 Strategy。'),
    (rr,'RECOVERY_GUIDANCE','仅修复 Builder 的 path_repair 模式。'),
]:
    parts.extend([f'### {name}\n',usage+' 源码：'+const_source(module,name)+'\n',block(getattr(module,name))])

add_prompt(parts,'strategy-initial','初始 Strategy Generator',portfolio,d._paper_strategy_portfolio_prompt,
           '当前增强版允许由化学价值决定卡片数量，可以返回空列表；不是固定三条。',())
add_prompt(parts,'strategy-review','初始 Strategy Portfolio Critic',
           d._paper_strategy_portfolio_critic_prompt(target=TARGET,strategy_cards=[CARD]),d._paper_strategy_portfolio_critic_prompt,
           '按输入顺序审查，每张卡返回一个结果；组合审查允许 discard，上游单卡审查没有该枚举。')
add_prompt(parts,'strategy-upstream','上游 Strategy Generator',milestone,d._milestone_strategy_prompt,
           '在已有分支的选定上游分子上生成下一 horizon；继承相关下游脉络，不继承兄弟叶的完整历史。',
           ('continuation_hint','selected_upstream_leaf_stereo','current_split_context','retired_strategy'))
body_delta(parts,'允许物料边界请求时的差异',milestone,d._milestone_strategy_prompt(**milestone_args,allow_material_boundary=True))
add_prompt(parts,'strategy-upstream-review','上游 Strategy Critic',
           d._upstream_strategy_critic_prompt(campaign_target=TARGET,selected_product=TARGET,selected_product_mapped=d._mapped_smiles(TARGET),branch_index=0,milestone_index=2,generated_card=CARD,completed_strategy_cards=[],accepted_route_steps=[]),
           d._upstream_strategy_critic_prompt,'审查新 horizon 与当前叶及下游已接受路径是否兼容。',
           ('continuation_hint','selected_upstream_leaf_stereo','current_split_context','retired_strategy'))
add_prompt(parts,'builder','普通一步 Route Builder',builder,d._node_prompt,
           '该正文为没有历史、失败、修复信息的基础分支；真实调用按状态加入下一节中的指令。输出一步图编辑，但要求在路线背景下选择。',
           ('selected_leaf_stereo','selected_leaf_topology','continuation_hint','connected_path_reactions','ancestor_smiles','current_split_context','last_rejection_for_this_leaf','pending_checkpoint_feedback','path_repair','boundary_observation'))
parts.extend(['<a id="builder-variants"></a>\n','## Builder 条件与修复变体\n'])
conditional_count=builder_conditionals(parts)
for mode in ('strategy_checkpoint','cut_frontier'):
    repair=d._node_prompt(**{**base,'repair':True,'strategy_card':CARD if mode=='strategy_checkpoint' else {},
                            'host_failure_feedback':{'path_repair':{'completion_mode':mode,'repair_goal':'<repair_goal>'}}})
    body_delta(parts,f'{mode} 修复相对普通 Builder 的变化',builder,repair)
parts.extend(['### 有 retained reconnect boundaries 时的私有规划句\n'])
node_fn=next(n for n in TREE.body if isinstance(n,ast.FunctionDef) and n.name=='_node_prompt')
private_assign=next(n for n in ast.walk(node_fn) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='private_path_instruction' for t in n.targets))
parts.append(block(ast.get_source_segment(SOURCE_TEXT,private_assign),'python'))

add_prompt(parts,'key-critic','Key-event Critic 初审',key_critic,d._critic_prompt,
           '只审查 focus edge 及直接下游接口；先前步骤是上下文，不在本轮逐一重审。',
           ('focus_step_topology','active_checkpoint_constraints','failure_basin'))
body_delta(parts,'选中新的直接上游证据后的复审',key_critic,d._critic_prompt(**critic_args,audit_kind='key_event_followup',focus_step_id='<focus_step_id>'))
parts.extend(['### 调用入口追加的定向不确定性复审句\n',
              '该句在 `_critic_prompt` 返回后追加，位于 KeyEventCriticInput JSON 之后。只有当前复审记录给出 uncertainty_source 时加入；下面两种尾句按来源选择。\n'])
followup=next(n for n in ast.walk(TREE) if isinstance(n,ast.AugAssign) and isinstance(n.target,ast.Name) and n.target.id=='prompt' and isinstance(n.value,ast.BinOp) and 'Targeted uncertainty follow-up:' in ast.unparse(n.value))
for source in ('evidence_missing','assessment_unresolved'):
    val=eval(compile(ast.Expression(followup.value),'<documentation-expression>','eval'),{'review':{'uncertainty_source':source}})
    parts.extend([f'`uncertainty_source={source}`；源码行 {followup.lineno}。其他非空原因使用第二种尾句。\n',block(val)])
add_prompt(parts,'route-critic','Whole-route Critic',route_critic,d._critic_prompt,
           '从前体到目标正向检查化学，仍保持 RouteJSON 的目标向上游存储顺序。整体评语限定为化学评价，库存由 Host 另算。',
           ('repair_requirements_to_reassess','repair_checkpoint_focus'))
recritic=d._critic_prompt(**critic_args,repair_completion={'completion_mode':'strategy_checkpoint','required_checkpoint_step_id':'<step_id>','repair_goal':'<repair_goal>','active_constraints':['<constraint>']})
body_delta(parts,'strategy_checkpoint 修复后的重评',route_critic,recritic)
parts.append('`_bounded_critic_prompt` 在 context 超预算时尝试更高压缩等级；它调用同一 `_critic_prompt`，不是另一套角色指令。完整动态字段投影以源码为准。\n')
parts.extend(['### 调用入口追加的物料边界评审说明\n',
              '当 `current_material_boundary(branch)` 非空时，在整路线 Critic 的主 context JSON 之后追加。这是待评审的当前物料边界数据，不是采购证明。\n'])
material_assign=next(n for n in ast.walk(TREE) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='material_context' for t in n.targets))
material_prefix=next(n.value for n in ast.walk(material_assign.value) if isinstance(n,ast.Constant) and isinstance(n.value,str) and 'Material sourcing review is pending' in n.value)
parts.append(block(material_prefix+'<MATERIAL_BOUNDARY_REVIEW_JSON>'))
add_prompt(parts,'editor','当前 Path Repair Editor：cut_frontier',editor,d._path_repair_editor_prompt,
           '当前意图式 Editor 输出修改范围与目标，不直接写完整替换图；后续 Builder 一步步实施。',
           ('strategy','provisional_rejected_step_ids'))
body_delta(parts,'strategy_checkpoint 模式',editor,d._path_repair_editor_prompt(**editor_args,repair_mode='strategy_checkpoint'))
body_delta(parts,'包含未准入的临时失败步骤',editor,d._path_repair_editor_prompt(**editor_args,repair_mode='cut_frontier',provisional_rejected_step_ids=['<step_id>']))

parts.append('''<a id="discussion"></a>

## 留给可插拔约束模块讨论的事实

这里只记录现有承载位置及可能相互影响的指令，不确定新模块接口，也不新增字段。

| 当前承载位置 | 已有内容 | 后续需要讨论什么 |
| --- | --- | --- |
| 角色正文 | 推理任务、策略多样性、一步输出、检查范围 | 哪些长期通用，哪些其实依赖任务范围 |
| `_with_target_constraints` | 非默认 constraints 和一段带 process_brief/深冷/分离语义的前缀 | 通用约束解释是否应与工艺专用文字分开 |
| `decorate_task` | 工具能力、额度、库存语义、起始物料推理及四处正文替换 | 能力模块与用户需求模块的边界 |
| Host context | 目标、当前叶、历史、失败反馈、修复边界 | 动态状态如何与稳定的任务限制区分 |
| Worker 包装和独立 developer 指令 | 输出纪律、证据权限、工具范围 | 用户需求如何使用这些能力而不覆盖权限 |
| 模型输出 schema | 字段、枚举和长度限制 | 优先复用现有评语；确需调度信号时再讨论最小新增 |

值得在后续讨论中直接对照的四处现状：初始 Strategy 已按任务瓶颈区分骨架合成与工艺开发；上游 Strategy 对当前叶的 horizon 要求；Critic 的“Evaluate chemistry only”；现有约束前缀要求 Critic 比较工艺目标。工艺偏好与化学否决仍应分开表达，不能仅靠把新段落放得更靠前就假定所有语义冲突已经解决。

`active_constraints` 这个名称还出现在 Key Critic 记忆和 Editor 修复任务中，它们是路线局部的化学修复要求，不等于用户全局需求。本册不把这几种用途合并成一个已确定的数据接口。

## 本次离线核对

英文主体由当前 Python prompt 函数直接生成；条件追加段由同一源码 AST 提取；输出 schema 由当前 `_worker_model_output_json_schema` 生成。没有用历史 IO 的一小部分冒充全部当前模板，也没有把待讨论方案混入原文。

人工解释与固定文本分离，动态 context 明确占位，所有普通/条件/兼容分支均有来源。`Return only` 文字和 schema 若有差异，本册保留差异，不静默改写。例如 Key-event Critic 在前文要求 uncertainty_source，后面的字段枚举句未再次列它，而当前 schema 要求该字段；以协议附录展示的实际 schema 为准。
''')

# Compatibility branches remain separate from the current mainline.
compat=['''# 基础提示词兼容分支附录

整理日期：2026-09-16。这里列出与主线共享当前源码、但由不同开关触发的模板。存在于代码不表示最新实验使用过。所有原文离线生成，不是调用记录。主册见 [BASE_PROMPTS_CURRENT.md](BASE_PROMPTS_CURRENT.md)。
''']
add_prompt(compat,'frozen-three','冻结/控制版固定三策略',d._paper_strategy_portfolio_prompt(target=TARGET,enhanced=False),d._paper_strategy_portfolio_prompt,'触发：`enhanced=False`；此时 context 带 strategy_count=3，schema 可以强制三条。')
for paper in (True,False):
    add_prompt(compat,'single-'+str(paper),'逐分支 Strategy：paper_matched='+str(paper),
               d._strategy_prompt(target=TARGET,branch_index=0,lens='<lens>',forbidden_strategy_cards=[],prior_rejections=[],paper_matched=paper),
               d._strategy_prompt,'与一次生成整个 portfolio 的入口区分；其约束与输出契约不能自动替代增强版。')
add_prompt(compat,'strategy-v2','非 paper strategy-v2 slot 分支',
           d._strategy_prompt(target=TARGET,branch_index=0,lens='strategy_v2_slot=<slot>',forbidden_strategy_cards=[],prior_rejections=[],paper_matched=False),
           d._strategy_prompt,'触发：lens 含 strategy_v2_slot=，并且 paper_matched=False。')
add_prompt(compat,'old-editor','直接输出 replace_span 的 RouteJSON Editor',d._node_prompt(**{**base,'editor_route_mutations':True}),d._node_prompt,
           '触发：`paper_matched and editor_route_mutations`；与当前 Path Editor 输出改造意图的模式不同。')
for ident,title,options in [
    ('generic-builder','非 paper 普通 Builder',{'paper_matched':False}),
    ('generic-repair','非 paper 局部修复 Builder',{'paper_matched':False,'repair':True}),
    ('full-route','完整 RouteJSON 输出分支',{'paper_matched':True,'complete_route_json':True}),
    ('generic-editor','非 paper 完整 RouteJSON/patch Editor',{'paper_matched':False,'editor_route_mutations':True}),
]:
    add_prompt(compat,ident,title,d._node_prompt(**{**base,**options}),d._node_prompt,'当前同一函数中的条件分支；不代表当前一步式搜索会同时收到这些指令。')
add_prompt(compat,'generic-critic','非 paper Route Critic',d._critic_prompt(**{**critic_args,'paper_matched':False}),d._critic_prompt,'触发：`paper_matched=False`。')
compat.extend(['## 通用分支剩余的条件段\n',
               '上述普通模板使用无历史、非生物域、默认候选数/深度的离线参数。下列条件开启时，原文按源码替换或扩充；候选数和最小路线深度由实际配置决定，不固定为示例中的 1。\n'])
for node in ast.walk(node_fn):
    if isinstance(node,ast.If) and ast.unparse(node.test) in {'strategy_anchor_fulfilled','strategy_domain in BIOLOGICAL_EXECUTION_DOMAINS','compact_editor_context'}:
        compat.extend(['触发：`'+ast.unparse(node.test)+'`。\n'])
        for stmt in node.body:
            for literal in ast.walk(stmt):
                if isinstance(literal,ast.Constant) and isinstance(literal.value,str) and len(literal.value)>70:
                    compat.append(block(literal.value))
compat.extend(['## 调度层摘要文本\n',
               'SequentialStrategyDirectorRunner.prompt_for 返回的是调度摘要，不是新增的 LLM 角色正文；实际 worker 调用仍使用主册所列专用模板。完整原文如下。\n',
               block(inspect.getsource(d.SequentialStrategyDirectorRunner.prompt_for),'python')])
compat.extend(['## 早期 return 之后仍保留的分支文字\n',
               '当前 `_node_prompt` 在 paper 一步 Builder 和 paper Editor 分支提前返回；后部通用分支中仍有下面三句 paper 条件文案。它们在当前控制流组合下不会成为这两条主线的额外指令，不能与主册叠加。保留于此，仅用于源码文字清点。\n'])
for value in (
    'Return one JSON object containing exactly one local ReactionJSON expansion for the selected node.',
    'Return one complete edited route_json or one coordinated route_patch, not a prose-only sketch and not multiple output candidates.',
    'Return exactly one local ReactionJSON transformation candidate; the Builder has no terminal action and must not return a prose-only route.',
):
    assert value in SOURCE_TEXT
    compat.append(block(value))
compat.extend(['## 非 strict worker 外包装\n',
               '非 strict 工作者另有完整 artifact wrapper、draft 要求及按 artifact 类型附加的指令，不应与主线紧凑输出包装叠加。以下函数原文保留所有包装条件。\n',
               block(inspect.getsource(w._codex_worker_prompt),'python'),
               '## artifact 指令分派原文\n',
               '该函数用于通用 worker 路径，不是当前 strict `_codex_worker_prompt` 必经层。列出原文是为了避免在抽取时误把它当成当前全部附加指令。\n',
               block(inspect.getsource(w._artifact_payload_instruction),'python'),
               '## 本册之外的独立入口\n',
               '以下位置有其他任务的 prompt，不属于本次 Strategy/Builder/Critic/Editor 基线。若未来模块要覆盖整个研究平台，需要另定这些入口的语义；不能据此宣称现有 `_with_target_constraints` 已覆盖它们。\n',
               '- `orchestration/global_campaign_director.py::director_prompt`：GlobalCampaignDirector。\n'
               '- `agent/condition_agent.py`：独立条件建议任务。\n'
               '- `agent/prior_generator.py`：独立规划先验生成器。\n'
               '- `agent/smiles_first.py`：独立 SMILES-first 研究任务组装。\n'
               '- `harness/`、`legacy/` 和离线评测脚本：各自的文献、视觉和比较任务。\n'])

# Export the exact response schemas without exercising workers.
schema_specs=[
    ('策略组合生成','paper_matched_strategy_generator','StrategyPortfolioReport',{}),
    ('冻结版三策略组合生成','paper_matched_strategy_generator','StrategyPortfolioReport',{'strategy_count':3}),
    ('策略组合审查','paper_matched_strategy_critic','StrategyPortfolioReport',{}),
    ('上游单策略生成','paper_matched_strategy_generator','StrategyCardReport',{}),
    ('带物料边界选择的上游生成','paper_matched_strategy_generator','StrategyCardReport',{'allow_material_boundary':True}),
    ('上游单策略审查','paper_matched_strategy_critic','StrategyCardReport',{}),
    ('一步 Builder','paper_matched_route_step','RetrosynthesisProposalReport',{}),
    ('修复 Builder','paper_matched_route_step','RetrosynthesisProposalReport',{'allow_repair_recovery':True}),
    ('关键事件 Critic','paper_matched_key_event_critic','ChemicalStrategyCritique',{}),
    ('整路线 Critic','paper_matched_route_critic','ChemicalStrategyCritique',{}),
    ('当前 Path Editor','path_repair_editor','RetrosynthesisProposalReport',{}),
    ('兼容 RouteJSON Editor','paper_matched_route_editor','RetrosynthesisProposalReport',{}),
]
schemas=['''# 当前模型输出协议与工具定义原文

整理日期：2026-09-16。主册：[BASE_PROMPTS_CURRENT.md](BASE_PROMPTS_CURRENT.md)。这些是模型可见输出 schema，不是 Host 事后添加的 artifact 包装。字段、required、additionalProperties、枚举、长度和数组限制均由当前代码直接生成；不做手工简化。

JSON schema 通过独立输出参数约束模型，不需要再把它当成普通 prompt 全文复制一次。中文标题不是模型指令。这里的“修复”及“物料边界”版本由 host_context 开关控制。
''','源码：'+link(w._worker_model_output_json_schema)+'\n']
schema_values=[]
for title,kind,artifact,host in schema_specs:
    value=w._worker_model_output_json_schema(task(kind,artifact,host_context=host))
    schema_values.append(value)
    schemas.extend([f'## {title}\n',f'任务：`{kind}`；artifact：`{artifact}`；host 开关：`{json.dumps(host)}`。\n',block(json.dumps(value,ensure_ascii=False,indent=2),'json')])
schemas.extend(['<a id="tools"></a>\n','## 工具定义原文\n',
                '定义来自 `application/chemistry_inspection_mcp.py`。query 的操作枚举和参数随当前可用操作裁剪；本段仅计算定义，不启动 MCP 服务，不调用网络。\n'])
mp=ROOT/'cascade_planner/application/chemistry_inspection_mcp.py'
mt=ast.parse(mp.read_text(encoding='utf-8'))
functions=[n for n in mt.body if isinstance(n,ast.FunctionDef) and n.name in {'_tool_definition','_query_definition'}]
env={'Any':object,'TOOL_NAME':'inspect_mapped_smiles','QUERY_NAME':'query_planning_evidence'}
exec(compile(ast.Module(body=functions,type_ignores=[]),str(mp),'exec'),env)
schemas.extend(['### inspect_mapped_smiles\n',block(json.dumps(env['_tool_definition'](),ensure_ascii=False,indent=2),'json')])
for ops in [['stock','list'],['stock','compound','search','read','list']]:
    env['_query_operations']=lambda ops=ops:ops
    schemas.extend(['### query_planning_evidence：'+', '.join(ops)+'\n',block(json.dumps(env['_query_definition'](),ensure_ascii=False,indent=2),'json')])

parts.extend(['## 当前会话方式\n',
              '停止共享 session 实验。Strategy、Builder、Critic、Editor 的每一次新模型调用都使用独立的 `codex exec --ephemeral` 和临时 CODEX_HOME，不执行 `exec resume`。多次 Builder 或 Critic 调用同样不共享会话历史。\n',
              '路线连续性通过 Host 显式提供的当前结构、已执行步骤、continuation_hint、约束和修复反馈传递；各次提示词保留适用的完整规则。共用 Python 常量用于维护一致性，不代表模型已在其他会话读过这些规则。\n',
              '已撤回共享会话入口、正文删减及专用缓存/累计用量处理。策略与酶步骤修复保留；此前单会话 IO 和实验报告作为历史记录保留，不代表后续运行配置。资源上限、ZINC 库存和外部检索设置沿用原配置。\n',
              '源码：' + link(w._run_codex_cli_worker) + '\n'])

for name,values in [('BASE_PROMPTS_CURRENT.md',parts),('BASE_PROMPTS_COMPATIBILITY.md',compat),('BASE_PROMPTS_SCHEMAS.md',schemas)]:
    text='\n'.join(values).rstrip()+'\n'
    (OUT/name).write_text(text,encoding='utf-8',newline='\n')
    print(name,len(text.splitlines()),len(text.encode('utf-8')))
print('Rendered prompt bodies:',len(RENDERED),'conditional Builder additions:',conditional_count,'model schemas:',len(schema_values))

# Documentation checks: no network, no model invocation, no runtime state writes.
all_text='\n'.join((OUT/f).read_text(encoding='utf-8') for f in ['BASE_PROMPTS_CURRENT.md','BASE_PROMPTS_COMPATIBILITY.md'])
assert all(body in all_text for _,body in RENDERED)
assert conditional_count==14,conditional_count
for value in schema_values:
    assert isinstance(value,dict) and value.get('type')=='object'
print('Offline verbatim/template/schema checks passed.')
