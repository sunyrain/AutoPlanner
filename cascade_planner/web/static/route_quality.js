/* Presentation only: no route edits, chemical telescoping or verdict changes. */
(function (root) {
  'use strict';
  const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'}[c]));

  function analyze(steps, target) {
    const rows = Array.isArray(steps) ? steps : [];
    const valid = i => Number.isInteger(i) && i >= 0 && i < rows.length;
    const groups = rows.map((step, index) => {
      const routeGroups = Array.isArray(step.display_precursors) ? step.display_precursors.map(group => ({
        ...group, child_step_indices: [...new Set((group.child_step_indices || []).filter(valid))],
      })) : [...new Set(step.precursor_smiles || [])].map(smiles => ({
      // Legacy data without topology can connect only by exact saved identity.
      // Array adjacency is never evidence of a reaction dependency.
        smiles, count: (step.precursor_smiles || []).filter(value => value === smiles).length,
        child_step_indices: rows.flatMap((child, i) => i !== index && child.product_smiles === smiles ? [i] : []),
      }));
      const additional = additionalReactionInputs(step);
      return [...routeGroups, ...[...new Set(additional)].map(smiles => ({
        smiles, count: additional.filter(value => value === smiles).length,
        co_reactant: true, child_step_indices: [],
      }))];
    });
    const children = groups.map(values => [...new Set(values.flatMap(group => group.child_step_indices))]);
    const parents = rows.map(() => new Set());
    children.forEach((values, parent) => values.forEach(child => parents[child].add(parent)));
    const roots = rows.flatMap((step, i) => step.product_smiles === target ? [i] : []);
    const reached = new Set(), visiting = new Set(), memo = new Map();
    let cycle = false;
    function depth(i) {
      reached.add(i);
      if (visiting.has(i)) { cycle = true; return 0; }
      if (memo.has(i)) return memo.get(i);
      visiting.add(i);
      const result = 1 + Math.max(0, ...children[i].map(depth));
      visiting.delete(i); memo.set(i, result); return result;
    }
    const longest = Math.max(0, ...roots.map(depth));
    const alternatives = roots.length > 1 || groups.some(values => values.some(group => group.has_route_alternatives || group.child_step_indices.length > 1));
    function anchor(i) {
      const step = rows[i];
      return step.checkpoint_relation === 'executes_checkpoint' || step.is_key === true ||
        step.critic_verdict === 'reject' || !(step.precursor_smiles || []).length;
    }
    function chain(start, seen = new Set()) {
      const result = [start], used = new Set([...seen, start]);
      if (!valid(start) || anchor(start) || cycle) return valid(start) ? result : [];
      let i = start;
      while (groups[i].length === 1) {
        const group = groups[i][0];
        if (group.has_route_alternatives || group.child_step_indices.length !== 1 || Number(group.count || 1) !== 1) break;
        const child = group.child_step_indices[0];
        if (used.has(child) || parents[child].size !== 1 || anchor(child)) break;
        // Keep a convergent junction, side reactant or alternative boundary visible.
        if (groups[child].length !== 1 || groups[child].some(value => value.has_route_alternatives || value.child_step_indices.length > 1)) break;
        result.push(child); used.add(child); i = child;
      }
      return result;
    }
    return {rows, groups, roots, chain, recordCount: rows.length,
      longest: cycle || (rows.length && !roots.length) ? null : longest,
      unreachable: rows.length - reached.size, alternatives, cycle};
  }

  function countingHtml(a, grouped = null) {
    const toggle = typeof grouped === 'boolean' ? `<label><input type="checkbox" data-paper-grouping ${grouped ? 'checked' : ''}>论文分组</label>` : '';
    return `<aside class="routeCounting" aria-label="路线粒度与计数">${toggle}<span>反应记录 <b data-route-record-count>${a.recordCount}</b>${a.alternatives ? '（含备选）' : ''}</span><span>最长线性路径 <b data-route-lls>${a.longest === null ? '未确定' : a.longest}</b> 条</span>${a.unreachable ? `<span>${a.unreachable} 条未连接目标</span>` : ''}<small>按保存的反应记录计数；折叠不减步数，实验操作数未确定。</small></aside>`;
  }

  function createView(onChange, onLayout) {
    let grouped = true, current = null;
    const opened = new Set();
    document.addEventListener('change', event => {
      if (!event.target.matches('[data-paper-grouping]')) return;
      grouped = event.target.checked; onChange();
    });
    document.addEventListener('toggle', event => {
      const detail = event.target;
      if (!detail.matches?.('details.routeSequence')) return;
      if (detail.open) opened.add(detail.dataset.sequenceId); else opened.delete(detail.dataset.sequenceId);
      onLayout?.();
    }, true);
    function prepare(steps, target) { current = analyze(steps, target); return current; }
    function mount(viewport) {
      let bar = viewport.previousElementSibling;
      if (!bar?.classList.contains('routeCountingHost')) {
        bar = document.createElement('div'); bar.className = 'routeCountingHost';
        viewport.before(bar);
      }
      bar.innerHTML = countingHtml(current, grouped);
    }
    function sequence(start, seen, frame, connector, molecule, tail) {
      const indices = grouped ? current.chain(start, seen) : [start];
      if (indices.length < 2) return null;
      indices.forEach(i => seen.add(i));
      const ids = indices.map(i => String(current.rows[i].step_id || i));
      const key = ids.join('|');
      const focused = [...(frame?.new_step_ids || []), ...(frame?.changed_step_ids || []), ...(frame?.proposed_step_ids || [])].some(id => ids.includes(String(id)));
      const open = focused || opened.has(key);
      const content = indices.map((i, pos) => `${pos ? molecule(current.rows[i].product_smiles) : ''}${connector(current.rows[i])}`).join('');
      const families = [...indices].reverse().map(i => `<li>${esc(current.rows[i].reaction_family || current.rows[i].transformation_rationale || '反应类型未记录')}</li>`).join('');
      return `<details class="routeSequence" data-sequence-id="${esc(key)}" data-sequence-count="${indices.length}" ${open ? 'open' : ''}><summary><b>${indices.length} 条转化记录</b><span>展开步骤、条件与中间体</span><ol aria-label="正向合成顺序">${families}</ol></summary><div class="routeSequenceContent">${content}</div></details>${tail(indices[indices.length - 1])}`;
    }
    return {prepare, mount, sequence, groups: index => current.groups[index]};
  }
  function reactionInputs(step) {
    const supplied = step.reaction_input_smiles || [];
    return supplied.length ? [...supplied] : [...(step.precursor_smiles || []), ...(step.auxiliary_reagent_smiles || [])];
  }
  function stockPresentation(smiles, snapshot, label = '') {
    const hit = (snapshot?.stock_hit_smiles || []).includes(smiles);
    return {
      className: hit ? 'stockHit' : '',
      label: hit ? label.replace(/^OPEN LEAF/, 'STOCK LEAF') : label,
      badge: hit ? '<div class="stockHitBadge" title="该任务保存的库存目录确认命中；不代表供应商实时现货。">✓ 库存命中</div>' : '',
    };
  }
  function additionalReactionInputs(step) {
    const remaining = [...(step.precursor_smiles || [])];
    return reactionInputs(step).filter(value => {
      const index = remaining.indexOf(value);
      if (index < 0) return true;
      remaining.splice(index, 1);
      return false;
    });
  }
  function reactionInputStructures(values, molecule) {
    const render = molecule || (smiles => `<img alt="${esc(smiles)}" src="/api/v4/molecule.svg?smiles=${encodeURIComponent(smiles)}">`);
    return `<div class="reactionInputStructures">${values.map(value => `<figure><div class="moleculeVisual" data-molecule-smiles="${esc(value)}"><div class="moleculeInline">${render(value)}</div></div><figcaption><code>${esc(value)}</code></figcaption></figure>`).join('')}</div>`;
  }
  function criticPresentation(step) {
    const labels = {pass: '通过 · pass', uncertain: '存在不确定性 · uncertain', reject: '建议拒绝 · reject'};
    const current = String(step.critic_verdict || '');
    if (labels[current]) return {verdict: current, label: labels[current], scope: '步骤 Critic 判断', note: '', review: step};
    const historical = step.historical_critic;
    if (historical && labels[historical.critic_verdict]) return {
      verdict: historical.critic_verdict, label: labels[historical.critic_verdict], scope: '历史步骤 Critic',
      note: '当前终审未绑定。以下为历史版本的评审及理由，不代表当前最终路线已通过。',
      review: historical, historical: true
    };
    return {verdict: 'unavailable', review: step, scope: '评审可用性',
      label: current === 'reviewed' ? '整路已评审 · 未单列步骤判定' : current === 'unavailable' ? '当前终审未绑定' : '未记录步骤评审',
      note: current === 'reviewed' ? '整体路线结论不能自动当作每一步的 pass。'
        : '没有适用于当前步骤版本的明确判断；这不等于 pass、uncertain 或 reject。'};
  }

  function historicalRouteReviewHtml(branch) {
    const review = branch?.historical_critic;
    if (!review || branch.chemical_critic_status !== 'unavailable') return '';
    const verdict = String(review.overall_assessment || ''), label = {viable:'可行 · viable',uncertain:'存在不确定性 · uncertain',reject:'拒绝 · reject'}[verdict];
    if (!label) return '';
    return `<section class="strategyDetailSection historicalCritic"><span>历史整路 Critic · 未绑定当前最终版本</span><b class="criticVerdict ${verdict==='viable'?'pass':esc(verdict)}">${esc(label)}</b><p>${esc(review.route_overall_evaluation || '已保存历史整体判断；没有保存整体评语。')}</p><small class="criticScopeNote">历史判断保留供审查，不替代当前终审。${review.critic_task_id ? ` 来源：${esc(review.critic_task_id)}` : ''}</small></section>`;
  }

  function reactionDetailHtml(step, {origin = '', status = '', relationLabel = '', verdictLabel = '', molecule = null} = {}) {
    // Keep authored entries intact: an array can contain alternatives as well as
    // stages. Do not invent an execution order by splitting sentences or numbering.
    const texts = value => (Array.isArray(value) ? value : [value]).filter(value => typeof value === 'string' && value.trim());
    const list = (values, className) => `<ul class="${className}">${values.map(value => `<li>${esc(value)}</li>`).join('')}</ul>`;
    const section = (title, content, className = '') => content ? `<details class="reactionDetailSection ${className}" open><summary>${esc(title)}</summary><div class="reactionSectionContent">${content}</div></details>` : '';
    const fact = (label, content) => `<div class="reactionDetailFact"><span>${esc(label)}</span><div>${content}</div></div>`;
    const conditions = [...new Set(texts(step.conditions))], catalyst = String(step.catalyst || '');
    const precursors = texts(reactionInputs(step)), assessment = criticPresentation(step), reviewData = assessment.review;
    const family = step.reaction_family || step.transformation_rationale || '', rationale = String(step.transformation_rationale || '');
    const catalystHtml = catalyst && !conditions.includes(catalyst) ? `<p class="reactionCatalyst"><span>催化体系</span>${esc(catalyst)}</p>` : '';
    const missing = (step.step_origins || []).some(value => value.kind === 'aizynthfinder_short_tail')
      ? 'AiZ 未提供实验条件；该步仅完成模板级结构衔接' : '未记录具体条件';
    const conditionHtml = section('条件 / 催化剂', catalystHtml + (conditions.length
      ? list(conditions, 'reactionConditionList') : `<p class="reactionDetailEmpty">${missing}</p>`), 'reactionConditions');
    const inputsHtml = precursors.length ? section(`本步反应物 · ${precursors.length}`, reactionInputStructures(precursors, molecule)) : '';
    const explanation = rationale && rationale !== family ? section('转换说明', `<p>${esc(rationale)}</p>`) : '';
    const reasons = texts(reviewData.critic_reasons), limitations = texts(step.builder_limitations), prefix = assessment.historical ? '历史 ' : '';
    const review = (reasons.length ? section(prefix + 'Critic 依据', list(reasons, 'reactionReviewList')) : '')
      + section(prefix + '条件判断', reviewData.critic_condition_assessment ? `<p>${esc(reviewData.critic_condition_assessment)}</p>` : '')
      + section(prefix + '建议修订', reviewData.critic_suggested_revision ? `<p>${esc(reviewData.critic_suggested_revision)}</p>` : '')
      + (limitations.length ? section('Builder 限制', list(limitations, 'reactionReviewList')) : '');
    const meta = fact('来源 / 状态', `${esc(origin)}${origin && status ? ' · ' : ''}${esc(status)}`)
      + (relationLabel ? fact('路线角色', esc(relationLabel)) : '')
      + fact(assessment.scope, `<span class="criticVerdict ${esc(assessment.verdict)}">${esc(!assessment.historical && ['pass','uncertain','reject'].includes(step.critic_verdict) ? verdictLabel || assessment.label : assessment.label)}</span>${assessment.note ? `<p class="criticScopeNote">${esc(assessment.note)}</p>` : ''}`);
    const technical = `<details class="reactionTechnical"><summary>结构与 SMILES · ${precursors.length} 个反应物</summary><div class="reactionTechnicalGrid"><code title="产物：${esc(step.product_smiles || '未记录')}">产物 · ${esc(step.product_smiles || '未记录')}</code>${precursors.length ? precursors.map(value => `<code title="反应物：${esc(value)}">反应物 · ${esc(value)}</code>`).join('') : '<code>尚无可投影反应物</code>'}</div></details>`;
    return `<div class="reactionDetailMeta">${meta}</div><div class="reactionDetailColumns${review ? ' hasReview' : ''}"><div class="reactionDetailMain">${inputsHtml}${conditionHtml}${explanation}</div>${review ? `<div class="reactionDetailReview" aria-label="模型审查">${review}</div>` : ''}</div>${technical}`;
  }

  const api = {analyze, countingHtml, createView, reactionDetailHtml, criticPresentation, historicalRouteReviewHtml, reactionInputs, additionalReactionInputs, reactionInputStructures, stockPresentation};
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.RoutePresentation = api;
})(typeof globalThis === 'undefined' ? this : globalThis);

(() => {
  'use strict';
  if (typeof document === 'undefined') return;
  const hosts = '.moleculeVisual, .molecule-art, .graph-molecule, .target-visual';
  const dialog = document.createElement('dialog');
  dialog.className = 'structure-dialog';
  dialog.setAttribute('aria-label', '分子结构高清查看');
  dialog.innerHTML = '<header><b>分子结构 · 高清查看</b><button type="button" data-close aria-label="关闭结构大图">关闭 ×</button></header><div class="structure-large"></div><code></code><footer><span>矢量结构可无损放大，SVG 可用于文档与打印。</span><a download="molecule.svg">保存 SVG</a></footer>';
  document.body.append(dialog);
  let downloadUrl = '', previousFocus = null;
  function releaseDownload() {
    if (downloadUrl) URL.revokeObjectURL(downloadUrl);
    downloadUrl = '';
  }
  dialog.querySelector('[data-close]').addEventListener('click', () => dialog.close());
  dialog.addEventListener('close', () => { releaseDownload(); previousFocus?.focus(); });
  dialog.addEventListener('click', event => { if (event.target === dialog) dialog.close(); });
  function decorate(scope) {
    const nodes = scope.matches?.(hosts) ? [scope] : [];
    nodes.push(...scope.querySelectorAll(hosts));
    nodes.forEach(host => {
      if (host.querySelector('.structure-expand') || !host.querySelector('svg, img')) return;
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'structure-expand';
      button.textContent = '放大';
      button.setAttribute('aria-label', '放大查看分子结构');
      host.append(button);
    });
  }
  document.addEventListener('click', event => {
    const button = event.target.closest('.structure-expand');
    if (!button) return;
    const host = button.closest(hosts), source = host.querySelector('svg, img');
    if (!source) return;
    event.stopPropagation();
    previousFocus = button;
    releaseDownload();
    dialog.querySelector('.structure-large').replaceChildren(source.cloneNode(true));
    const card = host.closest('article, .targetCard, .previewCard');
    dialog.querySelector('code').textContent = host.querySelector('[data-molecule-smiles]')?.dataset.moleculeSmiles || card?.querySelector('.nodeText, code')?.textContent || '';
    const link = dialog.querySelector('a');
    if (source.tagName.toLowerCase() === 'svg') {
      const svg = source.cloneNode(true);
      svg.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
      downloadUrl = URL.createObjectURL(new Blob([new XMLSerializer().serializeToString(svg)], {type: 'image/svg+xml'}));
      link.href = downloadUrl;
    } else {
      link.href = source.src;
    }
    dialog.showModal();
  });
  decorate(document);
  new MutationObserver(records => {
    records.forEach(record => {
      // A live SVG image arrives inside an existing molecule host.
      const host = record.target.closest?.(hosts);
      if (host) decorate(host);
      record.addedNodes.forEach(node => { if (node.nodeType === 1 && !dialog.contains(node)) decorate(node); });
    });
  }).observe(document.body, {childList: true, subtree: true});
})();
