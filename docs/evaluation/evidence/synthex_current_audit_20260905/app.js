/* SynthAtlas single-page app — vanilla JS, hash-routed, data from ./data/*.json */
(() => {
  const app = document.getElementById("app");
  if (window.cytoscape && window.cytoscapeDagre) window.cytoscape.use(window.cytoscapeDagre);
  const cache = {};
  // Every dataset fetch goes through here with a logical "data/…" path; config.js maps that
  // to same-origin site/data locally and to the versioned R2 prefix everywhere else (see the
  // note there). The cache stays keyed by the logical path, so the mapping is invisible here.
  const dataUrl = (window.SA_CONFIG || {}).dataUrl || ((p) => p);
  const jget = (p) =>
    (cache[p] ??= fetch(dataUrl(p))
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status} ${r.statusText} for ${dataUrl(p)}`);
        return r.json();
      })
      // Don't let one failed request poison the path for the rest of the session: a cached
      // rejected promise would make every later jget(p) fail too, so drop it and let the
      // next call retry.
      .catch((e) => { delete cache[p]; throw e; }));
  const ASSET_VER = "16";   // bump when structures are re-rendered, to bust image cache
  // where molecule SVGs are served from: same-origin "img/mol" for local builds, the R2
  // bucket otherwise (36k SVGs exceed Cloudflare Pages' 20k-file limit, so they live in
  // object storage rather than the Pages deploy). Resolved in config.js.
  const IMG_BASE = ((window.SA_CONFIG || {}).imgBase || "img/mol").replace(/\/+$/, "");
  // crossorigin="anonymous" on EVERY molecule image so the CDN only caches CORS-clean copies —
  // otherwise a plain <img> can cache a non-CORS copy that the <canvas> figure/PDF export and the
  // Cytoscape route tree then can't draw (tainted → blank nodes). Requires the R2 bucket's CORS to
  // allow this origin (see scripts/r2-cors.json); same-origin local dev is unaffected.
  const molImg = (h, cls = "", style = "") =>
    h ? `<img class="${cls}" data-hash="${h}" loading="lazy" crossorigin="anonymous" src="${IMG_BASE}/${h}.svg?v=${ASSET_VER}"${style ? ` style="${style}"` : ""} alt="structure ${h}">` : "";
  // Mini reaction-list schemes share one fixed fraction of each molecule's true render size
  // (w/h from molecules.json), so every card sizes structures consistently rather than each
  // <img> hitting a different CSS max-width/height clamp. 0.45 fits most schemes in a card.
  const MINI_SCALE = 0.45;
  const esc = (s) =>
    String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const feasClass = (f) => "feas feas-" + String(f || "").toLowerCase().replace(/[^a-z]/g, "");
  const FEAS_RENAME = { acceptable: "flawed" };   // these routes all contain major mistakes
  const feasLabel = (f) => FEAS_RENAME[String(f || "").toLowerCase()] || f;
  // Chemical-space ramp, built from the site's own tokens rather than plasma: accent-soft →
  // spruce (--accent). Monotonic in lightness, so "more" reads as "darker/denser" at a glance,
  // and it sits on the paper background instead of fighting it. Copper (--copper) is deliberately
  // NOT in the ramp — it stays reserved for the hovered node, so the highlight can never be
  // confused with a high value.
  const SA_RAMP = [[231,237,233],[168,192,180],[94,136,119],[31,77,63]];
  const ramp = (t) => {
    t = Math.max(0, Math.min(1, t));
    const x = t * (SA_RAMP.length - 1), i = Math.floor(x), f = x - i;
    const a = SA_RAMP[i], b = SA_RAMP[Math.min(i + 1, SA_RAMP.length - 1)];
    return `rgb(${Math.round(a[0]+(b[0]-a[0])*f)},${Math.round(a[1]+(b[1]-a[1])*f)},${Math.round(a[2]+(b[2]-a[2])*f)})`;
  };

  /* identity + ratings + comments live in data-access.js (window.SA) — the single storage
     seam. The active provider (localStorage prototype ↔ Supabase) is chosen by config.js. */
  const SA = window.SA;

  function starsHTML(key, size = "") {
    const { avg, n, mine } = SA.ratingSummary(key);
    const cur = mine || Math.round(avg);
    let s = `<span class="stars ${size}" data-rate="${esc(key)}">`;
    for (let i = 1; i <= 5; i++) s += `<span class="star ${i <= cur ? "on" : ""}" data-star="${i}">★</span>`;
    s += "</span>";
    s += avg ? `<span class="rate-summary">${avg.toFixed(1)} · ${n} vote${n>1?"s":""}</span>` : "";
    return s;
  }
  // Cytoscape renders to a live <canvas>, which prints blank. Rasterize the tree to a static PNG
  // and show that in the PDF instead (CSS hides the canvas whenever a .cy-print sibling exists).
  window.addEventListener("beforeprint", () => {
    closeFigModal?.();
    const el = document.getElementById("cy-tree-inline");
    if (!el || !inlineTreeCy) return;
    try {
      const uri = inlineTreeCy.png({ full: true, scale: 2, bg: "#ffffff" });
      let img = el.parentNode.querySelector(".cy-print");
      if (!img) { img = document.createElement("img"); img.className = "cy-print"; el.parentNode.insertBefore(img, el.nextSibling); }
      img.src = uri;
    } catch (e) {
      // fallback: at least resize + re-fit the canvas so it isn't clipped
      el.style.width = "680px"; el.style.height = "420px";
      inlineTreeCy.resize(); inlineTreeCy.fit(undefined, 8);
    }
  });
  window.addEventListener("afterprint", () => {
    const el = document.getElementById("cy-tree-inline");
    if (!el) return;
    el.parentNode.querySelector(".cy-print")?.remove();
    el.style.width = ""; el.style.height = "";
    if (inlineTreeCy) { inlineTreeCy.resize(); inlineTreeCy.fit(undefined, 8); }
  });

  // fullscreen route-figure modal
  let closeFigModal = null;
  let currentRouteDetail = null;   // set by viewRoute() — lets the tree's fullscreen button re-render at full detail
  function openFigureModal(box, html, onMount) {
    if (!box && !html) return;
    const overlay = document.createElement("div");
    overlay.className = "fig-modal";
    overlay.innerHTML = `<button class="fig-close" title="Close">✕</button><div class="fig-modal-body"></div>`;
    const bodyEl = overlay.querySelector(".fig-modal-body");
    if (html) {
      bodyEl.innerHTML = html;
    } else {
      const clone = box.cloneNode(true);
      clone.querySelector(".fig-expand")?.remove();
      bodyEl.appendChild(clone);
    }
    document.body.appendChild(overlay);
    document.body.style.overflow = "hidden";
    let modalCy = null;
    if (onMount) onMount(bodyEl).then((cy) => (modalCy = cy));
    const close = () => { modalCy?.destroy(); overlay.remove(); document.body.style.overflow = ""; closeFigModal = null; };
    overlay.querySelector(".fig-close").onclick = close;
    overlay.addEventListener("click", (e) => { if (e.target === overlay) close(); });
    document.addEventListener("keydown", function onKey(e) {
      if (e.key === "Escape") { close(); document.removeEventListener("keydown", onKey); }
    });
    closeFigModal = close;
  }
  document.addEventListener("click", (e) => {
    const btn = e.target.closest(".fig-expand"); if (!btn) return;
    if (btn.dataset.mode === "tree" && currentRouteDetail) {
      const d = currentRouteDetail;
      const html = `<div class="tree-detailed-wrap">
        <div class="ov-head"><span class="eyebrow">Full route</span>
          <span class="ov-hint">${esc(d.name || d.npaid || "")} · click a structure or step number to jump to that step · drag to pan, scroll to zoom</span></div>
        <div class="cy-tree detailed" id="cy-tree-modal"></div></div>`;
      openFigureModal(null, html, (bodyEl) => mountTree(bodyEl.querySelector("#cy-tree-modal"), d, { scale: 1, detailed: true }));
    } else {
      openFigureModal(btn.closest(".overview, .step-scheme"));
    }
  });

  // event delegation: overview node / arrow -> scroll to its step
  document.addEventListener("click", (e) => {
    const t = e.target.closest("[data-step]"); if (!t) return;
    closeFigModal?.();
    const el = document.getElementById("step-" + t.dataset.step); if (!el) return;
    el.scrollIntoView({ behavior: "smooth", block: "center" });
    el.classList.add("flash"); setTimeout(() => el.classList.remove("flash"), 1200);
  });
  // event delegation for star clicks
  document.addEventListener("click", (e) => {
    const star = e.target.closest(".star"); if (!star) return;
    const box = star.closest(".stars"); const key = box.dataset.rate;
    Promise.resolve(SA.rate(key, +star.dataset.star)).then((ok) => {
      if (!ok) return;
      // starsHTML returns TWO siblings (the widget + a vote-count span), so replacing the
      // widget's own outerHTML appends a second vote count every time it re-renders. Where a
      // .stars-slot wrapper exists, swap its contents instead so the pair is replaced as a unit.
      const size = box.classList.contains("sm") ? "sm" : "";
      const slot = box.closest(".stars-slot");
      if (slot) slot.innerHTML = starsHTML(key, size);
      else box.outerHTML = starsHTML(key, size);
      renderAuth();   // local mode may have just captured an identity via the name prompt
      toast(`Rating ${SA.saveMsg}`);
    }).catch((e) => toast(e.message || "Could not save rating"));
  });

  let toastT;
  function toast(msg) {
    let t = document.getElementById("toast");
    if (!t) { t = document.createElement("div"); t.id = "toast"; document.body.appendChild(t);
      t.style.cssText = "position:fixed;bottom:22px;left:50%;transform:translateX(-50%);background:#1A1E1C;color:#fff;padding:10px 18px;border-radius:10px;font-size:.86rem;z-index:200;opacity:0;transition:.2s"; }
    t.textContent = msg; t.style.opacity = "1";
    clearTimeout(toastT); toastT = setTimeout(() => (t.style.opacity = "0"), 1600);
  }

  /* ---------------- SMILES copy ---------------- */
  // navigator.clipboard needs a secure context (https or localhost); plain-http LAN access
  // falls back to the legacy textarea+execCommand path so copying works everywhere.
  function copyText(text, what = "SMILES") {
    const done = () => toast(what + " copied");
    const fail = () => toast("Copy failed");
    if (navigator.clipboard && window.isSecureContext) {
      navigator.clipboard.writeText(text).then(done, fail);
    } else {
      const ta = document.createElement("textarea");
      ta.value = text; ta.style.cssText = "position:fixed;opacity:0";
      document.body.appendChild(ta); ta.select();
      try { document.execCommand("copy") ? done() : fail(); } catch { fail(); }
      ta.remove();
    }
  }
  // any element carrying data-smiles copies it on click (buttons in step notes, cards, molecule page)
  document.addEventListener("click", (e) => {
    const b = e.target.closest("[data-smiles]"); if (!b) return;
    e.stopPropagation();
    copyText(b.dataset.smiles, b.dataset.smilesWhat || "SMILES");
  });
  // the same thing for text that isn't a SMILES (the About page's BibTeX entry)
  document.addEventListener("click", (e) => {
    const b = e.target.closest("[data-copy]"); if (!b) return;
    e.stopPropagation();
    copyText(b.dataset.copy, b.dataset.copyWhat || "Text");
  });
  // route page: clicking a structure in a step scheme copies that molecule's SMILES
  // (molecules.json is already fetched+cached by viewRoute, so jget resolves instantly)
  document.addEventListener("click", async (e) => {
    const img = e.target.closest(".step-scheme img[data-hash]"); if (!img) return;
    const mols = await jget("data/molecules.json").catch(() => null);
    const m = mols && mols[img.dataset.hash]; if (!m || !m.smiles) return;
    copyText(m.smiles, (m.name || m.formula || "molecule") + " SMILES");
  });
  // leaf-synthesis structures aren't in molecules.json (they're AiZynth-only intermediates);
  // their SMILES ride along in the leaf-route record, stashed here by leafRouteCard().
  let currentLeafMols = {};
  document.addEventListener("click", (e) => {
    const img = e.target.closest(".leaf-steps img[data-hash]"); if (!img) return;
    const m = currentLeafMols[img.dataset.hash]; if (!m || !m.s) return;
    copyText(m.s, "SMILES");
  });

  /* ---------------- reaction scheme ---------------- */
  function scheme(reactants, product, { cond = "", cls = "", mini = false, sizes = null } = {}) {
    // sizes: hash -> {w,h} from molecules.json; forces the shared MINI_SCALE so mini schemes
    // stay mutually consistent. Non-mini schemes pass no sizes -> each img at intrinsic size.
    const sizedImg = (h) => {
      const s = sizes && sizes[h];
      return s ? molImg(h, "", `width:${Math.round(s.w * MINI_SCALE)}px;height:${Math.round(s.h * MINI_SCALE)}px`) : molImg(h);
    };
    let left = "";
    reactants.forEach((h, i) => { left += `<div class="mol">${sizedImg(h)}</div>`; if (i < reactants.length - 1) left += `<span class="plus">+</span>`; });
    const inner = mini
      ? `<div class="arrow-wrap"><div class="arrow"></div></div>`
      : `<div class="arrow-wrap">${cond ? `<div class="cond">${esc(cond)}</div>` : ""}<div class="arrow"></div>${cls ? `<div class="rxn-class">${esc(cls)}</div>` : ""}</div>`;
    return `<div class="scheme${mini ? " mini" : ""}">${left}${inner}<div class="mol big">${sizedImg(product)}</div></div>`;
  }
  const clip = (s, n) => { s = String(s || ""); return s.length > n ? s.slice(0, n).replace(/\s+\S*$/, "") + "…" : s; };

  /* ---------------- LANDING ---------------- */
  async function viewHome() {
    const L = await jget("data/landing.json");
    const feat = await jget(`data/routes/${L.featured}.json`);
    const kstep = feat.steps.find((s) => s.is_key) || feat.steps[0];
    const fmt = (n) => (n || 0).toLocaleString();
    const topo = L.topology || {};
    const means = L.score_means || {};
    const mean = (k) => (means[k] != null ? means[k] : "—");
    app.innerHTML = `
      <header class="hero">
        <div class="wrap">
          <div class="hero-copy">
            <div class="eyebrow">Schwaller Group · EPFL</div>
            <!-- Three deliberate lines — the name, what the routes are, what they are for. Left to
                 wrap on its own the headline broke mid-phrase ("…synthesis / routes for natural
                 products"), which reads as an accident.
                 The middle line needs ~407px and a phone column gives ~311px, so below 520px it
                 splits at its own seam (AI-planned / total synthesis routes) rather than wrapping
                 wherever it runs out and orphaning "routes" on a line of its own. -->
            <h1><span class="hl">SynthAtlas:</span><span class="hl"><span
              class="hl-part">AI-planned</span> <span class="hl-part">total synthesis routes</span></span><span
              class="hl">for natural products</span></h1>
            <p class="lede">SynthAtlas is a database of AI-planned synthesis routes for natural products, built to
              aid synthetic chemists in brainstorming strategic disconnections for complex natural scaffolds.
              The dataset features:</p>
            <ul class="features">
              <li>${fmt(L.n_routes)} retrosynthetic routes for ${fmt(L.n_compounds)} natural products extracted from
                the <a class="ext" href="https://www.npatlas.org/" target="_blank" rel="noopener">Natural Products
                Atlas</a>.</li>
              <li>Up to three routes with distinct strategies for each target.</li>
              <li>Collection of strategic and elegant reactions - for example a
                <a class="ext" href="#/route/npa006547_s3/9">cascade Michael addition</a>, an
                <a class="ext" href="#/route/npa000776_s3/2">Ireland–Claisen rearrangement</a>, or a
                <a class="ext" href="#/route/npa023645_s2/3">Wolff rearrangement</a>.</li>
                <li>Reaction precedents are linked when available in Reaxys.</li>
            </ul>
            <div class="notice"><strong>These routes are AI-proposed and have not been experimentally
              validated. Treat each route as an idea of how a compound
              could be synthesized.</strong></div>
            <div class="cta"><a class="btn btn-primary" href="#/routes">Browse routes</a>
              <a class="btn btn-ghost" href="#/map">Chemical space map</a>
              <!-- Tinted rather than ghost: this is the one CTA that asks something of the reader
                   instead of pointing at another part of the atlas, and as a third plain ghost it
                   was indistinguishable from the map link beside it. -->
              <a class="btn btn-suggest" href="#/suggest">Suggest a target<svg viewBox="0 0 24 24"
                width="14" height="14" fill="none" stroke="currentColor" stroke-width="2.2"
                stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"
                ><path d="M5 12h14M13 6l6 6-6 6"/></svg></a></div>
            <!-- The evidence behind the notice above. A text link rather than a fourth button: the
                 CTA row is already three, and "Browse routes" should stay the loudest thing there.
                 The BibTeX entry lives on the About page — this only has to reach the paper. -->
            <p class="hero-paper">Read the preprint: <a class="ext"
              href="https://arxiv.org/abs/2608.07454" target="_blank" rel="noopener">Strategy-first
              synthesis planning for complex natural products</a>
              <span class="mono paper-id">arXiv:2608.07454</span></p>
          </div>
          <aside class="route-card-hero" onclick="location.hash='#/route/${feat.id}'">
            <div class="rc-head"><span>Example route</span></div>
            <div class="struct">${molImg(feat.target)}</div>
            <div class="rc-name">${esc(feat.name || "Natural product target")}</div>
            <div class="rc-id">${feat.npaid || feat.id}${feat.organism ? " · " + esc(feat.organism) : ""}</div>
            <div class="meta-badges" style="margin-top:12px">
              <span class="badge">${feat.total_steps} steps</span>
              <span class="badge">longest path ${feat.longest_linear_path}</span>
              <span class="badge alt">stereo: ${esc(feat.stereo_risk)}</span></div>
            ${scoreBar(feat.scores)}
          </aside>
        </div>
      </header>
      <section class="home-section"><div class="wrap">
        <h2 class="sec-h">Dataset summary</h2>
        <table class="summary-table">
          <tr><th>Target compounds</th><td>${fmt(L.n_compounds)} natural products (Natural Products Atlas)</td></tr>
          <tr><th>Retrosynthetic routes</th><td>${fmt(L.n_routes)} — up to three independent strategies per target</td></tr>
          <tr><th>Reaction steps</th><td>${fmt(L.n_reactions)}, each with proposed conditions and reaction class</td></tr>
          <tr><th>Unique molecules</th><td>${fmt(L.n_molecules)} targets and intermediates</td></tr>
          <tr><th>Route topology</th><td>${Object.keys(topo).sort((a, b) => topo[b] - topo[a]).map((t) => `${fmt(topo[t])} ${esc(t)}`).join(" · ")}</td></tr>
          <tr><th>Mean scores <span class="dim">(scale 1–9)</span></th><td>overall ${mean("overall")} · feasibility ${mean("feasibility")} · green ${mean("green")}</td></tr>
        </table>
      </div></section>
      <section class="home-section"><div class="wrap">
        <h2 class="sec-h">Browse the dataset</h2>
        <p class="sec-sub">Four views of the same data.</p>
        <div class="explore-grid">
          ${exploreCard(fmt(L.n_routes), "Routes", "Full retrosynthetic routes with per-step critic notes and scores.", "#/routes")}
          ${exploreCard(fmt(L.n_reactions), "Reactions", "All reaction steps, with proposed conditions and reaction class.", "#/reactions")}
          ${exploreCard(fmt(L.n_molecules), "Molecules", "All targets and intermediates.", "#/molecules")}
          ${exploreCard("TMAP", "Chemical space", "The target compounds arranged by Tanimoto similarity.", "#/map")}
        </div></div></section>
      <section class="home-section"><div class="wrap">
        <h2 class="sec-h">What each step records</h2>
        <p class="sec-sub">Every disconnection carries the model's strategic intent, the proposed reagents and
          conditions, the reaction class, and the critic's assessment — including the main risk it sees.
          The key step of the example route above:</p>
        <div class="step key">
          <div class="step-head"><span class="badge key">key step</span><span class="step-class">${esc(kstep.class)}</span></div>
          <div class="step-scheme">${scheme(kstep.reactants, kstep.product, { cond: kstep.conditions, cls: kstep.class })}</div>
          <div class="step-note"><div class="lbl">Why it's critical</div>${esc(kstep.why_critical || kstep.description)}</div>
          ${kstep.main_risk ? `<div class="step-note"><div class="lbl">Main risk flagged</div><span class="risk-note">${esc(kstep.main_risk)}</span></div>` : ""}
        </div>
        <p style="margin-top:16px"><a class="btn btn-ghost" href="#/route/${feat.id}">Open the full route →</a></p>
      </div></section>
      ${tmapSectionHTML(L.n_compounds)}`;
    // Canvas is mounted after the HTML lands; it fetches tmap.json + index.json itself, so the
    // rest of the landing page renders without waiting on them.
    mountTmap();
  }
  const exploreCard = (n, t, p, href) =>
    `<div class="explore-card" onclick="location.hash='${href}'"><div class="ec-n">${n}</div><h3>${t}</h3><p>${p}</p></div>`;
  // The site footer (with the EPFL logo + link and the Privacy Policy link) is static in
  // index.html so it appears on every route — see <footer class="foot"> there.

  /* ---------------- ROUTES gallery ---------------- */
  const RELIABLE_MIN = 7;   // default view: overall score >= this
  const routeState = { q: "", feas: "all", reliable: true, productive: true, sort: "favorites" };
  const CAP = 400;
  // Hand-picked elegant routes, surfaced by the "our favorites" sort (the default view).
  const FAVORITE_ROUTES = ["npa029946_s2", "npa035356_s1", "npa030041_s3", "npa006547_s3",
                           "npa032655_s3", "npa000108_s2", "npa020559_s1"];
  const scOf = (r, k) => (r.scores && r.scores[k] != null ? r.scores[k] : -1);
  async function viewRoutes() {
    const idx = await jget("data/index.json");
    app.innerHTML = `<div class="page"><div class="wrap">
      <div class="page-head"><div><h1>Routes</h1><div class="count" id="rcount"></div></div></div>
      <div class="toolbar">
        <label class="reliable-toggle"><input type="checkbox" id="rrel" ${routeState.reliable ? "checked" : ""}> High-confidence only <span class="hint2">overall ≥ ${RELIABLE_MIN}</span></label>
        <label class="reliable-toggle"><input type="checkbox" id="rprod" ${routeState.productive ? "checked" : ""}> Productive only <span class="hint2">small or purchasable starting materials</span></label>
        <input type="search" id="rq" placeholder="Search name, organism, or id…" value="${esc(routeState.q)}">
        <label>Feasibility <select id="rfeas">
          ${["all","excellent","good","acceptable","poor","infeasible"].map((f)=>`<option value="${f}" ${routeState.feas===f?"selected":""}>${f==="all"?"all":feasLabel(f)}</option>`).join("")}
        </select></label>
        <label>Sort <select id="rsort">
          <option value="favorites">our favorites ★</option>
          <option value="conv-desc">most convergent</option><option value="overall-desc">best overall</option><option value="feas-desc">most feasible</option>
          <option value="green-desc">greenest</option>
          <option value="steps-asc">fewest steps</option><option value="steps-desc">most steps</option>
          <option value="llp-desc">longest linear path</option><option value="id">route id</option>
        </select></label>
        <a class="info-i" href="#/glossary" title="How are the scores and metrics calculated? See the glossary">i</a>
      </div>
      <!-- Directly under the search, because that is where the thought occurs: you type a name,
           you do not see it, and the answer to "so what now?" should be right there rather than
           past a screen of other people's targets. Kept to one line for the same reason — it sits
           above the results, so it has to stay out of their way. -->
      <div class="suggest-strip">
        <div><b>Not finding your target?</b> Tell us what you would like SynthEx to plan next.</div>
        <a class="btn btn-suggest btn-sm" href="#/suggest">Suggest a target<svg viewBox="0 0 24 24"
          width="13" height="13" fill="none" stroke="currentColor" stroke-width="2.2"
          stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"
          ><path d="M5 12h14M13 6l6 6-6 6"/></svg></a>
      </div>
      <div class="grid routes" id="rgrid"></div></div></div>`;
    document.getElementById("rsort").value = routeState.sort;
    const draw = () => {
      let list = idx.slice();
      if (routeState.reliable) list = list.filter((r) => scOf(r, "overall") >= RELIABLE_MIN);
      // `!== false` so an index.json built before the flag existed keeps showing everything
      if (routeState.productive) list = list.filter((r) => r.productive !== false);
      const q = routeState.q.toLowerCase();
      const qmatch = (r) => !q || (r.id + " " + (r.name || "") + " " + (r.organism || "")).toLowerCase().includes(q);
      if (q) list = list.filter(qmatch);
      if (routeState.feas !== "all") list = list.filter((r) => String(r.feasibility).toLowerCase() === routeState.feas);
      const S = routeState.sort;
      list.sort((a, b) => S === "id" ? a.id.localeCompare(b.id)
        : S === "steps-asc" ? a.total_steps - b.total_steps
        : S === "steps-desc" ? b.total_steps - a.total_steps
        : S === "llp-desc" ? b.longest_linear_path - a.longest_linear_path
        : S === "feas-desc" ? scOf(b, "feasibility") - scOf(a, "feasibility")
        : S === "green-desc" ? scOf(b, "green") - scOf(a, "green")
        : S === "overall-desc" ? scOf(b, "overall") - scOf(a, "overall")
        // convergence ties heavily (most routes are 0 or 100) -> break ties by overall
        : scOf(b, "convergence") - scOf(a, "convergence") || scOf(b, "overall") - scOf(a, "overall"));
      // "Our favorites" sort: the hand-picked routes lead in curated order, everything else
      // follows in the default convergence order (the sort chain's final branch). Favorites
      // are pulled from the unfiltered index on purpose — under this sort they bypass the
      // feasibility filter and toggles — but an active search does apply to them: only
      // favorites matching the query stay pinned. Other sorts are pure.
      if (S === "favorites") {
        const favs = FAVORITE_ROUTES.map((id) => idx.find((r) => r.id === id)).filter(Boolean).filter(qmatch);
        if (favs.length) list = favs.concat(list.filter((r) => !FAVORITE_ROUTES.includes(r.id)));
      }
      const shown = list.slice(0, CAP);
      const capNote = list.length > CAP ? ` · showing top ${CAP}` : "";
      const activeTags = [routeState.reliable ? "high-confidence" : "", routeState.productive ? "productive" : ""].filter(Boolean).join(" ");
      document.getElementById("rcount").innerHTML = activeTags
        ? `${list.length} ${activeTags} routes${capNote} · <a href="#" id="rshowall">show all ${idx.length}</a>`
        : `${list.length} of ${idx.length} routes${capNote}`;
      document.getElementById("rgrid").innerHTML = shown.length ? shown.map(routeCard).join("")
        // No suggest link here on purpose — .suggest-strip sits directly below and says the same
        // thing with more weight. Two prompts a hundred pixels apart just read as clutter.
        : `<div class="empty">No routes match these filters.</div>`;
      const sa = document.getElementById("rshowall");
      if (sa) sa.onclick = (e) => { e.preventDefault();
        routeState.reliable = false; document.getElementById("rrel").checked = false;
        routeState.productive = false; document.getElementById("rprod").checked = false; draw(); };
    };
    const rq = document.getElementById("rq");
    rq.oninput = () => { routeState.q = rq.value; draw(); };
    document.getElementById("rrel").onchange = (e) => { routeState.reliable = e.target.checked; draw(); };
    document.getElementById("rprod").onchange = (e) => { routeState.productive = e.target.checked; draw(); };
    document.getElementById("rfeas").onchange = (e) => { routeState.feas = e.target.value; draw(); };
    document.getElementById("rsort").onchange = (e) => { routeState.sort = e.target.value; draw(); };
    draw();
  }
  // ⓘ links to the glossary; stopPropagation so it doesn't also open the card's route
  const glossaryI = `<a class="info-i" href="#/glossary" onclick="event.stopPropagation()" title="How are these calculated? See the glossary">i</a>`;
  const scoreBar = (s) => (s && s.overall != null)
    ? `<div class="scorebar">
        <span class="sc sc-o" title="Strategic quality (1–9)">${s.overall}<i>overall</i></span>
        <span class="sc sc-g" title="Green chemistry (1–9)">${s.green ?? "—"}<i>green</i></span>
        <span class="sc sc-f" title="Feasibility (1–10)">${s.feasibility ?? "—"}<i>feas.</i></span>
        <span class="sc sc-c" title="Convergence (0–100; 100 = ideally convergent tree, 0 = fully linear)">${s.convergence != null ? Math.round(s.convergence) : "—"}<i>conv.</i></span>
      </div>` : "";
  const routeCard = (r) => `<div class="card" onclick="location.hash='#/route/${r.id}'">
    ${FAVORITE_ROUTES.includes(r.id) ? `<div class="fav-star" title="One of our favorite routes">★</div>` : ""}
    <div class="thumb">${molImg(r.target)}</div>
    <div class="body">
      <div class="cid">${r.npaid || r.id}</div>
      <div class="ctitle">${esc(r.name || "Unnamed natural product")}</div>
      ${r.organism ? `<div class="org">${esc(r.organism)}</div>` : ""}
      ${scoreBar(r.scores)}
      <div class="meta-badges">
        <span class="badge gray">${r.total_steps} steps</span>
        <span class="badge ${feasClass(r.feasibility)}" style="background:#fff">${esc(feasLabel(r.feasibility))}</span>
      </div></div></div>`;

  /* ---------------- ROUTE detail ---------------- */
  async function viewRoute(id, fwdStep) {
    const d = await jget(`data/routes/${id}.json`).catch(() => null);
    if (!d) return notFound("route", id);
    // molecules.json is cached (jget) and needed by the tree anyway — here it lets the
    // linear strip pick each route's true starting material by heavy-atom count.
    const mols = await jget("data/molecules.json").catch(() => null);
    // leaf-synthesis index (hash -> {solved,n,longest}); null when the feature wasn't built.
    // Lets the building-block chips flag which leaves have an AiZynthFinder route.
    const lrIndex = await jget("data/leaf_routes.json").catch(() => null);
    currentRouteDetail = d;
    const rkey = "route:" + id;
    const total = d.steps.length;
    const fwd = d.steps.slice().sort((a, b) => b.idx - a.idx);   // forward: step 1 = first operation
    // Load shared ratings/comments for the route + each step before rendering. No-op in the
    // localStorage prototype; in Supabase mode this fills the caches starsHTML/engageBlock read.
    await SA.hydrate([rkey, ...fwd.map((s) => "rxn:" + id + "#" + s.idx)]);
    const steps = fwd.map((s) => stepBlock(s, total, id)).join("");
    const fnum = (i) => total - i;   // retro idx -> forward step number
    const stepLink = (idx) => idx != null
      ? ` <span class="step-jump" data-step="${idx}">→ jump to step ${fnum(idx)}</span>` : "";
    const risks = (d.top_risks || []).map((r) =>
      `<li><b>#${r.rank ?? "•"} ${esc(r.risk)}</b>${r.mitigation ? esc(r.mitigation) : ""}${stepLink(r.step_idx)}</li>`).join("");
    // sourcing category per building block (color-coded chip + tooltip). lrIndex may be null
    // when the leaf-routes feature wasn't built, in which case chips render uncoloured.
    const BB_CAT_LABEL = { buyable: "Commercially available (ZINC / eMolecules)",
      obtainable: "Short AiZynthFinder synthesis available",
      small: "Small — readily sourceable", unavailable: "Not readily available" };
    const blocks = (d.building_blocks || []).slice(0, 8).map((b) => {
      if (!b.hash) return "";
      const cat = (lrIndex && lrIndex[b.hash] && lrIndex[b.hash].cat) || "";
      const li = lrIndex && lrIndex[b.hash];
      const badge = cat === "obtainable"
        ? `<span class="bb-route" title="AiZynthFinder synthesis (${li.n} reaction${li.n > 1 ? "s" : ""})">⚗</span>` : "";
      const title = BB_CAT_LABEL[cat] || b.availability || "building block";
      return `<span class="bb-item${cat ? " bb-" + cat : ""}" onclick="location.hash='#/molecule/${b.hash}'" title="${esc(title)}">${molImg(b.hash)}${badge}</span>`;
    }).join("");
    // legend shows only the categories actually present among this route's building blocks
    const bbCatsPresent = new Set((d.building_blocks || []).slice(0, 8)
      .map((b) => lrIndex && b.hash && lrIndex[b.hash] && lrIndex[b.hash].cat).filter(Boolean));
    const BB_CAT_LEGEND = { buyable: "Buyable (ZINC / eMolecules)", obtainable: "AiZynthFinder route",
      small: "Small", unavailable: "Unavailable" };
    const bbLegend = bbCatsPresent.size
      ? `<div class="bb-legend">${["buyable", "obtainable", "small", "unavailable"].filter((c) => bbCatsPresent.has(c))
          .map((c) => `<span class="bb-leg bb-${c}">${BB_CAT_LEGEND[c]}</span>`).join("")}</div>`
      : "";
    const asym = (d.asymmetric || []).map((a) =>
      `<li><b>${esc(a.method || "")} — step ${fnum(a.step_index)}</b>${esc(a.catalyst || "")}${a.confidence ? ` · ${esc(a.confidence)}` : ""}${stepLink(a.step_index)}</li>`).join("");
    const casc = (d.cascade_opportunities || []).map((c) =>
      `<li><b>${esc(c.opportunity || "")}</b>${esc(c.suggested || "")}</li>`).join("");
    app.innerHTML = `<div class="page"><div class="wrap">
      <a class="back" href="#/routes">← All routes</a>
      <div class="print-only print-head">SynthAtlas — Agentic Synthesis Planning · Schwaller Group, EPFL</div>
      <div class="page-head"><div>
        <h1>${esc(d.name || "Unnamed natural product")}</h1>
        <div class="count">${d.npaid || id}${d.organism ? ` · <span class="org-inline">${esc(d.organism)}</span>${d.origin_type ? ` (${esc(d.origin_type)})` : ""}` : ""} · ${d.total_steps} steps · longest linear path ${d.longest_linear_path} · <span class="${feasClass(d.feasibility)}">${esc(feasLabel(d.feasibility))}</span></div></div>
        <div class="meta-badges" style="align-items:center">
          <button id="dl-pdf" class="btn btn-ghost btn-sm no-print">↓ Download PDF</button>
          ${d.solved ? '<span class="badge">✓ solved</span>' : ""}</div></div>
      ${routeSwitch(d)}
      <p class="ai-attrib no-print">The SynthEx AI agent suggested and critiqued this route.</p>
      ${d.steer_query ? `<div class="steer-note"><span class="lbl">SynthEx's strategy</span>${esc(d.steer_query)}</div>` : ""}
      ${routeFigure(d, mols)}
      ${d.summary ? `<div class="step-note"><div class="lbl">Route summary</div>${esc(d.summary)}</div>` : ""}
      ${d.route_score ? `<div class="step-note route-critic"><div class="lbl">Route critic</div>${esc(d.route_score)}</div>` : ""}
      <div class="detail">
        <div>
          ${engageBlock(rkey, "this route")}
          ${steps}
        </div>
        <div>
          ${d.scores && d.scores.overall != null ? `<div class="aside-card"><h3>Route score ${glossaryI}</h3>
            <div class="scorebar big">
              <span class="sc sc-o" title="Strategic quality (1–9)">${d.scores.overall}<i>overall</i></span>
              <span class="sc sc-g" title="Green chemistry (1–9)">${d.scores.green ?? "—"}<i>green</i></span>
              <span class="sc sc-f" title="Feasibility (1–10)">${d.scores.feasibility ?? "—"}<i>feas.</i></span>
            </div>
            ${d.score_rationale ? `<p class="aside-p">${esc(d.score_rationale)}</p>` : ""}</div>` : ""}
          <div class="aside-card"><h3>Route intelligence</h3>
            ${specRow("Total steps", d.total_steps)}
            ${specRow("Longest linear path", d.longest_linear_path)}
            ${specRow("Stereo risk", `<span class="${d.stereo_risk==='moderate'?'risk-note':''}">${esc(d.stereo_risk)}</span>`)}
            ${specRow("Feasibility", `<span class="${feasClass(d.feasibility)}">${esc(feasLabel(d.feasibility))}</span>`)}
            ${specRow("Redundant steps", d.redundant.length)}
          </div>
          ${blocks ? `<div class="aside-card"><h3>Building blocks <a class="info-i" href="#/glossary/building-block" title="Colour = how the block is sourced: catalogue availability (ZINC/eMolecules), a short AiZynthFinder synthesis, or small enough to source trivially — see the glossary">i</a></h3><div class="bb-row">${blocks}</div>${bbLegend}</div>` : ""}
          ${risks ? `<div class="aside-card"><h3>Top risks</h3><ul class="risk-list">${risks}</ul></div>` : ""}
          ${casc ? `<div class="aside-card"><h3>Cascade opportunities</h3><ul class="risk-list">${casc}</ul></div>` : ""}
          ${d.stereo_comment ? `<div class="aside-card"><h3>Stereochemistry</h3><p class="aside-p">${esc(d.stereo_comment)}</p></div>` : ""}
          ${asym ? `<div class="aside-card"><h3>Asymmetric methods</h3><ul class="risk-list">${asym}</ul></div>` : ""}
        </div>
      </div></div></div>`;
    bindEngage(rkey);
    mountInlineTree(d);
    bindPdfDownload(d);
    // deep link #/route/<id>/<n> — n is the forward (displayed) step number
    const n = parseInt(fwdStep, 10);
    if (n >= 1 && n <= total) {
      const el = document.getElementById("step-" + (total - n));
      if (el) {
        el.scrollIntoView({ behavior: "smooth", block: "center" });
        el.classList.add("flash"); setTimeout(() => el.classList.remove("flash"), 1200);
      }
    }
  }

  // Standalone, brand-consistent PDF built client-side from this route's JSON +
  // the already-published molecule SVGs (see pdf.js). Falls back to the browser's
  // window.print() if the PDF engine failed to load.
  function bindPdfDownload(d) {
    const btn = document.getElementById("dl-pdf");
    if (!btn) return;
    if (!window.SA_PDF || !window.PDFLib) { btn.onclick = () => window.print(); return; }
    btn.onclick = async () => {
      if (btn.dataset.busy) return;
      btn.dataset.busy = "1";
      const label = btn.textContent;
      btn.textContent = "Generating…";
      btn.disabled = true;
      try {
        await window.SA_PDF.downloadRoutePdf(d, { imgBase: IMG_BASE, assetVer: ASSET_VER });
      } catch (e) {
        console.error("PDF generation failed, falling back to print:", e);
        window.print();
      } finally {
        btn.textContent = label; btn.disabled = false; delete btn.dataset.busy;
      }
    };
  }

  // Dev helper (localhost only): build PDFs for n random routes and download each,
  // for eyeballing batches into gabriel/pdf_output/. Usage in console: SA.pdfSample(10)
  async function pdfSample(n = 10) {
    if (!window.SA_PDF) throw new Error("SA_PDF not loaded");
    const idx = await jget("data/index.json");
    const ids = idx.map((r) => r.id);
    const pick = [];
    for (let i = 0; i < Math.min(n, ids.length); i++) {
      pick.push(ids[Math.floor((i + 0.5) * ids.length / Math.min(n, ids.length))]);
    }
    for (const id of pick) {
      const d = await jget(`data/routes/${id}.json`);
      try { await window.SA_PDF.downloadRoutePdf(d, { imgBase: IMG_BASE, assetVer: ASSET_VER }); }
      catch (e) { console.error("pdfSample failed for", id, e); }
      await new Promise((r) => setTimeout(r, 400)); // let each download settle
    }
    console.log("pdfSample: generated", pick.length, "PDFs:", pick.join(", "));
  }
  if (SA) SA.pdfSample = pdfSample;   // console-reachable dev helper
  // Feasibility verdict rank (lower = better) — shared by the route-page strategy switcher
  // and the map click, so both agree on which sibling route is "best".
  const feasRank = (f) => ({ excellent: 0, good: 1, acceptable: 2, flawed: 2, poor: 3, infeasible: 4 })[String(f || "").toLowerCase()] ?? 5;
  // Best of a target's sibling routes, ranked like the route page's strategy switcher: feasibility
  // verdict, then overall score, then route id as the deterministic last resort (which is also the
  // fallback order for ids missing from the index). `idxById` maps route id -> data/index.json
  // entry. Shared by the t-map click and the suggest page's "we already have routes" notice so
  // every "take me to the route" link in the app lands on the same variant.
  const bestRouteId = (ids, idxById) => ids.slice().sort((a, b) => {
    const A = idxById[a], B = idxById[b];
    return (A ? feasRank(A.feasibility) : 6) - (B ? feasRank(B.feasibility) : 6)
      || ((B && B.scores && B.scores.overall != null ? B.scores.overall : -1)
        - (A && A.scores && A.scores.overall != null ? A.scores.overall : -1))
      || a.localeCompare(b);
  })[0];
  function routeSwitch(d) {
    const sibs = d.siblings || [];
    if (sibs.length < 2) return "";
    const rank = feasRank;
    const ov = (s) => (s.scores && s.scores.overall != null ? s.scores.overall : -1);
    const bySort = (a, b) => rank(a.feasibility) - rank(b.feasibility) || ov(b) - ov(a);
    const chip = (s) => `<a class="rs-chip ${s.id === d.id ? "on" : ""}" href="#/route/${s.id}">
      <b>${esc(s.variant)}</b> · ${s.total_steps} steps · <span class="${feasClass(s.feasibility)}">${esc(feasLabel(s.feasibility))}</span></a>`;
    // keep excellent/good routes inline (best rating first); collapse flawed/poor/infeasible
    // into "other routes" — matching the rating shown on the chip. The viewed route is
    // always shown, even if it's collapsed-tier.
    const keep = (s) => rank(s.feasibility) <= 1 || s.id === d.id;
    const primary = sibs.filter(keep).sort(bySort);
    const others = sibs.filter((s) => !keep(s)).sort(bySort);
    const more = others.length
      ? `<span class="rs-more"><button class="rs-more-btn" onclick="this.parentElement.classList.add('open')">+${others.length} other route${others.length > 1 ? "s" : ""}</button><span class="rs-more-list">${others.map(chip).join("")}</span></span>`
      : "";
    return `<div class="route-switch">
      <span class="rs-label">${sibs.length} routes to ${esc(d.name || "this compound")}:</span>${primary.map(chip).join("")}${more}</div>`;
  }
  // A route genuinely converges when at least one step unites two or more *synthesized*
  // intermediates. Building blocks join as leaves and don't count — child_idxs holds
  // exactly the non-building-block tree children captured at build time. Derived from the
  // real tree, so a route whose analysis mislabels it "linear" still shows its branch.
  function routeHasMerge(d) {
    return (d.steps || []).some((s) => (s.child_idxs || []).length >= 2);
  }
  function routeFigure(d, mols) {
    const labelledConvergent = d.topology === "convergent" || d.topology === "mixed";
    if ((labelledConvergent || routeHasMerge(d)) && buildTreeTopology(d)) {
      return `<div class="overview">
        <button class="fig-expand no-print" data-mode="tree" title="View full screen">⤢</button>
        <div class="ov-head"><span class="eyebrow">Full route</span>
          <span class="ov-hint">${d.total_steps} steps · fragments converge left → right · click a structure or step number to jump to that step</span></div>
        <div class="cy-tree" id="cy-tree-inline"></div></div>`;
    }
    return routeOverview(d, mols);
  }

  // linear routes: a wrapping JACS-style strip (step labels sit under the arrows)
  function routeOverview(d, mols) {
    const total = d.steps.length;
    const fwd = d.steps.slice().sort((a, b) => b.idx - a.idx).filter((s) => s.product);
    if (!fwd.length) return "";
    // Start node = the first step's MAIN precursor (most heavy atoms), not reactants[0] — the
    // data can list a reagent first (e.g. [H2O, HBr, geranyllinalool]), and showing water as
    // the route's "start" is nonsense. Ties/missing metadata keep the data's own order.
    // molecules.json can lag the route JSONs (partial rebuilds), so fall back to a heavy-atom
    // count derived from the route's own mol_key formulas ("C20H34O" -> 21).
    const byKey = {};
    (d.mol_key || []).forEach((m) => { if (m && m.hash) byKey[m.hash] = m.formula || ""; });
    const formulaHeavy = (f) => {
      let n = 0;
      for (const [, el, cnt] of String(f || "").matchAll(/([A-Z][a-z]?)(\d*)/g)) if (el !== "H") n += cnt ? +cnt : 1;
      return n;
    };
    const heavy = (h) => (mols && mols[h] && mols[h].heavy) || formulaHeavy(byKey[h]);
    const startHash = (fwd[0].reactants || []).slice().sort((a, b) => heavy(b) - heavy(a))[0];
    const nodes = [{ hash: startHash, role: "start" }].filter((n) => n.hash);
    fwd.forEach((s, k) => nodes.push({ hash: s.product, idx: s.idx, is_key: s.is_key, role: k === fwd.length - 1 ? "target" : "int" }));
    // ONE scale for the whole strip, so every structure shares the same bond length — the same
    // rule mountTree() applies to the convergent tree. Sizing by a CSS max-width clamp instead
    // gives each molecule its own scale (min(1, cell/naturalWidth)), which draws the *same* ring
    // at two different sizes in one figure: molecules narrower than the cell stay at 1.0 while
    // wider ones shrink. That is what shipped, and it is worst for structures carrying collapsed
    // abbreviations (TBS, TBDPS) — wide text labels for very little chemistry.
    //
    // Scale is per ROUTE, not per row. Per-row scaling would put the same molecule at two sizes
    // either side of a wrap, which is the same bug with a bigger blast radius.
    const OV_CELL = 96;        // px budget for a typical cell — the old fixed .ov-node width
    const OV_MIN_SCALE = 0.45; // one unusually wide molecule must not shrink the other sixteen to
                               // illegibility; it takes a wider cell and the flex row wraps.
    // Nothing here caps the cell against the viewport, deliberately. The widest published molecule
    // is 720px, which at the floor is a ~324px cell — wider than a 320px phone has to spare, on 6
    // of 724 real linear routes. Those are handled by `overflow-x` on .overview-inner (the figure
    // scrolls inside its own card) rather than by a viewport-derived cap here, which would mean
    // duplicating .wrap/.overview/.ov-node padding values in JS and keeping them in sync by hand.
    const natW = (h) => (mols && mols[h] && mols[h].w) || 0;
    const widest = Math.max(0, ...nodes.map((n) => natW(n.hash)));
    // Never upscale past intrinsic size: every SVG is rendered at one fixed bond length, so
    // blowing a tiny molecule (H2O, HBr) up to the cell width dwarfs its neighbours' labels.
    const ovScale = widest ? Math.max(OV_MIN_SCALE, Math.min(1, OV_CELL / widest)) : 1;
    const units = nodes.map((n, i) => {
      const next = nodes[i + 1];
      const keyInt = !!n.is_key && n.role === "int";               // product of a key step (not the target)
      const ncls = (n.role === "target" ? "target " : "") + (keyInt ? "rx-key" : "");
      const tag = n.role === "start" ? `<span class="ov-tag start">start</span>`
        : n.role === "target" ? `<span class="ov-tag target">target</span>`
        : keyInt ? `<span class="ov-tag keyint">key intermediate</span>` : "";
      const nstep = n.idx != null ? n.idx : (fwd[0] && fwd[0].idx);   // start node -> first forward step
      const nclick = nstep != null ? `data-step="${nstep}" title="Jump to step ${total - nstep}"` : "";
      let arrow = "";
      if (next) {
        const num = total - next.idx;
        arrow = `<div class="ov-arrow${next.is_key ? " key" : ""}" data-step="${next.idx}" title="Jump to step ${num}"><span class="line"></span><span class="ov-step">${num}</span></div>`;
      }
      // molecules.json can lag the route JSONs (partial rebuilds). Without w/h there is no honest
      // size to set, so that ONE image keeps the old cell clamp rather than guessing a size.
      const m = mols && mols[n.hash];
      const size = m && m.w
        ? `width:${(m.w * ovScale).toFixed(1)}px;height:${(m.h * ovScale).toFixed(1)}px`
        : `max-width:${OV_CELL}px`;
      return `<div class="ov-unit"><div class="ov-node ${ncls}" ${nclick}>${molImg(n.hash, "", size)}${tag}</div>${arrow}</div>`;
    }).join("");
    return `<div class="overview">
      <button class="fig-expand no-print" title="View full screen">⤢</button>
      <div class="ov-head"><span class="eyebrow">Full route</span>
        <span class="ov-hint">${fwd.length} step${fwd.length === 1 ? "" : "s · key intermediates highlighted"} · click a structure or step number to jump to that step</span></div>
      <div class="overview-inner">${units}</div></div>`;
  }

  // ---------------- tree topology (shared) — pure data, dagre computes the actual pixel layout ----------------
  function buildTreeTopology(d) {
    const byProd = {};
    d.steps.forEach((s) => { if (s.product) byProd[s.product] = { reactants: s.reactants || [], idx: s.idx, is_key: s.is_key }; });
    if (!d.target || !byProd[d.target]) return null;
    const seen = new Set();
    let counter = 0;
    const build = (hash) => {
      const st = byProd[hash];
      const leaf = !st || seen.has(hash);
      const node = { id: "n" + (++counter), hash, idx: leaf ? null : st.idx, is_key: leaf ? false : st.is_key, leaf, children: [] };
      if (!leaf) { seen.add(hash); node.children = st.reactants.map(build); }
      return node;
    };
    const root = build(d.target);
    let maxDepth = 0;
    const setDepth = (n, dep) => { n.depth = dep; maxDepth = Math.max(maxDepth, dep); n.children.forEach((c) => setDepth(c, dep + 1)); };
    setDepth(root, 0);
    if (maxDepth < 1) return null;   // not actually branching — caller falls back to the linear strip
    const nodes = [], edges = [];
    let num = 0;
    (function collect(n) {
      n.num = ++num; nodes.push(n);
      n.children.forEach((c) => { edges.push({ source: c.id, target: n.id, idx: n.idx, is_key: n.is_key }); collect(c); });
    })(root);
    return { nodes, edges };
  }

  // renders a tree topology into `container` as a Cytoscape + dagre graph — real per-molecule node sizes
  // (from molecules.json, captured at build time via RDKit flexicanvas) instead of one uniform box for every node
  async function mountTree(container, d, { scale, detailed }) {
    const topo = buildTreeTopology(d);
    if (!topo || !container) return null;
    const total = d.steps.length;
    const mols = await jget("data/molecules.json");
    // deliberately tiny floors — just anti-degenerate, not a real minimum size, so relative
    // proportions stay honest (a floor here was previously inflating small molecules ~6-9x)
    const MIN_W = detailed ? 46 : 24, MIN_H = detailed ? 32 : 16;
    const elements = [];
    topo.nodes.forEach((n) => {
      const m = mols[n.hash] || { w: 130, h: 100 };
      const w = Math.max(MIN_W, m.w * scale), h = Math.max(MIN_H, m.h * scale);
      const role = n.depth === 0 ? "target" : n.leaf ? "start" : n.is_key ? "key" : "int";
      elements.push({ group: "nodes", data: { id: n.id, hash: n.hash, w, h, num: n.num, idx: n.idx, role } });
    });
    // one reaction can have several reactant edges converging on the same product — show the
    // step number once (on the last/central branch into that product), not on every branch.
    const lastEdgeForTarget = {};
    topo.edges.forEach((e, i) => { if (e.idx != null) lastEdgeForTarget[e.target] = i; });
    topo.edges.forEach((e, i) => {
      const showNo = e.idx != null && lastEdgeForTarget[e.target] === i;
      elements.push({ group: "edges", data: { id: "e" + i, source: e.source, target: e.target, idx: e.idx,
                                               stepno: showNo ? total - e.idx : "" } });
    });
    const cy = window.cytoscape({
      container,
      elements,
      style: [
        { selector: "node", style: {
            shape: "round-rectangle", width: "data(w)", height: "data(h)",
            "background-color": "#fff", "background-opacity": 1,
            "background-image": (ele) => `${IMG_BASE}/${ele.data("hash")}.svg?v=${ASSET_VER}`,
            "background-fit": "contain", "background-clip": "none", "background-image-crossorigin": "anonymous",
            "border-width": 1.3, "border-color": "#E3DED3",
            label: "",
          } },
        { selector: "node[role='target']", style: { "border-color": "#C7D6CE", "border-width": 2.2, "background-color": "#F3F7F4" } },
        { selector: "node[role='key']", style: { "border-color": "#E7D6C4", "border-width": 2.2, "background-color": "#FCF8F2" } },
        { selector: "node[role='start']", style: { "border-style": "dashed" } },
        { selector: "edge", style: {
            "curve-style": "taxi", "taxi-direction": "horizontal", "taxi-turn": "50%", "taxi-turn-min-distance": 12,
            width: 1.8, "line-color": "#A9662E", "target-arrow-color": "#A9662E", "target-arrow-shape": "triangle", "arrow-scale": 1.05,
            label: "data(stepno)", "font-family": "SF Mono, ui-monospace, monospace", "font-size": detailed ? 12 : 10,
            "font-weight": 700, color: "#8A6320", "text-background-color": "#F7F5F0", "text-background-opacity": 1,
            "text-background-padding": 2, "text-background-shape": "roundrectangle",
          } },
      ],
      layout: { name: "null" },   // laid out explicitly below, after listeners are attached — see note
      userZoomingEnabled: detailed, userPanningEnabled: detailed, boxSelectionEnabled: false,
      autoungrabify: true, wheelSensitivity: 0.25,
    });
    cy.on("tap", "node", (evt) => {
      const idx = evt.target.data("idx");
      if (idx == null) { location.hash = "#/molecule/" + evt.target.data("hash"); return; }   // building block: no producing step
      closeFigModal?.();
      const el = document.getElementById("step-" + idx); if (!el) return;
      el.scrollIntoView({ behavior: "smooth", block: "center" });
      el.classList.add("flash"); setTimeout(() => el.classList.remove("flash"), 1200);
    });
    cy.on("tap", "edge", (evt) => {
      const idx = evt.target.data("idx"); if (idx == null) return;
      closeFigModal?.();
      const el = document.getElementById("step-" + idx); if (!el) return;
      el.scrollIntoView({ behavior: "smooth", block: "center" });
      el.classList.add("flash"); setTimeout(() => el.classList.remove("flash"), 1200);
    });
    // size the container to the graph's own aspect ratio (bounded) instead of a fixed height —
    // a short, wide tree should fill the available width at a much bigger zoom than a tall one.
    const pad = detailed ? 40 : 26;
    cy.one("layoutstop", () => {
      const bb = cy.elements().boundingBox();
      const availW = container.clientWidth || container.offsetWidth || 900;
      const minH = detailed ? 320 : 220;
      const maxH = detailed ? window.innerHeight * 0.86 - 40 : 640;
      const targetH = Math.max(minH, Math.min(maxH, availW * (bb.h / bb.w) + pad));
      container.style.height = targetH + "px";
      cy.resize();
      const w = container.clientWidth, h = container.clientHeight;
      const zoomByW = (w - 2 * pad) / (bb.x2 - bb.x1);
      const zoomByH = (h - 2 * pad) / (bb.y2 - bb.y1);
      // in the fullscreen view, panning/zooming is available — so don't shrink wide/tall routes past
      // legibility just to guarantee everything is visible at once; floor the zoom and let people pan
      const minZoom = detailed ? 0.5 : 0;
      const zoom = Math.max(Math.min(zoomByW, zoomByH), minZoom);
      // bias the pan to bring the TARGET node closer to vertical centre (it can land anywhere within
      // its rank if branches are lopsided) — but only by however much slack space is left over once
      // the bounding box is fit at this zoom. Centering the target exactly and never wasting space
      // are mutually exclusive for a genuinely lopsided tree, so this caps the correction rather than
      // forcing full centering (which — tried it — pushes everything to one half and leaves the rest
      // of the container empty). No slack (a tall/narrow tree) means no correction: falls back to a
      // plain centred fit.
      const target = cy.nodes('[role="target"]').first();
      const ty = target.position().y;
      const bbCenterY = (bb.y1 + bb.y2) / 2;
      const slack = Math.max(0, (h - 2 * pad) - zoom * (bb.y2 - bb.y1));
      const desiredShift = zoom * (bbCenterY - ty);
      const appliedShift = Math.max(-slack / 2, Math.min(slack / 2, desiredShift));
      cy.zoom(zoom);
      cy.pan({ x: w / 2 - zoom * (bb.x1 + bb.x2) / 2, y: (h / 2 - zoom * bbCenterY) + appliedShift });
    });
    // layout is run explicitly (not via the constructor's `layout` option) so the listener above,
    // registered after construction, is guaranteed to be attached before "layoutstop" can fire
    cy.layout({
      name: "dagre", rankDir: "LR",
      nodeSep: detailed ? 26 : 14, rankSep: detailed ? 100 : 48, edgeSep: 10,
      nodeDimensionsIncludeLabels: true,
    }).run();
    return cy;
  }

  let inlineTreeCy = null;
  async function mountInlineTree(d) {
    const el = document.getElementById("cy-tree-inline"); if (!el) return;
    inlineTreeCy?.destroy();
    inlineTreeCy = await mountTree(el, d, { scale: 0.78, detailed: false });
  }
  window.addEventListener("resize", () => { inlineTreeCy?.resize(); inlineTreeCy?.fit(undefined, 10); });
  const specRow = (k, v) => `<div class="spec-row"><span>${k}</span><span class="v">${v}</span></div>`;
  const REASON_THRESHOLD = 200;
  // Two-tier Reaxys precedents, rendered as a pill/chip. "close" = same transformation
  // (strong), "related" = same reaction type (softer). The count is the trust signal
  // (">10" past the cap); the link opens the closest IDs and resolves only for subscribers.
  // `band` picks close first, falling back to related when there is no close precedent.
  function reaxysBand(closeCount, closeUrl, relCount, relUrl) {
    if (closeCount) return { n: closeCount, u: closeUrl, cls: "", noun: "close precedent",
                             tip: "Signature match at radius 2 — the more specific of the two bands" };
    if (relCount)   return { n: relCount, u: relUrl, cls: " related", noun: "related report",
                             tip: "Signature match at radius 1 — broader, so weaker evidence" };
    return null;
  }
  // The count is the number of independent DOCUMENTS reporting the transformation, not a number of
  // reactions — that is the claim worth making, since repeated reports are evidence a step is
  // established while repeated reactions may just be a common pattern. Shown exactly up to ten and
  // as ">10" beyond: past that the precise figure stops informing a decision, and many matches run
  // to hundreds. The tooltip still carries the exact number for anyone who wants it.
  const MANY_DOCS = 10;
  function reaxysChipHTML(b, { full } = {}) {
    const n = b.n > MANY_DOCS ? ">10" : b.n;
    // Reactions browser: compact "7 in Reaxys". Route step: full "7 documents · close precedent".
    const label = full
      ? `${n} document${b.n === 1 ? "" : "s"} · ${b.noun} in Reaxys →`
      : `${n} in Reaxys →`;
    const title = `${b.tip}. Reported in ${b.n} document${b.n === 1 ? "" : "s"}; `
      + `the link opens up to 20 example reactions in Reaxys (subscription required).`;
    return `<a class="reaxys-chip${b.cls}" href="${esc(b.u)}" target="_blank" rel="noopener"`
      + ` title="${esc(title)}">${label}</a>`;
  }
  // Route step note: labelled chip.
  function reaxysNote(closeCount, closeUrl, relCount, relUrl) {
    const b = reaxysBand(closeCount, closeUrl, relCount, relUrl);
    if (!b) return "";
    return `<div class="step-note reaxys"><div class="lbl">Literature precedent</div>${reaxysChipHTML(b, { full: true })}</div>`;
  }
  // Compact chip for the Reactions browser cards.
  function reaxysChip(r) {
    const b = reaxysBand(r.reaxys_close_count, r.reaxys_close_url, r.reaxys_related_count, r.reaxys_related_url);
    return b ? reaxysChipHTML(b) : "";
  }
  function stepBlock(s, total, rid) {
    const rk = "rxn:" + rid + "#" + s.idx;
    const num = total - s.idx;   // forward step number (1 = first operation)
    // accepted steps carry no badge — acceptance is the norm, only a flag is worth surfacing
    const verdict = s.critic_verdict === false ? `<div class="verdict bad"><span class="tick">!</span> Critic flagged this step</div>` : "";
    const long = (s.strategy || "").length > REASON_THRESHOLD;
    let reasoning;
    if (s.strategy) {
      if (long) {
        // collapsed: "…"-clipped preview + "read more"; open: the preview is hidden and the
        // full text shows as one paragraph, ending in an explicit "read less" toggle
        const preview = clip(s.strategy, REASON_THRESHOLD);
        reasoning = `<details class="reason-details disclosure"><summary>${esc(preview)}</summary><p>${esc(s.strategy)} <button type="button" class="rd-less no-print" onclick="this.closest('details').removeAttribute('open')">▴ read less</button></p></details>`;
      } else {
        reasoning = esc(s.strategy);
      }
    } else {
      reasoning = s.description ? esc(s.description) : "";
    }
    const reasonLbl = s.strategy ? "Description" : "What happens";
    return `<div class="step ${s.is_key ? "key" : ""}" id="step-${s.idx}">
      <div class="step-head"><span class="step-class">Step ${num}</span>
        ${s.is_key ? '<span class="badge key">key step</span>' : ""}
        ${s.asym_method ? `<span class="badge alt">${esc(s.asym_method)}</span>` : ""}
        <span style="margin-left:auto">${starsHTML(rk, "sm")}</span></div>
      <div class="step-scheme">
        <button class="fig-expand no-print" title="View full screen">⤢</button>
        ${scheme(s.reactants, s.product, { cond: s.conditions, cls: "" })}
      </div>
      ${verdict}
      ${reasoning ? `<div class="step-note"><div class="lbl">${reasonLbl}</div>${reasoning}</div>` : ""}
      ${s.critic_reason ? `<div class="step-note"><div class="lbl">Critic assessment</div>${esc(s.critic_reason)}</div>` : ""}
      ${s.why_critical ? `<div class="step-note"><div class="lbl">Why it's critical</div>${esc(s.why_critical)}</div>` : ""}
      ${s.main_risk ? `<div class="step-note"><div class="lbl">Main risk</div><span class="risk-note">${esc(s.main_risk)}</span></div>` : ""}
      ${s.asym_method ? `<div class="step-note"><div class="lbl">Asymmetric method</div>${esc(s.asym_method)}${s.asym_catalyst ? ` · <span class="mono">${esc(s.asym_catalyst)}</span>` : ""}</div>` : ""}
      ${s.conditions ? `<div class="step-note"><div class="lbl">Conditions${s.assessment ? ` · ${esc(s.assessment)}` : ""}</div><span class="cond-chip">${esc(s.conditions)}</span></div>` : ""}
      ${reaxysNote(s.reaxys_close_count, s.reaxys_close_url, s.reaxys_related_count, s.reaxys_related_url)}
      ${s.incompatibility ? `<div class="step-note"><div class="lbl">Compatibility note</div><span class="risk-note">${esc(s.incompatibility)}</span></div>` : ""}
      ${s.rxn_smiles ? `<details class="step-details no-print"><summary>Reaction SMILES</summary><p class="smiles-p"><code>${esc(s.rxn_smiles)}</code><button type="button" class="smiles-chip" data-smiles="${esc(s.rxn_smiles)}" data-smiles-what="Reaction SMILES">⧉ Copy</button></p></details>` : ""}
      ${s.ops_analysis ? `<details class="step-details disclosure"><summary>Mechanistic analysis</summary><p>${esc(s.ops_analysis)}</p></details>` : ""}
    </div>`;
  }

  /* ---------------- engagement block ---------------- */
  function engageBlock(key, label) {
    const cs = SA.comments(key);
    const me = SA.identity();
    const note = SA.mode === "local"
      ? "Prototype: ratings &amp; comments are saved in your browser. The next version adds email-confirmed sign-in and shared, tracked feedback."
      : "Ratings &amp; comments are shared with the team and attributed to your confirmed account.";
    return `<div class="engage" id="engage">
      <div class="note">${note}</div>
      <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap"><b>Rate ${label}:</b> ${starsHTML(key)}</div>
      <div class="cmt-form">
        <textarea id="cmtText" placeholder="Leave a comment for the team…"></textarea>
        <div class="cmt-actions">
          <button class="btn btn-primary btn-sm" id="cmtBtn">Post comment</button>
          <label class="cmt-private" title="Only you and the SynthAtlas team see private comments.">
            <input type="checkbox" id="cmtPrivate"> Private</label>
          <span class="rate-summary" id="cmtWho">${me ? "as " + esc(me.name) : (SA.mode === "supabase" ? "sign in to comment" : "you'll be asked for your name")}</span>
        </div>
      </div>
      ${commentsSection(cs, key)}
    </div>`;
  }
  // comment list, collapsed by default behind a "N comments" toggle. data-key lets the edit
  // handler know which target to re-hydrate.
  function commentsSection(cs, key) {
    const n = cs.length;
    const summary = n ? `${n} comment${n !== 1 ? "s" : ""}` : "No comments yet";
    return `<details class="cmt-collapse" id="cmtDetails">
      <summary>${summary}</summary>
      <div class="cmt-list" id="cmtList" data-key="${esc(key)}">${cs.map(cmtHTML).join("")}</div>
    </details>`;
  }
  const cmtHTML = (c) => `<div class="cmt${c.is_public === false ? " private" : ""}" data-id="${esc(c.id)}">
    <span class="who">${esc(c.by)}</span> <span class="when">${new Date(c.at).toLocaleString()}</span>${c.is_public === false ? ' <span class="cmt-badge">private</span>' : ""}${c.edited ? ' <span class="cmt-edited">edited</span>' : ""}${c.mine ? ` <button class="cmt-edit" data-id="${esc(c.id)}" data-public="${c.is_public !== false}">Edit</button>` : ""}
    <div class="txt">${esc(c.text)}</div></div>`;
  function bindEngage(key) {
    const btn = document.getElementById("cmtBtn"); if (!btn) return;
    btn.onclick = () => {
      const ta = document.getElementById("cmtText"); const t = ta.value.trim(); if (!t) return;
      const isPublic = !(document.getElementById("cmtPrivate") || {}).checked;
      Promise.resolve(SA.addComment(key, t, isPublic)).then((arr) => {
        if (!arr) return;
        ta.value = "";
        const cp = document.getElementById("cmtPrivate"); if (cp) cp.checked = false;
        const details = document.getElementById("cmtDetails");
        if (details) { details.outerHTML = commentsSection(arr, key); document.getElementById("cmtDetails").open = true; }
        const me = SA.identity(); if (me) document.getElementById("cmtWho").textContent = "as " + me.name;
        renderAuth();
        toast(`Comment ${SA.saveMsg}`);
      }).catch((e) => toast(e.message || "Could not post comment"));
    };
  }
  // inline edit of your own comment (delegated so it survives list re-renders)
  document.addEventListener("click", (e) => {
    const eb = e.target.closest(".cmt-edit"); if (!eb) return;
    const cmt = eb.closest(".cmt"); if (cmt.querySelector(".cmt-editor")) return;
    const list = eb.closest(".cmt-list"); const key = list && list.dataset.key; const id = eb.dataset.id;
    const txt = cmt.querySelector(".txt"); const isPub = eb.dataset.public === "true";
    txt.style.display = "none";
    const ed = document.createElement("div");
    ed.className = "cmt-editor";
    ed.innerHTML = `<textarea class="cmt-edit-text"></textarea>
      <div class="cmt-edit-actions">
        <button class="btn btn-primary btn-sm cmt-save">Save</button>
        <button class="btn btn-ghost btn-sm cmt-cancel">Cancel</button>
        <label class="cmt-private"><input type="checkbox" class="cmt-edit-private"${isPub ? "" : " checked"}> Private</label>
      </div>`;
    ed.querySelector(".cmt-edit-text").value = txt.textContent;
    cmt.appendChild(ed);
    ed.querySelector(".cmt-edit-text").focus();
    ed.querySelector(".cmt-cancel").onclick = () => { ed.remove(); txt.style.display = ""; };
    ed.querySelector(".cmt-save").onclick = () => {
      const nt = ed.querySelector(".cmt-edit-text").value.trim(); if (!nt) return;
      const pub = !ed.querySelector(".cmt-edit-private").checked;
      Promise.resolve(SA.editComment(key, id, nt, pub)).then((arr) => {
        const details = document.getElementById("cmtDetails");
        if (details && arr) { details.outerHTML = commentsSection(arr, key); document.getElementById("cmtDetails").open = true; }
        toast(`Comment ${SA.saveMsg}`);
      }).catch((err) => toast(err.message || "Could not save edit"));
    };
  });

  /* ---------------- REACTIONS ---------------- */
  const rxnState = { q: "", key: false, sort: "favorites" };
  // Hand-picked elegant reactions, as [route id, forward step number] (the "step N" shown
  // on the card). Surfaced by the Reactions browser's "our favorites" sort, same mechanism
  // as FAVORITE_ROUTES on the Routes page.
  const FAVORITE_REACTIONS = [
    ["npa029946_s2", 11], ["npa035356_s1", 8], ["npa030041_s3", 1], ["npa006547_s3", 9],
    ["npa032655_s3", 9], ["npa020559_s1", 6], ["npa009851_s3", 2], ["npa000336_s1", 3],
    ["npa000656_s2", 2],
  ];
  const FAV_RXN_KEYS = new Set(FAVORITE_REACTIONS.map(([rt, n]) => rt + "@" + n));
  const isFavRxn = (r) => FAV_RXN_KEYS.has(r.route + "@" + r.step_no);
  async function viewReactions() {
    const [rx, mols] = await Promise.all([jget("data/reactions.json"), jget("data/molecules.json")]);
    // Suggested searches: the most common extracted class labels, minus the extraction
    // fallback ("step") and "global deprotection" (a noisy near-duplicate of "deprotection").
    const HIDDEN_TAGS = new Set(["step", "global deprotection"]);
    const counts = {};
    rx.forEach((r) => (counts[r.class] = (counts[r.class] || 0) + 1));
    const topClasses = Object.entries(counts).filter(([c]) => !HIDDEN_TAGS.has(c))
      .sort((a, b) => b[1] - a[1]).slice(0, 8).map((e) => e[0]);
    app.innerHTML = `<div class="page"><div class="wrap">
      <div class="page-head"><div><h1>Reactions</h1><div class="count" id="xcount"></div></div></div>
      <div class="toolbar">
        <input type="search" id="xq" placeholder="Search reactions…" value="${esc(rxnState.q)}">
        <label><input type="checkbox" id="xkey" ${rxnState.key ? "checked" : ""}> key steps only</label>
        <label>Sort <select id="xsort">
          <option value="favorites">our favorites ★</option>
          <option value="route">by route</option>
        </select></label>
      </div>
      <div class="tags" id="xtags" style="margin:-12px 0 22px">
        <span class="tag" data-c="">all</span>
        <span class="tag" data-c="cascade">cascade</span>
        ${topClasses.map((c)=>`<span class="tag" data-c="${esc(c)}">${esc(c)}</span>`).join("")}
      </div>
      <div class="grid rxns" id="xgrid"></div></div></div>`;
    // Generation counter: a hydrate that resolves after the query moved on must not repaint the
    // grid with stale results.
    let drawGen = 0;
    const draw = () => {
      const gen = ++drawGen;
      let list = rx.slice();
      if (rxnState.q) { const q = rxnState.q.toLowerCase();
        list = list.filter((r) => (r.class + " " + r.conditions + " " + (r.prose || "")).toLowerCase().includes(q)); }
      if (rxnState.key) list = list.filter((r) => r.is_key);
      // base order: by route, then forward step number (1 = first operation) so cards read
      // 1→N, matching the route page (step_no = total - idx; ascending step_no = descending idx).
      list.sort((a, b) => a.route.localeCompare(b.route) || ((a.step_no ?? -a.idx) - (b.step_no ?? -b.idx)));
      // "Our favorites" sort: hand-picked reactions lead in curated order (pulled from the
      // full list, so they bypass the key-step toggle) — but only while the search box is
      // empty; any query suppresses the pins entirely. "by route" is the pure base order.
      if (rxnState.sort === "favorites" && !rxnState.q) {
        const favs = FAVORITE_REACTIONS.map(([rt, n]) => rx.find((r) => r.route === rt && r.step_no === n)).filter(Boolean);
        if (favs.length) list = favs.concat(list.filter((r) => !isFavRxn(r)));
      }
      document.getElementById("xcount").textContent = `${list.length} of ${rx.length} reactions`;
      // a tag lights up when the search box holds exactly its text ("all" when empty)
      const ql = rxnState.q.trim().toLowerCase();
      document.querySelectorAll("#xtags .tag").forEach((x) => x.classList.toggle("on", (x.dataset.c || "").toLowerCase() === ql));
      const shown = list.slice(0, 300);
      // Paint immediately from whatever ratings are already cached — hydrate is NOT on the
      // critical path. It used to be awaited here, which put three Supabase queries over 300
      // target keys in front of every repaint (and, before the debounce below, of every
      // keystroke): slow for the user and a self-inflicted load spike at launch traffic.
      const paint = () => {
        document.getElementById("xgrid").innerHTML = shown.length ? shown.map((r) => rxnCard(r, mols)).join("")
          : `<div class="empty">No reactions match.</div>`;
      };
      paint();
      // Then fill in shared ratings, patching ONLY the star widgets. Repainting all 300 cards
      // would flicker; each card carries a .stars-slot whose contents we can swap instead.
      const keys = shown.map((r) => "rxn:" + r.id);
      if (keys.length) {
        Promise.resolve(SA.hydrate(keys))
          .then(() => {
            if (gen !== drawGen) return;              // query moved on — stale result
            document.querySelectorAll("#xgrid .stars-slot").forEach((slot) => {
              const key = slot.querySelector(".stars")?.dataset.rate;
              if (key) slot.innerHTML = starsHTML(key, "sm");
            });
          })
          .catch(() => { /* ratings unavailable — the cards themselves are already on screen */ });
      }
    };
    document.getElementById("xsort").value = rxnState.sort;
    // Debounced: the Molecules search already did this, the Reactions one did not, so every
    // keystroke re-filtered 33k reactions and re-queried Supabase.
    const xq = document.getElementById("xq");
    let xqDebounce = null;
    xq.oninput = () => { rxnState.q = xq.value; clearTimeout(xqDebounce); xqDebounce = setTimeout(draw, 180); };
    document.getElementById("xkey").onchange = (e) => { rxnState.key = e.target.checked; draw(); };
    document.getElementById("xsort").onchange = (e) => { rxnState.sort = e.target.value; draw(); };
    // Tags are search shortcuts: clicking one drops its text into the search box.
    document.getElementById("xtags").onclick = (e) => { const t = e.target.closest(".tag"); if (!t) return;
      rxnState.q = t.dataset.c || ""; xq.value = rxnState.q; draw(); };
    draw();
  }
  // Deep-link straight to this step on the route page (#/route/<id>/<n> scrolls + flashes
  // it); plain route link only for pre-step_no data.
  const rxnHref = (r) => `#/route/${r.route}${r.step_no ? "/" + r.step_no : ""}`;
  // reactions.json references molecules by hash only — assemble the reaction SMILES from
  // molecules.json (already loaded by the Reactions page), reactants dot-joined >> product.
  const rxnSmiles = (r, mols) => {
    const smi = (h) => (mols && mols[h] && mols[h].smiles) || "";
    const left = (r.reactants || []).map(smi).filter(Boolean).join(".");
    const p = smi(r.product);
    return left && p ? left + ">>" + p : "";
  };
  const rxnCard = (r, mols) => {
    const sm = rxnSmiles(r, mols);
    return `<div class="card rxn-card">
    ${isFavRxn(r) ? `<div class="fav-star" title="One of our favorite reactions">★</div>` : ""}
    <div class="rxn-top" onclick="location.hash='${rxnHref(r)}'">${scheme(r.reactants, r.product, { mini: true, sizes: mols })}</div>
    ${r.conditions ? `<div class="rxn-cond">${esc(clip(r.conditions, 96))}</div>` : ""}
    <div class="body">
      <div style="display:flex;justify-content:space-between;align-items:center;gap:8px">
        <span class="cid">${r.route} · step ${r.step_no ?? r.idx}</span>${r.is_key ? '<span class="badge key">key</span>' : ""}</div>
      ${r.prose ? `<div class="rxn-prose">${esc(clip(r.prose, 160))}</div>` : `<div class="ctitle">${esc(r.class)}</div>`}
      ${reaxysChip(r)}
      <div style="display:flex;justify-content:space-between;align-items:center">
        <span class="stars-slot">${starsHTML("rxn:" + r.id, "sm")}</span>
        <span style="display:flex;align-items:center;gap:12px">
          ${sm ? `<button type="button" class="smiles-chip" data-smiles="${esc(sm)}" data-smiles-what="Reaction SMILES" title="Copy reaction SMILES">⧉ SMILES</button>` : ""}
          <a class="rate-summary" href="${rxnHref(r)}" style="color:var(--accent);margin:0">view in route →</a></span></div>
    </div></div>`;
  };

  /* ---------------- MOLECULES ---------------- */
  // chem: ranked ChemSearch results when the query is a valid SMILES/SMARTS (null = text search);
  // gen: generation counter so stale async scans cancel when the query or view changes.
  const molState = { q: "", targets: false, sort: "heavy-desc", chem: null, gen: 0 };
  async function viewMolecules() {
    const mols = Object.values(await jget("data/molecules.json"));
    app.innerHTML = `<div class="page"><div class="wrap">
      <div class="page-head"><div><h1>Molecules</h1><div class="count" id="mcount"></div></div></div>
      <div class="toolbar">
        <input type="search" id="mq" placeholder="Search name, formula, SMILES or SMARTS...">
        <label><input type="checkbox" id="mtar"> targets only</label>
        <label>Sort <select id="msort">
          <option value="heavy-desc">largest (heavy atoms)</option><option value="heavy-asc">smallest</option>
          <option value="rings-desc">most rings</option><option value="chiral-desc">most stereocenters</option>
          <option value="routes-desc">most reused</option>
        </select></label>
      </div>
      <div class="grid mols" id="mgrid"></div></div></div>`;
    // controls persist across navigation (module-level molState) — reflect them in the UI
    document.getElementById("msort").value = molState.sort;
    const draw = () => {
      if (molState.chem) {   // structure search: ranking (exact, then overlap) overrides the sort dropdown
        let ranked = molState.chem;
        if (molState.targets) ranked = ranked.filter((x) => x.m.is_target);
        document.getElementById("mcount").textContent = `${ranked.length} of ${mols.length} molecules · structure match`;
        document.getElementById("mgrid").innerHTML = ranked.length ? ranked.slice(0, 400).map((x) => molCard(x.m, x.overlap)).join("")
          : `<div class="empty">No molecules match this structure.</div>`;
        return;
      }
      let list = mols.slice();
      if (molState.q) { const q = molState.q.toLowerCase(); list = list.filter((m) => (m.formula+" "+m.smiles+" "+(m.name||"")).toLowerCase().includes(q)); }
      if (molState.targets) list = list.filter((m) => m.is_target);
      const S = molState.sort;
      list.sort((a, b) => S==="heavy-asc"?a.heavy-b.heavy : S==="rings-desc"?b.rings-a.rings
        : S==="chiral-desc"?b.chiral-a.chiral : S==="routes-desc"?b.routes.length-a.routes.length : b.heavy-a.heavy);
      document.getElementById("mcount").textContent = `${list.length} of ${mols.length} molecules`;
      document.getElementById("mgrid").innerHTML = list.length ? list.slice(0, 400).map((m) => molCard(m)).join("")
        : `<div class="empty">No molecules match.</div>`;
    };
    // If the query parses as SMILES/SMARTS, ChemSearch scans structures (exact first, then
    // substructure by overlap); anything else falls through to the plain text filter above.
    const runSearch = () => {
      const gen = ++molState.gen;
      const live = () => molState.gen === gen && document.getElementById("mgrid");
      const q = molState.q.trim();
      if (!q || /\s/.test(q)) { molState.chem = null; draw(); return; }   // whitespace is never SMILES — skip WASM
      document.getElementById("mcount").textContent = "parsing query…";
      ChemSearch.run(mols, q, {
        canceled: () => !live(),
        onText: () => { if (!live()) return; molState.chem = null; draw(); },
        onProgress: (pct) => { if (live()) document.getElementById("mcount").textContent = `searching structures… ${pct}%`; },
        onDone: (ranked) => { if (!live()) return; molState.chem = ranked; draw(); },
      });
    };
    const mq = document.getElementById("mq"); mq.value = molState.q;
    let mqDebounce = null;
    mq.oninput = () => { molState.q = mq.value; molState.gen++; clearTimeout(mqDebounce); mqDebounce = setTimeout(runSearch, 180); };
    document.getElementById("mtar").onchange = (e) => { molState.targets = e.target.checked; draw(); };
    document.getElementById("msort").onchange = (e) => { molState.sort = e.target.value; draw(); };
    draw();
  }
  const molCard = (m, overlap) => `<div class="card mol-card" onclick="location.hash='#/molecule/${m.hash}'">
    <div class="thumb">${molImg(m.hash)}</div>
    <div class="body">
      ${m.name ? `<div class="ctitle" style="font-size:.92rem;margin:0 0 3px">${esc(m.name)}</div>` : ""}
      <div class="formula">${m.is_target ? '<span class="target-dot"></span>' : ""}${esc(m.formula)}</div>
      <div class="sub">MW ${m.mw} · ${m.rings} rings · ${m.chiral} stereocenters</div>
      <div class="sub">${m.routes.length} route${m.routes.length!==1?"s":""}</div>
      ${overlap != null ? `<div class="sub" style="color:var(--accent)">${overlap === 1 ? "exact structure match" : Math.round(overlap * 100) + "% structure match"}</div>` : ""}</div></div>`;

  async function viewMolecule(hash) {
    const mols = await jget("data/molecules.json"); const m = mols[hash];
    if (!m) return notFound("molecule", hash);
    const routes = m.routes.map((r) => `<a class="badge" href="#/route/${r}" style="cursor:pointer">${r}</a>`).join(" ");
    // "How to obtain it": AiZynthFinder synthesis for building-block leaves. The index tells
    // us whether a leaf was searched/solved without fetching the full (heavier) route file;
    // both degrade to null when the feature wasn't built into this dataset.
    const lrIndex = await jget("data/leaf_routes.json").catch(() => null);
    const li = lrIndex && lrIndex[hash];
    const lr = li && li.cat === "obtainable" ? await jget(`data/leaf_routes/${hash}.json`).catch(() => null) : null;
    app.innerHTML = `<div class="page"><div class="wrap">
      <a class="back" href="#/molecules">← All molecules</a>
      <div class="detail">
        <div class="aside-card mol-view-card" style="position:static"><div class="mol-view">${molImg(hash, "mol-view-img")}</div></div>
        <div>
          ${m.name ? `<h1 style="font-size:1.5rem">${esc(m.name)}</h1><div class="count mono" style="margin:4px 0 2px">${esc(m.formula)}${m.is_target ? " · target" : ""}</div>` : `<h1 class="mono" style="font-size:1.4rem">${esc(m.formula)}</h1>`}
          <div class="count" style="word-break:break-all;margin:6px 0 20px">${esc(m.smiles)} <button type="button" class="smiles-chip" data-smiles="${esc(m.smiles)}" data-smiles-what="${esc(m.name || m.formula)} SMILES">⧉ Copy</button></div>
          <div class="aside-card" style="position:static">
            <h3>Appears in ${m.routes.length} route${m.routes.length!==1?"s":""}</h3>
            <div class="tags">${routes}</div>
            ${m.is_building_block ? `<div class="spec-row" style="margin-top:12px;border-top:1px solid var(--line);padding-top:12px"><span>Building block${m.availability ? " · " + esc(m.availability) : ""}</span></div>` : ""}
          </div>
          ${sourcingCard(lr, li)}
        </div>
      </div></div></div>`;
  }

  // "How to obtain it" card for a building-block leaf: an AiZynthFinder synthesis from
  // purchasable starting materials, rendered as one mini scheme per reaction (synthesis order).
  function sourcingCard(lr, li) {
    const cat = li && li.cat;
    if (cat === "buyable") {
      return `<div class="aside-card leaf-route" style="position:static">
        <h3>How to obtain it</h3>
        <p class="aside-p"><span class="cat-dot cat-buyable"></span>Commercially available — listed in the ZINC / eMolecules catalogue.</p></div>`;
    }
    if (cat === "obtainable" && lr && lr.solved) {
      currentLeafMols = lr.mols || {};              // read by the leaf-structure copy handler
      const sizes = {};
      Object.entries(lr.mols || {}).forEach(([h, v]) => { if (v.w && v.h) sizes[h] = { w: v.w, h: v.h }; });
      const rows = (lr.route || []).map((s, i) =>
        `<div class="leaf-step"><span class="leaf-step-n">${i + 1}</span>${scheme(s.reactants, s.product, { mini: true, sizes })}</div>`).join("");
      const nrx = lr.n === 1 ? "1 reaction" : `${lr.n} reactions`;
      return `<div class="aside-card leaf-route" style="position:static">
        <h3><span class="cat-dot cat-obtainable"></span>How to obtain it</h3>
        <p class="aside-p">A ${nrx} synthesis${lr.longest ? ` (longest linear path ${lr.longest})` : ""} from commercially available starting materials.</p>
        <div class="leaf-steps">${rows}</div>
        <p class="ai-attrib">Route proposed by AiZynthFinder; terminal reagents drawn from the ZINC / eMolecules catalogue. Click a structure to copy its SMILES.</p>
      </div>`;
    }
    return "";   // small / unavailable: nothing shown
  }

  /* ---------------- CHEMICAL SPACE (round TMAP, landing page) ----------------
   * Replaces the former standalone "Map" tab. Two deliberate differences from that version:
   *
   *   • Round. The MST layout is a blob, so it was previously letterboxed inside a wide dark
   *     rectangle with dead corners. Normalising radially about the centroid (rather than
   *     stretching each axis) fits it to a disc while preserving angles and relative distances.
   *   • No wheel zoom. On a landing page, capturing the scroll wheel traps the reader mid-page.
   *     Hover and click are the whole interaction; there is nothing to pan back from.
   */
  /* Colour-by is limited to STRUCTURAL properties. The AI scores were dropped as options because
   * they barely vary across targets — 87% of routes score 8 or 9 overall, 80% score 6-7 on green —
   * so the ramp was near-flat and the map carried no information. These three span their ranges
   * far more widely (coefficient of variation 0.33 / 0.60 / 0.52 against 0.13 for the scores). */
  const mapState = { colorBy: "mw" };
  const MAP_METRICS = {
    // "mw" is the exact molecular weight from RDKit (CalcExactMolWt), so it is labelled as such —
    // the molecule detail card already reports the same number as "MW".
    mw:     { label: "molecular weight", lo: "light", hi: "heavy",      get: (n) => n.mw || 0 },
    chiral: { label: "stereocenters",  lo: "few",     hi: "many",        get: (n) => n.chiral || 0 },
    rings:  { label: "ring count",     lo: "acyclic", hi: "polycyclic",  get: (n) => n.rings || 0 },
  };
  const MAP_ORDER = ["mw", "chiral", "rings"];
  const tmapSectionHTML = (n) => `
    <section class="home-section tmap-section" id="chemical-space"><div class="wrap">
      <h2 class="sec-h">Chemical space</h2>
      <p class="sec-sub">${(n || 0).toLocaleString()} natural-product targets, arranged so structurally similar
        compounds sit near each other (minimum spanning tree over Morgan/Tanimoto similarity).
        Hover a point to see the structure; click to open its best route.</p>
      <div class="tmap-wrap">
        <div class="tmap-controls">
          ${MAP_ORDER.map((c) => `<button class="tmc${mapState.colorBy===c?" on":""}" data-c="${c}">${MAP_METRICS[c].label}</button>`).join("")}
        </div>
        <div class="tmap-round">
          <canvas id="tmap"></canvas>
          <div class="map-tip" id="tip"><img id="tipImg" crossorigin="anonymous" alt=""><div class="cap" id="tipCap"></div></div>
        </div>
        <div class="tmap-legend">
          <div class="grad"></div>
          <div class="ends"><span id="legLo">${MAP_METRICS[mapState.colorBy].lo}</span><span id="legHi">${MAP_METRICS[mapState.colorBy].hi}</span></div>
        </div>
        <div class="tmap-hint">Click a target to open its best route</div>
      </div>
    </div></section>`;

  /* Where #/map lands. scrollIntoView({block:"start"}) aligned the section's top edge with the top
     of the scrollport, which the 65px sticky .nav then covered — the heading sat half behind the
     bar — and on a 800px-tall window it still left the bottom of the disc below the fold.
     So: land on whichever is lower of two targets. The heading is preferred, because it names the
     map and carries the "click a target" instruction; the fallback is just far enough down to fit
     the whole disc. On a tall window the heading target is already low enough and nothing is
     given up; on a short one the map wins and the heading scrolls off. The cap keeps the top of
     the disc clear of the nav — without it, a window shorter than the disc would bury it. */
  const NAV_GAP = 15;   // the 80px landing the other in-page anchors get, measured off the nav
  const DISC_PAD = 16;  // breathing room under the disc when the map target is the one that wins
  function chemicalSpaceScrollTop(sec) {
    const navH = document.querySelector(".nav")?.getBoundingClientRect().height || 0;
    const top = (el) => el.getBoundingClientRect().top + window.scrollY;
    let y = top(sec.querySelector("h2")) - navH - NAV_GAP;
    const disc = sec.querySelector(".tmap-round");
    if (disc) {
      const fitsDisc = top(disc) + disc.getBoundingClientRect().height + DISC_PAD - window.innerHeight;
      y = Math.min(Math.max(y, fitsDisc), top(disc) - navH - NAV_GAP);
    }
    return Math.max(0, Math.round(y));
  }
  /* One aim is not enough, because the page grows underneath it. The molecule images above the map
     are lazy, and a lazy image that has not decoded yet has no intrinsic size and so occupies no
     height at all: at the moment of the jump the section sits 500px higher on a phone column than
     where it ends up, and the images only begin loading as the scroll brings them into view, which
     moves the target again mid-flight. The jump therefore finished above the figure — worst on a
     phone, where those images span the full column instead of sitting in a side card.
     So: switch the ones above the map to eager, which makes the layout settle on its own instead
     of waiting for the scroll to reach them, and re-aim until the target holds still. It gives up
     after SETTLE_MS in case an image never arrives, and abandons the whole thing the moment the
     reader scrolls — chasing a target under someone who has taken over is worse than landing
     short. Only wheel/touch/key count as taking over; listening for "scroll" would catch our own
     animation and cancel on the first frame. */
  const SETTLE_MS = 3000;
  const SETTLE_TICK = 120;
  function scrollToChemicalSpace() {
    const sec = document.getElementById("chemical-space");
    if (!sec || !sec.querySelector("h2")) return;
    const secTop = sec.getBoundingClientRect().top + window.scrollY;
    for (const img of document.images) {
      if (img.loading === "lazy" && !img.complete
          && img.getBoundingClientRect().top + window.scrollY < secTop) img.loading = "eager";
    }
    const ac = new AbortController();
    const stop = () => ac.abort();
    for (const ev of ["wheel", "touchstart", "keydown"]) {
      window.addEventListener(ev, stop, { passive: true, signal: ac.signal });
    }
    let aimed = null, held = 0;
    const reaim = () => {
      if (ac.signal.aborted) return;
      const y = chemicalSpaceScrollTop(sec);
      if (y === aimed) { if (++held >= 4) stop(); return; }
      held = 0; aimed = y;
      window.scrollTo({ top: y, behavior: "smooth" });
    };
    const iv = setInterval(reaim, SETTLE_TICK);
    const cap = setTimeout(stop, SETTLE_MS);
    ac.signal.addEventListener("abort", () => { clearInterval(iv); clearTimeout(cap); });
    reaim();
  }

  // One resize handler for the lifetime of the page, not one per visit to the landing page —
  // viewHome() re-runs on every navigation home, and the old Map view leaked a listener per visit.
  let tmapResize = null;
  window.addEventListener("resize", () => tmapResize && tmapResize());

  async function mountTmap() {
    const cv = document.getElementById("tmap");
    if (!cv) { tmapResize = null; return; }
    // index.json rides along to score-rank each target's routes for the click handler
    // (tmap nodes only carry alphabetical route ids); jget caches it across views.
    const [tm, idx] = await Promise.all([jget("data/tmap.json"), jget("data/index.json")]);
    if (!document.getElementById("tmap")) return;   // navigated away while loading
    const idxById = {}; idx.forEach((r) => (idxById[r.id] = r));
    const bestRoute = (ids) => bestRouteId(ids, idxById);

    const ctx = cv.getContext("2d"), tip = document.getElementById("tip");
    const nodes = tm.nodes;
    let W = 0, H = 0, R = 0, cx = 0, cy = 0;
    const dpr = window.devicePixelRatio || 1;
    let hovered = null;

    // Radial normalisation, computed once: centre on the point cloud's centroid, then divide by
    // the largest radius so the outermost node lands exactly on the disc edge. Scaling both axes
    // by the SAME factor is the point — an axis-independent fit would squash the layout and
    // misrepresent the similarity distances the map exists to show.
    let ox = 0, oy = 0;
    nodes.forEach((n) => { ox += n.x; oy += n.y; });
    ox /= nodes.length || 1; oy /= nodes.length || 1;
    let maxR = 0;
    nodes.forEach((n) => { maxR = Math.max(maxR, Math.hypot(n.x - ox, n.y - oy)); });
    maxR = maxR || 1;
    const unit = nodes.map((n) => ({ ux: (n.x - ox) / maxR, uy: (n.y - oy) / maxR }));
    const pxy = (i) => ({ x: cx + unit[i].ux * R, y: cy + unit[i].uy * R });

    const metric = (n) => (MAP_METRICS[mapState.colorBy] || MAP_METRICS.mw).get(n);
    // Normalise to the 2nd-98th percentile, not min-max, and clamp outside it. These properties
    // have long right tails — stereocenters run to 21 while the 90th percentile is 10 — so a
    // min-max ramp spends most of its range on a handful of outliers and leaves the bulk of the
    // targets crowded into the pale end, unable to be told apart.
    let lo = 0, hi = 1;
    const setRange = () => {
      const v = nodes.map(metric).sort((a, b) => a - b);
      const at = (p) => v[Math.min(v.length - 1, Math.max(0, Math.floor(p * (v.length - 1))))];
      lo = at(0.02); hi = at(0.98);
      if (hi <= lo) { lo = v[0]; hi = v[v.length - 1]; }   // degenerate spread: fall back to full range
    };
    setRange();

    function resize() {
      W = cv.clientWidth; H = cv.clientHeight;
      cv.width = Math.max(1, W * dpr); cv.height = Math.max(1, H * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      cx = W / 2; cy = H / 2;
      R = Math.max(10, Math.min(W, H) / 2 - 16);   // 16px breathing room inside the ring
      draw();
    }
    function draw() {
      ctx.clearRect(0, 0, W, H);
      // MST edges first. They carry the structure of the map, so they need enough contrast to
      // read between the dots — with FM^3 the median edge is under 1% of the map's width, so
      // oversized nodes bury them entirely.
      ctx.strokeStyle = "rgba(31,77,63,.30)"; ctx.lineWidth = 1;
      ctx.beginPath();
      tm.edges.forEach(([a, b]) => { const p = pxy(a), q = pxy(b); ctx.moveTo(p.x, p.y); ctx.lineTo(q.x, q.y); });
      ctx.stroke();
      // Node radius scales with the disc so the filaments stay legible at any width: at ~480px
      // the dots must not touch, or the branching structure the layout exists to show is lost.
      const rBase = Math.max(1.5, Math.min(3.0, R / 95));
      nodes.forEach((n, i) => {
        const p = pxy(i), v = (metric(n) - lo) / (hi - lo || 1);
        const isHot = hovered === i;
        const r = rBase * (isHot ? 2.4 : 1);
        ctx.beginPath(); ctx.arc(p.x, p.y, r, 0, 7);
        ctx.fillStyle = isHot ? "#A9662E" : ramp(v);           // copper = hover, never a value
        ctx.fill();
        ctx.strokeStyle = isHot ? "#8A5223" : "rgba(26,30,28,.22)";
        ctx.lineWidth = isHot ? 1.6 : 0.6;
        ctx.stroke();
      });
    }
    const nearest = (mx, my) => {
      let best = -1, bd = 15 * 15;
      for (let i = 0; i < nodes.length; i++) {
        const p = pxy(i), d = (p.x - mx) ** 2 + (p.y - my) ** 2;
        if (d < bd) { bd = d; best = i; }
      }
      return best;
    };
    const hideTip = () => { tip.style.display = "none"; if (hovered !== null) { hovered = null; draw(); } };
    cv.onmousemove = (e) => {
      const b = cv.getBoundingClientRect(), mx = e.clientX - b.left, my = e.clientY - b.top;
      const i = nearest(mx, my);
      if (i < 0) { cv.style.cursor = "default"; hideTip(); return; }
      const n = nodes[i];
      cv.style.cursor = "pointer";
      if (hovered !== i) { hovered = i; draw(); }
      tip.style.display = "block";
      // keep the card inside the canvas on both axes
      tip.style.left = Math.max(4, Math.min(mx + 16, W - 178)) + "px";
      tip.style.top = Math.max(4, Math.min(my - 10, H - 172)) + "px";
      document.getElementById("tipImg").src = `${IMG_BASE}/${n.hash}.svg?v=${ASSET_VER}`;
      document.getElementById("tipCap").textContent = n.name || `${n.formula}${n.is_target ? " · target" : ""}`;
    };
    cv.onmouseleave = () => { cv.style.cursor = "default"; hideTip(); };
    cv.onclick = (e) => {
      const b = cv.getBoundingClientRect();
      const i = nearest(e.clientX - b.left, e.clientY - b.top);
      if (i < 0) return;
      const n = nodes[i];
      location.hash = (n.is_target && n.routes.length) ? "#/route/" + bestRoute(n.routes) : "#/molecule/" + n.hash;
    };
    document.querySelector(".tmap-controls").onclick = (e) => {
      const c = e.target.dataset.c; if (!c) return;
      mapState.colorBy = c;
      document.querySelectorAll(".tmap-controls .tmc").forEach((b) => b.classList.toggle("on", b.dataset.c === c));
      const M = MAP_METRICS[c] || MAP_METRICS.mw;
      document.getElementById("legLo").textContent = M.lo;
      document.getElementById("legHi").textContent = M.hi;
      setRange(); draw();
    };
    tmapResize = resize;
    requestAnimationFrame(resize);
  }

  function notFound(kind, id) {
    app.innerHTML = `<div class="page"><div class="wrap"><div class="empty">No ${kind} found for “${esc(id)}”.<br><a class="back" href="#/">← Home</a></div></div></div>`;
  }

  /* ---------------- privacy policy ---------------- */
  /* ---------------- about ----------------
   * All editable content lives in this one object so the page can be updated without touching
   * markup. Anything whose value is empty renders as plain text rather than a broken link, and
   * empty sections are omitted entirely — so an unfilled field is invisible rather than shipping
   * a "TODO" onto a public page.
   *
   * ⚠️ `team` is a PROVISIONAL list derived from repository contributors. It is a credits page
   * for real people: confirm and complete it (and add affiliations if wanted) before the site
   * goes public. Missing an author here is worse than any layout problem on this page.
   */
  const ABOUT = {
    /* Author list and affiliations are transcribed from the manuscript
       "Strategy-first synthesis planning for complex natural products", grouped by lab.
       Armstrong and Nguyen contributed equally; Schwaller is corresponding author.

       The Arizona and Pittsburgh splits follow the paper's affiliation markers and are certain.
       The EPFL authors all share one affiliation marker and the manuscript's "Author contributions"
       section is empty, so the LIAC/LSPN split cannot be read off the paper. Susanu and Strunden
       are placed in LIAC on the group lead's own correction; the remaining EPFL placements are
       still inferred (from co-authorship on the group's other papers and from who ran the expert
       assessment) and are worth a final check before the gate comes off — misplacing someone under
       the wrong group is the worst thing this page can do. */
    groups: [
      {
        name: "Laboratory of Artificial Chemical Intelligence (LIAC)",
        pi: "Philippe Schwaller",
        where: "EPFL, Lausanne",
        url: "https://schwallergroup.github.io",
        // Ordered as in the manuscript's author list, so contribution order is preserved.
        members: ["Daniel Armstrong", "Xuan-Vu Nguyen", "Octavian Susanu", "Gabriel Gibberd",
                  "Théo A. Neukomm", "Taddäus Strunden", "Maarten R. Dobbelaere",
                  "Philippe Schwaller"],
      },
      {
        name: "Laboratory of Synthesis and Natural Products (LSPN)",
        pi: "Jieping Zhu",
        where: "EPFL, Lausanne",
        url: "https://www.epfl.ch/labs/lspn/",
        members: ["Dan Forster", "Morgane Delattre", "Shawn Teh", "Clément Rols", "Jieping Zhu"],
      },
      {
        name: "Njardarson Laboratory",
        pi: "Jon T. Njardarson",
        where: "University of Arizona",
        url: "https://sites.arizona.edu/njardarson-lab/",
        members: ["John Federice", "Hayden Leatherwood", "Jon T. Njardarson"],
      },
      {
        name: "Wipf Group",
        pi: "Peter Wipf",
        where: "University of Pittsburgh",
        url: "http://ccc.chem.pitt.edu/wipf/",
        members: ["M. Lavelle Barnes", "Peter Wipf"],
      },
    ],
    // Maarten Dobbelaere is also at the Laboratory for Chemical Technology, Ghent University.
    teamNote: "Daniel Armstrong and Xuan-Vu Nguyen contributed equally. Maarten R. Dobbelaere is "
      + "also affiliated with the Laboratory for Chemical Technology, Ghent University.",

    publications: [
      { title: "Strategy-first synthesis planning for complex natural products",
        note: "arXiv:2608.07454", url: "https://arxiv.org/abs/2608.07454",
        desc: "Describes SynthEx, the agentic framework that proposed, critiqued and improved every "
            + "route in this atlas, and the expert assessment of its key steps." },
    ],
    /* Kept in sync with the entry in the SynthEx README — that is the canonical copy, this is the
       one a reader can actually reach. Author order is the manuscript's, not the by-lab grouping
       above. The line breaks inside `author` are cosmetic — BibTeX collapses the whitespace — and
       are what keeps the block readable at the width it renders in. */
    bibtex:
      "@article{armstrong2026synthex,\n"
      + "  title  = {Strategy-first synthesis planning for complex natural products},\n"
      + "  author = {Armstrong, Daniel and Nguyen, Xuan-Vu and Susanu, Octavian and Gibberd, Gabriel\n"
      + "            and Neukomm, Th{\\'e}o A. and Strunden, Tadd{\\\"a}us and Forster, Dan\n"
      + "            and Delattre, Morgane and Teh, Shawn and Rols, Cl{\\'e}ment and Federice, John\n"
      + "            and Leatherwood, Hayden and Barnes, M. Lavelle and Dobbelaere, Maarten R.\n"
      + "            and Wipf, Peter and Njardarson, Jon T. and Zhu, Jieping and Schwaller, Philippe},\n"
      + "  year          = {2026},\n"
      + "  eprint        = {2608.07454},\n"
      + "  archivePrefix = {arXiv},\n"
      + "  primaryClass  = {cs.MA},\n"
      + "  url           = {https://arxiv.org/abs/2608.07454}\n"
      + "}",
    previousWork: [
      { title: "Synthegy", url: "https://www.cell.com/matter/fulltext/S2590-2385(26)00175-X",
        desc: "Chemical reasoning in LLMs unlocks strategy-aware synthesis planning and reaction "
            + "mechanism elucidation. Matter, 2026." },
      { title: "Synthelite", url: "https://arxiv.org/abs/2512.16424",
        desc: "Chemist-aligned and feasibility-aware synthesis planning with LLMs." },
      { title: "SynthStrategy", url: "https://arxiv.org/abs/2512.01507",
        desc: "Extracting and formalising latent strategic insights from LLMs in organic chemistry." },
      { title: "RXNClassifier", url: "https://arxiv.org/abs/2607.01061",
        desc: "Reaction classification, used here to place SynthEx's chemistry in reaction space." },
    ],
    // Transcribed from the manuscript's Funding statement. Grant numbers are part of the
    // acknowledgement obligation, so they are reproduced rather than paraphrased.
    funding: [
      { name: "Swiss National Science Foundation",
        detail: "through the National Centre of Competence in Research (NCCR) Catalysis (225147) and through grant 214915" },
      { name: "AiChemist", detail: "MSCA Doctoral Network (X.-V. Nguyen)" },
      { name: "LowDataML", detail: "MSCA Doctoral Network (G. Gibberd)" },
      { name: "Intel and Merck KGaA", detail: "via the AWASES programme (T. A. Neukomm)" },
      { name: "Research Foundation – Flanders (FWO Vlaanderen)",
        detail: "postdoctoral fellowship 1266226N and travel grant V414426N (M. R. Dobbelaere)" },
      { name: "Reaxys R&D collaboration network", detail: "" },
    ],
    // Reproduced verbatim: funder acknowledgements normally have required wording.
    fundingNote:
      "This research is with support from Google.org and the Google Cloud Research Credits program "
      + "for the Gemini Academic Program, additionally supported by Digital Schooling.",
    acknowledgement:
      "The authors thank collaborators in the EPFL Laboratory of Artificial Chemical Intelligence "
      + "(LIAC) for helpful discussions.",

    software: [
      { title: "SynthEx", url: "https://github.com/schwallergroup/SynthEx",
        desc: "The synthesis-planning framework behind every route here." },
      { title: "ReactionClassifier", url: "https://github.com/schwallergroup/ReactionClassifier",
        desc: "Deterministic reaction classification." },
      { title: "tmap2", url: "https://github.com/afloresep/tmap2",
        desc: "Lays out the chemical-space map: a minimum spanning tree over Morgan/Tanimoto "
            + "similarity, positioned with OGDF's FM³ algorithm." },
    ],
  };
  // A link only when we actually have a URL — never an anchor to nowhere. Entries without one
  // render as plain text.
  const aboutLink = (title, url) => (url
    ? `<a class="ext" href="${esc(url)}" target="_blank" rel="noopener">${esc(title)}</a>`
    : esc(title));
  function viewAbout() {
    const team = ABOUT.groups.length
      ? `<h2 id="team">Team</h2><div class="about-groups">${ABOUT.groups.map((g) => `
          <section class="about-group">
            <h3>${aboutLink(g.name, g.url)}</h3>
            <div class="about-group-meta">${esc(g.pi)} · ${esc(g.where)}</div>
            <ul class="about-team">${g.members.map((m) => `<li>${esc(m)}</li>`).join("")}</ul>
          </section>`).join("")}</div>
          ${ABOUT.teamNote ? `<p class="about-desc">${esc(ABOUT.teamNote)}</p>` : ""}`
      : "";
    const listSection = (heading, items, lede) => (items && items.length
      ? `<h2>${heading}</h2>${lede ? `<p class="about-lede">${lede}</p>` : ""}
         <ul class="about-list">${items.map((i) =>
          `<li><b>${aboutLink(i.title, i.url)}</b>${i.note ? ` <span class="gl-chip">${esc(i.note)}</span>` : ""}`
          + `${i.desc ? `<div class="about-desc">${esc(i.desc)}</div>` : ""}</li>`).join("")}</ul>`
      : "");
    const funding = ABOUT.funding.length
      ? `<ul class="about-list">${ABOUT.funding.map((f) =>
          `<li><b>${esc(f.name)}</b>${f.detail ? ` — ${esc(f.detail)}` : ""}</li>`).join("")}</ul>`
      : "";
    // The paper entry plus a ready-to-paste BibTeX block. data-copy is picked up by the generic
    // copy handler, so this needs no wiring of its own.
    const citation = listSection("SynthAtlas / SynthEx citation", ABOUT.publications)
      + (ABOUT.bibtex
        ? `<div class="bib">
             <div class="bib-head"><span class="mono">BibTeX</span>
               <button class="bib-copy" type="button" data-copy="${esc(ABOUT.bibtex)}"
                 data-copy-what="BibTeX entry">Copy</button></div>
             <pre class="bib-body">${esc(ABOUT.bibtex)}</pre>
           </div>`
        : "");
    app.innerHTML = `<div class="page"><div class="wrap legal about">
      <a class="back" href="#/">← Home</a>
      <h1>About SynthAtlas</h1>
      <p class="about-lede">SynthAtlas is a catalogue of AI-planned total-synthesis routes to natural
        products — an open resource covering targets that have no reported total synthesis, built so
        synthetic chemists can see how a machine proposes to reach them.</p>
      <p>Every route was proposed and critiqued by the SynthEx agent. <b>None has been
        experimentally validated</b> — each is an idea for how a compound could be made, not a
        procedure. Ratings and comments from expert chemists are what turn it into evidence, which is
        why feedback is built into every route and reaction page. The
        <a href="#/glossary">glossary</a> marks exactly which numbers are computed and which are
        AI-proposed.</p>
      <!-- Deliberately placed right after the "none has been validated" paragraph: the honest next
           question is "then where do I find synthesis that HAS been done?", and it deserves an
           answer on the same screen rather than nowhere. Both are free and both are the kind of
           record this atlas is not. -->
      <p>For chemistry that has actually been run, two open resources are worth keeping open
        alongside this one.
        <a class="ext" href="https://chemistrybydesign.oia.arizona.edu" target="_blank"
           rel="noopener">Chemistry by Design</a> is an interactive collection of published total
        syntheses, drawn step by step from the literature — the completed counterpart to what you
        will find here. <a class="ext" href="https://www.orgsyn.org" target="_blank"
           rel="noopener">Organic Syntheses</a> publishes detailed preparations only after they have
        been reproduced in an independent laboratory, which makes it about as close to a verified
        procedure as the literature gets.</p>
      ${team}
      ${citation}
      ${listSection("Previous work", ABOUT.previousWork,
        "SynthAtlas builds on a line of work on language models for synthesis planning.")}
      <h2>Funding &amp; support</h2>
      ${funding}
      ${ABOUT.fundingNote ? `<p class="about-fundnote">${esc(ABOUT.fundingNote)}</p>` : ""}
      ${ABOUT.acknowledgement ? `<p class="about-desc">${esc(ABOUT.acknowledgement)}</p>` : ""}
      ${listSection("Software", ABOUT.software)}
    </div></div>`;
  }

  /* ---------------- Suggest a target (#/suggest) ----------------
   * Replaces the Google Form this page used to link out to. Three things it does that the form
   * could not: identity comes from the sign-in session rather than being typed, the target SMILES
   * is parsed by RDKit before submit so nobody files an unreadable structure, and strategies are
   * repeatable blocks rather than one free-text box.
   *
   * WHETHER SUBMITTING NEEDS AN ACCOUNT is a one-line switch: `suggestRequiresSignIn` in config.js.
   * It ships ON, so today an account is required and the address on every row is a verified one.
   * Turning it off opens the form to anyone — a target request from a chemist who will not make an
   * account is still a target request — at the cost of unattributable rows and an insert path that
   * anyone can call. See the config comment for what else that needs.
   *
   * The switch reaches exactly two places: the submit gate and the signed-out copy below, and the
   * guard in submitSuggestion() (data-access.js). Nothing else in this file reads it.
   *
   * The draft lives in localStorage, not just in memory. Losing a half-written strategy is the
   * most likely way to lose the feedback this page exists to collect, and it can happen three
   * ways: SA.onAuthChange re-runs route() the moment someone signs in mid-form, navigating away
   * and back re-renders, and a tab that was closed after switching windows to fetch a DOI is gone
   * entirely. One localStorage key covers all three, and it works signed-out, so filling the form
   * first and signing in second is a supported path rather than a trap.
   */
  const SUGGEST_DRAFT_KEY = "sa_suggest_draft";
  const SUGGEST_NEEDS_AUTH = !!(window.SA_CONFIG || {}).suggestRequiresSignIn;
  const emptyStrategy = () => ({ description: "", reference: "" });
  const emptyDraft = () => ({ idType: "smiles", target: "", targetUrl: "",
                              strategies: [emptyStrategy()], feedback: "", contactOk: true });
  function loadDraft() {
    try {
      const d = JSON.parse(localStorage.getItem(SUGGEST_DRAFT_KEY));
      if (!d || typeof d !== "object") return emptyDraft();
      const base = emptyDraft();
      const s = Array.isArray(d.strategies) && d.strategies.length
        ? d.strategies.map((x) => ({ description: String(x?.description || ""), reference: String(x?.reference || "") }))
        : base.strategies;
      return { idType: d.idType === "name" ? "name" : "smiles", target: String(d.target || ""),
               targetUrl: String(d.targetUrl || ""), strategies: s,
               feedback: String(d.feedback || ""), contactOk: d.contactOk !== false };
    } catch { return emptyDraft(); }
  }
  const saveDraft = (d) => { try { localStorage.setItem(SUGGEST_DRAFT_KEY, JSON.stringify(d)); } catch {} };
  const clearDraft = () => { try { localStorage.removeItem(SUGGEST_DRAFT_KEY); } catch {} };
  const draftHasContent = (d) =>
    !!(d.target.trim() || d.targetUrl.trim() || d.feedback.trim() ||
       d.strategies.some((s) => s.description.trim() || s.reference.trim()));

  /* Writes inside the form are debounced (see `persist` below), which leaves the last few hundred
   * milliseconds of typing in memory only. Navigating within the app is safe — the timer outlives
   * the re-render — but the tab GOING AWAY is not, and that is the likeliest way to lose work here:
   * someone switches to another window to fetch a DOI, and the browser discards or freezes the tab
   * while they are gone. So flush on the way out.
   *
   * `visibilitychange → hidden` covers the switch-away itself and is the only one mobile Safari
   * reliably fires before a tab is frozen; `pagehide` covers closing and navigating off the site.
   * Both can fire for one departure, which is harmless — the second write is identical.
   *
   * ONE listener for the life of the page, holding a pointer to whichever render is current, so
   * revisiting #/suggest cannot stack up listeners that each write a stale draft over a newer one.
   * viewSuggest() sets this; submitting clears it, so a submitted draft is never resurrected. */
  let flushSuggestDraft = null;
  const flushDraftNow = () => { if (flushSuggestDraft) flushSuggestDraft(); };
  window.addEventListener("pagehide", flushDraftNow);
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "hidden") flushDraftNow();
  });

  // A bare DOI or a citation string is still useful to us, so a non-URL is a HINT, never a
  // blocker — this returns advice, and nothing reads it as a validity signal.
  function targetUrlHint(v) {
    const t = String(v || "").trim();
    if (!t) return "";
    try { const u = new URL(t); if (u.protocol === "http:" || u.protocol === "https:") return ""; } catch {}
    return "We'll keep this as a reference — a full https:// link is easiest for us to follow.";
  }

  async function viewSuggest() {
    const draft = loadDraft();
    const restored = draftHasContent(draft);
    // Result of the last SMILES check. `ok` gates submit in SMILES mode only.
    let check = { ok: false, canon: "", svg: "" };

    const strategyHTML = (s, i, n) => `
      <div class="sug-strategy" data-i="${i}">
        <div class="sug-strategy-head">
          <span class="sug-strategy-n">Strategy ${i + 1}</span>
          ${n > 1 ? `<button type="button" class="sug-remove" data-rm="${i}"
            aria-label="Remove strategy ${i + 1}">Remove</button>` : ""}
        </div>
        <textarea class="sug-ta" data-f="description" rows="4"
          placeholder="Describe the approach — a disconnection, a key transform, a reagent class. If you have a very particular idea, be specific.">${esc(s.description)}</textarea>
        <input class="auth-input" data-f="reference" type="text"
          placeholder="Reference or link for this strategy (optional)" value="${esc(s.reference)}">
      </div>`;

    app.innerHTML = `<div class="page"><div class="wrap legal sug">
      <a class="back" href="#/">← Home</a>
      <h1>Suggest a target</h1>
      <p class="about-lede">Request a target for us to plan with the SynthEx agent, and suggest the
        strategies you would like to see explored for it. This is how the atlas grows in the
        direction chemists actually care about — not just which routes we happened to run.</p>
      <div class="sug-who" id="sugWho"></div>
      <div class="sug-restored" id="sugRestored"${restored ? "" : " hidden"}>
        Draft restored from your last visit. <button type="button" class="sug-link" id="sugFresh">Start fresh</button>
      </div>

      <form class="sug-form" id="sugForm" novalidate>
        <section class="sug-field">
          <label class="sug-label" for="sugTarget">Which target would you like to explore with our
            SynthEx agent harness? <span class="sug-req">required</span></label>
          <p class="sug-hint">SMILES is preferred — it's the one format we can verify here and match
            against the atlas. If you don't have one to hand, switch to a common name.</p>
          <div class="sug-row">
            <select class="auth-input sug-type" id="sugType" aria-label="Target identifier type">
              <option value="smiles">SMILES</option>
              <option value="name">Common name</option>
            </select>
            <input class="auth-input sug-target" id="sugTarget" type="text" spellcheck="false"
              autocapitalize="off" autocorrect="off" value="${esc(draft.target)}">
          </div>
          <div class="sug-status" id="sugStatus" role="status" hidden></div>
          <div class="sug-known" id="sugKnown" role="status" hidden></div>
          <div class="sug-preview" id="sugPreview" hidden></div>
        </section>

        <section class="sug-field">
          <label class="sug-label" for="sugUrl">Link or reference for this target
            <span class="sug-opt">optional</span></label>
          <p class="sug-hint">An NPAtlas or PubChem entry, a DOI, or the isolation paper — anything
            that pins down exactly which compound you mean. Trivial names are often shared between
            several natural products.</p>
          <input class="auth-input" id="sugUrl" type="text" spellcheck="false"
            placeholder="https://www.npatlas.org/explore/compounds/…" value="${esc(draft.targetUrl)}">
          <div class="sug-status hint" id="sugUrlStatus" hidden></div>
        </section>

        <section class="sug-field">
          <label class="sug-label">Is there any strategy (or strategies) you would be interested in
            exploring for this target? <span class="sug-opt">optional</span></label>
          <p class="sug-hint">Add one block per approach. If there is a reference you would like us
            to work from, link it alongside the description.</p>
          <div id="sugStrategies"></div>
          <button type="button" class="btn btn-ghost btn-sm sug-add" id="sugAdd">+ Add another strategy</button>
        </section>

        <section class="sug-field">
          <label class="sug-label" for="sugFeedback">Do you have any feedback about the SynthAtlas
            website? <span class="sug-opt">optional</span></label>
          <p class="sug-hint">Such as whether the layout was intuitive, what information you did or
            didn't find useful, or whether you hit any technical issues.</p>
          <textarea class="sug-ta" id="sugFeedback" rows="4">${esc(draft.feedback)}</textarea>
        </section>

        <section class="sug-field">
          <label class="sug-label">Would you like to be contacted when we have run SynthEx for your
            target?</label>
          <div class="sug-radios">
            <label><input type="radio" name="sugContact" value="yes"${draft.contactOk ? " checked" : ""}> Yes</label>
            <label><input type="radio" name="sugContact" value="no"${draft.contactOk ? "" : " checked"}> No</label>
          </div>
        </section>

        <div class="sug-actions">
          <button type="submit" class="btn btn-primary" id="sugSubmit" disabled>Submit suggestion</button>
          <span class="sug-gate" id="sugGate"></span>
        </div>
      </form>
      <div id="sugMine"></div>
    </div></div>`;

    const $ = (id) => document.getElementById(id);
    const typeSel = $("sugType"), targetIn = $("sugTarget"), statusEl = $("sugStatus"),
          previewEl = $("sugPreview"), knownEl = $("sugKnown"),
          urlIn = $("sugUrl"), urlStatus = $("sugUrlStatus"),
          stratWrap = $("sugStrategies"), feedbackIn = $("sugFeedback"),
          submitBtn = $("sugSubmit"), gateEl = $("sugGate");
    typeSel.value = draft.idType;

    let saveT;
    const persist = () => { clearTimeout(saveT); saveT = setTimeout(() => saveDraft(draft), 400); };
    // Point the page-level flush at THIS render's draft, replacing any earlier one.
    flushSuggestDraft = () => { clearTimeout(saveT); saveDraft(draft); };

    /* -- strategies: draft.strategies is authoritative, the DOM is rendered from it -- */
    function renderStrategies() {
      stratWrap.innerHTML = draft.strategies
        .map((s, i) => strategyHTML(s, i, draft.strategies.length)).join("");
    }
    renderStrategies();
    stratWrap.addEventListener("input", (e) => {
      const block = e.target.closest(".sug-strategy"); if (!block) return;
      const s = draft.strategies[+block.dataset.i]; if (!s) return;
      s[e.target.dataset.f] = e.target.value;
      persist();
    });
    stratWrap.addEventListener("click", (e) => {
      const btn = e.target.closest(".sug-remove"); if (!btn) return;
      draft.strategies.splice(+btn.dataset.rm, 1);
      if (!draft.strategies.length) draft.strategies.push(emptyStrategy());
      renderStrategies(); saveDraft(draft);
    });
    $("sugAdd").onclick = () => {
      if (draft.strategies.length >= 10) { toast("Ten strategies is the limit — thank you!"); return; }
      draft.strategies.push(emptyStrategy());
      renderStrategies(); saveDraft(draft);
      stratWrap.lastElementChild?.querySelector("textarea")?.focus();
    };

    /* -- target field: placeholder, validation, preview -- */
    function setPlaceholder() {
      targetIn.placeholder = draft.idType === "smiles"
        ? "CC(=O)Oc1ccccc1C(=O)O" : "e.g. taxol, or a CAS number";
    }
    function showStatus(cls, text) {
      if (!text) { statusEl.hidden = true; statusEl.textContent = ""; return; }
      statusEl.hidden = false; statusEl.className = "sug-status " + cls; statusEl.textContent = text;
    }
    function showPreview(svg) {
      // RDKit renders the SVG from a parsed molecule, so nothing of the raw input survives into
      // markup — but this is still user-derived, so refuse anything that doesn't look like a plain
      // <svg> document rather than trusting that.
      const clean = svg && /^\s*<(\?xml|svg)/i.test(svg) && !/<script/i.test(svg) ? svg : "";
      previewEl.hidden = !clean; previewEl.innerHTML = clean;
    }
    /* -- "we already have routes to this" -- */
    // Purely informational: someone who already knows we have routes may still be suggesting a
    // strategy those routes don't take, so this never touches the submit gate. Only a line of text
    // and a link — the routes themselves stay on the route page.
    function showKnown(hit) {
      if (!hit) { knownEl.hidden = true; knownEl.innerHTML = ""; return; }
      const idxById = {}; hit.routes.forEach((r) => (idxById[r.id] = r));
      const best = bestRouteId(hit.routes.map((r) => r.id), idxById);
      const n = hit.routes.length;
      const what = hit.name ? `<b>${esc(hit.name)}</b>` : "this molecule";
      knownEl.hidden = false;
      knownEl.innerHTML = `SynthAtlas already has ${n} route${n === 1 ? "" : "s"} to ${what}
        (${esc(hit.npaid)}). <a href="#/route/${esc(best)}" id="sugKnownLink">Review them →</a>
        <span class="sug-known-note">Your draft is saved — carry on here if your strategy isn't
        one of them.</span>`;
      // Flush the autosave debounce before navigating away, so the last keystrokes before the
      // click can't be the ones that get lost.
      const link = $("sugKnownLink");
      if (link) link.addEventListener("click", flushDraftNow);
    }
    let knownSeq = 0;
    // Bumping the sequence is what makes this a real clear: an in-flight lookup for a superseded
    // query must not repaint the notice after the field has moved on.
    const clearKnown = () => { knownSeq++; showKnown(null); };
    async function checkKnown(canon) {
      const seq = ++knownSeq;
      let hit = null;
      try { hit = await ChemSearch.findExisting(canon, await jget("data/index.json")); }
      catch { return; }   // index unreachable: no notice, form unaffected
      if (seq === knownSeq) showKnown(hit);
    }

    let checkT, checkSeq = 0;
    async function validateTarget() {
      const raw = draft.target.trim();
      if (draft.idType === "name") { check = { ok: false, canon: "", svg: "" }; showStatus("", ""); showPreview(""); clearKnown(); updateGate(); return; }
      if (!raw) { check = { ok: false, canon: "", svg: "" }; showStatus("", ""); showPreview(""); clearKnown(); updateGate(); return; }
      const seq = ++checkSeq;
      showStatus("busy", "Checking…");
      const r = await ChemSearch.validate(raw);
      if (seq !== checkSeq) return;   // a newer keystroke already superseded this check
      check = { ok: !!r.ok, canon: r.canon || "", svg: r.svg || "" };
      if (r.ok) {
        showStatus("ok", r.unchecked ? "Structure checker unavailable — we'll take it as typed."
          : (r.canon && r.canon !== raw ? `Valid — canonical form ${r.canon}` : "Valid structure"));
        showPreview(r.svg);
        clearKnown();                       // drop the previous molecule's notice first
        if (r.canon && !r.unchecked) checkKnown(r.canon);   // then look this one up, off the critical path
      } else {
        showStatus("err", r.message); showPreview(""); clearKnown();
      }
      updateGate();
    }
    const queueValidate = () => { clearTimeout(checkT); checkT = setTimeout(validateTarget, 300); };

    targetIn.addEventListener("input", () => {
      draft.target = targetIn.value; persist();
      if (draft.idType === "smiles") { showPreview(""); clearKnown(); queueValidate(); }
      updateGate();
    });
    typeSel.addEventListener("change", () => {
      // The typed text is deliberately KEPT across a switch: someone who pasted a name into SMILES
      // mode should be able to flip the dropdown, not retype.
      draft.idType = typeSel.value === "name" ? "name" : "smiles";
      setPlaceholder(); saveDraft(draft);
      showPreview(""); showStatus("", ""); clearKnown();
      if (draft.idType === "smiles") validateTarget(); else updateGate();
    });
    setPlaceholder();

    const showUrlHint = () => {
      const h = targetUrlHint(draft.targetUrl);
      urlStatus.hidden = !h; urlStatus.textContent = h;
    };
    urlIn.addEventListener("input", () => { draft.targetUrl = urlIn.value; showUrlHint(); persist(); });
    showUrlHint();

    feedbackIn.addEventListener("input", () => { draft.feedback = feedbackIn.value; persist(); });
    document.querySelectorAll('input[name="sugContact"]').forEach((r) => {
      r.addEventListener("change", () => { draft.contactOk = r.value === "yes" && r.checked; saveDraft(draft); });
    });

    $("sugFresh").onclick = () => { clearDraft(); viewSuggest(); toast("Draft cleared"); };

    /* -- gating: viewable by anyone; whether submitting needs an account is SUGGEST_NEEDS_AUTH -- */
    function updateGate() {
      const me = SA.identity();
      const hasTarget = !!draft.target.trim();
      const targetOk = hasTarget && (draft.idType === "name" || check.ok);
      const authOk = !!me || !SUGGEST_NEEDS_AUTH;
      submitBtn.disabled = !(authOk && targetOk);
      gateEl.textContent = !authOk ? "Sign in to submit."
        : !hasTarget ? "Enter a target above."
        : !targetOk ? "Fix the SMILES above, or switch to “Common name”."
        : "";
    }
    function renderWho() {
      const me = SA.identity(), box = $("sugWho");
      box.innerHTML = me
        ? `Submitting as <b>${esc(me.email || me.name)}</b> — we record this automatically so you
           don't have to type it.`
        : SUGGEST_NEEDS_AUTH
        ? `<b>You can fill this in now.</b> Submitting needs a signed-in email address, so we know
           who to come back to — your draft is kept while you sign in.
           <button type="button" class="btn btn-ghost btn-sm" id="sugSignIn">Sign in</button>`
        // Signing in is offered, not demanded — it is what puts a verified address on the row and
        // what lets your past suggestions be listed below, so it is worth saying what it buys.
        : `<b>You can fill this in now.</b> Signing in is optional — it records a verified email
           rather than none at all, and keeps your past suggestions listed here.
           <button type="button" class="btn btn-ghost btn-sm" id="sugSignIn">Sign in</button>`;
      const b = $("sugSignIn"); if (b) b.onclick = openSignInModal;
    }
    renderWho(); updateGate();
    if (draft.idType === "smiles" && draft.target.trim()) validateTarget();

    /* -- submit -- */
    $("sugForm").addEventListener("submit", async (e) => {
      e.preventDefault();
      if (submitBtn.disabled) return;
      const smiles = draft.idType === "smiles";
      const payload = {
        identifier_type: draft.idType,
        smiles: smiles ? (check.canon || draft.target.trim()) : null,
        smiles_input: smiles ? draft.target.trim() : null,
        common_name: smiles ? null : draft.target.trim(),
        target_url: draft.targetUrl.trim() || null,
        strategies: draft.strategies
          .map((s) => ({ description: s.description.trim(), reference: s.reference.trim() }))
          .filter((s) => s.description || s.reference),
        site_feedback: draft.feedback.trim() || null,
        contact_ok: !!draft.contactOk,
      };
      submitBtn.disabled = true; gateEl.textContent = "Submitting…";
      try {
        await SA.submitSuggestion(payload);
        // Order matters: drop the pending flush BEFORE clearing, or leaving the tab afterwards
        // would write the just-submitted draft straight back and offer it again as "restored".
        flushSuggestDraft = null;
        clearTimeout(saveT);
        clearDraft();
        showThanks(payload);
      } catch (err) {
        gateEl.textContent = "";
        submitBtn.disabled = false;
        toast(err.message || "Could not submit — please try again.");
      }
    });

    function showThanks(p) {
      const name = p.common_name || p.smiles;
      $("sugForm").outerHTML = `<div class="sug-thanks">
        <h2>Thank you — your suggestion is in.</h2>
        <p>We have recorded <b>${esc(name)}</b>${p.strategies.length
          ? ` and ${p.strategies.length} ${p.strategies.length === 1 ? "strategy" : "strategies"}`
          : ""}.${p.contact_ok ? " We'll be in touch once SynthEx has run it." : ""}</p>
        <p><a class="btn btn-ghost btn-sm" href="#/suggest" id="sugAgain">Suggest another target</a>
           <a class="btn btn-ghost btn-sm" href="#/routes">Browse routes</a></p>
      </div>`;
      document.getElementById("sugRestored")?.setAttribute("hidden", "");
      const again = document.getElementById("sugAgain");
      if (again) again.onclick = (ev) => { ev.preventDefault(); viewSuggest(); };
      renderMine();
    }

    async function renderMine() {
      const box = $("sugMine"); if (!box || !SA.identity()) return;
      let rows = [];
      try { rows = (await SA.mySuggestions()) || []; } catch { return; }
      if (!rows.length) { box.innerHTML = ""; return; }
      const when = (t) => { try { return new Date(t).toLocaleDateString(); } catch { return ""; } };
      const ref = (u) => {
        if (!u) return "";
        const isUrl = /^https?:\/\//i.test(u);
        return ` · ${isUrl ? `<a class="ext" href="${esc(u)}" target="_blank" rel="noopener">reference</a>` : esc(u)}`;
      };
      box.innerHTML = `<details class="sug-mine"><summary>Your suggestions (${rows.length})</summary>
        <ul class="sug-mine-list">${rows.map((r) => {
          const n = (r.strategies || []).length;
          return `<li><b>${esc(r.common_name || r.smiles || "—")}</b>
            <span class="dim">${when(r.created_at)}${n ? ` · ${n} ${n === 1 ? "strategy" : "strategies"}` : ""}${ref(r.target_url)}</span></li>`;
        }).join("")}</ul></details>`;
    }
    renderMine();
  }

  function viewPrivacy() {
    app.innerHTML = `<div class="page"><div class="wrap legal">
      <a class="back" href="#/">← Home</a>
      <h1>Privacy Policy</h1>
      <p class="legal-meta">Last updated 10 August 2026</p>

      <p>SynthAtlas is a research project of the <b>Laboratory of Artificial Chemical
      Intelligence (LIAC)</b> at EPFL (École polytechnique fédérale de Lausanne). This page
      explains what personal data the site processes and why, in line with the Swiss Federal
      Act on Data Protection (FADP/LPD) and the EU GDPR. It complements
      <a href="https://www.epfl.ch/about/overview/regulations-and-guidelines/" target="_blank" rel="noopener">EPFL's institutional data-protection framework</a>.</p>

      <h2>Data controller</h2>
      <p>EPFL — Laboratory of Artificial Chemical Intelligence (LIAC), 1015 Lausanne,
      Switzerland. Contact: <a href="mailto:philippe.schwaller@epfl.ch">philippe.schwaller@epfl.ch</a>.</p>

      <h2>What we process</h2>
      <ul>
        <li><b>Browsing the atlas</b> (routes, reactions, molecules, map): no account is
        required, and we do not store personal data or build any profile of you. We do count
        visits in aggregate — see <i>Audience measurement</i> below.</li>
        <li><b>When you sign in to leave feedback:</b> your <b>email address</b>, a
        <b>display name</b>, and the <b>ratings and comments</b> you submit on routes or
        reaction steps, together with timestamps. Your email is used only to confirm your
        identity and is not shown publicly; your display name is shown next to your feedback.</li>
        <li><b>When you suggest a target</b> (<a href="#/suggest">Suggest a target</a>): the
        structure or name you submit, any reference link, the strategies you describe, any website
        feedback you add, whether you asked to be contacted, and your account <b>email address</b>,
        which is recorded automatically from your sign-in rather than typed. Suggestions are not
        public — only the SynthAtlas team can read them.</li>
      </ul>

      <h2>Audience measurement</h2>
      <p>To understand how much the atlas is used, we run <b>Cloudflare Web Analytics</b>. It is
      cookieless: it sets no cookie, stores no IP address, assigns you no identifier, and cannot
      follow you to any other website. It reports only aggregate counts — visits, page views,
      referring site, browser, and country — and there is no way for us to single you out from
      them. This is why the site shows no cookie banner. We use no advertising or profiling
      analytics of any kind.</p>

      <h2>Purpose and legal basis</h2>
      <p>We collect feedback to evaluate and improve computer-generated synthesis routes with
      input from expert chemists. Processing is based on your consent (given when you create an
      account and submit feedback) and on our legitimate interest in conducting research.</p>

      <h2>Where data is processed</h2>
      <p>The site is served through Cloudflare's global network. Accounts and feedback are
      stored with <b>Supabase in Switzerland (Zurich)</b>. Molecule images are served from
      Cloudflare R2. No data is sold or shared with third parties for advertising.</p>

      <h2>Retention</h2>
      <p>Account and feedback data are kept until you ask us to delete them or until the
      project ends, whichever comes first.</p>

      <h2>Cookies</h2>
      <p>We use only a strictly necessary authentication/session cookie to keep you signed in.
      We do not use tracking, advertising, or third-party analytics cookies.</p>

      <h2>Your rights</h2>
      <p>You may request access to, correction of, or deletion of your personal data, and you
      may withdraw consent at any time, by emailing
      <a href="mailto:philippe.schwaller@epfl.ch">philippe.schwaller@epfl.ch</a>. You also have
      the right to lodge a complaint with the Swiss Federal Data Protection and Information
      Commissioner (FDPIC).</p>

      <h2>Security</h2>
      <p>Data is transmitted over HTTPS/TLS. The database enforces row-level security so each
      user can modify only their own feedback, access is protected by Cloudflare (DDoS
      protection and web application firewall), and privileged credentials are never exposed in
      the browser.</p>
    </div></div>`;
  }

  /* ---------------- glossary ---------------- */
  // Deterministic definitions are transcribed from the code that computes them
  // (site_build/build.py, route_scoring_metric.py, site_build/match_reaxys.py).
  // AI-proposed items are described by their role in the pipeline, not re-defined here.
  function viewGlossary(focus) {
    const chip = (kind) => kind === "computed"
      ? `<span class="gl-chip computed">computed</span>`
      : `<span class="gl-chip ai">AI proposed</span>`;
    const entry = (term, kind, body) =>
      `<div class="gl-entry" id="gl-${term.toLowerCase().replace(/[^a-z0-9]+/g, "-")}">
        <h3>${term} ${chip(kind)}</h3><div class="gl-body">${body}</div></div>`;
    app.innerHTML = `<div class="page"><div class="wrap legal glossary">
      <a class="back" href="#/">← Home</a>
      <h1>Glossary</h1>
      <p class="legal-meta">How every number and term on SynthAtlas is produced.</p>
      <p>Each item below is marked either <span class="gl-chip computed">computed</span> —
      calculated deterministically from the route tree by our build code — or
      <span class="gl-chip ai">AI proposed</span> — suggested by the SynthEx planning agent or
      its automated critic, and not experimentally verified.</p>

      <h2>Metrics</h2>
      ${entry("Overall score (1–9)", "ai",
        `A strategy-quality score for the whole route, assigned by the automated route critic.
         Higher is better.`)}
      ${entry("Green score (1–9)", "ai",
        `A green-chemistry score for the whole route, assigned by the automated route critic.
         Higher is better.`)}
      ${entry("Feasibility score (1–10) & verdict", "ai",
        `A practical-feasibility score and a verdict (excellent, good, flawed, poor, infeasible)
         for the whole route, assigned by the automated route critic. Routes the model originally
         rated “acceptable” are displayed as “flawed”, since on inspection they generally contain
         significant mistakes.`)}
      ${entry("Convergence score (0–100)", "computed",
        `Measures how convergent the route is, independent of any AI judgment. In a fully linear
         route every starting material is dragged through the whole sequence; in an ideally
         convergent route, fragments of similar size are built in parallel and joined late. We
         count, on average, how many steps separate each starting material from the target, and
         place the route on a scale between those two extremes for the same number of starting
         materials: <b>0 = fully linear, 100 = ideally convergent</b>. A route with only one or
         two starting materials scores 100 by definition, and a reactant used twice in the same
         step is only counted once.`)}
      ${entry("Total steps", "computed",
        `The number of reactions in the route tree.`)}
      ${entry("Longest linear path", "computed",
        `The largest number of consecutive reactions on any single branch of the route tree, from
         a starting material to the target. For a fully linear route this equals the total number
         of steps; for a convergent route it is shorter.`)}
      ${entry("Stereo risk", "ai",
        `The critic's judgment of how risky the route's stereochemical outcome is.`)}
      ${entry("Redundant steps", "ai",
        `Pairs of steps the critic flags as redundant (e.g. a protection that is never used).
        `)}
      ${entry("Literature precedent (Reaxys)", "computed",
        `Each proposed reaction is given a structural signature and looked up against the reactions
         recorded in Reaxys. Two bands are reported: a <b>signature match at radius 2</b> — the more
         specific of the two, shown as a <b>close precedent</b> — and a <b>signature match at
         radius 1</b>, which is broader and therefore weaker evidence, shown as a
         <b>related report</b>. A step shows the more specific band when it has one.
         <br><br>
         The number is how many <b>independent documents</b> report the transformation — repeated
         reports being better evidence that a step is established than repeated reactions, which may
         only mean a common pattern. Counts above ten are shown as “>10”, since past that the exact
         figure stops changing the decision; hover the chip for it. The link opens up to twenty
         example reactions in Reaxys (subscription required).
         <br><br>
         A match means the literature contains reactions with the same signature — <b>not</b> that
         this exact substrate has been made. Treat it as evidence that the transformation is
         precedented, and follow the link to judge how closely the examples apply.`)}
      ${entry("Star ratings", "computed",
        `Nothing to do with AI: these are ratings left by users like you, averaged across votes.`)}

      <h2>Definitions</h2>
      ${entry("Route", "ai",
        `A synthetic plan from building blocks to the target natural
         product. Routes are proposed retrosynthetically by the SynthEx agent and displayed in the
         forward direction.`)}
      ${entry("Step", "ai",
        `One chemical transformation in a route. Steps are numbered in the forward direction —
         Step 1 is the first operation at the bench. The reaction itself, its conditions, and the
         accompanying reasoning are all proposed by the AI agent.`)}
      ${entry("Target", "computed",
        `The natural product a route is planned toward, taken from the
         <a href="https://www.npatlas.org/" target="_blank" rel="noopener">NP Atlas</a> database.`)}
      ${entry("Building block", "computed",
        `A starting material: a molecule the route consumes but never makes — a leaf of the route
         tree. Each is sorted into a sourcing category, shown as a colour on the chip:
         <b>buyable</b> — its InChIKey is listed in the eMolecules or ZINC catalogue;
         <b>obtainable</b> — not catalogued, but AiZynthFinder found a short synthesis to
         catalogued starting materials (see “How to obtain it” on the molecule page);
         <b>small</b> — neither of the above, but under 10 heavy atoms and so assumed easy to
         source; and <b>unavailable</b> — larger and not yet sourceable.`)}
      ${entry("Productive route", "computed",
        `A route whose building blocks are all sourceable — each is buyable (in the eMolecules or
         ZINC catalogue), obtainable via a short AiZynthFinder synthesis, or small enough (fewer
         than 10 heavy atoms) to assume easy to source. The Routes gallery shows only productive
         routes by default — untick “Productive only” to see the rest.`)}
      ${entry("Key step", "ai",
        `A step the AI analysis flags as strategically critical to the route — typically the bond
         construction the whole strategy hinges on.`)}
      ${entry("Critic", "ai",
        `An automated reviewer — a second AI pass — that inspects each proposed step and either
         accepts or flags it, with a short assessment. A route where the critic accepted every
         step can still be wrong: the critic is itself a language model.`)}
      ${entry("Conditions & assessment", "ai",
        `Reagents, solvent and temperature proposed by the agent for each step. The small tag next
         to “Conditions” (e.g. “vague”, “adequate”) is the critic's judgment of how completely the
         conditions are specified.`)}
      ${entry("Asymmetric methods", "ai",
        `Steps where the analysis detected an asymmetric method (e.g. catalyst system).
        `)}
      ${entry("Cascade opportunities", "ai",
        `Suggestions from the analysis for combining adjacent steps into one-pot cascades.
        `)}
      ${entry("Top risks", "ai",
        `The critic's ranked list of the most likely failure points of the route, each with a
         suggested mitigation.`)}
    </div></div>`;
    // deep link (#/glossary/<entry-id>): jump to and briefly highlight that definition
    if (focus) {
      const el = document.getElementById("gl-" + focus);
      if (el) { el.scrollIntoView(); el.classList.add("gl-hit"); }
    }
  }

  /* ---------------- router ---------------- */
  function setActive(tab) {
    document.querySelectorAll(".nav .links a").forEach((a) => a.classList.toggle("active", a.dataset.tab === tab));
  }
  async function route() {
    if (location.protocol === "file:") {
      app.innerHTML = `<div class="page"><div class="wrap"><div class="empty">
        <h2 style="margin-bottom:12px">Please run SynthAtlas from a local server</h2>
        <p>The app loads data with <code>fetch()</code>, which browsers block on <code>file://</code>.</p>
        <p style="margin-top:14px">From the project folder, run:</p>
        <pre class="mono" style="background:#fff;border:1px solid var(--line);border-radius:8px;padding:14px;display:inline-block;text-align:left;margin-top:10px">python3 -m http.server 4173 --directory site</pre>
        <p style="margin-top:14px">then open <a href="http://localhost:4173" style="color:var(--accent)">http://localhost:4173</a></p>
      </div></div></div>`;
      return;
    }
    const h = location.hash.replace(/^#\/?/, ""); const [seg, arg, arg2] = h.split("/");
    window.scrollTo(0, 0);
    // Every view is async, so each branch must be AWAITED inside the try — a bare `return
    // viewX()` hands the promise back before it settles and the catch below never runs, which
    // is how a failed data fetch used to leave the page sitting on "Loading the atlas…"
    // forever with nothing but an unhandled rejection in the console.
    try {
      if (!seg) { setActive("home"); return await viewHome(); }
      if (seg === "routes") { setActive("routes"); return await viewRoutes(); }
      if (seg === "route") { setActive("routes"); return await viewRoute(arg, arg2); }
      if (seg === "reactions") { setActive("reactions"); return await viewReactions(); }
      if (seg === "molecules") { setActive("molecules"); return await viewMolecules(); }
      if (seg === "molecule") { setActive("molecules"); return await viewMolecule(arg); }
      // The standalone Map tab is gone — the chemical-space map now lives at the bottom of the
      // landing page. Keep the route so existing links and bookmarks land somewhere sensible
      // instead of 404ing, and scroll to the section once it exists.
      if (seg === "map") {
        setActive("home");
        await viewHome();
        scrollToChemicalSpace();
        return;
      }
      // #/about/team lands on the Team heading rather than the top — the footer credit links
      // straight at the people it names. Same shape as the #/map branch: render, then scroll.
      if (seg === "about") {
        setActive("about");
        await viewAbout();
        if (arg) document.getElementById(arg)?.scrollIntoView({ behavior: "smooth", block: "start" });
        return;
      }
      if (seg === "suggest") { setActive(""); return await viewSuggest(); }
      if (seg === "privacy") { setActive(""); return await viewPrivacy(); }
      if (seg === "glossary") { setActive("glossary"); return await viewGlossary(arg); }
      await viewHome();
    } catch (err) { app.innerHTML = dataErrorHTML(err); }
  }
  // Shown when the dataset can't be loaded. The most likely cause by far is a deployment
  // whose pinned dataVer is not in the R2 bucket (or a bucket CORS policy that doesn't list
  // this origin), so name the version and origin instead of a bare "something went wrong".
  function dataErrorHTML(err) {
    const C = window.SA_CONFIG || {};
    return `<div class="page"><div class="wrap"><div class="empty">
      <h2 style="margin-bottom:12px">Couldn't load the atlas data</h2>
      <p>${esc(err && err.message || String(err))}</p>
      <p style="margin-top:14px">This deployment reads dataset
        <code>${esc(C.dataVer || "?")}</code> from <code>${esc(C.isLocal ? "site/data (local build)" : C.dataBase || "?")}</code>.</p>
      <p style="margin-top:6px">If this is a preview deployment, that dataset version may not be
        published yet, or the R2 bucket's CORS policy may not allow <code>${esc(location.origin)}</code>.</p>
      <p style="margin-top:14px"><a class="back" href="#/">← Home</a></p>
    </div></div></div>`;
  }
  window.addEventListener("hashchange", route);
  route();

  /* ---------------- Cloudflare Turnstile (sign-in challenge) ----------------
   * Guards the one action that costs money and consumes a project-wide quota: sending a sign-in
   * email. Inert unless SA_CONFIG.turnstileSiteKey is set (see the note in config.js).
   *
   * The script is injected on first use rather than from index.html, so browsing the atlas never
   * loads a third-party script — only opening the sign-in dialog does.
   */
  const TURNSTILE_KEY = (window.SA_CONFIG || {}).turnstileSiteKey || "";
  const TURNSTILE_SRC = "https://challenges.cloudflare.com/turnstile/v0/api.js?render=explicit";
  let turnstilePromise = null;
  const loadTurnstile = () => (turnstilePromise ??= new Promise((resolve, reject) => {
    if (window.turnstile) return resolve(window.turnstile);
    const s = document.createElement("script");
    s.src = TURNSTILE_SRC; s.async = true; s.defer = true;
    s.onload = () => (window.turnstile ? resolve(window.turnstile) : reject(new Error("turnstile did not initialise")));
    s.onerror = () => { turnstilePromise = null; reject(new Error("could not load the challenge script")); };
    document.head.appendChild(s);
  }));

  /* Renders the widget into `container` and hands back { token, reset }.
   * `token()` resolves with the current token, waiting for the challenge if it is still running —
   * the challenge is usually invisible and completes in about a second, so gating the button on it
   * would flicker for no reason; instead we let the user click and await the result here.
   * Tokens are SINGLE USE and expire (~5 min), so reset() must run after every send attempt. */
  const TURNSTILE_FAIL_MSG = "The human-verification check failed. Please try again.";
  const TURNSTILE_WAIT_MS = 60000;   // safety net only; a managed challenge may need real interaction
  function mountTurnstile(container) {
    if (!TURNSTILE_KEY || !container) return null;
    let current = null, lastError = null, waiters = [], widgetId = null;
    const settle = (t) => { current = t; lastError = null; waiters.splice(0).forEach((w) => w.resolve(t)); };
    // Record the failure as STATE, don't just notify current waiters. The challenge usually
    // resolves (or fails) before the user gets round to clicking "Send code", so an error with no
    // waiter yet would otherwise be dropped and the next token() call would hang forever.
    const fail = (msg) => { lastError = msg; waiters.splice(0).forEach((w) => w.reject(new Error(msg))); };
    const ready = loadTurnstile().then((ts) => {
      widgetId = ts.render(container, {
        sitekey: TURNSTILE_KEY,
        callback: settle,
        "error-callback": () => fail(TURNSTILE_FAIL_MSG),
        "expired-callback": () => { current = null; ts.reset(widgetId); },
        "timeout-callback": () => { current = null; ts.reset(widgetId); },
      });
    });
    return {
      token: async () => {
        await ready;                       // rejects if the script was blocked — surfaced by the caller
        if (current) return current;
        if (lastError) throw new Error(lastError);
        return new Promise((resolve, reject) => {
          const w = { resolve, reject };
          waiters.push(w);
          // Never leave the dialog wedged on an outcome that never arrives.
          setTimeout(() => {
            const i = waiters.indexOf(w);
            if (i !== -1) { waiters.splice(i, 1); reject(new Error(TURNSTILE_FAIL_MSG)); }
          }, TURNSTILE_WAIT_MS);
        });
      },
      reset: () => {
        current = null; lastError = null;   // a reset means "let them try again"
        if (window.turnstile && widgetId != null) window.turnstile.reset(widgetId);
      },
    };
  }

  /* ---------------- modal shell ----------------
     Overlay, card, the three ways out (✕ / Escape / click outside), a status line and a busy
     toggle. Both dialogs below are built on it, so dismissal and the scroll lock can only be
     right or wrong in one place. Returns null when a modal is already open — never stack two.
     `onClose` runs on every dismissal, however it was triggered. */
  function openModal(title, bodyHTML, { onClose } = {}) {
    if (document.querySelector(".auth-modal")) return null;
    const overlay = document.createElement("div");
    overlay.className = "auth-modal";
    overlay.innerHTML = `
      <div class="auth-card">
        <button class="auth-close" title="Close" aria-label="Close">✕</button>
        <h2 id="authTitle">${esc(title)}</h2>
        ${bodyHTML}
        <div class="auth-msg" id="authMsg" role="status"></div>
      </div>`;
    document.body.appendChild(overlay);
    document.body.style.overflow = "hidden";
    const $ = (s) => overlay.querySelector(s);
    const close = () => {
      overlay.remove(); document.body.style.overflow = ""; document.removeEventListener("keydown", onKey);
      onClose?.();
    };
    function onKey(e) { if (e.key === "Escape") close(); }
    document.addEventListener("keydown", onKey);
    $(".auth-close").onclick = close;
    overlay.addEventListener("click", (e) => { if (e.target === overlay) close(); });
    const msg = $("#authMsg");
    return {
      $, close,
      setTitle: (t) => { $("#authTitle").textContent = t; },
      setMsg: (t, cls) => { msg.textContent = t || ""; msg.className = "auth-msg" + (cls ? " " + cls : ""); },
      busy: (b) => overlay.querySelectorAll("button,input").forEach((el) => (el.disabled = b)),
    };
  }

  /* The display-name field, shared verbatim by the sign-in flow's last step and the standalone
     rename dialog — one field, one cap, one explanation of what the name is for. */
  const nameFieldHTML = (buttonLabel, hint) => `
    <label class="auth-label" for="authName">Display name</label>
    <input type="text" class="auth-input" id="authName" autocomplete="name" maxlength="80">
    <p class="auth-hint">${esc(hint)}</p>
    <button class="btn btn-primary auth-btn" id="authSaveName">${esc(buttonLabel)}</button>`;

  // Prefill for either dialog: the name on the account, or — when no profile row exists and
  // identity() has fallen back to the full email — the local part, since an email is not a name.
  const nameFor = (me, fallbackEmail = "") =>
    (me && me.hasName ? me.name : String((me && me.email) || fallbackEmail).split("@")[0]);

  /* Three-step sign-in dialog: email → code → name.
   *
   * The name step comes LAST, not alongside the code, because it is prefilled with the name the
   * account already has — and `profiles` is readable only by `authenticated` (RLS, 0001_init.sql),
   * so nothing can be looked up from an email until the code has been verified. That is also the
   * point: an anonymous visitor must not be able to turn an email address into a person's name.
   * By then the user is already signed in, so this step is skippable — closing it keeps the
   * existing name, and we only write when the field actually changed.
   */
  function openSignInModal() {
    // Once the code has been verified, dismissing the dialog is a legitimate way to skip the name
    // step, not an abandoned sign-in. Say so, or the user is left wondering whether it took.
    let signedIn = false, finished = false;
    const m = openModal("Sign in to SynthAtlas", `
        <div class="auth-step" data-step="email">
          <p class="auth-sub">Enter your email and we'll send you a sign-in code.</p>
          <input type="email" class="auth-input" id="authEmail" placeholder="you@example.com" autocomplete="email">
          <div class="auth-captcha" id="authCaptcha"></div>
          <button class="btn btn-primary auth-btn" id="authSend">Send code</button>
        </div>
        <div class="auth-step" data-step="code" hidden>
          <p class="auth-sub">Enter the sign-in code sent to <b id="authEmailShown"></b>.</p>
          <input type="text" class="auth-input" id="authCode" placeholder="Sign-in code" inputmode="numeric" autocomplete="one-time-code">
          <button class="btn btn-primary auth-btn" id="authVerify">Sign in</button>
          <button class="auth-alt" id="authBack">Use a different email</button>
        </div>
        <div class="auth-step" data-step="name" hidden>
          <p class="auth-sub">Signed in as <b id="authWho"></b>.</p>
          ${nameFieldHTML("Continue", "Shown next to your ratings and comments. Edit it here, or leave it as it is.")}
        </div>`,
      { onClose: () => { if (signedIn && !finished) toast("Signed in"); } });
    if (!m) return;
    const { $, close, setMsg, busy } = m;
    const finish = (t) => { finished = true; close(); toast(t); };
    let email = "";
    $("#authEmail").focus();
    // null when Turnstile is not configured — then no token is requested or sent
    const captcha = mountTurnstile($("#authCaptcha"));
    const send = async () => {
      email = $("#authEmail").value.trim(); if (!email) { $("#authEmail").focus(); return; }
      busy(true);
      try {
        let captchaToken;
        if (captcha) {
          setMsg("Verifying you're human…");
          captchaToken = await captcha.token();
        }
        setMsg("Sending…");
        const { error } = await SA.signIn(email, captchaToken); if (error) throw error;
        $("#authEmailShown").textContent = email;
        $('[data-step="email"]').hidden = true; $('[data-step="code"]').hidden = false;
        busy(false); $("#authCode").focus();
        setMsg("Code sent — check your email.", "ok");
      } catch (e) {
        busy(false); setMsg(e.message || "Could not send the code.", "err");
      } finally {
        // The token is single-use whether or not the send succeeded, so a retry (or a second
        // code request from the same dialog) needs a fresh challenge.
        captcha?.reset();
      }
    };
    const verify = async () => {
      const code = $("#authCode").value.trim(); if (!code) { $("#authCode").focus(); return; }
      setMsg("Verifying…"); busy(true);
      try {
        const { error } = await SA.verifyOtp(email, code); if (error) throw error;
        signedIn = true;              // onAuthChange has already re-rendered the nav + page behind us
        const me = SA.identity() || {};
        busy(false); setMsg("");
        m.setTitle("You're signed in");
        $("#authWho").textContent = me.email || email;
        // Prefilled, so a returning user sees the name they have and changes it only if they want to.
        $("#authName").value = nameFor(me, email);
        $('[data-step="code"]').hidden = true; $('[data-step="name"]').hidden = false;
        $("#authName").focus(); $("#authName").select();
      } catch (e) { busy(false); setMsg(e.message || "That code didn't work — try again.", "err"); }
    };
    // Only writes when the field actually differs from the name already on the account, so the
    // common "sign in, change nothing" path costs no round trip. A failed write does not undo the
    // sign-in — the user stays on this step and can retry or simply close it.
    const saveName = async () => {
      const name = $("#authName").value.trim();
      if (!name || name === ((SA.identity() || {}).name || "")) return finish("Signed in");
      setMsg("Saving…"); busy(true);
      try { await SA.setDisplayName(name); finish("Signed in as " + name); }
      catch (e) { busy(false); setMsg(e.message || "Could not save your name — you can change it from the top bar.", "err"); }
    };
    $("#authSend").onclick = send;
    $("#authVerify").onclick = verify;
    $("#authSaveName").onclick = saveName;
    $("#authBack").onclick = () => { $('[data-step="code"]').hidden = true; $('[data-step="email"]').hidden = false; setMsg(""); $("#authEmail").focus(); };
    $("#authEmail").addEventListener("keydown", (e) => { if (e.key === "Enter") send(); });
    $("#authCode").addEventListener("keydown", (e) => { if (e.key === "Enter") verify(); });
    $("#authName").addEventListener("keydown", (e) => { if (e.key === "Enter") saveName(); });
  }

  /* Standalone rename dialog, behind the name in the nav. Same field, same prefill and same
     "only write when it changed" rule as the sign-in flow's last step — the only differences are
     that there is no sign-in to report and the caller has to re-render, since changing a name
     does not change the identity SA.onAuthChange watches. */
  function openNameModal() {
    const me = SA.identity(); if (!me) return;
    const m = openModal("Change your name", `
        <div class="auth-step">
          ${me.email ? `<p class="auth-sub">Signed in as <b>${esc(me.email)}</b>.</p>` : ""}
          ${nameFieldHTML("Save", "Shown next to your ratings and comments.")}
        </div>`);
    if (!m) return;
    const { $, close, setMsg, busy } = m;
    const input = $("#authName");
    input.value = nameFor(me);
    input.focus(); input.select();
    const save = async () => {
      const name = input.value.trim();
      if (!name || name === (SA.identity() || {}).name) return close();
      setMsg("Saving…"); busy(true);
      try {
        await SA.setDisplayName(name);
        close(); renderAuth(); route(); toast("Name updated");
      } catch (e) { busy(false); setMsg(e.message || "Could not update your name.", "err"); }
    };
    $("#authSaveName").onclick = save;
    input.addEventListener("keydown", (e) => { if (e.key === "Enter") save(); });
  }

  /* ---------------- account / display name ----------------
     Shows the current display name (click to change it) plus Sign in / Sign out. In Supabase
     mode we also re-render once when the signed-in identity actually changes — note
     SA.onAuthChange fires after every hydrate() too, so we must NOT re-render unconditionally.
     In the localStorage prototype the box is empty until an identity exists (captured by the
     prompt on first rating/comment), then shows the name + a Clear button. */
  function renderAuth() {
    const box = document.getElementById("authbox"); if (!box) return;
    const me = SA.identity();
    if (me) {
      box.innerHTML = `<button class="who-nav" id="sa-name" title="Click to change your display name">${esc(me.name)}</button> `
        + `<button class="btn btn-ghost btn-sm" id="sa-signout">${SA.mode === "supabase" ? "Sign out" : "Clear"}</button>`;
    } else {
      box.innerHTML = SA.mode === "supabase" ? `<button class="btn btn-ghost btn-sm" id="sa-signin">Sign in</button>` : "";
    }
    const nm = document.getElementById("sa-name");
    if (nm) nm.onclick = openNameModal;
    const si = document.getElementById("sa-signin");
    if (si) si.onclick = openSignInModal;
    const so = document.getElementById("sa-signout");
    if (so) so.onclick = async () => {
      try { await SA.signOut(); } catch (e) { toast(e.message); return; }
      renderAuth();
      if (SA.mode !== "supabase") route();   // supabase: onAuthChange handles the re-render
      toast(SA.mode === "supabase" ? "Signed out" : "Cleared");
    };
  }
  renderAuth();
  if (SA.mode === "supabase" && SA.onAuthChange) {
    let lastId = (SA.identity() || {}).email || null;
    SA.onAuthChange(() => {
      renderAuth();
      const cur = (SA.identity() || {}).email || null;
      if (cur !== lastId) { lastId = cur; route(); }
    });
  }

  /* ---------------- mobile nav ----------------
     Below 960px the nav links collapse into a panel behind a menu button. Everything here is
     about not stranding the user once it is open: it closes on navigation, on Escape (returning
     focus to the button that opened it), on a click outside, and when the viewport grows back to
     desktop width — where the panel's CSS stops applying and a lingering `open` class would
     otherwise leave the state inconsistent. */
  (function () {
    const nav = document.querySelector(".nav");
    const btn = document.getElementById("navToggle");
    const panel = document.getElementById("navLinks");
    if (!nav || !btn || !panel) return;

    const isOpen = () => nav.classList.contains("open");
    const setOpen = (open) => {
      nav.classList.toggle("open", open);
      btn.setAttribute("aria-expanded", open ? "true" : "false");
      btn.setAttribute("aria-label", open ? "Close menu" : "Open menu");
    };
    const close = ({ refocus = false } = {}) => {
      if (!isOpen()) return;
      setOpen(false);
      if (refocus) btn.focus();
    };

    btn.addEventListener("click", (e) => { e.stopPropagation(); setOpen(!isOpen()); });
    // Tapping a link navigates; the panel must not stay over the page it just opened.
    panel.addEventListener("click", (e) => { if (e.target.closest("a")) close(); });
    // Hash changes also come from in-page links and the router, not just the panel.
    window.addEventListener("hashchange", () => close());
    document.addEventListener("keydown", (e) => { if (e.key === "Escape") close({ refocus: true }); });
    document.addEventListener("click", (e) => {
      if (isOpen() && !e.target.closest(".nav")) close();
    });
    window.addEventListener("resize", () => {
      if (isOpen() && window.innerWidth > 960) close();
    });
  })();

  /* ---------------- back-to-top ---------------- */
  (function () {
    const btn = document.createElement("button");
    btn.id = "to-top";
    btn.className = "no-print";
    btn.type = "button";
    btn.setAttribute("aria-label", "Back to top");
    btn.title = "Back to top";
    btn.innerHTML = `<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 19V6"/><path d="M6 12l6-6 6 6"/></svg>`;
    btn.addEventListener("click", () => window.scrollTo({ top: 0, behavior: "smooth" }));
    document.body.appendChild(btn);
    const onScroll = () => btn.classList.toggle("show", window.scrollY > 400);
    window.addEventListener("scroll", onScroll, { passive: true });
    onScroll();
  })();
})();
