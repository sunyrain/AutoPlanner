/* SynthAtlas runtime config — safe to commit.
 *
 * `backend`  : "local"  → ratings/comments live in the browser (localStorage prototype).
 *              "supabase" → shared, email-confirmed feedback via Supabase.
 * `supabaseUrl` / `supabaseAnonKey`: your Supabase project's URL + anon (public) key.
 *   The anon key is DESIGNED to be public — row-level-security in the database is what
 *   protects the data, so committing it here is expected and safe. NEVER put the
 *   service_role key in this file (it bypasses RLS).
 */
window.SA_CONFIG = {
  backend: "supabase",
  supabaseUrl: "https://xizqncmluvdpddwncilj.supabase.co",
  // Supabase "publishable" key (the client-safe replacement for the anon key). Public by
  // design — row-level security in the database is what protects the data.
  supabaseAnonKey: "sb_publishable_W8KzBzGOki_k-3rlNOo93w_2a-FYhVk",

  /* Does #/suggest require an account to submit?
   *
   *   true (current) — the form renders for anyone, but submitting is gated behind the sign-in
   *                    dialog. Every row then carries a VERIFIED address: the client sends the
   *                    session's email and the insert policy rejects anything that is not the JWT
   *                    claim, so a suggestion cannot be filed under someone else's name.
   *   false          — anyone can submit. user_id and email are null, so rows are unattributable,
   *                    and "your suggestions" stays empty because there is no one to scope it to.
   *
   * The switch reaches exactly two places: the submit gate + signed-out copy in viewSuggest()
   * (app.js), and the guard in submitSuggestion() (data-access.js).
   *
   * ⚠ SETTING THIS TO false IS NOT SUFFICIENT ON ITS OWN. The database has no anonymous insert
   * path yet — `suggestions_insert_anon` is committed in 0008_suggestions.sql but COMMENTED OUT,
   * so an anonymous submit would fail RLS. Enabling it means uncommenting that policy and
   * re-running the migration (it is written to be safe to run twice). It is parked rather than
   * applied because granting `anon` insert on a table is a spam surface, and there is no reason to
   * open one for a path the site does not currently use. If it is ever opened and abused, the
   * lever is Turnstile — `turnstileSiteKey` below is already provisioned, just not wired to this
   * form. See supabase/README.md.
   */
  suggestRequiresSignIn: true,

  /* ---------------------------------------------------------------- generated assets
   * The generated dataset (site/data/) and the 36k molecule SVGs (site/img/mol/) are NOT
   * committed and NOT part of the Cloudflare Pages deploy — they are published to R2 and
   * fetched from there. Two consequences worth knowing:
   *
   *   • a Pages deploy is ~12 code files, so EVERY branch / PR preview loads the atlas
   *     with no data-committing step (this is what used to leave previews stuck on
   *     "Loading the atlas…"), and
   *   • data lives under an immutable, versioned prefix, so a preview that rebuilt the
   *     dataset publishes its own `dataVer` instead of overwriting production's.
   *
   * `dataVer` pins which published dataset THIS checkout reads. site_build/build.py
   * computes it and writes it to site/data/manifest.json; ./scripts/upload-r2-data.sh
   * uploads that build to R2 and rewrites the line below to match. A branch may pin an
   * older or newer dataset deliberately — that is the whole point of the version.
   */
  dataVer: "20260809-00e8823-5a1cf6",
  dataBase: "https://data.synthatlas.xyz",   // R2 bucket `synthatlas-data`, public custom domain
  imgBaseRemote: "https://img.synthatlas.xyz/mol",   // R2 bucket `synthatlas-img`

  /* Cloudflare Turnstile on the sign-in form. Every sign-in sends an email, which costs money
   * and draws down a project-wide hourly cap in Supabase — so without a challenge, scripted
   * address submissions can either run up the bill or lock out every real user.
   *
   * This is the PUBLIC site key (safe to commit). The matching SECRET key goes only into
   * Supabase → Authentication → Attack Protection → CAPTCHA protection; never into this repo.
   *
   * Empty string = disabled: no script is loaded, no widget renders, and no token is sent, so
   * the app behaves exactly as it did before. Set it to activate.
   *
   * ORDER MATTERS when turning this on. Supabase rejects auth requests that arrive without a
   * token as soon as CAPTCHA protection is enabled, so:
   *   1. set this key and DEPLOY, then
   *   2. enable CAPTCHA protection in Supabase.
   * Doing it the other way round breaks sign-in for everyone in between.
   *
   * When creating the widget, note that Turnstile's hostname list takes PLAIN hostnames — it
   * rejects `*.synthatlas.pages.dev`, unlike scripts/r2-cors.json where that wildcard is valid.
   * See "Turnstile" in docs/DEPLOYMENT.md.
   */
  turnstileSiteKey: "0x4AAAAAAEGPt-yzLNdwbGe4",

  /* Cloudflare Web Analytics — cookieless audience measurement (visits, page views, referrers,
   * countries). No cookie, no cross-site identifier, no consent banner required, which is why it
   * is this and not Google Analytics on an EPFL site.
   *
   * This is the PUBLIC site token from Cloudflare → Web Analytics → Manage site → the JS snippet.
   * Safe to commit: it only says which site a hit belongs to and grants no read access to the
   * numbers. Empty string = disabled: no script is loaded and nothing is sent, exactly as before.
   *
   * ⚠ DO NOT ALSO ENABLE the Pages toggle (Workers & Pages → the project → Metrics → Web
   * Analytics → Enable). That injects a second beacon into every HTML response on the next
   * deploy, and every visit gets counted twice — which is the one failure mode you cannot spot
   * from the dashboard, because doubled numbers look plausible. Pick one. This file is the one
   * that is version-controlled and that skips localhost.
   *
   * ⚠ EXPECT VISITS, NOT PER-ROUTE PAGE VIEWS. Cloudflare's beacon detects in-app navigation by
   * patching History API pushState and listening for popstate; it explicitly does not support
   * hash routers, and SynthAtlas routes entirely on `#/…` (the `hashchange` listener in app.js).
   * So opening twenty routes in one session is one page view, not twenty. Per-route engagement
   * has to come from the Supabase ratings/comments/suggestions tables instead.
   */
  webAnalyticsToken: "04164af6d2bc478e94449209c2e07c75",
};

/* Resolve where data + images are actually read from.
 *
 * Local development (localhost, a LAN IP, *.local, file://) serves both straight out of the
 * working tree — `site/data/` and `site/img/mol/` as produced by ./build.sh — so a developer
 * sees their own build with no upload step. EVERY other host, including hosts we have never
 * heard of, reads from R2.
 *
 * That default direction matters: the previous version allowlisted the known production
 * hostnames and fell back to same-origin otherwise, so any new domain silently served
 * `img/mol` — which is not in the deploy — and every structure on the page came up blank.
 * Defaulting to R2 fails visibly at worst (a CORS error naming the missing origin) instead
 * of shipping an empty-looking site.
 */
(function (C) {
  const h = (typeof location !== "undefined" && location.hostname) || "";
  let isLocal =
    (typeof location !== "undefined" && location.protocol === "file:") || !h ||
    h === "localhost" || h === "127.0.0.1" || h === "::1" || h === "[::1]" ||
    h.endsWith(".local") ||
    /^10\./.test(h) || /^192\.168\./.test(h) || /^172\.(1[6-9]|2\d|3[01])\./.test(h);

  /* Two query-string overrides, for debugging a deployment and for testing the remote path
   * from a local checkout:
   *   ?dataRemote=1        read from R2 even on localhost (verifies bucket + CORS + version)
   *   ?dataVer=<version>   read a different published dataset than this build pins
   * Deliberately NOT overridable: `dataBase`. A link that could repoint the app at an
   * arbitrary host would let someone feed a SynthAtlas URL data they control, so overrides
   * can only ever select a version WITHIN the configured bucket.
   */
  try {
    const q = new URLSearchParams(location.search);
    if (q.get("dataRemote") === "1") isLocal = false;
    const v = q.get("dataVer");
    if (v && /^[A-Za-z0-9._-]{1,64}$/.test(v)) { C.dataVer = v; isLocal = false; }
  } catch { /* no URLSearchParams / no location — keep the computed default */ }

  C.isLocal = isLocal;

  // "data/routes/x.json" -> same-origin path locally, else <dataBase>/<dataVer>/routes/x.json.
  // Callers keep passing the logical "data/…" path, so the fetch cache stays keyed by it.
  C.dataUrl = (p) => {
    const rel = String(p).replace(/^\/?data\//, "");
    if (isLocal) return "data/" + rel;
    return `${String(C.dataBase).replace(/\/+$/, "")}/${C.dataVer}/${rel}`;
  };

  // Molecule SVG base. app.js falls back to same-origin "img/mol" when this is empty.
  C.imgBase = isLocal ? "" : C.imgBaseRemote;
})(window.SA_CONFIG);

/* Load the Cloudflare Web Analytics beacon, if a token is configured.
 *
 * Injected from here rather than hardcoded into index.html so the token sits with the rest of the
 * runtime config, and so the two skip conditions below live next to it instead of being invisible
 * markup in the page head.
 *
 * Skipped when `isLocal` — a developer reloading the atlas a hundred times a day would otherwise
 * land in the launch numbers, and those numbers are going into a pitch. `isLocal` is computed by
 * the IIFE above, so this block has to stay after it.
 *
 * Skipped when the token is empty, which is what makes PR previews silent by default: a preview
 * deploy shares this file, and preview traffic is us.
 *
 * Runs at end-of-body (index.html loads config.js there), so document.head already exists.
 */
(function (C) {
  if (!C.webAnalyticsToken || C.isLocal) return;
  const s = document.createElement("script");
  // `type="module"` mirrors the snippet Cloudflare hands out verbatim. The file itself is a
  // classic script (no import/export) served with `access-control-allow-origin: *`, so plain
  // `defer` works identically today — matching the vendor is purely so this does not drift if
  // they ever do ship module syntax. Module scripts are deferred by default; no `defer` needed.
  s.type = "module";
  s.src = "https://static.cloudflareinsights.com/beacon.min.js";
  s.setAttribute("data-cf-beacon", JSON.stringify({ token: C.webAnalyticsToken }));
  document.head.appendChild(s);
})(window.SA_CONFIG);
