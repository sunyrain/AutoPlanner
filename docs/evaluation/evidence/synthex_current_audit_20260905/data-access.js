/* SynthAtlas data-access layer — the single seam between the UI and stored
 * identity / ratings / comments.
 *
 * The UI (app.js) only ever talks to `window.SA`. Today the active provider is
 * the localStorage prototype. Switching the whole app to shared, email-confirmed
 * feedback (Phase 2) is meant to be: create the Supabase project, flip
 * config.backend to "supabase", and wire the two `SA.hydrate()` calls noted below
 * — with no other UI changes.
 *
 * Rating/comment KEYS carry their target type as a prefix:
 *   "route:<id>"        a whole route
 *   "rxn:<route>#<idx>" a single reaction step
 * `targetType(key)` derives "route" | "reaction" from that prefix.
 */
(function () {
  const CFG = window.SA_CONFIG || {};
  const MODE = CFG.backend === "supabase" ? "supabase" : "local";
  // provenance: the host a rating/comment was written from (e.g. deploy.synthatlas.pages.dev,
  // synthatlas.xyz, localhost) — lets one shared DB distinguish test vs production feedback.
  const ORIGIN = (typeof location !== "undefined" && location.hostname) || "";
  const targetType = (key) => (String(key).startsWith("rxn:") ? "reaction" : "route");
  // Whether #/suggest demands an account. See the note on `suggestRequiresSignIn` in config.js —
  // both providers below read it, so the two submit paths can never disagree with each other or
  // with the gate the form renders.
  const SUGGEST_NEEDS_AUTH = !!CFG.suggestRequiresSignIn;

  /* ============================ local (browser-only prototype) ============================ */
  const LS = {
    get: (k, d) => { try { return JSON.parse(localStorage.getItem(k)) ?? d; } catch { return d; } },
    set: (k, v) => localStorage.setItem(k, JSON.stringify(v)),
  };
  function ensureIdentityLocal() {
    let id = LS.get("sa_identity", null);
    if (id) return id;
    const name = window.prompt("Add your name to rate & comment (prototype — the next version verifies by email):");
    if (!name) return null;
    const email = window.prompt("Your email (optional — used to confirm identity in the next version):") || "";
    id = { name: name.trim(), email: email.trim() };
    LS.set("sa_identity", id);
    return id;
  }
  const rawRates = (key) => LS.get("sa_rate:" + key, []);
  const rawComments = (key) => LS.get("sa_cmt:" + key, []);

  const local = {
    mode: "local",
    saveMsg: "saved locally",
    ready: Promise.resolve(),
    identity: () => { const id = LS.get("sa_identity", null); return id ? { ...id, hasName: !!id.name } : null; },
    signOut() { localStorage.removeItem("sa_identity"); },
    setDisplayName(name) {
      const id = LS.get("sa_identity", null) || { name: "", email: "" };
      id.name = name; LS.set("sa_identity", id); return true;
    },
    ratingSummary(key) {
      const a = rawRates(key);
      const avg = a.length ? a.reduce((s, r) => s + r.stars, 0) / a.length : 0;
      const me = this.identity();
      const mine = (me && a.find((r) => r.by === me.name) || {}).stars || 0;
      return { avg, n: a.length, mine };
    },
    rate(key, stars) {
      const me = ensureIdentityLocal(); if (!me) return false;
      const arr = rawRates(key).filter((r) => r.by !== me.name);
      arr.push({ by: me.name, stars, at: new Date().toISOString(), origin: ORIGIN });
      LS.set("sa_rate:" + key, arr);
      return true;
    },
    comments: (key) => rawComments(key),
    addComment(key, text, isPublic = true) {
      const me = ensureIdentityLocal(); if (!me) return null;
      const arr = rawComments(key);
      arr.push({ id: "c" + Date.now() + Math.random().toString(36).slice(2, 7),
                 by: me.name, email: me.email, text, at: new Date().toISOString(),
                 is_public: isPublic, mine: true, edited: false, origin: ORIGIN });
      LS.set("sa_cmt:" + key, arr);
      return arr;
    },
    editComment(key, id, text, isPublic) {
      const arr = rawComments(key);
      const c = arr.find((x) => x.id === id); if (!c) return arr;
      c.text = text;
      if (typeof isPublic === "boolean") c.is_public = isPublic;
      c.edited = true; c.updated_at = new Date().toISOString();
      LS.set("sa_cmt:" + key, arr);
      return arr;
    },
    // Target requests & strategy suggestions (#/suggest). Mirrors the supabase methods so the page
    // works unchanged if config.backend is ever flipped back to "local".
    submitSuggestion(payload) {
      // Only *ensures* an identity when one is actually required — prompting for a name would
      // defeat the point of the signed-out path when suggestRequiresSignIn is off.
      const me = SUGGEST_NEEDS_AUTH ? ensureIdentityLocal() : LS.get("sa_identity", null);
      if (SUGGEST_NEEDS_AUTH && !me) throw new Error("sign in to submit");
      const arr = LS.get("sa_suggestions", []);
      const row = { ...payload, id: "s" + Date.now(), email: (me && me.email) || null,
                    origin: ORIGIN, created_at: new Date().toISOString() };
      arr.push(row); LS.set("sa_suggestions", arr);
      return row;
    },
    mySuggestions() {
      return LS.get("sa_suggestions", []).slice().reverse();
    },
    async hydrate() { /* localStorage is already local — nothing to fetch */ },
  };

  /* ============================ supabase (Phase 2 — written, dormant) ============================
   * Reads are served synchronously from an in-memory cache that `hydrate(keys)` fills, so the
   * UI's synchronous render stays unchanged. Phase 2 wiring is exactly two lines:
   *   • in viewRoute():      await SA.hydrate([routeKey, ...stepKeys]) before rendering, and
   *   • in viewReactions():  await SA.hydrate(reactionKeys)
   * then re-render (or render after the await). Writes are async and return a Promise.
   * The tables + RLS these calls assume live in supabase/migrations/0001_init.sql.
   */
  function makeSupabase() {
    let client = null, user = null, profile = null;
    const rateCache = {};   // key -> { avg, n, mine }
    const cmtCache = {};    // key -> [{ by, at, text }]
    const listeners = [];

    async function getClient() {
      if (client) return client;
      if (!CFG.supabaseUrl || !CFG.supabaseAnonKey) throw new Error("Supabase config missing (see site/config.js)");
      const { createClient } = await import("https://esm.sh/@supabase/supabase-js@2");
      client = createClient(CFG.supabaseUrl, CFG.supabaseAnonKey);
      const { data } = await client.auth.getSession();
      user = data.session?.user || null;
      if (user) profile = await fetchProfile(user.id);
      client.auth.onAuthStateChange(async (_e, session) => {
        user = session?.user || null;
        profile = user ? await fetchProfile(user.id) : null;
        listeners.forEach((cb) => cb());
      });
      return client;
    }
    async function fetchProfile(id) {
      const { data } = await client.from("profiles").select("display_name").eq("id", id).maybeSingle();
      return data || null;
    }

    const ready = getClient().catch((e) => console.warn("[SA] supabase init:", e.message));

    return {
      mode: "supabase",
      saveMsg: "saved",
      ready,
      onAuthChange: (cb) => listeners.push(cb),
      // hasName distinguishes "chose a display name" from "we fell back to the email", so the
      // sign-in dialog can tell whether the name it was given is worth writing.
      identity: () => (user
        ? { name: profile?.display_name || user.email, email: user.email, hasName: !!profile?.display_name }
        : null),
      // email-confirmed sign-in via a 6-digit CODE (not a magic link): links are single-use and
      // get pre-consumed by Microsoft Defender Safe Links on @epfl.ch inboxes, so we email the
      // code instead (the Supabase "Magic Link" template must render {{ .Token }}, no URL).
      // signIn sends the code; verifyOtp exchanges it for a session — no redirect, works cross-device.
      // captchaToken: a Cloudflare Turnstile token, when SA_CONFIG.turnstileSiteKey is set and
      // CAPTCHA protection is enabled in Supabase. Undefined otherwise — supabase-js omits the
      // field, which is exactly the pre-Turnstile behaviour.
      async signIn(email, captchaToken) {
        await getClient();
        // `data` is written to raw_user_meta_data only when this call creates the user, so ORIGIN
        // (the hostname) is captured once, at first sign-up — handle_new_user() copies it (and the
        // full email) into profiles. Existing users re-signing in ignore it.
        // The display name is deliberately NOT passed here: the dialog collects it after the code
        // is verified (prefilled from the existing profile) and writes it with setDisplayName, so
        // the same path serves first-time and returning users.
        return client.auth.signInWithOtp({
          email,
          options: { data: { origin: ORIGIN }, captchaToken },
        });
      },
      // No captcha token here on purpose: Supabase enforces CAPTCHA on the endpoints that SEND
      // mail (otp/signup/recover/resend), not on /verify. Verification is already rate-limited by
      // the attempt cap, and a token is single-use — reusing the send token would just fail.
      async verifyOtp(email, token) {
        await getClient();
        const res = await client.auth.verifyOtp({ email, token, type: "email" });
        // Adopt the session here rather than waiting for onAuthStateChange: that listener fetches
        // the profile asynchronously, so a caller reading identity() on the next line would
        // otherwise see the pre-sign-in state. The listener still fires and re-renders.
        if (!res.error && res.data?.user) {
          user = res.data.user;
          profile = await fetchProfile(user.id);
        }
        return res;
      },
      async signOut() { await getClient(); await client.auth.signOut(); },
      async setDisplayName(name) {
        await getClient();
        if (!user) throw new Error("sign in first");
        const { error } = await client.from("profiles").upsert({ id: user.id, display_name: name }, { onConflict: "id" });
        if (error) throw error;
        profile = await fetchProfile(user.id);
        listeners.forEach((cb) => cb());
        return true;
      },

      ratingSummary: (key) => rateCache[key] || { avg: 0, n: 0, mine: 0 },
      async rate(key, stars) {
        await getClient();
        if (!user) throw new Error("sign in to rate");
        const { error } = await client.from("ratings").upsert(
          { user_id: user.id, target_key: key, target_type: targetType(key), score: stars, origin: ORIGIN },
          { onConflict: "user_id,target_key" });
        if (error) throw error;
        await this.hydrate([key]);
        return true;
      },
      comments: (key) => cmtCache[key] || [],
      async addComment(key, text, isPublic = true) {
        await getClient();
        if (!user) throw new Error("sign in to comment");
        const { error } = await client.from("comments").insert(
          { user_id: user.id, target_key: key, target_type: targetType(key), body: text, is_public: isPublic, origin: ORIGIN });
        if (error) throw error;
        await this.hydrate([key]);
        return cmtCache[key];
      },
      // edit own comment; the DB trigger archives the previous body into comment_history
      async editComment(key, id, text, isPublic) {
        await getClient();
        if (!user) throw new Error("sign in first");
        const patch = { body: text };
        if (typeof isPublic === "boolean") patch.is_public = isPublic;
        const { error } = await client.from("comments").update(patch).eq("id", id);
        if (error) throw error;
        await this.hydrate([key]);
        return cmtCache[key];
      },
      /* ---- target requests & strategy suggestions (#/suggest) ----
       * `payload` carries only what the user typed (identifier_type, smiles, smiles_input,
       * common_name, target_url, strategies, site_feedback, contact_ok). Identity is added HERE,
       * never by the form: user_id and email come from the session. The insert policy in
       * 0008_suggestions.sql re-checks both against the verified JWT, so a tampered client cannot
       * file a suggestion under someone else's address.
       *
       * Submitting signed out is refused unless `suggestRequiresSignIn` is off (config.js) — and
       * note that turning it off ALSO needs the anonymous insert policy enabled in the database,
       * which is committed but commented out. See the note in config.js. */
      async submitSuggestion(payload) {
        await getClient();
        if (!user && SUGGEST_NEEDS_AUTH) throw new Error("sign in to submit");
        const row = user
          ? { ...payload, user_id: user.id, email: user.email }
          : { ...payload, user_id: null, email: null };
        const q = client.from("suggestions").insert({ ...row, origin: ORIGIN });
        // The row is read back only when signed in. There is no anonymous SELECT policy — by
        // design, since suggestions are a private channel to the team — so asking `anon` for a
        // representation returns nothing and PostgREST reports that as a failure on an insert that
        // in fact succeeded. Nothing downstream needs the stored row: showThanks() renders the
        // payload it already has.
        const { data, error } = user ? await q.select().single() : await q;
        if (error) throw error;
        return data || row;
      },
      // RLS scopes this to the caller's own rows (suggestions are a private channel to the team,
      // unlike ratings/comments), so no .eq("user_id", …) filter is needed for correctness.
      async mySuggestions() {
        await getClient();
        if (!user) return [];
        const { data, error } = await client.from("suggestions").select("*")
          .order("created_at", { ascending: false });
        if (error) throw error;
        return data || [];
      },

      async hydrate(keys) {
        await getClient();
        const list = [...new Set(keys)].filter(Boolean);
        if (!list.length) return;
        const [{ data: sums }, { data: cmts }] = await Promise.all([
          client.from("rating_summary").select("target_key,avg_score,vote_count").in("target_key", list),
          client.from("comments").select("id,body,created_at,updated_at,edited,target_key,is_public,user_id,profiles(display_name)")
            .in("target_key", list).order("created_at", { ascending: true }),
        ]);
        let mine = {};
        if (user) {
          const { data } = await client.from("ratings").select("target_key,score").eq("user_id", user.id).in("target_key", list);
          (data || []).forEach((r) => (mine[r.target_key] = r.score));
        }
        list.forEach((k) => {
          const s = (sums || []).find((x) => x.target_key === k);
          rateCache[k] = { avg: s ? Number(s.avg_score) : 0, n: s ? s.vote_count : 0, mine: mine[k] || 0 };
          cmtCache[k] = (cmts || []).filter((c) => c.target_key === k)
            .map((c) => ({ id: c.id, by: c.profiles?.display_name || "someone", at: c.created_at, text: c.body,
                           is_public: c.is_public, edited: c.edited, mine: c.user_id === (user && user.id) }));
        });
        listeners.forEach((cb) => cb());
      },
    };
  }

  window.SA = MODE === "supabase" ? makeSupabase() : local;
})();
