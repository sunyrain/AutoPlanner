/* Display-only route translations. Raw jobs, structures and replay events stay intact. */
(function (root) {
  'use strict';
  const STORAGE_KEY = 'autoplanner.route-language';
  const SKIP_FIELDS = new Set(['molecules', 'activities', 'replay', 'condition_predictions',
    'reaction_operations', 'operations']);

  function create(options = {}) {
    const storage = options.storage;
    let locale = options.locale === 'en' ? 'en' : 'zh-CN';
    try {
      const saved = storage?.getItem(STORAGE_KEY);
      if (saved === 'en' || saved === 'zh-CN') locale = saved;
    } catch (_) { /* Browser storage can be disabled. */ }
    let dictionary = new Map(), fields = new Set(), cache = new WeakMap();
    function text(value) {
      return locale === 'zh-CN' && typeof value === 'string'
        ? (dictionary.get(value) ?? value) : value;
    }
    function translateField(value) {
      if (!Array.isArray(value)) return text(value);
      const result = value.map(text);
      return result.some((item, index) => item !== value[index]) ? result : value;
    }
    function visit(value) {
      if (!value || typeof value !== 'object') return value;
      if (cache.has(value)) return cache.get(value);
      const result = Array.isArray(value) ? [] : {};
      let changed = false;
      for (const [key, child] of Object.entries(value)) {
        const localized = fields.has(key) ? translateField(child)
          : SKIP_FIELDS.has(key) ? child : visit(child);
        Object.defineProperty(result, key, {value: localized, enumerable: true,
          configurable: true, writable: true});
        changed ||= localized !== child;
      }
      const view = changed ? result : value;
      cache.set(value, view);
      cache.set(view, view);
      return view;
    }
    return {
      get locale() { return locale; },
      setLocale(value) {
        locale = value === 'en' ? 'en' : 'zh-CN';
        try { storage?.setItem(STORAGE_KEY, locale); } catch (_) { /* Optional preference. */ }
      },
      setDictionary(payload) {
        if (payload?.locale !== 'zh-CN' || !Array.isArray(payload.text_fields)
          || !payload.translations || typeof payload.translations !== 'object') {
          throw new Error('Invalid route translation asset');
        }
        dictionary = new Map(Object.entries(payload.translations)
          .filter(([, value]) => typeof value === 'string' && value.trim()));
        fields = new Set(payload.text_fields);
        cache = new WeakMap();
      },
      text,
      view(value) { return locale === 'zh-CN' && dictionary.size ? visit(value) : value; }
    };
  }
  const api = {create};
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.RouteLanguage = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
