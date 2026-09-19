// What every page needs, once. Loaded before each page's own script.

const $ = (id) => document.getElementById(id);

// For anything put into innerHTML. Quotes both ways, so a value is safe inside
// either kind of attribute.
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, c =>
  ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

// localStorage that never throws: private windows and blocked storage just
// forget, instead of taking the page down with them.
const store = {
  get(k, d) { try { const v = JSON.parse(localStorage.getItem(k)); return v ?? d; } catch { return d; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch {} },
};
