// What every page needs, once. Loaded before each page's own script.

const $ = (id) => document.getElementById(id);

// For anything put into innerHTML. Quotes both ways, so a value is safe inside
// either kind of attribute.
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, c =>
  ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

// Ignore a slow response after the user has moved to another source/word.
function latestRequest() {
  let sequence = 0;
  return {
    next: () => ++sequence,
    current: token => token === sequence,
    cancel: () => { ++sequence; },
  };
}

function drawMeasuredSpectrogram(ctx, data, w, h, gain, color) {
  if (!data || !data.values.length) return;
  const rows = data.values, nFreq = data.bins;
  const colW = Math.max(1, w / rows.length), rowH = h / nFreq;
  const boost = 20 * Math.log10(Math.max(0.01, gain)) / (data.db_max - data.db_min);
  rows.forEach((row, i) => row.forEach((value, f) => {
    ctx.fillStyle = color(value === 0 ? 0 : Math.min(1, Math.max(0, value / 255 + boost)));
    ctx.fillRect(i * w / rows.length, h - (f + 1) * rowH, colW, Math.ceil(rowH));
  }));
}

// localStorage that never throws: private windows and blocked storage just
// forget, instead of taking the page down with them.
const store = {
  get(k, d) { try { const v = JSON.parse(localStorage.getItem(k)); return v ?? d; } catch { return d; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch {} },
};
