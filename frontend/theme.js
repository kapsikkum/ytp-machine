// The one bit of the shared look CSS cannot do alone: WebKit has no way to
// colour a slider's track up to its thumb, so each slider carries the fill as
// --p, kept up to date here for every range input on the page -- including
// ones a page draws after load, and ones its own script moves.
(() => {
  const fill = (el) => {
    const min = Number(el.min || 0), max = Number(el.max || 100);
    const p = max > min ? (Number(el.value) - min) / (max - min) * 100 : 0;
    el.style.setProperty('--p', `${Math.min(100, Math.max(0, p))}%`);
  };
  const all = () => document.querySelectorAll('input[type=range]').forEach(fill);
  document.addEventListener('input', (e) => { if (e.target.type === 'range') fill(e.target); }, true);
  document.addEventListener('change', (e) => { if (e.target.type === 'range') fill(e.target); }, true);
  new MutationObserver(all).observe(document.documentElement, { childList: true, subtree: true });
  // A script setting .value does not fire an event; catch up now and then.
  setInterval(all, 500);
  document.addEventListener('DOMContentLoaded', all);
})();
