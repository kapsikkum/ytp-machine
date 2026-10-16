/* Layout only: part controls remain the original, single source of truth. */
window.createYtpmvWorkbench = function (editor) {
  const el = id => document.getElementById(id);
  const safe = text => String(text ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const node = (tag, id, html = '') => { const n = document.createElement(tag); if (id) n.id = id; n.innerHTML = html; return n; };
  let selected = null, analysis = null, tab = 'part';
  const workspace = el('parts-card');
  const tools = node('details', 'track-tools', '<summary>Find & manage tracks</summary>');
  tools.append(el('parts-tools')); workspace.prepend(tools);
  const timeline = node('div', 'track-lanes');
  timeline.setAttribute('role', 'group'); timeline.setAttribute('aria-label', 'MIDI arrangement');
  workspace.append(timeline);
  const tabs = node('div', 'editor-tabs'); tabs.setAttribute('role', 'tablist'); tabs.setAttribute('aria-label', 'Workbench controls');
  const dock = node('div', 'workbench-dock');
  const panels = {};
  for (const [key, title] of [['part','Part'], ['song','Song & render'], ['video','Video preview']]) {
    const button = node('button', `tab-${key}`, title); button.type = 'button'; button.dataset.tab = key;
    button.setAttribute('role', 'tab'); button.setAttribute('aria-controls', `panel-${key}`);
    button.addEventListener('click', () => setTab(key)); tabs.append(button);
    const panel = node('section', `panel-${key}`); panel.setAttribute('role', 'tabpanel'); panel.setAttribute('aria-labelledby', button.id);
    panels[key] = panel; dock.append(panel);
  }
  panels.part.append(el('parts'));
  const empty = node('div', 'no-track', 'Select a track to edit its voice or chip instrument.'); panels.part.append(empty);
  panels.song.append(el('machine'), el('options'), el('bulk-tools'), el('how'));
  const composition = node('div', 'composition-preview');
  const caption = node('div', 'composition-heading', '<b>Video composition</b><span>16:9 · fill frame · centred crop</span>');
  panels.video.append(caption, composition, el('render-card'));
  workspace.append(tabs, dock);
  const footer = node('div', 'workbench-footer'); footer.append(el('save-state')); workspace.append(footer);
  // The loaded project replaces the upload area; choosing a file remains one click.
  document.querySelector('body > .card')?.setAttribute('id', 'midi-import');
  const projectTools = node('div', 'project-tools');
  const upload = node('button', 'change-midi', 'Import MIDI'); upload.type = 'button'; upload.addEventListener('click', () => el('file').click());
  const library = node('button', 'open-midi-library', 'MIDI library'); library.type = 'button';
  library.setAttribute('aria-controls', 'library'); library.setAttribute('aria-expanded', 'false');
  library.addEventListener('click', () => { el('library').open = !el('library').open; library.setAttribute('aria-expanded', String(el('library').open)); });
  projectTools.append(upload, library); el('song').append(projectTools);
  el('song-desc').hidden = true;
  const renderButton = el('render-btn'); projectTools.append(renderButton);
  el('expand-parts').textContent = 'Expand part settings';
  el('expand-parts').hidden = true; // The selected part still has its settings button.

  function setTab(key) {
    tab = key;
    for (const [name, panel] of Object.entries(panels)) {
      panel.hidden = name !== key;
      if (panel.hidden) panel.querySelectorAll('video').forEach(video => video.pause());
      el(`tab-${name}`).setAttribute('aria-selected', String(name === key));
    }
    if (key === 'video') drawComposition();
  }
  function select(id, focus = false) {
    selected = id; setTab('part'); refresh();
    if (focus) el('tab-part').focus();
  }
  function drawTimeline(song) {
    const duration = Math.max(.001, song.duration);
    const ruler = `<div class="track-ruler"><span>Tracks</span><button type="button" class="time-ruler" aria-label="Set preview start on the song ruler">${[0,.25,.5,.75,1].map(f => `<span>${formatTime(f * duration / editor.speed())}</span>`).join('')}</button></div>`;
    timeline.innerHTML = ruler + song.parts.map(p => {
      const activity = p.activity || [];
      const pitches = activity.map(n => n[2]);
      const low = Math.min(...pitches, 48), high = Math.max(...pitches, 72);
      const marks = activity.map(([start, length, pitch]) => `<rect x="${Math.max(0,start / duration * 1000).toFixed(2)}" y="${(8 + (high-pitch) / Math.max(1,high-low) * 24).toFixed(2)}" width="${Math.max(1,Math.min(duration-start,length) / duration * 1000).toFixed(2)}" height="3" rx=".5"/>`).join('');
      return `<div class="track" data-track="${safe(p.id)}" style="--track-color:var(${p.is_drums?'--green':p.role==='bass'?'--amber':p.role==='chords'?'--violet':'--blue'})">
        <div class="track-head"><button type="button" data-select="${safe(p.id)}" class="track-name"><b>${safe(p.instrument)}</b><small data-track-source></small></button>
        <div class="track-cast"><select data-track-corpus="${safe(p.id)}" aria-label="Corpus for ${safe(p.instrument)} track"></select><button type="button" data-select="${safe(p.id)}" class="track-tone" hidden></button></div>
        <div class="track-switches"><button type="button" data-track-flag="mute" aria-label="Mute ${safe(p.instrument)}">M</button><button type="button" data-track-flag="solo" aria-label="Solo ${safe(p.instrument)}">S</button></div></div>
        <button type="button" class="track-region" data-select="${safe(p.id)}" aria-label="Select ${safe(p.instrument)} MIDI part"><span data-track-caption></span><svg viewBox="0 0 1000 44" preserveAspectRatio="none" role="img" aria-label="${safe(p.instrument)} MIDI activity overview">${marks}</svg></button></div>`;
    }).join('');
  }
  function updateTrack(id) {
    const song = editor.song(); if (!song) return;
    const p = song.parts.find(p => p.id === id), s = editor.parts()[id], r = editor.row(id);
    const track = [...timeline.querySelectorAll('.track')].find(t => t.dataset.track === id);
    if (!track || !p || !r) return;
    const synth = editor.onChip(p, s), source = r.querySelector('[data-el=source-note]').textContent;
    track.classList.toggle('muted', !!s.mute); track.classList.toggle('solo', !!s.solo);
    track.querySelector('[data-track-source]').textContent = synth ? editor.toneName(p,s) : editor.corpusName(s);
    track.querySelector('[data-track-caption]').textContent = synth ? editor.toneName(p,s) : (s.text || 'Choose a sample');
    track.querySelector('.track-tone').textContent = editor.toneName(p,s);
    track.querySelector('.track-tone').hidden = !synth;
    const select = track.querySelector('select'), original = r.querySelector('[data-k=corpus]');
    select.hidden = synth || editor.synth();
    const options = [...original.options].map(o => `<option value="${safe(o.value)}">${safe(o.value ? o.textContent.replace(/ \(\d+ words\)$/, '') : 'Page voice')}</option>`).join('');
    if (select.innerHTML !== options) select.innerHTML = options;
    select.value = original.value; select.disabled = r.getAttribute('aria-busy') === 'true';
    track.querySelectorAll('[data-track-flag]').forEach(b => { b.setAttribute('aria-pressed', String(!!s[b.dataset.trackFlag])); b.disabled = select.disabled; });
    track.querySelector('.track-name').title = `${p.instrument} · ${source}`;
  }
  function refresh() {
    const song = editor.song(); if (!song) return;
    if (song !== analysis) {
      analysis = song; selected = song.parts[0]?.id; drawTimeline(song);
      el('library').open = false; library.setAttribute('aria-expanded','false');
    }
    const shown = song.parts.filter(p => editor.row(p.id)?.dataset.filtered !== 'true');
    if (!shown.some(p => p.id === selected)) selected = shown[0]?.id;
    for (const p of song.parts) {
      const r = editor.row(p.id), track = [...timeline.querySelectorAll('.track')].find(t => t.dataset.track === p.id);
      if (!r || !track) continue;
      r.hidden = p.id !== selected; track.hidden = r.dataset.filtered === 'true';
      if (r.hidden) r.querySelectorAll('video').forEach(video => video.pause());
      track.classList.toggle('selected', p.id === selected);
      track.querySelectorAll('[data-select]').forEach(b => b.setAttribute('aria-pressed', String(p.id === selected)));
      updateTrack(p.id);
    }
    empty.hidden = !!selected;
    const songSeconds = Math.max(.001, song.duration / editor.speed());
    timeline.style.setProperty('--playhead', `${Math.min(100,Number(el('preview-from').value)/songSeconds*100)}%`);
    timeline.querySelectorAll('.time-ruler span').forEach((span,i) => span.textContent = formatTime(i/4*songSeconds));
    el('render-card').hidden = false;
    el('library').classList.add('loaded-project');
    drawComposition();
  }
  function drawComposition() {
    const song = editor.song(); if (!song) return;
    let parts = song.parts.filter(p => editor.parts()[p.id].visible && (editor.onChip(p,editor.parts()[p.id]) || editor.parts()[p.id].text));
    if (!parts.length) parts = song.parts.filter(p => !editor.parts()[p.id].mute).slice(0,1);
    const rows = Math.ceil(parts.length / Math.ceil(Math.sqrt(parts.length || 1)));
    let index = 0;
    composition.innerHTML = Array.from({length:rows}, (_,r) => {
      const count = Math.floor(parts.length/rows)+(r<parts.length%rows?1:0);
      return `<div class="composition-row">${parts.slice(index,index+=count).map(p=>{
        const s=editor.parts()[p.id]; return `<button type="button" data-composition-part="${safe(p.id)}"><b>${safe(editor.onChip(p,s)?editor.toneName(p,s):s.text)}</b><span>${safe(p.instrument)}</span></button>`;
      }).join('')}</div>`;
    }).join('') || '<span class="composition-empty">No visible tiles</span>';
  }
  timeline.addEventListener('click', event => {
    const pick = event.target.closest('[data-select]'); if (pick) select(pick.dataset.select);
    const flag = event.target.closest('[data-track-flag]'); if (flag) {
      const id = flag.closest('.track').dataset.track; select(id);
      const control = editor.row(id).querySelector(`[data-k=${flag.dataset.trackFlag}]`);
      control.checked = !control.checked; control.dispatchEvent(new Event('input',{bubbles:true}));
    }
    const ruler = event.target.closest('.time-ruler'); if (ruler) {
      const rect=ruler.getBoundingClientRect(), song=editor.song();
      const fraction = Math.max(0, Math.min(1, (event.clientX-rect.left) / rect.width));
      el('preview-from').value = Math.round(fraction * song.duration / editor.speed());
      el('preview-from').dispatchEvent(new Event('change',{bubbles:true})); refresh();
    }
  });
  timeline.addEventListener('change', event => { if(event.target.matches('[data-track-corpus]')) editor.selectCorpus(event.target.dataset.trackCorpus,event.target.value); });
  composition.addEventListener('click', event => { const b=event.target.closest('[data-composition-part]'); if(b)select(b.dataset.compositionPart); });
  el('preview-from').addEventListener('input',refresh);
  el('opt-speed').addEventListener('change',refresh);
  function formatTime(seconds) { return `${Math.floor(seconds/60)}:${String(Math.floor(seconds%60)).padStart(2,'0')}`; }
  setTab('part');
  return {refresh, updateTrack, select, setTab, get selected(){return selected;}, get tab(){return tab;}};
};
