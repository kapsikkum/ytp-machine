/* Layout only: part controls remain the original, single source of truth. */
window.createYtpmvWorkbench = function (editor) {
  const el = id => document.getElementById(id);
  const safe = text => String(text ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const node = (tag, id, html = '') => { const n = document.createElement(tag); if (id) n.id = id; n.innerHTML = html; return n; };
  let selected = null, analysis = null, tab = 'part';
  const workspace = el('parts-card');
  // Keep project sound defaults in view alongside the arrangement, regardless
  // of which inspector tab is open. Reuse the original controls and handlers.
  const sound = node('section', 'song-sound'); sound.className = 'card';
  sound.setAttribute('aria-label', 'Song sound defaults');
  const soundTitle = node('div', 'sound-title', '<b>Song sound</b><small>Used unless a track overrides</small>');
  const voiceLabel = el('voice-select').closest('label');
  voiceLabel.firstChild.textContent = 'Default voice ';
  voiceLabel.title = 'The voice inherited by tracks using the song default. Individual track voices stay as chosen.';
  const help = node('details', 'sound-help', '<summary>About defaults</summary><p>Tracks set to Auto follow the global chip. Choose an instrument in the selected track to override it. Voice menus let each track follow the default voice or use another installed voice.</p>');
  help.append(el('chip-about'));
  sound.append(soundTitle, voiceLabel, el('machine'), el('voice-note'), help);
  el('transport').before(sound);
  const tools = node('details', 'track-tools', '<summary>Find & manage tracks</summary>');
  tools.append(el('parts-tools')); workspace.prepend(tools);
  const timeline = node('div', 'track-lanes');
  timeline.setAttribute('role', 'group'); timeline.setAttribute('aria-label', 'MIDI arrangement');
  workspace.append(timeline);
  const timelineView = createYtpmvTimeline(timeline, {speed:editor.speed, mount:updateTrack,
    select:id=>select(id), stop:editor.stop, preview:editor.preview, notes:editor.notes});
  const tabs = node('div', 'editor-tabs'); tabs.setAttribute('role', 'tablist'); tabs.setAttribute('aria-label', 'Workbench controls');
  const dock = node('div', 'workbench-dock');
  const panels = {};
  for (const [key, title] of [['part','Track'], ['song','Song & render'], ['video','Video preview']]) {
    const button = node('button', `tab-${key}`, title); button.type = 'button'; button.dataset.tab = key;
    button.setAttribute('role', 'tab'); button.setAttribute('aria-controls', `panel-${key}`);
    button.addEventListener('click', () => setTab(key)); tabs.append(button);
    const panel = node('section', `panel-${key}`); panel.setAttribute('role', 'tabpanel'); panel.setAttribute('aria-labelledby', button.id);
    panels[key] = panel; dock.append(panel);
  }
  panels.part.append(el('parts'));
  const empty = node('div', 'no-track', 'Select a track to edit its voice or chip instrument.'); panels.part.append(empty);
  panels.song.append(el('options'), el('bulk-tools'), el('how'));
  const composition = node('div', 'composition-preview');
  const caption = node('div', 'composition-heading', '<b>Video composition</b><span id="composition-framing">16:9 · whole clips</span>');
  caption.append(el('video-layout-controls'));
  const compositionStatus=node('p','composition-status','Drag tiles to arrange · Alt + ← / → moves the focused tile');
  compositionStatus.setAttribute('role','status');
  panels.video.append(caption, composition, compositionStatus, el('render-card'));
  workspace.append(tabs, dock);
  const footer = node('div', 'workbench-footer'); footer.append(el('save-state')); workspace.append(footer);
  // The loaded project replaces the upload area; choosing a file remains one click.
  document.querySelector('body > .card')?.setAttribute('id', 'midi-import');
  const projectTools = node('div', 'project-tools');
  const upload = node('button', 'change-midi', 'Import MIDI'); upload.type = 'button'; upload.addEventListener('click', () => el('file').click());
  projectTools.append(upload); el('song').append(projectTools);
  el('song-desc').hidden = true;
  const renderButton = el('render-btn'); projectTools.append(renderButton);
  el('expand-parts').textContent = 'Expand part settings';
  el('expand-parts').hidden = true; // The selected part still has its settings button.
  const views = node('div', 'project-tabs'); views.setAttribute('role','tablist'); views.setAttribute('aria-label','YTPMV views');
  const editorButton = node('button','open-editor','Editor'), library = node('button','open-midi-library','MIDI library');
  const editorView = node('section','view-editor'), libraryView = node('section','view-library');
  for(const [button,panel] of [[editorButton,editorView],[library,libraryView]]) {
    button.type='button'; button.setAttribute('role','tab');button.setAttribute('aria-controls',panel.id);
    panel.setAttribute('role','tabpanel');panel.setAttribute('aria-labelledby',button.id);
    views.append(button);
  }
  el('midi-import').before(views,libraryView,editorView);
  libraryView.append(el('midi-import'),el('library'));el('library').hidden=false;
  editorView.append(el('song'),sound,el('transport'),workspace);
  function setView(name) {
    editorView.hidden=name!=='editor';libraryView.hidden=name!=='library';
    editorButton.disabled=!editor.song();
    editorButton.setAttribute('aria-selected',String(name==='editor'));
    library.setAttribute('aria-selected',String(name==='library'));
    if(name==='library')instrument?.close();
  }
  const instrument = createYtpmvInstrument(editor);
  editorButton.addEventListener('click',()=>setView('editor'));
  library.addEventListener('click',()=>setView('library'));
  setView('library');

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
    timelineView.setSong(song);
  }
  function updateTrack(id) {
    const song = editor.song(); if (!song) return;
    const p = song.parts.find(p => p.id === id), s = editor.parts()[id], r = editor.row(id);
    const track = timelineView.track(id);
    if (!track || !p || !r) return;
    const synth = editor.onChip(p, s), source = r.querySelector('[data-el=source-note]').textContent;
    track.classList.toggle('muted', !!s.mute); track.classList.toggle('solo', !!s.solo);
    track.querySelector('[data-track-source]').textContent = synth ? editor.toneName(p,s) : editor.corpusName(s);
    track.querySelector('[data-track-caption]').textContent = synth ? editor.toneName(p,s) : (s.text || 'Choose a sample');
    track.querySelector('.track-tone').textContent = editor.toneName(p,s);
    track.querySelector('.track-tone').hidden = !synth;
    const sample=track.querySelector('.track-sample');sample.hidden=synth;
    sample.textContent=`${s.text||'Choose sound'} · take ${Number(s.take||0)+1}`;
    sample.title=`Edit sound: ${s.text||'no sample'} · take ${Number(s.take||0)+1}`;
    const select = track.querySelector('select'), original = r.querySelector('[data-k=corpus]');
    select.hidden = synth || editor.synth();
    // Preserve the inheritance choice, but show the performer rather than a
    // placeholder. Menu groups explain default versus explicit track casting.
    const options = original.innerHTML;
    if (select.innerHTML !== options) select.innerHTML = options;
    select.value = original.value; select.disabled = r.getAttribute('aria-busy') === 'true';
    select.title = s.corpus ? `Track voice: ${editor.corpusName(s)}` : `Follows default voice: ${editor.corpusName(s)}`;
    track.querySelectorAll('[data-track-flag]').forEach(b => { b.setAttribute('aria-pressed', String(!!s[b.dataset.trackFlag])); b.disabled = select.disabled; });
    track.querySelector('.track-name').title = `${p.instrument} · ${source}`;
    if(synth)track.querySelector('.track-tone').style.gridColumn='1 / -1';
  }
  function refresh() {
    const song = editor.song(); if (!song) return;
    if (song !== analysis) {
      analysis = song; selected = song.parts[0]?.id; drawTimeline(song);
      setView('editor');
    }
    const shown = song.parts.filter(p => editor.row(p.id)?.dataset.filtered !== 'true');
    if (!shown.some(p => p.id === selected)) selected = shown[0]?.id;
    for (const p of song.parts) {
      const r = editor.row(p.id); if (!r) continue;
      if (!r.hidden && p.id !== selected) r.querySelectorAll('video').forEach(video => video.pause());
      r.hidden = p.id !== selected;
    }
    timelineView.refresh(shown,selected);
    for (const id of timelineView.mounted()) updateTrack(id);
    empty.hidden = !!selected;
    timelineView.sync();
    el('render-card').hidden = false;
    if (tab === 'video') drawComposition();
    instrument.refresh();
  }
  let tileDrag=null, tileCells=[];
  function tileOrder(){
    const ids=editor.song().parts.map(p=>p.id);let saved=[];
    try{saved=JSON.parse(el('opt-tile-order').value);}catch{}
    return [...new Set([...(Array.isArray(saved)?saved:[]),...ids])].filter(id=>ids.includes(id));
  }
  function gridCells(n,w=1920,h=1080,layout=el('opt-layout').value){
    if(layout==='classic'){
      const count=Math.max(1,n);let cols=1,rows=count,cost=Infinity;
      for(let c=1;c<=count;c++){const r=Math.ceil(count/c),score=Math.abs(Math.log(c/r))+.3*(c*r-count);if(score<cost){cols=c;rows=r;cost=score;}}
      const scale=Math.min(w/(cols*16),h/(rows*9)),tw=Math.max(1,Math.floor(16*scale)),th=Math.max(1,Math.floor(9*scale)),center=el('opt-center-tiles').checked;
      return Array.from({length:count},(_,i)=>{const r=Math.floor(i/cols),columns=Math.min(cols,count-r*cols);return [(center?Math.floor((w-columns*tw)/2):0)+(i%cols)*tw,(center?Math.floor((h-rows*th)/2):0)+r*th,tw,th];});
    }
    if(layout==='featured'&&n>1){const split=Math.floor(w*2/3);return [[0,0,split,h],...gridCells(n-1,w-split,h,'sidebar').map(([x,y,a,b])=>[x+split,y,a,b])];}
    const count=Math.max(1,n),rows=layout==='two-rows'?Math.min(count,2):layout==='two-columns'?Math.ceil(count/2):layout==='strips'?1:layout==='sidebar'?Math.min(count,Math.max(1,Math.round(Math.sqrt(count*h/w)))):Math.ceil(count/Math.ceil(Math.sqrt(count)));
    const cells=[];for(let r=0;r<rows;r++){
      const columns=Math.floor(count/rows)+(r<count%rows?1:0),y=Math.floor(r*h/rows),bottom=Math.floor((r+1)*h/rows);
      for(let c=0;c<columns;c++){const x=Math.floor(c*w/columns);cells.push([x,y,Math.floor((c+1)*w/columns)-x,bottom-y]);}
    }return cells;
  }
  function drawComposition() {
    const song = editor.song(); if (!song) return;
    const fit=el('opt-media-fit').value,classic=el('opt-layout').value==='classic';
    composition.dataset.mediaFit=fit;
    el('center-tiles-control').hidden=!classic;
    el('composition-framing').textContent=`16:9 · ${classic?'equal tiles · ':''}${fit==='cover'?'cropped to fill':'whole clips'}`;
    let parts = song.parts.filter(p => editor.parts()[p.id].visible && (editor.onChip(p,editor.parts()[p.id]) || editor.parts()[p.id].text));
    if (!parts.length) parts = song.parts.filter(p => !editor.parts()[p.id].mute).slice(0,1);
    const order=tileDrag?.order||tileOrder();parts.sort((a,b)=>order.indexOf(a.id)-order.indexOf(b.id));
    tileCells=gridCells(parts.length);
    const present=new Set(parts.map(p=>p.id));
    for(const button of composition.children)if(!present.has(button.dataset.compositionPart))button.remove();
    parts.forEach((p,i)=>{
      let button=composition.querySelector(`[data-composition-part="${CSS.escape(p.id)}"]`);
      if(!button){button=node('button');button.type='button';button.className='composition-tile';button.dataset.compositionPart=p.id;composition.append(button);}
      const s=editor.parts()[p.id],[x,y,w,h]=tileCells[i];
      const url=editor.onChip(p,s)?null:editor.sampleInfo(p.id)?.url;
      const html=`${url?`<video src="${safe(url)}" muted playsinline preload="metadata" aria-hidden="true"></video>`:''}<b>${safe(editor.onChip(p,s)?editor.toneName(p,s):s.text)}</b><span>${safe(p.instrument)}</span><small>⠿ ${i+1}</small>`;
      if(button.dataset.content!==html){
        button.innerHTML=html;button.dataset.content=html;
        // Metadata-only preload can leave a paused video black. Seek just
        // inside the clip so a real frame shows its fit/crop in the preview.
        const video=button.querySelector('video');
        video?.addEventListener('loadedmetadata',()=>{if(Number.isFinite(video.duration)&&video.duration>0)video.currentTime=Math.min(.05,video.duration/2);},{once:true});
      }
      button.setAttribute('aria-label',`Tile ${i+1}: ${p.instrument}. Drag to rearrange.`);
      Object.assign(button.style,{left:x/1920*100+'%',top:y/1080*100+'%',width:w/1920*100+'%',height:h/1080*100+'%'});
      button.dataset.slot=i;
    });
  }
  timeline.addEventListener('click', event => {
    const pick = event.target.closest('[data-select]'); if (pick) select(pick.dataset.select);
    const sound=event.target.closest('[data-track-sound]');if(sound){select(sound.dataset.trackSound);instrument.open(sound.dataset.trackSound,sound);}
    const flag = event.target.closest('[data-track-flag]'); if (flag) {
      const id = flag.closest('.track').dataset.track; select(id);
      const control = editor.row(id).querySelector(`[data-k=${flag.dataset.trackFlag}]`);
      control.checked = !control.checked; control.dispatchEvent(new Event('input',{bubbles:true}));
    }
  });
  timeline.addEventListener('change', event => { if(event.target.matches('[data-track-corpus]')) editor.selectCorpus(event.target.dataset.trackCorpus,event.target.value); });
  composition.addEventListener('pointerdown',event=>{
    const button=event.target.closest('[data-composition-part]');if(!button||event.button!==0)return;
    event.preventDefault();button.focus({preventScroll:true});
    const rect=button.getBoundingClientRect();
    tileDrag={id:button.dataset.compositionPart,pointer:event.pointerId,x:event.clientX,y:event.clientY,left:rect.left,top:rect.top,order:tileOrder(),moved:false,button};
    button.setPointerCapture(event.pointerId);button.classList.add('dragging');
  });
  composition.addEventListener('pointermove',event=>{
    if(!tileDrag||event.pointerId!==tileDrag.pointer)return;
    const dx=event.clientX-tileDrag.x,dy=event.clientY-tileDrag.y;
    if(Math.hypot(dx,dy)<6&&!tileDrag.moved)return;tileDrag.moved=true;
    const rect=composition.getBoundingClientRect(),x=(event.clientX-rect.left)/rect.width*1920,y=(event.clientY-rect.top)/rect.height*1080;
    const index=tileCells.findIndex(([a,b,w,h])=>x>=a&&x<a+w&&y>=b&&y<b+h);
    const buttons=[...composition.children].sort((a,b)=>Number(a.dataset.slot)-Number(b.dataset.slot)),target=buttons[index]?.dataset.compositionPart;
    if(target&&target!==tileDrag.id){
      const from=tileDrag.order.indexOf(tileDrag.id),to=tileDrag.order.indexOf(target);
      tileDrag.order.splice(from,1);tileDrag.order.splice(to,0,tileDrag.id);drawComposition();
    }
    const left=parseFloat(tileDrag.button.style.left)/100*rect.width+rect.left,top=parseFloat(tileDrag.button.style.top)/100*rect.height+rect.top;
    tileDrag.button.style.transform=`translate(${dx+tileDrag.left-left}px,${dy+tileDrag.top-top}px) scale(1.025)`;
  });
  function finishTileDrag(event,cancel=false){
    if(!tileDrag||event.pointerId!==tileDrag.pointer)return;
    const current=tileDrag;tileDrag=null;current.button.classList.remove('dragging');current.button.style.transform='';
    if(current.button.hasPointerCapture(event.pointerId))current.button.releasePointerCapture(event.pointerId);
    if(current.moved&&!cancel){el('opt-tile-order').value=JSON.stringify(current.order);el('opt-tile-order').dispatchEvent(new Event('change',{bubbles:true}));compositionStatus.textContent='Tile order saved · used in the rendered video';}
    else if(!current.moved&&!cancel)select(current.id);
    drawComposition();
  }
  composition.addEventListener('pointerup',event=>finishTileDrag(event));
  composition.addEventListener('pointercancel',event=>finishTileDrag(event,true));
  composition.addEventListener('keydown',event=>{
    const button=event.target.closest('[data-composition-part]');if(!button)return;
    if(event.altKey&&['ArrowLeft','ArrowRight'].includes(event.code)){
      event.preventDefault();const order=tileOrder(),from=order.indexOf(button.dataset.compositionPart),to=Math.max(0,Math.min(order.length-1,from+(event.code==='ArrowLeft'?-1:1)));
      order.splice(from,1);order.splice(to,0,button.dataset.compositionPart);el('opt-tile-order').value=JSON.stringify(order);el('opt-tile-order').dispatchEvent(new Event('change',{bubbles:true}));
    }else if(event.code==='Enter')select(button.dataset.compositionPart);
  });
  el('opt-layout').addEventListener('change',drawComposition);
  el('opt-media-fit').addEventListener('change',drawComposition);
  el('opt-center-tiles').addEventListener('change',drawComposition);
  el('opt-speed').addEventListener('change',refresh);
  setTab('part');
  return {refresh, updateTrack, select, setTab, previewReady:timelineView.ready,
    instrumentPreparing:instrument.preparing,instrumentReady:instrument.ready,
    stopInstrument:instrument.stop,
    sampleChanged(id){updateTrack(id);instrument.refresh();if(tab==='video')drawComposition();},
    clearSong(){analysis=null;selected=null;setView('library');},
    get selected(){return selected;}, get tab(){return tab;}};
};
