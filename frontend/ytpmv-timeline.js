/* Scroll/zoom/seek use the song's real clock. No simulated transport. */
window.createYtpmvTimeline = function (host, editor) {
  const safe = text => String(text ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const byId = id => document.getElementById(id);
  const player = byId('player'), from = byId('preview-from'), seconds = byId('preview-seconds');
  const selection = byId('preview-selection');
  const toolbar = document.createElement('div'); toolbar.id = 'timeline-toolbar';
  toolbar.innerHTML = `<div class="timeline-actions"><button type="button" data-view="fit">Fit song</button><button type="button" data-view="out" aria-label="Zoom out">−</button><label>Zoom <input id="timeline-zoom" type="range" min="0" max="70" step="1" value="0"></label><button type="button" data-view="in" aria-label="Zoom in">+</button><output id="timeline-zoom-value">100%</output><button type="button" data-view="range" disabled>Focus range</button><button type="button" data-view="clear" hidden>Clear range</button><button type="button" data-view="part">Part start</button><label class="timeline-follow"><input id="timeline-follow" type="checkbox" checked> Follow playback</label></div><div class="timeline-readout"><span id="timeline-position" aria-live="off">0:00.00</span><span id="timeline-span"></span></div>`;
  const help = document.createElement('details'); help.id = 'timeline-help';
  help.innerHTML = '<summary>Shortcuts & navigation</summary><p>Space: play / pause · Wheel: tracks · Shift + wheel: time · Ctrl + wheel: zoom · Drag: seek · Shift + drag: playback range · Esc: clear range</p>';
  host.before(toolbar); host.after(help);
  host.setAttribute('role','region'); host.setAttribute('aria-label','Scrollable MIDI timeline');
  host.setAttribute('aria-describedby','timeline-help'); host.tabIndex = 0;
  host.innerHTML = '<div class="timeline-content"><div class="track-ruler"><span>Tracks</span><button type="button" class="time-ruler" aria-label="Seek or drag to set preview start"><span class="timeline-range"></span><span class="timeline-ticks"></span></button></div><div class="timeline-rows"></div><div class="timeline-cursor" aria-hidden="true"></div></div>';
  const content=host.firstElementChild, ruler=host.querySelector('.time-ruler'), rows=host.querySelector('.timeline-rows');
  const ticks=host.querySelector('.timeline-ticks'), zoomInput=byId('timeline-zoom');
  let song=null, parts=[], selected=null, trackMap=new Map(), zoom=1, timeWidth=1, headWidth=330;
  let position=0, offset=0, rangeStart=0, rangeEnd=10, drag=null, frame=0, scrollFrame=0;
  let windowKey='', noteRequest=0, noteAbort=null, noteTimer=0, seekRequest=0;
  const details=new Map();
  const duration=()=>Math.max(.001,(song?.duration||1)/editor.speed());
  const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
  const fmt=(value,precise=false)=>{
    const units=precise?100:1, ticks=precise?Math.round(Math.max(0,value)*units):Math.floor(Math.max(0,value));
    return `${Math.floor(ticks/(60*units))}:${((ticks%(60*units))/units).toFixed(precise?2:0).padStart(precise?5:2,'0')}`;
  };
  function rowHeight(){return parseFloat(getComputedStyle(host).getPropertyValue('--track-height'))||68;}
  function svg(p){
    const data=details.get(p.id)||p.activity||[];
    const pitches=p.range_midi||((p.activity||[]).map(n=>n[2]));
    const low=Math.min(48,...pitches),high=Math.max(72,...pitches),length=Math.max(.001,song.duration);
    return `<svg viewBox="0 0 ${length} 44" preserveAspectRatio="none" role="img" aria-label="${safe(p.instrument)} MIDI notes">${data.map(([start,len,pitch])=>`<rect x="${Math.max(0,start)}" y="${8+(high-pitch)/Math.max(1,high-low)*24}" width="${Math.max(.001,Math.min(length-start,len))}" height="3" rx=".1" vector-effect="non-scaling-stroke"/>`).join('')}</svg>`;
  }
  function trackHTML(p){return `<div class="track" data-track="${safe(p.id)}" style="--track-color:var(${p.is_drums?'--green':p.role==='bass'?'--amber':p.role==='chords'?'--violet':'--blue'})"><div class="track-head"><button type="button" data-select="${safe(p.id)}" class="track-name"><b>${safe(p.instrument)}</b><small data-track-source></small></button><div class="track-cast"><button type="button" data-track-sound="${safe(p.id)}" class="track-sample" title="Open this track's instrument"></button><select data-track-corpus="${safe(p.id)}" aria-label="Corpus for ${safe(p.instrument)} track"></select><button type="button" data-track-sound="${safe(p.id)}" class="track-tone" hidden></button></div><div class="track-switches"><button type="button" data-track-flag="mute" aria-label="Mute ${safe(p.instrument)}">M</button><button type="button" data-track-flag="solo" aria-label="Solo ${safe(p.instrument)}">S</button></div></div><button type="button" class="track-region" data-seek-part="${safe(p.id)}" aria-label="Select or seek in ${safe(p.instrument)}"><span data-track-caption></span>${svg(p)}</button></div>`;}
  function renderRows(force=false){
    const h=rowHeight(), virtual=parts.length>80;
    const first=virtual?Math.max(0,Math.floor(Math.max(0,host.scrollTop-31)/h)-4):0;
    const last=virtual?Math.min(parts.length,first+Math.ceil((host.clientHeight||480)/h)+9):parts.length;
    const key=parts.slice(first,last).map(p=>p.id).join('|');
    if(force||key!==windowKey){
      windowKey=key;
      rows.innerHTML=`<div class="timeline-spacer" style="height:${first*h}px"></div>`+parts.slice(first,last).map(trackHTML).join('')+`<div class="timeline-spacer" style="height:${Math.max(0,parts.length-last)*h}px"></div>`;
      trackMap=new Map([...rows.querySelectorAll('.track')].map(r=>[r.dataset.track,r]));
      for(const id of trackMap.keys())editor.mount(id);
      scheduleNotes();
    }
    for(const [id,track] of trackMap){
      track.classList.toggle('selected',id===selected);
      track.querySelectorAll('[data-select],[data-seek-part]').forEach(b=>b.setAttribute('aria-pressed',String(id===selected)));
    }
  }
  function layout(){
    if(!song)return;
    headWidth=parseFloat(getComputedStyle(host).getPropertyValue('--track-head-width'))||330;
    const base=Math.max(100,host.clientWidth-headWidth);
    timeWidth=Math.min(200000,base*zoom);
    content.style.width=`${headWidth+timeWidth}px`;
    host.style.setProperty('--timeline-width',timeWidth+'px');
    zoomInput.value=String(Math.log2(zoom)*10);
    byId('timeline-zoom-value').textContent=Math.round(zoom*100)+'%';
    renderRows();drawRuler();drawCursor();scheduleNotes();
  }
  function drawRuler(){
    if(!song)return;
    const visible=Math.max(1,host.clientWidth-headWidth), start=host.scrollLeft/timeWidth*duration(),end=(host.scrollLeft+visible)/timeWidth*duration();
    const target=80/timeWidth*duration();
    const step=[.1,.25,.5,1,2,5,10,15,30,60,120,300,600,1800,3600].find(s=>s>=target)||3600;
    const labels=[];
    for(let t=Math.ceil(start/step)*step;t<=Math.min(duration(),end+step);t+=step)
      labels.push(`<span style="left:${t/duration()*100}%">${fmt(t,step<1)}</span>`);
    ticks.innerHTML=labels.join('');
    const a=clamp(rangeStart,0,duration()),b=clamp(rangeEnd,a,duration());
    const band=ruler.querySelector('.timeline-range');band.hidden=!selection.checked;band.style.left=a/duration()*100+'%';band.style.width=(b-a)/duration()*100+'%';
    byId('playback-scope').textContent=selection.checked?`Selection · ${fmt(a,true)}–${fmt(b,true)}`:'Whole song';
    toolbar.querySelector('[data-view=clear]').hidden=!selection.checked;
    toolbar.querySelector('[data-view=range]').disabled=!selection.checked;
    byId('timeline-span').textContent=`${parts.length} tracks · ${fmt(start)}–${fmt(Math.min(end,duration()))} of ${fmt(duration())}`;
  }
  function drawCursor(){
    content.style.setProperty('--cursor-x',headWidth+clamp(position,0,duration())/duration()*timeWidth+'px');
    byId('timeline-position').textContent=fmt(position,true);
    byId('transport-clock').textContent=fmt(position,true);
  }
  function setZoom(value,anchorX=host.clientWidth/2){
    const anchor=clamp((host.scrollLeft+anchorX-headWidth)/timeWidth,0,1);
    zoom=clamp(value,1,128);
    if(zoom===1){details.clear();windowKey='';}
    layout();
    host.scrollLeft=Math.max(0,anchor*timeWidth-(anchorX-headWidth));drawRuler();scheduleNotes();
  }
  function revealCursor(){
    const x=position/duration()*timeWidth, width=Math.max(1,host.clientWidth-headWidth);
    if(x<host.scrollLeft||x>host.scrollLeft+width-30)host.scrollLeft=Math.max(0,x-width*.25);
  }
  function syncAudio(){
    if(!song||drag)return;
    if(player.getAttribute('src'))position=clamp(offset+(player.currentTime||0),0,duration());
    drawCursor();
    if(!player.paused&&byId('timeline-follow').checked)revealCursor();
  }
  function animate(){syncAudio();if(!player.paused&&!player.ended)frame=requestAnimationFrame(animate);else frame=0;}
  player.addEventListener('play',()=>{cancelAnimationFrame(frame);animate();});
  for(const name of ['timeupdate','seeked','pause','ended'])player.addEventListener(name,syncAudio);
  player.addEventListener('emptied',()=>{cancelAnimationFrame(frame);frame=0;position=Number(from.value)||0;drawCursor();});
  async function seek(time,continuePlaying=false,keepRange=false){
    const request=++seekRequest;
    const clearedRange=selection.checked&&!keepRange;
    if(!keepRange)selection.checked=false;
    position=clamp(time,0,duration());from.value=position.toFixed(2);drawCursor();
    if(keepRange){rangeStart=position;rangeEnd=position+Number(seconds.value);}drawRuler();
    const local=position-offset;
    if(!clearedRange&&player.getAttribute('src')&&Number.isFinite(player.duration)&&local>=0&&local<=player.duration){
      player.currentTime=local;
      if(continuePlaying){try{await player.play();}catch(e){if(e.name!=='NotAllowedError'&&e.name!=='AbortError')throw e;}}
    }else{
      const stopped=await editor.stop();
      if(request!==seekRequest)return;
      position=Number(from.value)||0;drawCursor();
      if(continuePlaying&&stopped)editor.preview();
    }
  }
  async function clearRange(){
    selection.checked=false;from.value=0;position=0;
    drawCursor();drawRuler();await editor.stop();
  }
  function timeAt(clientX){return clamp((clientX-ruler.getBoundingClientRect().left)/timeWidth*duration(),0,duration());}
  host.addEventListener('pointerdown',event=>{
    const target=event.target.closest('.time-ruler,.track-region');
    if(!target||event.button!==0)return;
    event.preventDefault();
    if(target.dataset.seekPart)editor.select(target.dataset.seekPart);
    drag={id:event.pointerId,start:timeAt(event.clientX),range:event.shiftKey,playing:!player.paused,x:event.clientX};
    if(drag.range){selection.checked=true;rangeStart=drag.start;rangeEnd=drag.start;drawRuler();}
    position=drag.start;host.setPointerCapture(event.pointerId);drawCursor();
  });
  host.addEventListener('pointermove',event=>{
    if(!drag||event.pointerId!==drag.id)return;
    const rect=host.getBoundingClientRect();
    if(event.clientX>rect.right-24)host.scrollLeft+=18;
    if(event.clientX<rect.left+headWidth+18)host.scrollLeft-=18;
    position=timeAt(event.clientX);
    if(drag.range){rangeStart=Math.min(drag.start,position);rangeEnd=Math.max(drag.start,position);drawRuler();}
    drawCursor();
  });
  host.addEventListener('pointerup',async event=>{
    if(!drag||event.pointerId!==drag.id)return;
    const current=drag;drag=null;host.releasePointerCapture(event.pointerId);
    if(current.range){
      const length=Math.max(1,Math.round((rangeEnd-rangeStart)*10)/10);
      if(![...seconds.options].some(o=>Number(o.value)===length))seconds.add(new Option(`${length} seconds (selection)`,String(length)));
      seconds.value=String(length);await editor.stop();await seek(rangeStart,false,true);
    }else await seek(timeAt(event.clientX),current.playing);
  });
  host.addEventListener('pointercancel',()=>{drag=null;position=Number(from.value)||0;drawCursor();});
  host.addEventListener('click',event=>{
    const region=event.target.closest('[data-seek-part]');
    if(region&&event.detail===0){const p=parts.find(p=>p.id===region.dataset.seekPart);editor.select(p.id);seek((p.first??p.activity?.[0]?.[0]??0)/editor.speed());}
    if(event.target.closest('.time-ruler')&&event.detail===0)seek(Number(from.value)||0);
  });
  host.addEventListener('wheel',event=>{
    if(event.ctrlKey||event.metaKey){event.preventDefault();setZoom(zoom*Math.exp(-event.deltaY*.005),event.clientX-host.getBoundingClientRect().left);}
    else if(event.shiftKey&&!event.deltaX){event.preventDefault();host.scrollLeft+=event.deltaY;}
  },{passive:false});
  host.addEventListener('scroll',()=>{
    if(!scrollFrame)scrollFrame=requestAnimationFrame(()=>{scrollFrame=0;renderRows();drawRuler();scheduleNotes();});
  },{passive:true});
  host.addEventListener('keydown',event=>{
    if(event.target!==host)return;
    if(event.code==='Space'){event.preventDefault();if(!event.repeat)editor.preview();}
    else if(event.code==='Escape'){event.preventDefault();clearRange();}
    else if(['ArrowLeft','ArrowRight','Home','End'].includes(event.code)){
      event.preventDefault();seek(event.code==='Home'?0:event.code==='End'?duration():position+(event.code==='ArrowRight'?1:-1)*(event.shiftKey?10:1));
    }
  });
  toolbar.addEventListener('click',event=>{
    const key=event.target.closest('[data-view]')?.dataset.view;if(!key||!song)return;
    if(key==='in')setZoom(zoom*1.5);
    if(key==='out')setZoom(zoom/1.5);
    if(key==='fit'){zoom=1;host.scrollLeft=0;details.clear();windowKey='';layout();}
    if(key==='range'){
      const start=Number(from.value)||0,span=Math.max(.1,Math.min(Number(seconds.value),duration()-start));
      zoom=clamp(duration()/span*.9,1,128);layout();host.scrollLeft=Math.max(0,start/duration()*timeWidth-20);drawRuler();scheduleNotes();
    }
    if(key==='clear')clearRange();
    if(key==='part'){
      const index=parts.findIndex(p=>p.id===selected),p=parts[index];if(!p)return;
      host.scrollTop=Math.max(0,index*rowHeight()-rowHeight());renderRows();
      const start=p.first??p.activity?.[0]?.[0]??0;seek(start/editor.speed());revealCursor();
    }
  });
  zoomInput.addEventListener('input',()=>setZoom(2**(Number(zoomInput.value)/10)));
  byId('transport-start').addEventListener('click',()=>seek(0,!player.paused));
  byId('transport-back').addEventListener('click',()=>seek(position-10,!player.paused));
  byId('transport-forward').addEventListener('click',()=>seek(position+10,!player.paused));
  function scheduleNotes(){
    clearTimeout(noteTimer);noteAbort?.abort();++noteRequest;
    if(!song||zoom<=1||!editor.notes||!trackMap.size)return;
    noteTimer=setTimeout(loadNotes,140);
  }
  async function loadNotes(){
    const request=++noteRequest, current=song;
    noteAbort=new AbortController();
    const start=host.scrollLeft/timeWidth*song.duration;
    const end=Math.min(song.duration,(host.scrollLeft+host.clientWidth-headWidth)/timeWidth*song.duration);
    const ids=[...trackMap.keys()];
    try{
      const data=await editor.notes(song.midi_id,{start,end,parts:ids.join(',')},noteAbort.signal);
      if(request!==noteRequest||song!==current)return;
      for(const [id,notes] of Object.entries(data.parts||{})){
        details.set(id,notes);const track=trackMap.get(id),p=parts.find(p=>p.id===id);
        if(track&&p)track.querySelector('svg').outerHTML=svg(p);
      }
      while(details.size>96)details.delete(details.keys().next().value);
      if(data.sampled)byId('timeline-span').textContent+=' · zoom further for every note';
    }catch(error){if(error.name!=='AbortError'&&request===noteRequest)byId('timeline-span').textContent+=' · note overview (detail unavailable)';}
  }
  new ResizeObserver(()=>{windowKey='';layout();}).observe(host);
  from.addEventListener('input',()=>{selection.checked=true;position=Number(from.value)||0;rangeStart=position;rangeEnd=position+Number(seconds.value);drawCursor();drawRuler();});
  seconds.addEventListener('change',()=>{selection.checked=true;rangeStart=Number(from.value)||0;rangeEnd=rangeStart+Number(seconds.value);drawRuler();});
  selection.addEventListener('change',()=>{rangeStart=Number(from.value)||0;rangeEnd=rangeStart+Number(seconds.value);drawRuler();});
  return {
    setSong(value){song=value;selection.checked=false;parts=[];selected=null;zoom=1;host.scrollTop=host.scrollLeft=0;details.clear();windowKey='';position=Number(from.value)||0;rangeStart=position;rangeEnd=position+Number(seconds.value);layout();},
    refresh(list,id){
      const changed=parts.length!==list.length||parts.some((p,i)=>p!==list[i]);
      const picked=selected!==id;
      if(changed){parts=list;windowKey='';}
      selected=id;host.classList.toggle('many-tracks',parts.length>6);layout();
      const index=parts.findIndex(p=>p.id===id),top=31+index*rowHeight();
      if(index>=0&&(changed||picked)&&(top<host.scrollTop+31||top+rowHeight()>host.scrollTop+host.clientHeight)){
        host.scrollTop=Math.max(0,index*rowHeight()-rowHeight());renderRows();
      }
    },
    track(id){return trackMap.get(id);},
    mounted(){return [...trackMap.keys()];},
    ready(result){offset=Number(result.from)||0;position=offset;drawCursor();},
    sync(){rangeStart=Number(from.value)||0;rangeEnd=rangeStart+Number(seconds.value);if(player.getAttribute('src'))syncAudio();else position=rangeStart;layout();}
  };
};
