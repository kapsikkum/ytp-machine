/* A small instrument window: the track's original controls remain authoritative. */
window.createYtpmvInstrument = function(editor) {
  const panel=document.createElement('section');panel.id='track-instrument';panel.setAttribute('popover','auto');
  panel.setAttribute('aria-label','Track instrument');
  panel.innerHTML=`<header><div><small>TRACK INSTRUMENT</small><h3></h3></div><button type="button" data-close aria-label="Close instrument">×</button></header>
    <div class="instrument-fields"><label>Instrument<select data-proxy="instrument" aria-label="Quick instrument"></select></label><label data-voice-field>Voice<select data-proxy="corpus" aria-label="Quick voice"></select></label><label data-patch-field>Patch<select data-proxy="program" aria-label="Quick patch"></select></label></div>
    <div data-sample-fields><label class="instrument-sample">Sample<input data-proxy="text" type="text" aria-label="Quick sample"></label><div class="instrument-takes"><button type="button" data-action="prev" aria-label="Previous instrument take">◀</button><span data-take></span><button type="button" data-action="next" aria-label="Next instrument take">▶</button></div><div class="instrument-suggestions"><small>Suggested sounds</small><div data-suggestions></div></div></div>
    <canvas width="900" height="120" aria-label="Audition audio waveform"></canvas><p data-wave-status>Play a key to hear this instrument.</p>
    <div class="instrument-keyboard" aria-label="Audition keyboard"></div><div class="instrument-actions"><button type="button" data-pattern>▶ Play track pattern</button><button type="button" data-stop>■ Stop audition</button><span>Keys audition this instrument alone.</span></div>`;
  document.body.append(panel);
  const patternStatus=document.createElement('div');patternStatus.className='hint';patternStatus.setAttribute('role','status');panel.append(patternStatus);
  const player=document.getElementById('instrument-player'),canvas=panel.querySelector('canvas'),ctx=canvas.getContext('2d');
  let id=null,peaks=[],frame=0,sequence=0,audioContext=null,activeNote=60;
  const banks=new Map(),notes=Array.from({length:13},(_,i)=>60+i);
  let pending=null,bank=null,cacheTimer=0,keyIntent=0,waveClock=null;
  const voices=new Set();
  const context=()=>audioContext ||= new (window.AudioContext||window.webkitAudioContext)({latencyHint:'interactive'});
  const names=['C','C♯','D','D♯','E','F','F♯','G','G♯','A','A♯','B','C'];
  panel.querySelector('.instrument-keyboard').innerHTML=names.map((name,i)=>`<button type="button" data-note="${60+i}" class="${name.includes('♯')?'black':'white'}" ${name.includes('♯')?`style="left:${({1:12.5,3:25,6:50,8:62.5,10:75})[i]-3.5}%"`:''} aria-label="Audition ${name}${i===12?'5':'4'}">${name}${i===0?'4':i===12?'5':''}</button>`).join('');
  const row=()=>id?editor.row(id):null;
  function refresh(){
    const r=row(),p=editor.song()?.parts.find(p=>p.id===id);if(!r||!p)return;
    const s=editor.parts()[id],synth=editor.onChip(p,s);
    panel.querySelector('h3').textContent=`${p.instrument} · ${synth?editor.toneName(p,s):editor.corpusName(s)}`;
    for(const key of ['instrument','corpus','program']){
      const original=r.querySelector(`[data-k=${key}]`),control=panel.querySelector(`[data-proxy=${key}]`);
      if(!original)continue;
      if(control.innerHTML!==original.innerHTML)control.innerHTML=original.innerHTML;
      control.value=original.value;control.disabled=original.disabled;
    }
    panel.querySelector('[data-voice-field]').hidden=synth||editor.synth();
    panel.querySelector('[data-patch-field]').hidden=r.querySelector('[data-el=patch-wrap]')?.hidden??true;
    panel.querySelector('[data-sample-fields]').hidden=synth||editor.synth();
    const input=panel.querySelector('[data-proxy=text]');
    if(document.activeElement!==input)input.value=s.text||'';
    input.setAttribute('list',r.querySelector('[data-k=text]').getAttribute('list'));
    panel.querySelector('[data-take]').textContent=r.querySelector('[data-el=take]').textContent;
    const suggestions=panel.querySelector('[data-suggestions]');suggestions.replaceChildren();
    for(const original of r.querySelectorAll('[data-alt]')){
      const button=document.createElement('button');button.type='button';button.textContent=original.textContent;
      button.addEventListener('click',()=>{original.click();refresh();playNote(activeNote);});suggestions.append(button);
    }
    panel.querySelector('.instrument-suggestions').hidden=!suggestions.childElementCount;
    panel.querySelectorAll('[data-note],[data-pattern]').forEach(b=>b.disabled=r.getAttribute('aria-busy')==='true');
    if(panel.matches(':popover-open'))scheduleBank();
  }
  function cancelPreparation(){clearTimeout(cacheTimer);pending?.abort.abort();pending=null;}
  function stopNotes(){
    ++keyIntent;for(const voice of voices){try{voice.stop();}catch{}}voices.clear();
    waveClock=null;cancelAnimationFrame(frame);frame=0;
    panel.querySelectorAll('[data-note]').forEach(b=>b.classList.remove('playing'));draw();
  }
  function waveform(buffer,offset,duration){
    const data=buffer.getChannelData(0),start=Math.round(offset*buffer.sampleRate),end=Math.min(data.length,Math.round((offset+duration)*buffer.sampleRate));
    const hop=Math.max(1,Math.ceil((end-start)/canvas.width));let maximum=0;
    const values=Array.from({length:canvas.width},(_,x)=>{let peak=0;for(let i=start+x*hop;i<Math.min(end,start+(x+1)*hop);i++)peak=Math.max(peak,Math.abs(data[i]));maximum=Math.max(maximum,peak);return peak;});
    return values.map(value=>value/Math.max(.001,maximum));
  }
  function showBank(entry){
    bank=entry;panel.dataset.keyboardState='ready';panel.querySelector('.instrument-keyboard').setAttribute('aria-busy','false');
    panel.querySelector('[data-wave-status]').textContent=`Keyboard ready · ${entry.segments.size} notes cached in this tab`;
    if(!voices.size){peaks=entry.waves.get(activeNote)||[];draw();}
  }
  async function ensureBank(){
    const state=editor.keyboardState(id);
    if(!state||row()?.getAttribute('aria-busy')==='true')return null;
    if(banks.has(state.key)){
      const entry=banks.get(state.key);banks.delete(state.key);banks.set(state.key,entry);showBank(entry);return entry;
    }
    if(pending?.key===state.key)return pending.promise;
    cancelPreparation();bank=null;panel.dataset.keyboardState='loading';
    panel.querySelector('.instrument-keyboard').setAttribute('aria-busy','true');
    panel.querySelector('[data-wave-status]').textContent=`Preparing keyboard · ${notes.length} notes…`;
    const request={key:state.key,abort:new AbortController(),promise:null};pending=request;
    request.promise=(async()=>{
      const result=await editor.prepareKeyboard(state,notes,request.abort.signal);
      if(request.abort.signal.aborted)throw new DOMException('Keyboard changed','AbortError');
      const response=await fetch(result.audio_url,{signal:request.abort.signal});
      if(!response.ok)throw new Error('Could not load keyboard audio.');
      const buffer=await context().decodeAudioData(await response.arrayBuffer());
      if(request.abort.signal.aborted)throw new DOMException('Keyboard changed','AbortError');
      const segments=new Map((result.keyboard||[]).map(segment=>[segment.note,segment]));
      if(notes.some(note=>!segments.has(note))||[...segments.values()].some(s=>s.offset<0||s.duration<=0||s.offset+s.duration>buffer.duration+.05))throw new Error('The keyboard audio bank is incomplete.');
      const entry={key:state.key,buffer,segments,waves:new Map(),bytes:buffer.length*buffer.numberOfChannels*4};
      for(const [note,segment] of segments)entry.waves.set(note,waveform(buffer,segment.offset,segment.duration));
      banks.set(state.key,entry);
      let bytes=[...banks.values()].reduce((sum,b)=>sum+b.bytes,0);
      while((bytes>48*1024*1024||banks.size>6)&&banks.size>1){const key=banks.keys().next().value;bytes-=banks.get(key).bytes;banks.delete(key);}
      if(editor.keyboardState(id)?.key===state.key)showBank(entry);
      return entry;
    })().catch(error=>{
      if(error.name!=='AbortError'&&pending===request){panel.dataset.keyboardState='error';panel.querySelector('[data-wave-status]').textContent=`Keyboard unavailable: ${error.message} Press a key to retry.`;panel.querySelector('.instrument-keyboard').setAttribute('aria-busy','false');}
      throw error;
    }).finally(()=>{if(pending===request)pending=null;});
    return request.promise;
  }
  function scheduleBank(){
    const state=editor.keyboardState(id);if(!state)return;
    if(bank?.key===state.key||pending?.key===state.key)return;
    bank=null;cancelPreparation();stopNotes();panel.dataset.keyboardState='loading';
    cacheTimer=setTimeout(()=>ensureBank().catch(()=>{}),250);
  }
  async function playNote(note){
    activeNote=note;const intent=++keyIntent,state=editor.keyboardState(id);if(!state)return;
    const resumed=context().resume();
    try{
      clearTimeout(cacheTimer);const entry=await ensureBank();await resumed;
      if(!entry||intent!==keyIntent||editor.keyboardState(id)?.key!==state.key||!panel.matches(':popover-open'))return;
      // Stop the arrangement once, then subsequent cached notes can overlap.
      if(!voices.size){await editor.stopForKeyboard();}
      if(intent!==keyIntent||editor.keyboardState(id)?.key!==state.key)return;
      if(voices.size>=8){const oldest=voices.values().next().value;oldest.stop();voices.delete(oldest);}
      const segment=entry.segments.get(note),source=context().createBufferSource();source.buffer=entry.buffer;source.connect(context().destination);
      source.note=note;voices.add(source);const button=panel.querySelector(`[data-note="${note}"]`);button.classList.add('playing');
      source.onended=()=>{voices.delete(source);source.disconnect();if(![...voices].some(v=>v.note===note))button.classList.remove('playing');if(!voices.size){waveClock=null;cancelAnimationFrame(frame);frame=0;draw();}};
      source.start(0,segment.offset,segment.duration);peaks=entry.waves.get(note);waveClock={start:context().currentTime,duration:segment.duration};
      panel.querySelector('[data-wave-status]').textContent=`${names[note-60]} · cached instrument · ${entry.segments.size} keys ready`;
      cancelAnimationFrame(frame);animate();
    }catch(error){if(error.name!=='AbortError')panel.querySelector('[data-wave-status]').textContent=error.message;}
  }
  function open(trackId,anchor){
    const wasOpen=panel.matches(':popover-open');
    if(id!==trackId){id=trackId;peaks=[];++sequence;patternStatus.textContent='';panel.querySelector('[data-wave-status]').textContent='Play a key to hear this instrument.';draw();}
    refresh();
    if(wasOpen&&!anchor)return;
    if(!panel.matches(':popover-open'))panel.showPopover();
    clearTimeout(cacheTimer);ensureBank().catch(()=>{});
    const rect=anchor?.getBoundingClientRect();
    panel.style.left=`${Math.max(8,Math.min(innerWidth-panel.offsetWidth-8,rect?.right||innerWidth/2-panel.offsetWidth/2))}px`;
    panel.style.top=`${Math.max(8,Math.min(innerHeight-panel.offsetHeight-8,rect?.top||100))}px`;
  }
  panel.addEventListener('input',e=>{
    const key=e.target.dataset.proxy;if(!key||!row())return;
    const original=row().querySelector(`[data-k=${key}]`);original.value=e.target.value;
    original.dispatchEvent(new Event('input',{bubbles:true}));
    refresh();
  });
  panel.addEventListener('click',e=>{
    if(e.target.closest('[data-close]'))panel.hidePopover();
    const action=e.target.closest('[data-action]')?.dataset.action;
    if(action){row()?.querySelector(`[data-act=${action}]`).click();refresh();playNote(activeNote);}
    const note=e.target.closest('[data-note]')?.dataset.note;
    if(note)playNote(Number(note));
    if(e.target.closest('[data-pattern]')){cancelPreparation();editor.audition(id);}
    if(e.target.closest('[data-stop]'))editor.stop();
  });
  function draw(){
    ctx.clearRect(0,0,canvas.width,canvas.height);ctx.strokeStyle='#3ddc97';ctx.lineWidth=1;
    ctx.beginPath();for(let x=0;x<canvas.width;x++){
      const peak=peaks[x]||0;ctx.moveTo(x,60-peak*54);ctx.lineTo(x,60+peak*54);
    }ctx.stroke();
    if(player.getAttribute('src')&&Number.isFinite(player.duration)){
      ctx.fillStyle='#ffe500';ctx.fillRect(player.currentTime/player.duration*canvas.width,0,2,canvas.height);
    }
    if(waveClock){ctx.fillStyle='#ffe500';ctx.fillRect(Math.min(1,(context().currentTime-waveClock.start)/waveClock.duration)*canvas.width,0,2,canvas.height);}
  }
  function animate(){draw();if(voices.size||(!player.paused&&!player.ended))frame=requestAnimationFrame(animate);}
  player.addEventListener('play',()=>{cancelAnimationFrame(frame);animate();});
  for(const event of ['pause','ended','emptied'])player.addEventListener(event,()=>{if(voices.size)return;cancelAnimationFrame(frame);draw();panel.querySelectorAll('[data-note]').forEach(b=>b.classList.remove('playing'));});
  panel.addEventListener('toggle',event=>{if(event.newState==='closed'){++sequence;stopNotes();cancelPreparation();}});
  return {
    open,refresh,
    stop:stopNotes,
    close(){if(panel.matches(':popover-open'))panel.hidePopover();},
    preparing(trackId){
      // Only an explicit sound-control click opens this window. The dock's
      // part button auditions in place, with status beside the transport.
      if(trackId!==id||!panel.matches(':popover-open'))return;
      cancelPreparation();++sequence;peaks=[];draw();panel.querySelector('[data-wave-status]').textContent='Preparing track pattern…';
      return patternStatus;
    },
    async ready(trackId,result){
      if(trackId!==id||!panel.matches(':popover-open'))return;
      const request=++sequence;panel.querySelector('[data-wave-status]').textContent=`${result.duration.toFixed(2)}s · isolated instrument audition`;
      try{
        audioContext ||= new (window.AudioContext||window.webkitAudioContext)();
        const buffer=await (await fetch(result.audio_url)).arrayBuffer(),audio=await audioContext.decodeAudioData(buffer);
        if(request!==sequence)return;
        const samples=audio.getChannelData(0),hop=Math.max(1,Math.ceil(samples.length/canvas.width));
        let maximum=0;peaks=Array.from({length:canvas.width},(_,x)=>{
          let value=0;for(let i=x*hop;i<Math.min(samples.length,(x+1)*hop);i++)value=Math.max(value,Math.abs(samples[i]));
          maximum=Math.max(maximum,value);return value;
        });peaks=peaks.map(v=>v/Math.max(.001,maximum));draw();
      }catch(e){if(request===sequence)panel.querySelector('[data-wave-status]').textContent+=' · waveform unavailable';}
    }
  };
};
