// Model-free regressions against the actual page functions. Run: node tests/test_frontend.js
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const root = path.resolve(__dirname, '..');
const read = file => fs.readFileSync(path.join(root, 'frontend', file), 'utf8');
const section = (source, begin, end) => {
  const a = source.indexOf(begin), b = source.indexOf(end, a + begin.length);
  assert(a >= 0 && b > a, `missing source section: ${begin} -> ${end}`);
  return source.slice(a, b);
};
const deferred = () => { let resolve, reject; const promise = new Promise((a,b) => {resolve=a;reject=b;}); return {promise,resolve,reject}; };
function sandbox(extra = {}) {
  const elements = {};
  const element = id => elements[id] ||= {value:'', style:{}, dataset:{}, children:[],
    textContent:'', innerHTML:'', classList:{add(){},remove(){},toggle(){}}, querySelector(){return element('child');},
    addEventListener(type,handler){this.listeners ||= {};this.listeners[type]=handler;}};
  const context = vm.createContext({console, clearTimeout, setTimeout, URLSearchParams,
    document:{getElementById:element}, localStorage:{getItem(){return null;},setItem(){}},
    ...extra});
  vm.runInContext(read('common.js'), context);
  return {context, element, run: code => vm.runInContext(code, context)};
}

async function sourceRace() {
  const requests = [deferred(), deferred()]; let call = 0;
  const b = sandbox({corpusSlug:'test', sourceId:null, sel:null, env:null, clips:[], newSpan:null,
    alignTimer:null, envSeq:0, envLoading:false, vid:{}, api:()=>requests[call++].promise,
    cancelPlayback(){}, renderWords(){}, drawWave(){}, refreshAlign(){}, loadEnvelope(){}, say(){}});
  b.run('const sourceRequests = latestRequest();');
  b.run(section(read('editor.html'), 'async function selectSource', '// ── measuring'));
  const first = b.context.selectSource(1), second = b.context.selectSource(2);
  requests[1].resolve({clips:[{id:2,word:'second'}]}); await second;
  requests[0].resolve({clips:[{id:1,word:'first'}]}); await first;
  assert.equal(b.context.sourceId, 2);
  assert.equal(b.context.clips[0].word, 'second');
  assert.match(b.context.vid.src, /source\/2\/video/);
}

async function editorCorpusSwitch() {
  let loaded=null;
  const b=sandbox({corpusSlug:'old',sel:{},sourceId:1,env:{},clips:[],
    mayLeave:()=>true,cancelPlayback(){},vid:{removeAttribute(){},load(){}},say(){},
    fetch:async()=>({ok:true,json:async()=>({})}),loadSources:async()=>{loaded=b.context.corpusSlug;}});
  b.element('corpus').value='new';
  b.run(section(read('editor.html'),"$('corpus').onchange =",'// ── sources'));
  await b.element('corpus').onchange();
  assert.equal(loaded,'new');assert.equal(b.context.corpusSlug,'new');
  assert.equal(b.element('corpus').disabled,false);
}

async function saveAndDeleteSelection() {
  const editor = read('editor.html');
  const saved = deferred(), deleted = deferred(); let call = 0;
  const first = {id:1,kind:'word',word:'first',start_time:1}, second = {id:2,kind:'word',word:'second',start_time:2};
  let selected = null;
  const b = sandbox({sel:first,sourceId:1,corpusSlug:'test',clips:[first,second],
    api:()=>[saved,deleted][call++].promise, renderWords(){},drawWave(){},say(){},markDirty(){},
    select:clip=>{selected=clip;},confirm:()=>true});
  b.element('w-word').value = 'fixed'; b.element('w-start').value='1';b.element('w-end').value='2';
  b.run(section(editor, "$('save').onclick =", "$('del').onclick ="));
  b.run(section(editor, "$('del').onclick =", 'async function addClip'));
  const saving = b.element('save').onclick(); b.context.sel = second;
  saved.resolve({word:'fixed',start_time:1,end_time:2}); await saving;
  assert.equal(first.word,'fixed'); assert.equal(second.word,'second'); assert.equal(selected,null);
  const further=deferred();b.context.api=()=>further.promise;b.context.sel=first;
  const savingAgain=b.element('save').onclick();b.element('w-word').value='new unsaved draft';
  further.resolve({word:'fixed',start_time:1,end_time:2});await savingAgain;
  assert.equal(b.element('w-word').value,'new unsaved draft');assert.equal(selected,null);
  b.context.api=()=>deleted.promise;
  b.context.sel=first;
  const deleting = b.element('del').onclick(); b.context.sel=second;
  deleted.resolve({deleted:true}); await deleting;
  assert.equal(b.context.clips.length,1); assert.equal(b.context.clips[0],second); assert.equal(b.context.sel,second);
}

async function spliceCandidateRace() {
  const request = deferred(); let rendered=0;
  const b = sandbox({wordContext:1,corpusSlug:'test',sources:{},api:()=>request.promise,
    renderGroup(){rendered++;}});
  b.run(section(read('splice.html'),'async function fetchSources','function renderGroup'));
  const card={isConnected:true};
  const pending=b.context.fetchSources('0:1',{phones:['AA']},card);
  b.context.wordContext=2;
  request.resolve({sources:[{word:'old word'}]}); await pending;
  assert.equal(rendered,0); assert.equal(Object.keys(b.context.sources).length,0);
}

async function sampleRace() {
  const pending=[deferred(),deferred()];let call=0, badges=0;
  const settings={text:'old',take:0,hit:true};
  const b=sandbox({parts:{p:settings},info:{},sampleRequests:{},liveVoice:'test',corpusOf:s=>s.corpus || 'test',
    song:{parts:[{id:'p',is_drums:false}]},row(){return null;},post:()=>pending[call++].promise,
    drawBadge(){badges++;}});
  b.run(section(read('ytpmv.html'),'async function refreshSample','function changed'));
  const first=b.context.refreshSample('p');settings.text='new';const second=b.context.refreshSample('p');
  pending[1].resolve({take:0,takes:2,sample_url:'/new',hz:440});await second;
  pending[0].resolve({take:1,takes:7,sample_url:'/old',hz:220});await first;
  assert.equal(b.context.info.p.url,'/new');assert.equal(settings.take,0);assert.equal(badges,1);
  const replaced=deferred();b.context.post=()=>replaced.promise;
  const oldSong=b.context.refreshSample('p');b.context.song={parts:[{id:'p'}]};b.context.parts={p:{text:'third',take:0}};
  b.context.info={};replaced.resolve({take:0,takes:1,sample_url:'/wrong-song'});await oldSong;
  assert.equal(Object.keys(b.context.info).length,0);
  const oldCorpusResponse=deferred();let submitted;
  b.context.post=(_url,body,corpus)=>{submitted={body,corpus};return oldCorpusResponse.promise;};
  b.context.parts.p.corpus='first-voice';
  const oldCorpus=b.context.refreshSample('p');
  assert.equal(submitted.body.corpus,'first-voice');assert.equal(submitted.corpus,'first-voice');
  b.context.parts.p.corpus='second-voice';
  oldCorpusResponse.resolve({take:0,takes:1,sample_url:'/wrong-corpus'});await oldCorpus;
  assert.equal(Object.keys(b.context.info).length,0);
}

async function cancellationFailure() {
  const messages=[];
  const b=sandbox({activeJobId:'job',fetchJson:async()=>({ok:false,status:500,data:{detail:'cancel failed'}}),
    setStatus:(...args)=>messages.push(args)});
  b.run(section(read('index.html'), "$('cancel-job-btn').onclick =", 'async function poll'));
  await b.element('cancel-job-btn').onclick();
  assert.equal(b.element('cancel-job-btn').disabled,false);
  assert.equal(messages[0][0],'cancel failed');
  assert.equal(messages[0][1],true);
}

async function settingsFailure() {
  const messages=[], modal={style:{display:'flex'}};
  const b=sandbox({sModal:modal,postJson:async()=>({ok:false,status:400,data:{detail:'invalid encoder'}}),
    setStatus:(...args)=>messages.push(args)});
  b.run(section(read('index.html'), "$('btn-save-server-settings').addEventListener", 'async function loadSpliceMode'));
  await b.element('btn-save-server-settings').listeners.click();
  assert.equal(modal.style.display,'flex');assert.equal(b.element('btn-save-server-settings').disabled,false);
  assert.equal(messages[0][0],'invalid encoder');
}

async function suggestionBlurRace() {
  const response=deferred();let drawn=0;
  const input={};
  const b=sandbox({fetchSeq:0,suggestTimer:null,items:[],active:-1,textInput:input,
    suggestBox:{classList:{remove(){}}},caretParts:()=>({context:'',partial:'ca'}),
    fetchJson:()=>response.promise,drawSuggest(){drawn++;}});
  b.context.document.activeElement=input;
  b.run(section(read('index.html'),'function closeSuggest','function drawSuggest'));
  b.run(section(read('index.html'),'async function fetchSuggest',"textInput.addEventListener('blur'"));
  const pending=b.context.fetchSuggest();b.context.closeSuggest();
  b.context.document.activeElement=null;
  response.resolve({ok:true,data:{words:[{word:'cat',clips:1}]}});await pending;
  assert.equal(drawn,0);assert.equal(b.context.items.length,0);
}

function historyCorpusSafety() {
  let votes=0,reports=0;
  const node=()=>({children:[],dataset:{},appendChild(child){this.children.push(child);},addEventListener(){}});
  const words=node();
  const b=sandbox({corpusSlug:'current',marks:[],TITLES:{found:'found'},
    voteButton(){votes++;return node();},reportButton(){reports++;return node();},voiceName:x=>x,
    document:{getElementById:()=>words,createElement:node}});
  b.run(section(read('index.html'),'function drawWords','// The word being said'));
  const result={tokens:[{word:'cat',status:'found',clips:[1]}]};
  b.context.drawWords({...result,corpus:'other'});b.context.drawWords(result);
  assert.equal(votes,0);assert.equal(reports,0);
  b.context.drawWords({...result,corpus:'current'});assert.equal(votes,2);assert.equal(reports,1);
}

function spectrumSilenceGain() {
  const colors=[];
  const b=sandbox();
  b.context.ctx={set fillStyle(value){colors.push(value);},fillRect(){}};
  b.run('drawMeasuredSpectrogram(ctx, {values:[[0,255]],bins:2,db_min:-80,db_max:0}, 10,10, 10, x=>x);');
  assert.deepEqual(colors,[0,1]);
}

async function escapedDiagnostics() {
  const evil='<img src=x onerror="globalThis.injected=true">';
  const data={summary:{},unindexed_downloads:[evil],negative_margins:[{id:1,word:evil,reason:evil}],
    low_confidence_clips:[{id:1,word:evil,start_time:0,end_time:1,reason:evil}],
    words_missing_pronunciations:[evil],downvoted_clips:[{clip_id:1,clip_word:evil,target_word:evil}]};
  for(const page of ['editor.html','splice.html']) {
    const b=sandbox({corpusSlug:'test',fetch:async()=>({ok:true,json:async()=>data})});
    b.run(section(read(page),'async function loadDiagnostics',"$('diag-btn').onclick"));
    await b.context.loadDiagnostics();
    assert(!b.element('diag-details').innerHTML.includes('<img'),`${page} must quote stored labels`);
    assert(b.element('diag-details').innerHTML.includes('&lt;img'));
  }
  const b=sandbox();
  b.run(section(read('splice.html'),'function renderSummary','function renderCurrent'));
  b.context.renderSummary({word:evil,phones:null});
  assert(!b.element('summary').innerHTML.includes('<img'));
}

async function main() {
  for(const page of ['editor.html','splice.html','ytpmv.html','index.html']) {
    for(const script of read(page).matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/g))
      new vm.Script(script[1],{filename:page});
  }
  await sourceRace(); await editorCorpusSwitch();await saveAndDeleteSelection(); await spliceCandidateRace();
  await sampleRace(); await cancellationFailure(); await escapedDiagnostics();
  await settingsFailure();await suggestionBlurRace();historyCorpusSafety();spectrumSilenceGain();
  console.log('PASS frontend: script parsing, source/save/delete/candidate/sample/suggestion races, failure recovery, history corpus safety, stored HTML escaping, measured silence gain');
}
main().catch(error=>{console.error(error);process.exitCode=1;});
