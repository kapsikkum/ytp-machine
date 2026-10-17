const assert=require('node:assert/strict');
const path=require('node:path');
const time=text=>{const [m,s]=text.split(':');return Number(m)*60+Number(s);};
async function run(chromium,origin='http://127.0.0.1:8877') {
 const browser=await chromium.launch({headless:true,...(process.env.MRS_TEST_CHROMIUM?{executablePath:process.env.MRS_TEST_CHROMIUM}:{})});
 const page=await browser.newPage({viewport:{width:1440,height:1100}}),errors=[],requests=[];
 page.on('pageerror',e=>errors.push(e.message));
 page.on('request',r=>{if(r.url().endsWith('/api/ytpmv/preview-mix'))requests.push(r.postDataJSON());});
 try {
  await page.goto(origin+'/ytpmv.html');await page.locator('.lib-row[data-mid=fixture-song]').click();
  await page.waitForFunction(()=>document.querySelector('.track')&&!document.querySelector('.part[aria-busy=true]'));
  await page.locator('#song-title').click();await page.keyboard.press('Space');
  await page.waitForFunction(()=>!document.getElementById('player').paused);
  assert.equal(await page.locator('#player').isVisible(),false);
  assert.equal(await page.locator('#preview-mix').getAttribute('aria-pressed'),'true');
  await page.waitForFunction(()=>document.getElementById('player').currentTime>.25);
  const clock=await page.evaluate(()=>({audio:document.getElementById('player').currentTime,transport:document.getElementById('transport-clock').textContent,timeline:document.getElementById('timeline-position').textContent}));
  assert(clock.audio>.15);
  assert(Math.abs(clock.audio-time(clock.transport))<.08,JSON.stringify(clock));
  assert.equal(clock.transport,clock.timeline);
  await page.keyboard.press('Space');assert.equal(await page.locator('#player').evaluate(e=>e.paused),true);
  await page.locator('#transport-forward').click();assert(Number(await page.locator('#preview-from').inputValue())>=10);
  await page.locator('#transport-start').click();assert.equal(Number(await page.locator('#preview-from').inputValue()),0);
  // A dock audition must stay in place, including while audio is preparing.
  await page.evaluate(()=>{
    window.__unexpectedInstrument=false;
    document.getElementById('track-instrument').addEventListener('beforetoggle',e=>{if(e.newState==='open')window.__unexpectedInstrument=true;});
  });
  await page.locator('.part[data-id=lead] .sound [data-act=preview]').click();
  await page.waitForFunction(()=>!document.getElementById('instrument-player').paused);
  assert.equal(await page.evaluate(()=>window.__unexpectedInstrument),false,'part audition must not open the instrument window');
  assert.equal(await page.locator('#track-instrument').isVisible(),false);
  assert.equal(await page.locator('#instrument-status').isVisible(),true);
  assert.match(await page.locator('#instrument-status').textContent(),/Lead/);
  assert.equal(requests.findLast(r=>!r.notes).part_id,'lead');
  assert.deepEqual(requests.findLast(r=>!r.notes).parts.map(p=>[p.id,p.mute]),[['lead',false],['bass',true],['kick',true]]);
  assert.equal(await page.locator('#player').getAttribute('src'),null);
  await page.locator('#stop-preview').click();
  // Every track exposes the actual sound, and its plugin window edits the
  // original controls. Note auditions always mute other tracks in the request.
  assert.match(await page.locator('[data-track-sound=lead].track-sample').textContent(),/cat · take 1/);
  await page.locator('[data-track-sound=lead].track-sample').click();
  await page.locator('#track-instrument').waitFor({state:'visible'});
  assert.equal(await page.locator('#track-instrument [data-suggestions] button').count(),12);
  await page.getByRole('button',{name:'Audition C4',exact:true}).click();
  try { await page.waitForFunction(()=>document.querySelector('#track-instrument [data-note="60"]').classList.contains('playing'),{},{timeout:8000}); }
  catch(error) { console.log(await page.locator('#track-instrument').innerText(),errors,requests);throw error; }
  const note=requests.findLast(r=>r.notes);assert.deepEqual(note.notes,Array.from({length:13},(_,i)=>60+i));assert.equal(note.part_id,'lead');
  assert.deepEqual(note.parts.map(p=>[p.id,p.mute]),[['lead',false]]);
  assert.equal(await page.locator('#player').getAttribute('src'),null);
  await page.getByLabel('Quick sample',{exact:true}).fill('new sample');
  assert.equal(await page.locator('.part[data-id=lead] [data-k=text]').inputValue(),'new sample');
  await page.getByLabel('Quick instrument',{exact:true}).selectOption('opl2');
  await page.getByLabel('Quick patch',{exact:true}).selectOption('1');
  assert.match(await page.locator('[data-track=lead] .track-tone').textContent(),/Bright Piano/);
  await page.getByRole('button',{name:'Play track pattern',exact:false}).click();
  await page.waitForFunction(()=>document.getElementById('instrument-player').getAttribute('src'));
  assert.equal(requests.findLast(r=>!r.notes).note,null);
  assert.equal(requests.findLast(r=>!r.notes).part_id,'lead');
  assert.equal(await page.locator('#track-instrument').isVisible(),true);
  assert.equal(await page.locator('#track-instrument [data-note].playing').count(),0,'track patterns must not highlight an arbitrary piano key');
  await page.screenshot({path:path.resolve('output/ytpmv-smoke/instrument-window.png'),fullPage:true});
  await page.getByLabel('Close instrument',{exact:true}).click();
  await page.locator('.part[data-id=lead] .instrument-only [data-act=preview]').click();
  await page.waitForFunction(()=>!document.getElementById('instrument-player').paused);
  assert.equal(await page.locator('#track-instrument').isVisible(),false,'chip part audition must also stay in place after closing its instrument');
  assert.equal(await page.locator('#instrument-status').isVisible(),true);
  await page.locator('#stop-preview').click();
  await page.locator('#tab-video').click();
  assert.equal(await page.locator('.composition-tile').count(),3);
  await page.getByLabel('Video grid preset',{exact:true}).selectOption('featured');
  await page.waitForTimeout(250);
  await page.locator('#composition-preview').scrollIntoViewIfNeeded();
  let first=await page.locator('[data-composition-part=lead]').boundingBox(),box=await page.locator('#composition-preview').boundingBox();
  assert(Math.abs(first.width/box.width-2/3)<.02);
  const tile=page.locator('[data-composition-part=lead]'),target=page.locator('[data-composition-part=kick]');
  const a=await tile.boundingBox(),b=await target.boundingBox();
  await page.mouse.move(a.x+a.width/2,a.y+a.height/2);await page.mouse.down();
  await page.mouse.move(b.x+b.width/2,b.y+b.height/2,{steps:12});await page.mouse.up();
  const order=await page.locator('#opt-tile-order').inputValue();
  assert.equal(JSON.parse(order).length,3);
  assert.notEqual(JSON.parse(order)[0],'lead');
  await page.waitForTimeout(250);
  const visual=await page.locator('.composition-tile').evaluateAll(nodes=>nodes.sort((a,b)=>Number(a.dataset.slot)-Number(b.dataset.slot)).map(n=>n.dataset.compositionPart));
  assert.deepEqual(visual,JSON.parse(order));
  const expectedFirst=await page.locator(`[data-composition-part="${visual[0]}"]`).boundingBox();
  assert(Math.abs(expectedFirst.x-box.x)<3);
  await page.evaluate(()=>scrollTo(0,0));await page.screenshot({path:path.resolve('output/ytpmv-smoke/studio-video.png'),fullPage:true});
  const render=page.waitForRequest(r=>r.url().endsWith('/api/ytpmv/render'));
  await page.locator('#render-btn').click();const data=(await render).postDataJSON();
  assert.equal(data.options.layout,'featured');assert.deepEqual(data.options.tile_order,JSON.parse(order));
  await page.reload();await page.locator('.track').first().waitFor();
  assert.equal(await page.locator('#opt-layout').inputValue(),'featured');
  assert.equal(await page.locator('#opt-tile-order').inputValue(),order);
  await page.locator('#tab-video').click();
  for(const preset of ['balanced','two-rows','two-columns','strips','featured']){
    await page.locator('#opt-layout').selectOption(preset);await page.waitForTimeout(250);
    const area=await page.locator('.composition-tile').evaluateAll(nodes=>nodes.reduce((sum,n)=>{const r=n.getBoundingClientRect();return sum+r.width*r.height;},0));
    box=await page.locator('#composition-preview').boundingBox();assert(Math.abs(area/((box.width-2)*(box.height-2))-1)<.01);
  }
  await page.setViewportSize({width:320,height:900});
  await page.locator('[data-track-sound=lead].track-tone').click();
  const instrument=await page.locator('#track-instrument').boundingBox();
  assert(instrument.x>=0&&instrument.x+instrument.width<=320);
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
  await page.screenshot({path:path.resolve('output/ytpmv-smoke/instrument-mobile.png'),fullPage:true});
  assert.deepEqual(errors,[]);
  console.log('PASS studio: global Space, audio-coupled DAW clock/playhead, hidden native player, transport seek, quick sound editing, 12 suggestions, isolated note/pattern auditions, animated tile order, preset geometry/render payload/recall');
 }finally{await browser.close();}
}
module.exports=run;
if(require.main===module)run(require('playwright').chromium).catch(e=>{console.error(e);process.exitCode=1;});
