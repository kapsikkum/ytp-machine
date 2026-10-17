const assert=require('node:assert/strict');
async function run(chromium,origin='http://127.0.0.1:8877') {
 const browser=await chromium.launch({headless:true,...(process.env.MRS_TEST_CHROMIUM?{executablePath:process.env.MRS_TEST_CHROMIUM}:{})});
 const page=await browser.newPage({viewport:{width:1280,height:1050}}),errors=[],banks=[];
 page.on('pageerror',e=>errors.push(e.message));
 page.on('request',r=>{if(r.url().endsWith('/api/ytpmv/preview-mix')&&r.postDataJSON().notes)banks.push(r.postDataJSON());});
 await page.addInitScript(()=>{
   const Native=window.AudioContext;
   window.__keyboardStarts=[];window.__keyboardClicks=0;
   document.addEventListener('click',e=>{if(e.target.closest('[data-note]'))window.__keyboardClicks=performance.now();},true);
   window.AudioContext=class extends Native{
     createBufferSource(){
       const source=super.createBufferSource(),start=source.start.bind(source);
       source.start=(when,offset,duration)=>{window.__keyboardStarts.push({delay:performance.now()-window.__keyboardClicks,offset,duration});return start(when,offset,duration);};
       return source;
     }
   };
 });
 const ready=()=>page.waitForFunction(()=>document.getElementById('track-instrument').dataset.keyboardState==='ready');
 try {
  await page.goto(origin+'/ytpmv.html?mode=advanced');await page.locator('.lib-row[data-mid=fixture-song]').click();
  await page.waitForFunction(()=>document.querySelector('.track')&&!document.querySelector('.part[aria-busy=true]')&&document.querySelector('.part[data-id=lead] [data-el=take]').textContent.includes('/3'));
  await page.locator('[data-track-sound=lead].track-sample').click();await ready();
  assert.equal(banks.length,1);assert.equal(banks[0].notes.length,13);
  for(let i=0;i<13;i++){
    await page.locator(`#track-instrument [data-note="${60+i}"]`).click();
    await page.waitForFunction(count=>window.__keyboardStarts.length===count,i+1);
  }
  assert.equal(banks.length,1,'all key presses reuse one decoded audio bank');
  const starts=await page.evaluate(()=>window.__keyboardStarts);
  assert.deepEqual(starts.map(s=>s.offset),Array.from({length:13},(_,i)=>i*2));
  const delay=Math.max(...starts.slice(1).map(s=>s.delay));assert(delay<100,`cached scheduling delay ${delay}ms`);
  await page.getByLabel('Close instrument',{exact:true}).click();
  await page.locator('[data-track-sound=lead].track-sample').click();await ready();
  assert.equal(banks.length,1,'reopening the same sound reuses its cache');
  await page.getByLabel('Quick instrument',{exact:true}).selectOption('opl2');
  await page.getByLabel('Quick patch',{exact:true}).selectOption('1');await ready();
  assert.equal(banks.length,2,'a changed patch prepares one fresh bank');
  assert.equal(banks[1].parts[0].program,1);assert.equal(banks[1].parts[0].tone,'opl2');
  await page.getByLabel('Quick patch',{exact:true}).selectOption('0');
  await page.getByLabel('Quick instrument',{exact:true}).selectOption('');await ready();
  assert.equal(banks.length,2,'returning to the previous sound reuses its decoded bank');
  await page.locator('#track-instrument [data-note="60"]').click();
  await page.waitForFunction(()=>document.querySelector('#track-instrument [data-note="60"]').classList.contains('playing'));
  await page.getByRole('button',{name:'Stop audition',exact:false}).click();
  assert.equal(await page.locator('#track-instrument .playing').count(),0);
  assert.deepEqual(errors,[]);
  console.log(`PASS keyboard cache: one 13-note bank, local playback with ${delay.toFixed(1)}ms maximum warm scheduling, no per-key requests, correct note offsets, patch invalidation, cache reuse and stop`);
 }finally{await browser.close();}
}
module.exports=run;
if(require.main===module)run(require('playwright').chromium).catch(e=>{console.error(e);process.exitCode=1;});
