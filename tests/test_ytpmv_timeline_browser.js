const assert=require('node:assert/strict');
const path=require('node:path');
async function run(chromium,origin='http://127.0.0.1:8877') {
 const browser=await chromium.launch({headless:true,...(process.env.MRS_TEST_CHROMIUM?{executablePath:process.env.MRS_TEST_CHROMIUM}:{})});
 const page=await browser.newPage({viewport:{width:1280,height:1000}}),errors=[],previewRequests=[];
 page.on('pageerror',e=>errors.push(e.message));
 page.on('request',r=>{if(r.url().endsWith('/api/ytpmv/preview-mix'))previewRequests.push(r.postDataJSON());});
 try {
  await page.goto(origin+'/ytpmv.html');await page.locator('#open-midi-library').click();
  await page.locator('.lib-row[data-mid="fixture-song"]').click();await page.locator('.track[data-track=lead]').waitFor();
  const viewport=page.locator('#track-lanes');
  assert.equal(await page.locator('#preview-selection').isChecked(),false);
  assert.equal(await page.locator('.timeline-range').isVisible(),false);
  await page.locator('#preview-mix').click();await page.waitForFunction(()=>document.getElementById('player').getAttribute('src'));
  assert.equal(previewRequests.at(-1).seconds,null);
  assert.equal(previewRequests.at(-1).start,0);
  await page.locator('#player').evaluate(e=>{e.pause();e.currentTime=0;});
  const count=previewRequests.length;
  await page.locator('#preview-mix').click();
  assert.equal(await page.locator('#player').evaluate(e=>e.paused),false);
  await page.locator('#preview-mix').click();
  assert.equal(await page.locator('#player').evaluate(e=>e.paused),true);
  assert.equal(previewRequests.length,count,'pause/resume reuses prepared audio');
  await page.locator('#stop-preview').click();
  // Zoom creates a genuinely wider clock. Track headers remain pinned.
  await page.locator('#timeline-zoom').fill('40');
  await page.waitForResponse(r=>r.url().includes('/notes?')&&r.ok());
  assert(await viewport.evaluate(e=>e.scrollWidth>e.clientWidth*8));
  const head=await page.locator('.track[data-track=lead] .track-head').boundingBox();
  await viewport.evaluate(e=>{e.scrollLeft=2000;});
  await page.waitForTimeout(60);
  assert(Math.abs((await page.locator('.track[data-track=lead] .track-head').boundingBox()).x-head.x)<1);
  const ruler=page.locator('.time-ruler');
  const root=await viewport.boundingBox(),headerWidth=await page.locator('.track-head').first().evaluate(e=>e.getBoundingClientRect().width);
  await page.mouse.click(root.x+headerWidth+120,root.y+14);
  const expected=await ruler.evaluate((e,x)=>(x-e.getBoundingClientRect().left)/e.getBoundingClientRect().width*12,root.x+headerWidth+120);
  assert(Math.abs(Number(await page.locator('#preview-from').inputValue())-expected)<.03);
  await page.locator('[data-view=fit]').click();
  assert(await viewport.evaluate(e=>e.scrollWidth<=e.clientWidth+2));
  // Drag a range, then focus it. The same start reaches the preview worker.
  const r=await ruler.boundingBox();
  await page.keyboard.down('Shift');await page.mouse.move(r.x+r.width*.2,r.y+15);await page.mouse.down();
  await page.mouse.move(r.x+r.width*.6,r.y+15,{steps:8});await page.mouse.up();await page.keyboard.up('Shift');
  assert(Math.abs(Number(await page.locator('#preview-from').inputValue())-2.4)<.05);
  assert(Math.abs(Number(await page.locator('#preview-seconds').inputValue())-4.8)<.15);
  await page.locator('[data-view=range]').click();assert(await viewport.evaluate(e=>e.scrollWidth>e.clientWidth*1.5));
  await page.locator('#preview-mix').click();await page.waitForFunction(()=>document.getElementById('player').getAttribute('src'));
  assert(Math.abs(previewRequests.at(-1).start-2.4)<.05);
  assert(Math.abs(previewRequests.at(-1).seconds-4.8)<.15);
  assert.equal(await page.locator('#preview-selection').isChecked(),true);
  // An actual audio seek, rather than a CSS animation, positions the cursor.
  await page.locator('#player').evaluate(async e=>{e.pause();if(!Number.isFinite(e.duration))await new Promise(r=>e.addEventListener('loadedmetadata',r,{once:true}));e.currentTime=Math.min(.2,e.duration*.5);});
  await page.waitForTimeout(80);
  const audioTime=await page.locator('#player').evaluate(e=>e.currentTime);
  assert((await page.locator('#timeline-position').textContent()).includes(audioTime.toFixed(2).padStart(5,'0')));
  await page.locator('[data-view=clear]').click();
  assert.equal(await page.locator('.timeline-range').isVisible(),false);
  await page.locator('#preview-mix').click();await page.waitForFunction(()=>document.getElementById('player').getAttribute('src'));
  assert.equal(previewRequests.at(-1).seconds,null);
  assert.equal(previewRequests.at(-1).start,0);
  await viewport.focus();await page.keyboard.press('Home');
  assert.equal(Number(await page.locator('#preview-from').inputValue()),0);
  await page.keyboard.press('Shift+ArrowRight');
  assert.equal(Number(await page.locator('#preview-from').inputValue()),10);
  // Large arrangements use a bounded scroll area and a small moving window
  // of track rows. Filtering still searches all tracks, including unmounted ones.
  await page.route('**/api/ytpmv/midi/fixture-second',async route=>{
    const data=await (await route.fetch()).json(),template=data.parts[0];
    data.duration=900;data.parts=Array.from({length:300},(_,i)=>({...template,id:`huge-${i}`,instrument:`Track ${i+1}`,first:i,
      activity:[[i,1,60],[i+12,2,67]],settings:{...template.settings,id:`huge-${i}`,text:null,mute:true,visible:false},recommended:{}}));
    await route.fulfill({json:data});
  });
  await page.locator('#open-midi-library').click();await page.locator('.lib-row[data-mid="fixture-second"]').click();
  await page.locator('[data-track="huge-0"]').waitFor();
  assert(await viewport.evaluate(e=>e.scrollHeight>e.clientHeight*20));
  assert(await page.locator('.track').count()<35);
  assert((await viewport.boundingBox()).height<600);
  await viewport.evaluate(e=>{e.scrollTop=e.scrollHeight;});
  await page.locator('[data-track="huge-299"]').waitFor();
  const v=await viewport.boundingBox(),pinned=await page.locator('.track-ruler').boundingBox();
  assert(Math.abs(pinned.y-v.y)<2);
  await page.locator('[data-select="huge-299"]').first().click();
  assert.equal(await page.locator('.part[data-id="huge-299"]').isVisible(),true);
  await page.locator('[data-view=part]').click();
  assert.equal(Number(await page.locator('#preview-from').inputValue()),299);
  await page.locator('#track-tools summary').click();await page.locator('#part-search').fill('Track 251');
  await page.locator('[data-track="huge-250"]').waitFor();assert.equal(await page.locator('.track').count(),1);
  await page.locator('#part-search').fill('');
  await page.locator('#timeline-zoom').fill('30');
  await viewport.hover();const before=await viewport.evaluate(e=>e.scrollLeft);
  await page.keyboard.down('Shift');await page.mouse.wheel(0,300);await page.keyboard.up('Shift');
  await page.waitForTimeout(80);assert(await viewport.evaluate(e=>e.scrollLeft)>before);
  await page.screenshot({path:path.resolve('output/ytpmv-smoke/timeline-large.png'),fullPage:true});
  await page.setViewportSize({width:390,height:844});
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
  assert(await page.locator('.track').count()<25);
  assert.deepEqual(errors,[]);
  console.log('PASS timeline: zoom/detail, pinned headers/ruler, horizontal/vertical scrolling, real seek/range/audio clock, keyboard, 300-track virtualization and full-track filtering');
 }finally{await browser.close();}
}
module.exports=run;
if(require.main===module)run(require('playwright').chromium).catch(e=>{console.error(e);process.exitCode=1;});
