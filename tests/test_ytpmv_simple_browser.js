const assert=require('node:assert/strict');
const path=require('node:path');
async function run(chromium,origin='http://127.0.0.1:8877') {
 const browser=await chromium.launch({headless:true,...(process.env.MRS_TEST_CHROMIUM?{executablePath:process.env.MRS_TEST_CHROMIUM}:{})});
 const page=await browser.newPage({viewport:{width:1440,height:1050}}),errors=[];
 page.on('pageerror',e=>errors.push(e.message));
 try {
  await page.goto(origin+'/ytpmv.html?mode=simple');
  await page.locator('#simple-library summary').click();
  await page.locator('.lib-row[data-mid=fixture-song]').click();
  await page.waitForFunction(()=>document.querySelectorAll('.part').length===3&&!document.querySelector('.part[aria-busy=true]'));
  assert.equal(await page.locator('.part:visible').count(),3);
  assert.equal(await page.locator('#track-lanes').count(),0);
  assert.equal(await page.locator('#project-tabs').count(),0);
  assert.equal(await page.locator('#opt-chip').isVisible(),true);
  assert.equal(await page.locator('#options').isVisible(),true);
  assert.equal(await page.locator('#render-btn').isVisible(),true);
  assert.equal(await page.locator('#advanced-style').getAttribute('media'),'not all');
  assert.equal(await page.locator('#interface-mode [data-interface=simple]').getAttribute('aria-current'),'page');
  await page.locator('.part[data-id=lead] [data-k=volume]').fill('0.65');
  await page.locator('.part[data-id=bass] [data-k=corpus]').selectOption('fixture-other');
  await page.waitForFunction(()=>document.querySelector('.part[data-id=bass] [data-k=text]').value==='boom');
  await page.evaluate(()=>scrollTo(0,0));
  await page.screenshot({path:path.resolve('output/ytpmv-smoke/simple-desktop.png'),fullPage:true});
  await page.locator('#interface-mode [data-interface=advanced]').click();
  await page.locator('.track').first().waitFor();
  assert.equal(await page.locator('.part[data-id=lead] [data-k=volume]').inputValue(),'0.65');
  assert.equal(await page.locator('.part[data-id=bass] [data-k=corpus]').inputValue(),'fixture-other');
  await page.locator('#interface-mode [data-interface=simple]').click();
  await page.locator('.part').first().waitFor();
  assert.equal(await page.locator('.part:visible').count(),3);
  assert.equal(await page.locator('.part[data-id=lead] [data-k=volume]').inputValue(),'0.65');
  await page.goto(origin+'/ytpmv.html');
  await page.locator('.part').first().waitFor();
  assert.equal(await page.locator('#simple-library').count(),1,'preferred interface survives a bare URL');
  for(const width of [768,390,320]){
    await page.setViewportSize({width,height:900});
    if(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth))console.log(await page.evaluate(()=>[...document.querySelectorAll('body *')].filter(e=>{const r=e.getBoundingClientRect();return r.width&&r.right>innerWidth+1;}).map(e=>({id:e.id,tag:e.tagName,class:String(e.className),text:e.textContent.slice(0,60),right:e.getBoundingClientRect().right})).slice(0,12)));
    assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true,`simple overflow at ${width}`);
  }
  await page.evaluate(()=>scrollTo(0,0));await page.screenshot({path:path.resolve('output/ytpmv-smoke/simple-mobile.png'),fullPage:true});
  assert.deepEqual(errors,[]);
  console.log('PASS Simple interface: original stacked layout, all tracks/global controls, shared voice/settings across mode switches, preference recall, responsive layout');
 }finally{await browser.close();}
}
module.exports=run;
if(require.main===module)run(require('playwright').chromium).catch(e=>{console.error(e);process.exitCode=1;});
