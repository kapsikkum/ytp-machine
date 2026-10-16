const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

async function run(chromium, origin='http://127.0.0.1:8877') {
  const browser=await chromium.launch({headless:true,...(process.env.MRS_TEST_CHROMIUM?{executablePath:process.env.MRS_TEST_CHROMIUM}:{})});
  const page=await browser.newPage({viewport:{width:1440,height:1050}}), errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  const choose=async id=>page.locator(`[data-select="${id}"]`).first().click();
  const control=(id,key)=>page.locator(`.part[data-id="${id}"] [data-k="${key}"]`);
  const out=path.resolve(__dirname,'..','output','ytpmv-smoke');fs.mkdirSync(out,{recursive:true});
  try {
    await page.goto(origin+'/ytpmv.html');
    await page.locator('#library summary').click();
    await page.locator('.lib-row[data-mid="fixture-song"]').click();
    await page.locator('.track[data-track="lead"]').waitFor();
    await page.waitForFunction(()=>!document.querySelector('.part[aria-busy=true]'));
    assert.equal(await page.locator('.track:visible').count(),3);
    assert.equal(await page.locator('.part:visible').count(),1);
    assert.equal(await page.locator('.track[data-track="lead"] rect').count(),8);
    await choose('bass');
    assert.equal(await control('bass','text').isVisible(),true);
    assert.equal(await control('lead','text').isVisible(),false);
    await page.locator('[data-track-corpus="bass"]').selectOption('fixture-other');
    await page.waitForFunction(()=>document.querySelector('.part[data-id=bass] [data-k=text]').value==='boom');
    assert.equal(await control('bass','corpus').inputValue(),'fixture-other');
    await choose('lead');
    await control('lead','instrument').selectOption('opl2');
    assert.equal(await control('lead','corpus').isVisible(),false);
    assert.equal(await control('lead','program').isVisible(),true);
    await control('lead','program').selectOption('1');
    assert.match(await page.locator('.track[data-track=lead] .track-tone').textContent(),/Bright Piano/);
    await control('lead','instrument').selectOption('snes');
    assert.equal(await control('lead','program').inputValue(),'1');
    await control('lead','instrument').selectOption('clean');
    assert.equal(await control('lead','corpus').isVisible(),true);
    await page.locator('[data-track="bass"] [data-track-flag=solo]').click();
    assert.equal(await control('bass','solo').isChecked(),true);
    await page.locator('#preview-mix').getByText(/solo/).waitFor();
    await page.locator('.time-ruler').click({position:{x:200,y:10}});
    assert(Number(await page.locator('#preview-from').inputValue())>0);
    await page.locator('#tab-video').click();
    const cells=await page.locator('.composition-row button').evaluateAll(buttons=>buttons.map(b=>{const r=b.getBoundingClientRect();return {x:r.x,y:r.y,w:r.width,h:r.height};}));
    const box=await page.locator('#composition-preview').boundingBox();
    assert(Math.abs(box.width/box.height-16/9)<.01);
    assert.equal(cells.length,3);
    assert.equal(cells[0].y,cells[1].y);
    assert(cells[2].y>cells[0].y);
    assert(Math.abs(cells[2].w-cells[0].w-cells[1].w)<2);
    await page.locator('[data-composition-part=lead]').click();
    assert.equal(await control('lead','text').isVisible(),true);
    await page.screenshot({path:path.join(out,'workbench-desktop.png'),fullPage:true});
    await page.locator('#tab-video').click();
    await page.screenshot({path:path.join(out,'workbench-grid.png'),fullPage:true});
    for(const width of [768,390,320]) {
      await page.setViewportSize({width,height:900});
      const overflow=await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth);
      if(overflow)console.log(await page.evaluate(()=>[...document.querySelectorAll('body *')].filter(e=>{const r=e.getBoundingClientRect();return r.width&&r.right>innerWidth+1;}).map(e=>({tag:e.tagName,id:e.id,class:e.className,parent:e.parentElement.id||e.parentElement.className,text:e.textContent.slice(0,80),right:e.getBoundingClientRect().right})).slice(0,15)));
      assert.equal(overflow,false,`overflow at ${width}`);
      await choose('lead');
      assert.equal(await control('lead','instrument').isVisible(),true);
    }
    await page.screenshot({path:path.join(out,'workbench-mobile.png'),fullPage:true});
    assert.deepEqual(errors,[]);
    console.log('PASS workbench: real MIDI overview, selected-part dock, track corpus/solo, OPL/SNES patches, ruler seek, full 16:9 three-part composition, desktop/mobile');
  } finally { await browser.close(); }
}
module.exports=run;
if(require.main===module)run(require('playwright').chromium).catch(e=>{console.error(e);process.exitCode=1;});
