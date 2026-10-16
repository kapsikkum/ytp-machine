// Optional browser smoke using tests/frontend_fixture.py; requires Playwright.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

async function run(chromium, origin = 'http://127.0.0.1:8877') {
  const browser = await chromium.launch({headless:true,
    ...(process.env.MRS_TEST_CHROMIUM ? {executablePath:process.env.MRS_TEST_CHROMIUM} : {})});
  const errors=[];
  const page = await browser.newPage({viewport:{width:1280,height:1000}});
  page.on('pageerror', e=>errors.push(e.message));
  const out=path.resolve(__dirname,'..','output','frontend-smoke');fs.mkdirSync(out,{recursive:true});
  try {
    await page.goto(origin+'/editor.html');
    await page.locator('#words').getByText('cat',{exact:true}).waitFor();
    await page.locator('#source').selectOption('1');
    await page.locator('#source').selectOption('2');
    await page.locator('#words').getByText('different',{exact:true}).waitFor();
    await page.waitForTimeout(250);
    assert.equal(await page.locator('#words').getByText('cat',{exact:true}).count(),0);
    await page.locator('#source').selectOption('1');
    await page.locator('#words').getByText('cat',{exact:true}).click();
    // A focused word chip must play the unsaved boundaries, not select the
    // same clip again and restore the database values before playback.
    await page.evaluate(()=>{
      window.fixturePlayedSpans=[];
      const play=window.playSpan;
      window.playSpan=(a,b)=>{window.fixturePlayedSpans.push([a,b]);play(a,b);};
    });
    await page.locator('#w-start').fill('0.55');
    await page.locator('#w-end').fill('1.15');
    await page.locator('#words').getByText('cat',{exact:true}).focus();
    await page.keyboard.press('Space');
    assert.equal(await page.locator('#w-start').inputValue(),'0.55');
    assert.equal(await page.locator('#w-end').inputValue(),'1.15');
    assert.deepEqual(await page.evaluate(()=>window.fixturePlayedSpans),[[.55,1.15]]);
    await page.keyboard.press('Enter');
    assert.equal(await page.locator('#w-start').inputValue(),'0.55');
    await page.evaluate(()=>document.activeElement.blur());
    await page.keyboard.press('Space');
    assert.deepEqual(await page.evaluate(()=>window.fixturePlayedSpans),[[.55,1.15],[.55,1.15]]);
    await page.locator('#w-start').fill('0.400');
    await page.locator('#w-end').fill('1.400');
    await page.evaluate(()=>cancelPlayback());
    await page.locator('#spectrogram-toggle').check();
    await page.waitForResponse(r=>r.url().includes('/envelope')&&r.url().includes('spectrogram=true'));
    await page.waitForTimeout(100);
    await page.screenshot({path:path.join(out,'editor-spectrogram.png'),fullPage:true});
    await page.locator('#diag-btn').click();
    await page.locator('#diag-details').getByText('<img src=x onerror="window.fixtureInjected=true">',{exact:true}).first().waitFor();
    assert.equal(await page.locator('#diag-details img').count(),0);
    assert.equal(await page.evaluate(()=>window.fixtureInjected),undefined);
    await page.screenshot({path:path.join(out,'editor-diagnostics.png'),fullPage:true});
    await page.goto(origin+'/splice.html?word=cat');
    await page.locator('.clip[role=button]').first().waitFor();
    assert.equal(await page.locator('#summary img, #current img, .why img').count(),0);
    const second=page.locator('.clip[role=button]').nth(2);
    await second.focus();await page.keyboard.press('Enter');
    assert.equal(await page.locator('.clip.on').getAttribute('data-id'),'2');
    await page.locator('#spectrogram-toggle').check();
    await page.waitForTimeout(250);
    await page.screenshot({path:path.join(out,'splice-spectrogram.png'),fullPage:true});
    await page.locator('#diag-btn').click();
    await page.locator('#diag-details').getByText('<img src=x onerror="window.fixtureInjected=true">',{exact:true}).first().waitFor();
    assert.equal(await page.locator('#diag-details img').count(),0);
    assert.equal(await page.evaluate(()=>window.fixtureInjected),undefined);
    await page.goto(origin+'/index.html');
    await page.evaluate(()=>localStorage.setItem('ytp-history',JSON.stringify([{text:'Old fixture voice',
      url:'/api/edit/source/1/video',at:Date.now(),corpus:'other-voice',tokens:[{word:'cat',status:'found',clips:[1]}]}])));
    await page.reload();
    await page.locator('.past .replay').click();
    assert.equal(await page.locator('#word-results .acts').count(),0);
    await page.locator('#btn-server-settings').click();
    await page.locator('#server-encoder option').waitFor({state:'attached'});
    await page.locator('#btn-save-server-settings').click();
    await page.locator('#status').getByText('fixture write not implemented',{exact:true}).waitFor();
    assert.equal(await page.locator('#server-settings-modal').isVisible(),true);
    assert.equal(await page.locator('#btn-save-server-settings').isDisabled(),false);
    await page.goto(origin+'/ytpmv.html');
    await page.locator('#voice-select').getByText('Test voice',{exact:false}).waitFor({state:'attached'});
    assert.deepEqual(errors,[]);
    console.log('PASS browser: source race, measured spectra, literal hostile labels/diagnostics, keyboard clip choice, history corpus safety, settings failure recovery, YTPMV startup; '+out);
  } finally {await browser.close();}
}
module.exports=run;
if(require.main===module) run(require('playwright').chromium).catch(e=>{console.error(e);process.exitCode=1;});
