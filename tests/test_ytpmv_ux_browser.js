// Run against tests/frontend_fixture.py. Exercises real controls and recovery.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

async function run(chromium, origin = 'http://127.0.0.1:8877') {
  const browser = await chromium.launch({headless: true,
    ...(process.env.MRS_TEST_CHROMIUM ? {executablePath: process.env.MRS_TEST_CHROMIUM} : {})});
  const page = await browser.newPage({viewport: {width: 1280, height: 1000}});
  const errors = [], requests = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('request', request => {
    if (request.method() === 'POST' && request.headers()['content-type']?.includes('application/json'))
      requests.push({route: new URL(request.url()).pathname, data: request.postDataJSON()});
  });
  const part = id => page.locator(`.part[data-id="${id}"]`);
  const control = (id, key) => part(id).locator(`[data-k="${key}"]`);
  const selectPart = async id => { await page.locator(`[data-select="${id}"]`).first().click(); };
  const songSettings = async () => { await page.locator('#tab-song').click(); };
  const idle = () => page.waitForFunction(() =>
    !document.querySelector('.part[aria-busy="true"]'));
  const value = (id, key, expected) => page.waitForFunction(({id,key,expected}) =>
    document.querySelector(`.part[data-id="${id}"] [data-k="${key}"]`)?.value === expected,
    {id,key,expected});
  const openSong = async id => {
    await page.locator('#open-midi-library').click();
    await page.locator(`.lib-row[data-mid="${id}"]`).click();
    await page.waitForFunction(id => JSON.parse(localStorage.getItem('ytpmv-last-midi')) === id, id);
    await idle();
  };
  const latest = route => requests.filter(r => r.route === route).at(-1)?.data;
  const out = path.resolve(__dirname, '..', 'output', 'ytpmv-smoke');
  fs.mkdirSync(out, {recursive: true});
  try {
    await page.goto(origin + '/ytpmv.html');
    await page.locator('.lib-row[data-mid="fixture-song"]').waitFor({state: 'attached'});
    await openSong('fixture-song');
    await selectPart('lead');
    await part('lead').locator('[data-el=take]').getByText('take 1/3', {exact:true}).waitFor();
    await selectPart('lead');
    assert.equal(await control('lead', 'mode').isVisible(), false);
    await selectPart('lead');
    await part('lead').locator('[data-act=expand]').click();
    await selectPart('lead');
    await control('lead', 'volume-value').fill('65');
    await selectPart('lead');
    assert.equal(await control('lead', 'volume').inputValue(), '.65'.replace(/^\./,'0.'));
    await selectPart('lead');
    await control('lead', 'pan-value').fill('-35');
    await selectPart('lead');
    assert.equal(await control('lead', 'pan').inputValue(), '-0.35');
    await selectPart('lead');
    await part('lead').locator('[data-act=pan-center]').click();
    await selectPart('lead');
    assert.equal(await control('lead', 'pan-value').inputValue(), '0');
    await page.locator('#undo-btn').click();
    await selectPart('lead');
    assert.equal(await control('lead', 'pan-value').inputValue(), '-35');
    await selectPart('lead');
    await part('lead').locator('[data-act=reset-part]').click();
    await selectPart('lead');
    assert.equal(await control('lead', 'volume-value').inputValue(), '100');
    await page.locator('#undo-btn').click();
    await selectPart('lead');
    assert.equal(await control('lead', 'volume-value').inputValue(), '65');
    await selectPart('lead');
    await part('lead').locator('[data-act=expand]').click();

    // Source clips and numbered takes stay in the editor.
    await selectPart('lead');
    await control('lead', 'corpus').selectOption('fixture-other');
    await idle(); await value('lead', 'text', 'bell');
    await selectPart('lead');
    await part('lead').locator('[data-act=clip]').click();
    await selectPart('lead');
    assert.equal(await part('lead').locator('.sample-panel video').isVisible(), true);
    await selectPart('lead');
    await part('lead').locator('[data-el=sample-source]').getByText('Synthetic source recording').waitFor();
    await selectPart('lead');
    await control('lead', 'take-choice').fill('2');
    await selectPart('lead');
    await part('lead').locator('[data-el=take]').getByText('take 2/3', {exact:true}).waitFor();

    // The audition gets actual options and solo overrides without editing mute.
    await selectPart('bass');
    await control('bass', 'mute').check();
    await selectPart('bass');
    await control('bass', 'solo').check();
    await songSettings();
    await page.locator('#opt-speed').selectOption('1.25');
    await songSettings();
    await page.locator('#opt-transpose').selectOption('3');
    await songSettings();
    await page.locator('#opt-chip').selectOption('md-voice');
    await songSettings();
    await page.locator('#opt-draft').check();
    const previewRequest = page.waitForRequest(r => r.url().endsWith('/api/ytpmv/preview-mix'));
    await page.locator('#preview-mix').click(); await previewRequest;
    await page.waitForFunction(()=>document.getElementById('player').getAttribute('src'));
    const mix = latest('/api/ytpmv/preview-mix');
    assert.equal(mix.options.speed, 1.25); assert.equal(mix.options.transpose, 3);
    assert.equal(mix.options.chip, 'md-voice');
    assert.deepEqual(mix.parts.map(p => [p.id,p.mute]), [['lead',true],['bass',false],['kick',true]]);
    await selectPart('bass');
    assert.equal(await control('bass', 'mute').isChecked(), true);
    await page.locator('#stop-preview').click();
    assert.equal(await page.locator('#player').evaluate(el=>el.paused), true);
    // Stop cancels actual queued work; a failed cancellation remains retryable.
    let previewCancelled = false, cancellationFails = true;
    await page.route('**/api/ytpmv/preview-mix', route => route.fulfill({status:202,json:{id:'slow-preview',status:'queued'}}));
    await page.route('**/api/jobs/slow-preview', route => {
      if (route.request().method()==='DELETE') {
        if (cancellationFails) {cancellationFails=false;return route.abort('connectionrefused');}
        previewCancelled=true;return route.fulfill({json:{cancelled:true,status:'cancelled'}});
      }
      return route.fulfill({json:{status:'queued'}});
    });
    await page.locator('#preview-mix').click();
    await page.locator('#preview-status').getByText('Audio queued',{exact:true}).waitFor();
    await page.locator('#stop-preview').click();
    await page.locator('#preview-status').getByText(/could not confirm cancellation/).waitFor();
    await page.locator('#stop-preview').click();
    await page.locator('#preview-status').getByText('Playback stopped.',{exact:true}).waitFor();
    assert.equal(previewCancelled,true);
    await page.unroute('**/api/ytpmv/preview-mix'); await page.unroute('**/api/jobs/slow-preview');

    // Each library song has an independent full project, including options.
    await openSong('fixture-second');
    assert.equal(await page.locator('#opt-speed').inputValue(), '1');
    await selectPart('lead');
    assert.equal(await control('lead', 'volume').inputValue(), '1');
    await selectPart('lead');
    await control('lead', 'volume-value').fill('35');
    await openSong('fixture-song');
    assert.equal(await page.locator('#opt-speed').inputValue(), '1.25');
    assert.equal(await page.locator('#opt-transpose').inputValue(), '3');
    assert.equal(await page.locator('#opt-draft').isChecked(), true);
    await selectPart('lead');
    assert.equal(await control('lead', 'corpus').inputValue(), 'fixture-other');
    await selectPart('lead');
    assert.equal(await control('lead', 'volume').inputValue(), '0.65');
    await selectPart('lead');
    await page.reload(); await part('lead').waitFor(); await idle();
    assert.equal(await page.locator('#opt-speed').inputValue(), '1.25');
    assert.equal(await page.locator('#opt-chip').inputValue(), 'md-voice');

    // A failed/reloaded/new-song render keeps its own metadata and last export.
    await page.locator('#render-btn').click();
    await page.locator('#result video').waitFor();
    assert.equal(latest('/api/ytpmv/render').parts.find(p=>p.id==='bass').mute, true);
    const oldVideo = await page.locator('#result video').getAttribute('src');
    await songSettings();
    await page.locator('#opt-speed').selectOption('1.5');
    await page.locator('#tab-video').click();
    await page.locator('#result-state').getByText(/settings have changed/).waitFor();
    let state = 'queued', failPoll = true, queuedId = 'long-render';
    await page.route('**/api/ytpmv/render', route => route.fulfill({status:202, contentType:'application/json',
      body:JSON.stringify({id:queuedId,status:'queued'})}));
    await page.route('**/api/jobs/long-render*', route => {
      if (route.request().method()==='DELETE') {
        state='cancelled';return route.fulfill({json:{cancelled:true,status:'cancelled'}});
      }
      if (failPoll) {failPoll=false;return route.abort('connectionrefused');}
      return route.fulfill({json:{id:queuedId,status:state,result:state==='done' ? {
        video_url:'/api/fixture/video?render=second',duration:3,size:[480,270],problems:[]} : undefined}});
    });
    await page.locator('#render-btn').click();
    await page.locator('#reconnect-render').waitFor({state:'visible'});
    assert.equal(await page.locator('#render-btn').isDisabled(), true);
    assert.equal(await page.locator('#result video').getAttribute('src'), oldVideo);
    await selectPart('lead');
    await page.reload(); await part('lead').waitFor(); await idle();
    await page.locator('#cancel-render').waitFor({state:'visible'});
    await openSong('fixture-second');
    await selectPart('lead');
    assert.equal(await control('lead', 'volume').inputValue(), '0.35');
    state='done';
    await page.locator('#result .result-meta').getByText(/Synthetic trio.mid/).waitFor();
    assert.match(await page.locator('#result .download').getAttribute('download'), /^Synthetic trio/);
    await page.locator('#result-state').getByText(/from another song/).waitFor();
    await page.waitForFunction(()=>!document.getElementById('render-btn').disabled);
    queuedId='long-render-cancel'; state='queued';
    await page.locator('#render-btn').click();
    await page.locator('#cancel-render').waitFor({state:'visible'});
    await page.locator('#cancel-render').click();
    await page.waitForFunction(()=>!document.getElementById('render-btn').disabled);
    assert.equal(await page.locator('#result video').getAttribute('src'), '/api/fixture/video?render=second');

    // Filters, common edits and the responsive transport remain usable.
    await page.locator('#track-tools').evaluate(el=>{el.open=true;});
    await page.locator('#part-search').fill('bass');
    assert.equal(await page.locator('.track:visible').count(), 1);
    await page.locator('#part-search').fill('');
    await page.locator('#part-filter').selectOption('drums');
    assert.equal(await page.locator('.track:visible').count(), 1);
    await page.locator('#part-filter').selectOption('all');
    await songSettings();
    await page.locator('#bulk-tools').evaluate(el=>{el.open=true;});
    await page.locator('#bulk-corpus').selectOption('fixture-other');
    await page.locator('#apply-corpus').click(); await idle();
    await value('bass','text','boom');
    await selectPart('kick');
    assert.equal(await control('kick','corpus').inputValue(),'fixture-other');
    await songSettings();
    await page.locator('#bulk-tools').evaluate(el=>{el.open=false;});
    await page.screenshot({path:path.join(out,'ux-desktop.png'),fullPage:true});
    await page.setViewportSize({width:390,height:844});
    await page.evaluate(()=>scrollTo(0,document.body.scrollHeight));
    const header=await page.locator('header.top').boundingBox(), transport=await page.locator('#transport').boundingBox();
    assert(transport.y>=header.height-2 && transport.y<header.height+30,'transport stays below the header');
    assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
    await page.screenshot({path:path.join(out,'ux-mobile-viewport.png')});
    await page.evaluate(()=>scrollTo(0,0));
    await page.screenshot({path:path.join(out,'ux-mobile.png'),fullPage:true});
    assert.deepEqual(errors,[]);
    console.log('PASS YTPMV UX: numeric mixing, undo/reset, inline takes, accurate solo preview, per-MIDI full recall, render disconnect/reload/cancel/result identity, bulk voices, filters, desktop/mobile layout');
  } finally {await browser.close();}
}
module.exports=run;
if(require.main===module) run(require('playwright').chromium).catch(e=>{console.error(e);process.exitCode=1;});
