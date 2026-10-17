// Isolated cross-corpus browser regression; run tests/frontend_fixture.py first.
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
    const route = new URL(request.url()).pathname;
    if (!route.startsWith('/api/')) return;
    requests.push({route, method: request.method(), headers: request.headers(),
      data: request.headers()['content-type']?.startsWith('application/json') ? request.postDataJSON() : null});
  });
  const out = path.resolve(__dirname, '..', 'output', 'ytpmv-smoke');
  fs.mkdirSync(out, {recursive: true});
  const part = id => page.locator(`.part[data-id="${id}"]`);
  const control = (id, key) => part(id).locator(`[data-k="${key}"]`);
  const selectPart = async id => { await page.locator(`[data-select="${id}"]`).first().click(); };
  const songSettings = async () => { await page.locator('#tab-song').click(); };
  const waitValue = (id, key, value) => page.waitForFunction(({id, key, value}) =>
    document.querySelector(`.part[data-id="${id}"] [data-k="${key}"]`)?.value === value,
    {id, key, value});
  const idle = () => page.waitForFunction(() =>
    document.querySelectorAll('.part[aria-busy="true"]').length === 0 &&
    !document.getElementById('render-btn').disabled);
  const latest = route => requests.filter(request => request.route === route).at(-1);
  const withRequest = async (route, action) => {
    const requested = page.waitForRequest(request => new URL(request.url()).pathname === route);
    await action();
    const request = await requested;
    await request.response();
    return request;
  };
  try {
    await page.goto(origin + '/ytpmv.html');
    await page.locator('#voice-select option[value="fixture-other"]').waitFor({state: 'attached'});
    await page.locator('#open-midi-library').click();
    await page.locator('.lib-row[data-mid="fixture-song"]').click();
    await selectPart('lead');
    await part('lead').waitFor();
    await waitValue('lead', 'text', 'cat');
    await selectPart('lead');
    await part('lead').locator('[data-el=take]').getByText('take 1/3', {exact: true}).waitFor();
    await selectPart('lead');
    assert.deepEqual(await control('lead', 'corpus').locator('option').evaluateAll(options =>
      options.map(option => option.value)), ['', 'fixture', 'fixture-other']);

    // The corpus belongs to every voice treatment and auto voice, but not a chip tone.
    await selectPart('lead');
    assert.equal(await control('lead', 'corpus').isVisible(), true);
    await selectPart('lead');
    await control('lead', 'instrument').selectOption('md-fm');
    await selectPart('lead');
    assert.equal(await control('lead', 'corpus').isVisible(), false);
    await selectPart('lead');
    await control('lead', 'instrument').selectOption('clean');
    await selectPart('lead');
    assert.equal(await control('lead', 'corpus').isVisible(), true);
    await selectPart('lead');
    await control('lead', 'instrument').selectOption('robot');
    await selectPart('lead');
    assert.equal(await control('lead', 'corpus').isVisible(), true);
    await selectPart('lead');
    await control('lead', 'instrument').selectOption('');
    await songSettings();
    await page.locator('#opt-chip').selectOption('md');
    await selectPart('lead');
    assert.equal(await control('lead', 'corpus').isVisible(), false);
    await songSettings();
    await page.locator('#opt-chip').selectOption('md-voice');
    await selectPart('lead');
    assert.equal(await control('lead', 'corpus').isVisible(), true);
    await songSettings();
    await page.locator('#opt-chip').selectOption('');

    // Recast just one part. Its word, pitch, alternatives and vocabulary follow it.
    const alternateSample = page.waitForRequest(request =>
      new URL(request.url()).pathname === '/api/ytpmv/sample' &&
      request.postDataJSON().corpus === 'fixture-other');
    await selectPart('lead');
    await control('lead', 'corpus').selectOption('fixture-other');
    await waitValue('lead', 'text', 'bell');
    await idle();
    await selectPart('bass');
    assert.equal(await control('bass', 'text').inputValue(), 'second');
    await selectPart('bass');
    assert.equal(await control('bass', 'corpus').inputValue(), '');
    const sampleRequest = await alternateSample;
    assert.equal(sampleRequest.headers()['x-corpus'], 'fixture-other');
    assert.equal(sampleRequest.postDataJSON().text, 'bell');
    await selectPart('lead');
    await part('lead').locator('[data-el=alts] button[data-alt="chime"]').waitFor();
    await selectPart('lead');
    assert.match(await part('lead').locator('[data-el=badge]').innerText(), /C5/);
    const listId = await control('lead', 'text').getAttribute('list');
    await page.locator(`datalist[id="${listId}"] option[value="bell"]`).waitFor({state: 'attached'});
    assert.equal(await page.locator(`datalist[id="${listId}"] option[value="cat"]`).count(), 0);
    assert.equal(requests.some(request => request.route === '/api/words' &&
      request.headers['x-corpus'] === 'fixture-other'), true);

    // Mixer edits and explicit voices survive changing the page's default voice.
    await selectPart('lead');
    await control('lead', 'volume').fill('0.65');
    await selectPart('lead');
    await part('lead').locator('[data-act="expand"]').click();
    await selectPart('lead');
    await control('lead', 'pan').fill('0.25');
    await selectPart('lead');
    await control('lead', 'mode').selectOption('tape');
    await selectPart('lead');
    await part('lead').locator('[data-act="oct+"]').click();
    await page.locator('#voice-select').selectOption('fixture-other');
    await waitValue('bass', 'text', 'boom');
    await idle();
    await selectPart('lead');
    assert.equal(await control('lead', 'corpus').inputValue(), 'fixture-other');
    await selectPart('lead');
    assert.equal(await control('lead', 'text').inputValue(), 'bell');
    await selectPart('lead');
    assert.equal(await control('lead', 'volume').inputValue(), '0.65');
    await selectPart('lead');
    await part('lead').locator('[data-el=alts] button[data-alt="chime"]').waitFor();
    await selectPart('lead');
    assert.equal(await control('lead', 'pan').inputValue(), '0.25');
    await selectPart('lead');
    assert.equal(await control('lead', 'mode').inputValue(), 'tape');
    await selectPart('lead');
    assert.equal(await part('lead').locator('[data-el=oct]').innerText(), '0');
    await page.locator('#voice-select').selectOption('fixture');
    await waitValue('bass', 'text', 'second');
    await idle();

    const nextSample = await withRequest('/api/ytpmv/sample', () => part('lead').locator('[data-act="next"]').click());
    assert.equal(nextSample.postDataJSON().corpus, 'fixture-other');
    assert.equal(nextSample.postDataJSON().take, 1);
    const leadPreview = await withRequest('/api/ytpmv/preview-mix', () =>
      part('lead').locator('.sound [data-act="preview"]').click());
    assert.equal(leadPreview.postDataJSON().parts.find(p => p.id === 'lead').corpus, 'fixture-other');
    assert.equal(leadPreview.postDataJSON().parts.find(p => p.id === 'lead').text, 'bell');
    assert.equal(leadPreview.postDataJSON().part_id, 'lead');
    await selectPart('bass');
    const bassPreview = await withRequest('/api/ytpmv/preview-mix', () =>
      part('bass').locator('.sound [data-act="preview"]').click());
    assert.equal(bassPreview.postDataJSON().parts.find(p => p.id === 'bass').corpus, 'fixture');
    assert.equal(bassPreview.postDataJSON().parts.find(p => p.id === 'bass').text, 'second');
    await withRequest('/api/ytpmv/render', () => page.locator('#render-btn').click());
    const rendered = Object.fromEntries(latest('/api/ytpmv/render').data.parts.map(settings => [settings.id, settings]));
    assert.equal(rendered.lead.corpus, 'fixture-other');
    assert.equal(rendered.lead.volume, .65);
    assert.equal(rendered.lead.take, 1);
    assert.equal(rendered.bass.corpus, 'fixture');
    assert.equal(rendered.kick.corpus, 'fixture');
    await page.locator('#result video').waitFor();
    await page.screenshot({path: path.join(out, 'cross-corpus-desktop.png'), fullPage: true});

    // Both the part cast and page default survive an actual reload.
    await page.reload();
    await selectPart('lead');
    await part('lead').waitFor();
    await idle();
    assert.equal(await page.locator('#voice-select').inputValue(), 'fixture');
    await selectPart('lead');
    assert.equal(await control('lead', 'corpus').inputValue(), 'fixture-other');
    await selectPart('lead');
    assert.equal(await control('lead', 'text').inputValue(), 'bell');
    await selectPart('lead');
    assert.equal(await control('lead', 'volume').inputValue(), '0.65');
    await selectPart('bass');
    assert.equal(await control('bass', 'corpus').inputValue(), '');
    await selectPart('bass');
    assert.equal(await control('bass', 'text').inputValue(), 'second');
    await page.locator('#voice-select').selectOption('fixture-other');
    await waitValue('bass', 'text', 'boom');
    await idle();
    await page.reload();
    await selectPart('bass');
    await part('bass').waitFor();
    await idle();
    assert.equal(await page.locator('#voice-select').inputValue(), 'fixture-other');
    await selectPart('bass');
    assert.equal(await control('bass', 'text').inputValue(), 'boom');
    await selectPart('lead');
    assert.equal(await control('lead', 'corpus').inputValue(), 'fixture-other');

    // A rejected corpus switch keeps the previous usable cast and sample.
    const partRoute = '**/api/ytpmv/midi/fixture-song/part/lead?corpus=fixture';
    await page.route(partRoute, route => route.fulfill({status: 400,
      contentType: 'application/json', body: JSON.stringify({detail: 'Fixture corpus temporarily unavailable'})}));
    await selectPart('lead');
    await control('lead', 'corpus').selectOption('fixture');
    await idle();
    await selectPart('lead');
    assert.equal(await control('lead', 'corpus').inputValue(), 'fixture-other');
    await selectPart('lead');
    assert.equal(await control('lead', 'text').inputValue(), 'bell');
    await selectPart('lead');
    await part('lead').locator('.sound [data-el=err]').getByText('Fixture corpus temporarily unavailable', {exact: true}).waitFor();
    await page.unroute(partRoute);

    // A slow first choice cannot overwrite a newer choice or its sample.
    let slowFinished = false;
    await page.route(partRoute, async route => {
      const response = await route.fetch();
      await new Promise(resolve => setTimeout(resolve, 250));
      await route.fulfill({response});
      slowFinished = true;
    });
    await selectPart('lead');
    await control('lead', 'corpus').selectOption('fixture');
    await selectPart('lead');
    assert.equal(await part('lead').locator('.sound [data-act=preview]').isDisabled(), true);
    assert.equal(await page.locator('#render-btn').isDisabled(), true);
    await selectPart('lead');
    await control('lead', 'corpus').selectOption('fixture-other');
    await idle();
    await page.waitForResponse(response => response.url().includes('/part/lead?corpus=fixture'));
    await page.waitForTimeout(50);
    assert.equal(slowFinished, true);
    await selectPart('lead');
    assert.equal(await control('lead', 'text').inputValue(), 'bell');
    await selectPart('lead');
    assert.equal(await control('lead', 'corpus').inputValue(), 'fixture-other');
    await selectPart('lead');
    assert.match(await part('lead').locator('[data-el=badge]').innerText(), /C5/);
    assert.equal(requests.some(request => request.route === '/api/corpus' && request.method === 'POST'), false);
    await page.setViewportSize({width: 390, height: 844});
    await page.screenshot({path: path.join(out, 'cross-corpus-mobile.png'), fullPage: true});
    assert.deepEqual(errors, []);
    console.log('PASS YTPMV browser: installed corpus controls, voice/chip visibility, per-part suggestions/vocabulary, independent sample/preview/render voices, page-local default, mixer preservation, reload restoration, stale response protection; ' + out);
  } finally {
    await browser.close();
  }
}

module.exports = run;
if (require.main === module) run(require('playwright').chromium).catch(error => {
  console.error(error);
  process.exitCode = 1;
});
