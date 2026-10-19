// Exploratory browser measurements against tests/frontend_fixture.py.
// Fixture response/hydration timings are not production network timings.
const fs=require('node:fs'),path=require('node:path');
async function run(chromium,origin='http://127.0.0.1:8891'){
  const browser=await chromium.launch({headless:true,...(process.env.MRS_TEST_CHROMIUM?{executablePath:process.env.MRS_TEST_CHROMIUM}:{})});
  const page=await browser.newPage({viewport:{width:1280,height:1000}});
  const errors=[];let payloadBytes=0;
  page.on('pageerror',e=>errors.push(e.message));
  try{
    await page.addInitScript(()=>{
      window.auditLongTasks=[];
      new PerformanceObserver(list=>window.auditLongTasks.push(...list.getEntries().map(e=>({start:e.startTime,duration:e.duration})))).observe({type:'longtask',buffered:true});
    });
    await page.route('**/api/ytpmv/midi/fixture-second',async route=>{
      const data=await(await route.fetch()).json(),p=data.parts[0];
      data.duration=900;data.parts=Array.from({length:300},(_,i)=>({...p,id:`audit-${i}`,instrument:`Track ${i+1}`,first:i,
        activity:Array.from({length:512},(_,j)=>[j*1.5,.4,48+(i+j)%36]),
        settings:{...p.settings,id:`audit-${i}`,text:null,mute:true,visible:false},recommended:{}}));
      payloadBytes=Buffer.byteLength(JSON.stringify(data));await route.fulfill({json:data});
    });
    await page.goto(origin+'/ytpmv.html');await page.locator('#open-midi-library').click();
    const before=await page.evaluate(()=>performance.now());
    await page.locator('.lib-row[data-mid="fixture-second"]').click();
    await page.locator('[data-track="audit-0"]').waitFor();
    const hydration=await page.evaluate(start=>performance.now()-start,before);
    const scroll=[];
    for(let i=1;i<=6;i++){
      scroll.push(await page.locator('#track-lanes').evaluate(async(e,i)=>{
        const begin=performance.now();e.scrollTop=(e.scrollHeight-e.clientHeight)*i/6;
        await new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)));
        return performance.now()-begin;
      },i));
    }
    await page.waitForTimeout(100);
    const metrics=await page.evaluate(start=>({
      visibleTimelineRows:document.querySelectorAll('.track').length,
      partEditors:document.querySelectorAll('.part').length,
      totalElements:document.querySelectorAll('*').length,
      longTasks:window.auditLongTasks.filter(e=>e.start>=start),
    }),before);
    if(errors.length)throw new Error(errors.join('\n'));
    const report={fixture:{tracks:300,notes:153600,viewport:[1280,1000]},payloadBytes,
      responseAndHydrationMs:hydration,scrollTwoFramesMs:scroll,...metrics,
      limitations:['One exploratory browser run; includes fixture response serialization and browser scheduling.',
        'Two-frame scroll duration includes the display frame interval; not pure handler CPU time.']};
    const destination=process.env.MRS_PERFORMANCE_BROWSER_OUTPUT||'output/performance-audit/browser.json';
    fs.mkdirSync(path.dirname(destination),{recursive:true});fs.writeFileSync(destination,JSON.stringify(report,null,2));
    console.log(JSON.stringify(report));
  }finally{await browser.close();}
}
module.exports=run;
