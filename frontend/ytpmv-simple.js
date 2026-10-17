/* Original stacked page layout, sharing the current project and sound engine. */
window.createYtpmvSimple = function(editor) {
  const el=id=>document.getElementById(id);
  const view=document.createElement('main');view.id='view-editor';
  el('song').before(view);view.append(el('song'),el('parts-card'),el('render-card'));
  const library=document.createElement('details');library.id='simple-library';library.className='card';
  const summary=document.createElement('summary');summary.textContent='MIDI library';
  el('library').before(library);library.append(summary,el('library'));el('library').hidden=false;
  const actions=document.createElement('div');actions.id='actions';
  actions.append(el('render-btn'),el('cancel-render'),el('reconnect-render'));
  el('render-card').prepend(el('options'),actions,el('status'),el('progress'));
  const refresh=()=>{
    const song=editor.song();if(!song)return;
    for(const part of song.parts){const row=editor.row(part.id);if(row)row.hidden=row.dataset.filtered==='true';}
  };
  return {
    refresh,updateTrack(){},sampleChanged(){},stopInstrument(){},
    select(id){editor.row(id)?.scrollIntoView({block:'nearest'});},
    setTab(tab){if(tab==='video')el('result').scrollIntoView({block:'nearest'});},
    previewReady(){},instrumentPreparing(){},instrumentReady(){},
    clearSong(){for(const id of ['song','parts-card','render-card'])el(id).hidden=true;library.open=true;},
    get selected(){return null;},get tab(){return 'part';}
  };
};
