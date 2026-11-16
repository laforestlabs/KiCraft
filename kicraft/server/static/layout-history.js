/* Self-contained layout history player. Data and labels are never interpreted as HTML. */
(function (root) {
  'use strict';
  const LIMIT = 32 * 1024 * 1024;
  const objectBytes = value => value == null ? 0 : JSON.stringify(value).length * 2;
  function makeState() { return {objects: new Map(), templates: new Map(), size: 0, ended: false, seq: 0, event: null, sourceScene: null, sourceFilter: null, geometrySource: null}; }
  function cloneState(s) { return {objects: new Map(s.objects), templates: new Map(s.templates), size: s.size, ended: s.ended, seq: s.seq, event: s.event, sourceScene: s.sourceScene, sourceFilter: s.sourceFilter, geometrySource: s.geometrySource}; }
  function apply(s, e) {
    if (!e || e.seq !== s.seq+1 || !Number.isSafeInteger(e.seq)) throw Error('Out-of-order sequence');
    if (s.ended) throw Error('Record after scope end');
    if (!['define','scene','delta','decision','end'].includes(e.kind)) throw Error('Unknown record kind: '+e.kind);
    if (e.kind==='define' && (typeof e.template_id!=='string' || !e.geometry)) throw Error('Invalid template');
    if (e.kind==='define' && s.templates.has(e.template_id) && JSON.stringify(s.templates.get(e.template_id))!==JSON.stringify(e.geometry)) throw Error('Immutable template redefined');
    if (e.kind==='scene' || e.kind==='delta') {
      const objects=e.kind==='scene' ? e.objects : e.put, remove=e.remove || [];
      if (!Array.isArray(objects) || !Array.isArray(remove)) throw Error('Invalid scene or delta');
      if(e.kind==='scene' && new Set(objects.map(o=>o && o.id)).size!==objects.length) throw Error('Duplicate scene object');
      if(e.source_scene && (typeof e.source_scene.scope_id!=='string' || (e.source_scene.seq != null && !Number.isSafeInteger(e.source_scene.seq)))) throw Error('Invalid source scene');
      if(e.source_filter && (!Array.isArray(e.source_filter) || e.source_filter.some(x=>typeof x!=='string'))) throw Error('Invalid source filter');
      const removed=new Set();
      for (const id of remove) {
        if(e.kind==='scene' || removed.has(id) || !s.objects.has(id)) throw Error('Unknown or duplicate removal: '+id);
        removed.add(id);
      }
      for (const o of objects) {
        if(!o || typeof o.id!=='string' || typeof o.type!=='string') throw Error('Invalid object');
        if(!['footprint','group','track','arc','via','zone','outline','keepout'].includes(o.type)) throw Error('Unknown object type: '+o.type);
        if(o.template_id && !s.templates.has(o.template_id)) throw Error('Missing template: '+o.template_id);
        if(o.pose && !['x','y','rotation'].every(k=>typeof o.pose[k]==='number' && Number.isFinite(o.pose[k]))) throw Error('Invalid transform');
      }
    }
    const undo = {seq: s.seq, event: s.event, size: s.size, ended: s.ended, sourceScene: s.sourceScene, sourceFilter: s.sourceFilter, geometrySource: s.geometrySource, changes: [], template: null, scene: null};
    if (e.kind === 'define') {
      undo.template = [e.template_id, s.templates.has(e.template_id), s.templates.get(e.template_id)];
      s.size += objectBytes(e.geometry) - objectBytes(s.templates.get(e.template_id));
      s.templates.set(e.template_id, e.geometry);
    } else if (e.kind === 'scene') {
      undo.scene = s.objects;
      for (const o of s.objects.values()) s.size -= objectBytes(o);
      for (const o of e.objects) s.size += objectBytes(o);
      s.objects = new Map((e.objects || []).map(o => [o.id, o]));
      s.sourceScene = e.source_scene || null;
      s.sourceFilter = e.source_filter || null;
    } else if (e.kind === 'delta') {
      for (const id of e.remove || []) {
        if (!s.objects.has(id)) throw new Error('Unknown removal: ' + id);
        s.size -= objectBytes(s.objects.get(id));
        undo.changes.push([id, true, s.objects.get(id)]); s.objects.delete(id);
      }
      for (const o of e.put || []) {
        s.size += objectBytes(o) - objectBytes(s.objects.get(o.id));
        undo.changes.push([o.id, s.objects.has(o.id), s.objects.get(o.id)]); s.objects.set(o.id, o);
      }
    }
    if(e.kind==='scene')s.geometrySource=e.geometry_source || null;
    else if(e.kind==='delta' && e.geometry_source)s.geometrySource=e.geometry_source;
    if (e.kind==='end') s.ended=true;
    s.seq = e.seq; s.event = e; return undo;
  }
  function reverse(s, u) {
    if (u.scene) s.objects = u.scene;
    for (let i = u.changes.length - 1; i >= 0; --i) {
      const [id, had, obj] = u.changes[i]; if (had) s.objects.set(id, obj); else s.objects.delete(id);
    }
    if (u.template) { const [id, had, obj] = u.template; if (had) s.templates.set(id, obj); else s.templates.delete(id); }
    s.seq = u.seq; s.event = u.event; s.size=u.size; s.ended=u.ended; s.sourceScene = u.sourceScene; s.sourceFilter = u.sourceFilter; s.geometrySource=u.geometrySource;
  }
  // Rotation followed by translation; optional mirror is in the local frame.
  function transformPoint(point, pose) {
    const p = pose || {}, a = (+p.rotation || 0) * Math.PI / 180;
    const x = (p.side === 'back' || p.mirror ? -1 : 1) * point[0], y = point[1];
    return [(+p.x || 0) + x * Math.cos(a) - y * Math.sin(a), (+p.y || 0) + x * Math.sin(a) + y * Math.cos(a)];
  }
  function transformChild(point, origin, pose) {
    const o = Array.isArray(origin) ? origin : [origin && origin.x || 0, origin && origin.y || 0];
    const p = pose || {}, a = (+p.rotation || 0) * Math.PI / 180;
    const x = point[0] - o[0], y = point[1] - o[1];
    return [(+p.x || 0) + x * Math.cos(a) + y * Math.sin(a), (+p.y || 0) - x * Math.sin(a) + y * Math.cos(a)];
  }
  function mapNet(net, maps) {
    for (const map of maps) if (map && Object.prototype.hasOwnProperty.call(map, net)) net=map[net];
    return net;
  }
  function arcLength(a,m,b) {
    const d=2*(a[0]*(m[1]-b[1])+m[0]*(b[1]-a[1])+b[0]*(a[1]-m[1]));
    if(Math.abs(d)<1e-12)return Math.hypot(b[0]-a[0],b[1]-a[1]);
    const aa=a[0]*a[0]+a[1]*a[1],mm=m[0]*m[0]+m[1]*m[1],bb=b[0]*b[0]+b[1]*b[1];
    const x=(aa*(m[1]-b[1])+mm*(b[1]-a[1])+bb*(a[1]-m[1]))/d;
    const y=(aa*(b[0]-m[0])+mm*(a[0]-b[0])+bb*(m[0]-a[0]))/d;
    const start=Math.atan2(a[1]-y,a[0]-x), mid=Math.atan2(m[1]-y,m[0]-x), end=Math.atan2(b[1]-y,b[0]-x);
    const mod=v=>(v+2*Math.PI)%(2*Math.PI), sweep=mod(end-start);
    return Math.hypot(a[0]-x,a[1]-y)*(mod(mid-start)>sweep ? 2*Math.PI-sweep : sweep);
  }
  function overviewOrder(scopes, selectedId) {
    const sorted=scopes.slice().sort((a,b)=>(a.header.started_ns||0)-(b.header.started_ns||0));
    const selected=sorted.find(s=>s.header.scope_id===selectedId);
    return selected ? [...sorted.filter(s=>s!==selected),selected] : sorted;
  }
  root.kicraftLayoutHistory = {makeState, cloneState, apply, reverse, transformPoint, transformChild, mapNet, arcLength, overviewOrder};
  if (typeof document === 'undefined') return;
  const dataNode = document.getElementById('layout-history-data');
  if (!dataNode) return;
  let data;
  try { data = JSON.parse(dataNode.textContent); } catch (_) { dataNode.textContent = 'Invalid layout history data.'; return; }
  const mount = document.getElementById('layout-history-player') || document.body;
  const el = (tag, text, parent, cls) => { const x = document.createElement(tag); if (text != null) x.textContent = String(text); if (cls) x.className = cls; if (parent) parent.appendChild(x); return x; };
  const box = el('section', null, mount, 'lh-player');
  if (data.schema_version !== 1) { el('p', 'Unsupported layout history schema.', box, 'lh-warning'); return; }
  if (typeof DecompressionStream === 'undefined') { el('p', 'Playback requires a browser with DecompressionStream support.', box, 'lh-warning'); return; }
  const title = el('div', null, box, 'lh-title');
  el('strong', 'Layout history', title);
  const buildResult=data.build_rc == null ? 'unavailable' : String(data.build_rc)+(data.build_rc===0?' (success)':' (failed)');
  el('div','History: '+(data.status||'unknown')+'; Build exit code: '+buildResult,box,'lh-verdict');
  if(data.selection_role==='partial_preview')el('div','Selected partial preview — not promoted; not fabrication approval.',box,'lh-verdict lh-warning');
  const warning = el('div', null, box, 'lh-warning');
  const currentGaps = el('div', null, box, 'lh-current-gaps');
  const gapDetails = el('details', null, box, 'lh-gap-details');
  const gapSummary = el('summary', null, gapDetails);
  const gapList = el('ul', null, gapDetails, 'lh-gap-list');
  const toolbar = el('div', null, box, 'lh-toolbar');
  const scopeSelect = el('select', null, toolbar); scopeSelect.setAttribute('aria-label', 'Attempt or scope');
  const phaseSelect = el('select', null, toolbar); phaseSelect.setAttribute('aria-label', 'Phase');
  const back = el('button', '◀ Step', toolbar), play = el('button', 'Play', toolbar), next = el('button', 'Step ▶', toolbar);
  const mode = el('select', null, toolbar); mode.setAttribute('aria-label', 'Playback mode');
  [['overview','Overview (60 seconds)'],['step','Every step']].forEach(([v,t]) => { const o=el('option',t,mode); o.value=v; });
  const speed = el('select', null, toolbar); speed.setAttribute('aria-label','Speed');
  [0.25,0.5,1,2,4,8].forEach(v => { const o=el('option',v+'×',speed); o.value=v; }); speed.value='1';
  const copyButton = el('button','Copy step reference',toolbar);
  const copyFeedback=el('div',null,box,'lh-copy-feedback');copyFeedback.setAttribute('aria-live','polite');copyFeedback.hidden=true;
  const copyReference=el('input',null,copyFeedback);copyReference.type='text';copyReference.readOnly=true;copyReference.setAttribute('aria-label','Step reference to copy');copyReference.onclick=()=>copyReference.select();
  const copyMessage=el('span',null,copyFeedback);
  const slider = el('input', null, box, 'lh-slider'); slider.type='range'; slider.min='0'; slider.step='1'; slider.value='0'; slider.setAttribute('aria-label','Event');
  const status = el('div', '', box, 'lh-status');
  const area = el('div',null,box,'lh-area'); const canvas=el('canvas',null,area); canvas.setAttribute('aria-label','Board layout');
  const side=el('aside',null,area,'lh-sidebar');
  const highlight=el('input',null,side); highlight.placeholder='Highlight ref or net'; highlight.setAttribute('aria-label','Highlight reference or net');
  const layersBox=el('div',null,side); el('strong','Layers',layersBox);
  const info=el('div',null,side,'lh-info');
  const ctx=canvas.getContext('2d');
  const scopes=(data.scopes || []).slice().sort((a,b) => (a.header.started_ns || 0)-(b.header.started_ns || 0));
  const overviewOption=el('option','All attempts · overview',scopeSelect); overviewOption.value='';
  const selectedId=data.selection && data.selection.scope_id;
  const selectedScope=scopes.find(s=>s.header.scope_id===selectedId);
  const overviewScopes=overviewOrder(scopes,selectedId);
  const overviewEvents=overviewScopes.reduce((sum,s)=>sum+(s.last_seq||0),0);
  for (const s of scopes) { const h=s.header || {}, o=el('option', (h.label || h.scope_kind || 'Scope') + ' · ' + (h.scope_kind || '') + (h.scope_kind === 'leaf' ? ' (parallel)' : '') + ' · ' + (h.scope_id || '').slice(0,8), scopeSelect); o.value=h.scope_id; }
  const gapText=g=>typeof g==='string'?g:JSON.stringify(g);
  const allGaps=(data.gaps||[]).map(gapText);
  const gapSet=new Set(allGaps);
  for(const s of scopes)for(const gap of s.gaps||[]){
    const text=(s.header.scope_id||'scope')+': '+gapText(gap);
    if(!gapSet.has(text)){gapSet.add(text);allGaps.push(text);}
  }
  gapDetails.hidden=!allGaps.length;
  gapSummary.textContent='Show all '+allGaps.length+' recording gaps';
  for(const gap of allGaps)el('li',gap,gapList);
  if (!scopes.length) { warning.textContent='No recorded scopes are available.'; return; }
  let scope, state, memberIndex=-1, memberEvents=[], inverse=[], inverseBytes=0, checkpoints=[], checkpointBytes=0, phases=new Map(), layers=new Map();
  let playing=false, timer=0, busy=false, view={scale:1,x:0,y:0,fit:true}, lastTick=0, carry=0, grouped=0, token=0;
  let sources = new Map(), sourceBytes = 0, memberBytes = 0, sourceSignature = '';
  const hash = new URLSearchParams(location.hash.slice(1));
  scopeSelect.value=hash.get('scope') || '';
  if (hash.get('scope') && !scopeSelect.value) scopeSelect.value='';
  function bytes(value) { return JSON.stringify(value).length * 2; }
  function trimBudget(extra=0) {
    const total=()=>state.size+sourceBytes+memberBytes+inverseBytes+checkpointBytes+extra;
    while(total()>LIMIT && checkpoints.length) checkpointBytes-=checkpoints.shift().size;
    while(total()>LIMIT && inverse.length) inverseBytes-=inverse.shift().size;
    if(total()>LIMIT) throw Error('Playback geometry and decoded data exceed the 32 MiB memory limit');
  }
  function validateHeader(h) {
    if(!h || h.kind!=='header' || h.schema_version!==1 || h.capture_id!==data.capture_id || typeof h.scope_id!=='string') throw Error('Unsupported or invalid scope header');
  }
  function pause() { playing=false; play.textContent='Play'; if (timer) cancelAnimationFrame(timer); timer=0; }
  async function member(n) {
    if (n===memberIndex) return memberEvents;
    const expectedScope=scope, expectedToken=token;
    memberEvents=[]; memberIndex=-1; memberBytes=0;
    const m=scope.members[n], binary=atob(m.data), a=new Uint8Array(binary.length);
    for (let i=0;i<binary.length;i++) a[i]=binary.charCodeAt(i);
    const stream=new Blob([a]).stream().pipeThrough(new DecompressionStream('gzip'));
    const text=await new Response(stream).text();
    if(expectedScope!==scope || expectedToken!==token)return null;
    trimBudget(text.length*2+a.length+binary.length*2);
    const lines=text.split('\n').filter(Boolean);
    if(lines.length){
      const first=JSON.parse(lines[0]);
      if(first.kind==='header'){validateHeader(first);if(first.scope_id!==scope.header.scope_id)throw Error('Scope identity mismatch');lines.shift();}
    }
    memberEvents=lines;memberIndex=n;memberBytes=objectBytes(lines);trimBudget();
    return memberEvents;
  }
  async function eventAt(seq) {
    const n=scope.members.findIndex(m => seq>=m.first_seq && seq<=m.last_seq);
    if (n<0) throw Error('Missing event '+seq);
    const current=scope, version=token, events=await member(n);
    if(!events || current!==scope || version!==token)return null;
    const line=events[seq-current.members[n].first_seq];
    if(!line)throw Error('Missing event '+seq);
    const e=JSON.parse(line);
    if(e.seq!==seq)throw Error('Out-of-order member record');
    return e;
  }
  async function loadSource(ref, ancestry) {
    const expectedToken=token;
    const key=ref.scope_id+':'+ref.seq;
    if (ancestry.has(key)) throw Error('Cyclic child scene '+key);
    if (sources.has(key)) return sources.get(key);
    const source=scopes.find(s=>s.header.scope_id===ref.scope_id);
    if (!source) throw Error('Missing child scene '+key);
    validateHeader(source.header);
    const result=makeState(), target=ref.seq == null ? source.last_seq : ref.seq;
    for (const m of source.members) {
      if (m.first_seq > target) break;
      const raw=atob(m.data), packed=new Uint8Array(raw.length);
      for(let i=0;i<raw.length;i++) packed[i]=raw.charCodeAt(i);
      const text=await new Response(new Blob([packed]).stream().pipeThrough(new DecompressionStream('gzip'))).text();
      if(expectedToken!==token)throw Error('Playback scope changed');
      trimBudget(result.size+text.length*2+packed.length+raw.length*2);
      for(const line of text.split('\n')) {
        if(!line)continue; const e=JSON.parse(line); if(e.kind==='header'){validateHeader(e);if(e.scope_id!==source.header.scope_id)throw Error('Scope identity mismatch');continue;}
        if(e.seq>target)break;
        if(e.seq!==result.seq+1)throw Error('Missing child event '+(result.seq+1));
        apply(result,e);
        trimBudget(result.size+text.length*2+packed.length+raw.length*2);
      }
    }
    if(result.seq!==target)throw Error('Incomplete child scene '+key);
    const size=result.size;
    trimBudget(size);
    sourceBytes+=size;
    const next=new Set(ancestry);next.add(key);
    try {
      if(result.sourceScene) await loadSource(result.sourceScene,next);
      for(const o of result.objects.values()) if(o.type==='group'&&o.source_scene) await loadSource(o.source_scene,next);
    } catch(error) { if(expectedToken===token)sourceBytes-=size; throw error; }
    sources.set(key,result); return result;
  }
  async function loadSources() {
    const ref=state.sourceScene || (scope.header.identity || {}).board_scene;
    const signature=JSON.stringify([ref,...Array.from(state.objects.values()).filter(o=>o.type==='group').map(o=>o.source_scene)]);
    if(signature!==sourceSignature){sources=new Map();sourceBytes=0;sourceSignature=signature;}
    if(ref) await loadSource(ref,new Set());
    for(const o of state.objects.values()) if(o.type==='group') {
      if(!o.source_scene) throw Error('Group has no source scene');
      await loadSource(o.source_scene,new Set());
    }
  }
  function remember(u) {
    const size=bytes(u.scene ? {scene:Array.from(u.scene), changes:u.changes, template:u.template, event:u.event} : u);
    inverse.push({u,size}); inverseBytes+=size;
    trimBudget();
  }
  function checkpoint() {
    const size=state.size;
    if(checkpoints.length>=8)checkpointBytes-=checkpoints.shift().size;
    try { trimBudget(size); } catch (_) { return; }
    checkpoints.push({seq:state.seq,state:cloneState(state),size}); checkpointBytes+=size;
  }
  async function seek(target) {
    validateHeader(scope.header);
    target=Math.max(0,Math.min(+target || 0,scope.last_seq || 0)); const mine=++token;
    if (target < state.seq) {
      while (state.seq>target && inverse.length) { const x=inverse.pop(); inverseBytes-=x.size; reverse(state,x.u); }
      if (state.seq>target) {
        const c=checkpoints.filter(c=>c.seq<=target).sort((a,b)=>b.seq-a.seq)[0];
        state=c ? cloneState(c.state) : makeState(); inverse=[]; inverseBytes=0;
      }
    }
    let count=0;
    while (state.seq<target) {
      const e=await eventAt(state.seq+1); if (mine!==token) return;
      const u=apply(state,e); remember(u); count++;
      if (e.phase && !phases.has(e.phase)) { phases.set(e.phase,e.seq); const o=el('option',e.phase,phaseSelect); o.value=e.seq; }
      if (state.seq % 500 === 0) checkpoint();
      if (count % 250 === 0) await new Promise(resolve=>setTimeout(resolve,0));
    }
    grouped=count;
    try { await loadSources(); } catch(error) { if(mine===token)warning.textContent='History gap: '+error.message; }
    if(mine===token) update();
  }
  function report(error) { pause(); warning.textContent='History gap: '+error.message+'; last usable step '+state.seq+'.'; }
  function chooseScope(target=null, keepPlaying=false) {
    if(!keepPlaying)pause();
    token++; scope=target || scopes.find(s=>s.header.scope_id===scopeSelect.value) || overviewScopes[0];
    state=makeState(); memberIndex=-1; memberEvents=[]; memberBytes=0; inverse=[]; inverseBytes=0; checkpoints=[]; checkpointBytes=0; phases=new Map(); layers=new Map();
    sources=new Map(); sourceBytes=0; sourceSignature=''; layersBox.replaceChildren(); el('strong','Layers',layersBox);
    try { validateHeader(scope.header); } catch(error) { report(error); return; }
    phaseSelect.replaceChildren(); const o=el('option','All phases',phaseSelect); o.value='0';
    for(const [phase,seq] of Object.entries(scope.phases || {}).sort((a,b)=>a[1]-b[1])) {
      if(!Number.isSafeInteger(seq) || seq<1 || seq>(scope.last_seq||0)) continue;
      phases.set(phase,seq);const option=el('option',phase,phaseSelect);option.value=seq;
    }
    slider.max=scope.last_seq || 0; slider.value=0; view.fit=true;
    const gaps=scope.gaps || [];
    warning.textContent=(data.status !== 'complete' ? 'Partial history ('+(data.status || 'unknown')+'). ' : '')+(allGaps.length ? allGaps.length+' recording gap'+(allGaps.length===1?'':'s')+'. ' : '')+'Last recorded step in this scope: '+(scope.last_seq || 0)+'.';
    currentGaps.textContent=gaps.length ? 'Current scope: '+gaps.map(gapText).join('; ') : '';
    update();
  }
  async function firstSceneSequence(target) {
    if(Number.isSafeInteger(target.baseline_seq) && target.baseline_seq>0 && target.baseline_seq<=(target.last_seq||0))return target.baseline_seq;
    if(target.baseline_seq===null)return null;
    const version=token;
    for(let seq=1;seq<=(target.last_seq||0);seq++){
      if(version!==token || scope!==target)return null;
      const event=await eventAt(seq);
      if(version!==token || scope!==target)return null;
      if(event.kind==='scene')return seq;
      if(seq%250===0)await new Promise(resolve=>setTimeout(resolve,0));
    }
    return null;
  }
  async function activateInitial(allAttempts) {
    while(true){
      const target=scope, version=token, first=await firstSceneSequence(target);
      if(version!==token || target!==scope)return;
      if(first!=null){await seek(first);return;}
      if(!allAttempts)return;
      const index=overviewScopes.indexOf(scope);
      if(index>=overviewScopes.length-1)return;
      if(scope.last_seq){const ticket=token+1;await seek(scope.last_seq);if(ticket!==token || scope!==target)return;}
      chooseScope(overviewScopes[index+1]);
    }
  }
  function color(layer) { let n=0; for (const c of layer) n=(n*31+c.charCodeAt(0))>>>0; return `hsl(${n%360} 70% 60%)`; }
  function layerVisible(layer) {
    layer=layer || 'Other';
    if (!layers.has(layer)) { layers.set(layer,true); const label=el('label',null,layersBox,'lh-layer'); const cb=el('input',null,label); cb.type='checkbox'; cb.checked=true; cb.onchange=()=>{layers.set(layer,cb.checked); draw();}; const sw=el('span','■ ',label); sw.style.color=color(layer); el('span',layer,label); }
    return layers.get(layer);
  }
  function xy(p) { if (!p) return null; if (Array.isArray(p)) return p; return [p.x || 0,p.y || 0]; }
  function path(points, close) { if (!points || !points.length) return; ctx.beginPath(); points.forEach((p,i)=>{p=xy(p); if(i)ctx.lineTo(p[0],p[1]); else ctx.moveTo(p[0],p[1]);}); if(close)ctx.closePath(); }
  function appendRings(rings) {
    for(const ring of rings || []) {
      const points=ring.points || ring.contour || ring;
      if(!points.length) continue;
      points.forEach((p,i)=>{p=xy(p);if(i)ctx.lineTo(...p);else ctx.moveTo(...p);});
      ctx.closePath();
      for(const hole of ring.holes || []) {
        hole.forEach((p,i)=>{p=xy(p);if(i)ctx.lineTo(...p);else ctx.moveTo(...p);});
        ctx.closePath();
      }
    }
  }
  function traceArc(a,m,b) {
    const d=2*(a[0]*(m[1]-b[1])+m[0]*(b[1]-a[1])+b[0]*(a[1]-m[1]));
    if(Math.abs(d)<1e-12){ctx.lineTo(...b);return;}
    const aa=a[0]*a[0]+a[1]*a[1], mm=m[0]*m[0]+m[1]*m[1], bb=b[0]*b[0]+b[1]*b[1];
    const x=(aa*(m[1]-b[1])+mm*(b[1]-a[1])+bb*(a[1]-m[1]))/d;
    const y=(aa*(b[0]-m[0])+mm*(a[0]-b[0])+bb*(m[0]-a[0]))/d;
    const start=Math.atan2(a[1]-y,a[0]-x), mid=Math.atan2(m[1]-y,m[0]-x), end=Math.atan2(b[1]-y,b[0]-x);
    const mod=v=>(v+2*Math.PI)%(2*Math.PI);
    ctx.arc(x,y,Math.hypot(a[0]-x,a[1]-y),start,end,mod(mid-start)>mod(end-start));
  }
  function drawScene(scene, filter, netMaps=[]) {
    if(scene.sourceScene){
      const ref=scene.sourceScene, source=sources.get(ref.scope_id+':'+ref.seq);
      const own=scene.sourceFilter;
      const combined=filter && own ? own.filter(type=>filter.includes(type)) : (filter || own);
      if(source)drawScene(source,combined,netMaps);
    }
    for(const item of scene.objects.values()) if(!filter || filter.includes(item.type) || item.type==='group') drawObject(item,scene.templates,filter,netMaps);
  }
  function drawObject(o, templates, filter, netMaps) {
    if (o.type==='group') {
      const ref=o.source_scene, child=ref && sources.get(ref.scope_id+':'+ref.seq);
      if(child){const p=o.pose||{}, origin=xy(o.origin)||[0,0];ctx.save();ctx.translate(+p.x||0,+p.y||0);ctx.rotate(-(+p.rotation||0)*Math.PI/180);ctx.translate(-origin[0],-origin[1]);drawScene(child,filter,[o.net_map,...netMaps]);ctx.restore();}
      return;
    }
    const layer=o.layer || (o.type==='footprint' ? (o.pose && o.pose.side==='back'?'B.Cu':'F.Cu') : 'Other');
    if (o.type==='via' && o.layers && o.layers.length) { if(!o.layers.map(layerVisible).some(Boolean))return; }
    else if (o.type!=='footprint' && !layerVisible(layer)) return;
    const query=highlight.value.trim().toLowerCase(), match=!query || String(o.ref || '').toLowerCase().includes(query) || String(mapNet(o.net || '',netMaps)).toLowerCase().includes(query);
    ctx.globalAlpha=match?1:0.18; ctx.strokeStyle=color(layer); ctx.fillStyle=color(layer); ctx.lineWidth=Math.max(0.06,+o.width || 0.1);
    if (o.type==='track' || o.type==='arc') {
      const a=xy(o.start), b=xy(o.end), m=xy(o.mid); if(a&&b){ctx.beginPath();ctx.moveTo(...a); if(m)traceArc(a,m,b);else ctx.lineTo(...b);ctx.stroke();}
    } else if(o.type==='via') { const p=xy(o.position); if(p){ctx.beginPath();ctx.arc(p[0],p[1],(+o.diameter||0.6)/2,0,Math.PI*2);ctx.fill();ctx.fillStyle='#151a21';ctx.beginPath();ctx.arc(p[0],p[1],(+o.drill||0.3)/2,0,Math.PI*2);ctx.fill();} }
    else if(o.type==='footprint') {
      const p=o.pose||{}, t=templates.get(o.template_id)||{}, native=t.geometry_source==='native_board';
      const bodyVisible=layerVisible(layer);
      ctx.save();ctx.translate(+p.x||0,+p.y||0);ctx.rotate((+p.rotation||0)*Math.PI/180);
      if(native && p.side==='back')ctx.scale(-1,1);
      const w=+t.width_mm||1,h=+t.height_mm||1;
      if(bodyVisible){
        ctx.lineWidth=0.08;
        if(!native)ctx.strokeRect(-w/2,-h/2,w,h);
        for(const contour of t.courtyard||t.courtyards||t.contours||[]){path(contour.contour||contour.points||contour,true);ctx.stroke();}
      }
      for(const pad of t.pads||[]){
        const q=xy(pad.pos||pad.position);if(!q)continue;
        const padMatch=match || (query && String(mapNet(pad.net||'',netMaps)).toLowerCase().includes(query));
        ctx.globalAlpha=padMatch?1:0.18;
        const shapes=pad.copper_shapes;
        let visible=false;
        if(shapes && Object.keys(shapes).length){
          for(const [padLayer,rings] of Object.entries(shapes)){
            if(!layerVisible(padLayer))continue;
            visible=true;ctx.fillStyle=color(padLayer);ctx.beginPath();appendRings(rings);ctx.fill('evenodd');
          }
        } else {
          const padLayers=pad.layers && pad.layers.length ? pad.layers : [layer];
          visible=padLayers.map(layerVisible).some(Boolean);
          if(visible){
            const pw=+(pad.size||[])[0]||+pad.width_mm||+pad.width||0.7,ph=+(pad.size||[])[1]||+pad.height_mm||+pad.height||0.7;
            ctx.fillStyle=color(padLayers[0]);ctx.save();ctx.translate(...q);ctx.rotate((+pad.rotation||0)*Math.PI/180);ctx.fillRect(-pw/2,-ph/2,pw,ph);ctx.restore();
          }
        }
        const drill=xy(pad.drill);
        if(visible && drill && drill[0]>0 && drill[1]>0){
          ctx.save();ctx.translate(...q);
          ctx.rotate((+pad.rotation||0)*Math.PI/180*(native && p.side==='back'?-1:1));
          ctx.fillStyle='#151a21';ctx.beginPath();ctx.ellipse(0,0,drill[0]/2,drill[1]/2,0,0,2*Math.PI);ctx.fill();ctx.restore();
        }
      }
      if(bodyVisible){ctx.globalAlpha=match?1:0.18;ctx.fillStyle=color(layer);if(native && p.side==='back')ctx.scale(-1,1);ctx.scale(1,-1);ctx.font='1px sans-serif';ctx.fillText(String(o.ref||''),-w/2,-h/2-0.2);}
      ctx.restore();
    } else if(o.type==='outline'||o.type==='zone'||o.type==='keepout') {
      const t=templates.get(o.template_id)||o; const contours=t.contours||o.contours||[]; ctx.beginPath();
      appendRings(contours);
      if(o.type==='zone'){ctx.globalAlpha*=0.25;ctx.fill('evenodd');} else ctx.stroke();
    }
    ctx.globalAlpha=1;
  }
  function draw() {
    const r=canvas.getBoundingClientRect(), d=devicePixelRatio||1; if(!r.width||!r.height)return;
    canvas.width=Math.round(r.width*d);canvas.height=Math.round(r.height*d);ctx.setTransform(d,0,0,d,0,0);ctx.fillStyle='#151a21';ctx.fillRect(0,0,r.width,r.height);
    if(view.fit){ let minX=Infinity,minY=Infinity,maxX=-Infinity,maxY=-Infinity; const add=p=>{p=xy(p);if(p&&Number.isFinite(p[0])&&Number.isFinite(p[1])){minX=Math.min(minX,p[0]);maxX=Math.max(maxX,p[0]);minY=Math.min(minY,p[1]);maxY=Math.max(maxY,p[1]);}};
      for(const o of state.objects.values()){if(o.pose)add([o.pose.x,o.pose.y]);add(o.position);add(o.start);add(o.end);const t=state.templates.get(o.template_id)||o;for(const c of t.contours||[])for(const p of c.contour||c.points||c)add(p);}
      if(Number.isFinite(minX)){view.scale=Math.min(r.width/Math.max(10,maxX-minX+10),r.height/Math.max(10,maxY-minY+10));view.x=r.width/2-(minX+maxX)/2*view.scale;view.y=r.height/2+(minY+maxY)/2*view.scale;view.fit=false;}else {view.scale=5;view.x=r.width/2;view.y=r.height/2;}
    }
    ctx.translate(view.x,view.y);ctx.scale(view.scale,-view.scale);
    const fallback=(scope.header.identity || {}).board_scene;
    if(!state.sourceScene && fallback){
      const source=sources.get(fallback.scope_id+':'+fallback.seq);
      if(source)drawScene(source,['footprint','outline','zone','keepout']);
    }
    drawScene(state,null);
  }
  function measureScene(scene, filter, totals) {
    if(scene.sourceScene){
      const ref=scene.sourceScene, source=sources.get(ref.scope_id+':'+ref.seq), own=scene.sourceFilter;
      const combined=filter && own ? own.filter(type=>filter.includes(type)) : (filter || own);
      if(source)measureScene(source,combined,totals);
    }
    for(const o of scene.objects.values()){
      if(o.type==='group'){
        const ref=o.source_scene, child=ref && sources.get(ref.scope_id+':'+ref.seq);
        if(child)measureScene(child,filter,totals);
      } else if(!filter || filter.includes(o.type)){
        if(o.type==='via') totals.vias++;
        if(o.type==='track'||o.type==='arc'){
          const a=xy(o.start), b=xy(o.end), m=xy(o.mid);
          if(a&&b)totals.length+=m ? arcLength(a,m,b) : Math.hypot(b[0]-a[0],b[1]-a[1]);
        }
      }
    }
  }
  function geometryLabel(source) {
    if(source==='placement_model')return 'placement_model (pad/body proxies, not native shapes)';
    if(source==='native_board')return 'native_board (native geometry)';
    if(source==='router')return 'router (routing geometry)';
    return source || 'not yet recorded';
  }
  function sourceLabels(scene, seen=new Set()) {
    const refs=[scene.sourceScene,...Array.from(scene.objects.values()).filter(o=>o.type==='group').map(o=>o.source_scene)];
    const labels=[];
    for(const ref of refs){
      if(!ref)continue;const key=ref.scope_id+':'+ref.seq;if(seen.has(key))continue;seen.add(key);
      const child=sources.get(key), meta=scopes.find(s=>s.header.scope_id===ref.scope_id);
      labels.push((meta && (meta.header.label||meta.header.scope_kind) || ref.scope_id)+' ['+String(ref.scope_id).slice(0,8)+']: '+(child?geometryLabel(child.geometrySource):'unavailable'));
      if(child)labels.push(...sourceLabels(child,seen));
    }
    return labels;
  }
  function update() {
    slider.value=state.seq; const e=state.event||{}, plus=(e.put||e.objects||[]).length, minus=(e.remove||[]).length;
    status.textContent=(scope.header.label || scope.header.scope_kind || 'Scope')+(scope.header.scope_kind==='leaf'?' (parallel)':'')+(scope===selectedScope?' · selected scene':'')+' · Step '+state.seq+' / '+(scope.last_seq||0)+' · '+(e.phase||(scope.baseline_seq===null?'metadata (no recorded scene)':'baseline'))+(e.iteration!=null?' · iteration '+e.iteration:'')+(e.action?' · '+e.action:'')+(e.reason?' · '+e.reason:'')+' · +'+plus+' / −'+minus+(grouped>1?' · '+grouped+' grouped events':'');
    const totals={vias:0,length:0}; measureScene(state,null,totals);
    const labels=sourceLabels(state), fallback=(scope.header.identity||{}).board_scene;
    if(fallback && !state.sourceScene){const child=sources.get(fallback.scope_id+':'+fallback.seq);labels.push('Pre-route board ['+String(fallback.scope_id).slice(0,8)+']: '+(child?geometryLabel(child.geometrySource):'unavailable'));}
    info.textContent='Geometry: '+geometryLabel(state.geometrySource)+(labels.length?'\nReferenced scenes:\n'+labels.join('\n'):'')+'\nVias: '+totals.vias+'\nTrace length: '+totals.length.toFixed(2)+' mm'+(e.metrics?'\nMeasured: '+JSON.stringify(e.metrics):'')+(e.score!=null?'\nScore: '+e.score:'')+(e.drc!=null?'\nDRC: '+JSON.stringify(e.drc):''); draw();
  }
  async function advance(count) {
    if(!scopeSelect.value){
      let i=overviewScopes.indexOf(scope);
      while(state.seq>=scope.last_seq && i<overviewScopes.length-1){
        chooseScope(overviewScopes[++i],true);
        if(mode.value!=='step'){
          const nextScope=scope, version=token, baseline=await firstSceneSequence(nextScope);
          if(version!==token || scope!==nextScope)return 0;
          if(baseline!=null && baseline>0){await seek(baseline);return baseline;}
        }
      }
    }
    const amount=Math.min(count,Math.max(0,(scope.last_seq||0)-state.seq));
    if(amount)await seek(state.seq+amount);
    return amount;
  }
  async function stepBack() {
    if(state.seq>0){await seek(state.seq-1);return;}
    if(!scopeSelect.value){
      const i=overviewScopes.indexOf(scope);
      if(i>0){chooseScope(overviewScopes[i-1]);await seek(scope.last_seq||0);}
    }
  }
  async function tick(now) {
    if(!playing)return; if(document.hidden){pause();return;} if(now-lastTick<1000/30){timer=requestAnimationFrame(tick);return;}
    const dt=lastTick ? Math.min(0.25,(now-lastTick)/1000) : 0; lastTick=now;
    carry+=mode.value==='step' ? dt*(+speed.value) : dt*(scopeSelect.value ? scope.last_seq||0 : overviewEvents)/60*(+speed.value);
    const count=mode.value==='step'?Math.min(1,Math.floor(carry)):Math.floor(carry);
    if(count&&!busy){busy=true;try{const done=await advance(count);carry-=done;if(!done)pause();}catch(e){report(e);}finally{busy=false;}}
    if(playing)timer=requestAnimationFrame(tick);
  }
  play.onclick=()=>{if(playing){pause();return;}token++;if(state.seq>=+slider.max && (scopeSelect.value || scope===overviewScopes[overviewScopes.length-1])){if(!scopeSelect.value)chooseScope(overviewScopes[0]);else seek(0).catch(report);}playing=true;play.textContent='Pause';lastTick=0;carry=0;if((scopeSelect.value ? scope.last_seq : overviewEvents)>0)timer=requestAnimationFrame(tick);else pause();};
  back.onclick=()=>{pause();token++;stepBack().catch(report);}; next.onclick=()=>{pause();token++;advance(1).catch(report);};
  slider.oninput=()=>{pause();seek(slider.value).catch(report);}; phaseSelect.onchange=()=>{pause();seek(phaseSelect.value).catch(report);}; scopeSelect.onchange=()=>{chooseScope();activateInitial(!scopeSelect.value).catch(report);};
  highlight.oninput=draw;
  copyButton.onclick=()=>{
    const h=new URLSearchParams({capture:data.capture_id,scope:scope.header.scope_id,event:String(state.seq),phase:(state.event||{}).phase||''});
    const ref=highlight.value.trim();if(ref)h.set('highlight',ref);
    const text='#'+h.toString(), temporary=el('textarea',null,box);
    temporary.value=text;temporary.readOnly=true;temporary.setAttribute('aria-hidden','true');
    temporary.style.position='fixed';temporary.style.left='-10000px';temporary.style.top='0';
    temporary.focus();temporary.select();let copied=false;
    try { copied=document.execCommand('copy'); } catch (_) { /* Report or try Clipboard API below. */ }
    temporary.remove();
    function success(){copyFeedback.hidden=false;copyReference.hidden=true;copyMessage.textContent='Step reference copied.';}
    function failure(){copyFeedback.hidden=false;copyReference.hidden=false;copyReference.value=text;copyMessage.textContent='Automatic copy unavailable. Select and copy the reference manually.';copyReference.focus();copyReference.select();}
    if(copied){success();return;}
    try { if(navigator.clipboard && navigator.clipboard.writeText){navigator.clipboard.writeText(text).then(success,failure);}else failure(); }
    catch (_) { failure(); }
  };
  canvas.onwheel=e=>{e.preventDefault();const r=canvas.getBoundingClientRect(),x=e.clientX-r.left,y=e.clientY-r.top,f=Math.exp(-e.deltaY*0.001);view.x=x-(x-view.x)*f;view.y=y-(y-view.y)*f;view.scale*=f;draw();};
  let drag=null; canvas.onpointerdown=e=>{drag=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId);};canvas.onpointermove=e=>{if(!drag)return;view.x+=e.clientX-drag[0];view.y+=e.clientY-drag[1];drag=[e.clientX,e.clientY];draw();};canvas.onpointerup=()=>{drag=null;};
  document.addEventListener('visibilitychange',()=>{if(document.hidden)pause();}); window.addEventListener('resize',draw);
  chooseScope(); if(hash.get('highlight'))highlight.value=hash.get('highlight');
  if(hash.has('event'))seek(hash.get('event')).catch(report);else activateInitial(!scopeSelect.value).catch(report);
})(typeof window === 'undefined' ? globalThis : window);
