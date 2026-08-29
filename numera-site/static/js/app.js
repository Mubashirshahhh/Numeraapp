/* ── Numera — MathQuill edition ───────────────────────────── */
'use strict';

let MQ, mqField;
let currentLatex   = '';
let currentParsed  = null;
let currentJobId   = null;
let currentVideoUrl= null;
let activeLibId    = null;
let showGrid       = true;
let consoleOpen    = true;
let debounceTimer  = null;
let solutionAbort  = null;
let isEmpty        = true;

const COLORS = ['#58C4DD','#FC6255','#83C167','#f9e2af','#9B59B6','#fab387'];

/* ── Boot ─────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', async () => {
  initMathQuill();
  initGraph();
  await loadLibrary();
  setupVideoSlider();
  log('info', 'Numera ready — start typing or pick an equation from the library.');
});

/* ── MathQuill ─────────────────────────────────────────── */
function initMathQuill() {
  MQ = MathQuill.getInterface(2);

  const host = document.getElementById('mathField');
  mqField = MQ.MathField(host, {
    spaceBehavesLikeTab: false,
    handlers: {
      edit: onMQEdit,
      enter: () => generateAnimation(),
    },
    autoCommands: 'pi theta sqrt sum int',
    autoOperatorNames: 'sin cos tan ln log exp abs',
  });

  // After MQ transforms the span, force it to fill the container
  // MQ may convert the span to a block-like element internally
  const mqEl = host; // MQ mutates host in place
  Object.assign(mqEl.style, {
    flex: '1',
    minWidth: '0',
    width: '100%',
    background: 'transparent',
    border: 'none',
    boxShadow: 'none',
    fontSize: '22px',
    padding: '12px 4px',
    cursor: 'text',
    outline: 'none',
  });

  // Focus on wrap click
  document.getElementById('mqWrap').addEventListener('click', () => mqField.focus());

  // Show placeholder when empty
  updatePlaceholder();
}

function onMQEdit() {
  const latex = mqField.latex().trim();
  currentLatex = latex;
  isEmpty = latex === '';
  updatePlaceholder();

  clearTimeout(debounceTimer);
  if (!latex) { clearResults(); return; }
  debounceTimer = setTimeout(() => fetchAnalysis(latex), 380);
}

function updatePlaceholder() {
  document.getElementById('mqPlaceholder').style.display = isEmpty ? 'flex' : 'none';
}

function clearMQ() {
  mqField.latex('');
  currentLatex = '';
  isEmpty = true;
  updatePlaceholder();
  clearResults();
  document.querySelectorAll('.lib-card').forEach(c => c.classList.remove('active'));
  activeLibId = null;
  mqField.focus();
}

function insertExample(latex) {
  mqField.latex(latex);
  mqField.focus();
  onMQEdit();
}

/* ── Equation Library ──────────────────────────────────── */
async function loadLibrary() {
  try {
    const eqs = await fetch('/api/equations').then(r => r.json());
    const wrap = document.getElementById('libCards');
    wrap.innerHTML = '';
    eqs.forEach(eq => {
      const card = document.createElement('div');
      card.className = 'lib-card';
      card.id = `lc-${eq.id}`;
      card.innerHTML = `
        <span class="lc-name" style="color:${eq.color}">${eq.name}</span>
        <span class="lc-cat">${eq.category}</span>
        <span class="lc-latex" id="lc-lt-${eq.id}"></span>`;
      card.onclick = () => pickLib(eq);
      wrap.appendChild(card);
      // KaTeX render in card
      waitKatex(() => {
        const el = document.getElementById(`lc-lt-${eq.id}`);
        if (el) tryKatex(eq.latex, el, false);
      });
    });
  } catch(e) { log('err', 'Library load failed: ' + e); }
}

function pickLib(eq) {
  document.querySelectorAll('.lib-card').forEach(c => c.classList.remove('active'));
  document.getElementById(`lc-${eq.id}`)?.classList.add('active');
  activeLibId = eq.id;
  mqField.latex(eq.latex);
  isEmpty = false;
  updatePlaceholder();
  mqField.focus();
  onMQEdit();
}

/* ── Analysis ──────────────────────────────────────────── */
async function fetchAnalysis(latex) {
  try {
    const resp = await fetch('/api/analyze-latex', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ latex })
    });
    const data = await resp.json();

    if (data.error) {
      setHint('err', '⚠ ' + data.error.slice(0, 55));
      clearGraph();
      return;
    }

    currentParsed = data;
    setHint('ok', '✓ parsed');
    showPropsBar(data);
    renderGraph(data);
    fetchSolution(latex);

  } catch(e) {
    setHint('err', 'Connection error');
    log('err', String(e));
  }
}

function setHint(type, msg) {
  const el = document.getElementById('mqHint');
  el.textContent = msg;
  el.className = 'mq-hint ' + (type || '');
}

function showPropsBar(data) {
  document.getElementById('propsBar').style.display = 'flex';
  document.getElementById('propExpr').textContent  = data.expression || '—';
  document.getElementById('propDeriv').textContent = data.derivative || '—';
  document.getElementById('propInteg').textContent = data.integral   || '—';
}

/* ── Graph ─────────────────────────────────────────────── */
const LAYOUT = {
  paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)',
  font:{color:'#666', family:'Inter,sans-serif', size:11},
  xaxis:{gridcolor:'#1e1e1e', zerolinecolor:'#444', zerolinewidth:1.5, tickfont:{color:'#555'}, showgrid:true, zeroline:true},
  yaxis:{gridcolor:'#1e1e1e', zerolinecolor:'#444', zerolinewidth:1.5, tickfont:{color:'#555'}, showgrid:true, zeroline:true},
  margin:{l:40,r:16,t:16,b:36}, showlegend:false,
  hovermode:'x unified',
  hoverlabel:{bgcolor:'#161616', bordercolor:'#2e2e2e', font:{color:'#e8e8e8'}},
};
const CONFIG = {responsive:true, displayModeBar:false, scrollZoom:true, doubleClick:'reset'};

function initGraph() { Plotly.newPlot('graphDiv', [], LAYOUT, CONFIG); }

function renderGraph(data) {
  if (!data.plot) return;
  document.getElementById('graphEmpty').classList.add('hidden');
  Plotly.react('graphDiv', [{
    x:data.plot.x, y:data.plot.y, type:'scatter', mode:'lines',
    name: data.expression || 'f(x)',
    line:{color:COLORS[0], width:2.8, shape:'spline', smoothing:0.4},
    hovertemplate:`x=%{x:.3f}<br>y=%{y:.3f}<extra></extra>`
  }], LAYOUT, CONFIG);
  // Scroll graph into view smoothly so user sees it immediately
  setTimeout(() => {
    document.querySelector('.top-panels')?.scrollIntoView({ behavior:'smooth', block:'nearest' });
  }, 80);
}

function clearGraph() {
  Plotly.react('graphDiv', [], LAYOUT, CONFIG);
  document.getElementById('graphEmpty').classList.remove('hidden');
}

function clearResults() {
  setHint('', '');
  clearGraph();
  document.getElementById('propsBar').style.display = 'none';
  resetSol();
  currentParsed = null;
}

function zoomIn()  { adjZoom(.65); }
function zoomOut() { adjZoom(1.5); }
function adjZoom(s) {
  const l = document.getElementById('graphDiv').layout || {};
  const r = ax => { const [a,b]=(l[ax]||{}).range||[-6,6]; const m=(a+b)/2,h=(b-a)/2*s; return [m-h,m+h]; };
  Plotly.relayout('graphDiv', {'xaxis.range':r('xaxis'),'yaxis.range':r('yaxis')});
}
function resetView() { Plotly.relayout('graphDiv', {'xaxis.autorange':true,'yaxis.autorange':true}); }
function toggleGrid() {
  showGrid = !showGrid;
  document.getElementById('gridBtn').classList.toggle('tog', showGrid);
  Plotly.relayout('graphDiv', {'xaxis.showgrid':showGrid,'yaxis.showgrid':showGrid});
}
function screenshotGraph() {
  Plotly.downloadImage('graphDiv', {format:'png', filename:'numera-graph', width:1400, height:900});
}

/* ── MathGPT Solution ──────────────────────────────────── */
async function fetchSolution(latex) {
  if (solutionAbort) solutionAbort.abort();
  solutionAbort = new AbortController();
  resetSol();
  document.getElementById('solEmpty').style.display = 'none';
  document.getElementById('solLoading').style.display = 'flex';

  try {
    const resp = await fetch('/api/solution', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ latex }),
      signal: solutionAbort.signal
    });
    const data = await resp.json();
    document.getElementById('solLoading').style.display = 'none';

    if (data.error) {
      document.getElementById('solEmpty').style.display = 'flex';
      return;
    }
    renderSolution(data);

    // Dump to LaTeX console tab
    let dump = `% MathGPT — ${latex}\n\n`;
    if (data.latex_display) dump += `% Display\n$$${data.latex_display}$$\n\n`;
    (data.steps||[]).forEach((s,i) => {
      dump += `% Step ${i+1}: ${s.heading}\n$$${s.latex}$$\n`;
      if (s.explanation) dump += `% ${s.explanation}\n`;
      dump += '\n';
    });
    document.getElementById('latexOut').textContent = dump;

  } catch(e) {
    if (e.name === 'AbortError') return;
    document.getElementById('solLoading').style.display = 'none';
    document.getElementById('solEmpty').style.display = 'flex';
    log('err', 'Solution: ' + e);
  }
}

function renderSolution(data) {
  document.getElementById('solContent').style.display = 'flex';

  // Meta badges
  const meta = document.getElementById('solMeta');
  meta.innerHTML = '';
  if (data.function_type) { const b=mkBadge(data.function_type,'type'); meta.appendChild(b); }
  if (data.domain) {
    const b = mkBadge('', '');
    b.innerHTML = 'Domain: ';
    meta.appendChild(b);
    waitKatex(() => { const s=document.createElement('span'); b.appendChild(s); tryKatex(data.domain,s,false); });
  }
  if (data.range) {
    const b = mkBadge('','');
    b.innerHTML = 'Range: ';
    meta.appendChild(b);
    waitKatex(() => { const s=document.createElement('span'); b.appendChild(s); tryKatex(data.range,s,false); });
  }

  // Big equation
  const eq = document.getElementById('solEq');
  eq.innerHTML = '';
  if (data.latex_display) waitKatex(() => tryKatex(data.latex_display, eq, true));

  // Steps
  const list = document.getElementById('stepsList');
  list.innerHTML = '';
  (data.steps||[]).forEach((s, i) => {
    const card = document.createElement('div');
    card.className = 'step-card';
    card.style.animationDelay = `${i*70}ms`;
    card.innerHTML = `
      <div class="step-hd">
        <div class="step-num">${i+1}</div>
        <div class="step-title">${esc(s.heading||'')}</div>
      </div>
      <div class="step-body">
        <div class="step-latex" id="sl-${i}"></div>
        ${s.explanation ? `<div class="step-explain">${esc(s.explanation)}</div>` : ''}
      </div>`;
    list.appendChild(card);
    if (s.latex) waitKatex(() => { const el=document.getElementById(`sl-${i}`); if(el) tryKatex(s.latex,el,true); });
  });

  // Insight
  if (data.key_insight) {
    document.getElementById('insightBox').style.display = 'flex';
    document.getElementById('insightText').textContent = data.key_insight;
  }
  log('ok', `MathGPT: ${(data.steps||[]).length} steps · ${data.function_type||'solved'}`);
}

function mkBadge(text, cls) {
  const b = document.createElement('div');
  b.className = 'mbadge' + (cls ? ` ${cls}` : '');
  b.textContent = text;
  return b;
}

function resetSol() {
  ['solEmpty','solLoading','solContent','insightBox'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.style.display = 'none';
  });
  document.getElementById('solMeta').innerHTML = '';
  document.getElementById('solEq').innerHTML = '';
  document.getElementById('stepsList').innerHTML = '';
}

/* ═══════════════════════════════════════════════════════════
   PREMIUM AI REASONING TIMELINE
   ═══════════════════════════════════════════════════════════ */

const TL_STEPS_DATA = [
  { title:'Thinking through the problem',         desc:()=>'Understanding the mathematical request.'                        },
  { title:'Detecting equation type',              desc:()=>_tl.eqType+' Function'                                          },
  { title:'Running mathematical validation',      desc:()=>'Verifying syntax using SymPy.'                                  },
  { title:'Extracting graph characteristics',     desc:()=>'Finding intercepts, vertex, domain, range and extrema.'        },
  { title:'Selecting visualization strategy',     desc:()=>'Choosing the optimal Manim scene.'                             },
  { title:'Generating Manim animation',           desc:()=>'Writing optimized animation code.'                             },
  { title:'Running mathematical accuracy checks', desc:()=>'Ensuring graph matches symbolic solution.'                     },
  { title:'Rendering animation',                  desc:()=>'Rendering frames with GPU acceleration.'                       },
  { title:'Finalizing visualization',             desc:()=>'Preparing interactive playback.'                               },
];

const TL_PCTS   = [5,14,22,33,45,58,70,85,100];
const TL_ICONS  = [
  `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M12 2v3M12 19v3M4.22 4.22l2.12 2.12M17.66 17.66l2.12 2.12M2 12h3M19 12h3M4.22 19.78l2.12-2.12M17.66 6.34l2.12-2.12"/></svg>`,
  `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>`,
  `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><polyline points="9 12 11 14 15 10"/></svg>`,
  `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/><line x1="2" y1="20" x2="22" y2="20"/></svg>`,
  `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5M2 12l10 5 10-5"/></svg>`,
  `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg>`,
  `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 11-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>`,
  `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="6" width="20" height="12" rx="2"/><line x1="7" y1="6" x2="7" y2="18"/><line x1="17" y1="6" x2="17" y2="18"/><line x1="2" y1="12" x2="22" y2="12"/></svg>`,
  `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg>`,
];
const TL_CHECK = `<svg viewBox="0 0 24 24" fill="none" stroke="#22c55e" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>`;

const _tl = {
  current:-1, hidden:false, timers:[], etaTimer:null,
  particleRAF:null, eqType:'Polynomial',
};

function _tlDetectType() {
  const e = (currentParsed && currentParsed.expression) || '';
  if (/sin|cos|tan/.test(e))           return 'Trigonometric';
  if (/\*\*4/.test(e))                 return 'Quartic';
  if (/\*\*3/.test(e))                 return 'Cubic';
  if (/\*\*2/.test(e))                 return 'Quadratic';
  if (/exp|log/.test(e))               return 'Transcendental';
  if (/^[\d\s\*x\+\-\.\/]+$/.test(e)) return 'Linear';
  return 'Polynomial';
}

function _typeWriter(el, text) {
  if (!el || !text) return;
  el.textContent = '';
  let i = 0;
  const tick = () => { if (i < text.length) { el.textContent += text[i++]; setTimeout(tick, 17); } };
  tick();
}

function _tlBuild() {
  const c = document.getElementById('tlSteps');
  if (!c) return;
  c.innerHTML = '';
  TL_STEPS_DATA.forEach((s, i) => {
    const d = document.createElement('div');
    d.className = 'tl-step'; d.id = `tls-${i}`;
    d.innerHTML = `
      <div class="tl-step-dot" id="tls-dot-${i}">
        <span class="tl-step-icon">${TL_ICONS[i]}</span>
        <span class="tl-step-chk" style="display:none">${TL_CHECK}</span>
      </div>
      <div class="tl-step-body">
        <div class="tl-step-title">${s.title}</div>
        <div class="tl-step-desc" id="tls-desc-${i}"></div>
        <div class="tl-step-adots" id="tls-adots-${i}" style="display:none"><span></span><span></span><span></span></div>
      </div>`;
    c.appendChild(d);
  });
}

function _tlActivate(i) {
  if (_tl.hidden) return;
  _tl.current = i;
  const el = document.getElementById(`tls-${i}`);
  if (!el) return;
  el.classList.add('tl-active');
  el.scrollIntoView({ behavior:'smooth', block:'nearest' });
  _typeWriter(document.getElementById(`tls-desc-${i}`), TL_STEPS_DATA[i].desc());
  const ad = document.getElementById(`tls-adots-${i}`);
  if (ad) ad.style.display = 'flex';
  _tlSetPct(TL_PCTS[i]);
  // ETA
  const etaEl = document.getElementById('tlEta');
  if (!etaEl) return;
  if (i < 7) { etaEl.textContent = '~'+(7-i)*2+'s remaining'; }
  else if (i === 7) {
    etaEl.textContent = 'Rendering…';
    clearInterval(_tl.etaTimer);
    const t0 = Date.now();
    _tl.etaTimer = setInterval(() => {
      if (etaEl) etaEl.textContent = Math.round((Date.now()-t0)/1000)+'s elapsed';
    }, 1000);
  } else {
    clearInterval(_tl.etaTimer);
    etaEl.textContent = 'Almost done!';
  }
}

function _tlComplete(i) {
  if (_tl.hidden) return;
  const el = document.getElementById(`tls-${i}`);
  if (!el) return;
  el.classList.remove('tl-active'); el.classList.add('tl-done');
  const icon = el.querySelector('.tl-step-icon');
  const chk  = el.querySelector('.tl-step-chk');
  if (icon) icon.style.display = 'none';
  if (chk)  chk.style.display  = '';
  const ad = document.getElementById(`tls-adots-${i}`);
  if (ad) ad.style.display = 'none';
  const fill = document.getElementById('tlVlineFill');
  if (fill) fill.style.height = Math.min(100, ((i+1)/TL_STEPS_DATA.length)*100)+'%';
}

function _tlSetPct(p) {
  const el = document.getElementById('tlPct');
  if (el) el.textContent = p+'%';
  const f = document.getElementById('tlTopFill');
  if (f) f.style.width = p+'%';
}

function startReasoningTimeline() {
  _tl.current = -1; _tl.hidden = false;
  _tl.eqType  = _tlDetectType();
  _tl.timers.forEach(clearTimeout); _tl.timers = [];
  clearInterval(_tl.etaTimer);

  _tlBuild();
  _tlSetPct(0);
  const vf = document.getElementById('tlVlineFill');
  if (vf) vf.style.height = '0%';
  const tf = document.getElementById('tlTopFill');
  if (tf) tf.style.width = '0%';

  const ov = document.getElementById('thinkingOverlay');
  ov.style.opacity = '0';
  ov.style.display = 'flex';
  requestAnimationFrame(() => {
    ov.style.transition = 'opacity .35s ease';
    ov.style.opacity = '1';
    setTimeout(() => { ov.style.transition = ''; }, 380);
  });
  _tlStartParticles();

  const MS = 2100;
  // Steps 0-6 auto-play; step 7 activates and waits for SSE done
  for (let i = 0; i <= 6; i++) {
    _tl.timers.push(setTimeout(()=>_tlActivate(i),  i*MS));
    _tl.timers.push(setTimeout(()=>_tlComplete(i),  i*MS+MS-250));
  }
  _tl.timers.push(setTimeout(()=>_tlActivate(7), 7*MS));
}

function tlRenderDone(videoUrl) {
  clearInterval(_tl.etaTimer);
  _tl.timers.forEach(clearTimeout); _tl.timers = [];

  const cur = _tl.current;
  const QMS = 280; let d = 0;

  // Complete current + any skipped steps up to 7
  for (let i = Math.max(cur,0); i <= 7; i++) {
    if (i > cur) { setTimeout(()=>_tlActivate(i), d); d += QMS; }
    setTimeout(()=>_tlComplete(i), d); d += QMS;
  }
  // Step 8 — finalize
  setTimeout(()=>_tlActivate(8), d); d += 1100;
  setTimeout(()=>{
    _tlComplete(8); _tlSetPct(100);
    const etaEl = document.getElementById('tlEta');
    if (etaEl) etaEl.textContent = 'Complete!';
  }, d); d += 700;

  // Hide overlay then show video
  setTimeout(()=>_tlHide(), d);
  setTimeout(()=>{
    showVideo(videoUrl);
    document.getElementById('exportBtn').style.display = '';
    document.getElementById('videoMeta').textContent =
      `Manim CE · 480p · ${new Date().toLocaleTimeString()}`;
    log('ok','✓ 3B1B Manim explainer ready!');
    resetRenderUI();
  }, d + 450);
}

function tlRenderError() { _tlHide(); }

function _tlHide() {
  if (_tl.hidden) return;
  _tl.hidden = true;
  _tl.timers.forEach(clearTimeout);
  clearInterval(_tl.etaTimer);
  _tlStopParticles();
  const ov = document.getElementById('thinkingOverlay');
  ov.style.transition = 'opacity .4s ease';
  ov.style.opacity = '0';
  setTimeout(()=>{ ov.style.display='none'; ov.style.opacity=''; ov.style.transition=''; }, 430);
}

function _tlStartParticles() {
  const cv = document.getElementById('tlParticles');
  if (!cv) return;
  const ctx = cv.getContext('2d');
  cv.width = window.innerWidth; cv.height = window.innerHeight;
  const pts = Array.from({length:60},()=>({
    x:Math.random()*cv.width, y:Math.random()*cv.height,
    r:Math.random()*1.4+0.3,
    dx:(Math.random()-.5)*.3, dy:(Math.random()-.5)*.3,
    a:Math.random()*.4+0.05,
    c:Math.random()>.5?'79,140,255':'139,92,246',
  }));
  const tick = ()=>{
    if (_tl.hidden) return;
    ctx.clearRect(0,0,cv.width,cv.height);
    pts.forEach(p=>{
      p.x+=p.dx; p.y+=p.dy;
      if(p.x<0)p.x=cv.width; if(p.x>cv.width)p.x=0;
      if(p.y<0)p.y=cv.height; if(p.y>cv.height)p.y=0;
      ctx.beginPath(); ctx.arc(p.x,p.y,p.r,0,Math.PI*2);
      ctx.fillStyle=`rgba(${p.c},${p.a})`; ctx.fill();
    });
    _tl.particleRAF = requestAnimationFrame(tick);
  };
  _tl.particleRAF = requestAnimationFrame(tick);
}

function _tlStopParticles() {
  if (_tl.particleRAF) { cancelAnimationFrame(_tl.particleRAF); _tl.particleRAF=null; }
  const cv = document.getElementById('tlParticles');
  if (cv) cv.getContext('2d').clearRect(0,0,cv.width,cv.height);
}

/* ── Animation ─────────────────────────────────────────── */
async function generateAnimation() {
  if (!currentParsed) { log('warn','Enter an equation first.'); return; }

  const btn   = document.getElementById('genBtn');
  const inner = document.getElementById('genBtnInner');
  btn.disabled = true;
  inner.innerHTML = '<span class="spinner"></span>&nbsp;Generating…';

  document.getElementById('animEmpty').style.display    = 'none';
  document.getElementById('videoWrap').style.display    = 'none';
  document.getElementById('renderStatus').style.display = 'none';
  log('info', `Generating 3B1B explainer for: ${currentParsed.expression}`);

  // Show premium reasoning timeline immediately, API runs in parallel
  startReasoningTimeline();

  try {
    const resp = await fetch('/api/render', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ latex: currentLatex })
    });
    const { job_id, error } = await resp.json();
    if (error) throw new Error(error);
    currentJobId = job_id;
    listenToJob(job_id);
  } catch(e) {
    log('err','Render failed: '+e);
    tlRenderError();
    resetRenderUI();
  }
}

function listenToJob(jobId) {
  const es = new EventSource(`/api/stream/${jobId}`);
  es.onmessage = ev => handleEvent(JSON.parse(ev.data), es);
  es.onerror = () => { es.close(); resetRenderUI(); };
}

/* stage → which pill lights up */
const STAGE_PILL = {
  analyzing: 1, codegen: 2,
  rendering: 3, rendering_attempt_1: 3, rendering_attempt_2: 3,
  rendering_attempt_3: 3, healing: 3,
};

function handleEvent(data, es) {
  const { stage, message, video_url, progress } = data;

  if (message) {
    log(stage === 'error' ? 'err' : 'info', message);
    document.getElementById('rpLabel').textContent = message;
  }

  // Progress bar
  const progMap = {
    analyzing: 10, codegen: 35,
    rendering: 55, rendering_attempt_1: 62,
    rendering_attempt_2: 74, rendering_attempt_3: 86,
    healing: 68,
  };
  const pct = progress ?? progMap[stage];
  if (pct !== undefined) setProgress(pct);

  // Stage pills
  const pillIdx = STAGE_PILL[stage];
  if (pillIdx) {
    for (let i = 1; i <= 3; i++) {
      const el = document.getElementById(`rpStage${i}`);
      if (!el) continue;
      el.classList.remove('active', 'done');
      if (i < pillIdx)  el.classList.add('done');
      if (i === pillIdx) el.classList.add('active');
    }
  }

  if (stage === 'done' && video_url) {
    es.close();
    currentVideoUrl = video_url;
    // Drive the timeline to completion, then reveal video
    tlRenderDone(video_url);
  }
  if (stage === 'error') {
    es.close();
    tlRenderError();
    resetRenderUI();
    document.getElementById('animEmpty').style.display = 'flex';
    document.getElementById('renderStatus').style.display = 'none';
  }
  if (stage === 'end') es.close();
}

function setProgress(pct) {
  const f = document.getElementById('rpFill');
  f.className = 'rp-fill';
  f.style.width = pct + '%';
}

function showVideo(url) {
  const wrap = document.getElementById('videoWrap');
  const vid  = document.getElementById('videoEl');
  document.getElementById('animEmpty').style.display = 'none';
  document.getElementById('renderStatus').style.display = 'none';
  wrap.style.display = 'block';
  vid.src = url + '?t=' + Date.now();
  vid.load();
  vid.play().catch(() => {});
  // Scroll video into view
  setTimeout(() => wrap.scrollIntoView({ behavior:'smooth', block:'nearest' }), 150);
}

function resetRenderUI() {
  document.getElementById('genBtn').disabled = false;
  document.getElementById('genBtnInner').innerHTML = '▶  Generate 3B1B Animation';
}

function exportVideo() {
  const url = currentVideoUrl || (currentJobId ? `/api/video/${currentJobId}` : null);
  if (!url) { alert('Generate an animation first.'); return; }
  const a = document.createElement('a'); a.href=url; a.download='numera-3b1b.mp4'; a.click();
}

/* ── Video Slider ──────────────────────────────────────── */
function setupVideoSlider() {
  const vid    = document.getElementById('videoEl');
  const slider = document.getElementById('videoSlider');
  const tEl    = document.getElementById('vtime');
  const playBtn= document.getElementById('playBtn');

  vid.ontimeupdate = () => {
    if (!slider.matches(':active') && vid.duration) {
      slider.value = Math.round((vid.currentTime / vid.duration) * 1000);
    }
    tEl.textContent = `${fmt(vid.currentTime)} / ${fmt(vid.duration||0)}`;
  };

  slider.oninput = () => {
    if (vid.duration) vid.currentTime = (slider.value / 1000) * vid.duration;
  };

  vid.onplay  = () => { playBtn.textContent = '⏸'; };
  vid.onpause = () => { playBtn.textContent = '▶'; };
  vid.onended = () => { playBtn.textContent = '▶'; slider.value = 0; };
}

function togglePlay() {
  const vid = document.getElementById('videoEl');
  vid.paused ? vid.play() : vid.pause();
}

function fmt(s) {
  if (!isFinite(s)) return '0:00';
  const m = Math.floor(s/60), ss = Math.floor(s%60);
  return `${m}:${String(ss).padStart(2,'0')}`;
}

/* ── Theme ─────────────────────────────────────────────── */
function toggleTheme() {
  document.body.classList.toggle('light');
  const l = document.body.classList.contains('light');
  LAYOUT.xaxis.gridcolor = l ? '#e0e0e0' : '#1e1e1e';
  LAYOUT.yaxis.gridcolor = l ? '#e0e0e0' : '#1e1e1e';
  LAYOUT.xaxis.zerolinecolor = l ? '#aaa' : '#444';
  LAYOUT.yaxis.zerolinecolor = l ? '#aaa' : '#444';
  LAYOUT.font.color = l ? '#555' : '#666';
  Plotly.relayout('graphDiv', LAYOUT);
}

/* ── Console ─────────────────────────────────────────────── */
function toggleConsole() {
  consoleOpen = !consoleOpen;
  document.getElementById('consoleBar').classList.toggle('collapsed', !consoleOpen);
  document.getElementById('consoleArrow').textContent = consoleOpen ? '▲' : '▼';
}
function switchTab(name, e) {
  e?.stopPropagation();
  document.querySelectorAll('.ctab').forEach(t => t.classList.remove('active'));
  e?.target?.classList.add('active');
  document.querySelectorAll('.cpane').forEach(p => p.classList.remove('active'));
  document.getElementById(`pane-${name}`)?.classList.add('active');
}
function clearConsole(e) {
  e?.stopPropagation();
  document.getElementById('logOut').innerHTML = '';
  document.getElementById('errOut').innerHTML = '';
  document.getElementById('latexOut').textContent = '% LaTeX solution appears here after MathGPT analysis';
}
function log(type, msg) {
  if (!consoleOpen) {
    consoleOpen = true;
    document.getElementById('consoleBar').classList.remove('collapsed');
    document.getElementById('consoleArrow').textContent = '▲';
  }
  const out  = document.getElementById('logOut');
  const line = document.createElement('div');
  line.className = 'll ' + (type==='ok'?'ok':type==='warn'?'warn':type==='err'?'err':'info');
  line.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
  out.appendChild(line);
  out.scrollTop = out.scrollHeight;
  if (type === 'err') {
    const el = document.getElementById('errOut');
    const l2 = document.createElement('div');
    l2.className = 'll err'; l2.textContent = line.textContent;
    el.appendChild(l2); el.scrollTop = el.scrollHeight;
  }
}

/* ── KaTeX ─────────────────────────────────────────────── */
function waitKatex(fn) {
  if (window.katex) { fn(); return; }
  const p = setInterval(() => { if (window.katex) { clearInterval(p); fn(); } }, 50);
}
function tryKatex(latex, el, display) {
  try { katex.render(latex, el, {throwOnError:false, displayMode:display, output:'html'}); return true; }
  catch { el.textContent = latex; return false; }
}

/* ── Helpers ─────────────────────────────────────────────── */
function esc(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
