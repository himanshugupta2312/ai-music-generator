// aimusic web UI: talks to the local server's JSON API and drives the 3D scene.
const $ = (s) => document.querySelector(s);
const root = document.documentElement;
const fmt = (s) => { s = Math.max(0, Math.round(s || 0)); return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`; };
const FUN = [
  '🎹 Warming up the piano…', '🥁 Convincing the drummer to keep time…', '🎸 Tuning the guitars…',
  '🎛️ Twiddling the knobs…', '🎻 Politely asking the violins to stop squeaking…', '☕ Brewing coffee for the band…',
  '✨ Sprinkling a little reverb…', '🎧 Still cooking, good music takes a moment…',
];
const RING = 326.7;
const DEFAULT_PALETTE = ['#8b5cf6', '#ec4899'];

let info = null, scene = null, vibesById = {}, selectedVibe = null;
let job = null, pollTimer = null, previewTimer = null;
let actx = null, analyser = null, freq = null;
const audio = $('#audio');

async function api(path, body) {
  const opts = body === undefined ? {} : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) };
  const r = await fetch(path, opts);
  let data = {};
  try { data = await r.json(); } catch { /* non-JSON error body */ }
  if (!r.ok) throw new Error(data.error || `Request failed (${r.status})`);
  return data;
}

let toastTimer = null;
function toast(msg) {
  const t = $('#toast'); t.textContent = msg; t.classList.add('show');
  clearTimeout(toastTimer); toastTimer = setTimeout(() => t.classList.remove('show'), 5200);
}

function setPalette(a, b) {
  root.style.setProperty('--c1', a); root.style.setProperty('--c2', b);
  if (scene) scene.setPalette(a, b);
}
const PALETTE_RULES = [
  [/punk|rock|workout/, 'workout'], [/jazz/, 'jazz'], [/cinematic|epic|trailer/, 'trailer'],
  [/sleep|lullab/, 'sleep'], [/lofi|lo-fi|hip hop|beats/, 'lofi'], [/forest|nature/, 'forest'],
  [/rain|study|calm|piano|focus|soothing/, 'study'],
];
function paletteForText(text) {
  const t = text.toLowerCase();
  for (const [re, id] of PALETTE_RULES) if (re.test(t) && vibesById[id]) return [vibesById[id].c1, vibesById[id].c2];
  return DEFAULT_PALETTE;
}

// ---- vibes (3D tilt cards) ------------------------------------------------
function renderVibes() {
  const box = $('#vibes');
  for (const v of info.vibes) {
    vibesById[v.id] = v;
    const b = document.createElement('button');
    b.type = 'button'; b.className = 'vibe'; b.dataset.id = v.id;
    b.style.setProperty('--a', v.c1); b.style.setProperty('--b', v.c2);
    const mk = (cls, text) => { const s = document.createElement('span'); s.className = cls; s.textContent = text; return s; };
    const sheen = document.createElement('i'); sheen.className = 'sheen';
    b.append(sheen, mk('emoji', v.emoji), mk('name', v.name), mk('sub', v.sub));
    b.addEventListener('pointermove', (e) => {
      const r = b.getBoundingClientRect(), x = (e.clientX - r.left) / r.width, y = (e.clientY - r.top) / r.height;
      b.style.setProperty('--ry', `${(x - 0.5) * 18}deg`); b.style.setProperty('--rx', `${(0.5 - y) * 18}deg`);
      b.style.setProperty('--mx', `${x * 100}%`); b.style.setProperty('--my', `${y * 100}%`);
    });
    b.addEventListener('pointerleave', () => { b.style.setProperty('--rx', '0deg'); b.style.setProperty('--ry', '0deg'); });
    b.addEventListener('click', () => pickVibe(v));
    box.append(b);
  }
}
function markVibe(id) {
  selectedVibe = id;
  document.querySelectorAll('.vibe').forEach((b) => b.classList.toggle('active', b.dataset.id === id));
}
function pickVibe(v) {
  $('#prompt').value = v.prompt; markVibe(v.id); setPalette(v.c1, v.c2);
  if (scene) scene.burst();
  schedulePreview(0);
}

// ---- backends --------------------------------------------------------------
function renderBackends() {
  const box = $('#backends');
  for (const b of info.backends) {
    const label = document.createElement('label'); label.className = 'backend';
    const input = document.createElement('input');
    input.type = 'radio'; input.name = 'backend'; input.value = b.id; input.disabled = !b.available;
    input.checked = b.id === info.default_backend;
    input.addEventListener('change', () => schedulePreview(0));
    const span = document.createElement('span');
    const name = document.createElement('b'); name.textContent = b.name;
    const hint = document.createElement('small'); hint.textContent = b.available ? b.hint : `${b.hint} (not installed)`;
    span.append(name, hint); label.append(input, span); box.append(label);
  }
}
const backend = () => (document.querySelector('input[name=backend]:checked') || {}).value || 'synth';

// ---- live preview ("how I'll read it") -------------------------------------
const num = (el) => (el.value === '' ? null : Number(el.value));
function schedulePreview(delay = 250) { clearTimeout(previewTimer); previewTimer = setTimeout(loadPreview, delay); }
async function loadPreview() {
  const box = $('#read'); box.replaceChildren();
  const prompt = $('#prompt').value;
  if (!prompt.trim()) { const h = document.createElement('span'); h.className = 'hint'; h.textContent = "Pick a vibe or type something, and I'll show what I understood."; box.append(h); return; }
  try {
    const out = await api('/api/preview', { prompt, backend: backend(), duration: num($('#duration')) || null,
      tuning: num($('#tuning')), ambience: $('#ambience').checked, level: num($('#level')) });
    const pill = (label, value, warn) => {
      const p = document.createElement('span'); p.className = warn ? 'pill warn' : 'pill';
      if (label) { const b = document.createElement('b'); b.textContent = label; p.append(b); }
      p.append(document.createTextNode(value)); box.append(p);
    };
    if (out.error) pill('', `⚠️ ${out.error}`, true);
    out.rows.forEach((r) => pill(r.label, r.value));
    out.warnings.forEach((w) => pill('', `⚠️ ${w}`, true));
  } catch (e) { /* preview is best-effort */ }
}

// ---- generation ------------------------------------------------------------
function collect(newSeed) {
  return { prompt: $('#prompt').value, backend: backend(), duration: num($('#duration')) || null,
    seed: newSeed ? null : num($('#seed')), tuning: num($('#tuning')), ambience: $('#ambience').checked,
    level: num($('#level')), lyrics: $('#lyrics').value };
}
function setBusy(busy) {
  $('#go').disabled = busy; $('#again').disabled = busy; $('#stop').disabled = !busy;
  $('#working').hidden = !busy;
  if (scene) scene.setMode(busy ? 'working' : (!audio.paused ? 'playing' : 'idle'));
}
function setStatus(text, err) { const s = $('#status'); s.textContent = text; s.classList.toggle('err', !!err); }

async function generate(again) {
  if (job) return;
  if (!$('#prompt').value.trim()) { toast('Pick a vibe or describe the music first.'); return; }
  try {
    audio.pause();
    const r = await api('/api/generate', collect(again));
    job = { id: r.job, backend: backend() };
    setBusy(true); setStatus(FUN[0]);
    $('#ring').style.strokeDashoffset = RING;
    pollTimer = setInterval(pollJob, 700); pollJob();
  } catch (e) { toast(e.message); }
}

async function pollJob() {
  if (!job) return;
  let snap;
  try { snap = await api(`/api/jobs/${job.id}`); } catch (e) { return; }
  if (snap.state === 'running') {
    $('#elapsed').textContent = fmt(snap.elapsed);
    $('#est').textContent = `~${fmt(snap.estimate)} est.`;
    const frac = Math.min(snap.elapsed / Math.max(snap.estimate, 1), 0.97);
    $('#ring').style.strokeDashoffset = RING * (1 - frac);
    setStatus(FUN[Math.floor(snap.elapsed / 6) % FUN.length]);
    return;
  }
  clearInterval(pollTimer); pollTimer = null; job = null; setBusy(false);
  if (snap.state === 'done') {
    $('#ring').style.strokeDashoffset = 0;
    setStatus(`🎉 Ta-da! Done in ${fmt(snap.result.took_s)}. Not quite right? Try Another take.`);
    if (scene) scene.burst();
    loadTrack(snap.result, true); loadHistory();
  } else if (snap.state === 'cancelled') { setStatus('⏹ Stopped. Nothing was saved.');
  } else { setStatus(snap.error || 'Something went wrong.', true); toast(snap.error || 'Something went wrong.'); }
}

async function stop() {
  if (!job) return;
  try { await api(`/api/jobs/${job.id}/cancel`, {}); setStatus('⏹ Stopping…'); } catch (e) { toast(e.message); }
}

// ---- player + audio-reactive visuals ---------------------------------------
function ensureGraph() {
  if (actx) { if (actx.state === 'suspended') actx.resume(); return; }
  try {
    actx = new (window.AudioContext || window.webkitAudioContext)();
    const src = actx.createMediaElementSource(audio);
    analyser = actx.createAnalyser(); analyser.fftSize = 512; analyser.smoothingTimeConstant = 0.78;
    src.connect(analyser); analyser.connect(actx.destination);
    freq = new Uint8Array(analyser.frequencyBinCount);
  } catch (e) { actx = null; analyser = null; }
}
const avg = (a, lo, hi) => { let s = 0; for (let i = lo; i < hi; i++) s += a[i]; return s / ((hi - lo) * 255); };
// Each band is compared with its own slow moving average, so steady loudness sits at a calm
// baseline and beats/changes make the visuals jump (instead of everything being maxed out).
const slow = [0, 0, 0];
function reactive(i, value) {
  slow[i] += (value - slow[i]) * 0.015;
  return Math.max(0, Math.min(1, 0.22 + (value - slow[i]) * 5 + value * 0.25));
}
function frame() {
  if (scene && analyser && !audio.paused) {
    analyser.getByteFrequencyData(freq);
    scene.setLevels(reactive(0, avg(freq, 0, 4)), reactive(1, avg(freq, 4, 32)), reactive(2, avg(freq, 32, 120) * 1.6));
  }
  requestAnimationFrame(frame);
}
function loadTrack(result, autoplay) {
  $('#player').hidden = false;
  $('#title').textContent = result.label;
  audio.src = result.audio_url;
  const dl = $('#download'); dl.href = result.audio_url; dl.download = result.name;
  document.querySelectorAll('#history button').forEach((b) => b.classList.toggle('on', b.dataset.url === result.audio_url));
  if (result.settings && result.settings.style_prompt && !selectedVibe) { const [a, b] = paletteForText(result.label); setPalette(a, b); }
  if (autoplay) { ensureGraph(); audio.play().catch(() => { /* needs a click; the play button is right there */ }); }
}
function togglePlay() {
  if (!audio.src) return;
  ensureGraph();
  if (audio.paused) audio.play().catch(() => {}); else audio.pause();
}
audio.addEventListener('play', () => { $('#play').textContent = '⏸'; $('#play').setAttribute('aria-label', 'Pause'); if (scene && !job) scene.setMode('playing'); });
const stopped = () => { $('#play').textContent = '▶'; $('#play').setAttribute('aria-label', 'Play'); if (scene && !job) { scene.setMode('idle'); scene.setLevels(0, 0, 0); } };
audio.addEventListener('pause', stopped); audio.addEventListener('ended', stopped);
audio.addEventListener('timeupdate', () => {
  const d = audio.duration || 0;
  $('#time').textContent = `${fmt(audio.currentTime)} / ${fmt(d)}`;
  if (d && document.activeElement !== $('#seek')) $('#seek').value = String(Math.round((audio.currentTime / d) * 1000));
});
$('#seek').addEventListener('input', (e) => { if (audio.duration) audio.currentTime = (Number(e.target.value) / 1000) * audio.duration; });
$('#play').addEventListener('click', togglePlay);
document.addEventListener('keydown', (e) => {
  if (e.code !== 'Space' || $('#player').hidden) return;
  if (/^(INPUT|TEXTAREA|SELECT|BUTTON|SUMMARY|A)$/.test((e.target.tagName || ''))) return;
  e.preventDefault(); togglePlay();
});

// ---- history ----------------------------------------------------------------
async function loadHistory() {
  try {
    const { tracks } = await api('/api/history');
    const ol = $('#history'); ol.replaceChildren();
    $('#historybox').hidden = tracks.length === 0;
    tracks.forEach((t) => {
      const li = document.createElement('li'), b = document.createElement('button');
      b.type = 'button'; b.dataset.url = t.audio_url; b.textContent = t.label;
      if (audio.src.endsWith(t.audio_url)) b.classList.add('on');
      b.addEventListener('click', () => { loadTrack(t, true); setStatus(`Replaying: ${t.label}`); });
      li.append(b); ol.append(li);
    });
  } catch (e) { /* ignore */ }
}

// ---- boot ---------------------------------------------------------------------
async function boot() {
  try { info = await api('/api/info'); } catch (e) { setStatus(`Could not reach the aimusic server: ${e.message}`, true); return; }
  renderVibes(); renderBackends();
  try {
    const mod = await import('./scene.js');
    scene = mod.createScene($('#scene'));
    const rm = window.matchMedia('(prefers-reduced-motion: reduce)');
    scene.setReduced(rm.matches); rm.addEventListener('change', () => scene.setReduced(rm.matches));
    scene.setPalette(...DEFAULT_PALETTE);
  } catch (e) { document.body.classList.add('no-webgl'); console.warn('3D scene unavailable:', e); }
  requestAnimationFrame(frame);
  $('#prompt').addEventListener('input', () => {
    markVibe(null); const [a, b] = paletteForText($('#prompt').value); setPalette(a, b); schedulePreview();
  });
  ['#duration', '#tuning', '#ambience', '#level'].forEach((s) => $(s).addEventListener('input', () => schedulePreview()));
  $('#level').addEventListener('input', () => { $('#levelout').textContent = Number($('#level').value).toFixed(2); });
  $('#surprise').addEventListener('click', async () => {
    try { const r = await api('/api/surprise', {}); $('#prompt').value = r.prompt; markVibe(null);
      const [a, b] = paletteForText(r.prompt); setPalette(a, b); if (scene) scene.burst(); schedulePreview(0); } catch (e) { toast(e.message); }
  });
  $('#dice').addEventListener('click', () => { $('#seed').value = String(1 + Math.floor(Math.random() * 99999)); });
  $('#go').addEventListener('click', () => generate(false));
  $('#again').addEventListener('click', () => generate(true));
  $('#stop').addEventListener('click', stop);
  setBusy(false); loadPreview(); loadHistory();
}
boot();
