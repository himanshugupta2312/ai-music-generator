// 3D scene: a glowing audio-reactive orb, a particle halo, orbit rings and a wave floor, with bloom.
// Three.js r160 is vendored in ./vendor so the UI works offline.
import * as THREE from 'three';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';

const NOISE = /* glsl */ `
float hash(vec3 p){ p = fract(p*0.3183099 + .1); p *= 17.0; return fract(p.x*p.y*p.z*(p.x+p.y+p.z)); }
float noise(vec3 x){
  vec3 i = floor(x), f = fract(x); f = f*f*(3.0-2.0*f);
  return mix(mix(mix(hash(i+vec3(0,0,0)), hash(i+vec3(1,0,0)), f.x),
                 mix(hash(i+vec3(0,1,0)), hash(i+vec3(1,1,0)), f.x), f.y),
             mix(mix(hash(i+vec3(0,0,1)), hash(i+vec3(1,0,1)), f.x),
                 mix(hash(i+vec3(0,1,1)), hash(i+vec3(1,1,1)), f.x), f.y), f.z);
}`;

const ORB_VERT = /* glsl */ `
uniform float uTime, uBass, uMid, uTreble, uImpulse;
varying vec3 vN; varying vec3 vV; varying float vD;
${NOISE}
void main(){
  vec3 p = position;
  float n  = noise(p*1.5 + uTime*0.35);
  float n2 = noise(p*4.0 - uTime*0.9);
  float d = (n-0.5)*(0.30 + uBass*0.7 + uImpulse*0.7) + (n2-0.5)*(0.06 + uTreble*0.35 + uMid*0.12);
  p += normal * d;
  vD = d;
  vN = normalize(normalMatrix * normal);
  vec4 mv = modelViewMatrix * vec4(p, 1.0);
  vV = normalize(-mv.xyz);
  gl_Position = projectionMatrix * mv;
}`;

const ORB_FRAG = /* glsl */ `
uniform vec3 uC1, uC2; uniform float uTime, uBass;
varying vec3 vN; varying vec3 vV; varying float vD;
void main(){
  vec3 N = normalize(vN); vec3 V = normalize(vV);
  float fres = pow(1.0 - max(dot(N, V), 0.0), 2.4);
  float band = 0.5 + 0.5*sin(vD*16.0 + uTime*1.3);
  vec3 col = mix(uC1, uC2, clamp(vD*2.4 + 0.5 + 0.25*sin(uTime*0.5), 0.0, 1.0));
  col = col*(0.26 + 0.45*band) + fres * mix(uC2, uC1, 0.5) * (0.9 + uBass*0.5);
  gl_FragColor = vec4(col, 1.0);
}`;

const HALO_VERT = /* glsl */ `
attribute float aSeed;
uniform float uTime, uSpeed, uTreble, uImpulse, uPx;
varying float vMix; varying float vA;
void main(){
  float r   = 2.3 + aSeed*2.7;
  float ang = aSeed*43.98 + uTime*uSpeed*(0.12 + aSeed*0.28);
  float y   = (fract(aSeed*13.7) - 0.5)*3.4 + sin(uTime*0.6 + aSeed*40.0)*0.15;
  vec3 p = vec3(cos(ang)*r, y, sin(ang)*r);
  p *= 1.0 + uImpulse*0.9*(0.3 + aSeed);
  vec4 mv = modelViewMatrix * vec4(p, 1.0);
  gl_PointSize = (1.6 + uTreble*4.5 + aSeed*2.2) * uPx * (9.0 / -mv.z);
  vMix = aSeed; vA = 0.35 + 0.65*fract(aSeed*91.7);
  gl_Position = projectionMatrix * mv;
}`;

const HALO_FRAG = /* glsl */ `
uniform vec3 uC1, uC2;
varying float vMix; varying float vA;
void main(){
  float d = length(gl_PointCoord - 0.5);
  float a = smoothstep(0.5, 0.0, d);
  gl_FragColor = vec4(mix(uC1, uC2, vMix) * 1.2, a * vA * 0.85);
}`;

const FLOOR_VERT = /* glsl */ `
uniform float uTime, uBass, uImpulse, uPx;
varying float vH; varying float vF;
void main(){
  vec3 p = position;
  float r = length(p.xz);
  float h = sin(p.x*0.9 + uTime*0.7)*0.12 + sin(p.z*1.1 - uTime*0.9)*0.12;
  h += sin(r*2.2 - uTime*2.6) * exp(-r*0.2) * (0.14 + uBass*1.2 + uImpulse*0.9);
  p.y = h; vH = h; vF = 1.0 - smoothstep(4.0, 9.0, r);
  vec4 mv = modelViewMatrix * vec4(p, 1.0);
  gl_PointSize = 2.4 * uPx * (8.0 / -mv.z);
  gl_Position = projectionMatrix * mv;
}`;

const FLOOR_FRAG = /* glsl */ `
uniform vec3 uC1, uC2;
varying float vH; varying float vF;
void main(){
  float d = length(gl_PointCoord - 0.5);
  float a = smoothstep(0.5, 0.1, d) * vF;
  vec3 col = mix(uC1, uC2, clamp(vH*2.2 + 0.5, 0.0, 1.0)) * 1.1;
  gl_FragColor = vec4(col, a);
}`;

export function createScene(canvas) {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance' });
  const DPR = Math.min(window.devicePixelRatio || 1, 1.75);
  renderer.setPixelRatio(DPR);
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x07060f);
  scene.fog = new THREE.FogExp2(0x07060f, 0.045);
  const camera = new THREE.PerspectiveCamera(48, 1, 0.1, 100);
  camera.position.set(0, 0.5, 7);

  const c1 = new THREE.Color('#8b5cf6'), c2 = new THREE.Color('#ec4899');
  const t1 = c1.clone(), t2 = c2.clone();
  const shared = { uC1: { value: c1 }, uC2: { value: c2 }, uTime: { value: 0 },
    uBass: { value: 0 }, uMid: { value: 0 }, uTreble: { value: 0 }, uImpulse: { value: 0 },
    uSpeed: { value: 0.6 }, uPx: { value: DPR } };

  const world = new THREE.Group();
  scene.add(world);

  // orb
  const orb = new THREE.Mesh(
    new THREE.IcosahedronGeometry(1.2, 6),
    new THREE.ShaderMaterial({ uniforms: shared, vertexShader: ORB_VERT, fragmentShader: ORB_FRAG }));
  world.add(orb);

  // particle halo
  const N = 5200, seeds = new Float32Array(N), zeros = new Float32Array(N * 3);
  for (let i = 0; i < N; i++) seeds[i] = Math.random();
  const haloGeo = new THREE.BufferGeometry();
  haloGeo.setAttribute('position', new THREE.BufferAttribute(zeros, 3));
  haloGeo.setAttribute('aSeed', new THREE.BufferAttribute(seeds, 1));
  const halo = new THREE.Points(haloGeo, new THREE.ShaderMaterial({
    uniforms: shared, vertexShader: HALO_VERT, fragmentShader: HALO_FRAG,
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending }));
  halo.frustumCulled = false;
  world.add(halo);

  // wave floor
  const G = 110, S = 0.135, fp = new Float32Array(G * G * 3);
  for (let i = 0; i < G; i++) for (let j = 0; j < G; j++) {
    const k = (i * G + j) * 3; fp[k] = (i - G / 2) * S; fp[k + 1] = 0; fp[k + 2] = (j - G / 2) * S;
  }
  const floorGeo = new THREE.BufferGeometry();
  floorGeo.setAttribute('position', new THREE.BufferAttribute(fp, 3));
  const floor = new THREE.Points(floorGeo, new THREE.ShaderMaterial({
    uniforms: shared, vertexShader: FLOOR_VERT, fragmentShader: FLOOR_FRAG,
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending }));
  floor.position.y = -2.3; floor.frustumCulled = false;
  world.add(floor);

  // orbit rings
  const ringMats = [], rings = [];
  [[2.0, 0.5, 0.2], [2.5, -0.9, 0.5]].forEach(([r, rx, rz]) => {
    const m = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.55 });
    const ring = new THREE.Mesh(new THREE.TorusGeometry(r, 0.007, 8, 220), m);
    ring.rotation.set(rx, 0, rz); world.add(ring); rings.push(ring); ringMats.push(m);
  });

  // post-processing: bloom for the glow
  const composer = new EffectComposer(renderer);
  composer.addPass(new RenderPass(scene, camera));
  const bloom = new UnrealBloomPass(new THREE.Vector2(256, 256), 0.6, 0.6, 0.22);
  composer.addPass(bloom);
  composer.addPass(new OutputPass());

  const state = { mode: 'idle', bass: 0, mid: 0, treble: 0, impulse: 0, reduced: false,
    lb: 0, lm: 0, lt: 0, mx: 0, my: 0, offset: 0, offsetTarget: 0 };

  function resize() {
    const w = canvas.clientWidth || window.innerWidth, h = canvas.clientHeight || window.innerHeight;
    renderer.setSize(w, h, false); composer.setSize(w, h);
    camera.aspect = w / h; camera.updateProjectionMatrix();
    state.offsetTarget = w / h > 1.25 ? 2.1 : 0;   // push the orb to the right on wide screens
  }
  window.addEventListener('resize', resize); resize();
  window.addEventListener('pointermove', (e) => {
    state.mx = (e.clientX / window.innerWidth) * 2 - 1; state.my = (e.clientY / window.innerHeight) * 2 - 1;
  });

  const clock = new THREE.Clock();
  renderer.setAnimationLoop(() => {
    const dt = Math.min(clock.getDelta(), 0.05), t = clock.elapsedTime;
    const calm = state.reduced ? 0.35 : 1;
    shared.uTime.value = t * calm;

    // target levels per mode
    let b = state.bass, m = state.mid, tr = state.treble, speed = 0.6;
    if (state.mode === 'idle') { b = 0.10 + 0.07 * Math.sin(t * 1.1); m = 0.08; tr = 0.06; }
    if (state.mode === 'working') { b = 0.28 + 0.22 * Math.sin(t * 5.5); m = 0.25; tr = 0.2; speed = 2.6; }
    if (state.mode === 'playing') speed = 1.1;
    const k = 1 - Math.pow(0.001, dt);              // smoothing
    state.lb += (b - state.lb) * k; state.lm += (m - state.lm) * k; state.lt += (tr - state.lt) * k;
    state.impulse *= Math.pow(0.04, dt);
    shared.uBass.value = state.lb * calm; shared.uMid.value = state.lm * calm; shared.uTreble.value = state.lt * calm;
    shared.uImpulse.value = state.impulse * calm;
    shared.uSpeed.value += (speed * calm - shared.uSpeed.value) * 0.05;

    // palette easing
    c1.lerp(t1, 0.05); c2.lerp(t2, 0.05);
    ringMats[0].color.copy(c1).multiplyScalar(1.15); ringMats[1].color.copy(c2).multiplyScalar(1.15);
    rings[0].rotation.z += dt * 0.25 * calm; rings[1].rotation.x -= dt * 0.18 * calm;
    orb.rotation.y += dt * 0.12 * calm;
    halo.rotation.y += dt * 0.02 * calm;

    // camera parallax + layout offset
    state.offset += (state.offsetTarget - state.offset) * 0.06;
    world.position.x = state.offset;
    const px = state.reduced ? 0 : state.mx * 0.9, py = state.reduced ? 0 : -state.my * 0.4;
    camera.position.x += (px - camera.position.x) * 0.04;
    camera.position.y += (0.5 + py - camera.position.y) * 0.04;
    camera.lookAt(state.offset * 0.35, 0, 0);
    bloom.strength = 0.42 + state.lb * 0.25 + state.impulse * 0.4;
    composer.render();
  });

  return {
    setPalette(a, b) { t1.set(a); t2.set(b); },
    setMode(mode) { state.mode = mode; },
    setLevels(b, m, t) { state.bass = b; state.mid = m; state.treble = t; },
    burst() { state.impulse = 1; },
    setReduced(flag) { state.reduced = !!flag; },
  };
}
