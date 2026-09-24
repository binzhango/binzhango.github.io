import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { vertex, frontFragment, edgeFragment } from './card-shaders.js';

// The homepage owns its canvas and pointer events. No iframe or drag gesture.
const host = document.querySelector('[data-home-card]');
if (host) mountCard(host);

async function mountCard(host) {
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  let targetX = -.035, targetY = -.15, hovering = false, visible = true;
  let renderer, root, uniforms, frame = 0, disposed = false;
  const textures = [], materials = [];
  const applyPointer = (event) => {
    if (reduced.matches || event.pointerType === 'touch') return;
    const rect = host.getBoundingClientRect();
    targetX = THREE.MathUtils.clamp(((event.clientY - rect.top) / rect.height - .5) * .7, -.35, .35);
    targetY = THREE.MathUtils.clamp(((event.clientX - rect.left) / rect.width - .5) * 1.2, -.6, .6);
    hovering = true;
    host.style.setProperty('--card-x', `${-targetX}rad`);
    host.style.setProperty('--card-y', `${targetY}rad`);
    host.style.setProperty('--shine-angle', `${115 + targetY * 70}deg`);
  };
  const resetPointer = () => {
    hovering = false;
    targetX = -.035; targetY = -.15;
    host.style.removeProperty('--card-x');
    host.style.removeProperty('--card-y');
    host.style.removeProperty('--shine-angle');
  };
  host.addEventListener('pointermove', applyPointer);
  host.addEventListener('pointerleave', resetPointer);
  reduced.addEventListener('change', resetPointer);
  const visibility = new IntersectionObserver(([entry]) => { visible = entry.isIntersecting; });
  visibility.observe(host);
  const scene = new THREE.Scene();
  const camera = new THREE.OrthographicCamera(-4, 4, 5.45, -5.45, .1, 100);
  camera.position.set(0, 0, 20);
  const inverse = new THREE.Matrix4();
  const resize = () => {
    if (!renderer) return;
    const { width, height } = host.getBoundingClientRect();
    const aspect = width / height;
    const halfHeight = Math.max(5.45, 3.05 / aspect);
    camera.left = -halfHeight * aspect; camera.right = halfHeight * aspect;
    camera.top = halfHeight; camera.bottom = -halfHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height, false);
  };
  const sizeObserver = new ResizeObserver(resize);
  sizeObserver.observe(host);
  const cleanup = () => {
    disposed = true;
    cancelAnimationFrame(frame);
    visibility.disconnect(); sizeObserver.disconnect();
    host.removeEventListener('pointermove', applyPointer);
    host.removeEventListener('pointerleave', resetPointer);
    reduced.removeEventListener('change', resetPointer);
    root?.traverse((object) => object.geometry?.dispose());
    textures.forEach((texture) => texture.dispose());
    materials.forEach((material) => material.dispose());
    renderer?.dispose();
  };
  document.addEventListener('astro:before-swap', cleanup, { once: true });

  try {
    const base = new URL('/card-studio/', location.origin);
    const response = await fetch(new URL('card-config.json', base));
    if (!response.ok) throw new Error('Card configuration unavailable');
    const config = await response.json();
    const p = config.parameters;
    const loader = new THREE.TextureLoader();
    const load = async (name) => {
      const texture = config.assets[name]
        ? await loader.loadAsync(new URL(config.assets[name], base).href)
        : new THREE.DataTexture(new Uint8Array([0, 0, 0, 0]), 1, 1);
      texture.colorSpace = THREE.NoColorSpace;
      texture.needsUpdate = true;
      textures.push(texture);
      return { value: texture };
    };
    const [tSubject, tBackground, tText, tLine, tEffects, gltf] = await Promise.all([
      load('subject'), load('background'), load('text'), load('lineart'), load('effects'),
      new GLTFLoader().loadAsync(new URL(config.assets.model, base).href),
    ]);
    if (disposed) { textures.forEach(t => t.dispose()); return; }
    renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true, powerPreference: 'default' });
    renderer.setClearColor(0x000000, 0);
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.domElement.setAttribute('aria-hidden', 'true');
    uniforms = {
      tSubject, tBackground, tText, tLine, tEffects,
      uTime: { value: 0 }, uView: { value: new THREE.Vector3(0, 0, 1) },
      uFit: { value: new THREE.Vector2(...(config.artworkFit || [1, 1])) },
      uFoil: { value: p.foil }, uScale: { value: p.subjectScale },
      uDepth: { value: p.subjectDepth }, uBgDepth: { value: p.backgroundDepth },
      uFxDepth: { value: p.effectsDepth }, uHasFx: { value: config.assets.effects ? 1 : 0 },
      uHasLine: { value: config.assets.lineart ? 1 : 0 }, uRelief: { value: 0 },
      uFinish: { value: { pearl: 0, silver: 1, original: 2, gold: 3 }[config.appearance.finish] ?? 0 },
      uSafeScale: { value: config.safeArea?.scale ?? 1 },
      uSafeOffset: { value: new THREE.Vector2(config.safeArea?.offset?.[0] ?? 0, 1 - (config.safeArea?.scale ?? 1) - (config.safeArea?.offset?.[1] ?? 0)) },
    };
    const material = (fragment) => new THREE.ShaderMaterial({ uniforms, vertexShader: vertex, fragmentShader: fragment });
    const front = material(frontFragment), edge = material(edgeFragment);
    materials.push(front, edge);
    root = new THREE.Group();
    root.add(gltf.scene); scene.add(root);
    root.traverse((object) => {
      if (!object.isMesh) return;
      const role = object.material.name;
      object.material.dispose();
      object.material = role === 'web_front' ? front : edge;
      if (['web_subject', 'web_text', 'web_effects'].includes(role)) object.visible = false;
    });
    resize();
    renderer.compile(scene, camera);
    if (renderer.info.programs.some(program => program.diagnostics && !program.diagnostics.runnable)) throw new Error('Card shader unavailable');
    host.append(renderer.domElement);
    root.rotation.set(targetX, targetY, 0);
    let last = 0;
    const animate = (now) => {
      if (disposed) return;
      frame = requestAnimationFrame(animate);
      const dt = Math.min((now - last) / 1000, .06); last = now;
      if (!visible || document.hidden) return;
      const ease = reduced.matches ? 1 : 1 - Math.exp(-dt * 12);
      root.rotation.x += (targetX - root.rotation.x) * ease;
      root.rotation.y += (targetY - root.rotation.y) * ease;
      root.updateMatrixWorld(true);
      uniforms.uView.value.copy(camera.position).applyMatrix4(inverse.copy(root.matrixWorld).invert()).normalize();
      if (hovering && !reduced.matches) uniforms.uTime.value += dt;
      renderer.render(scene, camera);
    };
    animate(performance.now());
    host.dataset.renderer = 'webgl';
    renderer.domElement.addEventListener('webglcontextlost', (event) => {
      event.preventDefault();
      cancelAnimationFrame(frame);
      host.dataset.renderer = 'layers';
    });
  } catch (error) {
    renderer?.dispose();
    renderer?.domElement.remove();
    host.dataset.renderer = 'layers';
    console.warn('Card preview is using its layered artwork fallback.', error);
  }
}
