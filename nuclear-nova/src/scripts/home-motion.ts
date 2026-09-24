/** Progressive enhancement: all content is visible before motion initializes. */
export function setupHomepageMotion() {
  const hero = document.querySelector<HTMLElement>('[data-parallax-hero]');
  if (!hero) return;
  const media = matchMedia('(min-width: 768px) and (prefers-reduced-motion: no-preference)');
  const native = CSS.supports('animation-timeline', 'view()');
  const layers = Array.from(hero.querySelectorAll<HTMLElement>('[data-parallax-layer]'));
  const featured = document.querySelector<HTMLElement>('[data-featured-reveal]');
  let active = false;
  let frame = 0;
  let top = 0;
  let height = 1;
  let reveal: Animation | undefined;

  function update() {
    frame = 0;
    if (!active || !media.matches || native) return;
    const progress = Math.max(0, Math.min(1, (window.scrollY - top) / height));
    layers.forEach((layer) => {
      layer.style.transform = `translateY(${progress * Number(layer.dataset.travel)}px)`;
    });
  }
  function onScroll() { if (!frame && active) frame = requestAnimationFrame(update); }
  function measure() {
    const bounds = hero!.getBoundingClientRect();
    top = bounds.top + window.scrollY;
    height = Math.max(1, bounds.height);
    onScroll();
  }
  function sync() {
    window.removeEventListener('scroll', onScroll);
    cancelAnimationFrame(frame);
    frame = 0;
    layers.forEach((layer) => layer.style.removeProperty('transform'));
    if (!media.matches) reveal?.cancel();
    if (media.matches && !native && active) {
      measure();
      window.addEventListener('scroll', onScroll, { passive: true });
    }
  }
  const observer = new IntersectionObserver(([entry]) => {
    active = entry.isIntersecting;
    // Freeze the last position offscreen; avoid resetting it during exit.
    if (active) sync();
    else { window.removeEventListener('scroll', onScroll); cancelAnimationFrame(frame); frame = 0; }
  });
  observer.observe(hero);
  const resize = new ResizeObserver(measure);
  resize.observe(hero);
  const revealObserver = new IntersectionObserver(([entry]) => {
    if (!entry.isIntersecting) return;
    if (media.matches && !native && featured) {
      reveal = featured.animate([{ transform: 'translateY(12px)', opacity: .8 }, { transform: 'none', opacity: 1 }], { duration: 450, easing: 'ease-out' });
    }
    revealObserver.disconnect();
  }, { threshold: .15 });
  if (featured) revealObserver.observe(featured);
  media.addEventListener('change', sync);
  // Native full-page navigation tears down the document; also support Astro swaps.
  document.addEventListener('astro:before-swap', () => {
    observer.disconnect(); resize.disconnect(); revealObserver.disconnect();
    window.removeEventListener('scroll', onScroll);
    media.removeEventListener('change', sync);
    cancelAnimationFrame(frame); reveal?.cancel();
  }, { once: true });
}
