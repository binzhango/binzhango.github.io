import { afterEach, describe, expect, it, vi } from 'vitest';
import { setupHomepageMotion } from '../src/scripts/home-motion';

function browser({ native = false, enabled = true } = {}) {
  const styles = [16, 32, 48].map((travel) => ({ dataset: { travel }, style: { transform: '', removeProperty() { this.transform = ''; } } }));
  const hero = { querySelectorAll: () => styles, getBoundingClientRect: () => ({ top: 100, height: 600 }) };
  const listeners = new Map<string, () => void>();
  const observers: ((entries: { isIntersecting: boolean }[]) => void)[] = [];
  const media = { matches: enabled, addEventListener: vi.fn(), removeEventListener: vi.fn() };
  const window = { scrollY: 0, addEventListener: vi.fn((event: string, callback: () => void) => listeners.set(event, callback)), removeEventListener: vi.fn((event: string) => listeners.delete(event)) };
  let scheduled: (() => void) | undefined;
  vi.stubGlobal('window', window);
  vi.stubGlobal('document', { querySelector: (selector: string) => selector === '[data-parallax-hero]' ? hero : null, addEventListener: vi.fn() });
  vi.stubGlobal('matchMedia', () => media);
  vi.stubGlobal('CSS', { supports: () => native });
  vi.stubGlobal('requestAnimationFrame', vi.fn((callback: () => void) => { scheduled = callback; return 1; }));
  vi.stubGlobal('cancelAnimationFrame', vi.fn(() => { scheduled = undefined; }));
  vi.stubGlobal('IntersectionObserver', class { constructor(callback: typeof observers[number]) { observers.push(callback); } observe() {} disconnect() {} });
  vi.stubGlobal('ResizeObserver', class { observe() {} disconnect() {} });
  setupHomepageMotion();
  return { styles, listeners, observers, media, window, flush: () => { const callback = scheduled; scheduled = undefined; callback?.(); } };
}

afterEach(() => vi.unstubAllGlobals());

describe('homepage motion enhancement', () => {
  it('moves decorative layers proportionally, clamps the travel, and stops work offscreen', () => {
    const page = browser();
    page.observers[0]([{ isIntersecting: true }]);
    page.window.scrollY = 400;
    page.flush();
    expect(page.styles.map((layer) => layer.style.transform)).toEqual(['translateY(8px)', 'translateY(16px)', 'translateY(24px)']);
    page.window.scrollY = 1200;
    page.listeners.get('scroll')?.(); page.flush();
    expect(page.styles.map((layer) => layer.style.transform)).toEqual(['translateY(16px)', 'translateY(32px)', 'translateY(48px)']);
    page.observers[0]([{ isIntersecting: false }]);
    expect(page.listeners.has('scroll')).toBe(false);
  });

  it('clears motion immediately when reduced motion or mobile layout becomes active', () => {
    const page = browser();
    page.observers[0]([{ isIntersecting: true }]); page.flush();
    page.media.matches = false;
    page.media.addEventListener.mock.calls[0][1]();
    expect(page.listeners.has('scroll')).toBe(false);
    expect(page.styles.every((layer) => layer.style.transform === '')).toBe(true);
  });

  it.each([{ native: true }, { enabled: false }])('avoids a JS scroll listener when native motion or a static layout applies: %j', (options) => {
    const page = browser(options);
    page.observers[0]([{ isIntersecting: true }]); page.flush();
    expect(page.listeners.has('scroll')).toBe(false);
    expect(page.styles.every((layer) => layer.style.transform === '')).toBe(true);
  });
});
