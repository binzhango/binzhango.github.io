/// <reference types="astro/client" />

declare global {
  interface Window {
    theme?: {
      setTheme(theme?: 'auto' | 'light' | 'dark'): void;
      getTheme(): 'auto' | 'light' | 'dark';
      getSystemTheme(): 'light' | 'dark';
      getDefaultTheme(): 'auto' | 'light' | 'dark' | null;
    };
  }
}

export {};
