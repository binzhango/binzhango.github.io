#!/usr/bin/env bash
set -euo pipefail

# Vercel uses Amazon Linux 2023, so Playwright's apt-based --with-deps
# installer does not apply. The build image already includes GTK/ATK,
# ALSA, CUPS, Pango, and the X11 libraries; add Chromium's remaining libs.
if [[ "${VERCEL:-}" == "1" && "$(uname -s)" == "Linux" ]]; then
  dnf install -y nss nspr mesa-libgbm libdrm libxkbcommon
fi

# Log the actual runtime because Vercel and GitHub may use different Node patches.
node --version
node node_modules/playwright/cli.js --version

# Mermaid uses Playwright's default headless shell during content rendering.
# Bound the entire installer, including archive extraction, not only the download.
if [[ "${VERCEL:-}" == "1" && "$(uname -s)" == "Linux" ]]; then
  timeout --kill-after=10s 300s node node_modules/playwright/cli.js install --only-shell chromium
else
  node node_modules/playwright/cli.js install --only-shell chromium
fi

# Fail during installation if the downloaded browser cannot run on the build image.
node --input-type=module -e '
  import { chromium } from "playwright";
  const browser = await chromium.launch({ timeout: 30_000 });
  console.log(`Chromium ${browser.version()} launch verified`);
  await browser.close();
'
