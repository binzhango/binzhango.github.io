#!/usr/bin/env bash
set -euo pipefail

# Vercel uses Amazon Linux 2023, so Playwright's apt-based --with-deps
# installer does not apply. The build image already includes GTK/ATK,
# ALSA, CUPS, Pango, and the X11 libraries; add Chromium's remaining libs.
if [[ "${VERCEL:-}" == "1" && "$(uname -s)" == "Linux" ]]; then
  dnf install -y nss nspr mesa-libgbm libdrm libxkbcommon
fi

# Mermaid uses Playwright's default headless shell during content rendering.
node node_modules/playwright/cli.js install --only-shell chromium
