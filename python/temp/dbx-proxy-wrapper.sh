#!/bin/zsh
#wrapper for launchd: sources .zshrc so DBX_TOKEN is available

source "$HOME/.zshrc" 2>/dev/null
exec python <path-to>/dbx-proxy.pyx