#!/usr/bin/env bash
# Build the Windows installer on macOS (Tauri + cargo-xwin) and pack the Windows bundle:
#   release/windows/VinaNote-Offline-Windows.zip = installer + ai-worker + browser-extension + setup scripts + guide
# One-time setup:
#   brew install llvm lld nsis && rustup target add x86_64-pc-windows-msvc && cargo install --locked cargo-xwin
# (cargo-xwin downloads the Microsoft CRT / Windows SDK on first use, accepting Microsoft's license.)
set -euo pipefail
cd "$(dirname "$0")/../.."
export PATH="$(brew --prefix llvm)/bin:$(brew --prefix lld)/bin:$HOME/.cargo/bin:$HOME/.bun/bin:$PATH"
T=x86_64-pc-windows-msvc

cargo xwin build --release --target "$T" --manifest-path audio-capture-win/Cargo.toml
cp "audio-capture-win/target/$T/release/vinanote-audiocap.exe" src-tauri/resources/
bun install
bun run tauri build --runner cargo-xwin --target "$T"
setup=$(ls "src-tauri/target/$T/release/bundle/nsis/"*-setup.exe | head -1)

stage=release/windows/VinaNote-Offline
rm -rf release/windows && mkdir -p "$stage/scripts/windows" "$stage/docs"
rsync -a --exclude .venv --exclude __pycache__ --exclude .DS_Store --exclude verify_offline_pipeline.py \
  ai-worker browser-extension "$stage/"
cp setup-windows.cmd README.md AGENTS.md CLAUDE.md "$stage/"
cp scripts/windows/setup.ps1 "$stage/scripts/windows/"
cp docs/INSTALL-WINDOWS.md "$stage/docs/"
cp "$setup" "$stage/"
(cd release/windows && zip -qrX VinaNote-Offline-Windows.zip VinaNote-Offline)
echo "Installer: $setup"
echo "Bundle:    $(pwd)/release/windows/VinaNote-Offline-Windows.zip"
