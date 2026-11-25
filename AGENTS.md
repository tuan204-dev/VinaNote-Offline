# AGENTS.md — VinaNote Offline

Context for AI coding agents (Codex, Claude Code, …) working in this repo, above all when the user asks you to
**install / set up VinaNote Offline on this machine**. Talk to the user in Vietnamese. The human install guide is
`docs/INSTALL-WINDOWS.md` (Vietnamese); this file is its operational version for agents.

## What the app is

- **Tauri 2 desktop app**: React UI in `src/`, Rust backend in `src-tauri/src/lib.rs`.
- **Python AI worker** `ai-worker/server.py`: FastAPI on `127.0.0.1:38479`, spawned by the app, 100% offline once
  models are downloaded (~7 GB from Hugging Face into `~/.cache/huggingface/hub`). Backend picked by env
  `VINANOTE_BACKEND` (default: `mlx` on macOS Apple Silicon, `portable` elsewhere):
  - `ai-worker/models.py`: MLX (mlx-whisper + mlx-lm Qwen2.5-7B) for Apple Silicon.
  - `ai-worker/models_portable.py`: Windows/Linux/Intel Mac (faster-whisper + llama-cpp-python Qwen2.5-7B GGUF);
    subclasses the MLX engine and reuses all its Vietnamese text rules.
- **Native audio helper** (system audio + mic, JSON-lines over stdin/stdout, protocol at the top of
  `audio-capture-win/src/main.rs`):
  - macOS: `src-tauri/resources/vinanote-audiocap` (prebuilt from `audio-capture/audiocap.swift`, committed);
  - Windows: `vinanote-audiocap.exe` built from `audio-capture-win/` (not committed: build scripts produce it).
- **Chrome/Edge MV3 extension** `browser-extension/`: talks to the worker (38479) and to the app's bridge (38478).
- **The cloned repo is the runtime home.** The app looks for `ai-worker/` and `browser-extension/` via env
  `VINANOTE_ROOT` (set by the Windows setup script), the cwd, the folders above its own executable, then a fallback
  (`C:\VinaNote-Offline` on Windows). Do not move or delete the repo after installing; re-run setup if it moves.

## Install on Windows 10/11 x64

Work in PowerShell from the repo root. Call the `.ps1` scripts directly: `setup-windows.cmd` ends with `pause`,
which blocks non-interactive shells. Expect long downloads (CUDA wheels ~2 GB, models ~7 GB): use generous
timeouts or run in the background and report progress.

1. **Check the machine** and tell the user what you found:
   - x64 (`[Environment]::Is64BitOperatingSystem`); RAM ≥ 16 GB; ≥ 15 GB free disk.
   - OS build `[Environment]::OSVersion.Version.Build`: ≥ 22000 is Windows 11. On Windows 10 the app captures
     system audio without excluding itself, so live dubbing turns off while system audio is captured.
   - GPU: NVIDIA if `nvidia-smi` works; otherwise ask about AMD Radeon / Intel Arc (→ `-Gpu vulkan`).
   - Preferred repo location `C:\VinaNote-Offline` (any path works). Avoid OneDrive-synced folders.
2. **AI engine** (Python env, native runtimes, optional model download):
   ```powershell
   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\windows\setup.ps1 -Gpu auto -DownloadModels -Unattended
   ```
   - Installs the Visual C++ 2015-2022 x64 runtime if missing. **This shows a UAC prompt: ask the user to approve
     it.** It needs an interactive desktop session (not SSH).
   - Then: uv → Python 3.12 venv `ai-worker\.venv` → `ai-worker\requirements-windows.txt` → the prebuilt
     llama-cpp-python wheel (cpu 0.3.38 / cu124 0.3.37 / vulkan 0.3.38, falls back to cpu if the GPU wheel cannot
     load) → user env `VINANOTE_ROOT` and `VINANOTE_ASR_DEVICE` → smoke test of CTranslate2, ONNX Runtime,
     llama.cpp, ZeroTTS and ffmpeg.
   - `-Gpu auto|cuda|vulkan|cpu`. `-DownloadModels` can be dropped and done later in the app.
   - Do not `pip install llama-cpp-python` from PyPI (source build needs Visual Studio + CMake), and do not install
     the CUDA Toolkit (the wheels bring their CUDA DLLs).
3. **App installer** `VinaNote Offline_<version>_x64-setup.exe` (NSIS, per-user, no admin). Get it in this order:
   1. Already in the repo root (e.g. copied from `VinaNote-Offline-Windows.zip`, which the user can build on their
      Mac with `scripts/windows/cross-build-on-mac.sh`), or attached to a GitHub Release of this repo if there is one.
   2. Otherwise build it here (prerequisites via winget; Build Tools need UAC):
      ```powershell
      winget install -e --id Rustlang.Rustup --accept-package-agreements --accept-source-agreements
      winget install -e --id Microsoft.VisualStudio.2022.BuildTools --accept-package-agreements --accept-source-agreements --override "--quiet --wait --norestart --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
      winget install -e --id Oven-sh.Bun --accept-package-agreements --accept-source-agreements
      $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [Environment]::GetEnvironmentVariable("Path", "User")
      powershell -NoProfile -ExecutionPolicy Bypass -File scripts\windows\build.ps1   # -> installer in the repo root
      ```
      (A newer Visual Studio Build Tools with the C++ workload works too.)
4. **Install the app silently** (or re-run step 2's script with `-Unattended`, which installs any `*setup.exe` in
   the repo root):
   ```powershell
   $Setup = Get-ChildItem . -Filter "VinaNote*setup.exe" | Select-Object -First 1
   (Start-Process $Setup.FullName -ArgumentList "/S" -PassThru).WaitForExit()
   ```
   This installs to `%LOCALAPPDATA%\VinaNote Offline\vinanote-offline.exe`, with the helper in `...\resources\`.
5. **Verify** (see below), then **hand over to the user** the steps an agent cannot do:
   - approve UAC prompts;
   - SmartScreen "More info → Run anyway", if they start the unsigned installer by hand;
   - Windows mic privacy ("Let desktop apps access your microphone");
   - loading the browser extension: `chrome://extensions` → Developer mode → Load unpacked → `<repo>\browser-extension`;
   - if models were not downloaded: app → tab "Trình duyệt & Model AI" → "Tải trọn bộ Model Offline".

### Verify a Windows install

```powershell
cd ai-worker
.\.venv\Scripts\python.exe test_portable_rules.py; .\.venv\Scripts\python.exe test_translation_rules.py
.\.venv\Scripts\python.exe server.py        # foreground: shows any error; Ctrl+C when done
# in another shell:
Invoke-RestMethod http://127.0.0.1:38479/models/status   # backend=portable, ready=true once models are downloaded
& "$env:LOCALAPPDATA\VinaNote Offline\resources\vinanote-audiocap.exe" --check   # one "permissions" JSON line
& "$env:LOCALAPPDATA\VinaNote Offline\vinanote-offline.exe"                      # launch the app
```

The app reuses a worker already listening on 38479, so you can stop the manual one first or leave it running.
After the first live session, `models/status` → `devices` shows `asr: cuda|cpu` and `llm: gpu|cpu`. When the app
starts the worker itself, its stdout/stderr are discarded: to debug, run `server.py` by hand as above.

## Install on macOS (Apple Silicon)

Needs Xcode Command Line Tools, Rust (`rustup`) and Bun (<https://bun.sh>) to build the app.

```bash
brew install ffmpeg uv
uv venv --python 3.12 ai-worker/.venv
uv pip install --python ai-worker/.venv/bin/python -r ai-worker/requirements-macos.txt
bun install
bun run tauri dev          # or: bun run tauri build → src-tauri/target/release/bundle/macos/VinaNote Offline.app
```

- Grant Microphone and Screen & System Audio Recording when macOS asks.
- Download the MLX models from the app (tab "Trình duyệt & Model AI").
- A `.app` run from inside the repo's `src-tauri/target/` finds the repo by itself. If you copy it elsewhere
  (e.g. /Applications), run `launchctl setenv VINANOTE_ROOT "<repo path>"`; GUI apps do not read shell profiles.

## Repo conventions

- **Never commit build outputs:** `release/`, `*-setup.exe`, `src-tauri/resources/vinanote-audiocap.exe`,
  `audio-capture-win/target/`, `ai-worker/.venv` (see `.gitignore`).
- **`.ps1` encoding:** scripts are UTF-8 *with BOM*. Windows PowerShell 5.1 reads BOM-less files as ANSI and
  garbles the Vietnamese strings. Keep `setup-windows.cmd` ASCII with CRLF line endings.
- **Model-free checks:** `ai-worker/test_*.py`, `cargo test --manifest-path audio-capture-win/Cargo.toml`,
  `cargo check` in `src-tauri`, `bun run build`.
- **Audio-helper protocol changes** must update all four places: `audio-capture/audiocap.swift`,
  `audio-capture-win/src/main.rs`, `src-tauri/src/lib.rs`, `ai-worker/server.py` (`/ws/live`).
- **Not yet verified on real Windows hardware** (the port was built and tested on macOS): WASAPI capture with
  process-loopback exclusion, the CUDA/Vulkan paths, the VC++ runtime install, and the installer flow. Check
  these first when something fails on Windows.
