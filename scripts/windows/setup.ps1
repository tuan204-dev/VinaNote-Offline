﻿<#
  VinaNote Offline - cài AI Engine (Python) trên Windows 10/11 x64.
  Nhấp đúp setup-windows.cmd ở thư mục gốc, hoặc:
    powershell -ExecutionPolicy Bypass -File scripts\windows\setup.ps1 [-Gpu auto|cuda|vulkan|cpu] [-DownloadModels] [-Unattended]
  -Gpu auto   : có card NVIDIA (nvidia-smi) -> cuda, còn lại -> cpu
  -Gpu vulkan : card AMD / Intel Arc (llama.cpp chạy trên GPU, Whisper chạy CPU)
  -Unattended : không hỏi gì (cho AI agent / script); bộ cài *setup.exe ở thư mục gốc được cài im lặng (/S)
#>
param(
    [ValidateSet("auto", "cuda", "vulkan", "cpu")] [string]$Gpu = "auto",
    [switch]$DownloadModels,
    [switch]$Unattended
)
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Worker = Join-Path $Root "ai-worker"
$Venv = Join-Path $Worker ".venv"
$Py = Join-Path $Venv "Scripts\python.exe"

function Step([string]$Msg) { Write-Host "`n==> $Msg" -ForegroundColor Cyan }
function Invoke-Native([scriptblock]$Cmd) {
    & $Cmd
    if ($LASTEXITCODE -ne 0) { throw "Lệnh thất bại (mã $LASTEXITCODE): $Cmd" }
}
function Set-UserEnv([string]$Name, [string]$Value) {
    # Saved for apps started later, and set in this process so the installer / app started from here see it.
    # An empty value deletes the variable ([NullString]: PowerShell would pass $null to .NET as "").
    $V = if ($Value) { $Value } else { [NullString]::Value }
    [Environment]::SetEnvironmentVariable($Name, $V, "User")
    [Environment]::SetEnvironmentVariable($Name, $V, "Process")
}
function Install-Llama([string]$Variant) {
    $Wheel = @{ cpu = @("cpu", "0.3.38"); vulkan = @("vulkan", "0.3.38"); cuda = @("cu124", "0.3.37") }[$Variant]
    # --only-binary: never fall back to building llama.cpp from source (needs Visual Studio + CMake)
    Invoke-Native {
        uv pip install --python $Py --only-binary llama-cpp-python --reinstall-package llama-cpp-python `
            "llama-cpp-python==$($Wheel[1])" --extra-index-url "https://abetlen.github.io/llama-cpp-python/whl/$($Wheel[0])"
    }
}

Step "1/6 Kiểm tra Microsoft Visual C++ Redistributable (cần cho faster-whisper, ONNX Runtime, llama.cpp)"
$Vc = Get-ItemProperty "HKLM:\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64" -ErrorAction SilentlyContinue
if (-not $Vc -or $Vc.Installed -ne 1) {
    $VcExe = Join-Path $env:TEMP "vc_redist.x64.exe"
    Invoke-WebRequest "https://aka.ms/vs/17/release/vc_redist.x64.exe" -OutFile $VcExe -UseBasicParsing
    Write-Host "Windows sẽ hỏi quyền Admin để cài Visual C++ Redistributable (Microsoft)."
    $P = Start-Process $VcExe -ArgumentList "/install", "/quiet", "/norestart" -Verb RunAs -Wait -PassThru
    if ($P.ExitCode -notin 0, 1638, 3010) { throw "Cài Visual C++ Redistributable thất bại (mã $($P.ExitCode))" }
}

Step "2/6 Kiểm tra uv (trình quản lý Python, không cần quyền Admin)"
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    $env:Path = "$env:USERPROFILE\.local\bin;$env:USERPROFILE\.cargo\bin;$env:Path"
}

Step "3/6 Tạo môi trường Python 3.12 tại $Venv"
# A .venv copied over from macOS (bin\ instead of Scripts\) cannot run here: start clean.
if ((Test-Path $Venv) -and -not (Test-Path $Py)) { Remove-Item $Venv -Recurse -Force }
Invoke-Native { uv venv --python 3.12 --allow-existing $Venv }

Step "4/6 Cài thư viện AI (faster-whisper, ZeroTTS, FastAPI, ffmpeg...)"
Invoke-Native { uv pip install --python $Py -r (Join-Path $Worker "requirements-windows.txt") }

if ($Gpu -eq "auto") {
    $Gpu = "cpu"
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) { $Gpu = "cuda" }
}
Step "5/6 Cài llama.cpp cho bộ dịch Qwen2.5 (bản: $Gpu)"
if ($Gpu -eq "cuda") {
    # cuBLAS + cuDNN for faster-whisper on the GPU (the llama.cpp CUDA wheel bundles its own CUDA DLLs, ~600 MB)
    Invoke-Native { uv pip install --python $Py nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*" }
}
Install-Llama $Gpu
& $Py -c "import llama_cpp"
if ($LASTEXITCODE -ne 0 -and $Gpu -ne "cpu") {
    Write-Warning "Bản llama.cpp '$Gpu' không nạp được trên máy này (driver / GPU), chuyển sang bản CPU."
    $Gpu = "cpu"
    Install-Llama "cpu"
}
# Whisper uses CUDA only with the CUDA install: after -Gpu cpu|vulkan an older CUDA setup must not win.
Set-UserEnv "VINANOTE_ASR_DEVICE" $(if ($Gpu -eq "cuda") { "" } else { "cpu" })

Step "6/6 Kiểm tra AI Engine"
Set-UserEnv "VINANOTE_ROOT" $Root  # the app finds ai-worker\ and browser-extension\ here
Push-Location $Worker
try {
    # Loads every native runtime (CTranslate2, ONNX Runtime, llama.cpp) and runs ffmpeg, so a missing DLL fails here.
    Invoke-Native {
        & $Py -c "import ctranslate2, faster_whisper, onnxruntime, zerotts, llama_cpp, server, subprocess; subprocess.run([server.find_ffmpeg(), '-version'], check=True, capture_output=True); print('AI Engine OK | llama.cpp GPU:', llama_cpp.llama_supports_gpu_offload(), '| GPU CUDA:', ctranslate2.get_cuda_device_count(), '| model da tai:', server.engine.status()['ready'])"
    }
    if ($DownloadModels) {
        Step "Tải model AI (~7 GB, chỉ cần một lần)"
        Invoke-Native { & $Py -c "import server; server.engine.download_all(print)" }
    }
} finally {
    Pop-Location
}

Write-Host "`nHoàn tất! AI Engine ($Gpu) đã sẵn sàng. VINANOTE_ROOT = $Root" -ForegroundColor Green
$Installer = Get-ChildItem $Root -Filter "VinaNote*setup.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
if ($Installer) {
    $Answer = if ($Unattended) { "y" } else { Read-Host "Cài / cập nhật ứng dụng VinaNote Offline ngay bây giờ? (Y/n)" }
    if ($Answer -notmatch '^[nN]') {
        # WaitForExit, not -Wait: -Wait would also wait for the app if the installer's last page starts it.
        $StartArgs = @{ FilePath = $Installer.FullName; PassThru = $true }
        if ($Unattended) { $StartArgs.ArgumentList = "/S" }  # NSIS silent install for the current user, no UI
        $P = Start-Process @StartArgs
        $P.WaitForExit()
        if ($P.ExitCode -ne 0) { Write-Warning "Bộ cài kết thúc với mã $($P.ExitCode)." }
    }
}
Write-Host "Mở VinaNote Offline > tab 'Trình duyệt & Model AI' > 'Tải trọn bộ Model Offline' nếu chưa tải model."
