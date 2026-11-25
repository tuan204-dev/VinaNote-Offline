<#
  Build bộ cài VinaNote Offline (.exe NSIS) từ mã nguồn, ngay trên Windows 10/11 x64.
  Cần: Rust (rustup, toolchain MSVC), Visual Studio Build Tools "Desktop development with C++", Bun.
    powershell -ExecutionPolicy Bypass -File scripts\windows\build.ps1
  Kết quả: VinaNote Offline_<version>_x64-setup.exe ở thư mục gốc.
#>
$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path

function Invoke-Native([scriptblock]$Cmd) {
    & $Cmd
    if ($LASTEXITCODE -ne 0) { throw "Lệnh thất bại (mã $LASTEXITCODE): $Cmd" }
}

Push-Location $Root
try {
    Write-Host "==> Build bộ thu âm WASAPI (vinanote-audiocap.exe)" -ForegroundColor Cyan
    Invoke-Native { cargo build --release --manifest-path audio-capture-win\Cargo.toml }
    Copy-Item audio-capture-win\target\release\vinanote-audiocap.exe src-tauri\resources\ -Force

    Write-Host "==> Build ứng dụng Tauri + bộ cài NSIS" -ForegroundColor Cyan
    Invoke-Native { bun install }
    Invoke-Native { bun run tauri build }

    $Setup = Get-ChildItem src-tauri\target\release\bundle\nsis\*-setup.exe | Select-Object -First 1
    Copy-Item $Setup.FullName $Root -Force
    Write-Host "`nXong: $(Join-Path $Root $Setup.Name)" -ForegroundColor Green
} finally {
    Pop-Location
}
