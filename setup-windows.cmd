@echo off
rem VinaNote Offline - cai AI Engine tren Windows (nhap dup file nay).
rem Tuy chon: setup-windows.cmd -Gpu cuda / vulkan / cpu -DownloadModels
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\windows\setup.ps1" %*
set "ERR=%ERRORLEVEL%"
pause
exit /b %ERR%
