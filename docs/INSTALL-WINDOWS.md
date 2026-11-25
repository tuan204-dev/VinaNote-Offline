# Cài đặt VinaNote Offline trên Windows

VinaNote Offline bản Windows: ghi âm cuộc họp, chép lời, dịch phụ đề và đọc lồng tiếng tiếng Việt **chạy 100% trên máy**. Sau khi tải model lần đầu, ứng dụng không cần Internet và không gửi âm thanh hay tài liệu ra ngoài.

## 1. Yêu cầu

| Hạng mục | Tối thiểu | Khuyến nghị |
|---|---|---|
| Hệ điều hành | Windows 10 64-bit (x64) | Windows 11 (lồng tiếng được cả khi thu âm thanh hệ thống, xem mục 6) |
| RAM | 16 GB | 32 GB nếu không có GPU |
| Ổ cứng trống | 10 GB (model ~7 GB + Python ~1 GB) | 15 GB (bản CUDA thêm ~4 GB thư viện) |
| GPU | Không bắt buộc | NVIDIA RTX/GTX 16xx trở lên, VRAM ≥ 8 GB, driver mới (≥ 551.61) |
| Internet | Chỉ lần cài đầu tiên và khi tải model | |

Trình duyệt dùng tiện ích: Chrome, Edge hoặc Brave. WebView2 có sẵn trên Windows 11; trên Windows 10, bộ cài tự tải về nếu máy chưa có.

## 2. Cài đặt nhanh

1. **Giải nén** `VinaNote-Offline-Windows.zip` vào ổ `C:\`. Trong zip đã có sẵn thư mục `VinaNote-Offline`, nên kết quả là `C:\VinaNote-Offline\setup-windows.cmd`. Đừng giải nén vào một thư mục `VinaNote-Offline` khác, vì sẽ bị lồng hai cấp.
   Nên dùng đúng đường dẫn này vì ứng dụng tự tìm thấy nó. Nếu để chỗ khác, script ở bước 2 sẽ ghi lại vị trí vào biến môi trường `VINANOTE_ROOT`.
2. **Nhấp đúp `setup-windows.cmd`**. Script sẽ:
   - cài **Microsoft Visual C++ Redistributable** nếu máy chưa có (cần cho faster-whisper, ONNX Runtime và llama.cpp; Windows sẽ hỏi quyền Admin đúng một lần);
   - cài `uv` (trình quản lý Python, không cần quyền Admin) và một bản Python 3.12 riêng cho ứng dụng;
   - tạo `ai-worker\.venv` và cài faster-whisper, llama.cpp, ZeroTTS, FastAPI, ffmpeg. Nếu máy có card NVIDIA (lệnh `nvidia-smi` chạy được), script tự cài bản CUDA; nếu bản GPU không nạp được thì tự chuyển về bản CPU;
   - ghi biến môi trường `VINANOTE_ROOT`;
   - nạp thử toàn bộ thư viện AI và ffmpeg, rồi hỏi có chạy bộ cài ứng dụng luôn không.

   Mất khoảng 5–15 phút tùy tốc độ mạng. Bản CUDA tải thêm khoảng 2 GB.
3. **Cài ứng dụng**: chạy `VinaNote Offline_1.0.0_x64-setup.exe`. Ứng dụng được cài cho tài khoản hiện tại vào `%LOCALAPPDATA%\VinaNote Offline`, không cần quyền Admin.
   Bộ cài chưa được ký số, nên SmartScreen có thể báo *"Windows protected your PC"*. Khi đó bấm **More info → Run anyway**.
4. **Tải model AI** (khoảng 7 GB, chỉ làm một lần): mở VinaNote Offline, vào tab **Trình duyệt & Model AI** và bấm **Tải trọn bộ Model Offline**.
   Cách khác: chạy `setup-windows.cmd -DownloadModels` ngay ở bước 2.
5. **Dùng thử**: vào tab **Họp & Âm thanh Máy** và bấm **Bắt đầu Nghe & Dịch**.

**Cách khác: clone từ Git rồi nhờ AI cài hộ.** Clone repo vào `C:\VinaNote-Offline`, mở Claude Code hoặc Codex trong thư mục đó, rồi nói *"Cài đặt VinaNote Offline trên máy này"*. File `AGENTS.md` (Claude Code đọc qua `CLAUDE.md`) mô tả đủ các bước cho AI: kiểm tra máy, chạy `setup.ps1 -Unattended`, build bộ cài nếu trong repo chưa có, cài im lặng rồi kiểm tra. Bạn chỉ cần bấm đồng ý các hộp thoại quyền Admin (UAC) và tự nạp tiện ích trình duyệt.

Kiểm tra lại cài đặt: trong tab **Trình duyệt & Model AI**, cả ba model phải báo *Đã cài đặt Offline*. Dòng **Thiết bị đang chạy** cho biết phần nhận dạng giọng nói và phần dịch đang chạy trên `cuda`/`gpu` hay `cpu`.

## 3. Cài tiện ích trình duyệt (YouTube, TikTok, Douyin, Bilibili, Facebook, Coursera, Udemy)

1. Mở `chrome://extensions` (hoặc `edge://extensions`).
2. Bật **Developer mode** (Chế độ dành cho nhà phát triển).
3. Bấm **Load unpacked** và chọn thư mục `C:\VinaNote-Offline\browser-extension`.
4. Mở một video, bấm biểu tượng **VinaNote Offline** trên thanh công cụ, rồi chọn **Bật Dịch & Thuyết Minh Tab Này**.

Tiện ích chỉ hoạt động khi ứng dụng VinaNote Offline đang mở, vì AI Engine chạy ở `127.0.0.1:38479`.

## 4. Chọn GPU / CPU

Chạy lại `setup-windows.cmd` kèm tùy chọn để đổi bản llama.cpp:

| Lệnh | Khi nào dùng | Nhận dạng giọng nói (Whisper) | Dịch, tóm tắt (Qwen2.5) |
|---|---|---|---|
| `setup-windows.cmd` | Mặc định: tự nhận card NVIDIA | CUDA hoặc CPU | CUDA hoặc CPU |
| `setup-windows.cmd -Gpu cuda` | Card NVIDIA | CUDA | CUDA |
| `setup-windows.cmd -Gpu vulkan` | Card AMD Radeon / Intel Arc | CPU | GPU qua Vulkan |
| `setup-windows.cmd -Gpu cpu` | Không có GPU rời, hoặc GPU gặp lỗi | CPU | CPU |

Nếu GPU không dùng được (thiếu DLL, driver cũ, không đủ VRAM), AI Engine tự chuyển sang CPU chứ không dừng.

**Máy không có GPU.** Ứng dụng vẫn chạy, chỉ chậm hơn:
- **Họp trực tiếp**: mỗi câu mất khoảng 1–3 giây để nhận dạng (bản Windows chỉ encode đoạn âm thanh thật, không đệm lên 30 giây) và 2–5 giây để dịch bằng Qwen2.5-7B.
- **Dịch file video và tiện ích trình duyệt**: cần mốc thời gian từng từ, nên mỗi đoạn mất xấp xỉ bằng hoặc lâu hơn độ dài của chính đoạn đó. Nên dùng GPU cho hai tính năng này.

Có thể đổi sang model nhẹ hơn bằng biến môi trường. Chạy lệnh dưới đây trong PowerShell, mở lại ứng dụng, rồi bấm **Tải trọn bộ Model Offline** lại:

```powershell
# Bộ dịch Qwen2.5-3B: nhanh gấp khoảng 2 lần trên CPU, chất lượng thấp hơn một chút
setx VINANOTE_LLM_GGUF "bartowski/Qwen2.5-3B-Instruct-GGUF/Qwen2.5-3B-Instruct-Q4_K_M.gguf"
# (Tùy chọn) Whisper nhỏ hơn cho máy rất yếu
setx VINANOTE_WHISPER_REPO "Systran/faster-whisper-small"
```

`VINANOTE_LLM_GGUF` cũng nhận đường dẫn tới một file `.gguf` có sẵn trên máy, ví dụ file tải từ LM Studio. Muốn quay lại model mặc định, xóa biến bằng `[Environment]::SetEnvironmentVariable("VINANOTE_LLM_GGUF", $null, "User")`.

## 5. Bản Windows khác bản macOS ở đâu

| Thành phần | macOS (Apple Silicon) | Windows |
|---|---|---|
| Nhận dạng giọng nói | mlx-whisper large-v3-turbo (Metal) | faster-whisper large-v3-turbo (CTranslate2, CUDA hoặc CPU int8) |
| Dịch, tóm tắt, hỏi đáp | mlx-lm Qwen2.5-7B 4-bit | llama.cpp Qwen2.5-7B GGUF Q4_K_M (CUDA / Vulkan / CPU) |
| Lồng tiếng | ZeroTTS (CPU) | ZeroTTS (CPU), giống hệt |
| Thu âm hệ thống | ScreenCaptureKit, loại trừ tiếng của app | WASAPI process loopback, loại trừ tiếng của app |
| Lưu ghi chú | `~/Library/Application Support/local.vinanote.offline/notes` | `%APPDATA%\local.vinanote.offline\notes` |
| Kho model | `~/.cache/huggingface/hub` | `%USERPROFILE%\.cache\huggingface\hub` |

Hai bản dùng chung luật dịch "chỉ ra tiếng Việt": chặn chữ Hán/Cyrillic và phạt từ tiếng Anh chưa dịch ngay trong lúc sinh chữ. Giao diện, tiện ích trình duyệt và định dạng ghi chú cũng giống nhau.

## 6. Xử lý sự cố

**Ứng dụng báo "Không tìm thấy môi trường AI Worker…"**
Bạn chưa chạy `setup-windows.cmd`, hoặc thư mục `VinaNote-Offline` đã bị di chuyển. Chạy lại `setup-windows.cmd` ở vị trí mới.

**AI Engine không khởi động, hoặc muốn xem lỗi chi tiết.** Đóng ứng dụng rồi chạy AI Engine bằng tay trong PowerShell:

```powershell
cd C:\VinaNote-Offline\ai-worker
.\.venv\Scripts\python.exe server.py
```

Lỗi sẽ hiện ngay trong cửa sổ đó. Khi AI Engine đã chạy, mở lại ứng dụng: ứng dụng sẽ dùng luôn AI Engine đang chạy ở cổng `38479`.

**Micro báo `denied` / không thu được giọng nói**
Mở **Cài đặt > Quyền riêng tư & bảo mật > Micro** rồi bật **Quyền truy cập micro** và **Cho phép ứng dụng máy tính truy cập micro**. Sau đó kiểm tra micro mặc định trong **Cài đặt > Hệ thống > Âm thanh**.

**Windows 10: "Đọc lồng tiếng" tự tắt khi thu âm thanh hệ thống**
Windows 11 cho phép thu âm thanh hệ thống mà bỏ qua tiếng của chính VinaNote. Windows 10 (kể cả bản cuối 22H2) chưa có tính năng này, nên bộ thu phải nghe toàn bộ âm thanh phát ra, kể cả qua tai nghe. Nếu vẫn lồng tiếng, giọng đọc tiếng Việt sẽ bị chép lại thành lời thoại mới. Vì vậy, trên Windows 10 ứng dụng tự tắt **Đọc lồng tiếng** cho phiên đó và hiện thông báo. Hiếm khi, nếu tính năng này lỗi, Windows 11 cũng rơi vào trường hợp này và được xử lý y hệt. Chỉ thu Micro, hoặc dùng tab **Xem & Dịch File Video** và tiện ích trình duyệt, thì vẫn lồng tiếng bình thường.

**Video không phát (màn hình đen) dù vẫn có phụ đề**
Trình phát của ứng dụng (WebView2) hỗ trợ tốt nhất MP4 (H.264/AAC) và WebM. Hãy chuyển MKV/MOV sang MP4. Trên Windows bản **N** (châu Âu), cần cài thêm **Media Feature Pack** trong *Cài đặt > Ứng dụng > Tính năng tùy chọn*.

**Có card NVIDIA nhưng "Thiết bị đang chạy" vẫn báo `cpu`**
Kiểm tra driver bằng lệnh `nvidia-smi`, cập nhật driver nếu cần, rồi chạy `setup-windows.cmd -Gpu cuda`. Whisper chỉ chuyển sang CUDA sau khi một bài thử trên GPU chạy thành công; nếu bài thử lỗi (ví dụ thiếu cuDNN), nó ở lại CPU để không làm sập app. Khi cài bằng `-Gpu cpu` hoặc `-Gpu vulkan`, script đặt biến `VINANOTE_ASR_DEVICE=cpu` để Whisper luôn chạy CPU.

**Lỗi thiếu DLL (`MSVCP140.dll`, `VCRUNTIME140.dll`...)**
Cài [Microsoft Visual C++ Redistributable x64](https://aka.ms/vs/17/release/vc_redist.x64.exe) rồi mở lại ứng dụng. `setup-windows.cmd` thường đã tự cài bước này.

**Cổng 38478 / 38479 đã bị chiếm**
Có một bản VinaNote khác (hoặc AI Engine chạy tay) đang mở. Tắt hết rồi mở lại ứng dụng.

**Windows Defender chặn `vinanote-audiocap.exe`**
Đây là bộ thu âm (WASAPI) của VinaNote, nằm ở `%LOCALAPPDATA%\VinaNote Offline\resources`. Hãy cho phép nó trong Windows Security.

## 7. Gỡ cài đặt

1. **Cài đặt > Ứng dụng > VinaNote Offline > Gỡ cài đặt.**
2. Xóa thư mục `C:\VinaNote-Offline`.
3. (Tùy chọn) Xóa model (khoảng 7 GB) trong `%USERPROFILE%\.cache\huggingface\hub`, gồm các thư mục `models--dropbox-dash--faster-whisper-large-v3-turbo`, `models--bartowski--Qwen2.5-7B-Instruct-GGUF` và `models--zeroweight-ai--ZeroTTS`.
4. (Tùy chọn) Xóa biến môi trường: `[Environment]::SetEnvironmentVariable("VINANOTE_ROOT", $null, "User")`.

## 8. Dành cho nhà phát triển: build bộ cài

**Trên Windows.** Cần cài trước:
- Rust: <https://rustup.rs>, chọn toolchain MSVC;
- Visual Studio Build Tools với workload **Desktop development with C++**;
- Bun: `powershell -c "irm bun.sh/install.ps1 | iex"`.

Sau đó chạy:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\windows\build.ps1
```

Bộ cài `VinaNote Offline_<version>_x64-setup.exe` sẽ nằm ở thư mục gốc.

**Cross-build trên macOS** (đây là cách bộ cài hiện tại được build):

```bash
brew install llvm lld nsis
rustup target add x86_64-pc-windows-msvc
cargo install --locked cargo-xwin
scripts/windows/cross-build-on-mac.sh   # -> release/windows/VinaNote-Offline-Windows.zip
```

Lần đầu chạy, cargo-xwin tải Microsoft CRT và Windows SDK, nghĩa là bạn đồng ý với giấy phép của Microsoft.

**Mã nguồn của bản Windows:**
- `ai-worker/models_portable.py`: backend AI cho Windows (faster-whisper + llama.cpp), kế thừa toàn bộ luật dịch của `models.py`. Chọn backend bằng biến `VINANOTE_BACKEND=mlx|portable`. Mặc định: `mlx` trên Mac Apple Silicon, `portable` ở mọi nơi khác.
- `ai-worker/requirements-windows.txt`: thư viện Python; llama-cpp-python được `setup.ps1` cài riêng theo bản CPU / CUDA / Vulkan.
- `audio-capture-win/`: bộ thu âm WASAPI `vinanote-audiocap.exe`. Nó dùng cùng giao thức JSON-lines với `audio-capture/audiocap.swift`.
- `src-tauri/tauri.windows.conf.json`: cấu hình đóng gói NSIS, ghi đè `tauri.conf.json` khi build cho Windows.
- Kiểm thử không cần model: `.venv\Scripts\python.exe test_portable_rules.py` và `test_translation_rules.py` trong `ai-worker`.
