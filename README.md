# VinaNote Offline (macOS Apple Silicon & Windows — 100% On-Device AI)

> **Nhờ AI cài hộ:** clone repo, mở Claude Code / Codex trong thư mục và nói "cài đặt VinaNote Offline trên máy này". Hướng dẫn cho AI nằm ở [AGENTS.md](AGENTS.md).
>
> **Windows 10/11:** xem hướng dẫn cài đặt tại [docs/INSTALL-WINDOWS.md](docs/INSTALL-WINDOWS.md). Bản Windows dùng faster-whisper + llama.cpp (CUDA / Vulkan / CPU) thay cho MLX, và bộ thu âm WASAPI thay cho ScreenCaptureKit. Giao diện, tiện ích trình duyệt và luật dịch tiếng Việt giữ nguyên.

Ứng dụng ghi âm cuộc họp, dịch phụ đề thời gian thực và thuyết minh lồng tiếng tiếng Việt **chạy 100% nội bộ (Offline)** trên máy Mac dùng chip Apple Silicon (M1/M2/M3/M4), đi kèm Tiện ích mở rộng trình duyệt cho **YouTube, TikTok, Douyin, Bilibili, Facebook, Coursera, Udemy**.

## 1. Kiến trúc & Mô hình AI Nội bộ
- **Nhận dạng giọng nói (ASR):** `mlx-whisper` (`mlx-community/whisper-large-v3-turbo`) chạy trên Apple Metal GPU.
- **Dịch thuật, Tóm tắt & Hỏi đáp (LLM):** `mlx-lm` (`mlx-community/Qwen2.5-7B-Instruct-4bit`) chạy trên Apple Metal GPU kèm bộ lọc sửa lỗi sót chữ Hán tự động.
- **Đọc lồng tiếng tiếng Việt (TTS):** `ZeroTTS` (`zeroweight-ai/ZeroTTS`) chạy song song trên CPU (ONNX Runtime), hỗ trợ nhiều giọng Nam/Nữ và tự động khớp tốc độ phát.
- **Thu âm hệ thống & Khử vọng:** Bộ thu `ScreenCaptureKit` + `AVAudioEngine` (`vinanote-audiocap`) tự loại trừ tiếng của chính ứng dụng, kết hợp bộ lọc khử vọng âm 2 tầng (`AcousticEchoSuppressor` + `TextEchoFilter`).
- **Không cần Internet sau khi tải model:** Mọi suy luận đọc trực tiếp từ đường dẫn snapshot cục bộ (`local_files_only=True`), không giới hạn số phút, không gửi âm thanh hay tài liệu ra ngoài.

## 2. Đường dẫn Bản đóng gói `.app` & Tiện ích Trình duyệt
- **Ứng dụng macOS (`.app`):**
  `src-tauri/target/release/bundle/macos/VinaNote Offline.app`
- **Tiện ích Chrome / Edge / Brave / Arc (Load Unpacked):**
  `browser-extension/`

### Cách cài Tiện ích dịch YouTube / TikTok:
1. Mở `chrome://extensions` (hoặc `edge://extensions`).
2. Bật **Developer mode (Chế độ dành cho nhà phát triển)**.
3. Nhấn **Load unpacked** và chọn thư mục `/Users/tuanda/Projects/VinaNote-Offline/browser-extension`.
4. Mở bất kỳ video YouTube, TikTok, Douyin, Bilibili, Facebook nào -> bấm icon **VinaNote Offline** trên thanh công cụ -> **Bật Dịch & Thuyết Minh Tab Này**.

## 3. Lệnh Phát triển & Đóng gói lại
```bash
# Chạy chế độ phát triển
bun run tauri dev

# Đóng gói lại file .app
PATH="$HOME/.cargo/bin:$PATH" bun run tauri build
```

## 4. Bản Windows
```powershell
setup-windows.cmd                      # cài AI Engine (Python, llama.cpp CPU/CUDA/Vulkan, VC++ runtime)
scripts\windows\build.ps1              # build bộ cài NSIS ngay trên Windows
```
```bash
scripts/windows/cross-build-on-mac.sh  # cross-build bộ cài + gói release/windows/VinaNote-Offline-Windows.zip từ macOS
```
Chi tiết: [docs/INSTALL-WINDOWS.md](docs/INSTALL-WINDOWS.md).
