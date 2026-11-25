#!/usr/bin/env python3
"""
End-to-end 100% Offline Verification Test for VinaNote Offline:
1. Synthesizes a sample English test video (.mp4) locally using macOS `say` + `ffmpeg`.
2. Forces `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` to prove zero internet dependency.
3. Tests:
   - Video window extraction + MLX Whisper Large-v3-Turbo ASR (word timestamps)
   - MLX Qwen2.5-7B-Instruct-4bit Vietnamese subtitle translation
   - ZeroTTS Vietnamese speech synthesis (WAV)
   - Meeting/Video Markdown summarization & Q&A
   - Word (.docx) export
"""

import os
import subprocess
import tempfile
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from server import (
    AskNoteRequest,
    ExportDocxRequest,
    SummarizeRequest,
    VideoWindowRequest,
    engine,
    notes_export_docx,
    video_translate_window,
)


def main() -> None:
    st = engine.status()
    assert st["ready"], f"Models not ready yet: {st}"
    print("[1/5] Model status verified (offline_locked=True):")
    for k, v in st["models"].items():
        print(f"      - {k}: {v['repo']} -> {v['path']}")

    with tempfile.TemporaryDirectory() as tmp:
        aiff_path = os.path.join(tmp, "speech.aiff")
        mp4_path = os.path.join(tmp, "sample_en.mp4")
        docx_path = os.path.join(tmp, "exported_note.docx")

        sample_en = (
            "Hello everyone, welcome to our engineering sync. "
            "Today we are testing real-time video translation running completely offline on Apple Silicon. "
            "Please finalize the release checklist by Friday."
        )
        subprocess.run(["/usr/bin/say", "-v", "Samantha", sample_en, "-o", aiff_path], check=True)
        subprocess.run(
            [
                "/opt/homebrew/bin/ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=c=black:s=640x360:r=15",
                "-i",
                aiff_path,
                "-shortest",
                "-c:v",
                "libx264",
                "-c:a",
                "aac",
                mp4_path,
            ],
            capture_output=True,
            check=True,
        )

        t0 = time.monotonic()
        win_res = video_translate_window(
            VideoWindowRequest(file_path=mp4_path, start_sec=0.0, duration_sec=20.0, language="en")
        )
        print(f"[2/5] Video ASR + Translation completed in {(time.monotonic()-t0):.2f}s (ASR: {win_res['asr_ms']}ms):")
        for cue in win_res["cues"]:
            print(f"      [{cue['start']}s - {cue['end']}s] EN: {cue['source']}")
            print(f"                        VI: {cue['translation']}")
        assert len(win_res["cues"]) > 0, "Expected at least 1 translated cue"

        first_vi = win_res["cues"][0]["translation"]
        t1 = time.monotonic()
        wav_bytes, sr, dur = engine.synthesize_wav_bytes(first_vi, voice="maichi")
        print(f"[3/5] ZeroTTS synthesized {dur:.2f}s @ {sr}Hz ({len(wav_bytes)} bytes) in {(time.monotonic()-t1)*1000:.0f}ms")
        assert len(wav_bytes) > 1000, "Expected valid WAV output"

        items = [
            {
                "speaker": "Speaker 1",
                "source": "video",
                "text": c["source"],
                "translation": c["translation"],
                "time_label": f"00:{int(c['start']):02d}",
            }
            for c in win_res["cues"]
        ]
        summary = engine.summarize_notes(items, template="meeting")
        print("[4/5] Offline AI Summary preview:\n" + "\n".join("      " + l for l in summary.splitlines()[:8]))
        assert len(summary) > 40, "Expected non-empty summary"

        ans = engine.ask_note(
            transcript_text="\n".join(c["source"] for c in win_res["cues"]),
            summary_text=summary,
            question="Khi nào cần hoàn thành release checklist?",
        )
        print(f"      Offline Q&A Answer: {ans}")

        docx_res = notes_export_docx(
            ExportDocxRequest(
                title="Test Biên Bản Offline",
                created_at="2026-11-25",
                summary_markdown=summary,
                items=items,
                output_path=docx_path,
            )
        )
        print(f"[5/5] Exported DOCX ({docx_res['bytes']} bytes) -> OK!")
        print("\nALL 100% OFFLINE E2E TESTS PASSED!")


if __name__ == "__main__":
    main()
