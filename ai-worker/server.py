#!/usr/bin/env python3
"""
VinaNote Offline AI Engine Daemon (127.0.0.1:38479)
Exposes:
- REST endpoints for Model Status/Download, Video Rolling-Window ASR+Translation,
  YouTube/TikTok Caption Batch Translation, ZeroTTS Vietnamese Dubbing, Meeting Summary, Note Q&A, DOCX Export.
- WebSocket `/ws/live` for real-time dual-source (System Audio + Mic / Browser MediaStream)
  VAD -> Acoustic/Lexical Echo Cancellation -> MLX Whisper -> MLX Qwen2.5 Translation -> ZeroTTS Dubbing.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

from dsp import AcousticEchoSuppressor, PitchGenderTracker, SAMPLE_RATE, StreamingVAD, TextEchoFilter
from models import needs_vi_translation

# MLX needs Apple Silicon; Windows/Linux/Intel Macs run faster-whisper + llama.cpp (models_portable.py).
_MLX_DEFAULT = sys.platform == "darwin" and platform.machine() == "arm64"
if os.environ.get("VINANOTE_BACKEND", "mlx" if _MLX_DEFAULT else "portable") == "mlx":
    from models import OfflineModelEngine as ModelEngine
else:
    from models_portable import PortableModelEngine as ModelEngine

engine = ModelEngine()
# The worker runs without a console on Windows; console children (ffmpeg) would each pop up a window.
NO_WINDOW: Dict[str, Any] = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}

# Under backlog the live loop merges queued VAD segments of one source into a single ASR/MT pass
# (one encoder run instead of several), so it catches up without dropping speech.
LIVE_MERGE_MAX_SEC = 8.0
# Dubbing further behind than this many lines is stale: newer lines replace it.
LIVE_TTS_MAX_QUEUED = 2
PARTIAL_EMIT_INTERVAL_SEC = 0.08
# Interim results of the segment still being spoken (low latency over GPU economy).
INTERIM_MIN_SEC = 1.0
INTERIM_EVERY_SEC = 0.6


def split_clauses(text: str, min_words: int = 3) -> List[str]:
    """Clause-sized dubbing pieces: the first plays while the rest is still being synthesized."""
    clauses: List[str] = []
    for piece in re.split(r"(?<=[,;:.!?…])\s+", text.strip()):
        if clauses and (len(piece.split()) < min_words or len(clauses[-1].split()) < min_words):
            clauses[-1] += " " + piece
        elif piece:
            clauses.append(piece)
    return clauses


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Pre-loads Qwen2.5-7B + Frozen System-Prompt KV-Cache + Whisper + ZeroTTS into Unified Memory on boot."""
    st = engine.status()
    if st.get("ready"):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        threading.Thread(target=engine.warmup_all, daemon=True, name="vinanote-warmup").start()
    yield


app = FastAPI(title="VinaNote Offline AI Engine", version="1.1.0", lifespan=lifespan)

# Allow Tauri WebView, localhost dev, and Chrome/Edge Extension on 127.0.0.1
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def find_ffmpeg() -> str:
    for candidate in ("/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"):
        if Path(candidate).exists():
            return candidate
    if shutil.which("ffmpeg"):
        return "ffmpeg"
    try:
        import imageio_ffmpeg  # Windows: ffmpeg.exe bundled by pip (requirements-windows.txt)

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def extract_pcm16k_from_media(
    media_path: str,
    start_sec: float = 0.0,
    duration_sec: Optional[float] = None,
) -> np.ndarray:
    """Uses local FFmpeg to decode any video/audio slice directly into 16 kHz mono float32 PCM in RAM."""
    cmd = [find_ffmpeg(), "-nostdin", "-hide_banner", "-loglevel", "error"]
    if start_sec > 0:
        cmd += ["-ss", f"{start_sec:.3f}"]
    cmd += ["-i", media_path]
    if duration_sec is not None and duration_sec > 0:
        cmd += ["-t", f"{duration_sec:.3f}"]
    cmd += ["-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "pipe:1"]

    proc = subprocess.run(cmd, capture_output=True, check=False, timeout=120, **NO_WINDOW)
    if proc.returncode != 0 and not proc.stdout:
        err_msg = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"Không giải mã được âm thanh từ media: {err_msg[:240]}")
    return np.frombuffer(proc.stdout, dtype="<f4").copy()


def decode_upload_pcm16k(raw_bytes: bytes, suffix: str) -> np.ndarray:
    """FFmpeg needs a file; temp file, decode and cleanup run together off the event loop
    (Windows cannot delete a file FFmpeg still has open)."""
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(raw_bytes)
    try:
        return extract_pcm16k_from_media(tmp.name)
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


def probe_media_duration(media_path: str) -> float:
    ffmpeg = find_ffmpeg()
    ffprobe = str(Path(ffmpeg).with_name(Path(ffmpeg).name.replace("ffmpeg", "ffprobe")))
    if not shutil.which(ffprobe):  # pip-bundled ffmpeg ships without ffprobe: read "Duration:" from `ffmpeg -i`
        proc = subprocess.run([ffmpeg, "-hide_banner", "-i", media_path], capture_output=True, text=True,
                              errors="replace", check=False, timeout=15, **NO_WINDOW)
        m = re.search(r"Duration:\s*(\d+):(\d+):([\d.]+)", proc.stderr)
        return round(int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3]), 2) if m else 0.0
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        media_path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=15, **NO_WINDOW)
    try:
        return round(float(proc.stdout.strip()), 2)
    except Exception:
        return 0.0


# ---------------- Request Schemas ----------------


class VideoWindowRequest(BaseModel):
    file_path: str
    start_sec: float = 0.0
    duration_sec: float = 15.0
    language: str = "auto"
    style: str = "natural"
    context: List[str] = []


class CaptionBatchRequest(BaseModel):
    cues: List[Dict[str, Any]]
    language: str = "auto"
    style: str = "natural"
    context: List[str] = []


class UtteranceTranslateRequest(BaseModel):
    text: str
    language: str = "auto"
    style: str = "natural"
    context: List[str] = []


class TTSRequest(BaseModel):
    text: str
    voice: str = "maichi"


class SummarizeRequest(BaseModel):
    items: List[Dict[str, Any]]
    template: str = "meeting"


class AskNoteRequest(BaseModel):
    transcript_text: str
    summary_text: str = ""
    question: str
    history: List[Dict[str, str]] = []


class ExportDocxRequest(BaseModel):
    title: str
    created_at: str = ""
    summary_markdown: str = ""
    items: List[Dict[str, Any]] = []
    output_path: Optional[str] = None


# ---------------- REST Endpoints ----------------


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"ok": True, "service": "vinanote-offline-ai", "status": engine.status()}


@app.get("/models/status")
def models_status() -> Dict[str, Any]:
    return engine.status()


@app.post("/models/download")
async def models_download() -> StreamingResponse:
    loop = asyncio.get_running_loop()
    q: asyncio.Queue[Optional[Dict[str, Any]]] = asyncio.Queue()

    def progress_cb(ev: Dict[str, Any]) -> None:
        loop.call_soon_threadsafe(q.put_nowait, dict(ev))

    def worker() -> None:
        try:
            engine.download_all(on_progress=progress_cb)
        except Exception as exc:
            loop.call_soon_threadsafe(q.put_nowait, {"state": "error", "step": str(exc), "percent": 0})
        finally:
            loop.call_soon_threadsafe(q.put_nowait, None)

    threading.Thread(target=worker, daemon=True).start()

    async def event_gen():
        while True:
            item = await q.get()
            if item is None:
                break
            yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")


@app.get("/video/probe")
def video_probe(file_path: str) -> Dict[str, Any]:
    p = Path(file_path)
    if not p.is_file():
        raise HTTPException(status_code=404, detail="Không tìm thấy file video.")
    dur = probe_media_duration(str(p))
    # Check sidecar SRT if present alongside video
    srt_path = p.with_suffix(".srt")
    return {
        "name": p.name,
        "path": str(p),
        "size_bytes": p.stat().st_size,
        "duration_sec": dur,
        "sidecar_srt": str(srt_path) if srt_path.is_file() else None,
    }


@app.post("/video/translate-window")
def video_translate_window(req: VideoWindowRequest) -> Dict[str, Any]:
    if not Path(req.file_path).is_file():
        raise HTTPException(status_code=404, detail=f"File không tồn tại: {req.file_path}")

    pcm = extract_pcm16k_from_media(req.file_path, start_sec=req.start_sec, duration_sec=req.duration_sec)
    if pcm.size < SAMPLE_RATE // 4:
        return {"start_sec": req.start_sec, "end_sec": req.start_sec + req.duration_sec, "language": req.language, "cues": []}

    asr = engine.transcribe(pcm, language=req.language, word_timestamps=True)
    raw_cues = engine.group_segments_into_cues(asr["segments"], asr["words"])
    # Shift timestamps by window start_sec
    for c in raw_cues:
        c["start"] = round(req.start_sec + float(c["start"]), 2)
        c["end"] = round(req.start_sec + float(c["end"]), 2)

    translated_cues = engine.translate_caption_batch(
        raw_cues,
        source_lang=asr["language"],
        context=req.context,
        style=req.style,
    )
    return {
        "start_sec": req.start_sec,
        "end_sec": round(req.start_sec + (pcm.size / SAMPLE_RATE), 2),
        "language": asr["language"],
        "asr_ms": asr["asr_ms"],
        "cues": translated_cues,
    }


@app.post("/browser/translate-audio-chunk")
async def browser_translate_audio_chunk(
    audio: UploadFile = File(...),
    offset_sec: float = Form(0.0),
    language: str = Form("auto"),
    style: str = Form("natural"),
    context_json: str = Form("[]"),
    tts_voice: str = Form(""),
) -> Dict[str, Any]:
    """Used by the Chrome/Edge Extension for TikTok, Douyin, Bilibili, Facebook videos without captions!"""
    raw_bytes = await audio.read()
    if not raw_bytes:
        return {"cues": [], "language": language}

    suffix = Path(audio.filename or "chunk.webm").suffix or ".webm"
    pcm = await asyncio.to_thread(decode_upload_pcm16k, raw_bytes, suffix)

    if pcm.size < SAMPLE_RATE // 3:
        return {"cues": [], "language": language}

    # Quick energy check to skip pure silence chunks instantly
    rms = float(np.sqrt(np.mean(pcm * pcm)))
    if rms < 0.0025:
        return {"cues": [], "language": language}

    try:
        ctx = json.loads(context_json) if context_json else []
    except Exception:
        ctx = []

    asr = await asyncio.to_thread(engine.transcribe, pcm, language, True)
    if not asr["text"].strip():
        return {"cues": [], "language": asr["language"]}

    raw_cues = engine.group_segments_into_cues(asr["segments"], asr["words"])
    for c in raw_cues:
        c["start"] = round(offset_sec + float(c["start"]), 2)
        c["end"] = round(offset_sec + float(c["end"]), 2)

    translated = await asyncio.to_thread(
        engine.translate_caption_batch, raw_cues, asr["language"], ctx, style
    )

    if tts_voice:
        for cue in translated:
            vi_text = (cue.get("translation") or "").strip()
            if vi_text:
                wav_bytes, sr, dur = await asyncio.to_thread(engine.synthesize_wav_bytes, vi_text, tts_voice)
                if wav_bytes:
                    cue["wav_b64"] = base64.b64encode(wav_bytes).decode("ascii")
                    cue["tts_duration"] = round(dur, 2)

    return {"cues": translated, "language": asr["language"], "asr_ms": asr["asr_ms"]}


@app.post("/captions/translate")
async def captions_translate(req: CaptionBatchRequest) -> Dict[str, Any]:
    out = await asyncio.to_thread(
        engine.translate_caption_batch, req.cues, req.language, req.context, req.style
    )
    return {"cues": out}


@app.get("/youtube/captions")
def youtube_captions_fallback(url: str, lang: str = "auto") -> Dict[str, Any]:
    """Backup YouTube subtitle extractor via yt-dlp when DOM timedtext is not yet available."""
    try:
        import urllib.request
        import yt_dlp

        ydl_opts = {
            "skip_download": True,
            "quiet": True,
            "no_warnings": True,
            "extract_flat": False,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
        subs = info.get("subtitles") or {}
        auto_subs = info.get("automatic_captions") or {}

        chosen_formats = None
        chosen_lang = "en"
        is_auto = False

        if lang != "auto" and lang in subs:
            chosen_formats, chosen_lang, is_auto = subs[lang], lang, False
        elif "vi" in subs:
            chosen_formats, chosen_lang, is_auto = subs["vi"], "vi", False
        elif subs:
            first_k = next(iter(subs.keys()))
            chosen_formats, chosen_lang, is_auto = subs[first_k], first_k, False
        elif lang != "auto" and lang in auto_subs:
            chosen_formats, chosen_lang, is_auto = auto_subs[lang], lang, True
        elif "en" in auto_subs:
            chosen_formats, chosen_lang, is_auto = auto_subs["en"], "en", True
        elif auto_subs:
            first_k = next(iter(auto_subs.keys()))
            chosen_formats, chosen_lang, is_auto = auto_subs[first_k], first_k, True

        if not chosen_formats:
            return {"ok": False, "reason": "no_captions", "cues": []}

        json3_fmt = next((f for f in chosen_formats if f.get("ext") == "json3"), chosen_formats[0])
        sub_url = json3_fmt.get("url")
        if not sub_url:
            return {"ok": False, "reason": "no_url", "cues": []}

        raw = urllib.request.urlopen(sub_url, timeout=10).read().decode("utf-8", errors="ignore")
        data = json.loads(raw)
        events = data.get("events") or []

        words = []
        raw_cues = []
        has_offsets = any(
            ev.get("segs") and any(isinstance(s.get("tOffsetMs"), (int, float)) and s.get("tOffsetMs", 0) > 0 for s in ev["segs"])
            for ev in events
        )
        if has_offsets:
            for ev in events:
                if not ev.get("segs") or "tStartMs" not in ev:
                    continue
                base = float(ev["tStartMs"])
                for s in ev["segs"]:
                    w = (s.get("utf8") or "").replace("\n", " ").strip()
                    if w:
                        t_s = (base + float(s.get("tOffsetMs") or 0)) / 1000.0
                        words.append({"word": w, "start": round(t_s, 2), "end": round(t_s + 0.35, 2)})
            cues = engine.group_segments_into_cues([], words)
        else:
            for ev in events:
                if not ev.get("segs") or "tStartMs" not in ev:
                    continue
                txt = "".join(s.get("utf8") or "" for s in ev["segs"]).replace("\n", " ").strip()
                if txt:
                    st = float(ev["tStartMs"]) / 1000.0
                    du = float(ev.get("dDurationMs") or 2500) / 1000.0
                    raw_cues.append({"start": st, "end": st + du, "text": txt})
            cues = []
            for idx, c in enumerate(raw_cues):
                nxt = raw_cues[idx + 1]["start"] if idx + 1 < len(raw_cues) else c["end"]
                cues.append({
                    "id": idx,
                    "start": round(c["start"], 2),
                    "end": round(max(c["start"] + 0.4, min(c["end"], nxt)), 2),
                    "text": c["text"],
                })
        return {"ok": len(cues) > 0, "language": chosen_lang, "isAuto": is_auto, "cues": cues}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "cues": []}


@app.post("/translate/utterance")
async def translate_utterance(req: UtteranceTranslateRequest) -> Dict[str, Any]:
    vi = await asyncio.to_thread(
        engine.translate_utterance, req.text, req.language, req.context, req.style
    )
    return {"translation": vi}


@app.post("/tts/speak")
async def tts_speak(req: TTSRequest) -> Response:
    wav_bytes, sr, dur = await asyncio.to_thread(engine.synthesize_wav_bytes, req.text, req.voice)
    if not wav_bytes:
        raise HTTPException(status_code=400, detail="Không có nội dung văn bản để đọc.")
    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={"X-Audio-Duration": f"{dur:.3f}", "X-Sample-Rate": str(sr)},
    )


@app.post("/tts/speak-json")
async def tts_speak_json(req: TTSRequest) -> Dict[str, Any]:
    wav_bytes, sr, dur = await asyncio.to_thread(engine.synthesize_wav_bytes, req.text, req.voice)
    return {
        "wav_b64": base64.b64encode(wav_bytes).decode("ascii") if wav_bytes else "",
        "sample_rate": sr,
        "duration_sec": round(dur, 3),
    }


@app.post("/notes/summarize")
async def notes_summarize(req: SummarizeRequest) -> Dict[str, Any]:
    md = await asyncio.to_thread(engine.summarize_notes, req.items, req.template)
    return {"summary": md}


@app.post("/notes/ask")
async def notes_ask(req: AskNoteRequest) -> Dict[str, Any]:
    answer = await asyncio.to_thread(
        engine.ask_note, req.transcript_text, req.summary_text, req.question, req.history
    )
    return {"answer": answer}


@app.post("/notes/export-docx")
def notes_export_docx(req: ExportDocxRequest) -> Dict[str, Any]:
    from docx import Document

    doc = Document()
    doc.add_heading(req.title or "Biên bản Ghi chú — VinaNote Offline", level=0)
    if req.created_at:
        doc.add_paragraph(f"Thời gian: {req.created_at}")

    if req.summary_markdown.strip():
        doc.add_heading("1. Tóm tắt AI (Offline)", level=1)
        for line in req.summary_markdown.splitlines():
            s = line.strip()
            if not s:
                continue
            if s.startswith("### "):
                doc.add_heading(s[4:], level=3)
            elif s.startswith("## "):
                doc.add_heading(s[3:], level=2)
            elif s.startswith("# "):
                doc.add_heading(s[2:], level=1)
            elif s.startswith(("- ", "* ")):
                doc.add_paragraph(s[2:], style="List Bullet")
            else:
                doc.add_paragraph(s)

    if req.items:
        doc.add_heading("2. Lời thoại & Bản dịch chi tiết", level=1)
        for item in req.items:
            spk = item.get("speaker") or ("Âm thanh máy" if item.get("source") == "system" else "Micro")
            ts_label = item.get("time_label") or ""
            src = (item.get("text") or item.get("source") or "").strip()
            vi = (item.get("translation") or "").strip()
            p = doc.add_paragraph()
            hdr = p.add_run(f"[{ts_label}] {spk}: " if ts_label else f"{spk}: ")
            hdr.bold = True
            p.add_run(src)
            if vi and vi != src:
                p2 = doc.add_paragraph()
                r2 = p2.add_run(f"   → Dịch: {vi}")
                r2.italic = True

    buf = io.BytesIO()
    doc.save(buf)
    raw = buf.getvalue()
    if req.output_path:
        out_p = Path(req.output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_bytes(raw)
        return {"saved_path": str(out_p), "bytes": len(raw)}
    return {"docx_b64": base64.b64encode(raw).decode("ascii"), "bytes": len(raw)}


# ---------------- Real-time Live Meeting & Browser Stream WebSocket ----------------


@app.websocket("/ws/live")
async def ws_live(ws: WebSocket) -> None:
    await ws.accept()
    loop = asyncio.get_running_loop()
    send_q: asyncio.Queue[Optional[Dict[str, Any]]] = asyncio.Queue()

    session_cfg: Dict[str, Any] = {
        "language": "auto",
        "style": "natural",
        "translate": True,
        "speak_vi": False,
        "tts_voice": "auto",
        "interim": True,
    }

    vads = {
        "system": StreamingVAD(trailing_silence_ms=250, soft_split_sec=1.9, max_segment_sec=3.5),
        "microphone": StreamingVAD(trailing_silence_ms=260, soft_split_sec=2.0, max_segment_sec=3.6),
    }
    text_echo = TextEchoFilter()
    pitch_tracker = PitchGenderTracker()
    context_window: List[str] = []
    # (pcm, started_at, source, segment_id, interim)
    job_q: asyncio.Queue[Optional[Tuple[np.ndarray, float, str, int, bool]]] = asyncio.Queue(maxsize=12)
    tts_q: asyncio.Queue[Optional[Tuple[str, str, str]]] = asyncio.Queue()
    session_tag = uuid.uuid4().hex[:6]  # utterance ids stay unique across sessions
    gpu_busy = [False]
    last_interim_at: Dict[str, float] = {}
    shown_interims: Dict[str, set] = {name: set() for name in vads}  # segment ids with interim text shown

    def utt_id_of(source: str, seg_id: int) -> str:
        return f"{source}-{session_tag}-{seg_id}"

    def speaker_of(source: str) -> str:
        return "Âm thanh máy" if source == "system" else "Tôi (Micro)"

    def emit(msg: Dict[str, Any]) -> None:
        loop.call_soon_threadsafe(send_q.put_nowait, msg)

    def put_job(job: Tuple[np.ndarray, float, str, int, bool]) -> None:
        try:
            job_q.put_nowait(job)
        except asyncio.QueueFull:
            # Drop oldest segment to preserve real-time live edge!
            try:
                job_q.get_nowait()
                job_q.put_nowait(job)
            except Exception:
                pass

    def on_clean_pcm(pcm: np.ndarray, ts: float, source: str) -> None:
        vad = vads.get(source)
        if not vad:
            return
        for seg_pcm, dur, seg_id in vad.feed(pcm):
            pitch_tracker.observe(source, seg_pcm)
            put_job((seg_pcm, ts - dur, source, seg_id, False))
        # Interim result of the segment still being spoken, only while the GPU loop is idle:
        # Vietnamese shows up during speech instead of after the speaker pauses.
        active = vad.active_audio() if session_cfg["interim"] else None
        now = time.monotonic()
        if (
            active is not None
            and not gpu_busy[0]
            and job_q.empty()
            and active[0].size >= INTERIM_MIN_SEC * SAMPLE_RATE
            and now - last_interim_at.get(source, 0.0) >= INTERIM_EVERY_SEC
        ):
            last_interim_at[source] = now
            put_job((active[0], ts - active[0].size / SAMPLE_RATE, source, active[1], True))

    echo_suppressor = AcousticEchoSuppressor(on_clean_pcm)

    async def sender_loop() -> None:
        try:
            while True:
                msg = await send_q.get()
                if msg is None:
                    break
                await ws.send_text(json.dumps(msg, ensure_ascii=False))
        except Exception:
            pass

    def queue_tts(item: Tuple[str, str, str]) -> None:
        while tts_q.qsize() >= LIVE_TTS_MAX_QUEUED:
            tts_q.get_nowait()
        tts_q.put_nowait(item)

    async def tts_loop() -> None:
        """ZeroTTS runs on CPU here, off the GPU loop: it used to block ASR+MT of the next segment
        for 1-2 s per line, so the live pipeline fell further behind with dubbing on."""
        while True:
            item = await tts_q.get()
            if item is None:
                break
            utt_id, vi_text, source = item
            try:
                chosen_voice = (
                    pitch_tracker.preferred_voice(source)
                    if session_cfg["tts_voice"] == "auto"
                    else session_cfg["tts_voice"]
                )
                for c_idx, clause_txt in enumerate(split_clauses(vi_text)):
                    wav_bytes, sr, dur = await asyncio.to_thread(
                        engine.synthesize_wav_bytes, clause_txt, chosen_voice
                    )
                    if wav_bytes:
                        emit({
                            "type": "tts_audio",
                            "id": f"{utt_id}-{c_idx}",
                            "voice": chosen_voice,
                            "duration_sec": round(dur, 2),
                            "wav_b64": base64.b64encode(wav_bytes).decode("ascii"),
                        })
            except Exception as exc:
                emit({"type": "error", "message": str(exc)})

    def retract(source: str, seg_ids: List[int]) -> None:
        for seg_id in seg_ids:
            if seg_id in shown_interims[source]:
                shown_interims[source].discard(seg_id)
                emit({"type": "retract", "id": utt_id_of(source, seg_id), "source": source})

    async def translate(utt_id: str, text: str, lang: str, source: str, should_stop=None) -> str:
        last_partial = [0.0]

        def on_partial(partial: str) -> None:
            now = time.monotonic()
            if now - last_partial[0] >= PARTIAL_EMIT_INTERVAL_SEC:
                last_partial[0] = now
                emit({"type": "translation_partial", "id": utt_id, "source": source,
                      "speaker": speaker_of(source), "text": partial})

        return await asyncio.to_thread(
            engine.translate_utterance,
            text,
            lang,
            list(context_window[-3:]),
            session_cfg["style"],
            on_partial,
            should_stop,
        )

    async def run_interim(pcm: np.ndarray, started_at: float, source: str, seg_id: int) -> None:
        asr = await asyncio.to_thread(engine.transcribe_fast, pcm, session_cfg["language"])
        text = asr["text"].strip()
        if not text or not job_q.empty() or (source != "system" and text_echo.is_mic_echo(text)):
            return
        utt_id = utt_id_of(source, seg_id)
        lang = asr["language"]
        needs_translation = bool(session_cfg["translate"]) and needs_vi_translation(text, lang)
        shown_interims[source].add(seg_id)
        emit({
            "type": "transcript",
            "interim": True,
            "id": utt_id,
            "source": source,
            "speaker": speaker_of(source),
            "language": lang,
            "text": text,
            "needs_translation": needs_translation,
            "started_at": round(started_at, 2),
            "duration_sec": round(pcm.size / SAMPLE_RATE, 2),
            "asr_ms": asr["asr_ms"],
        })
        if needs_translation:
            # Gives way the moment a final segment is queued; partials already shown stay until replaced.
            vi_text = await translate(utt_id, text, lang, source, should_stop=lambda: not job_q.empty())
            if vi_text:
                emit({"type": "translation_partial", "id": utt_id, "source": source,
                      "speaker": speaker_of(source), "text": vi_text})

    async def run_final(pcm: np.ndarray, started_at: float, source: str, seg_ids: List[int]) -> None:
        seg_id = seg_ids[0]
        # Interim lines replaced by this merged final, or whose segment never got a final.
        retract(source, [i for i in shown_interims[source] if i <= seg_ids[-1] and i != seg_id])
        asr = await asyncio.to_thread(engine.transcribe_fast, pcm, session_cfg["language"])
        text = asr["text"].strip()
        if not text or (source != "system" and text_echo.is_mic_echo(text)):
            retract(source, [seg_id])
            return
        if source == "system":
            text_echo.record_system(text)
        shown_interims[source].discard(seg_id)

        utt_id = utt_id_of(source, seg_id)
        lang = asr["language"]
        needs_translation = bool(session_cfg["translate"]) and needs_vi_translation(text, lang)
        emit({
            "type": "transcript",
            "interim": False,
            "id": utt_id,
            "source": source,
            "speaker": speaker_of(source),
            "language": lang,
            "text": text,
            "needs_translation": needs_translation,
            "started_at": round(started_at, 2),
            "duration_sec": round(pcm.size / SAMPLE_RATE, 2),
            "asr_ms": asr["asr_ms"],
        })
        if needs_translation:
            t_trans = time.monotonic()
            vi_text = await translate(utt_id, text, lang, source)
            if vi_text:
                emit({
                    "type": "translation",
                    "id": utt_id,
                    "source": source,
                    "speaker": speaker_of(source),
                    "text": vi_text,
                    "translate_ms": round((time.monotonic() - t_trans) * 1000, 1),
                })
                if session_cfg["speak_vi"]:
                    queue_tts((utt_id, vi_text, source))

        context_window.append(text)
        if len(context_window) > 10:
            del context_window[:-10]

    async def inference_loop() -> None:
        held: List[Any] = []  # a dequeued job of the other source, processed next
        while True:
            job = held.pop() if held else await job_q.get()
            if job is None:
                break
            seg_pcm, started_at, source, seg_id, interim = job
            if interim and not job_q.empty():
                continue  # newer audio is already queued
            seg_ids = [seg_id]
            while not interim and not job_q.empty() and seg_pcm.size < LIVE_MERGE_MAX_SEC * SAMPLE_RATE:
                nxt = job_q.get_nowait()
                if nxt is None or nxt[2] != source:
                    held.append(nxt)
                    break
                if not nxt[4]:  # an interim snapshot is superseded by the merged final
                    seg_pcm = np.concatenate((seg_pcm, nxt[0]))
                    seg_ids.append(nxt[3])
            gpu_busy[0] = True
            try:
                if interim:
                    await run_interim(seg_pcm, started_at, source, seg_id)
                else:
                    await run_final(seg_pcm, started_at, source, seg_ids)
            except Exception as exc:
                emit({"type": "error", "message": str(exc)})
            finally:
                gpu_busy[0] = False

    sender_task = asyncio.create_task(sender_loop())
    infer_task = asyncio.create_task(inference_loop())
    tts_task = asyncio.create_task(tts_loop())

    emit({"type": "connected", "status": engine.status()})

    try:
        while True:
            raw_msg = await ws.receive_text()
            data = json.loads(raw_msg)
            mtype = data.get("type")
            if mtype == "config":
                for k in ("language", "style", "translate", "speak_vi", "tts_voice", "interim"):
                    if k in data:
                        session_cfg[k] = data[k]
                emit({"type": "config_ack", "config": session_cfg})
            elif mtype == "audio":
                source = data.get("source", "system")
                pcm_b64 = data.get("pcm_b64", "")
                ts = float(data.get("ts") or time.time())
                if pcm_b64:
                    raw_pcm = base64.b64decode(pcm_b64)
                    pcm = np.frombuffer(raw_pcm, dtype="<f4")
                    if pcm.size > 0 and np.isfinite(pcm).all():
                        echo_suppressor.push(pcm, ts, source)
            elif mtype == "flush":
                echo_suppressor.flush()
                for src_name, vad in vads.items():
                    rem = vad.flush()
                    if rem is not None:
                        seg_pcm, dur, seg_id = rem
                        await job_q.put((seg_pcm, time.time() - dur, src_name, seg_id, False))
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        await job_q.put(None)
        await infer_task
        await tts_q.put(None)
        await tts_task
        await send_q.put(None)
        await sender_task


def main() -> None:
    parser = argparse.ArgumentParser(description="VinaNote Offline AI Engine")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=38479)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
