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
import subprocess
import tempfile
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

from dsp import AcousticEchoSuppressor, PitchGenderTracker, SAMPLE_RATE, StreamingVAD, TextEchoFilter
from models import OfflineModelEngine, detect_script_lang

engine = OfflineModelEngine()


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
    for candidate in ("/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg", "ffmpeg"):
        if Path(candidate).exists() or candidate == "ffmpeg":
            return candidate
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

    proc = subprocess.run(cmd, capture_output=True, check=False, timeout=120)
    if proc.returncode != 0 and not proc.stdout:
        err_msg = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"Không giải mã được âm thanh từ media: {err_msg[:240]}")
    return np.frombuffer(proc.stdout, dtype="<f4").copy()


def probe_media_duration(media_path: str) -> float:
    ffprobe = find_ffmpeg().replace("ffmpeg", "ffprobe")
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
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=15)
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
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(raw_bytes)
        tmp_path = tmp.name

    try:
        pcm = extract_pcm16k_from_media(tmp_path)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

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
    }

    vads = {
        "system": StreamingVAD(trailing_silence_ms=250, soft_split_sec=1.9, max_segment_sec=3.5),
        "microphone": StreamingVAD(trailing_silence_ms=260, soft_split_sec=2.0, max_segment_sec=3.6),
    }
    text_echo = TextEchoFilter()
    pitch_tracker = PitchGenderTracker()
    context_window: List[str] = []
    job_q: asyncio.Queue[Optional[Tuple[np.ndarray, float, str]]] = asyncio.Queue(maxsize=12)

    def emit(msg: Dict[str, Any]) -> None:
        loop.call_soon_threadsafe(send_q.put_nowait, msg)

    def on_clean_pcm(pcm: np.ndarray, ts: float, source: str) -> None:
        vad = vads.get(source)
        if not vad:
            return
        for seg_pcm, dur in vad.feed(pcm):
            pitch_tracker.observe(source, seg_pcm)
            started_at = ts - dur
            try:
                job_q.put_nowait((seg_pcm, started_at, source))
            except asyncio.QueueFull:
                # Drop oldest segment to preserve real-time live edge!
                try:
                    job_q.get_nowait()
                    job_q.put_nowait((seg_pcm, started_at, source))
                except Exception:
                    pass

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

    async def inference_loop() -> None:
        while True:
            job = await job_q.get()
            if job is None:
                break
            seg_pcm, started_at, source = job
            utt_id = f"{source}-{uuid.uuid4().hex[:8]}"
            try:
                asr = await asyncio.to_thread(
                    engine.transcribe, seg_pcm, session_cfg["language"], False
                )
                text = asr["text"].strip()
                if not text:
                    continue
                if source == "system":
                    text_echo.record_system(text)
                elif text_echo.is_mic_echo(text):
                    continue

                lang = asr["language"]
                speaker = "Âm thanh máy" if source == "system" else "Tôi (Micro)"
                emit({
                    "type": "transcript",
                    "id": utt_id,
                    "source": source,
                    "speaker": speaker,
                    "language": lang,
                    "text": text,
                    "started_at": round(started_at, 2),
                    "duration_sec": round(seg_pcm.size / SAMPLE_RATE, 2),
                    "asr_ms": asr["asr_ms"],
                })

                if session_cfg["translate"] and lang != "vi" and detect_script_lang(text) != "vi":
                    t_trans = time.monotonic()
                    vi_text = await asyncio.to_thread(
                        engine.translate_utterance,
                        text,
                        lang,
                        list(context_window[-3:]),
                        session_cfg["style"],
                    )
                    trans_ms = round((time.monotonic() - t_trans) * 1000, 1)
                    if vi_text:
                        emit({
                            "type": "translation",
                            "id": utt_id,
                            "source": source,
                            "speaker": speaker,
                            "text": vi_text,
                            "translate_ms": trans_ms,
                        })
                        if session_cfg["speak_vi"]:
                            chosen_voice = (
                                pitch_tracker.preferred_voice(source)
                                if session_cfg["tts_voice"] == "auto"
                                else session_cfg["tts_voice"]
                            )
                            # Split long Vietnamese sentence at first clause (>=4 words) so first audio plays in ~280ms!
                            clauses = [vi_text]
                            for sep in (", ", "; ", ". ", "! ", "? "):
                                if sep in vi_text:
                                    idx_sep = vi_text.find(sep)
                                    left = vi_text[: idx_sep + 1].strip()
                                    right = vi_text[idx_sep + len(sep) :].strip()
                                    if len(left.split()) >= 4 and len(right.split()) >= 2:
                                        clauses = [left, right]
                                        break
                            for c_idx, clause_txt in enumerate(clauses):
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

                context_window.append(text)
                if len(context_window) > 10:
                    del context_window[:-10]
            except Exception as exc:
                emit({"type": "error", "message": str(exc)})

    sender_task = asyncio.create_task(sender_loop())
    infer_task = asyncio.create_task(inference_loop())

    emit({"type": "connected", "status": engine.status()})

    try:
        while True:
            raw_msg = await ws.receive_text()
            data = json.loads(raw_msg)
            mtype = data.get("type")
            if mtype == "config":
                for k in ("language", "style", "translate", "speak_vi", "tts_voice"):
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
                        seg_pcm, dur = rem
                        await job_q.put((seg_pcm, time.time() - dur, src_name))
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        await job_q.put(None)
        await infer_task
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
