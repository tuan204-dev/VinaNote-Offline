import React, { useEffect, useRef, useState } from "react";
import { invoke, convertFileSrc } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { open, save } from "@tauri-apps/plugin-dialog";
import {
  Mic,
  Volume2,
  Video,
  FileText,
  Cpu,
  Play,
  Square,
  Download,
  Sparkles,
  Send,
  Trash2,
  FolderOpen,
  Globe,
  CheckCircle2,
  AlertCircle,
} from "lucide-react";

const AI_URL = "http://127.0.0.1:38479";

export interface TranscriptItem {
  id: string;
  speaker: string;
  source: string;
  text: string;
  translation?: string;
  time_label: string;
  asr_ms?: number;
  translate_ms?: number;
  needs_translation?: boolean;
  partial?: boolean;
  interim?: boolean;
}

export interface VideoCue {
  id: number;
  start: number;
  end: number;
  source: string;
  translation: string;
}

export interface SavedNote {
  id: string;
  title: string;
  kind: string;
  source_url?: string | null;
  created_at: string;
  updated_at: string;
  summary: string;
  items: TranscriptItem[];
}

function formatSec(sec: number): string {
  const s = Math.max(0, Math.floor(sec));
  const m = Math.floor(s / 60);
  const r = s % 60;
  return `${String(m).padStart(2, "0")}:${String(r).padStart(2, "0")}`;
}

export default function App() {
  const [tab, setTab] = useState<"live" | "video" | "notes" | "models">("live");
  const [engineStatus, setEngineStatus] = useState<any>(null);
  const [extPath, setExtPath] = useState<string>("/Users/tuanda/Projects/VinaNote-Offline/browser-extension");
  const [perms, setPerms] = useState<{ microphone?: string; system_audio?: string }>({});

  // Live Meeting State
  const [isCapturing, setIsCapturing] = useState(false);
  const [useSystemAudio, setUseSystemAudio] = useState(true);
  const [useMic, setUseMic] = useState(true);
  const [liveLang, setLiveLang] = useState("auto");
  const [liveStyle, setLiveStyle] = useState("natural");
  const [liveSpeakVi, setLiveSpeakVi] = useState(true);
  const [liveVoice, setLiveVoice] = useState("auto");
  const [sysLevel, setSysLevel] = useState(0);
  const [micLevel, setMicLevel] = useState(0);
  const [liveItems, setLiveItems] = useState<TranscriptItem[]>([]);
  const [liveBanner, setLiveBanner] = useState("");
  const ttsQueueRef = useRef<string[]>([]);
  const ttsPlayingRef = useRef(false);

  // Local Video State
  const [videoPath, setVideoPath] = useState<string>("");
  const [videoSrc, setVideoSrc] = useState<string>("");
  const [videoTitle, setVideoTitle] = useState<string>("");
  const [videoDuration, setVideoDuration] = useState<number>(0);
  const [videoCues, setVideoCues] = useState<VideoCue[]>([]);
  const [currentTime, setCurrentTime] = useState<number>(0);
  const [videoLang, setVideoLang] = useState<string>("auto");
  const [videoDub, setVideoDub] = useState<boolean>(true);
  const [videoVoice, setVideoVoice] = useState<string>("maichi");
  const [videoBusyMsg, setVideoBusyMsg] = useState<string>("");
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const processedToRef = useRef<number>(0);
  const fetchingWindowRef = useRef<boolean>(false);
  const spokenCueIdsRef = useRef<Set<string>>(new Set());
  const dubCacheRef = useRef<Map<string, { url: string; dur: number }>>(new Map());
  const dubAudioRef = useRef<HTMLAudioElement | null>(null);

  // Notes State
  const [notes, setNotes] = useState<SavedNote[]>([]);
  const [selectedNote, setSelectedNote] = useState<SavedNote | null>(null);
  const [summaryTpl, setSummaryTpl] = useState("meeting");
  const [isSummarizing, setIsSummarizing] = useState(false);
  const [chatQuestion, setChatQuestion] = useState("");
  const [chatHistory, setChatHistory] = useState<{ q: string; a: string }[]>([]);
  const [isAsking, setIsAsking] = useState(false);

  // Model Download State
  const [dlProgress, setDlProgress] = useState<any>(null);

  const playNextTts = () => {
    if (ttsPlayingRef.current || ttsQueueRef.current.length === 0) return;
    const b64 = ttsQueueRef.current.shift()!;
    ttsPlayingRef.current = true;
    const audio = new Audio(`data:audio/wav;base64,${b64}`);
    audio.playbackRate = 1.12;
    audio.onended = () => {
      ttsPlayingRef.current = false;
      playNextTts();
    };
    audio.onerror = () => {
      ttsPlayingRef.current = false;
      playNextTts();
    };
    audio.play().catch(() => {
      ttsPlayingRef.current = false;
    });
  };

  const refreshStatus = async () => {
    try {
      const st = await invoke<any>("ensure_ai_worker");
      setEngineStatus(st);
    } catch {
      try {
        const r = await fetch(`${AI_URL}/models/status`);
        if (r.ok) setEngineStatus(await r.json());
      } catch {}
    }
    try {
      const p = await invoke<any>("check_audio_permissions");
      setPerms(p || {});
    } catch {}
  };

  const loadNotesList = async () => {
    try {
      const list = await invoke<SavedNote[]>("list_saved_notes");
      setNotes(list);
      if (list.length > 0 && !selectedNote) {
        setSelectedNote(list[0]);
      }
    } catch {}
  };

  useEffect(() => {
    refreshStatus();
    loadNotesList();
    invoke<string>("get_extension_path").then(setExtPath).catch(() => {});

    const unlistenAi = listen<any>("live-ai-event", (ev) => {
      const msg = ev.payload;
      if (!msg) return;
      if (msg.type === "transcript") {
        const timeLabel = new Date().toLocaleTimeString("vi-VN", { hour12: false });
        setLiveItems((prev) => {
          if (prev.some((it) => it.id === msg.id)) {
            return prev.map((it) =>
              it.id === msg.id
                ? {
                    ...it,
                    text: msg.text,
                    needs_translation: msg.needs_translation,
                    asr_ms: msg.asr_ms,
                    interim: msg.interim,
                  }
                : it
            );
          }
          const entry: TranscriptItem = {
            id: msg.id,
            speaker: msg.speaker || "Người nói",
            source: msg.source || "system",
            text: msg.text,
            time_label: timeLabel,
            asr_ms: msg.asr_ms,
            needs_translation: msg.needs_translation,
            interim: msg.interim,
          };
          return [...prev, entry];
        });
      } else if (msg.type === "retract") {
        setLiveItems((prev) => prev.filter((it) => it.id !== msg.id));
      } else if (msg.type === "translation_partial") {
        setLiveItems((prev) =>
          prev.map((it) => {
            if (it.id !== msg.id) return it;
            if (it.translation !== undefined && it.partial !== true) return it;
            return { ...it, translation: msg.text, partial: true };
          })
        );
      } else if (msg.type === "translation") {
        setLiveItems((prev) =>
          prev.map((it) =>
            it.id === msg.id
              ? { ...it, translation: msg.text, translate_ms: msg.translate_ms, partial: false }
              : it
          )
        );
      } else if (msg.type === "tts_audio" && msg.wav_b64) {
        if (ttsQueueRef.current.length > 4) ttsQueueRef.current.shift();
        ttsQueueRef.current.push(msg.wav_b64);
        playNextTts();
      } else if (msg.type === "error") {
        setLiveBanner(`Lỗi xử lý: ${msg.message}`);
      }
    });

    const unlistenLvl = listen<any>("audio-level", (ev) => {
      const { source, rms } = ev.payload || {};
      const pct = Math.min(100, Math.round(Math.sqrt(rms || 0) * 280));
      if (source === "system") setSysLevel(pct);
      else setMicLevel(pct);
    });

    const unlistenCap = listen<any>("capture-status", (ev) => {
      const p = ev.payload || {};
      if (p.type === "error") setLiveBanner(p.message || "Lỗi thu âm");
    });

    const unlistenBrowserNote = listen<SavedNote>("browser-note-saved", (ev) => {
      loadNotesList();
      setSelectedNote(ev.payload);
      setTab("notes");
    });

    return () => {
      unlistenAi.then((f) => f());
      unlistenLvl.then((f) => f());
      unlistenCap.then((f) => f());
      unlistenBrowserNote.then((f) => f());
    };
  }, []);

  // ---------------- Live Meeting Handlers ----------------

  const toggleLiveCapture = async () => {
    if (isCapturing) {
      await invoke("stop_live_capture");
      setIsCapturing(false);
      setSysLevel(0);
      setMicLevel(0);
      setLiveBanner("Đã dừng thu âm.");
      return;
    }
    if (!engineStatus?.ready) {
      setTab("models");
      return;
    }
    setLiveBanner("Đang khởi tạo luồng thu âm Offline...");
    try {
      await invoke("start_live_capture", {
        args: {
          system_audio: useSystemAudio,
          microphone: useMic,
          language: liveLang,
          style: liveStyle,
          translate: true,
          speak_vi: liveSpeakVi,
          tts_voice: liveVoice,
        },
      });
      setIsCapturing(true);
      setLiveBanner("● Đang nghe & dịch trực tiếp 100% trên máy (Offline)");
    } catch (e: any) {
      setLiveBanner(String(e));
    }
  };

  const saveLiveToNotes = async (autoSummarize: boolean) => {
    if (liveItems.length === 0) return;
    if (isCapturing) {
      await invoke("stop_live_capture");
      setIsCapturing(false);
    }
    const now = new Date().toLocaleString("vi-VN");
    const newNote: SavedNote = {
      id: crypto.randomUUID(),
      title: `Cuộc họp ngày ${now}`,
      kind: "meeting",
      source_url: null,
      created_at: now,
      updated_at: now,
      summary: "",
      items: liveItems,
    };
    await invoke("save_note_disk", { note: newNote });
    await loadNotesList();
    setSelectedNote(newNote);
    setTab("notes");
    if (autoSummarize) {
      await runSummarizeNote(newNote, "meeting");
    }
  };

  // ---------------- Local Video Rolling-Window Engine ----------------

  const fetchNextVideoWindow = async (filePath: string, fromSec: number, durSec: number, totalDur: number) => {
    if (fetchingWindowRef.current || fromSec >= totalDur - 0.5) return;
    fetchingWindowRef.current = true;
    const sliceLen = Math.min(durSec, totalDur - fromSec);
    setVideoBusyMsg(`Đang nhận dạng & dịch đoạn ${formatSec(fromSec)} – ${formatSec(fromSec + sliceLen)} (MLX Offline)...`);
    try {
      const ctx = videoCues.slice(-3).map((c) => c.source);
      const resp = await fetch(`${AI_URL}/video/translate-window`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          file_path: filePath,
          start_sec: fromSec,
          duration_sec: sliceLen,
          language: videoLang,
          style: "natural",
          context: ctx,
        }),
      });
      if (resp.ok) {
        const data = await resp.json();
        processedToRef.current = Math.max(processedToRef.current, data.end_sec || fromSec + sliceLen);
        const incoming: VideoCue[] = data.cues || [];
        setVideoCues((prev) => {
          const filtered = prev.filter((p) => p.end <= fromSec + 0.2 || p.start >= (data.end_sec || fromSec + sliceLen));
          return [...filtered, ...incoming].sort((a, b) => a.start - b.start);
        });

        // Pre-synthesize ZeroTTS audio for upcoming cues if dubbing is enabled
        if (videoDub) {
          for (const cue of incoming.slice(0, 4)) {
            prefetchCueDub(cue, videoVoice);
          }
        }
      }
    } catch (e: any) {
      setVideoBusyMsg(`Lỗi dịch đoạn video: ${e}`);
    } finally {
      fetchingWindowRef.current = false;
      setVideoBusyMsg("");
    }
  };

  const prefetchCueDub = async (cue: VideoCue, voice: string) => {
    const key = `${cue.start.toFixed(2)}|${voice}|${cue.translation}`;
    if (!cue.translation.trim() || dubCacheRef.current.has(key)) return;
    try {
      const r = await fetch(`${AI_URL}/tts/speak`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: cue.translation, voice }),
      });
      if (r.ok) {
        const dur = Number(r.headers.get("X-Audio-Duration") || "2.0");
        const blob = await r.blob();
        const url = URL.createObjectURL(blob);
        dubCacheRef.current.set(key, { url, dur });
      }
    } catch {}
  };

  const pickAndOpenVideo = async () => {
    const selected = await open({
      multiple: false,
      filters: [{ name: "Video & Audio", extensions: ["mp4", "mkv", "mov", "webm", "m4v", "mp3", "wav"] }],
    });
    if (!selected || typeof selected !== "string") return;

    setVideoPath(selected);
    setVideoSrc(convertFileSrc(selected));
    setVideoCues([]);
    processedToRef.current = 0;
    spokenCueIdsRef.current.clear();

    try {
      const r = await fetch(`${AI_URL}/video/probe?file_path=${encodeURIComponent(selected)}`);
      const info = await r.json();
      setVideoTitle(info.name || "Video cục bộ");
      const dur = info.duration_sec || 3600;
      setVideoDuration(dur);
      // Fast 12-second cold-start window!
      fetchNextVideoWindow(selected, 0, 12, dur);
    } catch {
      setVideoTitle(selected.split("/").pop() || "Video");
      setVideoDuration(3600);
      fetchNextVideoWindow(selected, 0, 12, 3600);
    }
  };

  const onVideoTimeUpdate = () => {
    const v = videoRef.current;
    if (!v) return;
    const t = v.currentTime;
    setCurrentTime(t);

    // Rolling 45s lookahead when playhead approaches within 20s of processed horizon, or if user seeked ahead!
    if (videoPath && videoDuration > 0 && !fetchingWindowRef.current) {
      if (t > processedToRef.current + 1.5) {
        processedToRef.current = t;
        spokenCueIdsRef.current.clear();
        fetchNextVideoWindow(videoPath, t, 14, videoDuration);
      } else if (t >= processedToRef.current - 20 && processedToRef.current < videoDuration - 0.5) {
        fetchNextVideoWindow(videoPath, processedToRef.current, 45, videoDuration);
      }
    }

    // Pre-synthesize next 3 cues ahead of playhead
    if (videoDub) {
      const upcoming = videoCues.filter((c) => c.start >= t && c.start <= t + 28).slice(0, 3);
      for (const c of upcoming) prefetchCueDub(c, videoVoice);
    }

    // Trigger time-stretched dubbing + audio ducking!
    if (videoDub && !v.paused) {
      const idx = videoCues.findIndex((c) => t >= c.start - 0.05 && t < c.end - 0.25);
      if (idx >= 0) {
        const active = videoCues[idx];
        const key = `${active.start.toFixed(2)}|${videoVoice}|${active.translation}`;
        const cached = dubCacheRef.current.get(key);
        if (cached && !spokenCueIdsRef.current.has(key)) {
          spokenCueIdsRef.current.add(key);
          const nextStart = videoCues[idx + 1]?.start ?? active.end + 1.2;
          const slot = Math.max(0.5, nextStart - t - 0.15);
          const rate = Math.min(1.32, Math.max(1.0, cached.dur / slot));

          if (dubAudioRef.current) {
            dubAudioRef.current.pause();
          }
          v.volume = 0.22; // Duck original video audio while Vietnamese voice-over speaks
          const dubAudio = new Audio(cached.url);
          dubAudio.playbackRate = rate;
          dubAudio.onended = () => {
            if (videoRef.current) videoRef.current.volume = 1.0;
          };
          dubAudio.onerror = () => {
            if (videoRef.current) videoRef.current.volume = 1.0;
          };
          dubAudioRef.current = dubAudio;
          dubAudio.play().catch(() => {
            if (videoRef.current) videoRef.current.volume = 1.0;
          });
        }
      }
    }
  };

  const activeCue = videoCues.find((c) => currentTime >= c.start && currentTime <= c.end + 0.25);

  const saveVideoToNotes = async () => {
    if (videoCues.length === 0) return;
    const now = new Date().toLocaleString("vi-VN");
    const note: SavedNote = {
      id: crypto.randomUUID(),
      title: videoTitle || `Video ${now}`,
      kind: "video",
      source_url: videoPath,
      created_at: now,
      updated_at: now,
      summary: "",
      items: videoCues.map((c, i) => ({
        id: `cue-${i}`,
        speaker: "Video",
        source: "video",
        text: c.source,
        translation: c.translation,
        time_label: formatSec(c.start),
      })),
    };
    await invoke("save_note_disk", { note });
    await loadNotesList();
    setSelectedNote(note);
    setTab("notes");
    await runSummarizeNote(note, "video");
  };

  // ---------------- Notes & AI Summary / Q&A / Export ----------------

  const runSummarizeNote = async (note: SavedNote, tpl: string) => {
    setIsSummarizing(true);
    try {
      const r = await fetch(`${AI_URL}/notes/summarize`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ items: note.items, template: tpl }),
      });
      const data = await r.json();
      const updated: SavedNote = {
        ...note,
        summary: data.summary || "",
        updated_at: new Date().toLocaleString("vi-VN"),
      };
      await invoke("save_note_disk", { note: updated });
      setSelectedNote(updated);
      await loadNotesList();
    } finally {
      setIsSummarizing(false);
    }
  };

  const askSelectedNote = async () => {
    if (!selectedNote || !chatQuestion.trim() || isAsking) return;
    const q = chatQuestion.trim();
    setChatQuestion("");
    setIsAsking(true);
    try {
      const fullTranscript = selectedNote.items
        .map((i) => `[${i.time_label}] ${i.speaker}: ${i.text}${i.translation ? ` -> ${i.translation}` : ""}`)
        .join("\n");
      const r = await fetch(`${AI_URL}/notes/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          transcript_text: fullTranscript,
          summary_text: selectedNote.summary,
          question: q,
          history: chatHistory,
        }),
      });
      const data = await r.json();
      setChatHistory((prev) => [...prev, { q, a: data.answer || "Không có câu trả lời." }]);
    } finally {
      setIsAsking(false);
    }
  };

  const exportDocx = async () => {
    if (!selectedNote) return;
    const outPath = await save({
      defaultPath: `${selectedNote.title.replace(/[/\\?%*:|"<>]/g, "-")}.docx`,
      filters: [{ name: "Word Document", extensions: ["docx"] }],
    });
    if (!outPath) return;
    await fetch(`${AI_URL}/notes/export-docx`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        title: selectedNote.title,
        created_at: selectedNote.created_at,
        summary_markdown: selectedNote.summary,
        items: selectedNote.items,
        output_path: outPath,
      }),
    });
  };

  const exportMarkdown = async () => {
    if (!selectedNote) return;
    const outPath = await save({
      defaultPath: `${selectedNote.title.replace(/[/\\?%*:|"<>]/g, "-")}.md`,
      filters: [{ name: "Markdown", extensions: ["md"] }],
    });
    if (!outPath) return;
    const mdLines = [
      `# ${selectedNote.title}`,
      `> Thời gian: ${selectedNote.created_at}`,
      "",
      "## 1. Tóm tắt AI (Offline)",
      selectedNote.summary || "_Chưa tạo tóm tắt._",
      "",
      "## 2. Lời thoại & Bản dịch",
      ...selectedNote.items.map(
        (i) => `- **[${i.time_label}] ${i.speaker}:** ${i.text}${i.translation ? `\n  - *Dịch:* ${i.translation}` : ""}`
      ),
    ].join("\n");
    const blob = new Blob([mdLines], { type: "text/markdown;charset=utf-8" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = outPath.split("/").pop() || "note.md";
    a.click();
  };

  const triggerDownloadModels = async () => {
    setDlProgress({ state: "downloading", step: "Đang bắt đầu tải bộ model Offline...", percent: 2 });
    const resp = await fetch(`${AI_URL}/models/download`, { method: "POST" });
    const reader = resp.body?.getReader();
    const decoder = new TextDecoder();
    if (!reader) return;
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      const chunk = decoder.decode(value, { stream: true });
      for (const line of chunk.split("\n")) {
        if (line.startsWith("data: ")) {
          try {
            const ev = JSON.parse(line.slice(6));
            setDlProgress(ev);
          } catch {}
        }
      }
    }
    await refreshStatus();
  };

  const testVoice = async (voiceName: string) => {
    const r = await fetch(`${AI_URL}/tts/speak`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text: "Xin chào, đây là giọng đọc tiếng Việt chạy hoàn toàn nội bộ trên máy Mac của bạn, không cần kết nối mạng.",
        voice: voiceName === "auto" ? "maichi" : voiceName,
      }),
    });
    if (r.ok) {
      const blob = await r.blob();
      new Audio(URL.createObjectURL(blob)).play();
    }
  };

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-badge">VN</div>
          <div>
            <h1>VinaNote Offline</h1>
            <span>100% On-Device Apple MLX</span>
          </div>
        </div>

        <button className={`nav-btn ${tab === "live" ? "active" : ""}`} onClick={() => setTab("live")}>
          <Mic size={17} /> Họp &amp; Âm thanh Máy
        </button>
        <button className={`nav-btn ${tab === "video" ? "active" : ""}`} onClick={() => setTab("video")}>
          <Video size={17} /> Xem &amp; Dịch File Video
        </button>
        <button className={`nav-btn ${tab === "notes" ? "active" : ""}`} onClick={() => setTab("notes")}>
          <FileText size={17} /> Ghi chú &amp; Tóm tắt ({notes.length})
        </button>
        <button className={`nav-btn ${tab === "models" ? "active" : ""}`} onClick={() => setTab("models")}>
          <Globe size={17} /> Trình duyệt &amp; Model AI
        </button>

        <div className="sidebar-footer">
          <div style={{ marginBottom: 8 }}>
            {engineStatus?.ready ? (
              <span className="status-pill ok">
                <CheckCircle2 size={13} /> 100% Offline Sẵn sàng
              </span>
            ) : (
              <span className="status-pill warn">
                <AlertCircle size={13} /> Đang chuẩn bị Model
              </span>
            )}
          </div>
          <div style={{ color: "var(--muted)", lineHeight: 1.45 }}>
            • ASR: Whisper Large-v3-Turbo<br />
            • Dịch/LLM: Qwen2.5-7B 4-bit<br />
            • Lồng tiếng: ZeroTTS CPU<br />
            • Không giới hạn số phút
          </div>
        </div>
      </aside>

      <main className="main-area">
        <header className="topbar">
          <h2>
            {tab === "live" && "Dịch Trực Tiếp Cuộc Họp & Âm Thanh Hệ Thống"}
            {tab === "video" && "Xem & Dịch/Thuyết Minh File Video Offline"}
            {tab === "notes" && "Thư Viện Biên Bản, Tóm Tắt & Hỏi Đáp AI"}
            {tab === "models" && "Tiện Ích YouTube/TikTok & Quản Lý Mô Hình AI Nội Bộ"}
          </h2>
          <div className="controls-row">
            <button className="btn" onClick={refreshStatus}>
              <Cpu size={14} /> Kiểm tra Engine
            </button>
          </div>
        </header>

        <div className="content-scroll">
          {tab === "live" && (
            <>
              <div className="card">
                <div className="controls-row" style={{ justifyContent: "space-between" }}>
                  <div className="controls-row">
                    <button
                      className={`btn ${isCapturing ? "btn-danger" : "btn-primary"}`}
                      onClick={toggleLiveCapture}
                    >
                      {isCapturing ? <Square size={15} /> : <Play size={15} />}
                      {isCapturing ? "Dừng Ghi & Dịch" : "Bắt đầu Nghe & Dịch"}
                    </button>

                    <label className="controls-row" style={{ fontSize: 13 }}>
                      <input
                        type="checkbox"
                        checked={useSystemAudio}
                        disabled={isCapturing}
                        onChange={(e) => setUseSystemAudio(e.target.checked)}
                      />
                      Âm thanh máy (Zoom/Meet/Browser)
                    </label>

                    <label className="controls-row" style={{ fontSize: 13 }}>
                      <input
                        type="checkbox"
                        checked={useMic}
                        disabled={isCapturing}
                        onChange={(e) => setUseMic(e.target.checked)}
                      />
                      Microphone
                    </label>
                  </div>

                  <div className="controls-row">
                    <div className="meter-wrap">
                      <span>Máy</span>
                      <div className="meter-bar">
                        <div className="meter-fill" style={{ width: `${sysLevel}%` }} />
                      </div>
                    </div>
                    <div className="meter-wrap">
                      <span>Mic</span>
                      <div className="meter-bar">
                        <div className="meter-fill" style={{ width: `${micLevel}%` }} />
                      </div>
                    </div>
                  </div>
                </div>

                <div className="controls-row" style={{ marginTop: 12 }}>
                  <select value={liveLang} onChange={(e) => setLiveLang(e.target.value)}>
                    <option value="auto">Ngôn ngữ gốc: Tự động</option>
                    <option value="en">Tiếng Anh → Tiếng Việt</option>
                    <option value="zh">Tiếng Trung → Tiếng Việt</option>
                    <option value="ja">Tiếng Nhật → Tiếng Việt</option>
                    <option value="ko">Tiếng Hàn → Tiếng Việt</option>
                    <option value="vi">Tiếng Việt (Chỉ chép lời)</option>
                  </select>

                  <select value={liveStyle} onChange={(e) => setLiveStyle(e.target.value)}>
                    <option value="natural">Cách dịch: Tự nhiên · Dễ nghe</option>
                    <option value="literal">Cách dịch: Sát nghĩa · Học ngoại ngữ</option>
                  </select>

                  <label className="controls-row" style={{ fontSize: 13 }}>
                    <input
                      type="checkbox"
                      checked={liveSpeakVi}
                      onChange={(e) => setLiveSpeakVi(e.target.checked)}
                    />
                    <Volume2 size={15} /> Đọc lồng tiếng Việt (ZeroTTS)
                  </label>

                  <select value={liveVoice} onChange={(e) => setLiveVoice(e.target.value)}>
                    <option value="auto">Giọng: Tự khớp Nam / Nữ</option>
                    <option value="maichi">Mai Chi (Nữ)</option>
                    <option value="baotrang">Bảo Trang (Nữ)</option>
                    <option value="quangminh">Quang Minh (Nam)</option>
                    <option value="tiendat">Tiến Đạt (Nam)</option>
                  </select>

                  {liveItems.length > 0 && (
                    <button className="btn btn-primary" onClick={() => saveLiveToNotes(true)}>
                      <Sparkles size={14} /> Lưu &amp; Tóm tắt Biên bản
                    </button>
                  )}
                </div>

                {liveBanner && (
                  <div style={{ marginTop: 10, fontSize: 12.5, color: "#34d399" }}>{liveBanner}</div>
                )}
              </div>

              <div className="transcript-list">
                {liveItems.length === 0 ? (
                  <div className="card" style={{ textAlign: "center", color: "var(--muted)", padding: 42 }}>
                    Bấm <b>Bắt đầu Nghe &amp; Dịch</b> để thu đồng thời âm thanh cuộc họp / trình duyệt và micro.
                    Mọi quá trình nhận dạng (Whisper Turbo), dịch thuật (Qwen2.5-7B) và đọc lồng tiếng (ZeroTTS) đều chạy 100% trên chip Apple M2 Pro của bạn.
                  </div>
                ) : (
                  liveItems.map((item) => (
                    <div key={item.id} className="utterance-card">
                      <div className="utterance-meta">
                        <span>
                          <b>{item.speaker}</b> · {item.time_label}
                        </span>
                        <span>
                          {item.asr_ms ? `ASR ${Math.round(item.asr_ms)}ms` : ""}
                          {item.translate_ms ? ` · Dịch ${Math.round(item.translate_ms)}ms` : ""}
                        </span>
                      </div>
                      {item.needs_translation !== false && (
                        <div
                          className="utterance-src"
                          style={item.interim ? { opacity: 0.6, fontStyle: "italic" } : undefined}
                        >
                          {item.text}
                        </div>
                      )}
                      {item.needs_translation === false ? (
                        <div
                          className="utterance-vi"
                          style={{
                            marginTop: 0,
                            paddingTop: 0,
                            borderTop: "none",
                            ...(item.interim ? { opacity: 0.6, fontStyle: "italic" } : undefined),
                          }}
                        >
                          {item.text}
                        </div>
                      ) : item.translation ? (
                        (item.needs_translation === true || item.partial || item.translation !== item.text) && (
                          <div
                            className="utterance-vi"
                            style={item.partial ? { opacity: 0.75 } : undefined}
                          >
                            {item.translation}
                          </div>
                        )
                      ) : item.needs_translation === true && item.partial !== false ? (
                        <div
                          className="utterance-vi"
                          style={{ color: "var(--muted)", fontWeight: 400, fontStyle: "italic" }}
                        >
                          Đang dịch…
                        </div>
                      ) : null}
                    </div>
                  ))
                )}
              </div>
            </>
          )}

          {tab === "video" && (
            <>
              <div className="card">
                <div className="controls-row" style={{ justifyContent: "space-between" }}>
                  <div className="controls-row">
                    <button className="btn btn-primary" onClick={pickAndOpenVideo}>
                      <FolderOpen size={15} /> Mở File Video / Audio (.mp4, .mkv, .mov...)
                    </button>
                    <select value={videoLang} onChange={(e) => setVideoLang(e.target.value)}>
                      <option value="auto">Ngôn ngữ video: Tự động</option>
                      <option value="en">Tiếng Anh</option>
                      <option value="zh">Tiếng Trung</option>
                      <option value="ja">Tiếng Nhật</option>
                    </select>
                    <label className="controls-row" style={{ fontSize: 13 }}>
                      <input
                        type="checkbox"
                        checked={videoDub}
                        onChange={(e) => setVideoDub(e.target.checked)}
                      />
                      Thuyết minh tiếng Việt khớp hình (Tự hạ nhỏ tiếng gốc)
                    </label>
                    <select value={videoVoice} onChange={(e) => setVideoVoice(e.target.value)}>
                      <option value="maichi">Giọng thuyết minh: Mai Chi (Nữ)</option>
                      <option value="baotrang">Bảo Trang (Nữ)</option>
                      <option value="quangminh">Quang Minh (Nam)</option>
                      <option value="tiendat">Tiến Đạt (Nam)</option>
                    </select>
                  </div>

                  {videoCues.length > 0 && (
                    <button className="btn" onClick={saveVideoToNotes}>
                      <Sparkles size={14} /> Lưu vào Ghi chú &amp; Tóm tắt
                    </button>
                  )}
                </div>
                {videoBusyMsg && (
                  <div style={{ marginTop: 8, fontSize: 12.5, color: "#fbbf24" }}>{videoBusyMsg}</div>
                )}
              </div>

              <div className="video-grid">
                <div className="video-stage">
                  {videoSrc ? (
                    <>
                      <video
                        ref={videoRef}
                        src={videoSrc}
                        controls
                        autoPlay
                        onTimeUpdate={onVideoTimeUpdate}
                      />
                      {activeCue && (
                        <div className="video-sub-overlay">
                          <div className="video-sub-vi">{activeCue.translation}</div>
                          {activeCue.source !== activeCue.translation && (
                            <div className="video-sub-src">{activeCue.source}</div>
                          )}
                        </div>
                      )}
                    </>
                  ) : (
                    <div style={{ color: "var(--muted)", textAlign: "center", padding: 28 }}>
                      Chọn một file video bất kỳ (kể cả video không có phụ đề).<br />
                      VinaNote sẽ cắt nhanh <b>12 giây đầu</b> để dịch ngay, sau đó tự động dịch cuốn chiếu <b>45 giây</b> phía trước khi bạn đang xem!
                    </div>
                  )}
                </div>

                <div className="cue-sidebar">
                  <div style={{ padding: "12px 14px", borderBottom: "1px solid var(--border)", fontWeight: 600, fontSize: 13.5 }}>
                    Phụ đề song ngữ ({videoCues.length} câu)
                  </div>
                  <div className="cue-scroll">
                    {videoCues.map((c, idx) => {
                      const isAct = activeCue?.start === c.start;
                      return (
                        <div
                          key={`${c.start}-${idx}`}
                          className={`cue-item ${isAct ? "active" : ""}`}
                          onClick={() => {
                            if (videoRef.current) videoRef.current.currentTime = c.start;
                          }}
                        >
                          <div style={{ fontSize: 11, color: "var(--muted)" }}>
                            {formatSec(c.start)} – {formatSec(c.end)}
                          </div>
                          <div style={{ fontSize: 13.5, fontWeight: 600, color: "#6ee7b7", marginTop: 2 }}>
                            {c.translation}
                          </div>
                          <div style={{ fontSize: 12, color: "#94a3b8", marginTop: 2 }}>{c.source}</div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>
            </>
          )}

          {tab === "notes" && (
            <div className="notes-layout">
              <div className="notes-list-pane">
                {notes.length === 0 ? (
                  <div style={{ color: "var(--muted)", fontSize: 13, padding: 12 }}>
                    Chưa có ghi chú nào. Hãy lưu từ mục Họp, Xem Video hoặc Tiện ích YouTube/TikTok.
                  </div>
                ) : (
                  notes.map((n) => (
                    <div
                      key={n.id}
                      className={`cue-item ${selectedNote?.id === n.id ? "active" : ""}`}
                      style={{ marginBottom: 8 }}
                      onClick={() => {
                        setSelectedNote(n);
                        setChatHistory([]);
                      }}
                    >
                      <div style={{ fontWeight: 600, fontSize: 13.5 }}>{n.title}</div>
                      <div style={{ fontSize: 11.5, color: "var(--muted)", marginTop: 3 }}>
                        {n.created_at} · {n.items.length} câu
                      </div>
                    </div>
                  ))
                )}
              </div>

              <div className="note-detail-pane">
                {selectedNote ? (
                  <>
                    <div className="controls-row" style={{ justifyContent: "space-between" }}>
                      <input
                        type="text"
                        value={selectedNote.title}
                        style={{ fontSize: 16, fontWeight: 700, width: "52%" }}
                        onChange={(e) => {
                          const updated = { ...selectedNote, title: e.target.value };
                          setSelectedNote(updated);
                          invoke("save_note_disk", { note: updated });
                        }}
                      />
                      <div className="controls-row">
                        <select value={summaryTpl} onChange={(e) => setSummaryTpl(e.target.value)}>
                          <option value="meeting">Mẫu: Biên bản họp</option>
                          <option value="lecture">Mẫu: Bài giảng / Khóa học</option>
                          <option value="video">Mẫu: Tóm tắt Video</option>
                        </select>
                        <button
                          className="btn btn-primary"
                          disabled={isSummarizing}
                          onClick={() => runSummarizeNote(selectedNote, summaryTpl)}
                        >
                          <Sparkles size={14} /> {isSummarizing ? "Đang tóm tắt..." : "Tóm tắt AI"}
                        </button>
                        <button className="btn" onClick={exportDocx}>
                          <Download size={14} /> Xuất .docx
                        </button>
                        <button className="btn" onClick={exportMarkdown}>
                          <Download size={14} /> Xuất .md
                        </button>
                        <button
                          className="btn btn-danger"
                          onClick={async () => {
                            await invoke("delete_note_disk", { id: selectedNote.id });
                            setSelectedNote(null);
                            loadNotesList();
                          }}
                        >
                          <Trash2 size={14} />
                        </button>
                      </div>
                    </div>

                    {selectedNote.summary && (
                      <div>
                        <h4 style={{ margin: "4px 0 8px" }}>Tóm tắt AI (Qwen2.5-7B Offline)</h4>
                        <div className="summary-box">{selectedNote.summary}</div>
                      </div>
                    )}

                    <div className="card" style={{ marginBottom: 0 }}>
                      <h4 style={{ margin: "0 0 8px" }}>Hỏi đáp Offline với nội dung bản ghi</h4>
                      {chatHistory.map((turn, idx) => (
                        <div key={idx} style={{ marginBottom: 10, fontSize: 13.5 }}>
                          <div style={{ color: "#93c5fd", fontWeight: 600 }}>Bạn: {turn.q}</div>
                          <div style={{ color: "var(--text)", marginTop: 3, whiteSpace: "pre-wrap" }}>
                            AI: {turn.a}
                          </div>
                        </div>
                      ))}
                      <div className="controls-row">
                        <input
                          type="text"
                          placeholder="Hỏi bất kỳ chi tiết nào trong cuộc họp / video này..."
                          style={{ flex: 1 }}
                          value={chatQuestion}
                          onChange={(e) => setChatQuestion(e.target.value)}
                          onKeyDown={(e) => e.key === "Enter" && askSelectedNote()}
                        />
                        <button className="btn btn-primary" disabled={isAsking} onClick={askSelectedNote}>
                          <Send size={14} /> {isAsking ? "Đang suy luận..." : "Hỏi AI"}
                        </button>
                      </div>
                    </div>

                    <div>
                      <h4 style={{ margin: "4px 0 8px" }}>Lời thoại chi tiết ({selectedNote.items.length})</h4>
                      <div className="transcript-list">
                        {selectedNote.items.map((it) => (
                          <div key={it.id} className="utterance-card">
                            <div className="utterance-meta">
                              <span>
                                <b>{it.speaker}</b> · {it.time_label}
                              </span>
                            </div>
                            <div className="utterance-src">{it.text}</div>
                            {it.translation && it.translation !== it.text && (
                              <div className="utterance-vi">{it.translation}</div>
                            )}
                          </div>
                        ))}
                      </div>
                    </div>
                  </>
                ) : (
                  <div style={{ color: "var(--muted)" }}>Chọn một ghi chú bên trái để xem chi tiết.</div>
                )}
              </div>
            </div>
          )}

          {tab === "models" && (
            <>
              <div className="card">
                <h3 style={{ marginTop: 0 }}>1. Tiện ích Trình duyệt (Dịch YouTube, TikTok, Douyin, Bilibili, Facebook 100% Offline)</h3>
                <p style={{ color: "var(--muted)", fontSize: 13.5, lineHeight: 1.55 }}>
                  Tiện ích sử dụng API <code>chrome.tabCapture</code> kết hợp <code>Offscreen Audio</code> để bắt trực tiếp âm thanh tab video trên mọi nền tảng (kể cả TikTok/Douyin/Reels không có phụ đề và chặn CORS CDN), truyền về AI Engine nội bộ <code>127.0.0.1:38479</code> để hiện phụ đề nổi song ngữ, tự hạ nhỏ tiếng video gốc và đọc lồng tiếng tiếng Việt!
                </p>
                <div style={{ fontSize: 13, marginBottom: 8 }}>
                  <b>Cách cài vào Chrome / Edge / Brave / Arc (chỉ mất 15 giây):</b>
                </div>
                <ol style={{ fontSize: 13, color: "#cbd5e1", lineHeight: 1.7 }}>
                  <li>Mở trình duyệt và truy cập <code>chrome://extensions</code> (hoặc <code>edge://extensions</code>).</li>
                  <li>Bật công tắc <b>Developer mode (Chế độ dành cho nhà phát triển)</b> ở góc phải trên.</li>
                  <li>Bấm <b>Load unpacked (Tải tiện ích đã giải nén)</b> và chọn thư mục dưới đây:</li>
                </ol>
                <div className="code-block">{extPath}</div>
              </div>

              <div className="card">
                <div className="controls-row" style={{ justifyContent: "space-between" }}>
                  <h3 style={{ margin: 0 }}>2. Trạng thái Bộ Mô Hình AI Nội Bộ (Apple Silicon MLX + ONNX)</h3>
                  <div className="controls-row">
                    <button className="btn" onClick={() => testVoice("maichi")}>
                      <Volume2 size={14} /> Nghe thử giọng Nữ (Mai Chi)
                    </button>
                    <button className="btn" onClick={() => testVoice("quangminh")}>
                      <Volume2 size={14} /> Nghe thử giọng Nam (Quang Minh)
                    </button>
                    {!engineStatus?.ready && (
                      <button className="btn btn-primary" onClick={triggerDownloadModels}>
                        <Download size={14} /> Tải trọn bộ Model Offline
                      </button>
                    )}
                  </div>
                </div>

                {dlProgress && (
                  <div style={{ marginTop: 12, color: "#34d399", fontSize: 13 }}>
                    {dlProgress.step} ({dlProgress.percent}%)
                  </div>
                )}

                <div className="transcript-list" style={{ marginTop: 14 }}>
                  {(["whisper", "llm", "tts"] as const).map((k) => {
                    const m = engineStatus?.models?.[k];
                    return (
                      <div key={k} className="utterance-card">
                        <div className="controls-row" style={{ justifyContent: "space-between" }}>
                          <div>
                            <b>{m?.repo || k}</b>
                            <div style={{ fontSize: 12, color: "var(--muted)", marginTop: 3 }}>
                              {m?.path || "Chưa có trong bộ nhớ đệm"}
                            </div>
                          </div>
                          <span className={`status-pill ${m?.installed ? "ok" : "warn"}`}>
                            {m?.installed ? "Đã cài đặt Offline" : "Chưa tải"}
                          </span>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              <div className="card">
                <h3 style={{ marginTop: 0 }}>3. Quyền Hệ Thống macOS</h3>
                <div className="controls-row" style={{ gap: 20, fontSize: 13.5 }}>
                  <div>
                    Microphone: <b>{perms.microphone || "Chưa kiểm tra"}</b>
                  </div>
                  <div>
                    Thu âm Hệ thống (ScreenCaptureKit): <b>{perms.system_audio || "Chưa kiểm tra"}</b>
                  </div>
                </div>
              </div>
            </>
          )}
        </div>
      </main>
    </div>
  );
}
