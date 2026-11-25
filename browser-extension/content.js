// Content Script for YouTube, TikTok, Douyin, Bilibili, Facebook, Coursera, Udemy...
// Supports TWO seamless modes:
// 1. ZERO-DELAY CAPTION MODE (Automatic whenever YouTube / HTML5 <video> has subtitles!):
//    - Extracts full timed subtitle cues immediately via page-hook.js (or yt-dlp fallback).
//    - Translates the playhead cue first, then the next 3, and prefetches +120s ahead in 4-cue batches.
//    - Pre-synthesizes ZeroTTS Vietnamese voice-over +25s ahead of playhead with dynamic 1.0x..1.35x speed & volume ducking!
//    - Zero lag: subtitles & dubbing fire at the EXACT millisecond `video.currentTime >= cue.start`!
//    - Injects a 1-click "VN Dịch" button directly into YouTube's player control bar (.ytp-right-controls)!
// 2. LIVE AUDIO STREAM MODE (Automatic fallback for TikTok, Douyin, Reels, or YouTube videos with 0 subtitles).

(() => {
  if (window.__vinanoteInjected) return;
  window.__vinanoteInjected = true;

  let overlayRoot = null;
  let viEl = null;
  let srcEl = null;
  let statusEl = null;
  let modeBadgeEl = null;
  let ytBtn = null;

  // Shared state
  let activeMode = null; // "captions" | "live-audio" | null
  let opts = { language: "auto", style: "natural", speakVi: true, voice: "maichi" };
  let historyItems = [];

  // Caption-Mode state
  let currentVideoId = "";
  let captionLanguage = "";
  let rawCues = []; // [{ id, start, end, text }]
  let translatedMap = new Map(); // id -> viText
  let inFlightBatch = false;
  let planEpoch = 0; // incremented on seek or video change to pre-empt old batches!
  let syncTimer = null;
  let lastShownCueId = -1;

  // Pre-synthesized Dubbing state (for Caption Mode)
  let dubCache = new Map(); // key -> { wavB64, durationSec }
  let dubPending = new Set();
  let spokenKeys = new Set();
  let currentDubAudio = null;
  let lastPlayhead = -1;

  function formatSec(sec) {
    const s = Math.max(0, Math.floor(sec));
    const m = Math.floor(s / 60);
    const r = s % 60;
    return `${String(m).padStart(2, "0")}:${String(r).padStart(2, "0")}`;
  }

  function findActiveVideo() {
    const vids = Array.from(document.querySelectorAll("video"));
    if (vids.length === 0) return null;
    // Pick largest visible playing video
    return (
      vids.sort((a, b) => b.clientWidth * b.clientHeight - a.clientWidth * a.clientHeight)[0] ||
      vids[0]
    );
  }

  function ensureOverlay() {
    if (overlayRoot) return;
    overlayRoot = document.createElement("div");
    overlayRoot.id = "vinanote-offline-overlay";
    overlayRoot.innerHTML = `
      <div class="vn-topbar">
        <span class="vn-badge">VinaNote Offline</span>
        <span class="vn-mode-pill"></span>
        <span class="vn-status">Sẵn sàng</span>
        <button class="vn-btn vn-save" title="Lưu phụ đề song ngữ sang ứng dụng VinaNote để tóm tắt">Lưu Ghi chú</button>
        <button class="vn-btn vn-close" title="Tắt dịch phụ đề">✕</button>
      </div>
      <div class="vn-vi">Đang chuẩn bị phụ đề tiếng Việt...</div>
      <div class="vn-src"></div>
    `;
    document.documentElement.appendChild(overlayRoot);

    viEl = overlayRoot.querySelector(".vn-vi");
    srcEl = overlayRoot.querySelector(".vn-src");
    statusEl = overlayRoot.querySelector(".vn-status");
    modeBadgeEl = overlayRoot.querySelector(".vn-mode-pill");

    overlayRoot.querySelector(".vn-close").addEventListener("click", () => {
      stopAllModes(true);
    });

    overlayRoot.querySelector(".vn-save").addEventListener("click", () => {
      const itemsToSave =
        activeMode === "captions" && rawCues.length > 0
          ? rawCues
              .filter((c) => translatedMap.has(c.id))
              .map((c) => ({
                id: `yt-${c.id}`,
                speaker: "Video",
                source: "browser",
                text: c.text,
                translation: translatedMap.get(c.id) || "",
                time_label: formatSec(c.start),
              }))
          : historyItems;

      if (itemsToSave.length === 0) {
        statusEl.textContent = "Chưa có câu đã dịch nào để lưu.";
        return;
      }
      chrome.runtime.sendMessage(
        {
          type: "save-browser-transcript",
          title: document.title.replace(/ - YouTube$/, "") || "Video Trình duyệt",
          url: location.href,
          items: itemsToSave,
        },
        (res) => {
          statusEl.textContent = res?.ok
            ? `✓ Đã lưu ${itemsToSave.length} câu sang VinaNote!`
            : "Hãy mở ứng dụng VinaNote Offline trên máy tính để lưu.";
        }
      );
    });

    // Draggable overlay
    let dragging = false;
    let startX = 0;
    let startY = 0;
    let origLeft = 0;
    let origBottom = 0;
    const bar = overlayRoot.querySelector(".vn-topbar");
    bar.addEventListener("mousedown", (e) => {
      if (e.target.tagName === "BUTTON") return;
      dragging = true;
      startX = e.clientX;
      startY = e.clientY;
      const rect = overlayRoot.getBoundingClientRect();
      origLeft = rect.left;
      origBottom = window.innerHeight - rect.bottom;
      overlayRoot.style.transform = "none";
      overlayRoot.style.left = `${origLeft}px`;
      overlayRoot.style.bottom = `${origBottom}px`;
      e.preventDefault();
    });
    window.addEventListener("mousemove", (e) => {
      if (!dragging) return;
      overlayRoot.style.left = `${origLeft + (e.clientX - startX)}px`;
      overlayRoot.style.bottom = `${Math.max(10, origBottom - (e.clientY - startY))}px`;
    });
    window.addEventListener("mouseup", () => {
      dragging = false;
    });
  }

  // ---------------- Ask MAIN-world page-hook.js for YouTube Captions ----------------

  function requestPageYouTubeCaptions(language) {
    return new Promise((resolve) => {
      if (!location.hostname.includes("youtube.com")) {
        resolve({ ok: false, cues: [] });
        return;
      }
      const reqId = Math.random().toString(36).slice(2);
      const timeout = setTimeout(() => {
        window.removeEventListener("message", onMsg);
        resolve({ ok: false, reason: "timeout", cues: [] });
      }, 3200);

      function onMsg(ev) {
        if (
          ev.source === window &&
          ev.data?.source === "vinanote-page" &&
          ev.data?.type === "yt-captions-result" &&
          ev.data?.requestId === reqId
        ) {
          clearTimeout(timeout);
          window.removeEventListener("message", onMsg);
          resolve(ev.data.result || { ok: false, cues: [] });
        }
      }
      window.addEventListener("message", onMsg);
      window.postMessage(
        { source: "vinanote-content", type: "extract-yt-captions", requestId: reqId, language },
        "*"
      );
    });
  }

  // Also check standard HTML5 <video> textTracks on non-YouTube sites (Coursera, Udemy, etc.)
  function extractHtml5TextTracks() {
    const v = findActiveVideo();
    if (!v || !v.textTracks) return { cues: [], language: "" };
    for (const track of Array.from(v.textTracks)) {
      if (track.cues && track.cues.length > 0) {
        const cues = Array.from(track.cues)
          .map((c, idx) => ({
            id: idx,
            start: Number(c.startTime.toFixed(2)),
            end: Number(c.endTime.toFixed(2)),
            text: (c.text || "").replace(/\s*\n+\s*/g, " ").trim(),
          }))
          .filter((c) => c.text.length > 0);
        return { cues, language: track.language || "" };
      }
    }
    return { cues: [], language: "" };
  }

  // ---------------- Zero-Delay Caption Scheduler & Pre-Dubbing ----------------

  function stopDubAudio() {
    if (currentDubAudio) {
      try {
        currentDubAudio.pause();
      } catch {}
      currentDubAudio = null;
    }
    const v = findActiveVideo();
    if (v && v.__vnOriginalVolume !== undefined) {
      v.volume = v.__vnOriginalVolume;
    }
  }

  function dubKey(cue) {
    return `${cue.id}|${opts.voice}|${translatedMap.get(cue.id) || ""}`;
  }

  function prefetchUpcomingDubs(nowSec) {
    if (!opts.speakVi || activeMode !== "captions") return;
    if (dubPending.size >= 3) return;

    for (const cue of rawCues) {
      if (cue.start > nowSec + 36) break;
      if (cue.end <= nowSec + 0.2) continue;
      const vi = translatedMap.get(cue.id);
      if (!vi || !vi.trim()) continue;
      const key = dubKey(cue);
      if (dubCache.has(key) || dubPending.has(key)) continue;

      dubPending.add(key);
      chrome.runtime.sendMessage(
        { type: "synthesize-tts", text: vi, voice: opts.voice },
        (res) => {
          dubPending.delete(key);
          if (res?.ok && res.wav_b64) {
            dubCache.set(key, {
              wavB64: res.wav_b64,
              durationSec: Number(res.duration_sec || 1.8),
            });
          }
        }
      );
      if (dubPending.size >= 3) break;
    }
  }

  function scheduleCaptionTranslation() {
    if (activeMode !== "captions" || inFlightBatch || rawCues.length === 0) return;
    const v = findActiveVideo();
    const nowSec = v ? v.currentTime : 0;

    // Find pivot index at current playhead
    let pivot = rawCues.findIndex((c) => c.end >= nowSec - 0.1);
    if (pivot < 0) return;

    // Priority 1A: Instant SINGLE cue right at current playhead (~0.5s with cached KV prefix on M2 Pro!)
    let batchToRun = [];
    let isIdleBackground = false;
    if (!translatedMap.has(rawCues[pivot].id)) {
      batchToRun = [rawCues[pivot]];
    } else {
      // Priority 1B: Next 3 cues immediately following playhead
      for (let i = pivot + 1; i < Math.min(rawCues.length, pivot + 4); i++) {
        if (!translatedMap.has(rawCues[i].id)) {
          batchToRun.push(rawCues[i]);
        }
      }
    }

    if (batchToRun.length === 0) {
      // Priority 2: Rolling lookahead up to +120s ahead in 4-cue batches
      const lookahead = [];
      for (let i = pivot; i < rawCues.length; i++) {
        if (rawCues[i].start > nowSec + 120) break;
        if (!translatedMap.has(rawCues[i].id)) {
          lookahead.push(rawCues[i]);
          if (lookahead.length >= 4) break;
        }
      }
      batchToRun = lookahead;
    }

    if (batchToRun.length === 0) {
      // Priority 3: Idle full-video prefetch in 4-cue batches, wrapping to earlier cues
      isIdleBackground = true;
      const rest = [];
      for (let i = pivot; i < rawCues.length && rest.length < 4; i++) {
        if (!translatedMap.has(rawCues[i].id)) rest.push(rawCues[i]);
      }
      for (let i = 0; i < pivot && rest.length < 4; i++) {
        if (!translatedMap.has(rawCues[i].id)) rest.push(rawCues[i]);
      }
      batchToRun = rest;
    }

    if (batchToRun.length === 0) {
      statusEl.textContent = `✓ Đã dịch sẵn 100% (${translatedMap.size}/${rawCues.length} câu · 0s trễ)`;
      prefetchUpcomingDubs(nowSec);
      return;
    }

    const firstIdx = rawCues.findIndex((c) => c.id === batchToRun[0].id);
    const ctx = rawCues
      .slice(Math.max(0, firstIdx - 2), firstIdx)
      .map((c) => c.text);

    inFlightBatch = true;
    const myEpoch = planEpoch;
    const t0 = performance.now();
    statusEl.textContent =
      batchToRun.length === 1
        ? `⚡ Đang dịch tức thì câu hiện tại...`
        : isIdleBackground
          ? `⚡ Đang dịch nền toàn video (${translatedMap.size}/${rawCues.length} câu)...`
          : `⚡ Đang dịch đón đầu +${Math.max(1, Math.round(batchToRun[batchToRun.length - 1].end - nowSec))}s (${translatedMap.size}/${rawCues.length})...`;

    chrome.runtime.sendMessage(
      {
        type: "translate-captions-batch",
        cues: batchToRun,
        language: captionLanguage || opts.language,
        style: opts.style,
        context: ctx,
      },
      (res) => {
        inFlightBatch = false;
        if (activeMode !== "captions") return;
        if (res?.ok && Array.isArray(res.cues)) {
          for (const item of res.cues) {
            if (item && item.translation) {
              translatedMap.set(item.id, item.translation);
            }
          }
          const ms = Math.round(performance.now() - t0);
          statusEl.textContent =
            translatedMap.size >= rawCues.length
              ? `✓ Đã dịch sẵn 100% (${translatedMap.size}/${rawCues.length} câu · 0s trễ)`
              : `⚡ Sẵn sàng (${translatedMap.size}/${rawCues.length} câu · ${ms}ms)`;
          renderCaptionAtPlayhead();
          const curV = findActiveVideo();
          if (curV) prefetchUpcomingDubs(curV.currentTime);
        } else if (res?.error) {
          statusEl.textContent = `Chờ AI Engine: ${res.error}`;
        }
        if (myEpoch === planEpoch) {
          setTimeout(scheduleCaptionTranslation, isIdleBackground ? 220 : 60);
        }
      }
    );
  }

  function renderCaptionAtPlayhead() {
    if (activeMode !== "captions" || !overlayRoot) return;
    const v = findActiveVideo();
    if (!v) return;
    const t = v.currentTime;

    // Detect user seek (> 1.8s jump) -> cancel old dubbing & immediately prioritize new playhead!
    if (lastPlayhead >= 0 && Math.abs(t - lastPlayhead) > 1.8) {
      planEpoch++;
      spokenKeys.clear();
      stopDubAudio();
      scheduleCaptionTranslation();
    }
    lastPlayhead = t;

    if (v.paused && currentDubAudio && !currentDubAudio.paused) {
      stopDubAudio();
    }

    // Find active cue at playhead `t` (show 150ms early so viewer reads on the very first syllable!)
    let idx = -1;
    for (let i = 0; i < rawCues.length; i++) {
      if (t >= rawCues[i].start - 0.15 && t <= rawCues[i].end + 0.15) {
        idx = i;
        break;
      }
    }

    if (idx < 0) {
      if (lastShownCueId !== -1) {
        viEl.style.opacity = "0.35";
        lastShownCueId = -1;
      }
      return;
    }

    const cue = rawCues[idx];
    const vi = translatedMap.get(cue.id);
    viEl.style.opacity = "1";
    srcEl.textContent = cue.text;
    viEl.textContent = vi || "⚡ Đang dịch câu này...";
    lastShownCueId = cue.id;

    // Trigger pre-synthesized ZeroTTS dubbing right on cue start!
    if (opts.speakVi && !v.paused && vi) {
      const key = dubKey(cue);
      const cached = dubCache.get(key);
      if (cached && !spokenKeys.has(key) && t < cue.end - 0.3) {
        spokenKeys.add(key);
        const nextStart = rawCues[idx + 1]?.start ?? cue.end + 1.0;
        const slot = Math.max(0.45, nextStart - t - 0.1);
        const rate = Math.min(1.42, Math.max(1.06, cached.durationSec / slot));

        stopDubAudio();
        if (v.__vnOriginalVolume === undefined) {
          v.__vnOriginalVolume = v.volume;
        }
        v.volume = Math.min(v.__vnOriginalVolume, 0.22);

        const audio = new Audio(`data:audio/wav;base64,${cached.wavB64}`);
        audio.playbackRate = rate;
        audio.onended = () => {
          if (currentDubAudio === audio) {
            currentDubAudio = null;
            if (v.__vnOriginalVolume !== undefined) v.volume = v.__vnOriginalVolume;
          }
        };
        audio.onerror = () => {
          if (currentDubAudio === audio) {
            currentDubAudio = null;
            if (v.__vnOriginalVolume !== undefined) v.volume = v.__vnOriginalVolume;
          }
        };
        currentDubAudio = audio;
        audio.play().catch(() => {});
      }
    }
  }

  async function tryActivateCaptionMode(newOpts) {
    if (newOpts) opts = { ...opts, ...newOpts };
    captionLanguage = "";
    ensureOverlay();
    overlayRoot.style.display = "flex";
    modeBadgeEl.textContent = "Đang tìm phụ đề...";
    viEl.textContent = "Đang kiểm tra phụ đề gốc của video...";
    viEl.style.opacity = "1";
    srcEl.textContent = "";

    let cues = [];
    let language = "";
    let vidId = "";

    if (location.hostname.includes("youtube.com")) {
      const ytRes = await requestPageYouTubeCaptions(opts.language);
      vidId = ytRes?.videoId || "";
      if (ytRes?.ok && ytRes.cues?.length > 0) {
        cues = ytRes.cues;
        language = ytRes.language || "";
      } else {
        // Tier-4 Fallback: ask local AI worker yt-dlp subtitle extractor!
        const fb = await new Promise((resolve) =>
          chrome.runtime.sendMessage(
            { type: "fetch-yt-captions-fallback", url: location.href, language: opts.language },
            (r) => resolve(r || { ok: false, cues: [] })
          )
        );
        if (fb?.ok && fb.cues?.length > 0) {
          cues = fb.cues;
          language = fb.language || "";
        }
      }
    } else {
      ({ cues, language } = extractHtml5TextTracks());
    }

    if (!cues || cues.length === 0) {
      return { ok: false, mode: "none", cueCount: 0 };
    }

    // Activate Zero-Delay Caption Mode!
    activeMode = "captions";
    currentVideoId = vidId || location.href;
    rawCues = cues;
    captionLanguage = language;
    translatedMap.clear();
    dubCache.clear();
    dubPending.clear();
    spokenKeys.clear();
    planEpoch++;
    lastPlayhead = -1;

    document.documentElement.classList.add("vinanote-hide-yt-cc");
    modeBadgeEl.textContent = `⚡ Phụ đề gốc (${cues.length} câu · 0s trễ)`;
    modeBadgeEl.className = "vn-mode-pill fast";
    updateYouTubeButtonState(true);

    chrome.runtime.sendMessage({ type: "register-caption-mode", active: true }).catch(() => {});

    clearInterval(syncTimer);
    syncTimer = setInterval(() => {
      renderCaptionAtPlayhead();
      scheduleCaptionTranslation();
    }, 150);

    scheduleCaptionTranslation();
    renderCaptionAtPlayhead();
    return { ok: true, mode: "captions", cueCount: cues.length };
  }

  function stopAllModes(notifyBg = false) {
    activeMode = null;
    captionLanguage = "";
    rawCues = [];
    translatedMap.clear();
    spokenKeys.clear();
    clearInterval(syncTimer);
    syncTimer = null;
    stopDubAudio();
    document.documentElement.classList.remove("vinanote-hide-yt-cc");
    if (overlayRoot) overlayRoot.style.display = "none";
    updateYouTubeButtonState(false);
    if (notifyBg) {
      chrome.runtime.sendMessage({ type: "stop-tab-translate" }).catch(() => {});
    }
  }

  // ---------------- 1-Click Button Inside YouTube Player Controls ----------------

  function updateYouTubeButtonState(on) {
    if (!ytBtn) return;
    ytBtn.classList.toggle("active", Boolean(on));
    ytBtn.textContent = on ? "VN Đang dịch" : "VN Dịch";
  }

  function injectYouTubePlayerButton() {
    if (!location.hostname.includes("youtube.com")) return;
    const rightControls = document.querySelector(".ytp-right-controls");
    if (!rightControls || rightControls.querySelector(".vinanote-ytp-btn")) return;

    const btn = document.createElement("button");
    btn.className = "ytp-button vinanote-ytp-btn";
    btn.title = "Dịch phụ đề & Thuyết minh tiếng Việt tức thì (VinaNote Offline)";
    btn.textContent = activeMode ? "VN Đang dịch" : "VN Dịch";
    if (activeMode) btn.classList.add("active");

    btn.addEventListener("click", async (e) => {
      e.stopPropagation();
      e.preventDefault();
      if (activeMode) {
        stopAllModes(true);
      } else {
        const st = await new Promise((r) => chrome.runtime.sendMessage({ type: "get-state" }, r));
        if (st?.options) opts = { ...opts, ...st.options };
        const res = await tryActivateCaptionMode(opts);
        if (!res.ok) {
          ensureOverlay();
          overlayRoot.style.display = "flex";
          modeBadgeEl.textContent = "Video không có CC";
          viEl.textContent =
            "Video YouTube này không có phụ đề gốc. Hãy bấm biểu tượng VinaNote trên thanh công cụ trình duyệt để bật chế độ Nghe âm thanh trực tiếp!";
        }
      }
    });

    rightControls.prepend(btn);
    ytBtn = btn;
  }

  // Handle YouTube SPA navigation to a new video while Caption Mode is ON!
  window.addEventListener("yt-navigate-finish", () => {
    setTimeout(injectYouTubePlayerButton, 600);
    if (activeMode === "captions") {
      const u = new URL(location.href);
      const newVid = u.searchParams.get("v") || location.pathname;
      if (newVid && newVid !== currentVideoId) {
        stopDubAudio();
        tryActivateCaptionMode(opts);
      }
    }
  });

  setInterval(injectYouTubePlayerButton, 2000);

  // ---------------- Extension Message Listener ----------------

  chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
    if (msg.type === "vinanote-try-captions") {
      tryActivateCaptionMode(msg.options).then(sendResponse);
      return true;
    }

    if (msg.type === "vinanote-stop-all") {
      stopAllModes(false);
      sendResponse({ ok: true });
      return false;
    }

    if (msg.type === "vinanote-options-updated") {
      const prevStyle = opts.style;
      opts = { ...opts, ...(msg.options || {}) };
      if (!opts.speakVi) stopDubAudio();
      if (activeMode === "captions" && prevStyle !== opts.style) {
        translatedMap.clear();
        planEpoch++;
        scheduleCaptionTranslation();
      }
      sendResponse({ ok: true });
      return false;
    }

    if (msg.type === "vinanote-show-overlay") {
      ensureOverlay();
      if (!msg.active) {
        stopAllModes(false);
      } else {
        activeMode = msg.mode || "live-audio";
        overlayRoot.style.display = "flex";
        modeBadgeEl.textContent = "🎙 Nghe trực tiếp âm thanh";
        modeBadgeEl.className = "vn-mode-pill live";
        viEl.textContent = "Đang lắng nghe âm thanh video (Offline MLX)...";
        viEl.style.opacity = "1";
        srcEl.textContent = "";
        updateYouTubeButtonState(true);
      }
      return false;
    }

    if (msg.type === "vinanote-overlay") {
      if (activeMode === "captions") return false;
      ensureOverlay();
      activeMode = "live-audio";
      overlayRoot.style.display = "flex";
      const p = msg.payload || {};
      if (p.kind === "status") {
        statusEl.textContent = p.text;
      } else if (p.type === "transcript") {
        srcEl.textContent = p.text || "";
        const isVietnamese = p.needs_translation === false || p.language === "vi";
        const found = historyItems.find((x) => x.id === p.id);
        if (found) {
          found.text = p.text || "";
          if (isVietnamese) found.translation = p.text;
        } else {
          const timeStr = new Date().toLocaleTimeString("vi-VN", { hour12: false });
          historyItems.push({
            id: p.id,
            speaker: "Video",
            source: "browser",
            text: p.text || "",
            translation: isVietnamese ? p.text : "",
            time_label: timeStr,
          });
        }
        if (isVietnamese) {
          viEl.textContent = p.text;
          viEl.style.opacity = "1";
        }
        statusEl.textContent = `ASR ${Math.round(p.asr_ms || 0)}ms`;
      } else if (p.type === "retract") {
        historyItems = historyItems.filter((x) => x.id !== p.id);
      } else if (p.type === "translation_partial") {
        viEl.textContent = p.text || "";
        viEl.style.opacity = "0.85";
      } else if (p.type === "translation") {
        viEl.textContent = p.text || "";
        const found = historyItems.find((x) => x.id === p.id);
        if (found) found.translation = p.text;
        statusEl.textContent = `● Đã dịch (${Math.round(p.translate_ms || 0)}ms)`;
        viEl.style.opacity = "1";
      } else if (p.type === "error") {
        statusEl.textContent = `Lỗi: ${p.message}`;
      }
      return false;
    }
  });
})();
