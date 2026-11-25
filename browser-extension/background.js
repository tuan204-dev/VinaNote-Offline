// Background Service Worker:
// 1. Proxies fast Caption Micro-Batch Translation (/captions/translate) & Pre-Synthesized ZeroTTS (/tts/speak-json)
//    for Zero-Delay YouTube Caption Mode (no tabCapture needed when captions exist!).
// 2. Fallback YouTube caption fetcher (/youtube/captions via yt-dlp) if YouTube DOM timedtext is empty.
// 3. Coordinates chrome.tabCapture + Offscreen Audio Document for videos WITHOUT subtitles (TikTok, Douyin, Bilibili, Facebook, unsubtitled YouTube).
// 4. Saves bilingual transcripts to VinaNote Offline Desktop App (127.0.0.1:38478).

const AI_HTTP = "http://127.0.0.1:38479";
const BRIDGE_HTTP = "http://127.0.0.1:38478";

let capturingTabId = null;
let captionModeTabs = new Set();
let currentOptions = {
  language: "auto",
  style: "natural",
  speakVi: true,
  voice: "maichi",
};

chrome.storage.local.get(["vinanoteOpts"], (res) => {
  if (res?.vinanoteOpts) {
    currentOptions = { ...currentOptions, ...res.vinanoteOpts };
  }
});

async function ensureOffscreenDocument() {
  const existing = await chrome.runtime.getContexts({
    contextTypes: ["OFFSCREEN_DOCUMENT"],
  });
  if (existing.length > 0) return;
  await chrome.offscreen.createDocument({
    url: "offscreen.html",
    reasons: ["USER_MEDIA", "AUDIO_PLAYBACK"],
    justification: "Capture tab video audio and play offline Vietnamese dubbing",
  });
}

chrome.tabs.onRemoved.addListener((tabId) => {
  captionModeTabs.delete(tabId);
  if (capturingTabId === tabId) {
    capturingTabId = null;
    chrome.runtime.sendMessage({ target: "offscreen", type: "stop" }).catch(() => {});
  }
});

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.target === "background" && msg.type === "overlay-event") {
    if (msg.tabId) {
      chrome.tabs.sendMessage(msg.tabId, { type: "vinanote-overlay", payload: msg.payload }).catch(() => {});
    }
    return false;
  }

  if (msg.type === "get-state") {
    const reqTabId = msg.tabId || sender.tab?.id;
    fetch(`${AI_HTTP}/models/status`)
      .then((r) => r.json())
      .then((st) => {
        sendResponse({
          ok: true,
          capturingTabId,
          isCaptionMode: Boolean(reqTabId && captionModeTabs.has(reqTabId)),
          options: currentOptions,
          engine: st,
        });
      })
      .catch(() => {
        sendResponse({
          ok: false,
          capturingTabId,
          isCaptionMode: Boolean(reqTabId && captionModeTabs.has(reqTabId)),
          options: currentOptions,
          engine: null,
        });
      });
    return true;
  }

  if (msg.type === "register-caption-mode") {
    const tId = msg.tabId || sender.tab?.id;
    if (tId) {
      if (msg.active) {
        captionModeTabs.add(tId);
        // Ensure live audio tabCapture is stopped if switching to zero-delay caption mode
        if (capturingTabId === tId) {
          capturingTabId = null;
          chrome.runtime.sendMessage({ target: "offscreen", type: "stop" }).catch(() => {});
        }
      } else {
        captionModeTabs.delete(tId);
      }
    }
    sendResponse({ ok: true });
    return false;
  }

  if (msg.type === "start-tab-translate") {
    (async () => {
      try {
        const tabId = msg.tabId || sender.tab?.id;
        if (!tabId) throw new Error("Không xác định được Tab hiện tại.");
        currentOptions = { ...currentOptions, ...(msg.options || {}) };
        chrome.storage.local.set({ vinanoteOpts: currentOptions });

        // STEP 1: Ask content script to try ZERO-DELAY CAPTION MODE first (YouTube captions / HTML5 <track>)!
        try {
          const capTry = await chrome.tabs.sendMessage(tabId, {
            type: "vinanote-try-captions",
            options: currentOptions,
          });
          if (capTry?.ok && capTry?.mode === "captions") {
            captionModeTabs.add(tabId);
            sendResponse({ ok: true, mode: "captions", cueCount: capTry.cueCount });
            return;
          }
        } catch {}

        // STEP 2: Fallback to live tabCapture audio stream when video has NO subtitles (TikTok, Douyin, Reels, unsubtitled YT)!
        await ensureOffscreenDocument();
        const streamId = await chrome.tabCapture.getMediaStreamId({ targetTabId: tabId });

        const res = await chrome.runtime.sendMessage({
          target: "offscreen",
          type: "start",
          streamId,
          tabId,
          options: currentOptions,
        });
        if (!res?.ok) throw new Error(res?.error || "Không mở được luồng âm thanh Tab.");
        capturingTabId = tabId;
        captionModeTabs.delete(tabId);
        chrome.tabs.sendMessage(tabId, { type: "vinanote-show-overlay", active: true, mode: "live-audio" }).catch(() => {});
        sendResponse({ ok: true, mode: "live-audio", capturingTabId });
      } catch (e) {
        sendResponse({ ok: false, error: String(e) });
      }
    })();
    return true;
  }

  if (msg.type === "stop-tab-translate") {
    (async () => {
      const tabId = msg.tabId || capturingTabId || sender.tab?.id;
      if (tabId) {
        captionModeTabs.delete(tabId);
        chrome.tabs.sendMessage(tabId, { type: "vinanote-stop-all" }).catch(() => {});
      }
      if (capturingTabId) {
        capturingTabId = null;
        await chrome.runtime.sendMessage({ target: "offscreen", type: "stop" }).catch(() => {});
      }
      sendResponse({ ok: true });
    })();
    return true;
  }

  if (msg.type === "update-options") {
    currentOptions = { ...currentOptions, ...(msg.options || {}) };
    chrome.storage.local.set({ vinanoteOpts: currentOptions });
    chrome.runtime.sendMessage({
      target: "offscreen",
      type: "update-config",
      options: currentOptions,
    }).catch(() => {});
    // Broadcast updated options to all caption-mode tabs as well
    for (const tId of captionModeTabs) {
      chrome.tabs.sendMessage(tId, { type: "vinanote-options-updated", options: currentOptions }).catch(() => {});
    }
    sendResponse({ ok: true });
    return false;
  }

  if (msg.type === "translate-captions-batch") {
    fetch(`${AI_HTTP}/captions/translate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        cues: msg.cues || [],
        language: msg.language || currentOptions.language,
        style: msg.style || currentOptions.style,
        context: msg.context || [],
      }),
    })
      .then((r) => r.json())
      .then((data) => sendResponse({ ok: true, cues: data.cues || [] }))
      .catch((e) => sendResponse({ ok: false, error: String(e) }));
    return true;
  }

  if (msg.type === "fetch-yt-captions-fallback") {
    fetch(`${AI_HTTP}/youtube/captions?url=${encodeURIComponent(msg.url || "")}&lang=${encodeURIComponent(msg.language || "auto")}`)
      .then((r) => r.json())
      .then((data) => sendResponse(data))
      .catch((e) => sendResponse({ ok: false, error: String(e), cues: [] }));
    return true;
  }

  if (msg.type === "synthesize-tts") {
    const voice = msg.voice === "auto" || !msg.voice ? "maichi" : msg.voice;
    fetch(`${AI_HTTP}/tts/speak-json`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: msg.text || "", voice }),
    })
      .then((r) => r.json())
      .then((data) => sendResponse({ ok: true, ...data }))
      .catch((e) => sendResponse({ ok: false, error: String(e) }));
    return true;
  }

  if (msg.type === "save-browser-transcript") {
    fetch(`${BRIDGE_HTTP}/v1/save-note`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        title: msg.title || "Video Trình duyệt",
        source_url: msg.url || "",
        items: msg.items || [],
      }),
    })
      .then((r) => r.json())
      .then((d) => sendResponse({ ok: true, data: d }))
      .catch((e) => sendResponse({ ok: false, error: String(e) }));
    return true;
  }
});
