// Runs in MAIN world on YouTube (run_at: document_start) so we can:
// 1. Sniff YouTube's own signed `/api/timedtext` requests (which carry YouTube's POToken `&pot=...`!)
// 2. Query `#movie_player.getPlayerResponse()` directly (always accurate across YouTube SPA navigations!)
// 3. Force YouTube's player to load a caption track invisibly if needed so YouTube generates the signed timedtext request,
//    then parse both manual captions and word-level auto-captions (`json3` / `srv3` / XML) into clean non-overlapping cues!

(() => {
  if (window.__vinanotePageHookInstalled) return;
  window.__vinanotePageHookInstalled = true;

  // Cache of signed timedtext URLs or raw bodies observed per videoId
  const sniffedTimedText = new Map(); // videoId -> { url, bodyText }

  function recordTimedTextUrl(urlStr, bodyText) {
    try {
      const u = new URL(urlStr, location.origin);
      if (!u.pathname.includes("/api/timedtext")) return;
      const v = u.searchParams.get("v");
      if (!v) return;
      const prev = sniffedTimedText.get(v) || {};
      sniffedTimedText.set(v, {
        url: u.toString(),
        lang: u.searchParams.get("lang") || prev.lang || "",
        bodyText: bodyText || prev.bodyText || null,
      });
    } catch {}
  }

  // Hook fetch in MAIN world to capture signed /api/timedtext URLs & responses
  const origFetch = window.fetch;
  window.fetch = async function (...args) {
    const reqUrl = typeof args[0] === "string" ? args[0] : args[0]?.url || "";
    const resp = await origFetch.apply(this, args);
    if (reqUrl.includes("/api/timedtext") && !reqUrl.includes("_vn_internal=1")) {
      try {
        const clone = resp.clone();
        clone.text().then((txt) => {
          if (txt && txt.length > 10) {
            recordTimedTextUrl(reqUrl, txt);
            window.postMessage({ source: "vinanote-page", type: "timedtext-sniffed", videoId: new URL(reqUrl, location.origin).searchParams.get("v") }, "*");
          }
        }).catch(() => {});
      } catch {}
    }
    return resp;
  };

  // Hook XMLHttpRequest as well (YouTube's player frequently uses XHR for /api/timedtext!)
  const origOpen = XMLHttpRequest.prototype.open;
  const origSend = XMLHttpRequest.prototype.send;
  XMLHttpRequest.prototype.open = function (method, url, ...rest) {
    this.__vnUrl = typeof url === "string" ? url : String(url || "");
    return origOpen.call(this, method, url, ...rest);
  };
  XMLHttpRequest.prototype.send = function (...args) {
    if (this.__vnUrl && this.__vnUrl.includes("/api/timedtext") && !this.__vnUrl.includes("_vn_internal=1")) {
      this.addEventListener("load", function () {
        try {
          if (this.responseText && this.responseText.length > 10) {
            recordTimedTextUrl(this.__vnUrl, this.responseText);
            const v = new URL(this.__vnUrl, location.origin).searchParams.get("v");
            window.postMessage({ source: "vinanote-page", type: "timedtext-sniffed", videoId: v }, "*");
          }
        } catch {}
      });
    }
    return origSend.apply(this, args);
  };

  function getCurrentVideoId() {
    try {
      const u = new URL(location.href);
      if (u.searchParams.get("v")) return u.searchParams.get("v");
      const shortsMatch = u.pathname.match(/\/shorts\/([a-zA-Z0-9_-]{8,15})/);
      if (shortsMatch) return shortsMatch[1];
    } catch {}
    const player = document.getElementById("movie_player");
    return player?.getVideoData?.()?.video_id || "";
  }

  function getPlayerCaptionTracks(videoId) {
    const player = document.getElementById("movie_player");
    let resp = null;
    try {
      resp = player?.getPlayerResponse?.();
    } catch {}
    if (!resp || resp?.videoDetails?.videoId !== videoId) {
      if (window.ytInitialPlayerResponse?.videoDetails?.videoId === videoId) {
        resp = window.ytInitialPlayerResponse;
      }
    }
    const tracks = resp?.captions?.playerCaptionsTracklistRenderer?.captionTracks || [];
    return tracks;
  }

  function pickBestTrack(tracks, preferredLang) {
    if (!tracks || tracks.length === 0) return null;
    // If preferredLang is set (e.g. 'en', 'zh'), prefer manual track of that lang, then auto track
    if (preferredLang && preferredLang !== "auto") {
      const exactManual = tracks.find((t) => t.languageCode?.startsWith(preferredLang) && t.kind !== "asr");
      if (exactManual) return exactManual;
      const exactAny = tracks.find((t) => t.languageCode?.startsWith(preferredLang));
      if (exactAny) return exactAny;
    }
    // Prefer existing Vietnamese manual track if someone uploaded human Vietnamese subs!
    const viManual = tracks.find((t) => t.languageCode === "vi" && t.kind !== "asr");
    if (viManual) return viManual;
    // Otherwise prefer manual original track over auto-generated ('asr')
    const manualNonVi = tracks.find((t) => t.kind !== "asr");
    if (manualNonVi) return manualNonVi;
    return tracks[0];
  }

  function parseJson3ToCues(jsonObj) {
    const events = jsonObj?.events || [];
    // Detect whether this is word-level auto-captions (segs with tOffsetMs) or clean manual cues
    let hasWordOffsets = false;
    for (const ev of events) {
      if (ev.segs && ev.segs.some((s) => typeof s.tOffsetMs === "number" && s.tOffsetMs > 0)) {
        hasWordOffsets = true;
        break;
      }
    }

    if (hasWordOffsets) {
      // Reconstruct clean, non-overlapping sentences from word-level timestamps!
      const words = [];
      for (const ev of events) {
        if (!ev.segs || typeof ev.tStartMs !== "number") continue;
        const baseMs = ev.tStartMs;
        for (const seg of ev.segs) {
          const raw = (seg.utf8 || "").replace(/\n/g, " ").trim();
          if (!raw) continue;
          const startSec = (baseMs + (seg.tOffsetMs || 0)) / 1000;
          words.push({ text: raw, start: startSec });
        }
      }
      if (words.length === 0) return [];
      const cues = [];
      let buf = [];
      for (let i = 0; i < words.length; i++) {
        buf.push(words[i]);
        const w = words[i].text;
        const nextStart = i + 1 < words.length ? words[i + 1].start : words[i].start + 1.2;
        const gap = nextStart - words[i].start;
        const dur = words[i].start - buf[0].start;
        const endsSentence = /[.!?。！？]$/.test(w);
        const endsClause = /[,;:，；、]$/.test(w);

        if (
          (endsSentence && buf.length >= 3) ||
          (gap >= 0.65 && buf.length >= 4) ||
          (endsClause && buf.length >= 8) ||
          buf.length >= 13 ||
          dur >= 5.5
        ) {
          const joined = buf.map((x) => x.text).join(" ").replace(/\s+([.,!?;:])/g, "$1").trim();
          const start = Number(buf[0].start.toFixed(2));
          const end = Number(Math.max(start + 0.8, Math.min(nextStart, words[i].start + 2.2)).toFixed(2));
          if (joined) {
            cues.push({ id: cues.length, start, end, text: joined });
          }
          buf = [];
        }
      }
      if (buf.length > 0) {
        const joined = buf.map((x) => x.text).join(" ").replace(/\s+([.,!?;:])/g, "$1").trim();
        const start = Number(buf[0].start.toFixed(2));
        const end = Number((buf[buf.length - 1].start + 1.5).toFixed(2));
        if (joined) cues.push({ id: cues.length, start, end, text: joined });
      }
      return cues;
    }

    // Manual captions: each event is a clean subtitle cue
    const rawCues = [];
    for (const ev of events) {
      if (!ev.segs || typeof ev.tStartMs !== "number") continue;
      const txt = ev.segs
        .map((s) => s.utf8 || "")
        .join("")
        .replace(/\s*\n+\s*/g, " ")
        .trim();
      if (!txt) continue;
      const start = ev.tStartMs / 1000;
      const dur = (ev.dDurationMs || 2500) / 1000;
      rawCues.push({ start, end: start + dur, text: txt });
    }
    // Clamp overlapping ends so subtitles never collide
    const cues = [];
    for (let i = 0; i < rawCues.length; i++) {
      const c = rawCues[i];
      const nextStart = i + 1 < rawCues.length ? rawCues[i + 1].start : c.end;
      const end = Math.max(c.start + 0.4, Math.min(c.end, nextStart));
      cues.push({
        id: cues.length,
        start: Number(c.start.toFixed(2)),
        end: Number(end.toFixed(2)),
        text: c.text,
      });
    }
    return cues;
  }

  function parseXmlTimedText(xmlStr) {
    const parser = new DOMParser();
    const doc = parser.parseFromString(xmlStr, "text/xml");
    const nodes = Array.from(doc.querySelectorAll("text, p"));
    const rawCues = [];
    for (const node of nodes) {
      const txt = (node.textContent || "").replace(/\s*\n+\s*/g, " ").trim();
      if (!txt) continue;
      let start = 0;
      let dur = 2.5;
      if (node.hasAttribute("t")) {
        start = Number(node.getAttribute("t") || 0) / 1000;
        dur = Number(node.getAttribute("d") || 2500) / 1000;
      } else {
        start = Number(node.getAttribute("start") || 0);
        dur = Number(node.getAttribute("dur") || 2.5);
      }
      rawCues.push({ start, end: start + dur, text: txt });
    }
    return rawCues.map((c, i) => {
      const nextStart = i + 1 < rawCues.length ? rawCues[i + 1].start : c.end;
      return {
        id: i,
        start: Number(c.start.toFixed(2)),
        end: Number(Math.max(c.start + 0.4, Math.min(c.end, nextStart)).toFixed(2)),
        text: c.text,
      };
    });
  }

  function parseAnyTimedText(bodyText) {
    if (!bodyText || bodyText.length < 5) return [];
    const trimmed = bodyText.trim();
    if (trimmed.startsWith("{")) {
      try {
        return parseJson3ToCues(JSON.parse(trimmed));
      } catch {}
    }
    if (trimmed.startsWith("<")) {
      try {
        return parseXmlTimedText(trimmed);
      } catch {}
    }
    return [];
  }

  async function tryFetchUrl(urlStr) {
    const u = new URL(urlStr, location.origin);
    u.searchParams.set("fmt", "json3");
    u.searchParams.set("_vn_internal", "1");
    const r = await origFetch(u.toString(), { credentials: "include" });
    if (!r.ok) return [];
    const txt = await r.text();
    return parseAnyTimedText(txt);
  }

  async function extractYouTubeCaptions(preferredLang) {
    const videoId = getCurrentVideoId();
    if (!videoId) return { ok: false, reason: "no_video_id", cues: [] };

    const tracks = getPlayerCaptionTracks(videoId);
    if (!tracks || tracks.length === 0) {
      return { ok: false, reason: "no_captions_on_video", videoId, cues: [] };
    }

    const chosen = pickBestTrack(tracks, preferredLang);
    if (!chosen?.baseUrl) {
      return { ok: false, reason: "no_track_url", videoId, cues: [] };
    }

    // Attempt 1 (0 ms): Check if YouTube's player already fetched a POToken-signed /api/timedtext!
    const cachedFirst = sniffedTimedText.get(videoId);
    if (cachedFirst?.bodyText) {
      const parsed = parseAnyTimedText(cachedFirst.bodyText);
      if (parsed.length > 0) {
        return { ok: true, videoId, language: cachedFirst.lang || chosen.languageCode || "auto", isAuto: chosen.kind === "asr", cues: parsed };
      }
    }

    // Attempt 2: Direct fetch of chosen.baseUrl + &fmt=json3 with YouTube cookies
    try {
      const cues1 = await tryFetchUrl(chosen.baseUrl);
      if (cues1.length > 0) {
        return {
          ok: true,
          videoId,
          language: chosen.languageCode || "auto",
          isAuto: chosen.kind === "asr",
          cues: cues1,
        };
      }
    } catch {}

    // Attempt 3 (POToken Trigger): Hide native CC box & programmatically toggle .ytp-subtitles-button
    // so YouTube's own player issues the genuine POToken-signed /api/timedtext XHR within ~100-200ms!
    const player = document.getElementById("movie_player");
    if (player) {
      try {
        document.documentElement.classList.add("vinanote-hide-yt-cc");
        const ccBtn = document.querySelector(".ytp-subtitles-button");
        if (ccBtn && ccBtn.getAttribute("aria-pressed") === "false") {
          ccBtn.click();
        }
        player.loadModule?.("captions");
        player.setOption?.("captions", "track", { languageCode: chosen.languageCode });
        for (let wait = 0; wait < 16; wait++) {
          await new Promise((r) => setTimeout(r, 120));
          const nowSniffed = sniffedTimedText.get(videoId);
          if (nowSniffed?.bodyText) {
            const parsed = parseAnyTimedText(nowSniffed.bodyText);
            if (parsed.length > 0) {
              return { ok: true, videoId, language: chosen.languageCode || "auto", isAuto: chosen.kind === "asr", cues: parsed };
            }
          }
          if (nowSniffed?.url) {
            const u = new URL(nowSniffed.url);
            u.searchParams.set("lang", chosen.languageCode);
            if (chosen.kind === "asr") u.searchParams.set("kind", "asr");
            const cues3 = await tryFetchUrl(u.toString());
            if (cues3.length > 0) {
              return { ok: true, videoId, language: chosen.languageCode || "auto", isAuto: chosen.kind === "asr", cues: cues3 };
            }
          }
        }
      } catch {}
    }

    return { ok: false, reason: "timedtext_empty", videoId, cues: [] };
  }

  window.addEventListener("message", async (ev) => {
    if (ev.source !== window || ev.data?.source !== "vinanote-content") return;
    if (ev.data.type === "extract-yt-captions") {
      const result = await extractYouTubeCaptions(ev.data.language || "auto");
      window.postMessage(
        {
          source: "vinanote-page",
          type: "yt-captions-result",
          requestId: ev.data.requestId,
          result,
        },
        "*"
      );
    }
  });
})();
