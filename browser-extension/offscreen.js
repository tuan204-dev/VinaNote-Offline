// Offscreen Document: Captures tab audio via chrome.tabCapture streamId (bypassing all CDN CORS limits
// on TikTok, Douyin, Bilibili, Facebook, YouTube), keeps tab audio audible with automatic volume ducking
// during Vietnamese ZeroTTS dubbing, resamples to 16 kHz mono Float32 PCM, and streams directly to
// ws://127.0.0.1:38479/ws/live!

const AI_WS_URL = "ws://127.0.0.1:38479/ws/live";
const TARGET_SR = 16000;

let mediaStream = null;
let audioCtx = null;
let gainNode = null;
let processorNode = null;
let ws = null;
let activeTabId = null;
let dubQueue = [];
let dubPlaying = false;

function float32ToBase64(float32Arr) {
  const bytes = new Uint8Array(float32Arr.buffer, float32Arr.byteOffset, float32Arr.byteLength);
  let binary = "";
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode.apply(null, bytes.subarray(i, i + chunk));
  }
  return btoa(binary);
}

function downsampleBuffer(buffer, inputSampleRate, outSampleRate) {
  if (outSampleRate === inputSampleRate) return buffer;
  const ratio = inputSampleRate / outSampleRate;
  const newLen = Math.round(buffer.length / ratio);
  const result = new Float32Array(newLen);
  let offsetResult = 0;
  let offsetBuffer = 0;
  while (offsetResult < result.length) {
    const nextOffsetBuffer = Math.round((offsetResult + 1) * ratio);
    let accum = 0;
    let count = 0;
    for (let i = offsetBuffer; i < nextOffsetBuffer && i < buffer.length; i++) {
      accum += buffer[i];
      count++;
    }
    result[offsetResult] = count > 0 ? accum / count : 0;
    offsetResult++;
    offsetBuffer = nextOffsetBuffer;
  }
  return result;
}

function playNextDub() {
  if (dubPlaying || dubQueue.length === 0) {
    if (!dubPlaying && gainNode && audioCtx) {
      gainNode.gain.setTargetAtTime(1.0, audioCtx.currentTime, 0.08);
    }
    return;
  }
  const wavB64 = dubQueue.shift();
  dubPlaying = true;
  if (gainNode && audioCtx) {
    // Duck original YouTube / TikTok audio to 22% while Vietnamese dub speaks!
    gainNode.gain.setTargetAtTime(0.22, audioCtx.currentTime, 0.05);
  }
  const audio = new Audio(`data:audio/wav;base64,${wavB64}`);
  // Speed up dynamically if another clause/utterance is waiting behind this one!
  audio.playbackRate = dubQueue.length >= 1 ? 1.36 : 1.2;
  audio.onended = () => {
    dubPlaying = false;
    playNextDub();
  };
  audio.onerror = () => {
    dubPlaying = false;
    playNextDub();
  };
  audio.play().catch(() => {
    dubPlaying = false;
    playNextDub();
  });
}

async function startCapture(streamId, tabId, options) {
  await stopCapture();
  activeTabId = tabId;

  mediaStream = await navigator.mediaDevices.getUserMedia({
    audio: {
      mandatory: {
        chromeMediaSource: "tab",
        chromeMediaSourceId: streamId,
      },
    },
    video: false,
  });

  audioCtx = new AudioContext();
  const source = audioCtx.createMediaStreamSource(mediaStream);
  gainNode = audioCtx.createGain();
  gainNode.gain.value = 1.0;

  // Keep tab audio playing to speakers through GainNode (for smooth ducking)
  source.connect(gainNode);
  gainNode.connect(audioCtx.destination);

  // Connect WebSocket to local Offline AI Worker
  ws = new WebSocket(AI_WS_URL);
  ws.onopen = () => {
    ws.send(
      JSON.stringify({
        type: "config",
        language: options.language || "auto",
        style: options.style || "natural",
        translate: true,
        speak_vi: Boolean(options.speakVi),
        tts_voice: options.voice || "auto",
      })
    );
    chrome.runtime.sendMessage({
      target: "background",
      type: "overlay-event",
      tabId: activeTabId,
      payload: { kind: "status", text: "● Đang nghe & dịch Offline (Apple MLX)" },
    });
  };

  ws.onmessage = (ev) => {
    try {
      const msg = JSON.parse(ev.data);
      if (msg.type === "transcript" || msg.type === "translation" || msg.type === "error") {
        chrome.runtime.sendMessage({
          target: "background",
          type: "overlay-event",
          tabId: activeTabId,
          payload: msg,
        });
      } else if (msg.type === "tts_audio" && msg.wav_b64) {
        if (dubQueue.length > 3) dubQueue.shift();
        dubQueue.push(msg.wav_b64);
        playNextDub();
      }
    } catch {}
  };

  ws.onerror = () => {
    chrome.runtime.sendMessage({
      target: "background",
      type: "overlay-event",
      tabId: activeTabId,
      payload: {
        kind: "status",
        text: "Không kết nối được VinaNote Offline (hãy mở ứng dụng VinaNote trên Mac)",
      },
    });
  };

  // Tap PCM audio & downsample to 16 kHz mono
  processorNode = audioCtx.createScriptProcessor(4096, 1, 1);
  source.connect(processorNode);
  processorNode.connect(audioCtx.destination); // silent pass-through required by Chromium ScriptProcessor

  processorNode.onaudioprocess = (audioProcessingEvent) => {
    const inputData = audioProcessingEvent.inputBuffer.getChannelData(0);
    // Ensure processor output buffer is zeroed so we don't double-play audio alongside gainNode
    const outBuf = audioProcessingEvent.outputBuffer.getChannelData(0);
    outBuf.fill(0);

    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    const pcm16k = downsampleBuffer(inputData, audioCtx.sampleRate, TARGET_SR);
    ws.send(
      JSON.stringify({
        type: "audio",
        source: "system",
        ts: Date.now() / 1000,
        pcm_b64: float32ToBase64(pcm16k),
      })
    );
  };
}

async function stopCapture() {
  dubQueue = [];
  dubPlaying = false;
  if (processorNode) {
    try { processorNode.disconnect(); } catch {}
    processorNode = null;
  }
  if (gainNode) {
    try { gainNode.disconnect(); } catch {}
    gainNode = null;
  }
  if (mediaStream) {
    mediaStream.getTracks().forEach((t) => t.stop());
    mediaStream = null;
  }
  if (audioCtx) {
    try { await audioCtx.close(); } catch {}
    audioCtx = null;
  }
  if (ws) {
    try {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: "flush" }));
      }
      ws.close();
    } catch {}
    ws = null;
  }
  activeTabId = null;
}

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (msg.target !== "offscreen") return;
  if (msg.type === "start") {
    startCapture(msg.streamId, msg.tabId, msg.options || {})
      .then(() => sendResponse({ ok: true }))
      .catch((e) => sendResponse({ ok: false, error: String(e) }));
    return true;
  }
  if (msg.type === "update-config") {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(
        JSON.stringify({
          type: "config",
          language: msg.options.language,
          style: msg.options.style,
          speak_vi: Boolean(msg.options.speakVi),
          tts_voice: msg.options.voice,
        })
      );
    }
    sendResponse({ ok: true });
    return false;
  }
  if (msg.type === "stop") {
    stopCapture().then(() => sendResponse({ ok: true }));
    return true;
  }
});
