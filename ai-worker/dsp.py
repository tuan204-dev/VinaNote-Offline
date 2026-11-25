"""
Clean-room Digital Signal Processing (DSP) utilities for VinaNote Offline:
1. StreamingVAD: 16 kHz adaptive energy Voice Activity Detector with pre-roll & max-turn split.
2. AcousticEchoSuppressor: Envelope cross-correlation suppressor so speaker playback doesn't echo into mic.
3. TextEchoFilter: Catches delayed speaker echo (e.g. Bluetooth) at the transcript token level.
4. PitchGenderTracker: FFT autocorrelation F0 tracker to pick matching male/female Vietnamese TTS voices.
"""

from __future__ import annotations

import time
import re
from collections import deque
from typing import Callable, List, Optional, Tuple
import numpy as np

SAMPLE_RATE = 16000
FRAME_SAMPLES = 320  # 20 ms @ 16 kHz


class StreamingVAD:
    """20 ms frame adaptive energy VAD with pre-roll and bounded continuous-speech cuts."""

    def __init__(
        self,
        base_threshold: float = 0.007,
        warmup_frames: int = 8,
        trailing_silence_ms: int = 260,
        soft_split_sec: float = 2.0,
        max_segment_sec: float = 3.6,
    ) -> None:
        self.base_threshold = base_threshold
        self.noise_floor = 0.0
        self.warmup_left = warmup_frames
        self.warmup_vals: List[float] = []
        self.residual = np.empty(0, dtype=np.float32)
        self.preroll: deque[np.ndarray] = deque(maxlen=10)  # 200 ms pre-roll
        self.active_frames: List[np.ndarray] = []
        self.onset_count = 0
        self.silent_count = 0
        self.voiced_count = 0
        self.trailing_frames = max(5, round(trailing_silence_ms / 20))
        self.soft_frames = max(40, round(soft_split_sec * 50))
        self.max_frames = max(60, round(max_segment_sec * 50))
        # Id of the segment being collected: its interim results and its final cut share it.
        self.segment_id = 0

    @property
    def gate_threshold(self) -> float:
        return max(self.base_threshold, min(self.noise_floor, 0.018) * 1.75)

    def reset(self) -> None:
        self.residual = np.empty(0, dtype=np.float32)
        self.preroll.clear()
        self.active_frames.clear()
        self.onset_count = 0
        self.silent_count = 0
        self.voiced_count = 0

    def feed(self, pcm: np.ndarray) -> List[Tuple[np.ndarray, float, int]]:
        """Feed 16 kHz mono float32 samples. Returns list of (segment_pcm, duration_sec, segment_id)."""
        if pcm.size == 0:
            return []
        self.residual = np.concatenate((self.residual, pcm.astype(np.float32, copy=False)))
        ready: List[Tuple[np.ndarray, float, int]] = []

        while self.residual.size >= FRAME_SAMPLES:
            frame = self.residual[:FRAME_SAMPLES].copy()
            self.residual = self.residual[FRAME_SAMPLES:]

            centered = frame - float(np.mean(frame))
            rms = float(np.sqrt(np.mean(centered * centered)))

            if self.warmup_left > 0:
                self.warmup_vals.append(rms)
                self.preroll.append(frame)
                self.warmup_left -= 1
                if self.warmup_left == 0:
                    arr = np.asarray(self.warmup_vals, dtype=np.float32)
                    p10, p50, p90 = np.percentile(arr, (10, 50, 90))
                    spread = (p90 - p10) / max(float(p50), 1e-6)
                    floor_est = min(float(np.percentile(arr, 25)), 0.015)
                    self.noise_floor = min(floor_est, self.base_threshold / 1.75) if spread > 0.25 else floor_est
                    # If user/video was already speaking during warmup, promote preroll immediately!
                    voiced_warmup = sum(1 for v in self.warmup_vals if v >= self.gate_threshold)
                    if voiced_warmup >= 3:
                        self.active_frames = list(self.preroll)
                        self.preroll.clear()
                        self.voiced_count = voiced_warmup
                continue

            is_voiced = rms >= self.gate_threshold
            if not self.active_frames:
                if not is_voiced:
                    self.onset_count = 0
                    self.noise_floor = rms if self.noise_floor == 0.0 else (self.noise_floor * 0.97 + rms * 0.03)
                    self.preroll.append(frame)
                    continue
                self.onset_count += 1
                self.preroll.append(frame)
                if self.onset_count < 3:  # require 60 ms onset to ignore clicks
                    continue
                self.active_frames = list(self.preroll)
                self.preroll.clear()
                self.voiced_count = self.onset_count
                self.onset_count = 0
            else:
                self.active_frames.append(frame)

            if is_voiced:
                self.voiced_count += 1
                self.silent_count = 0
            else:
                self.silent_count += 1

            hit_max = len(self.active_frames) >= self.max_frames
            hit_soft_breath = len(self.active_frames) >= self.soft_frames and self.silent_count >= 5  # 100 ms micro-pause after 2.0s!
            hit_silence = self.silent_count >= self.trailing_frames

            if hit_max or hit_soft_breath or hit_silence:
                if self.voiced_count >= 7:  # >= 140 ms of actual speech
                    seg = np.concatenate(self.active_frames)
                    ready.append((seg, seg.size / SAMPLE_RATE, self.segment_id))
                self.segment_id += 1
                # Keep small 160 ms overlap tail only if forced cut mid-phoneme without breath pause
                overlap_tail = self.active_frames[-8:] if (hit_max and is_voiced) else []
                self.active_frames = list(overlap_tail)
                self.voiced_count = len(overlap_tail) // 2
                self.silent_count = 0

        return ready

    def active_audio(self) -> Optional[Tuple[np.ndarray, int]]:
        """(audio so far, segment_id) of the segment still being spoken, for interim results.
        None while the speaker pauses or a forced cut is <0.6 s away: its final is about to come,
        and an interim ASR run then would only delay it."""
        if (not self.active_frames or self.voiced_count < 7 or self.silent_count >= 3
                or len(self.active_frames) > self.max_frames - 30):
            return None
        return np.concatenate(self.active_frames), self.segment_id

    def flush(self) -> Optional[Tuple[np.ndarray, float, int]]:
        seg_id = self.segment_id
        self.segment_id += 1
        if self.active_frames and self.voiced_count >= 7:
            seg = np.concatenate(self.active_frames)
            self.reset()
            return (seg, seg.size / SAMPLE_RATE, seg_id)
        self.reset()
        return None


class _LoudnessTrack:
    def __init__(self) -> None:
        self.log_rms: dict[int, float] = {}
        self.raw_rms: dict[int, float] = {}
        self.latest_ts: Optional[float] = None

    def push(self, pcm: np.ndarray, ended_at: float) -> None:
        n_frames = pcm.size // FRAME_SAMPLES
        if n_frames <= 0:
            return
        start_ts = ended_at - (pcm.size / SAMPLE_RATE)
        for i in range(n_frames):
            sl = pcm[i * FRAME_SAMPLES : (i + 1) * FRAME_SAMPLES]
            rms = float(np.sqrt(np.mean(sl * sl)))
            idx = round((start_ts + (i * 0.02)) / 0.02)
            self.raw_rms[idx] = rms
            self.log_rms[idx] = float(np.log10(rms + 1e-4))
        self.latest_ts = ended_at
        cutoff = round(ended_at / 0.02) - 260
        for old_k in [k for k in self.log_rms if k < cutoff]:
            self.log_rms.pop(old_k, None)
            self.raw_rms.pop(old_k, None)


class AcousticEchoSuppressor:
    """Suppresses laptop speaker playback bleeding into the microphone without headphones."""

    def __init__(self, on_clean_audio: Callable[[np.ndarray, float, str], None]) -> None:
        self.on_clean_audio = on_clean_audio
        self.sys_track = _LoudnessTrack()
        self.mic_track = _LoudnessTrack()
        self.mic_queue: deque[Tuple[np.ndarray, float, float]] = deque()
        self.suppress_until = 0.0

    def push(self, pcm: np.ndarray, ts: float, source: str) -> None:
        if source == "system":
            self.sys_track.push(pcm, ts)
            self.on_clean_audio(pcm, ts, "system")
        else:
            self.mic_track.push(pcm, ts)
            self.mic_queue.append((pcm, ts, time.monotonic()))
        self._drain(force=False)

    def flush(self) -> None:
        self._drain(force=True)

    def _drain(self, force: bool) -> None:
        while self.mic_queue:
            pcm, ts, queued_at = self.mic_queue[0]
            horizon = ts + 0.25
            sys_idle = self.sys_track.latest_ts is None or (self.sys_track.latest_ts < ts - 1.0)
            both_ready = (
                self.mic_track.latest_ts is not None
                and self.sys_track.latest_ts is not None
                and self.mic_track.latest_ts >= horizon
                and self.sys_track.latest_ts >= horizon
            )
            if not (force or sys_idle or both_ready or (time.monotonic() - queued_at > 0.5)):
                return
            self.mic_queue.popleft()
            if not sys_idle and self._correlates(min(horizon, self.mic_track.latest_ts or ts)):
                self.suppress_until = horizon + 0.55
            start_ts = ts - (pcm.size / SAMPLE_RATE)
            cleaned = np.zeros_like(pcm) if start_ts < self.suppress_until else pcm
            self.on_clean_audio(cleaned, ts, "microphone")

    def _correlates(self, until_ts: float) -> bool:
        end_idx = round(until_ts / 0.02)
        idxs = [i for i in range(end_idx - 90, end_idx) if i in self.mic_track.log_rms]
        if len(idxs) < 12:
            return False
        mic_vec = np.asarray([self.mic_track.log_rms[i] for i in idxs], dtype=np.float32)
        mic_loud = mic_vec > (np.percentile(mic_vec, 12) + 0.45)
        min_active = max(6, len(idxs) // 4)
        for lag in range(-2, 12):  # -40 ms .. +220 ms room propagation lag
            if not all((i - lag) in self.sys_track.log_rms for i in idxs):
                continue
            if max(self.sys_track.raw_rms[i - lag] for i in idxs) < 0.004:
                continue
            sys_vec = np.asarray([self.sys_track.log_rms[i - lag] for i in idxs], dtype=np.float32)
            active = mic_loud | (sys_vec > (sys_vec.max() - 1.25))
            if int(active.sum()) < min_active:
                continue
            a, b = mic_vec[active], sys_vec[active]
            if float(a.std()) > 0.04 and float(b.std()) > 0.04:
                if float(np.corrcoef(a, b)[0, 1]) >= 0.58:
                    return True
        return False


_WORD_RE = re.compile(r"\w+", re.UNICODE)


class TextEchoFilter:
    """Drops microphone transcripts that repeat system transcripts heard within the last 12 seconds."""

    def __init__(self) -> None:
        self.sys_history: deque[Tuple[float, List[str]]] = deque()

    def record_system(self, text: str) -> None:
        toks = _WORD_RE.findall(text.lower())
        if not toks:
            return
        now = time.monotonic()
        self.sys_history.append((now, toks))
        while self.sys_history and now - self.sys_history[0][0] > 14.0:
            self.sys_history.popleft()

    def is_mic_echo(self, text: str) -> bool:
        toks = _WORD_RE.findall(text.lower())
        if not toks:
            return False
        now = time.monotonic()
        win = 4.5 if len(toks) <= 2 else 12.0
        heard = {w for ts, ws in self.sys_history if (now - ts) <= win for w in ws}
        if not heard:
            return False
        if len(toks) <= 2:
            return all(w in heard for w in toks)
        return (sum(1 for w in toks if w in heard) / len(toks)) >= 0.62


class PitchGenderTracker:
    """Estimates speaker F0 base pitch (25th percentile) to auto-select male/female Vietnamese voice."""

    def __init__(self) -> None:
        self.history: dict[str, np.ndarray] = {}

    def observe(self, key: str, pcm: np.ndarray) -> None:
        if pcm.size < 640:
            return
        win = 640  # 40 ms
        hop = 320  # 20 ms
        n = 1 + (pcm.size - win) // hop
        frames = np.lib.stride_tricks.sliding_window_view(pcm, win)[::hop][:n]
        frames = frames - frames.mean(axis=1, keepdims=True)
        rms = np.sqrt((frames * frames).mean(axis=1))
        frames = frames[rms >= 0.012]
        if len(frames) == 0:
            return
        spec = np.fft.rfft(frames, 2 * win, axis=1)
        corr = np.fft.irfft(spec * np.conj(spec), axis=1)[:, :win]
        corr /= np.maximum(corr[:, :1], 1e-9)
        lo, hi = int(SAMPLE_RATE / 340), int(SAMPLE_RATE / 75)
        sub = corr[:, lo:hi]
        peaks = sub.max(axis=1)
        lags = lo + np.argmax(sub >= 0.9 * peaks[:, None], axis=1)
        f0 = (SAMPLE_RATE / lags[peaks >= 0.55]).astype(np.float32)
        if f0.size > 0:
            prev = self.history.get(key, np.empty(0, dtype=np.float32))
            self.history[key] = np.concatenate((prev, f0))[-400:]

    def preferred_voice(self, key: str, default_voice: str = "maichi") -> str:
        vals = self.history.get(key)
        if vals is None or vals.size < 6:
            return default_voice
        base_hz = float(np.percentile(vals, 25))
        return "quangminh" if base_hz < 160.0 else "maichi"
