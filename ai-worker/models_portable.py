"""
Portable inference backend (Windows / Linux / Intel Mac) with the same API as the MLX `OfflineModelEngine`,
whose Vietnamese output rules, caption grouping, summaries/Q&A prompts and ZeroTTS it inherits:
- ASR: faster-whisper (CTranslate2) `large-v3-turbo` — NVIDIA CUDA when usable, else CPU int8
- LLM: llama.cpp (`llama-cpp-python`) Qwen2.5-7B-Instruct GGUF Q4_K_M — CUDA / Vulkan / CPU per installed wheel
- TTS: ZeroTTS on CPU (ONNX Runtime), unchanged
Offline like the MLX engine: weights load only from the local Hugging Face cache.
"""

from __future__ import annotations

import os
import re
import site
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from models import (
    CJK_RE,
    GPU_LOCK,
    KEEP_LATIN_WORDS,
    LATIN_BAN,
    LATIN_PENALTY,
    TRANSLATE_FEW_SHOT,
    TRANSLATE_SYS_PROMPTS,
    TTS_REPO,
    VI_DIACRITICS,
    OfflineModelEngine,
    _VI_SYLLABLE_SHAPE,
    _is_foreign_script,
    _resolve_local_snapshot,
    clean_vi_output,
    detect_script_lang,
    is_valid_vi_translation,
    is_whisper_hallucination,
)

WHISPER_REPO = os.environ.get("VINANOTE_WHISPER_REPO", "dropbox-dash/faster-whisper-large-v3-turbo")
# "<HF repo>/<file.gguf>" or a local .gguf path. On CPU-only PCs
# "bartowski/Qwen2.5-3B-Instruct-GGUF/Qwen2.5-3B-Instruct-Q4_K_M.gguf" translates about twice as fast.
LLM_GGUF = os.environ.get("VINANOTE_LLM_GGUF", "bartowski/Qwen2.5-7B-Instruct-GGUF/Qwen2.5-7B-Instruct-Q4_K_M.gguf")
# ponytail: fixed context (~0.9 GB KV cache for 7B); longer summaries/Q&A prompts are cut in the middle to fit.
LLM_CTX = 16384
DEFAULT_VOICES = ["maichi", "baotrang", "kimoanh", "quangminh", "tiendat", "huuduc"]
# The worker runs without a console on Windows; console children would each pop up a window.
NO_WINDOW: Dict[str, Any] = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}


def _add_cuda_dll_dirs() -> None:
    """Windows: cuBLAS/cuDNN from the pip `nvidia-*-cu12` wheels (CTranslate2 GPU) are not on the DLL search path."""
    if os.name != "nt":
        return
    for sp in site.getsitepackages():
        for bin_dir in Path(sp, "nvidia").glob("*/bin"):
            os.add_dll_directory(str(bin_dir))
            os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"


_add_cuda_dll_dirs()


def _resolve_gguf(spec: str) -> Optional[str]:
    """Local path of a GGUF given as "<repo>/<file>" (HF cache only, no network) or as a file path."""
    if Path(spec).is_file():
        return spec
    try:
        from huggingface_hub import hf_hub_download

        repo, fname = spec.rsplit("/", 1)
        return hf_hub_download(repo, fname, local_files_only=True)
    except Exception:
        return None


def _cuda_whisper_works(path: str) -> bool:
    """CTranslate2 can abort the whole process when a cuBLAS/cuDNN DLL is missing, so CUDA is tried in a child first."""
    probe = ("import sys, numpy as np, models_portable; from faster_whisper import WhisperModel; "
             "m = WhisperModel(sys.argv[1], device='cuda', compute_type='auto'); "
             "list(m.transcribe(np.zeros(16000, np.float32), language='en', beam_size=1)[0])")
    try:
        return subprocess.run([sys.executable, "-c", probe, path], cwd=Path(__file__).parent,
                              capture_output=True, timeout=300, **NO_WINDOW).returncode == 0
    except Exception:
        return False


def _load_whisper(path: str) -> Tuple[Any, str]:
    import ctranslate2
    from faster_whisper import WhisperModel

    # setup.ps1 -Gpu cpu|vulkan sets VINANOTE_ASR_DEVICE=cpu: a CUDA install left behind must not win.
    if (os.environ.get("VINANOTE_ASR_DEVICE", "auto") != "cpu" and ctranslate2.get_cuda_device_count() > 0
            and _cuda_whisper_works(path)):
        return WhisperModel(path, device="cuda", compute_type="auto"), "cuda"
    threads = max(4, (os.cpu_count() or 8) // 2)  # faster-whisper defaults to 4 CPU threads
    return WhisperModel(path, device="cpu", compute_type="auto", cpu_threads=threads), "cpu"


_GLUED_RE = re.compile(r"(\w*?\w)([A-Z]\w*)")


def unglue_names(vi: str) -> str:
    """"cuộc họpEngineering Sync" -> "cuộc họp Engineering Sync": the Qwen2.5 GGUF sometimes picks a name's
    space-less token right after a Vietnamese word. Only a lowercase word with Vietnamese diacritics is split,
    so "iPhone", "macOS" and all-caps Vietnamese stay intact."""
    return _GLUED_RE.sub(
        lambda m: f"{m[1]} {m[2]}" if m[1][-1].islower() and any(c in VI_DIACRITICS for c in m[1].lower()) else m[0],
        vi,
    )


def _chatml(role: str, content: str) -> str:
    """One Qwen2.5 chat-template turn, as `apply_chat_template` renders it."""
    return f"<|im_start|>{role}\n{content}<|im_end|>\n"


ASSISTANT_PROMPT = "<|im_start|>assistant\n"


class PortableModelEngine(OfflineModelEngine):
    def __init__(self) -> None:
        super().__init__()  # shared caption cache / TTS / download state; MLX model paths are replaced below
        self.whisper_path = _resolve_local_snapshot(WHISPER_REPO)
        self.fast_llm_path = None
        self.llm_path = _resolve_gguf(LLM_GGUF)
        self._whisper: Any = None
        self._llm: Any = None
        self._devices: Dict[str, str] = {}

    # ---------------- Status / download ----------------

    def status(self) -> Dict[str, Any]:
        self.whisper_path = self.whisper_path or _resolve_local_snapshot(WHISPER_REPO)
        self.llm_path = self.llm_path or _resolve_gguf(LLM_GGUF)
        self.tts_path = self.tts_path or _resolve_local_snapshot(TTS_REPO)
        all_ready = bool(self.whisper_path and self.llm_path and self.tts_path)
        voices = DEFAULT_VOICES
        if self._tts_engine is not None:
            try:
                voices = list(self._tts_engine.list_voices()) or voices
            except Exception:
                pass
        return {
            "ready": all_ready,
            "offline_locked": all_ready,
            "backend": "portable",
            "devices": self._devices,  # {"asr": "cuda"|"cpu", "llm": "gpu"|"cpu"} once loaded
            "models": {
                "whisper": {"repo": WHISPER_REPO, "installed": bool(self.whisper_path), "path": self.whisper_path},
                "llm": {"repo": LLM_GGUF, "installed": bool(self.llm_path), "path": self.llm_path},
                "tts": {"repo": TTS_REPO, "installed": bool(self.tts_path), "path": self.tts_path},
            },
            "loaded": {
                "whisper": self._whisper is not None,
                "llm": self._llm is not None,
                "prefix_kv_cached": bool(self._prefix_caches),
                "tts": self._tts_engine is not None,
            },
            "voices": voices,
            "download": self._download_progress,
        }

    def download_all(self, on_progress: Optional[Callable[[Dict[str, Any]], None]] = None) -> Dict[str, Any]:
        from huggingface_hub import hf_hub_download, snapshot_download

        if self._downloading:
            return self._download_progress
        self._downloading = True
        steps = [
            ("whisper_path", lambda: snapshot_download(WHISPER_REPO), "Đang tải Whisper Large-v3-Turbo (faster-whisper ASR)..."),
            # A local .gguf path (e.g. C:\Models\x.gguf) resolves as is; only "<repo>/<file>" specs are fetched.
            ("llm_path", lambda: _resolve_gguf(LLM_GGUF) or hf_hub_download(*LLM_GGUF.rsplit("/", 1)),
             f"Đang tải {Path(LLM_GGUF).stem} (llama.cpp Dịch & Tóm tắt)..."),
            ("tts_path", lambda: snapshot_download(TTS_REPO), "Đang tải ZeroTTS (Giọng đọc tiếng Việt Offline)..."),
        ]
        try:
            for idx, (attr, fetch, label) in enumerate(steps):
                self._download_progress = {"state": "downloading", "step": label, "percent": int(idx / len(steps) * 100)}
                if on_progress:
                    on_progress(self._download_progress)
                setattr(self, attr, fetch())
            self._download_progress = {"state": "completed", "step": "Đã tải xong toàn bộ model Offline!", "percent": 100}
            if on_progress:
                on_progress(self._download_progress)
            return self.status()
        except Exception as exc:
            self._download_progress = {"state": "error", "step": str(exc), "percent": 0}
            if on_progress:
                on_progress(self._download_progress)
            raise
        finally:
            self._downloading = False

    def warmup_all(self) -> None:
        """Loads Qwen (prompt prefix + vocabulary masks), Whisper and ZeroTTS before the first request."""
        try:
            if self.llm_path or _resolve_gguf(LLM_GGUF):
                self._translate_one("Hello everyone.")
                self._translate_one("So the first thing you want to do is open the settings panel and turn it on.")
            if self.whisper_path or _resolve_local_snapshot(WHISPER_REPO):
                self.transcribe_fast(np.zeros(16000, dtype=np.float32), "en")
        except Exception:
            pass
        try:
            if self.tts_path or _resolve_local_snapshot(TTS_REPO):
                self.ensure_tts()
        except Exception:
            pass

    # ---------------- Model loading ----------------

    def ensure_llm(self) -> Any:
        if self._llm is not None:
            return self._llm
        self.llm_path = self.llm_path or _resolve_gguf(LLM_GGUF)
        if not self.llm_path:
            raise RuntimeError(f"Model LLM chưa được tải ({LLM_GGUF}). Vui lòng tải model một lần trước khi dùng offline.")
        from llama_cpp import Llama, llama_supports_gpu_offload

        try:
            self._llm = Llama(model_path=self.llm_path, n_ctx=LLM_CTX, n_gpu_layers=-1, verbose=False)
            self._devices["llm"] = "gpu" if llama_supports_gpu_offload() else "cpu"
        except Exception:  # GPU build without enough VRAM / a working driver: run on CPU instead
            self._llm = Llama(model_path=self.llm_path, n_ctx=LLM_CTX, n_gpu_layers=0, verbose=False)
            self._devices["llm"] = "cpu"
        return self._llm

    def ensure_fast_llm(self) -> Any:
        return self.ensure_llm()

    def _ensure_whisper(self) -> Any:
        if self._whisper is None:
            self.whisper_path = self.whisper_path or _resolve_local_snapshot(WHISPER_REPO)
            if not self.whisper_path:
                raise RuntimeError(f"Model Whisper chưa được tải ({WHISPER_REPO}).")
            self._whisper, self._devices["asr"] = _load_whisper(self.whisper_path)
        return self._whisper

    # ---------------- ASR (faster-whisper) ----------------

    def _whisper_segments(self, audio: np.ndarray, language: Optional[str], **opts: Any) -> Tuple[List[Any], str]:
        """Greedy decoding like the MLX engine. Segments are a lazy generator, so they are drained under the lock."""
        lang = None if (not language or language == "auto") else language
        with GPU_LOCK:
            model = self._ensure_whisper()
            segs, info = model.transcribe(
                audio, language=lang, beam_size=1, temperature=0.0,
                condition_on_previous_text=False, vad_filter=False, **opts,
            )
            segs = list(segs)
            self._whisper_warmed = True
        return segs, lang or info.language

    def transcribe(self, audio_f32: np.ndarray, language: Optional[str] = None, word_timestamps: bool = False) -> Dict[str, Any]:
        t0 = time.monotonic()
        segs, lang = self._whisper_segments(audio_f32, language, word_timestamps=word_timestamps)
        elapsed_ms = round((time.monotonic() - t0) * 1000, 1)
        raw_text = " ".join(s.text.strip() for s in segs).strip()
        if is_whisper_hallucination(raw_text):
            return {"text": "", "language": lang or "en", "segments": [], "words": [], "asr_ms": elapsed_ms}

        segments_out: List[Dict[str, Any]] = []
        words_out: List[Dict[str, Any]] = []
        for seg in segs:
            stext = seg.text.strip()
            if not stext or is_whisper_hallucination(stext) or seg.no_speech_prob > 0.75:
                continue
            segments_out.append({"start": round(seg.start, 2), "end": round(seg.end, 2), "text": stext})
            for w in seg.words or []:
                wt = w.word.strip()
                if wt:
                    words_out.append({"word": wt, "start": round(w.start, 2), "end": round(w.end, 2)})

        return {
            "text": " ".join(s["text"] for s in segments_out).strip() or raw_text,
            "language": lang or detect_script_lang(raw_text),
            "segments": segments_out,
            "words": words_out,
            "asr_ms": elapsed_ms,
        }

    def transcribe_fast(self, audio_f32: np.ndarray, language: Optional[str] = None) -> Dict[str, Any]:
        """
        Live-segment ASR without timings, the same shortcut as the MLX engine: `WhisperModel.transcribe` pads
        every 2-4 s VAD segment to Whisper's 30 s window (~5.5 s of encoder on a 6-thread CPU). Here the
        CTranslate2 encoder runs once over a window cut to the audio (min 10 s) and its output feeds both
        language detection and greedy decoding: ~5x less work, the same text on the EN/ZH/VI test clips.
        """
        from faster_whisper.tokenizer import Tokenizer
        from faster_whisper.transcribe import get_compression_ratio, get_suppressed_tokens

        t0 = time.monotonic()
        lang = None if (not language or language == "auto") else language
        with GPU_LOCK:
            model = self._ensure_whisper()
            fe = model.feature_extractor
            # ponytail: 10 s floor keeps Whisper near its 30 s training window; lower it only after testing noisy audio.
            frames = min(fe.nb_max_frames, max(1000, int((audio_f32.size / fe.sampling_rate + 1.5) * 100)))
            frames += frames % 2  # conv2 has stride 2
            mel = fe(audio_f32, padding=max(0, frames * fe.hop_length - audio_f32.size))[:, :frames]
            enc = model.encode(np.pad(mel, ((0, 0), (0, frames - mel.shape[-1]))))
            if lang is None:
                lang = max(model.model.detect_language(enc)[0], key=lambda p: p[1])[0][2:-2]  # "<|en|>" -> "en"
            tok = Tokenizer(model.hf_tokenizer, model.model.is_multilingual, task="transcribe", language=lang)
            res = model.model.generate(
                enc, [tok.sot_sequence + [tok.no_timestamps]], beam_size=1, max_length=224,
                return_scores=True, return_no_speech_prob=True,
                suppress_blank=True, suppress_tokens=get_suppressed_tokens(tok, [-1]),
            )[0]
            self._whisper_warmed = True
        text = tok.decode([t for t in res.sequences_ids[0] if t < tok.eot]).strip()
        seq_len = len(res.sequences_ids[0])
        avg_logprob = res.scores[0] * seq_len / (seq_len + 1)  # faster-whisper's avg_logprob (length_penalty 1)
        silent = res.no_speech_prob > 0.6 and avg_logprob < -1.0
        if silent or get_compression_ratio(text) > 2.4 or is_whisper_hallucination(text):
            text = ""
        return {"text": text, "language": lang, "segments": [], "words": [],
                "asr_ms": round((time.monotonic() - t0) * 1000, 1)}

    # ---------------- LLM (llama.cpp) ----------------

    def _tokens(self, text: str, special: bool) -> List[int]:
        return self._llm.tokenize(text.encode("utf-8"), add_bos=False, special=special)

    def _get_or_build_prefix_cache(self, style: str) -> Tuple[List[int], List[int]]:
        """Token ids around the cue for the few-shot translation prompt (must hold GPU_LOCK).
        llama.cpp keeps the KV cache of the previous prompt and re-evaluates only what follows the
        longest shared prefix, so the frozen system prompt + examples cost nothing per cue."""
        key = "literal" if style == "literal" else "natural"
        if key not in self._prefix_caches:
            self.ensure_llm()
            prefix = _chatml("system", TRANSLATE_SYS_PROMPTS[key])
            for src_ex, vi_ex in TRANSLATE_FEW_SHOT:
                prefix += _chatml("user", f"Dịch sang tiếng Việt: {src_ex}") + _chatml("assistant", vi_ex)
            prefix += "<|im_start|>user\n"
            self._prefix_caches[key] = (self._tokens(prefix, True), self._tokens("<|im_end|>\n" + ASSISTANT_PROMPT, True))
        return self._prefix_caches[key]

    def _vocab_masks(self, tokenizer: Any, vocab_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """(foreign-script tokens, untranslated-Latin-word tokens) over the GGUF vocabulary, built once."""
        cached = self._vocab_masks_cache
        if cached is not None and cached[0].size == vocab_size:
            return cached
        foreign = np.zeros(vocab_size, dtype=bool)
        latin = np.zeros(vocab_size, dtype=bool)
        for tid in range(vocab_size):
            raw = tokenizer.detokenize([tid])  # raw bytes; control tokens (<|im_end|>...) come back empty
            if _is_foreign_script(raw):
                foreign[tid] = True
            elif raw.isascii():
                # Punctuation is stripped: the model dodged the check with "-idle", "sẽ-discuss".
                word = raw.decode("ascii").strip().strip("-_'.,:;!?\"()[]/").lower()
                latin[tid] = (len(word) >= 2 and word.isalpha() and word not in KEEP_LATIN_WORDS
                              and not _VI_SYLLABLE_SHAPE.fullmatch(word))
        self._vocab_masks_cache = (foreign, latin)
        return self._vocab_masks_cache

    def _vi_logit_bias(self, model: Any, tokenizer: Any, src: str, latin_penalty: float) -> np.ndarray:
        """Additive logits for Vietnamese output (see the MLX engine); `model` and `tokenizer` are the Llama."""
        vocab = int(model.n_vocab())
        foreign, latin = self._vocab_masks(tokenizer, vocab)
        if latin_penalty <= 0:
            return np.where(foreign, -1e9, 0.0).astype(np.float32)
        latin = latin.copy()
        # Names, acronyms and model numbers written in the source ("iPhone", "GPU", "M2") stay writable.
        for word in set(re.findall(r"[A-Za-z0-9][\w+#.-]*", src)):
            if any(c.isupper() or c.isdigit() for c in word):
                for form in (word, " " + word):
                    latin[[i for i in self._tokens(form, False) if i < vocab]] = False
        return np.where(foreign, -1e9, np.where(latin, -latin_penalty, 0.0)).astype(np.float32)

    @staticmethod
    def _bias_processor(bias: np.ndarray) -> Any:
        from llama_cpp import LogitsProcessorList

        return LogitsProcessorList([lambda _input_ids, scores: scores + bias])

    def _translate_one(
        self,
        src: str,
        style: str = "natural",
        on_partial: Optional[Callable[[str], None]] = None,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> Optional[str]:
        """Same contract as the MLX engine: one streamed greedy pass steered to Vietnamese, retried once with
        untranslated Latin words banned; None when `should_stop()` turned true mid-way."""
        units = len(re.findall(r"\w+", CJK_RE.sub(" ", src))) + sum(len(m) for m in CJK_RE.findall(src))
        max_tokens = min(200, 24 + 3 * units)
        vi = ""
        with GPU_LOCK:
            llm = self.ensure_llm()
            prefix_ids, suffix_ids = self._get_or_build_prefix_cache(style)
            # The cue is tokenized without special tokens, so caption text cannot inject chat-template markers.
            prompt = prefix_ids + self._tokens(f"Dịch sang tiếng Việt: {src}", False) + suffix_ids
            for penalty in (LATIN_PENALTY, LATIN_BAN):
                bias = self._vi_logit_bias(llm, llm, src, penalty)
                raw = ""
                for chunk in llm.create_completion(prompt, max_tokens=max_tokens, temperature=0.0, stream=True,
                                                   logits_processor=self._bias_processor(bias)):
                    if should_stop and should_stop():
                        return None
                    piece = chunk["choices"][0]["text"]
                    raw += piece
                    if on_partial and piece.strip():
                        on_partial(unglue_names(clean_vi_output(raw)))
                vi = unglue_names(clean_vi_output(raw))
                if is_valid_vi_translation(src, vi):
                    break
        return vi

    def _llm_chat(self, system_prompt: str, user_prompt: str, max_tokens: int = 1024, fast: bool = False) -> str:
        with GPU_LOCK:
            llm = self.ensure_llm()
            head = self._tokens(_chatml("system", system_prompt) + "<|im_start|>user\n", True)
            tail = self._tokens("<|im_end|>\n" + ASSISTANT_PROMPT, True)
            body = self._tokens(user_prompt, False)
            room = max(256, LLM_CTX - max_tokens - len(head) - len(tail))
            if len(body) > room:  # keep the start and the end (where the Q&A question sits)
                body = body[: room // 2] + body[-(room - room // 2):]
            out = llm.create_completion(head + body + tail, max_tokens=max_tokens, temperature=0.0,
                                        logits_processor=self._bias_processor(self._vi_logit_bias(llm, llm, "", 0.0)))
        return out["choices"][0]["text"].strip()
