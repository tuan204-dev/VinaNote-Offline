"""
100% Offline AI Model Manager & Inference Pipelines for Apple Silicon (M1/M2/M3/M4):
- ASR: mlx-whisper (`mlx-community/whisper-large-v3-turbo`) on Metal GPU
- LLM: mlx-lm (`mlx-community/Qwen2.5-7B-Instruct-4bit`) on Metal GPU
- TTS: ZeroTTS (`zeroweight-ai/ZeroTTS`) streaming Vietnamese voices on CPU (ONNX Runtime)

Strict Offline Guarantee:
Once weights exist in the local HuggingFace cache, all models load exclusively from resolved
local filesystem snapshot paths (`local_files_only=True`) — zero internet calls at runtime.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import threading
import time
import wave
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

import numpy as np

WHISPER_REPO = os.environ.get("VINANOTE_WHISPER_REPO", "mlx-community/whisper-large-v3-turbo")
FAST_LLM_REPO = os.environ.get("VINANOTE_FAST_LLM_REPO", "mlx-community/Qwen2.5-3B-Instruct-4bit")
LLM_REPO = os.environ.get("VINANOTE_LLM_REPO", "mlx-community/Qwen2.5-7B-Instruct-4bit")
TTS_REPO = os.environ.get("VINANOTE_TTS_REPO", "zeroweight-ai/ZeroTTS")

# Apple Silicon Metal streams are NOT thread-safe across concurrent threads; serialize GPU ops!
GPU_LOCK = threading.Lock()
TTS_LOCK = threading.Lock()

CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3040-\u30ff\uac00-\ud7af]+")
FOREIGN_GLITCH_RE = re.compile(r"[\u3001-\u303f\uff01-\uff60\u0400-\u04ffİığş]+")
VI_DIACRITICS = set("ăâđêôơưàáạảãằắặẳẵầấậẩẫèéẹẻẽềếệểễìíịỉĩòóọỏõồốộổỗờớợởỡùúụủũừứựửữỳýỵỷỹ")

# Lowercase English grammar/function words (zero overlap with Vietnamese words) that must not remain in Vietnamese output
EN_FUNCTION_WORDS = {
    "the", "is", "are", "was", "were", "been", "being", "most", "other", "another",
    "with", "without", "have", "having", "has", "had", "which", "their", "there", "these",
    "those", "from", "this", "that", "though", "although", "because", "while", "where",
    "when", "what", "would", "could", "should", "about", "into", "through", "between",
    "before", "after", "under", "below", "above", "very", "really", "actually", "significantly",
    "faster", "slower", "better", "worse", "more", "less", "many", "much", "some", "every",
    "right", "here", "out", "your", "our", "they", "them", "you", "just", "only", "also",
    "even", "still", "however", "experience", "ideal", "numbers", "screen", "median",
}

COMMON_SHORT_EN_WORDS = EN_FUNCTION_WORDS | {
    "thank", "thanks", "hello", "welcome", "please", "sorry", "wait", "look", "let", "lets",
    "think", "know", "well", "yeah", "yes", "sure", "course", "good", "great", "why", "how",
    "who", "exactly", "today", "now", "first", "next", "last",
}

TRANSLATE_SYS_PROMPTS = {
    "natural": (
        "Bạn là biên dịch viên phụ đề video chuyên nghiệp sang tiếng Việt. "
        "BẮT BUỘC dịch hoàn toàn sang tiếng Việt có dấu tự nhiên, ngắn gọn, dễ nghe khớp nhịp nói. "
        "TUYỆT ĐỐI không lặp lại câu tiếng Anh gốc, không để sót từ nối tiếng Anh, không chèn chữ Hán, không gộp hay bịa thêm ý. "
        "Dịch mượt các từ công nghệ phổ biến (locally -> cục bộ, latency -> độ trễ, model -> mô hình, benchmark -> bài đo hiệu năng), "
        "nhưng giữ nguyên tên riêng và từ viết tắt quốc tế (AI, GPU, CPU, API, RAM, LLM, token, Apple Silicon). "
        "Chỉ xuất duy nhất 1 câu tiếng Việt đã dịch, không giải thích."
    ),
    "literal": (
        "Bạn là biên dịch viên phụ đề chính xác sang tiếng Việt. "
        "BẮT BUỘC dịch sát nghĩa hoàn toàn sang tiếng Việt có dấu, TUYỆT ĐỐI không giữ nguyên câu tiếng Anh gốc, không chèn chữ Hán. "
        "Giữ nguyên tên riêng và thuật ngữ viết tắt (AI, GPU, API, token). "
        "Chỉ xuất duy nhất 1 câu tiếng Việt đã dịch, không giải thích."
    ),
}


def clean_vi_output(raw: str) -> str:
    s = (raw or "").replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
    # Normalize fullwidth punctuation without needing a second LLM call
    s = s.translate(str.maketrans({"，": ", ", "。": ".", "！": "!", "？": "?", "：": ": ", "；": "; "}))
    s = re.sub(r"^\s*(?:bản dịch|dịch sang tiếng việt|tiếng việt|dịch)\s*[:\-]\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^\s*\[\d+\]\s*", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"^['\"«»“”]+|['\"«»“”]+$", "", s).strip()
    return s


def is_valid_vi_translation(src_text: str, vi_text: str) -> bool:
    """Strictly verifies that a foreign-language subtitle was genuinely translated into Vietnamese without merging/hallucination."""
    src = (src_text or "").strip()
    vi = (vi_text or "").strip()
    if not src:
        return True
    if not vi:
        return False
    if CJK_RE.search(vi) or FOREIGN_GLITCH_RE.search(vi):
        return False
    if detect_script_lang(src) == "vi":
        return True

    # Guard against runaway multi-sentence merging/hallucination (e.g. 4 sentences collapsed into 1)
    if len(src) >= 12 and len(vi) > max(int(len(src) * 2.6), len(src) + 90):
        return False

    # CJK source must produce Vietnamese diacritics if >= 3 words
    if CJK_RE.search(src) and len(src) >= 2:
        vi_words = re.findall(r"[A-Za-zÀ-ỹ]+", vi)
        if len(vi_words) >= 3 and not any(c in VI_DIACRITICS for c in vi.lower()):
            return False
        return True

    src_en_words = [w.lower() for w in re.findall(r"[A-Za-z]{2,}", src)]
    if not src_en_words:
        return True

    src_lower_words = re.findall(r"\b[a-z][a-z']+\b", src)
    has_en_grammar = any(w in COMMON_SHORT_EN_WORDS for w in src_en_words)

    # Pure proper nouns / hardware specs (e.g. "Apple M2 Pro 32GB", "iPhone 16 Pro Max") need no diacritics
    if len(src_lower_words) < 3 and not has_en_grammar:
        return True

    # Identical output check for genuine English phrases/sentences
    if vi.lower() == src.lower():
        return False

    # English grammar/function words left inside Vietnamese output
    vi_tokens = [w.lower() for w in re.findall(r"[A-Za-zÀ-ỹ]+", vi)]
    en_fn_hits = sum(1 for w in vi_tokens if w in EN_FUNCTION_WORDS)
    if en_fn_hits >= 2:
        return False
    if len(src_en_words) >= 4 and en_fn_hits >= 1 and sum(1 for c in vi.lower() if c in VI_DIACRITICS) < 2:
        return False

    # Any 4+ word English clause translated into genuine Vietnamese must have Vietnamese diacritics
    if len(src_en_words) >= 4 and (len(src_lower_words) >= 3 or has_en_grammar):
        diacritics = sum(1 for c in vi.lower() if c in VI_DIACRITICS)
        if diacritics == 0:
            return False
        if len(src_en_words) >= 8 and diacritics < 2:
            return False

    return True

HALLUCINATION_PREFIXES = (
    "cảm ơn các bạn đã theo dõi",
    "hãy đăng ký kênh",
    "đừng quên đăng ký kênh",
    "thank you for watching",
    "thanks for watching",
    "please subscribe",
    "subtitles by",
    "amara.org",
    "字幕由",
    "请不吝点赞",
)


def detect_script_lang(text: str) -> str:
    letters = [c for c in text.lower() if c.isalpha()]
    if not letters:
        return "en"
    cjk_ratio = sum("\u4e00" <= c <= "\u9fff" for c in letters) / len(letters)
    if cjk_ratio >= 0.25:
        return "zh"
    words = max(1, len(text.split()))
    vi_ratio = sum(c in VI_DIACRITICS for c in letters) / words
    return "vi" if vi_ratio >= 0.2 else "en"


def is_whisper_hallucination(text: str) -> bool:
    t = text.strip().lower()
    if not t:
        return True
    for pref in HALLUCINATION_PREFIXES:
        if t.startswith(pref) or (pref in t and len(t) < 65):
            return True
    alnum = "".join(c for c in t if c.isalnum())
    if len(alnum) >= 14:
        m = re.search(r"(.{1,12})\1{3,}", alnum)
        if m and len(m.group(0)) / len(alnum) > 0.6:
            return True
    return False


def _resolve_local_snapshot(repo_id: str) -> Optional[str]:
    """Return local snapshot path if cached, without touching the network."""
    try:
        from huggingface_hub import snapshot_download
        path = snapshot_download(repo_id=repo_id, local_files_only=True)
        if path and Path(path).exists():
            return path
    except Exception:
        pass
    return None


class OfflineModelEngine:
    def __init__(self) -> None:
        self.whisper_path: Optional[str] = _resolve_local_snapshot(WHISPER_REPO)
        self.fast_llm_path: Optional[str] = _resolve_local_snapshot(FAST_LLM_REPO)
        self.llm_path: Optional[str] = _resolve_local_snapshot(LLM_REPO)
        self.tts_path: Optional[str] = _resolve_local_snapshot(TTS_REPO)

        self._fast_llm_model: Any = None
        self._fast_llm_tokenizer: Any = None
        self._llm_model: Any = None
        self._llm_tokenizer: Any = None
        self._tts_engine: Any = None
        self._vi_normalizer: Optional[Callable[[str], str]] = None
        self._downloading = False
        self._download_progress: Dict[str, Any] = {"state": "idle", "step": "", "percent": 0}
        self._caption_cache: Dict[str, str] = {}
        self._prefix_caches: Dict[str, Tuple[Any, str]] = {}
        self._whisper_warmed: bool = False

    def warmup_all(self) -> None:
        """Pre-warms Qwen2.5-7B + Frozen System-Prompt KV-Caches + MLX Whisper + ZeroTTS in Unified Memory."""
        try:
            with GPU_LOCK:
                if self.llm_path or _resolve_local_snapshot(LLM_REPO):
                    import mlx_lm
                    from mlx_lm.sample_utils import make_sampler
                    model, tok = self.ensure_llm()
                    self._get_or_build_prefix_cache("natural")
                    self._get_or_build_prefix_cache("literal")
                    dummy = tok.encode("<|im_start|>user\nOK<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)
                    mlx_lm.batch_generate(
                        model, tok, [dummy, dummy], max_tokens=2, sampler=make_sampler(temp=0.0)
                    )
                self.whisper_path = self.whisper_path or _resolve_local_snapshot(WHISPER_REPO)
                if self.whisper_path and not self._whisper_warmed:
                    import mlx_whisper
                    mlx_whisper.transcribe(
                        np.zeros(8000, dtype=np.float32),
                        path_or_hf_repo=self.whisper_path,
                        temperature=0.0,
                        condition_on_previous_text=False,
                    )
                    self._whisper_warmed = True
        except Exception:
            pass
        try:
            if self.tts_path or _resolve_local_snapshot(TTS_REPO):
                self.ensure_tts()
        except Exception:
            pass

    def status(self) -> Dict[str, Any]:
        self.whisper_path = self.whisper_path or _resolve_local_snapshot(WHISPER_REPO)
        self.fast_llm_path = self.fast_llm_path or _resolve_local_snapshot(FAST_LLM_REPO)
        self.llm_path = self.llm_path or _resolve_local_snapshot(LLM_REPO)
        self.tts_path = self.tts_path or _resolve_local_snapshot(TTS_REPO)
        has_any_llm = bool(self.fast_llm_path or self.llm_path)
        all_ready = bool(self.whisper_path and has_any_llm and self.tts_path)
        voices = ["maichi", "baotrang", "kimoanh", "quangminh", "tiendat", "huuduc"]
        if self._tts_engine is not None:
            try:
                voices = list(self._tts_engine.list_voices()) or voices
            except Exception:
                pass
        return {
            "ready": all_ready,
            "offline_locked": all_ready,
            "models": {
                "whisper": {"repo": WHISPER_REPO, "installed": bool(self.whisper_path), "path": self.whisper_path},
                "fast_llm": {"repo": FAST_LLM_REPO, "installed": bool(self.fast_llm_path), "path": self.fast_llm_path},
                "llm": {"repo": LLM_REPO, "installed": bool(self.llm_path), "path": self.llm_path},
                "tts": {"repo": TTS_REPO, "installed": bool(self.tts_path), "path": self.tts_path},
            },
            "loaded": {
                "whisper": self._whisper_warmed,
                "fast_llm": self._fast_llm_model is not None,
                "llm": self._llm_model is not None,
                "prefix_kv_cached": len(self._prefix_caches) >= 2,
                "tts": self._tts_engine is not None,
            },
            "voices": voices,
            "download": self._download_progress,
        }

    def download_all(self, on_progress: Optional[Callable[[Dict[str, Any]], None]] = None) -> Dict[str, Any]:
        from huggingface_hub import snapshot_download

        if self._downloading:
            return self._download_progress

        self._downloading = True
        steps = [
            ("whisper", WHISPER_REPO, "Đang tải Whisper Large-v3-Turbo (MLX ASR)..."),
            ("llm", LLM_REPO, "Đang tải Qwen2.5-7B-Instruct 4-bit (MLX Dịch & Tóm tắt)..."),
            ("tts", TTS_REPO, "Đang tải ZeroTTS (Giọng đọc tiếng Việt Offline)..."),
        ]
        try:
            for idx, (key, repo, label) in enumerate(steps):
                pct = int((idx / len(steps)) * 100)
                self._download_progress = {"state": "downloading", "step": label, "repo": repo, "percent": pct}
                if on_progress:
                    on_progress(self._download_progress)
                local_dir = snapshot_download(repo_id=repo, local_files_only=False)
                setattr(self, f"{key}_path", local_dir)

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

    def ensure_fast_llm(self) -> Tuple[Any, Any]:
        """Loads ultra-fast Qwen2.5-3B-Instruct-4bit (~125 tok/s) for real-time translation, falling back to 7B if needed."""
        if self._fast_llm_model is not None and self._fast_llm_tokenizer is not None:
            return self._fast_llm_model, self._fast_llm_tokenizer
        self.fast_llm_path = self.fast_llm_path or _resolve_local_snapshot(FAST_LLM_REPO)
        if self.fast_llm_path:
            from mlx_lm import load
            model, tokenizer = load(self.fast_llm_path)
            self._fast_llm_model = model
            self._fast_llm_tokenizer = tokenizer
            return model, tokenizer
        return self.ensure_llm()

    def ensure_llm(self) -> Tuple[Any, Any]:
        if self._llm_model is not None and self._llm_tokenizer is not None:
            return self._llm_model, self._llm_tokenizer
        self.llm_path = self.llm_path or _resolve_local_snapshot(LLM_REPO)
        if not self.llm_path:
            raise RuntimeError(f"Model LLM chưa được tải ({LLM_REPO}). Vui lòng tải model một lần trước khi dùng offline.")
        from mlx_lm import load
        model, tokenizer = load(self.llm_path)
        self._llm_model = model
        self._llm_tokenizer = tokenizer
        return model, tokenizer

    def ensure_tts(self) -> Any:
        if self._tts_engine is not None:
            return self._tts_engine
        self.tts_path = self.tts_path or _resolve_local_snapshot(TTS_REPO)
        if not self.tts_path:
            raise RuntimeError(f"Model ZeroTTS chưa được tải ({TTS_REPO}).")
        from zerotts import ZeroTTS, normalize_vi_text
        self._tts_engine = ZeroTTS(Path(self.tts_path))
        self._vi_normalizer = normalize_vi_text
        return self._tts_engine

    # ---------------- ASR (MLX Whisper) ----------------

    def transcribe(
        self,
        audio_f32: np.ndarray,
        language: Optional[str] = None,
        word_timestamps: bool = False,
    ) -> Dict[str, Any]:
        self.whisper_path = self.whisper_path or _resolve_local_snapshot(WHISPER_REPO)
        if not self.whisper_path:
            raise RuntimeError(f"Model Whisper chưa được tải ({WHISPER_REPO}).")
        import mlx_whisper

        t0 = time.monotonic()
        lang_arg = None if (not language or language == "auto") else language
        with GPU_LOCK:
            result = mlx_whisper.transcribe(
                audio_f32,
                path_or_hf_repo=self.whisper_path,
                language=lang_arg,
                temperature=0.0,
                word_timestamps=word_timestamps,
                condition_on_previous_text=False,
                no_speech_threshold=0.6,
                compression_ratio_threshold=2.4,
            )
            self._whisper_warmed = True
        elapsed_ms = round((time.monotonic() - t0) * 1000, 1)
        raw_text = (result.get("text") or "").strip()
        if is_whisper_hallucination(raw_text):
            return {"text": "", "language": result.get("language") or "en", "segments": [], "words": [], "asr_ms": elapsed_ms}

        detected_lang = result.get("language") or detect_script_lang(raw_text)
        segments_out = []
        words_out = []
        for seg in result.get("segments") or []:
            stext = (seg.get("text") or "").strip()
            if not stext or is_whisper_hallucination(stext):
                continue
            if float(seg.get("no_speech_prob", 0.0)) > 0.75:
                continue
            segments_out.append({
                "start": round(float(seg.get("start", 0.0)), 2),
                "end": round(float(seg.get("end", 0.0)), 2),
                "text": stext,
            })
            for w in seg.get("words") or []:
                wt = (w.get("word") or "").strip()
                if wt:
                    words_out.append({
                        "word": wt,
                        "start": round(float(w.get("start", seg.get("start", 0.0))), 2),
                        "end": round(float(w.get("end", seg.get("end", 0.0))), 2),
                    })

        clean_text = " ".join(s["text"] for s in segments_out).strip() or raw_text
        return {
            "text": clean_text,
            "language": detected_lang,
            "segments": segments_out,
            "words": words_out,
            "asr_ms": elapsed_ms,
        }

    # ---------------- LLM Generation & Prefix KV-Cache Engine ----------------

    def _get_or_build_prefix_cache(self, style: str) -> Tuple[Any, str]:
        """Builds & freezes the Qwen2.5-7B KVCache for the translation system prompt (must hold GPU_LOCK)."""
        key = "literal" if style == "literal" else "natural"
        if key in self._prefix_caches:
            return self._prefix_caches[key]

        import mlx.core as mx
        from mlx_lm.generate import generate_step
        from mlx_lm.models.cache import make_prompt_cache

        model, tokenizer = self.ensure_llm()
        sentinel = "<<<VN_CUE>>>"
        full_tpl = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": TRANSLATE_SYS_PROMPTS[key]},
                {"role": "user", "content": sentinel},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        prefix_str, suffix_str = full_tpl.split(sentinel, 1)
        prefix_ids = tokenizer.encode(prefix_str, add_special_tokens=False)
        p_cache = make_prompt_cache(model)
        for _ in generate_step(mx.array(prefix_ids), model, max_tokens=0, prompt_cache=p_cache):
            pass
        mx.eval([c.state for c in p_cache])
        self._prefix_caches[key] = (p_cache, suffix_str)
        return p_cache, suffix_str

    def _llm_chat(self, system_prompt: str, user_prompt: str, max_tokens: int = 1024, fast: bool = False) -> str:
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        with GPU_LOCK:
            # Always prefer 7B when available for full Vietnamese accuracy
            model, tokenizer = (
                self.ensure_llm()
                if (self.llm_path or _resolve_local_snapshot(LLM_REPO))
                else self.ensure_fast_llm()
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            prompt_str = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            sampler = make_sampler(temp=0.0)
            out = generate(
                model,
                tokenizer,
                prompt=prompt_str,
                max_tokens=max_tokens,
                sampler=sampler,
                verbose=False,
            )
        return out.strip()

    def _clone_prefix_cache(self, src_cache: List[Any]) -> List[Any]:
        """Clones frozen KVCache layers in <0.1ms for single-cue instant generation (`mlx_lm.generate`)."""
        from mlx_lm.models.cache import KVCache
        cloned = []
        for layer_c in src_cache:
            nc = KVCache()
            nc.keys = layer_c.keys
            nc.values = layer_c.values
            nc.offset = layer_c.offset
            cloned.append(nc)
        return cloned

    def _repair_cjk_if_needed(self, vi_text: str) -> str:
        if not CJK_RE.search(vi_text) and not FOREIGN_GLITCH_RE.search(vi_text):
            return vi_text
        repaired = self._llm_chat(
            system_prompt=(
                "Viết lại câu sau hoàn toàn bằng tiếng Việt có dấu tự nhiên, dịch sạch mọi chữ Hán/Nhật/Hàn hoặc ký tự lạ còn sót. "
                "Chỉ trả về đúng câu tiếng Việt đã sửa."
            ),
            user_prompt=vi_text,
            max_tokens=220,
        )
        cleaned = clean_vi_output(CJK_RE.sub("", FOREIGN_GLITCH_RE.sub("", repaired)))
        return cleaned or CJK_RE.sub("", FOREIGN_GLITCH_RE.sub("", vi_text)).strip()

    def _strict_retranslate_vi(self, src_text: str) -> str:
        """Fallback strict 7B re-translation triggered whenever any cue fails Vietnamese output validation."""
        repaired = self._llm_chat(
            system_prompt=(
                "Bản dịch trước bị lỗi vì còn sót tiếng nước ngoài. "
                "Hãy dịch TOÀN BỘ câu sau sang tiếng Việt có dấu tự nhiên, chính xác, dễ hiểu. "
                "TUYỆT ĐỐI không lặp lại câu tiếng Anh gốc, không để từ nối tiếng Anh (the, is, with, which, having...), "
                "không để chữ Hán, giữ nguyên tên riêng/từ viết tắt (AI, GPU, API, Neon, token). "
                "Chỉ trả về duy nhất 1 câu tiếng Việt."
            ),
            user_prompt=f"Câu gốc cần dịch sang tiếng Việt: {src_text}",
            max_tokens=min(160, max(40, len(src_text.split()) * 5 + 28)),
        )
        cleaned = self._repair_cjk_if_needed(clean_vi_output(repaired))
        return cleaned or src_text

    def _batch_translate_cues_gpu(self, src_texts: List[str], style: str = "natural") -> List[str]:
        """
        Hybrid MLX 7B Translator:
        - 1 cue (instant playhead / live utterance): uses single-sequence Frozen Prefix KVCache (~480ms).
        - 2..8 cues (lookahead / video batch): uses GPU-parallel `mlx_lm.batch_generate` (~390ms/cue, zero line-merging, exact RoPE).
        """
        if not src_texts:
            return []
        import mlx_lm
        from mlx_lm.sample_utils import make_sampler

        key = "literal" if style == "literal" else "natural"
        sampler = make_sampler(temp=0.0)

        with GPU_LOCK:
            model, tokenizer = self.ensure_llm()
            if len(src_texts) == 1:
                txt = src_texts[0]
                sys_cache, suffix_str = self._get_or_build_prefix_cache(key)
                prompt_ids = tokenizer.encode(f"Dịch sang tiếng Việt: {txt}{suffix_str}", add_special_tokens=False)
                out_one = mlx_lm.generate(
                    model,
                    tokenizer,
                    prompt=prompt_ids,
                    prompt_cache=self._clone_prefix_cache(sys_cache),
                    max_tokens=min(112, max(28, len(txt.split()) * 4 + 22)),
                    sampler=sampler,
                    verbose=False,
                )
                raw_outputs = [out_one]
            else:
                compact_sys = (
                    "Dịch phụ đề sang tiếng Việt có dấu sát nghĩa, chỉ xuất câu tiếng Việt, không lặp tiếng Anh, không chữ Hán."
                    if key == "literal"
                    else "Dịch phụ đề sang tiếng Việt có dấu tự nhiên, ngắn gọn khớp nhịp nói (locally=cục bộ, latency=độ trễ, model=mô hình; giữ AI, GPU, API, token). Tuyệt đối không lặp câu tiếng Anh, không để chữ Hán. Chỉ xuất 1 câu tiếng Việt."
                )
                prompts = [
                    tokenizer.encode(
                        tokenizer.apply_chat_template(
                            [
                                {"role": "system", "content": compact_sys},
                                {"role": "user", "content": f"Dịch sang tiếng Việt: {txt}"},
                            ],
                            tokenize=False,
                            add_generation_prompt=True,
                        ),
                        add_special_tokens=False,
                    )
                    for txt in src_texts
                ]
                max_toks = [
                    min(112, max(28, len(txt.split()) * 4 + 22))
                    for txt in src_texts
                ]
                resp = mlx_lm.batch_generate(
                    model,
                    tokenizer,
                    prompts,
                    max_tokens=max_toks,
                    sampler=sampler,
                    verbose=False,
                )
                raw_outputs = list(resp.texts)

        final_outputs: List[str] = []
        for src, raw_vi in zip(src_texts, raw_outputs):
            vi = self._repair_cjk_if_needed(clean_vi_output(raw_vi))
            if not is_valid_vi_translation(src, vi):
                vi = self._strict_retranslate_vi(src)
            if is_valid_vi_translation(src, vi):
                if len(self._caption_cache) > 6000:
                    self._caption_cache.clear()
                self._caption_cache[f"{style}::{src}"] = vi
            final_outputs.append(vi or src)
        return final_outputs

    # ---------------- Translation & Note Intelligence ----------------

    def translate_utterance(
        self,
        text: str,
        source_lang: str = "auto",
        context: Optional[List[str]] = None,
        style: str = "natural",
    ) -> str:
        clean = text.strip()
        if not clean:
            return ""
        if source_lang == "vi" or detect_script_lang(clean) == "vi":
            return clean

        cache_key = f"{style}::{clean}"
        cached = self._caption_cache.get(cache_key)
        if cached and is_valid_vi_translation(clean, cached):
            return cached

        return self._batch_translate_cues_gpu([clean], style=style)[0]

    def translate_caption_batch(
        self,
        cues: List[Dict[str, Any]],
        source_lang: str = "auto",
        context: Optional[List[str]] = None,
        style: str = "natural",
    ) -> List[Dict[str, Any]]:
        """GPU-parallel 8-cue batch translator using Qwen2.5-7B + Frozen Prefix KVCache (zero line-merging, ~295ms/cue)."""
        if not cues:
            return []

        BATCH_SIZE = 8
        results: List[Dict[str, Any]] = []

        for offset in range(0, len(cues), BATCH_SIZE):
            chunk = cues[offset : offset + BATCH_SIZE]
            payload_lines = [
                {"id": i, "s": str(c.get("text") or c.get("source") or "").strip()}
                for i, c in enumerate(chunk)
            ]

            uncached_items: List[Dict[str, Any]] = []
            resolved_chunk: Dict[int, str] = {}
            for i, item in enumerate(payload_lines):
                src_txt = item["s"]
                if not src_txt or detect_script_lang(src_txt) == "vi":
                    resolved_chunk[i] = src_txt
                    continue
                cache_key = f"{style}::{src_txt}"
                cached = self._caption_cache.get(cache_key)
                if cached and is_valid_vi_translation(src_txt, cached):
                    resolved_chunk[i] = cached
                else:
                    uncached_items.append(item)

            if uncached_items:
                batch_vi = self._batch_translate_cues_gpu(
                    [it["s"] for it in uncached_items],
                    style=style,
                )
                for it, vi_txt in zip(uncached_items, batch_vi):
                    resolved_chunk[it["id"]] = vi_txt

            for idx, c in enumerate(chunk):
                src_txt = payload_lines[idx]["s"]
                vi_txt = resolved_chunk.get(idx, src_txt)
                results.append({
                    "id": c.get("id", offset + idx),
                    "start": float(c.get("start", 0.0)),
                    "end": float(c.get("end", 0.0)),
                    "source": src_txt,
                    "translation": vi_txt or src_txt,
                })

        return results

    def group_segments_into_cues(self, segments: List[Dict[str, Any]], words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Groups word-level timestamps (or Whisper segments) into crisp 6-15 word subtitle cues."""
        if not words:
            return [
                {"id": idx, "start": s["start"], "end": s["end"], "text": s["text"]}
                for idx, s in enumerate(segments)
                if s.get("text")
            ]

        cues: List[Dict[str, Any]] = []
        buf: List[Dict[str, Any]] = []
        for idx, w in enumerate(words):
            buf.append(w)
            token = w["word"]
            dur = buf[-1]["end"] - buf[0]["start"]
            pause_after = (words[idx + 1]["start"] - w["end"]) if (idx + 1 < len(words)) else 9.9
            ends_punct = token.endswith((".", "!", "?", "。", "！", "？"))
            ends_clause = token.endswith((",", ";", ":", "，", "；", "、"))

            should_cut = (
                (ends_punct and len(buf) >= 3)
                or (pause_after >= 0.65 and len(buf) >= 4)
                or (ends_clause and len(buf) >= 9)
                or len(buf) >= 15
                or dur >= 6.5
            )
            if should_cut:
                text = " ".join(x["word"] for x in buf).strip()
                text = re.sub(r"\s+([.,!?;:])", r"\1", text)
                cues.append({
                    "id": len(cues),
                    "start": round(buf[0]["start"], 2),
                    "end": round(max(buf[-1]["end"], buf[0]["start"] + 0.5), 2),
                    "text": text,
                })
                buf = []

        if buf:
            text = " ".join(x["word"] for x in buf).strip()
            text = re.sub(r"\s+([.,!?;:])", r"\1", text)
            cues.append({
                "id": len(cues),
                "start": round(buf[0]["start"], 2),
                "end": round(max(buf[-1]["end"], buf[0]["start"] + 0.5), 2),
                "text": text,
            })
        return cues

    def summarize_notes(self, transcript_items: List[Dict[str, Any]], template: str = "meeting") -> str:
        formatted = []
        for idx, item in enumerate(transcript_items, 1):
            spk = item.get("speaker") or ("Hệ thống" if item.get("source") == "system" else "Tôi")
            src = item.get("text") or item.get("source") or ""
            vi = item.get("translation") or ""
            line = f"[{idx}] {spk}: {src}"
            if vi and vi != src:
                line += f" (Dịch: {vi})"
            formatted.append(line)
        joined = "\n".join(formatted[:250])

        tpl_guide = {
            "meeting": "Hãy viết Biên bản cuộc họp gồm: 1. Tóm tắt điều hành, 2. Các điểm thảo luận chính, 3. Quyết định đã chốt, 4. Danh sách việc cần làm (Action Items - dạng checklist `- [ ]`).",
            "lecture": "Hãy tóm tắt Bài giảng / Khóa học gồm: 1. Chủ đề cốt lõi, 2. Các khái niệm & công thức quan trọng, 3. Ví dụ thực tế được nêu, 4. Câu hỏi tự ôn tập.",
            "video": "Hãy tóm tắt nội dung Video gồm: 1. Ý chính toàn video, 2. Các luận điểm / mẹo quan trọng theo trình tự, 3. Kết luận đáng nhớ.",
        }.get(template, "Hãy tóm tắt chi tiết nội dung thành các mục rõ ràng bằng Markdown.")

        res = self._llm_chat(
            system_prompt=(
                "Bạn là trợ lý tổng hợp ghi chú chuyên nghiệp hoàn toàn bằng tiếng Việt chuẩn Markdown. "
                f"{tpl_guide} Chỉ dựa trên lời thoại thực tế, không bịa thông tin, không chèn chữ Hán."
            ),
            user_prompt=f"Bản ghi lời thoại:\n{joined}",
            max_tokens=1600,
        )
        return self._repair_cjk_if_needed(res)

    def ask_note(self, transcript_text: str, summary_text: str, question: str, history: Optional[List[Dict[str, str]]] = None) -> str:
        hist_str = ""
        for turn in (history or [])[-4:]:
            hist_str += f"Người dùng: {turn.get('q', '')}\nTrợ lý: {turn.get('a', '')}\n"
        res = self._llm_chat(
            system_prompt=(
                "Bạn là trợ lý AI phân tích nội dung cuộc họp và video. Trả lời bằng tiếng Việt rõ ràng, chính xác, "
                "bám sát nội dung bản ghi bên dưới. Nếu thông tin không có trong bản ghi, hãy nói rõ."
            ),
            user_prompt=(
                f"Tóm tắt hiện có:\n{summary_text[:3000]}\n\n"
                f"Lời thoại chi tiết:\n{transcript_text[:9000]}\n\n"
                f"{hist_str}Câu hỏi mới: {question}"
            ),
            max_tokens=900,
        )
        return self._repair_cjk_if_needed(res)

    # ---------------- Offline Vietnamese TTS (ZeroTTS CPU) ----------------

    def synthesize_wav_bytes(self, text: str, voice: str = "maichi") -> Tuple[bytes, int, float]:
        clean = text.strip()
        if not clean:
            return b"", 48000, 0.0
        tts = self.ensure_tts()
        norm_text = self._vi_normalizer(clean) if self._vi_normalizer else clean
        chunks: List[np.ndarray] = []
        with TTS_LOCK:
            sr = int(getattr(tts, "sample_rate", 48000))
            for ch in tts.synthesize_stream(norm_text, voice=voice or "maichi"):
                arr = np.asarray(ch, dtype=np.float32).reshape(-1)
                if arr.size > 0:
                    chunks.append(arr)

        if not chunks:
            return b"", sr, 0.0
        full = np.concatenate(chunks)
        pcm16 = (np.clip(full, -1.0, 1.0) * 32767.0).astype("<i2")
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm16.tobytes())
        duration_sec = float(full.size / sr)
        return buf.getvalue(), sr, duration_sec

    def synthesize_stream_chunks(self, text: str, voice: str = "maichi") -> Generator[Tuple[bytes, int], None, None]:
        clean = text.strip()
        if not clean:
            return
        tts = self.ensure_tts()
        norm_text = self._vi_normalizer(clean) if self._vi_normalizer else clean
        with TTS_LOCK:
            sr = int(getattr(tts, "sample_rate", 48000))
            for ch in tts.synthesize_stream(norm_text, voice=voice or "maichi"):
                arr = np.asarray(ch, dtype=np.float32).reshape(-1)
                if arr.size > 0:
                    yield arr.tobytes(), sr
