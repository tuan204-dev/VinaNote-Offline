"""Model-free checks for the Vietnamese-output rules in models.py. Run: .venv/bin/python test_translation_rules.py"""
from models import (
    KEEP_LATIN_WORDS, _VI_SYLLABLE_SHAPE, _bpe_byte_decoder, _is_foreign_script,
    detect_script_lang, has_phrase_loop, is_valid_vi_translation, needs_vi_translation,
)


def test_language_detection():
    for vi in ("Mô hình này chạy hoàn toàn trên máy tính của bạn.", "Thật lòng, tôi không ngờ kết quả lại tốt đến vậy."):
        assert detect_script_lang(vi) == "vi", vi
    # Romance accents used to count as Vietnamese, so these lines were never translated.
    for foreign in ("¿Qué está pasando aquí? No lo entiendo.", "C'est très intéressant, n'est-ce pas ?",
                    "Não sei o que está acontecendo.", "So the first thing you want to do is open settings."):
        assert detect_script_lang(foreign) != "vi", foreign
        assert needs_vi_translation(foreign), foreign
    assert detect_script_lang("大家好，今天我们来聊一聊这个新手机") == "zh"
    assert not needs_vi_translation("Hello", "vi") and not needs_vi_translation("♪ ♪ ♪")
    assert needs_vi_translation("Bonjour à tous", "fr")


def test_latin_leak_shapes():
    # ASCII forms of real Vietnamese syllables (and their prefixes) must stay writable.
    for syl in "anh cho trong nguyen nghieng khuya truong nguoi duoc quyen giuong hoach xoong ngh tr".split():
        assert _VI_SYLLABLE_SHAPE.fullmatch(syl), syl
    for piece in "th ng nh ch tr kh ph gi qu la an in ly".split():  # 2-letter BPE pieces of Vietnamese
        assert _VI_SYLLABLE_SHAPE.fullmatch(piece), piece
    # Untranslated English seen in real outputs: "baThings", "harder", "sẽ-discuss", "như-idle" (= "-id" + "le").
    for word in "things harder discuss with model button subscribe id is ed er".split():
        assert not _VI_SYLLABLE_SHAPE.fullmatch(word) and word not in KEEP_LATIN_WORDS, word


def test_foreign_script_bytes():
    for text in ("你好", "の", "한국", "Привет", "😀", "，"):
        assert _is_foreign_script(text.encode()), text
    for text in ("Việt Nam", "người", "đường", "ữ", "“GPU” – 4K…", " harder"):
        assert not _is_foreign_script(text.encode()), text
    dec = _bpe_byte_decoder()
    assert len(dec) == 256 and dec["Ġ"] == 0x20  # Qwen/GPT-2 byte-level space


def test_output_checks():
    assert has_phrase_loop("Hàm năng này thực sự rất hữu ích, đặc biệt là khi đi công hàm năng này thực sự rất hữu ích")
    assert not has_phrase_loop("Tính năng này thực ra rất hữu ích, nhất là khi đi công tác.")
    zh = "这个功能其实非常实用，尤其是在出差的时候。"
    assert is_valid_vi_translation(zh, "Tính năng này thực ra rất hữu ích, nhất là khi đi công tác, đặc biệt là những chuyến đi dài ngày.")
    assert not is_valid_vi_translation("We're going to talk about three things today.", "We're going to talk about three things today.")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
