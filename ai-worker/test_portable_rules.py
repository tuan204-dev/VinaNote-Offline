"""Model-free checks for models_portable.py (Windows backend). Run: python test_portable_rules.py"""
from models_portable import _chatml, unglue_names


def test_unglue_names():
    assert unglue_names("Chào mừng đến với cuộc họpEngineering Sync.") == "Chào mừng đến với cuộc họp Engineering Sync."
    assert unglue_names("Mô hình chạy trênGPU.") == "Mô hình chạy trên GPU."
    for kept in ("iPhone mới chạy macOS và YouTube.", "ĐẠT CHUẨN VIỆT NAM", "VinaNote Offline", "Đầu tiên, bạn mở cài đặt."):
        assert unglue_names(kept) == kept, kept


def test_chatml_matches_qwen_template():
    assert _chatml("user", "Xin chào") == "<|im_start|>user\nXin chào<|im_end|>\n"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
