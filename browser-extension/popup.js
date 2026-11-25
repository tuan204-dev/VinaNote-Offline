const statusPill = document.getElementById("statusPill");
const langSel = document.getElementById("langSel");
const styleSel = document.getElementById("styleSel");
const speakChk = document.getElementById("speakChk");
const voiceSel = document.getElementById("voiceSel");
const toggleBtn = document.getElementById("toggleBtn");
const msgBox = document.getElementById("msgBox");

let activeTab = null;
let isTranslatingThisTab = false;

function currentOpts() {
  return {
    language: langSel.value,
    style: styleSel.value,
    speakVi: speakChk.checked,
    voice: voiceSel.value,
  };
}

async function init() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  activeTab = tab;

  chrome.runtime.sendMessage({ type: "get-state", tabId: tab?.id }, (res) => {
    if (!res?.ok || !res?.engine?.ready) {
      statusPill.textContent = res?.ok ? "Đang tải Model" : "Chưa bật Engine";
      statusPill.classList.add("err");
      msgBox.textContent = res?.ok
        ? "Đang chờ tải xong bộ model AI Offline..."
        : "Hãy mở ứng dụng VinaNote Offline trên máy Mac của bạn trước.";
    } else {
      statusPill.textContent = "100% Offline Sẵn sàng";
      statusPill.classList.remove("err");
    }

    if (res?.options) {
      langSel.value = res.options.language || "auto";
      styleSel.value = res.options.style || "natural";
      speakChk.checked = res.options.speakVi !== false;
      voiceSel.value = res.options.voice || "maichi";
    }

    isTranslatingThisTab = Boolean(
      activeTab && (res?.capturingTabId === activeTab.id || res?.isCaptionMode)
    );
    renderButton();
  });
}

function renderButton() {
  if (isTranslatingThisTab) {
    toggleBtn.textContent = "Dừng Dịch Tab Này";
    toggleBtn.classList.add("stop");
  } else {
    toggleBtn.textContent = "Bật Dịch & Thuyết Minh Tab Này";
    toggleBtn.classList.remove("stop");
  }
}

for (const el of [langSel, styleSel, speakChk, voiceSel]) {
  el.addEventListener("change", () => {
    chrome.runtime.sendMessage({ type: "update-options", options: currentOpts() });
  });
}

toggleBtn.addEventListener("click", () => {
  if (!activeTab) return;
  if (isTranslatingThisTab) {
    chrome.runtime.sendMessage({ type: "stop-tab-translate", tabId: activeTab.id }, () => {
      isTranslatingThisTab = false;
      renderButton();
    });
  } else {
    toggleBtn.disabled = true;
    toggleBtn.textContent = "Đang kiểm tra phụ đề / âm thanh...";
    chrome.runtime.sendMessage(
      {
        type: "start-tab-translate",
        tabId: activeTab.id,
        options: currentOpts(),
      },
      (res) => {
        toggleBtn.disabled = false;
        if (res?.ok) {
          isTranslatingThisTab = true;
          renderButton();
          window.close();
        } else {
          msgBox.textContent = `Lỗi: ${res?.error || "Không bật được"}`;
          renderButton();
        }
      }
    );
  }
});

init();
