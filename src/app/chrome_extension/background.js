const API_URLS = ["http://localhost:8000/scan", "http://127.0.0.1:8000/scan"];
const MAX_CHARS = 5000;
const TIMEOUT_MS = 5000;

chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "scan-for-spam",
    title: "Scan for spam",
    contexts: ["selection"]
  });
});

async function fetchWithTimeout(url, options) {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), TIMEOUT_MS);
  try {
    const res = await fetch(url, { ...options, signal: controller.signal });
    return res;
  } finally {
    clearTimeout(id);
  }
}

function getModelChoice() {
  return new Promise((resolve) => {
    chrome.storage.local.get(["modelChoice"], (res) => {
      resolve(res.modelChoice || "auto");
    });
  });
}

chrome.contextMenus.onClicked.addListener(async (info) => {
  if (info.menuItemId !== "scan-for-spam") return;
  const raw = (info.selectionText || "").trim();
  if (!raw) return;
  const text = raw.length > MAX_CHARS ? raw.slice(0, MAX_CHARS) : raw;
  const modelChoice = await getModelChoice();

  let payload = {
    label: null,
    score: 0,
    latency_ms: 0,
    truncated: raw.length > MAX_CHARS,
    error: null,
    scan_id: null,
    ts: Date.now(),
    model_choice: modelChoice,
    model_path: null
  };

  let lastError = null;
  for (const url of API_URLS) {
    try {
      const res = await fetchWithTimeout(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, model: modelChoice })
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: "Unknown error" }));
        throw new Error(err.detail || "Request failed");
      }
      const data = await res.json();
      payload = {
        label: data.label,
        score: data.score,
        latency_ms: data.latency_ms,
        truncated: data.truncated,
        error: null,
        scan_id: data.scan_id,
        ts: Date.now(),
        model_choice: modelChoice,
        model_path: data.model_path || null
      };
      lastError = null;
      break;
    } catch (err) {
      lastError = err?.message || "Fetch failed";
    }
  }

  if (lastError) {
    payload = {
      label: "ERROR",
      score: 0,
      latency_ms: 0,
      truncated: raw.length > MAX_CHARS,
      error: lastError,
      scan_id: null,
      ts: Date.now(),
      model_choice: modelChoice,
      model_path: null
    };
  }

  const playSound = payload.label === "PHISH";
  chrome.storage.local.set({ lastScan: payload, playSound }, () => {
    if (chrome.action && chrome.action.openPopup) {
      chrome.action.openPopup().catch(() => {});
    }
  });
});
