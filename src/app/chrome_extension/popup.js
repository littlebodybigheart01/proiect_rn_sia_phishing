const API_FEEDBACK_URLS = ["http://localhost:8000/feedback", "http://127.0.0.1:8000/feedback"];
const APP_URL = "http://localhost:8501";

const logo = document.getElementById("logo");
const content = document.getElementById("content");
const modelSelect = document.getElementById("model-select");
const openAppBtn = document.getElementById("open-app");
const alertAudio = document.getElementById("alert-audio");

// Simple smiley as base64 PNG
logo.src = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADAAAAAwCAIAAADYYG7QAAAABnRSTlMA/wD/AP83WBt9AAABGUlEQVR4nO2Z0Q3CMAxFq+2kC6oB0mALbQdO0gI7gQnG1m9h0bqkC8h8H3I9kW0xQzH3b8kFhWQJY0o1xA8oD6k4E5bQ4kZ8PqvO8AqW9B3s9x2+4m3b0d8OQJw3y0gB6Cw7hN4r2pQd5i7i4u5Xv0N+J3m2zvC8g1s4r5gA2Y6t7w2aU8m0zQ3p2A0y0m+3FJpFq3o7z7s0JqB9S+JrO8hQf8qJm0m3bYVJm3v2QJ0p1bKcH3B5g0aQp7yK4X3J8r5WbQh8y9QkO7mCw8s8Xh7gO+8yC9oJ9rK0bKJ1g7n6mQ1m2Hf3oGQkq7k8o3wQq1s6Q6s0f7mA+f8B4Q8y4oZt3WQAAAAASUVORK5CYII=";

function formatTs(ts) {
  if (!ts) return "-";
  try {
    const d = new Date(ts);
    return d.toLocaleString();
  } catch (_) {
    return "-";
  }
}

function normalizeModelChoice(value) {
  const raw = String(value || "auto").toLowerCase().trim();
  if (raw === "standard" || raw === "optimized" || raw === "auto") return raw;
  if (raw === "baseline" || raw === "default" || raw === "normal") return "standard";
  if (raw === "optimised" || raw === "opt") return "optimized";
  return "auto";
}

function render(data) {
  if (!data || !data.label) {
    content.innerHTML = "<div class='label'>NO SCAN YET</div><div class='small'>Select text, right click → Scan for spam</div>";
    return;
  }
  if (data.label === "ERROR") {
    content.innerHTML = `
      <div class='label phish'>ERROR</div>
      <div class='error'>${data.error || "Scan failed"}</div>
      <div class='small'>Check API: http://localhost:8000/health</div>
      <div class='small'>LAST SCAN: ${formatTs(data.ts)}</div>
    `;
    return;
  }

  const label = data.label;
  const score = (data.score ?? 0).toFixed(4);
  const latency = (typeof data.latency_ms === "number") ? data.latency_ms.toFixed(2) : data.latency_ms;
  const cls = label === "PHISH" ? "phish" : (label === "SUSPECT" ? "sus" : "safe");
  const trunc = data.truncated ? "Text truncated to 5000 chars" : "";
  const model = data.model_choice ? data.model_choice.toUpperCase() : "AUTO";

  content.innerHTML = `
    <div class='label ${cls}'>${label}</div>
    <div class='row'><span class='mono'>score</span><span class='mono'>${score}</span></div>
    <div class='row'><span class='mono'>latency</span><span class='mono'>${latency} ms</span></div>
    <div class='row'><span class='mono'>model</span><span class='mono'>${model}</span></div>
    <div class='small'>LAST SCAN: ${formatTs(data.ts)}</div>
    <div class='small'>${trunc}</div>
    <div class='btns'>
      <button id='btn-ok'>CORRECT</button>
      <button id='btn-wrong'>WRONG</button>
    </div>
    <div id='status' class='status'></div>
  `;

  document.getElementById("btn-ok").onclick = () => sendFeedback(data, "correct");
  document.getElementById("btn-wrong").onclick = () => sendFeedback(data, "wrong");
}

async function sendFeedback(data, action) {
  const status = document.getElementById("status");
  status.textContent = "Sending feedback...";
  const payload = {
    scan_id: data.scan_id,
    user_action: action,
    label: data.label,
    score: data.score,
    latency_ms: data.latency_ms
  };

  let ok = false;
  for (const url of API_FEEDBACK_URLS) {
    try {
      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      if (res.ok) { ok = true; break; }
    } catch (e) {
      // try next url
    }
  }

  status.textContent = ok ? "Feedback saved." : "Feedback failed.";
}

openAppBtn.onclick = () => {
  chrome.tabs.create({ url: APP_URL });
};

chrome.storage.local.get(["lastScan", "modelChoice", "playSound"], (res) => {
  const normalized = normalizeModelChoice(res.modelChoice);
  modelSelect.value = normalized;
  chrome.storage.local.set({ modelChoice: normalized });
  render(res.lastScan);

  if (res.playSound && res.lastScan && res.lastScan.label === "PHISH") {
    try { alertAudio.play(); } catch (e) {}
    chrome.storage.local.set({ playSound: false });
  }
});

modelSelect.onchange = () => {
  chrome.storage.local.set({ modelChoice: normalizeModelChoice(modelSelect.value) });
};
