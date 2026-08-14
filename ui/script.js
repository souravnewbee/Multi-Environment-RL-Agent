const API_BASE = "http://localhost:8000";

const queryInput = document.getElementById("queryInput");
const domainSelect = document.getElementById("domainSelect");
const runBtn = document.getElementById("runBtn");
const resultArea = document.getElementById("resultArea");
const historyList = document.getElementById("historyList");
const statusDot = document.getElementById("statusDot");
const statusText = document.getElementById("statusText");
const stages = document.querySelectorAll(".stage");

// ---------- init ----------
checkHealth();
loadDomains();
loadHistory();

async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/api/health`);
    if (!res.ok) throw new Error();
    statusDot.classList.add("ok");
    statusText.textContent = "backend connected";
  } catch {
    statusDot.classList.add("err");
    statusText.textContent = "backend unreachable — start the API on :8000";
  }
}

async function loadDomains() {
  try {
    const res = await fetch(`${API_BASE}/api/domains`);
    const domains = await res.json();
    for (const [key, val] of Object.entries(domains)) {
      const opt = document.createElement("option");
      opt.value = key;
      opt.textContent = val.label;
      domainSelect.appendChild(opt);
    }
  } catch {
    /* health check already surfaces the error */
  }
}

async function loadHistory() {
  try {
    const res = await fetch(`${API_BASE}/api/history`);
    const items = await res.json();
    renderHistory(items);
  } catch {
    /* silent — history is non-critical */
  }
}

// ---------- run query ----------
runBtn.addEventListener("click", runQuery);
queryInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) runQuery();
});

async function runQuery() {
  const query = queryInput.value.trim();
  if (!query) {
    queryInput.focus();
    return;
  }

  runBtn.disabled = true;
  resetStages();
  resultArea.classList.add("empty");
  resultArea.innerHTML = `<p class="empty-msg">Running pipeline…</p>`;

  try {
    await animateStage("route", 300);
    const res = await fetch(`${API_BASE}/api/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, domain: domainSelect.value || null }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: "Request failed" }));
      throw new Error(err.detail || "Request failed");
    }

    const data = await res.json();

    await animateStage("extract", 250);
    await animateStage("decide", 250);
    await animateStage("explain", 250);

    renderResult(data);
    loadHistory();
  } catch (e) {
    resultArea.innerHTML = `<p class="empty-msg" style="color:#c23b3b;">Error: ${escapeHtml(e.message)}</p>`;
  } finally {
    runBtn.disabled = false;
  }
}

function resetStages() {
  stages.forEach((s) => s.classList.remove("active", "done"));
}

function animateStage(name, delay) {
  return new Promise((resolve) => {
    const el = document.querySelector(`.stage[data-stage="${name}"]`);
    el.classList.add("active");
    setTimeout(() => {
      el.classList.remove("active");
      el.classList.add("done");
      resolve();
    }, delay);
  });
}

// ---------- render ----------
function renderResult(data) {
  resultArea.classList.remove("empty");

  const confPct = Math.round(data.decide.confidence * 100);
  const maxQ = Math.max(...data.decide.q_values.map((q) => Math.abs(q.q_value)), 1);

  const qRows = data.decide.q_values
    .map((q, i) => {
      const widthPct = Math.max(4, (Math.abs(q.q_value) / maxQ) * 100);
      return `
        <div class="qrow">
          <span>${escapeHtml(q.action)}</span>
          <div class="qbar-track"><div class="qbar-fill ${i === 0 ? "best" : ""}" style="width:${widthPct}%"></div></div>
          <span>${q.q_value.toFixed(2)}</span>
        </div>`;
    })
    .join("");

  resultArea.innerHTML = `
    <div class="result-grid">
      <div class="result-card">
        <span class="k">Domain</span>
        <span class="v">${escapeHtml(data.route.domain_label)}</span>
      </div>
      <div class="result-card">
        <span class="k">Task</span>
        <span class="v">${escapeHtml(data.extract.task)}</span>
      </div>
      <div class="result-card">
        <span class="k">Chosen Action</span>
        <span class="v action">${escapeHtml(data.decide.chosen_action)}</span>
      </div>
      <div class="result-card">
        <span class="k">Confidence</span>
        <span class="v">${confPct}%</span>
        <div class="confidence-bar-track"><div class="confidence-bar-fill" style="width:${confPct}%"></div></div>
      </div>
    </div>

    <div class="qvalues">
      <h3>Q-values (top actions)</h3>
      ${qRows}
    </div>

    <div class="explanation">
      <h3>Explanation</h3>
      <p>${escapeHtml(data.explain)}</p>
    </div>
  `;
}

function renderHistory(items) {
  if (!items || items.length === 0) {
    historyList.innerHTML = `<li class="history-empty">No queries yet.</li>`;
    return;
  }
  historyList.innerHTML = items
    .map(
      (h) => `
      <li class="history-item">
        <div class="hq">${escapeHtml(h.query)}</div>
        <div class="hmeta">
          <span class="domain-tag">${escapeHtml(h.domain)}</span>
          <span>${escapeHtml(h.action)} · ${Math.round(h.confidence * 100)}%</span>
        </div>
      </li>`
    )
    .join("");
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}
