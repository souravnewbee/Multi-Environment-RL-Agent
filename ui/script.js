const API_BASE = "http://localhost:8000";

const queryInput = document.getElementById("queryInput");
const domainSelect = document.getElementById("domainSelect");
const runBtn = document.getElementById("runBtn");
const resultArea = document.getElementById("resultArea");
const historyList = document.getElementById("historyList");
const statusDot = document.getElementById("statusDot");
const statusText = document.getElementById("statusText");
const stages = document.querySelectorAll(".stage");
const stateTableArea = document.getElementById("stateTableArea");
let lastQueriedTask = null;   // used only to highlight what just changed

// ---------- init ----------
checkHealth();
loadDomains();
loadHistory();
loadStateDashboard();

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

async function loadStateDashboard() {
  try {
    const res = await fetch(`${API_BASE}/api/state`);
    const domains = await res.json();
    renderStateDashboard(domains);
  } catch {
    stateTableArea.innerHTML = `<p class="empty-msg small">Couldn't load live state — start the API on :8000.</p>`;
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
      const detail = err.detail || "Request failed";
      // 422 = the pipeline is working correctly, it just needs more detail
      // from you — render that as a clarifying question, not a hard error.
      if (res.status === 422) {
        renderClarification(detail);
        loadHistory();
        loadStateDashboard();
        return;
      }
      throw new Error(detail);
    }

    const data = await res.json();

    await animateStage("extract", 250);
    await animateStage("decide", 250);
    await animateStage("explain", 250);

    lastQueriedTask = data.route.task;
    renderResult(data);
    loadHistory();
    loadStateDashboard();
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
function renderClarification(question) {
  resultArea.classList.remove("empty");
  resultArea.innerHTML = `
    <div class="clarify-box">
      <span class="clarify-icon">🤔</span>
      <div>
        <div class="clarify-title">Need a bit more detail</div>
        <p class="clarify-question">${escapeHtml(question)}</p>
        <p class="clarify-hint">Type your answer in the box above and run the query again — the conversation continues from here.</p>
      </div>
    </div>`;
}

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

  // Explanation may contain \n\n-separated parts (main explanation, network
  // suggestion, live state-diff line) — render each as its own paragraph
  // instead of one run-on block.
  const explainParts = (data.explain || "")
    .split("\n\n")
    .filter(Boolean)
    .map((p) => `<p>${escapeHtml(p)}</p>`)
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
      ${explainParts}
    </div>
  `;
}

// Renders every domain's live values in the sidebar, under History -- e.g.
// Hospital's bed/queue/staff counts, Finance's cash/portfolio, Energy's
// battery/solar levels, Agriculture's soil/water readings, Traffic's car
// counts and wait times. This is a persistent dashboard: it always shows
// every domain's current numbers, not just whichever domain the last query
// happened to touch. The task most recently updated by a query is
// highlighted so it's obvious what just changed.
function renderStateDashboard(domainsData) {
  const domainOrder = ["hospital", "traffic", "energy", "finance", "agriculture"];
  const keys = domainOrder.filter((d) => domainsData[d]);

  if (keys.length === 0) {
    stateTableArea.innerHTML = `<p class="empty-msg small">No state data yet.</p>`;
    return;
  }

  const sections = keys
    .map((domainKey) => {
      const domain = domainsData[domainKey];
      const taskBlocks = Object.entries(domain.tasks)
        .map(([taskKey, task]) => {
          const isRecent = taskKey === lastQueriedTask;
          const fieldRows = Object.entries(task.fields)
            .map(
              ([fieldKey, value]) => `
                <div class="state-row-values">
                  <span class="state-field-key">${escapeHtml(fieldKey.replace(/_/g, " "))}</span>
                  <span class="state-val">${escapeHtml(String(value))}</span>
                </div>`
            )
            .join("");
          return `
            <div class="state-task-block ${isRecent ? "state-task-recent" : ""}">
              <span class="state-task-label">${escapeHtml(task.task_label)}</span>
              ${fieldRows}
            </div>`;
        })
        .join("");

      return `
        <div class="state-domain-block">
          <span class="state-domain-label">${escapeHtml(domain.label)}</span>
          ${taskBlocks}
        </div>`;
    })
    .join("");

  stateTableArea.innerHTML = sections;
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