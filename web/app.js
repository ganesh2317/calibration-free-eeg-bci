/**
 * Calibration-Free Cross-Subject EEG-BCI Web Demo
 * Client-side interactivity, data fetching, and canvas waveform rendering.
 */

// Application state
const state = {
  subject: "S001",
  model: "DANN (EEGNet Backbone)",
  trialIndex: 0,
  channelMode: "triad",
  benchmarks: null,
  subjectData: null,
  cachedSubjects: {}
};

// DOM Elements
const el = {
  subjectSelect: document.getElementById("subjectSelect"),
  modelSelect: document.getElementById("modelSelect"),
  trialSlider: document.getElementById("trialSlider"),
  trialDisplayBadge: document.getElementById("trialDisplayBadge"),
  btnRandom: document.getElementById("btnRandom"),
  btnNextLeft: document.getElementById("btnNextLeft"),
  btnNextRight: document.getElementById("btnNextRight"),
  channelRadios: document.querySelectorAll("input[name='channelMode']"),
  
  metricSubject: document.getElementById("metricSubject"),
  metricTrialNum: document.getElementById("metricTrialNum"),
  metricGt: document.getElementById("metricGt"),
  metricCueInfo: document.getElementById("metricCueInfo"),
  metricPred: document.getElementById("metricPred"),
  metricStatus: document.getElementById("metricStatus"),
  metricConf: document.getElementById("metricConf"),
  
  probValLeft: document.getElementById("probValLeft"),
  probBarLeft: document.getElementById("probBarLeft"),
  probValRight: document.getElementById("probValRight"),
  probBarRight: document.getElementById("probBarRight"),
  
  foldSubjectName: document.getElementById("foldSubjectName"),
  foldAccVal: document.getElementById("foldAccVal"),
  foldF1Val: document.getElementById("foldF1Val"),
  foldModelName: document.getElementById("foldModelName"),
  
  canvas: document.getElementById("waveformCanvas"),
  tabButtons: document.querySelectorAll(".tab-btn"),
  tabPanes: document.querySelectorAll(".tab-pane"),
  benchmarkTableBody: document.getElementById("benchmarkTableBody"),
  foldMatrixTableBody: document.getElementById("foldMatrixTableBody")
};

// Canvas context
const ctx = el.canvas.getContext("2d");

// Initialize application
async function init() {
  setupEventListeners();
  await loadBenchmarks();
  await loadSubject(state.subject);
  setupCanvas();
}

// Event Listeners
function setupEventListeners() {
  el.subjectSelect.addEventListener("change", async (e) => {
    state.subject = e.target.value;
    state.trialIndex = 0;
    el.trialSlider.value = 1;
    await loadSubject(state.subject);
    updateUI();
  });

  el.modelSelect.addEventListener("change", (e) => {
    state.model = e.target.value;
    updateUI();
  });

  el.trialSlider.addEventListener("input", (e) => {
    state.trialIndex = parseInt(e.target.value, 10) - 1;
    updateUI();
  });

  el.btnRandom.addEventListener("click", () => {
    if (!state.subjectData) return;
    const count = state.subjectData.trials.length;
    state.trialIndex = Math.floor(Math.random() * count);
    el.trialSlider.value = state.trialIndex + 1;
    updateUI();
  });

  el.btnNextLeft.addEventListener("click", () => {
    if (!state.subjectData) return;
    const trials = state.subjectData.trials;
    const cur = state.trialIndex;
    let nextIdx = trials.findIndex((t, idx) => idx > cur && t.ground_truth === 0);
    if (nextIdx === -1) nextIdx = trials.findIndex(t => t.ground_truth === 0);
    if (nextIdx !== -1) {
      state.trialIndex = nextIdx;
      el.trialSlider.value = nextIdx + 1;
      updateUI();
    }
  });

  el.btnNextRight.addEventListener("click", () => {
    if (!state.subjectData) return;
    const trials = state.subjectData.trials;
    const cur = state.trialIndex;
    let nextIdx = trials.findIndex((t, idx) => idx > cur && t.ground_truth === 1);
    if (nextIdx === -1) nextIdx = trials.findIndex(t => t.ground_truth === 1);
    if (nextIdx !== -1) {
      state.trialIndex = nextIdx;
      el.trialSlider.value = nextIdx + 1;
      updateUI();
    }
  });

  el.channelRadios.forEach(radio => {
    radio.addEventListener("change", (e) => {
      state.channelMode = e.target.value;
      drawWaveforms();
    });
  });

  // Tab switching
  el.tabButtons.forEach(btn => {
    btn.addEventListener("click", () => {
      el.tabButtons.forEach(b => b.classList.remove("active"));
      el.tabPanes.forEach(p => p.classList.remove("active"));

      btn.classList.add("active");
      const target = btn.getAttribute("data-tab");
      document.getElementById(target).classList.add("active");
    });
  });

  window.addEventListener("resize", () => {
    setupCanvas();
    drawWaveforms();
  });
}

// Fetch Benchmarks Data
async function loadBenchmarks() {
  try {
    const res = await fetch("data/benchmarks.json");
    state.benchmarks = await res.json();
    renderBenchmarkTable();
    renderFoldMatrix();
  } catch (err) {
    console.error("Failed to load benchmarks.json", err);
  }
}

// Fetch Subject Trial Data
async function loadSubject(subjectId) {
  if (state.cachedSubjects[subjectId]) {
    state.subjectData = state.cachedSubjects[subjectId];
    return;
  }

  try {
    const res = await fetch(`data/${subjectId}.json`);
    const data = await res.json();
    state.cachedSubjects[subjectId] = data;
    state.subjectData = data;
  } catch (err) {
    console.error(`Failed to load data for ${subjectId}`, err);
  }
}

// Update UI
function updateUI() {
  if (!state.subjectData) return;
  const trial = state.subjectData.trials[state.trialIndex];
  if (!trial) return;

  // Sidebar badge
  el.trialDisplayBadge.textContent = `Trial ${state.trialIndex + 1} / 90`;

  // Metric 1: Subject & Fold
  const foldNum = parseInt(state.subject.replace("S", ""), 10);
  el.metricSubject.textContent = `${state.subject} (Fold ${foldNum}/10)`;
  el.metricTrialNum.textContent = `Trial #${state.trialIndex + 1} of 90`;

  // Metric 2: Ground-Truth
  const isLeftGt = trial.ground_truth === 0;
  el.metricGt.innerHTML = isLeftGt 
    ? `<span class="badge-gt badge-left">✋ Left Hand</span>`
    : `<span class="badge-gt badge-right">✋ Right Hand</span>`;
  el.metricCueInfo.textContent = isLeftGt ? "Class 0 (T1 Cue)" : "Class 1 (T2 Cue)";

  // Metric 3 & 4: Model Prediction & Confidence
  const predInfo = trial.predictions[state.model] || {
    pred_class: 0,
    confidence: 0.5,
    probs: [0.5, 0.5],
    is_correct: false
  };

  const isPredLeft = predInfo.pred_class === 0;
  el.metricPred.innerHTML = isPredLeft
    ? `<span class="badge-gt badge-left">✋ Left Hand</span>`
    : `<span class="badge-gt badge-right">✋ Right Hand</span>`;

  const isMatch = predInfo.pred_class === trial.ground_truth;
  el.metricStatus.innerHTML = isMatch
    ? `<span class="badge-status badge-match">✅ MATCH</span>`
    : `<span class="badge-status badge-mismatch">❌ MISMATCH</span>`;

  const confPct = (predInfo.confidence * 100).toFixed(2);
  el.metricConf.textContent = `${confPct}%`;
  el.metricConf.className = "metric-val " + (predInfo.confidence >= 0.6 ? "text-green" : (predInfo.confidence >= 0.5 ? "text-blue" : "text-orange"));

  // Probability Bars
  const pLeft = (predInfo.probs[0] * 100).toFixed(1);
  const pRight = (predInfo.probs[1] * 100).toFixed(1);
  el.probValLeft.textContent = `${pLeft}%`;
  el.probBarLeft.style.width = `${pLeft}%`;
  el.probValRight.textContent = `${pRight}%`;
  el.probBarRight.style.width = `${pRight}%`;

  // Fold Context
  el.foldSubjectName.textContent = state.subject;
  el.foldModelName.textContent = state.model;
  if (state.benchmarks && state.benchmarks[state.model]) {
    const foldEntry = state.benchmarks[state.model].per_fold?.find(f => f.test_subject === state.subject);
    if (foldEntry) {
      el.foldAccVal.textContent = `${(foldEntry.accuracy * 100).toFixed(2)}%`;
      el.foldF1Val.textContent = `${(foldEntry.f1 * 100).toFixed(2)}%`;
    }
  }

  drawWaveforms();
}

// Canvas Waveform Drawing
function setupCanvas() {
  const container = el.canvas.parentElement;
  const dpr = window.devicePixelRatio || 1;
  const width = container.clientWidth;
  const height = container.clientHeight;

  el.canvas.width = width * dpr;
  el.canvas.height = height * dpr;
  ctx.scale(dpr, dpr);
}

function drawWaveforms() {
  if (!state.subjectData) return;
  const trial = state.subjectData.trials[state.trialIndex];
  if (!trial) return;

  const container = el.canvas.parentElement;
  const width = container.clientWidth;
  const height = container.clientHeight;

  ctx.clearRect(0, 0, width, height);

  // Background
  ctx.fillStyle = "#FAFAFA";
  ctx.fillRect(0, 0, width, height);

  // Margins
  const margin = { top: 25, right: 30, bottom: 40, left: 50 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;

  // Gridlines & Time axis
  ctx.strokeStyle = "#E2E8F0";
  ctx.lineWidth = 1;

  // Time grid (0.0s to 4.0s)
  for (let s = 0; s <= 4; s += 1) {
    const x = margin.left + (s / 4.0) * plotWidth;
    ctx.beginPath();
    ctx.moveTo(x, margin.top);
    ctx.lineTo(x, margin.top + plotHeight);
    ctx.stroke();

    // Axis label
    ctx.fillStyle = "#64748B";
    ctx.font = "11px Inter, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(`${s}.0s`, x, margin.top + plotHeight + 18);
  }

  // Horizontal baseline (0 uV)
  const yMid = margin.top + plotHeight / 2;
  ctx.strokeStyle = "#CBD5E1";
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(margin.left, yMid);
  ctx.lineTo(margin.left + plotWidth, yMid);
  ctx.stroke();
  ctx.setLineDash([]);

  // Amplitude scale (+/- 3 standard deviations)
  const yMax = 3.2;
  const yMin = -3.2;

  // Cue Onset vertical line at t=0
  ctx.strokeStyle = "#334155";
  ctx.lineWidth = 2;
  ctx.setLineDash([2, 2]);
  ctx.beginPath();
  ctx.moveTo(margin.left, margin.top);
  ctx.lineTo(margin.left, margin.top + plotHeight);
  ctx.stroke();
  ctx.setLineDash([]);

  // Channels to draw
  const channels = [];
  if (state.channelMode === "triad" || state.channelMode === "c3") {
    channels.push({ data: trial.c3, color: "#1F77B4", label: "C3" });
  }
  if (state.channelMode === "triad") {
    channels.push({ data: trial.cz, color: "#10B981", label: "Cz" });
  }
  if (state.channelMode === "triad" || state.channelMode === "c4") {
    channels.push({ data: trial.c4, color: "#EF4444", label: "C4" });
  }

  // Draw each waveform
  channels.forEach(ch => {
    ctx.strokeStyle = ch.color;
    ctx.lineWidth = 1.8;
    ctx.beginPath();

    const nPoints = ch.data.length;
    for (let i = 0; i < nPoints; i++) {
      const x = margin.left + (i / (nPoints - 1)) * plotWidth;
      const val = Math.max(yMin, Math.min(yMax, ch.data[i]));
      const y = margin.top + plotHeight - ((val - yMin) / (yMax - yMin)) * plotHeight;

      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  });

  // Y-axis label
  ctx.fillStyle = "#64748B";
  ctx.font = "10px Inter, sans-serif";
  ctx.textAlign = "right";
  ctx.fillText("+3.0 SD", margin.left - 8, margin.top + 8);
  ctx.fillText("0 SD", margin.left - 8, yMid + 4);
  ctx.fillText("-3.0 SD", margin.left - 8, margin.top + plotHeight - 2);

  // Axis title
  ctx.save();
  ctx.translate(14, margin.top + plotHeight / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = "center";
  ctx.fillStyle = "#334155";
  ctx.font = "11px Inter, sans-serif";
  ctx.fillText("Normalized Amplitude (Z-Score)", 0, 0);
  ctx.restore();
}

// Render Viva Benchmark Table
function renderBenchmarkTable() {
  if (!state.benchmarks || !el.benchmarkTableBody) return;

  const rows = [];
  for (const [name, b] of Object.entries(state.benchmarks)) {
    const acc = (b.mean_accuracy * 100).toFixed(2);
    const accStd = (b.std_accuracy * 100).toFixed(2);
    const prec = (b.mean_precision * 100).toFixed(2);
    const rec = (b.mean_recall * 100).toFixed(2);
    const f1 = (b.mean_f1 * 100).toFixed(2);
    const kappa = b.mean_kappa.toFixed(3);

    const isDANN = name.includes("DANN");
    rows.push(`
      <tr style="${isDANN ? 'font-weight: 700; background: #F0F9FF;' : ''}">
        <td><strong>${name}</strong></td>
        <td>Zero-shot (None)</td>
        <td class="text-blue">${acc} ± ${accStd}%</td>
        <td>${prec}%</td>
        <td>${rec}%</td>
        <td>${f1}%</td>
        <td>${kappa}</td>
      </tr>
    `);
  }
  el.benchmarkTableBody.innerHTML = rows.join("");
}

// Render Subject-by-Subject Fold Matrix
function renderFoldMatrix() {
  if (!state.benchmarks || !el.foldMatrixTableBody) return;

  const subjects = ["S001", "S002", "S003", "S004", "S005", "S006", "S007", "S008", "S009", "S010"];
  const modelKeys = [
    "CSP + LDA (Baseline)",
    "EEGNet",
    "SpatialCNN",
    "CNN + BiLSTM",
    "DANN (EEGNet Backbone)"
  ];

  const rows = subjects.map(s => {
    const cells = modelKeys.map(mKey => {
      const b = state.benchmarks[mKey];
      const fold = b?.per_fold?.find(f => f.test_subject === s);
      if (!fold) return "<td>-</td>";
      const acc = (fold.accuracy * 100).toFixed(2);
      const isTop = mKey.includes("DANN") || mKey.includes("CSP");
      return `<td class="${isTop ? 'text-blue' : ''}">${acc}%</td>`;
    });

    return `
      <tr>
        <td><strong>${s}</strong></td>
        ${cells.join("")}
      </tr>
    `;
  });

  el.foldMatrixTableBody.innerHTML = rows.join("");
}

// Run app
document.addEventListener("DOMContentLoaded", init);
