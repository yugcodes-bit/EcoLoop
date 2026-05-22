/* ============================================================
   ECO-LOOP Simulator — Main Application Logic
   Modules 1-4 active
   ============================================================ */

console.log("ECO-LOOP Simulator — Modules 1-4 loaded ✓");

// ── Application state ───────────────────────────────────
const appState = {
  selectedSample: null,
  selectedAppleData: null,
  isScanning: false,
  lastReading: null,
  lastPrediction: null,
  scanStartTime: null,
};

// ── API Configuration ──────────────────────────────────
const API_URL = "http://localhost:8000";

// ── Class display info ─────────────────────────────────
const CLASS_DISPLAY = {
  0: { name: "FRESH",       cssClass: "class-fresh"  },
  1: { name: "EARLY ROT",   cssClass: "class-early"  },
  2: { name: "SEVERE ROT",  cssClass: "class-severe" }
};

// ── DOM elements ────────────────────────────────────────
const sampleListEl   = document.getElementById('sampleList');
const scanEmptyEl    = document.getElementById('scanEmpty');
const appleStageEl   = document.getElementById('appleStage');
const appleDisplayEl = document.getElementById('appleDisplay');
const sampleBadgeEl  = document.getElementById('sampleBadge');
const scanAreaEl     = document.getElementById('scanArea');
const statusBarEl    = document.querySelector('.status-bar');
const statusTextEl   = document.querySelector('.status-text');
const runBtn         = document.querySelector('.btn-primary');
const cutBtn         = document.querySelector('.btn-secondary');
const readingsEmptyEl   = document.getElementById('readingsEmpty');
const readingsContentEl = document.getElementById('readingsContent');
const channelsListEl    = document.getElementById('channelsList');
const scanTimeValueEl   = document.getElementById('scanTimeValue');
const resultCardEl       = document.getElementById('resultCard');
const resultClassEl      = document.getElementById('resultClass');
const resultPhEl         = document.getElementById('resultPh');
const resultConfidenceEl = document.getElementById('resultConfidence');
const recipeActionEl     = document.getElementById('recipeAction');
const recipeCNEl         = document.getElementById('recipeCN');
const recipeAmendmentEl  = document.getElementById('recipeAmendment');

// ══════════════════════════════════════════════════════════
// SAMPLE LIST — render the 4 sample cards
// ══════════════════════════════════════════════════════════

function renderSampleList() {
  sampleListEl.innerHTML = '';

  APPLE_SAMPLES.forEach(sample => {
    const card = document.createElement('button');
    card.className = 'sample-card';
    card.dataset.id = sample.id;

    let thumbHTML;
    if (sample.isRandom) {
      thumbHTML = `<div class="sample-thumb mystery-icon">?</div>`;
      card.classList.add('mystery');
    } else {
      thumbHTML = `<div class="sample-thumb">${getAppleSVG(sample.appearance, 48)}</div>`;
    }

    card.innerHTML = `
      ${thumbHTML}
      <div class="sample-info">
        <div class="sample-label">${sample.label}</div>
        <div class="sample-name">${sample.displayName}</div>
        <div class="sample-desc">${sample.description}</div>
      </div>
    `;

    card.addEventListener('click', () => selectSample(sample.id));
    sampleListEl.appendChild(card);
  });
}

// ══════════════════════════════════════════════════════════
// SELECT SAMPLE
// ══════════════════════════════════════════════════════════

function selectSample(sampleId) {
  if (appState.isScanning) return;

  const sample = APPLE_SAMPLES.find(s => s.id === sampleId);
  if (!sample) return;

  let resolved;
  if (sample.isRandom) {
    const random = generateMysteryApple();
    resolved = {
      ...sample,
      appearance: random.appearance,
      truth: random.truth,
      spectralProfile: random.spectralProfile
    };
    console.log("Mystery apple →", resolved.truth.class);
  } else {
    resolved = sample;
  }

  appState.selectedSample = sampleId;
  appState.selectedAppleData = resolved;
  appState.lastReading = null;
  appState.lastPrediction = null;

  document.querySelectorAll('.sample-card').forEach(card => {
    card.classList.toggle('active', card.dataset.id === sampleId);
  });

  scanEmptyEl.style.display  = 'none';
  // Remove verdict overlay if present
  const oldVerdict = document.getElementById('verdictOverlay');
  if (oldVerdict) oldVerdict.remove();
  appleStageEl.style.display = 'flex';

  appleDisplayEl.innerHTML = `
    ${getAppleSVG(resolved.appearance, 240)}
    <svg class="light-paths" viewBox="0 0 280 280" xmlns="http://www.w3.org/2000/svg">
      <path class="light-ray ray-in"        d="M 140,280 Q 135,220 130,180" />
      <path class="light-ray ray-scatter-1" d="M 130,180 Q 110,160 115,200" />
      <path class="light-ray ray-scatter-2" d="M 130,180 Q 150,155 165,190" />
      <path class="light-ray ray-scatter-3" d="M 130,180 Q 125,150 145,170" />
      <path class="light-ray ray-out-1"     d="M 115,200 Q 125,240 145,280" />
      <path class="light-ray ray-out-2"     d="M 165,190 Q 160,240 155,280" />
      <path class="light-ray ray-out-3"     d="M 145,170 Q 150,230 150,280" />
    </svg>
  `;

  sampleBadgeEl.textContent = `${resolved.label} • READY TO SCAN`;
  runBtn.disabled = false;
  cutBtn.disabled = true;

  // Reset readings panel
  readingsEmptyEl.style.display   = 'flex';
  readingsContentEl.style.display = 'none';
  // Hide result card from previous scan
  resultCardEl.style.display = 'none';

  setStatus('SYSTEM READY', '');
}

// ══════════════════════════════════════════════════════════
// STATUS BAR helper
// ══════════════════════════════════════════════════════════

function setStatus(text, state) {
  statusTextEl.textContent = text;
  statusBarEl.classList.remove('scanning', 'complete');
  if (state) statusBarEl.classList.add(state);
}

// ══════════════════════════════════════════════════════════
// RUN SIMULATION
// ══════════════════════════════════════════════════════════

async function runSimulation() {
  if (!appState.selectedAppleData || appState.isScanning) return;

  appState.isScanning = true;
  appState.scanStartTime = Date.now();
  runBtn.disabled = true;
  cutBtn.disabled = true;

  // Hide previous readings
  readingsEmptyEl.style.display   = 'flex';
  readingsContentEl.style.display = 'none';
  channelsListEl.innerHTML = '';

  // Phase 1
  setStatus('INITIALIZING SENSOR...', 'scanning');
  appleStageEl.classList.add('scanning');
  sampleBadgeEl.textContent = `${appState.selectedAppleData.label} • SCANNING...`;
  await wait(400);

  // Phase 2
  setStatus('SCANNING — 18 CHANNELS ACTIVE', 'scanning');
  await wait(2000);

  // Phase 3
  setStatus('PROCESSING DATA...', 'scanning');
  await wait(600);

  // Generate readings based on apple's spectral profile
  const profile = appState.selectedAppleData.spectralProfile;
  const readings = generateSpectralReading(profile);
  appState.lastReading = readings;

  console.log("Generated readings:", readings);

  // Phase 4
  triggerCompletionFlash();
  appleStageEl.classList.remove('scanning');
  setStatus('SCAN COMPLETE', 'complete');
  sampleBadgeEl.textContent = `${appState.selectedAppleData.label} • SCAN COMPLETE`;

  // Show readings panel
// Show readings panel
  const elapsed = ((Date.now() - appState.scanStartTime) / 1000).toFixed(2);
  scanTimeValueEl.textContent = `${elapsed}s`;
  readingsEmptyEl.style.display   = 'none';
  readingsContentEl.style.display = 'flex';

  renderChannelReadings(readings);

  // ─── Phase 5: Call ML model via FastAPI ─────────────────
  setStatus('ANALYZING WITH ML MODEL...', 'scanning');
  await wait(300);

  try {
    const prediction = await callPredictionAPI(readings);
    appState.lastPrediction = prediction;
    displayPrediction(prediction);
    setStatus('PREDICTION COMPLETE', 'complete');
  } catch (err) {
    console.error("Prediction API error:", err);
    displayPredictionError();
    setStatus('API ERROR — IS SERVER RUNNING?', '');
  }

  await wait(500);

  appState.isScanning = false;
  cutBtn.disabled = false;
  runBtn.disabled = false;

  console.log("Scanning complete ✓");
}

// ══════════════════════════════════════════════════════════
// API CALL — send readings to FastAPI prediction endpoint
// ══════════════════════════════════════════════════════════

async function callPredictionAPI(readings) {
  // Convert reading array to the object format FastAPI expects
  const payload = {};
  readings.forEach(r => {
    payload[`ch_${r.wavelength}nm`] = r.value;
  });

  console.log("Sending to API:", payload);

  const response = await fetch(`${API_URL}/predict/normalized`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });

  if (!response.ok) {
    throw new Error(`API returned status ${response.status}`);
  }

  const data = await response.json();
  console.log("API response:", data);
  return data;
}

// ══════════════════════════════════════════════════════════
// DISPLAY PREDICTION — show result card
// ══════════════════════════════════════════════════════════

function displayPrediction(pred) {
  const classInfo = CLASS_DISPLAY[pred.rot_class] || CLASS_DISPLAY[0];

  // Remove any existing class styles, add new one
  resultCardEl.className = 'result-card ' + classInfo.cssClass;

  resultClassEl.textContent      = classInfo.name;
  resultPhEl.textContent         = pred.predicted_pH;
  resultConfidenceEl.textContent = (pred.confidence || 'HIGH').toUpperCase();

  // Compost recommendation
  const rec = pred.compost_recommendation || {};
  recipeActionEl.textContent     = rec.action || '—';
  recipeCNEl.textContent         = rec.cn_ratio || 'N/A';
  recipeAmendmentEl.textContent  = rec.primary_amendment || 'N/A';

  // Show the card
  resultCardEl.style.display = 'block';
}

function displayPredictionError() {
  const resultCardEl       = document.getElementById('resultCard');
const resultClassEl      = document.getElementById('resultClass');
const resultPhEl         = document.getElementById('resultPh');
const resultConfidenceEl = document.getElementById('resultConfidence');
const recipeActionEl     = document.getElementById('recipeAction');
const recipeCNEl         = document.getElementById('recipeCN');
const recipeAmendmentEl  = document.getElementById('recipeAmendment');
}

// ══════════════════════════════════════════════════════════
// RENDER READINGS — 18 channels with animated bars
// ══════════════════════════════════════════════════════════

function renderChannelReadings(readings) {
  channelsListEl.innerHTML = '';

  readings.forEach((reading, index) => {
    const row = document.createElement('div');
    row.className = 'channel-row';
    if (reading.isKeyBand) row.classList.add('key-band');
    row.style.animationDelay = `${index * 0.04}s`;

    const pct = (reading.value * 100).toFixed(1);

    row.innerHTML = `
      <div class="channel-meta">
        <span class="channel-name">${reading.wavelength}nm</span>
        <span class="channel-value">${reading.value.toFixed(3)}</span>
      </div>
      <div class="channel-bar">
        <div class="channel-bar-fill" data-target="${pct}"></div>
      </div>
      ${reading.keyBandInfo
        ? `<div class="channel-info">${reading.keyBandInfo}</div>`
        : ''}
    `;

    channelsListEl.appendChild(row);
  });

  // Animate bars filling after a brief delay
  setTimeout(() => {
    document.querySelectorAll('.channel-bar-fill').forEach(bar => {
      bar.style.width = bar.dataset.target + '%';
    });
  }, 50);
}

// ══════════════════════════════════════════════════════════
// Completion flash
// ══════════════════════════════════════════════════════════

function triggerCompletionFlash() {
  const flash = document.createElement('div');
  flash.className = 'scan-complete-flash';
  scanAreaEl.appendChild(flash);
  setTimeout(() => flash.remove(), 700);
}

// ══════════════════════════════════════════════════════════
// Utility
// ══════════════════════════════════════════════════════════

function wait(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// ══════════════════════════════════════════════════════════
// CUT APPLE — Reveal internal truth + compare with prediction
// ══════════════════════════════════════════════════════════

async function cutApple() {
  if (!appState.selectedAppleData || !appState.lastPrediction) return;

  const apple = appState.selectedAppleData;
  const prediction = appState.lastPrediction;
  const truth = apple.truth;

  cutBtn.disabled = true;
  runBtn.disabled = true;

  // Hide result card during cut animation
  resultCardEl.style.display = 'none';

  // Replace whole apple with cut halves
  appleDisplayEl.innerHTML = `
    <div class="apple-cut" id="appleCut">
      <div class="apple-half left">
        ${getAppleCrossSectionSVG(truth, "left", 200)}
      </div>
      <div class="apple-half right">
        ${getAppleCrossSectionSVG(truth, "right", 200)}
      </div>
    </div>
  `;

  // Brief pause so the user sees the closed apple becoming the cut version
  await wait(150);

  // Trigger the split animation
  const cutEl = document.getElementById('appleCut');
  cutEl.classList.add('split');

  // Wait for split animation to finish, then show verdict
  await wait(800);

  // Determine if prediction was correct
  const predictedClass = prediction.rot_class;
  const actualClass    = truth.classCode;
  const isCorrect      = predictedClass === actualClass;

  showVerdict(prediction, truth, isCorrect);

  setStatus(isCorrect ? 'PREDICTION CORRECT ✓' : 'PREDICTION MISMATCH',
            isCorrect ? 'complete' : '');

  sampleBadgeEl.textContent = `${apple.label} • TRUTH REVEALED`;

  await wait(600);
  runBtn.disabled = false;
}


function showVerdict(prediction, truth, isCorrect) {
  // Remove any existing verdict
  const existing = document.getElementById('verdictOverlay');
  if (existing) existing.remove();

  // Class display info
  const PRED_NAMES   = { 0: "FRESH", 1: "EARLY-ROT", 2: "SEVERE-ROT" };
  const PRED_CSS     = { 0: "fresh", 1: "early", 2: "severe" };

  const predName = PRED_NAMES[prediction.rot_class] || "UNKNOWN";
  const predCss  = PRED_CSS[prediction.rot_class]   || "fresh";
  const truthName = truth.class.toUpperCase();
  const truthCss  = PRED_CSS[truth.classCode] || "fresh";

  const overlay = document.createElement('div');
  overlay.id = 'verdictOverlay';
  overlay.className = 'verdict-overlay ' + (isCorrect ? 'correct' : 'incorrect');

  overlay.innerHTML = `
    <div class="verdict-header">
      <div class="verdict-icon">${isCorrect ? '✓' : '✗'}</div>
      <div>
        <div class="verdict-title">
          ${isCorrect ? 'Prediction Validated' : 'Prediction Mismatch'}
        </div>
        <div class="verdict-subtitle">
          ${isCorrect
            ? 'ML model matched ground truth'
            : 'ML model did not match ground truth'}
        </div>
      </div>
    </div>

    <div class="verdict-comparison">
      <div class="verdict-cell">
        <div class="verdict-cell-label">ML PREDICTION</div>
        <div class="verdict-cell-value ${predCss}">${predName}</div>
      </div>
      <div class="verdict-cell">
        <div class="verdict-cell-label">ACTUAL TRUTH</div>
        <div class="verdict-cell-value ${truthCss}">${truthName}</div>
      </div>
    </div>

    <div class="verdict-description">
      "${truth.internalDescription}"
    </div>
  `;

  scanAreaEl.appendChild(overlay);
}

// ══════════════════════════════════════════════════════════
// INIT
// ══════════════════════════════════════════════════════════

window.addEventListener('DOMContentLoaded', () => {
  renderSampleList();
  runBtn.addEventListener('click', runSimulation);
  cutBtn.addEventListener('click', cutApple);
  console.log("App initialized — all modules loaded");
});
