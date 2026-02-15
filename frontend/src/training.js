/**
 * Training section — live training via Web Worker + pre-trained fallback.
 * Loss curve, checkpoint samples, weight inspector heatmaps.
 */

import { loadWeights, getStateDict } from './gpt.js';
import { get, set, subscribe } from './state.js';
import { t } from './content.js';
import { createChart, FONT_FAMILY, monoTooltip, tooltipWrap } from './echarts-setup.js';
import { drawCanvasHeatmap } from './viz-utils.js';

let checkpointWeights = null;
let currentCheckpointStep = 500;
const CHECKPOINT_STEPS = [1, 10, 50, 100, 200, 300, 400, 500];

function nearestCheckpoint(step) {
  for (let i = CHECKPOINT_STEPS.length - 1; i >= 0; i--) {
    if (CHECKPOINT_STEPS[i] <= step) return CHECKPOINT_STEPS[i];
  }
  return CHECKPOINT_STEPS[0];
}

// --- Weight Inspector (ECharts heatmaps) ---

const WEIGHT_MATRICES = [
  { key: 'wte', tKey: 'weight.wte', dims: [27, 16] },
  { key: 'wpe', tKey: 'weight.wpe', dims: [8, 16] },
  { key: 'layer0.attn_wq', tKey: 'weight.attn_wq', dims: [16, 16] },
  { key: 'layer0.attn_wk', tKey: 'weight.attn_wk', dims: [16, 16] },
  { key: 'layer0.attn_wv', tKey: 'weight.attn_wv', dims: [16, 16] },
  { key: 'layer0.attn_wo', tKey: 'weight.attn_wo', dims: [16, 16] },
  { key: 'lm_head', tKey: 'weight.lm_head', dims: [27, 16] },
];

let weightCharts = {};

function matrixToHeatmapData(matrix) {
  const data = [];
  let absMax = 0;
  for (const row of matrix) for (const v of row) absMax = Math.max(absMax, Math.abs(v));
  for (let r = 0; r < matrix.length; r++) {
    for (let c = 0; c < matrix[0].length; c++) {
      data.push([c, r, matrix[r][c]]);
    }
  }
  return { data, absMax };
}

function renderWeightInspector(stateDict) {
  const container = document.getElementById('weight-inspector');
  if (!stateDict) {
    container.innerHTML = '<p class="inter-empty">No weights loaded</p>';
    disposeWeightCharts();
    return;
  }

  // Create chart containers if not present
  const existingGrid = container.querySelector('.weight-heatmap-grid');
  if (!existingGrid) {
    container.innerHTML = '<div class="weight-heatmap-grid">' + WEIGHT_MATRICES.map(({ key, tKey, dims }) => {
      const label = t(tKey);
      return `<div class="weight-heatmap">
        <div class="heatmap-label">${label}</div>
        <div data-weight-chart="${key}" aria-label="${label} weight matrix"></div>
        <div class="dims">${dims[0]} × ${dims[1]}</div>
      </div>`;
    }).join('') + '</div>';
    // Set dynamic heights via JS
    for (const { key, dims } of WEIGHT_MATRICES) {
      const el = container.querySelector(`[data-weight-chart="${key}"]`);
      if (el) el.style.height = `${Math.max(dims[0] * 10, 160)}px`;
    }
    disposeWeightCharts();
  }

  // Initialize or update ECharts heatmaps
  for (const { key, dims } of WEIGHT_MATRICES) {
    const el = container.querySelector(`[data-weight-chart="${key}"]`);
    if (!el || !stateDict[key]) continue;

    if (!weightCharts[key]) {
      weightCharts[key] = createChart(el);
    }

    const { data, absMax } = matrixToHeatmapData(stateDict[key]);
    const rows = dims[0];
    const cols = dims[1];

    const showLabels = rows > 4;
    const labelInterval = rows > 16 ? 1 : 0;

    weightCharts[key].setOption({
      tooltip: {
        formatter: (params) => {
          const [col, row, val] = params.value;
          return monoTooltip(`${key}[${row}][${col}]`, val.toFixed(4));
        },
      },
      xAxis: {
        type: 'category',
        data: Array.from({ length: cols }, (_, i) => i),
        axisLabel: { show: showLabels, fontSize: 9, fontFamily: 'monospace', color: '#6B7585', interval: labelInterval },
        axisTick: { show: false },
        splitArea: { show: false },
      },
      yAxis: {
        type: 'category',
        data: Array.from({ length: rows }, (_, i) => i),
        axisLabel: { show: showLabels, fontSize: 9, fontFamily: 'monospace', color: '#6B7585', interval: labelInterval },
        axisTick: { show: false },
        splitArea: { show: false },
        inverse: true,
      },
      visualMap: {
        show: true,
        min: -absMax,
        max: absMax,
        orient: 'horizontal',
        left: 'center',
        bottom: 0,
        itemWidth: 12,
        itemHeight: 60,
        text: ['+', '\u2212'],
        textStyle: { color: '#6B7585', fontSize: 9 },
        inRange: {
          color: ['#F87171', '#3D1515', '#141926', '#152850', '#5B8DEF'],
        },
      },
      series: [{
        type: 'heatmap',
        data,
        emphasis: {
          itemStyle: { shadowBlur: 6, shadowColor: 'rgba(91, 141, 239, 0.4)' },
        },
        animationDuration: 400,
      }],
      grid: { left: showLabels ? 24 : 4, right: 8, top: 8, bottom: 28 },
    }, true);
  }
}

function disposeWeightCharts() {
  for (const chart of Object.values(weightCharts)) chart.dispose();
  weightCharts = {};
}

// --- Weight Evolution Filmstrip ---
let filmstripPlaying = false;
let filmstripTimer = null;
let filmstripDiffMode = false;
let activeFilmstripStep = 500;

const FILMSTRIP_MATRICES = [
  { key: 'wte', label: 'Token Embed', dims: [27, 16] },
  { key: 'wpe', label: 'Pos Embed', dims: [8, 16] },
  { key: 'layer0.attn_wq', label: 'W_Q', dims: [16, 16] },
  { key: 'layer0.attn_wv', label: 'W_V', dims: [16, 16] },
  { key: 'lm_head', label: 'LM Head', dims: [27, 16] },
];

function drawHeatmapDiff(canvas, matrix, prevMatrix) {
  const rows = matrix.length;
  const cols = matrix[0].length;
  canvas.width = cols;
  canvas.height = rows;
  const ctx = canvas.getContext('2d');

  // Find max absolute diff for color scale
  let maxDiff = 0;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const diff = Math.abs(matrix[r][c] - (prevMatrix ? prevMatrix[r][c] : 0));
      if (diff > maxDiff) maxDiff = diff;
    }
  }
  if (maxDiff < 0.001) maxDiff = 0.001;

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const diff = Math.abs(matrix[r][c] - (prevMatrix ? prevMatrix[r][c] : 0));
      const intensity = diff / maxDiff;
      // Orange scale for changes
      const red = Math.round(30 + intensity * 221);
      const green = Math.round(20 + intensity * 126);
      const blue = Math.round(15 + intensity * 45);
      ctx.fillStyle = `rgb(${red}, ${green}, ${blue})`;
      ctx.fillRect(c, r, 1, 1);
    }
  }
}

function renderWeightFilmstrip(container, allCheckpointWeights, vocabSize) {
  if (!allCheckpointWeights) {
    container.innerHTML = '<p class="inter-empty">Loading checkpoint weights...</p>';
    return;
  }

  const steps = CHECKPOINT_STEPS.filter(s => allCheckpointWeights[String(s)]);
  if (steps.length === 0) return;

  // Controls: play button + step buttons + diff toggle
  let html = '<div class="filmstrip-controls">';
  html += `<button class="filmstrip-play" id="filmstrip-play-btn" aria-pressed="false" aria-label="Play weight evolution" title="Auto-advance through checkpoints">&#9654;</button>`;
  html += '<div class="filmstrip-step-btns">';
  for (const s of steps) {
    const active = s === activeFilmstripStep ? ' active' : '';
    html += `<button class="filmstrip-step-btn${active}" data-filmstrip-step="${s}">Step ${s}</button>`;
  }
  html += '</div>';
  html += `<button class="filmstrip-diff-toggle" id="filmstrip-diff-btn" aria-pressed="${filmstripDiffMode}" title="Highlight weight changes between checkpoints">Diff</button>`;
  html += '</div>';

  // One row per matrix, with all checkpoints as small canvases
  for (const { key, label } of FILMSTRIP_MATRICES) {
    html += `<div class="filmstrip-matrix-label">${label}</div>`;
    html += '<div class="filmstrip-canvases">';
    for (const s of steps) {
      const active = s === activeFilmstripStep ? ' active' : '';
      html += `<div class="filmstrip-frame${active}" data-filmstrip-step="${s}">`;
      html += `<canvas data-filmstrip-key="${key}" data-filmstrip-step="${s}" aria-label="${label} at step ${s}"></canvas>`;
      html += `<div class="filmstrip-frame-label">${s}</div>`;
      html += '</div>';
    }
    html += '</div>';
  }

  container.innerHTML = html;

  // Draw all canvases
  drawFilmstripCanvases(container, allCheckpointWeights, vocabSize, steps);

  // Event delegation for step buttons
  container.addEventListener('click', (e) => {
    const stepBtn = e.target.closest('[data-filmstrip-step]');
    const playBtn = e.target.closest('#filmstrip-play-btn');
    const diffBtn = e.target.closest('#filmstrip-diff-btn');

    if (stepBtn && !stepBtn.classList.contains('filmstrip-frame')) {
      const step = parseInt(stepBtn.dataset.filmstripStep);
      selectFilmstripStep(container, allCheckpointWeights, vocabSize, steps, step);
    } else if (playBtn) {
      toggleFilmstripPlay(container, allCheckpointWeights, vocabSize, steps);
    } else if (diffBtn) {
      filmstripDiffMode = !filmstripDiffMode;
      diffBtn.setAttribute('aria-pressed', String(filmstripDiffMode));
      drawFilmstripCanvases(container, allCheckpointWeights, vocabSize, steps);
    }
  });
}

function selectFilmstripStep(container, allCheckpointWeights, vocabSize, steps, step) {
  activeFilmstripStep = step;

  // Update active class on step buttons and frames
  container.querySelectorAll('.filmstrip-step-btn').forEach(btn => {
    btn.classList.toggle('active', parseInt(btn.dataset.filmstripStep) === step);
  });
  container.querySelectorAll('.filmstrip-frame').forEach(frame => {
    frame.classList.toggle('active', parseInt(frame.dataset.filmstripStep) === step);
  });

  // Load these weights into the model
  if (allCheckpointWeights[String(step)]) {
    loadWeights(allCheckpointWeights[String(step)], vocabSize);
    set('weightsVersion', (get('weightsVersion') || 0) + 1);
  }

  // Update slider to match
  const slider = document.getElementById('step-slider');
  const stepValue = document.getElementById('step-value');
  slider.value = step - 1;
  stepValue.textContent = step;
  currentCheckpointStep = step;
  set('trainingStep', step);
}

function toggleFilmstripPlay(container, allCheckpointWeights, vocabSize, steps) {
  const btn = container.querySelector('#filmstrip-play-btn');
  filmstripPlaying = !filmstripPlaying;
  btn.setAttribute('aria-pressed', String(filmstripPlaying));
  btn.innerHTML = filmstripPlaying ? '&#9646;&#9646;' : '&#9654;';

  if (filmstripPlaying) {
    let idx = steps.indexOf(activeFilmstripStep);
    if (idx < 0 || idx >= steps.length - 1) idx = -1;

    filmstripTimer = setInterval(() => {
      idx++;
      if (idx >= steps.length) {
        idx = 0;
      }
      selectFilmstripStep(container, allCheckpointWeights, vocabSize, steps, steps[idx]);

      // Update checkpoints display
      const checkpointContainer = document.getElementById('checkpoint-container');
      if (checkpointContainer) {
        // Re-render handled by selectFilmstripStep via set('trainingStep')
      }
    }, 400);
  } else {
    clearInterval(filmstripTimer);
    filmstripTimer = null;
  }
}

function drawFilmstripCanvases(container, allCheckpointWeights, vocabSize, steps) {
  for (const { key } of FILMSTRIP_MATRICES) {
    for (let si = 0; si < steps.length; si++) {
      const s = steps[si];
      const canvas = container.querySelector(`canvas[data-filmstrip-key="${key}"][data-filmstrip-step="${s}"]`);
      if (!canvas) continue;

      // Temporarily load these weights to get the state dict
      loadWeights(allCheckpointWeights[String(s)], vocabSize);
      const sd = getStateDict();
      const matrix = sd[key];
      if (!matrix) continue;

      if (filmstripDiffMode && si > 0) {
        loadWeights(allCheckpointWeights[String(steps[si - 1])], vocabSize);
        const prevSd = getStateDict();
        drawHeatmapDiff(canvas, matrix, prevSd[key]);
      } else {
        drawCanvasHeatmap(canvas, matrix);
      }
    }
  }

  // Restore active checkpoint weights
  if (allCheckpointWeights[String(activeFilmstripStep)]) {
    loadWeights(allCheckpointWeights[String(activeFilmstripStep)], vocabSize);
  }
}

// --- Loss Curve (ECharts) ---
let lossCurveChart = null;

function lossCurveBaseOption(trainingLog) {
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const steps = trainingLog.map(d => d.step);
  const losses = trainingLog.map(d => d.loss);
  const lrs = trainingLog.map(d => d.lr);
  const maxLoss = Math.ceil(Math.max(...losses));

  return {
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'cross', lineStyle: { color: '#E8ECF1', opacity: 0.4 } },
      formatter: (params) => {
        const step = params[0]?.axisValue;
        const loss = params.find(p => p.seriesName === 'Loss');
        const lr = params.find(p => p.seriesName === 'Learning Rate');
        return tooltipWrap(`Step <strong>${step}</strong><br/>Loss: <strong>${loss ? loss.value.toFixed(4) : '—'}</strong><br/>LR: <strong>${lr ? lr.value.toFixed(6) : '—'}</strong>`);
      },
    },
    grid: { left: 60, right: 60, top: 20, bottom: 40 },
    xAxis: {
      type: 'category',
      data: steps,
      axisLabel: { interval: 99, formatter: v => v > 0 ? v : '' },
    },
    yAxis: [
      { type: 'value', name: 'Loss', nameTextStyle: { color: '#5B8DEF', fontWeight: 600, fontSize: 12 }, max: maxLoss, splitNumber: 4 },
      { type: 'value', name: 'Learning Rate', nameTextStyle: { color: '#9B7AEA', fontWeight: 600, fontSize: 12 }, position: 'right', splitLine: { show: false } },
    ],
    series: [
      {
        name: 'Loss',
        type: 'line',
        data: losses,
        yAxisIndex: 0,
        showSymbol: false,
        lineStyle: { color: '#5B8DEF', width: 2 },
        itemStyle: { color: '#5B8DEF' },
        animationDuration: reduceMotion ? 0 : 2000,
        markPoint: {
          symbol: 'circle',
          symbolSize: 10,
          data: CHECKPOINT_STEPS.map(cp => {
            const entry = trainingLog[cp - 1];
            if (!entry) return null;
            return { coord: [cp - 1, entry.loss], name: String(cp), value: cp };
          }).filter(Boolean),
          label: { show: true, position: 'top', fontSize: 9, color: '#6B7585', formatter: p => p.value },
          itemStyle: { color: '#5B8DEF', borderColor: '#0B0F1A', borderWidth: 1.5 },
        },
      },
      {
        name: 'Learning Rate',
        type: 'line',
        data: lrs,
        yAxisIndex: 1,
        showSymbol: false,
        lineStyle: { color: '#9B7AEA', width: 1.5, opacity: 0.5 },
        itemStyle: { color: '#9B7AEA' },
        animationDuration: reduceMotion ? 0 : 2000,
      },
    ],
  };
}

function createPretrainedChart(container, trainingLog) {
  container.innerHTML = '<div class="loss-curve-chart" aria-label="Training loss curve over 500 steps"></div>';
  const el = container.querySelector('.loss-curve-chart');
  el.style.height = '300px';

  lossCurveChart = createChart(el);
  const option = lossCurveBaseOption(trainingLog);
  lossCurveChart.setOption(option);
  return lossCurveChart;
}

function setScrubberPosition(chart, step, trainingLog) {
  const entry = trainingLog[step - 1];
  if (!chart || !entry) return;
  chart.setOption({
    series: [{
      markLine: {
        silent: true,
        symbol: 'none',
        lineStyle: { color: '#E8ECF1', type: 'dashed', width: 1, opacity: 0.4 },
        data: [{ xAxis: step - 1 }],
        label: { show: false },
        animation: false,
      },
    }],
  });
}

function createLiveChart(container) {
  container.innerHTML = '<div class="loss-curve-chart" aria-label="Live training loss curve"></div>';
  const el = container.querySelector('.loss-curve-chart');
  el.style.height = '300px';

  lossCurveChart = createChart(el);

  const lossData = [];
  const lrData = [];
  const steps = [];

  lossCurveChart.setOption({
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'cross', lineStyle: { color: '#E8ECF1', opacity: 0.4 } },
      formatter: (params) => {
        const step = params[0]?.axisValue;
        const loss = params.find(p => p.seriesName === 'Loss');
        const lr = params.find(p => p.seriesName === 'Learning Rate');
        return tooltipWrap(`Step <strong>${step}</strong><br/>Loss: <strong>${loss ? loss.value.toFixed(4) : '—'}</strong><br/>LR: <strong>${lr ? lr.value.toFixed(6) : '—'}</strong>`);
      },
    },
    grid: { left: 60, right: 60, top: 20, bottom: 40 },
    xAxis: { type: 'category', data: steps },
    yAxis: [
      { type: 'value', name: 'Loss', nameTextStyle: { color: '#5B8DEF', fontWeight: 600, fontSize: 12 }, max: 4, splitNumber: 4 },
      { type: 'value', name: 'Learning Rate', nameTextStyle: { color: '#9B7AEA', fontWeight: 600, fontSize: 12 }, position: 'right', splitLine: { show: false } },
    ],
    series: [
      { name: 'Loss', type: 'line', data: lossData, yAxisIndex: 0, showSymbol: false, lineStyle: { color: '#5B8DEF', width: 2 }, itemStyle: { color: '#5B8DEF' } },
      { name: 'Learning Rate', type: 'line', data: lrData, yAxisIndex: 1, showSymbol: false, lineStyle: { color: '#9B7AEA', width: 1.5, opacity: 0.5 }, itemStyle: { color: '#9B7AEA' } },
    ],
    animation: false,
  });

  function addPoint(step, loss, lr) {
    steps.push(step);
    lossData.push(loss);
    lrData.push(lr);
    lossCurveChart.setOption({
      xAxis: { data: steps },
      series: [{ data: lossData }, { data: lrData }],
    });
  }

  return { chart: lossCurveChart, addPoint };
}

function renderCheckpoints(checkpoints, activeStep, trainingLog) {
  const container = document.getElementById('checkpoint-container');
  const steps = Object.keys(checkpoints).map(Number).sort((a, b) => a - b);

  container.innerHTML = steps.map(step => {
    const names = checkpoints[String(step)];
    const isActive = step <= activeStep;
    const isCurrent = step === steps.reduce((prev, s) => s <= activeStep ? s : prev, steps[0]);
    const lossEntry = trainingLog ? trainingLog[step - 1] : null;
    const lossStr = lossEntry ? `Loss: ${lossEntry.loss.toFixed(3)}` : '';
    return `
      <div class="checkpoint-card ${isCurrent ? 'active' : ''}${!isActive ? ' checkpoint-inactive' : ''}">
        <div class="step-label">Step ${step}</div>
        ${lossStr ? `<div class="loss-value">${lossStr}</div>` : ''}
        <div class="names">${names.join('\n')}</div>
      </div>
    `;
  }).join('');
}

export function initTraining({ trainingLog, checkpoints, vocab, weights }, onWeightsUpdate) {
  const svgContainer = document.getElementById('training-svg-container');
  const sliderRow = document.getElementById('step-slider-row');
  const slider = document.getElementById('step-slider');
  const stepValue = document.getElementById('step-value');
  const btnPretrained = document.getElementById('btn-pretrained');
  const btnTrainScratch = document.getElementById('btn-train-scratch');
  const trainStatus = document.getElementById('train-status');
  const progressText = document.getElementById('train-progress-text');
  const progressBar = document.getElementById('train-progress-bar');

  let mode = 'pretrained'; // 'pretrained' | 'live'
  let worker = null;
  let liveChart = null;
  let liveCheckpoints = {};
  let sliderController = null;

  const vocabSize = vocab.chars.length + 1;

  // --- Pre-trained mode ---
  function showPretrained() {
    mode = 'pretrained';
    btnPretrained.setAttribute('aria-pressed', 'true');
    btnTrainScratch.setAttribute('aria-pressed', 'false');
    trainStatus.classList.add('train-status-hidden');
    sliderRow.style.display = 'flex';

    if (worker) { worker.terminate(); worker = null; }
    if (filmstripTimer) { clearInterval(filmstripTimer); filmstripTimer = null; filmstripPlaying = false; }
    if (lossCurveChart) { lossCurveChart.dispose(); lossCurveChart = null; }

    // Restore original weights when entering pretrained mode
    loadWeights(weights, vocabSize);

    const chart = createPretrainedChart(svgContainer, trainingLog);

    set('trainingStep', 500);
    currentCheckpointStep = 500;
    activeFilmstripStep = 500;
    renderCheckpoints(checkpoints, 500, trainingLog);
    setScrubberPosition(chart, 500, trainingLog);
    stepValue.textContent = 500;

    // Re-render weight inspector with pre-trained weights
    renderWeightInspector(getStateDict());

    const weightInspector = document.getElementById('weight-inspector');

    // Lazy-fetch checkpoint weights, then render filmstrip
    if (!checkpointWeights) {
      fetch('/data/checkpoint-weights.json')
        .then(r => r.ok ? r.json() : null)
        .then(data => {
          checkpointWeights = data;
          if (data && mode === 'pretrained') {
            renderWeightFilmstrip(weightInspector, checkpointWeights, vocabSize);
          }
        })
        .catch(() => {});
    } else {
      renderWeightFilmstrip(weightInspector, checkpointWeights, vocabSize);
    }

    // Checkpoint marker clicks on ECharts
    chart.on('click', 'markPoint', (params) => {
      if (!checkpointWeights) return;
      const step = params.value;
      slider.value = step - 1;
      set('trainingStep', step);
      setScrubberPosition(chart, step, trainingLog);
      stepValue.textContent = step;
      renderCheckpoints(checkpoints, step, trainingLog);
      currentCheckpointStep = step;
      activeFilmstripStep = step;
      loadWeights(checkpointWeights[String(step)], vocabSize);
      renderWeightFilmstrip(weightInspector, checkpointWeights, vocabSize);
      set('weightsVersion', (get('weightsVersion') || 0) + 1);
    });

    // Abort previous slider listener to prevent accumulation
    if (sliderController) sliderController.abort();
    sliderController = new AbortController();

    slider.addEventListener('input', () => {
      const step = parseInt(slider.value) + 1;
      set('trainingStep', step);
      setScrubberPosition(chart, step, trainingLog);
      stepValue.textContent = step;
      renderCheckpoints(checkpoints, step, trainingLog);

      if (!checkpointWeights) return;
      const cp = nearestCheckpoint(step);
      if (cp !== currentCheckpointStep) {
        currentCheckpointStep = cp;
        activeFilmstripStep = cp;
        loadWeights(checkpointWeights[String(cp)], vocabSize);
        renderWeightFilmstrip(weightInspector, checkpointWeights, vocabSize);
        set('weightsVersion', (get('weightsVersion') || 0) + 1);
      }
    }, { signal: sliderController.signal });
  }

  // --- Train from scratch mode ---
  function startLiveTraining() {
    mode = 'live';
    btnPretrained.setAttribute('aria-pressed', 'false');
    btnTrainScratch.setAttribute('aria-pressed', 'true');
    trainStatus.classList.remove('train-status-hidden');
    sliderRow.style.display = 'none';
    set('trainingStep', 0);
    liveCheckpoints = {};
    renderCheckpoints({}, 0, null);

    if (lossCurveChart) { lossCurveChart.dispose(); lossCurveChart = null; }

    // Create live chart
    liveChart = createLiveChart(svgContainer);

    // Clear weight inspector
    renderWeightInspector(null);

    // Fetch training docs and start worker
    fetch('/data/training-docs.json')
      .then(r => r.ok ? r.json() : null)
      .catch(() => null)
      .then(docs => {
        if (!docs) {
          // If training docs not available, generate a simple dataset
          const names = [];
          for (const cp of Object.values(checkpoints)) {
            names.push(...cp);
          }
          docs = names.length > 0 ? names : ['alice', 'bob', 'charlie'];
        }

        worker = new Worker(new URL('./train-worker.js', import.meta.url), { type: 'module' });
        worker.postMessage({ type: 'start', docs });

        worker.onmessage = (e) => {
          const msg = e.data;
          if (msg.type === 'step') {
            set('trainingStep', msg.step);
            liveChart.addPoint(msg.step, msg.loss, msg.lr);
            progressText.textContent = `${msg.step} / 500`;
            progressBar.style.width = `${(msg.step / 500) * 100}%`;
          } else if (msg.type === 'checkpoint') {
            liveCheckpoints[String(msg.step)] = msg.names;
            renderCheckpoints(liveCheckpoints, msg.step, null);
          } else if (msg.type === 'checkpoint-weights') {
            loadWeights(Array.from(msg.weights), vocabSize);
            renderWeightInspector(getStateDict());
            set('weightsVersion', (get('weightsVersion') || 0) + 1);
          } else if (msg.type === 'weights') {
            // Load the freshly trained weights
            loadWeights(Array.from(msg.weights), vocab.chars.length + 1);
            renderWeightInspector(getStateDict());
            if (onWeightsUpdate) onWeightsUpdate();
          } else if (msg.type === 'done') {
            trainStatus.querySelector('.spinner').style.display = 'none';
            progressText.textContent = '500 / 500 — Done';
            worker = null;
          }
        };
      });
  }

  // Mode toggle handlers
  btnPretrained.addEventListener('click', showPretrained);
  btnTrainScratch.addEventListener('click', startLiveTraining);

  // Re-render weight inspector labels on ELI5 toggle
  subscribe('eli5', () => {
    if (mode === 'pretrained' && checkpointWeights) {
      renderWeightFilmstrip(document.getElementById('weight-inspector'), checkpointWeights, vocabSize);
    } else {
      renderWeightInspector(getStateDict());
    }
  });

  // Initialize with pre-trained view
  showPretrained();
}
