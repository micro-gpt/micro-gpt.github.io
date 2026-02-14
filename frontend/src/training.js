/**
 * Training section — live training via Web Worker + pre-trained fallback.
 * Loss curve, checkpoint samples, weight inspector heatmaps.
 */

import { loadWeights, getStateDict } from './gpt.js';
import { get, set, subscribe } from './state.js';
import { t } from './content.js';
import { createChart, FONT_FAMILY } from './echarts-setup.js';

const SVG_NS = 'http://www.w3.org/2000/svg';

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

// Legacy drawHeatmap still used by filmstrip canvases
function drawHeatmap(canvas, matrix) {
  const rows = matrix.length;
  const cols = matrix[0].length;
  canvas.width = cols;
  canvas.height = rows;
  const ctx = canvas.getContext('2d');

  let absMax = 0;
  for (const row of matrix) for (const v of row) absMax = Math.max(absMax, Math.abs(v));
  if (absMax < 0.001) absMax = 0.001;

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const v = matrix[r][c] / absMax;
      if (v >= 0) {
        ctx.fillStyle = `rgb(${Math.round(15 + (1 - v) * 30)}, ${Math.round(23 + (1 - v) * 30)}, ${Math.round(80 + v * 175)})`;
      } else {
        const a = Math.abs(v);
        ctx.fillStyle = `rgb(${Math.round(80 + a * 175)}, ${Math.round(23 + (1 - a) * 30)}, ${Math.round(15 + (1 - a) * 30)})`;
      }
      ctx.fillRect(c, r, 1, 1);
    }
  }
}

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
    container.innerHTML = '<p style="color:var(--text-dim);font-size:0.85rem">No weights loaded</p>';
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
        <div data-weight-chart="${key}" style="height:${Math.max(dims[0] * 10, 160)}px" aria-label="${label} weight matrix"></div>
        <div class="dims">${dims[0]} × ${dims[1]}</div>
      </div>`;
    }).join('') + '</div>';
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

    weightCharts[key].setOption({
      tooltip: {
        formatter: (params) => {
          const [col, row, val] = params.value;
          return `<span style="font-family:monospace">${key}[${row}][${col}] = ${val.toFixed(4)}</span>`;
        },
      },
      xAxis: {
        type: 'category',
        data: Array.from({ length: cols }, (_, i) => i),
        axisLabel: { show: false },
        axisTick: { show: false },
        splitArea: { show: false },
      },
      yAxis: {
        type: 'category',
        data: Array.from({ length: rows }, (_, i) => i),
        axisLabel: { show: false },
        axisTick: { show: false },
        splitArea: { show: false },
        inverse: true,
      },
      visualMap: {
        show: false,
        min: -absMax,
        max: absMax,
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
      grid: { left: 4, right: 4, top: 4, bottom: 4 },
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
    container.innerHTML = '<p style="color:var(--text-dim);font-size:0.85rem">Loading checkpoint weights...</p>';
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
        drawHeatmap(canvas, matrix, key);
      }
    }
  }

  // Restore active checkpoint weights
  if (allCheckpointWeights[String(activeFilmstripStep)]) {
    loadWeights(allCheckpointWeights[String(activeFilmstripStep)], vocabSize);
  }
}

// --- Loss Curve SVG ---
function createTrainingSVG(trainingLog) {
  const w = 800;
  const h = 300;
  const padL = 55;
  const padR = 55;
  const padT = 20;
  const padB = 40;
  const plotW = w - padL - padR;
  const plotH = h - padT - padB;

  const maxStep = trainingLog.length;
  const maxLoss = Math.ceil(Math.max(...trainingLog.map(d => d.loss)));
  const maxLR = Math.max(...trainingLog.map(d => d.lr));

  const xScale = (step) => padL + ((step - 1) / (maxStep - 1)) * plotW;
  const yLoss = (loss) => padT + (1 - loss / maxLoss) * plotH;
  const yLR = (lr) => padT + (1 - lr / maxLR) * plotH;

  const svg = document.createElementNS(SVG_NS, 'svg');
  svg.setAttribute('viewBox', `0 0 ${w} ${h}`);
  svg.setAttribute('width', '100%');
  svg.setAttribute('height', h);
  svg.setAttribute('aria-label', `Training loss curve over ${maxStep} steps`);
  svg.setAttribute('role', 'img');
  svg.style.maxWidth = `${w}px`;

  // Grid lines
  for (let i = 0; i <= 4; i++) {
    const yVal = (i / 4) * maxLoss;
    const y = yLoss(yVal);
    const line = document.createElementNS(SVG_NS, 'line');
    line.setAttribute('x1', padL);
    line.setAttribute('y1', y);
    line.setAttribute('x2', w - padR);
    line.setAttribute('y2', y);
    line.setAttribute('stroke', '#1E2433');
    line.setAttribute('stroke-width', '1');
    svg.appendChild(line);

    const label = document.createElementNS(SVG_NS, 'text');
    label.setAttribute('x', padL - 8);
    label.setAttribute('y', y + 4);
    label.setAttribute('text-anchor', 'end');
    label.setAttribute('fill', '#6B7585');
    label.setAttribute('font-size', '11');
    label.setAttribute('font-family', '-apple-system, BlinkMacSystemFont, sans-serif');
    label.textContent = yVal.toFixed(1);
    svg.appendChild(label);
  }

  // X axis labels
  for (let s = 0; s <= 500; s += 100) {
    if (s === 0) continue;
    const x = xScale(s);
    const label = document.createElementNS(SVG_NS, 'text');
    label.setAttribute('x', x);
    label.setAttribute('y', h - 8);
    label.setAttribute('text-anchor', 'middle');
    label.setAttribute('fill', '#6B7585');
    label.setAttribute('font-size', '11');
    label.setAttribute('font-family', '-apple-system, BlinkMacSystemFont, sans-serif');
    label.textContent = s;
    svg.appendChild(label);
  }

  // Axis labels
  const lossLabel = document.createElementNS(SVG_NS, 'text');
  lossLabel.setAttribute('x', 14);
  lossLabel.setAttribute('y', padT + plotH / 2);
  lossLabel.setAttribute('text-anchor', 'middle');
  lossLabel.setAttribute('transform', `rotate(-90, 14, ${padT + plotH / 2})`);
  lossLabel.setAttribute('fill', '#5B8DEF');
  lossLabel.setAttribute('font-size', '12');
  lossLabel.setAttribute('font-weight', '600');
  lossLabel.setAttribute('font-family', '-apple-system, BlinkMacSystemFont, sans-serif');
  lossLabel.textContent = 'Loss';
  svg.appendChild(lossLabel);

  const lrLabel = document.createElementNS(SVG_NS, 'text');
  lrLabel.setAttribute('x', w - 14);
  lrLabel.setAttribute('y', padT + plotH / 2);
  lrLabel.setAttribute('text-anchor', 'middle');
  lrLabel.setAttribute('transform', `rotate(90, ${w - 14}, ${padT + plotH / 2})`);
  lrLabel.setAttribute('fill', '#9B7AEA');
  lrLabel.setAttribute('font-size', '12');
  lrLabel.setAttribute('font-weight', '600');
  lrLabel.setAttribute('font-family', '-apple-system, BlinkMacSystemFont, sans-serif');
  lrLabel.textContent = 'Learning Rate';
  svg.appendChild(lrLabel);

  // LR curve
  let lrPathD = '';
  trainingLog.forEach((d, i) => {
    const x = xScale(d.step);
    const y = yLR(d.lr);
    lrPathD += i === 0 ? `M${x},${y}` : `L${x},${y}`;
  });
  const lrPath = document.createElementNS(SVG_NS, 'path');
  lrPath.setAttribute('d', lrPathD);
  lrPath.setAttribute('fill', 'none');
  lrPath.setAttribute('stroke', '#9B7AEA');
  lrPath.setAttribute('stroke-width', '1.5');
  lrPath.setAttribute('opacity', '0.5');
  svg.appendChild(lrPath);

  // Loss curve
  let lossPathD = '';
  trainingLog.forEach((d, i) => {
    const x = xScale(d.step);
    const y = yLoss(d.loss);
    lossPathD += i === 0 ? `M${x},${y}` : `L${x},${y}`;
  });
  const lossPath = document.createElementNS(SVG_NS, 'path');
  lossPath.setAttribute('d', lossPathD);
  lossPath.setAttribute('fill', 'none');
  lossPath.setAttribute('stroke', '#5B8DEF');
  lossPath.setAttribute('stroke-width', '2');
  lossPath.id = 'loss-path';
  svg.appendChild(lossPath);

  // Draw-on animation
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  if (!reduceMotion) {
    for (const path of [lossPath, lrPath]) {
      const len = path.getTotalLength();
      path.style.strokeDasharray = len;
      path.style.strokeDashoffset = len;
      path.style.transition = 'stroke-dashoffset 2s ease-out';
      requestAnimationFrame(() => { path.style.strokeDashoffset = '0'; });
    }
  }

  // Checkpoint markers
  for (const cp of CHECKPOINT_STEPS) {
    const entry = trainingLog[cp - 1];
    if (!entry) continue;
    const cx = xScale(cp);
    const cy = yLoss(entry.loss);

    const g = document.createElementNS(SVG_NS, 'g');
    g.setAttribute('class', 'checkpoint-marker');
    g.setAttribute('data-step', cp);
    g.setAttribute('role', 'button');
    g.setAttribute('tabindex', '0');
    g.setAttribute('aria-label', `Checkpoint at step ${cp}, loss ${entry.loss.toFixed(2)}`);

    const markerLine = document.createElementNS(SVG_NS, 'line');
    markerLine.setAttribute('x1', cx);
    markerLine.setAttribute('y1', padT);
    markerLine.setAttribute('x2', cx);
    markerLine.setAttribute('y2', padT + plotH);
    markerLine.setAttribute('stroke', '#5B8DEF');
    markerLine.setAttribute('stroke-width', '1');
    markerLine.setAttribute('opacity', '0.2');
    markerLine.setAttribute('stroke-dasharray', '3 3');
    g.appendChild(markerLine);

    const dot = document.createElementNS(SVG_NS, 'circle');
    dot.setAttribute('cx', cx);
    dot.setAttribute('cy', cy);
    dot.setAttribute('r', '4');
    dot.setAttribute('fill', '#5B8DEF');
    dot.setAttribute('stroke', '#0B0F1A');
    dot.setAttribute('stroke-width', '1.5');
    g.appendChild(dot);

    const label = document.createElementNS(SVG_NS, 'text');
    label.setAttribute('x', cx);
    label.setAttribute('y', padT - 6);
    label.setAttribute('text-anchor', 'middle');
    label.setAttribute('fill', '#6B7585');
    label.setAttribute('font-size', '9');
    label.setAttribute('font-family', '-apple-system, BlinkMacSystemFont, sans-serif');
    label.textContent = cp;
    g.appendChild(label);

    svg.appendChild(g);
  }

  // Scrubber
  const scrubLine = document.createElementNS(SVG_NS, 'line');
  scrubLine.setAttribute('y1', padT);
  scrubLine.setAttribute('y2', padT + plotH);
  scrubLine.setAttribute('stroke', '#E8ECF1');
  scrubLine.setAttribute('stroke-width', '1');
  scrubLine.setAttribute('opacity', '0.4');
  scrubLine.setAttribute('stroke-dasharray', '4 3');
  scrubLine.id = 'scrub-line';
  svg.appendChild(scrubLine);

  const scrubDot = document.createElementNS(SVG_NS, 'circle');
  scrubDot.setAttribute('r', '5');
  scrubDot.setAttribute('fill', '#5B8DEF');
  scrubDot.setAttribute('stroke', '#E8ECF1');
  scrubDot.setAttribute('stroke-width', '2');
  scrubDot.id = 'scrub-dot';
  svg.appendChild(scrubDot);

  return { svg, xScale, yLoss, yLR, maxStep, maxLoss, maxLR, lossPath, lrPath };
}

// --- Live loss curve (for training from scratch) ---
function createLiveChart() {
  const w = 800, h = 300;
  const padL = 55, padR = 55, padT = 20, padB = 40;
  const plotW = w - padL - padR;
  const plotH = h - padT - padB;
  const maxStep = 500;
  const maxLoss = 4; // initial estimate, may increase

  const svg = document.createElementNS(SVG_NS, 'svg');
  svg.setAttribute('viewBox', `0 0 ${w} ${h}`);
  svg.setAttribute('width', '100%');
  svg.setAttribute('height', h);
  svg.setAttribute('aria-label', 'Live training loss curve');
  svg.setAttribute('role', 'img');
  svg.style.maxWidth = `${w}px`;

  // Grid
  for (let i = 0; i <= 4; i++) {
    const yVal = (i / 4) * maxLoss;
    const y = padT + (1 - yVal / maxLoss) * plotH;
    const line = document.createElementNS(SVG_NS, 'line');
    line.setAttribute('x1', padL);
    line.setAttribute('y1', y);
    line.setAttribute('x2', w - padR);
    line.setAttribute('y2', y);
    line.setAttribute('stroke', '#1E2433');
    svg.appendChild(line);

    const label = document.createElementNS(SVG_NS, 'text');
    label.setAttribute('x', padL - 8);
    label.setAttribute('y', y + 4);
    label.setAttribute('text-anchor', 'end');
    label.setAttribute('fill', '#6B7585');
    label.setAttribute('font-size', '11');
    label.setAttribute('font-family', '-apple-system, BlinkMacSystemFont, sans-serif');
    label.textContent = yVal.toFixed(1);
    svg.appendChild(label);
  }

  for (let s = 100; s <= 500; s += 100) {
    const x = padL + ((s - 1) / (maxStep - 1)) * plotW;
    const label = document.createElementNS(SVG_NS, 'text');
    label.setAttribute('x', x);
    label.setAttribute('y', h - 8);
    label.setAttribute('text-anchor', 'middle');
    label.setAttribute('fill', '#6B7585');
    label.setAttribute('font-size', '11');
    label.setAttribute('font-family', '-apple-system, BlinkMacSystemFont, sans-serif');
    label.textContent = s;
    svg.appendChild(label);
  }

  // Axis labels
  const lossLabel = document.createElementNS(SVG_NS, 'text');
  lossLabel.setAttribute('x', 14);
  lossLabel.setAttribute('y', padT + plotH / 2);
  lossLabel.setAttribute('text-anchor', 'middle');
  lossLabel.setAttribute('transform', `rotate(-90, 14, ${padT + plotH / 2})`);
  lossLabel.setAttribute('fill', '#5B8DEF');
  lossLabel.setAttribute('font-size', '12');
  lossLabel.setAttribute('font-weight', '600');
  lossLabel.setAttribute('font-family', '-apple-system, BlinkMacSystemFont, sans-serif');
  lossLabel.textContent = 'Loss';
  svg.appendChild(lossLabel);

  // Loss path (starts empty, appended to as steps come in)
  const lossPath = document.createElementNS(SVG_NS, 'path');
  lossPath.setAttribute('fill', 'none');
  lossPath.setAttribute('stroke', '#5B8DEF');
  lossPath.setAttribute('stroke-width', '2');
  svg.appendChild(lossPath);

  // LR path
  const lrPath = document.createElementNS(SVG_NS, 'path');
  lrPath.setAttribute('fill', 'none');
  lrPath.setAttribute('stroke', '#9B7AEA');
  lrPath.setAttribute('stroke-width', '1.5');
  lrPath.setAttribute('opacity', '0.5');
  svg.appendChild(lrPath);

  const lrLabel = document.createElementNS(SVG_NS, 'text');
  lrLabel.setAttribute('x', w - 14);
  lrLabel.setAttribute('y', padT + plotH / 2);
  lrLabel.setAttribute('text-anchor', 'middle');
  lrLabel.setAttribute('transform', `rotate(90, ${w - 14}, ${padT + plotH / 2})`);
  lrLabel.setAttribute('fill', '#9B7AEA');
  lrLabel.setAttribute('font-size', '12');
  lrLabel.setAttribute('font-weight', '600');
  lrLabel.setAttribute('font-family', '-apple-system, BlinkMacSystemFont, sans-serif');
  lrLabel.textContent = 'Learning Rate';
  svg.appendChild(lrLabel);

  const maxLR = 0.01;
  let lossD = '';
  let lrD = '';

  function addPoint(step, loss, lr) {
    const x = padL + ((step - 1) / (maxStep - 1)) * plotW;
    const yL = padT + (1 - Math.min(loss, maxLoss) / maxLoss) * plotH;
    const yR = padT + (1 - lr / maxLR) * plotH;
    lossD += lossD === '' ? `M${x},${yL}` : `L${x},${yL}`;
    lrD += lrD === '' ? `M${x},${yR}` : `L${x},${yR}`;
    lossPath.setAttribute('d', lossD);
    lrPath.setAttribute('d', lrD);
  }

  return { svg, addPoint };
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
      <div class="checkpoint-card ${isCurrent ? 'active' : ''}" ${!isActive ? 'style="opacity:0.4"' : ''}>
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
    trainStatus.style.display = 'none';
    sliderRow.style.display = 'flex';

    if (worker) { worker.terminate(); worker = null; }
    if (filmstripTimer) { clearInterval(filmstripTimer); filmstripTimer = null; filmstripPlaying = false; }

    // Restore original weights when entering pretrained mode
    loadWeights(weights, vocabSize);

    const { svg, xScale, yLoss } = createTrainingSVG(trainingLog);
    svgContainer.innerHTML = '';
    svgContainer.appendChild(svg);

    set('trainingStep', 500);
    currentCheckpointStep = 500;
    activeFilmstripStep = 500;
    renderCheckpoints(checkpoints, 500, trainingLog);
    updateScrubber(500);

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

    function updateScrubber(step) {
      const entry = trainingLog[step - 1];
      const x = xScale(entry.step);
      const y = yLoss(entry.loss);
      document.getElementById('scrub-line').setAttribute('x1', x);
      document.getElementById('scrub-line').setAttribute('x2', x);
      document.getElementById('scrub-dot').setAttribute('cx', x);
      document.getElementById('scrub-dot').setAttribute('cy', y);
      stepValue.textContent = step;
      renderCheckpoints(checkpoints, step, trainingLog);
    }

    // Checkpoint marker clicks on SVG
    svg.addEventListener('click', (e) => {
      const marker = e.target.closest('.checkpoint-marker');
      if (!marker || !checkpointWeights) return;
      const step = parseInt(marker.dataset.step);
      slider.value = step - 1;
      set('trainingStep', step);
      updateScrubber(step);
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
      updateScrubber(step);

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
    trainStatus.style.display = 'flex';
    sliderRow.style.display = 'none';
    set('trainingStep', 0);
    liveCheckpoints = {};
    renderCheckpoints({}, 0, null);

    // Create live chart
    liveChart = createLiveChart();
    svgContainer.innerHTML = '';
    svgContainer.appendChild(liveChart.svg);

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
