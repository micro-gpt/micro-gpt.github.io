/**
 * Training section — live training via Web Worker + pre-trained fallback.
 * Loss curve, checkpoint samples, weight inspector heatmaps.
 */

import { loadWeights, getStateDict } from './gpt.js';

const SVG_NS = 'http://www.w3.org/2000/svg';

// --- Weight Inspector ---
function drawHeatmap(canvas, matrix, label) {
  const rows = matrix.length;
  const cols = matrix[0].length;
  canvas.width = cols;
  canvas.height = rows;
  const ctx = canvas.getContext('2d');

  // Find global min/max for color scale
  let min = Infinity, max = -Infinity;
  for (const row of matrix) {
    for (const v of row) {
      if (v < min) min = v;
      if (v > max) max = v;
    }
  }
  const absMax = Math.max(Math.abs(min), Math.abs(max), 0.001);

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const v = matrix[r][c] / absMax; // -1 to 1
      let red, green, blue;
      if (v >= 0) {
        // Blue scale for positive
        red = Math.round(15 + (1 - v) * 30);
        green = Math.round(23 + (1 - v) * 30);
        blue = Math.round(80 + v * 175);
      } else {
        // Red scale for negative
        const a = Math.abs(v);
        red = Math.round(80 + a * 175);
        green = Math.round(23 + (1 - a) * 30);
        blue = Math.round(15 + (1 - a) * 30);
      }
      ctx.fillStyle = `rgb(${red}, ${green}, ${blue})`;
      ctx.fillRect(c, r, 1, 1);
    }
  }
}

function renderWeightInspector(stateDict) {
  const container = document.getElementById('weight-inspector');
  if (!stateDict) {
    container.innerHTML = '<p style="color:var(--text-dim);font-size:0.85rem">No weights loaded</p>';
    return;
  }

  const matrices = [
    { key: 'wte', label: 'Token Embeddings (wte)', dims: '27 × 16' },
    { key: 'wpe', label: 'Position Embeddings (wpe)', dims: '8 × 16' },
    { key: 'layer0.attn_wq', label: 'Attention Wq', dims: '16 × 16' },
    { key: 'layer0.attn_wk', label: 'Attention Wk', dims: '16 × 16' },
    { key: 'layer0.attn_wv', label: 'Attention Wv', dims: '16 × 16' },
    { key: 'layer0.attn_wo', label: 'Attention Wo', dims: '16 × 16' },
    { key: 'lm_head', label: 'LM Head', dims: '27 × 16' },
  ];

  container.innerHTML = matrices.map(({ key, label, dims }) =>
    `<div class="weight-heatmap">
      <div class="heatmap-label">${label}</div>
      <canvas data-weight="${key}" title="Click to inspect ${label}"></canvas>
      <div class="dims">${dims}</div>
    </div>`
  ).join('');

  // Draw heatmaps
  for (const { key } of matrices) {
    const canvas = container.querySelector(`canvas[data-weight="${key}"]`);
    if (canvas && stateDict[key]) {
      drawHeatmap(canvas, stateDict[key], key);
    }
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
    line.setAttribute('stroke', '#1e293b');
    line.setAttribute('stroke-width', '1');
    svg.appendChild(line);

    const label = document.createElementNS(SVG_NS, 'text');
    label.setAttribute('x', padL - 8);
    label.setAttribute('y', y + 4);
    label.setAttribute('text-anchor', 'end');
    label.setAttribute('fill', '#64748b');
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
    label.setAttribute('fill', '#64748b');
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
  lossLabel.setAttribute('fill', '#3b82f6');
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
  lrLabel.setAttribute('fill', '#8b5cf6');
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
  lrPath.setAttribute('stroke', '#8b5cf6');
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
  lossPath.setAttribute('stroke', '#3b82f6');
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

  // Scrubber
  const scrubLine = document.createElementNS(SVG_NS, 'line');
  scrubLine.setAttribute('y1', padT);
  scrubLine.setAttribute('y2', padT + plotH);
  scrubLine.setAttribute('stroke', '#f1f5f9');
  scrubLine.setAttribute('stroke-width', '1');
  scrubLine.setAttribute('opacity', '0.4');
  scrubLine.setAttribute('stroke-dasharray', '4 3');
  scrubLine.id = 'scrub-line';
  svg.appendChild(scrubLine);

  const scrubDot = document.createElementNS(SVG_NS, 'circle');
  scrubDot.setAttribute('r', '5');
  scrubDot.setAttribute('fill', '#3b82f6');
  scrubDot.setAttribute('stroke', '#f1f5f9');
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
    line.setAttribute('stroke', '#1e293b');
    svg.appendChild(line);

    const label = document.createElementNS(SVG_NS, 'text');
    label.setAttribute('x', padL - 8);
    label.setAttribute('y', y + 4);
    label.setAttribute('text-anchor', 'end');
    label.setAttribute('fill', '#64748b');
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
    label.setAttribute('fill', '#64748b');
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
  lossLabel.setAttribute('fill', '#3b82f6');
  lossLabel.setAttribute('font-size', '12');
  lossLabel.setAttribute('font-weight', '600');
  lossLabel.setAttribute('font-family', '-apple-system, BlinkMacSystemFont, sans-serif');
  lossLabel.textContent = 'Loss';
  svg.appendChild(lossLabel);

  // Loss path (starts empty, appended to as steps come in)
  const lossPath = document.createElementNS(SVG_NS, 'path');
  lossPath.setAttribute('fill', 'none');
  lossPath.setAttribute('stroke', '#3b82f6');
  lossPath.setAttribute('stroke-width', '2');
  svg.appendChild(lossPath);

  // LR path
  const lrPath = document.createElementNS(SVG_NS, 'path');
  lrPath.setAttribute('fill', 'none');
  lrPath.setAttribute('stroke', '#8b5cf6');
  lrPath.setAttribute('stroke-width', '1.5');
  lrPath.setAttribute('opacity', '0.5');
  svg.appendChild(lrPath);

  const lrLabel = document.createElementNS(SVG_NS, 'text');
  lrLabel.setAttribute('x', w - 14);
  lrLabel.setAttribute('y', padT + plotH / 2);
  lrLabel.setAttribute('text-anchor', 'middle');
  lrLabel.setAttribute('transform', `rotate(90, ${w - 14}, ${padT + plotH / 2})`);
  lrLabel.setAttribute('fill', '#8b5cf6');
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

function renderCheckpoints(checkpoints, activeStep) {
  const container = document.getElementById('checkpoint-container');
  const steps = Object.keys(checkpoints).map(Number).sort((a, b) => a - b);

  container.innerHTML = steps.map(step => {
    const names = checkpoints[String(step)];
    const isActive = step <= activeStep;
    const isCurrent = step === steps.reduce((prev, s) => s <= activeStep ? s : prev, steps[0]);
    return `
      <div class="checkpoint-card ${isCurrent ? 'active' : ''}" ${!isActive ? 'style="opacity:0.4"' : ''}>
        <div class="step-label">Step ${step}</div>
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

  // --- Pre-trained mode ---
  function showPretrained() {
    mode = 'pretrained';
    btnPretrained.setAttribute('aria-pressed', 'true');
    btnTrainScratch.setAttribute('aria-pressed', 'false');
    trainStatus.style.display = 'none';
    sliderRow.style.display = 'flex';

    if (worker) { worker.terminate(); worker = null; }

    const { svg, xScale, yLoss } = createTrainingSVG(trainingLog);
    svgContainer.innerHTML = '';
    svgContainer.appendChild(svg);

    renderCheckpoints(checkpoints, 500);
    updateScrubber(500);

    // Re-render weight inspector with pre-trained weights
    renderWeightInspector(getStateDict());

    function updateScrubber(step) {
      const entry = trainingLog[step - 1];
      const x = xScale(entry.step);
      const y = yLoss(entry.loss);
      document.getElementById('scrub-line').setAttribute('x1', x);
      document.getElementById('scrub-line').setAttribute('x2', x);
      document.getElementById('scrub-dot').setAttribute('cx', x);
      document.getElementById('scrub-dot').setAttribute('cy', y);
      stepValue.textContent = step;
      renderCheckpoints(checkpoints, step);
    }

    slider.addEventListener('input', () => {
      updateScrubber(parseInt(slider.value) + 1);
    });
  }

  // --- Train from scratch mode ---
  function startLiveTraining() {
    mode = 'live';
    btnPretrained.setAttribute('aria-pressed', 'false');
    btnTrainScratch.setAttribute('aria-pressed', 'true');
    trainStatus.style.display = 'flex';
    sliderRow.style.display = 'none';
    liveCheckpoints = {};
    renderCheckpoints({}, 0);

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
            liveChart.addPoint(msg.step, msg.loss, msg.lr);
            progressText.textContent = `${msg.step} / 500`;
            progressBar.style.width = `${(msg.step / 500) * 100}%`;
          } else if (msg.type === 'checkpoint') {
            liveCheckpoints[String(msg.step)] = msg.names;
            renderCheckpoints(liveCheckpoints, msg.step);
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

  // Initialize with pre-trained view
  showPretrained();
}
