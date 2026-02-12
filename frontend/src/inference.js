/**
 * Inference section — temperature control, token-by-token generation,
 * raw logits + probabilities, token history, attention heatmaps,
 * collapsible intermediate values viewer.
 */

import { gptForward, softmax, sampleFrom, N_LAYER, N_HEAD, BLOCK_SIZE } from './gpt.js';

let generating = false;

function renderBars(containerId, vocab) {
  const container = document.getElementById(containerId);
  const chars = [...vocab.chars, 'BOS'];

  container.innerHTML = chars.map((ch, i) => `
    <div class="prob-bar-row" data-token="${i}">
      <span class="token-label">${ch === ' ' ? '␣' : ch}</span>
      <div class="bar-track"><div class="bar-fill" style="width:0%"></div></div>
      <span class="prob-value">—</span>
    </div>
  `).join('');
}

function updateProbBars(containerId, values, selectedToken, cssClass) {
  const container = document.getElementById(containerId);
  const rows = container.querySelectorAll('.prob-bar-row');

  // Sort by value for display
  const indexed = Array.from(values).map((v, i) => ({ v, i }));
  indexed.sort((a, b) => b.v - a.v);

  const sortedRows = indexed.map(({ i }) => rows[i]);
  sortedRows.forEach(row => container.appendChild(row));

  // Find max absolute for logit scale
  const maxAbs = Math.max(...values.map(Math.abs), 0.001);

  rows.forEach((row, i) => {
    const val = values[i];
    const fill = row.querySelector('.bar-fill');
    const valSpan = row.querySelector('.prob-value');

    // Remove old classes
    fill.classList.remove('selected', 'logit-bar');

    if (cssClass === 'logit') {
      // For logits, scale relative to max absolute value
      const pct = (Math.max(0, val) / maxAbs) * 100;
      fill.style.width = `${Math.min(pct, 100)}%`;
      fill.classList.add('logit-bar');
      valSpan.textContent = val.toFixed(2);
    } else {
      // For probabilities
      const pct = val * 100;
      fill.style.width = `${Math.min(pct, 100)}%`;
      valSpan.textContent = pct < 1 ? `${pct.toFixed(1)}%` : `${pct.toFixed(0)}%`;
    }

    if (i === selectedToken) {
      fill.classList.add('selected');
    }
  });
}

function updateHeatmaps(allStepAttn, seqLen) {
  const canvases = document.querySelectorAll('#heatmap-container canvas');

  canvases.forEach((canvas, headIdx) => {
    canvas.width = seqLen;
    canvas.height = seqLen;
    const ctx = canvas.getContext('2d');

    for (let row = 0; row < seqLen; row++) {
      const stepAttn = allStepAttn[row];
      const headWeights = stepAttn ? stepAttn[headIdx] : null;

      for (let col = 0; col < seqLen; col++) {
        let val = 0;
        if (headWeights && col < headWeights.length) {
          val = headWeights[col];
        }

        const r = Math.round(15 + val * 220);
        const g = Math.round(23 + val * 180);
        const b = Math.round(42 + val * 213);
        ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
        ctx.fillRect(col, row, 1, 1);
      }
    }
  });
}

function renderTokenHistory(tokens, vocab) {
  const container = document.getElementById('token-history');
  if (tokens.length === 0) {
    container.innerHTML = '';
    return;
  }

  const chars = vocab.chars;
  container.innerHTML = tokens.map((t, i) => {
    const label = t === vocab.bos ? 'BOS' : chars[t];
    const isCurrent = i === tokens.length - 1;
    return `<span class="token-chip ${isCurrent ? 'current' : ''}">${label}</span>`;
  }).join('');
}

function renderIntermediateViewer(intermediates) {
  const content = document.getElementById('infer-inter-content');
  if (!intermediates) {
    content.innerHTML = '<p style="color:var(--text-dim);font-size:0.85rem;padding:0.5rem">Run a generation step to see intermediate values.</p>';
    return;
  }

  const sections = [
    { key: 'tokEmb', label: 'Token Embedding', dim: 16 },
    { key: 'posEmb', label: 'Position Embedding', dim: 16 },
    { key: 'combined', label: 'Combined (tok + pos)', dim: 16 },
    { key: 'postNorm0', label: 'After initial RMSNorm', dim: 16 },
    { key: 'postNorm1', label: 'After pre-attention RMSNorm', dim: 16 },
    { key: 'q', label: 'Q projection', dim: 16 },
    { key: 'k', label: 'K projection', dim: 16 },
    { key: 'v', label: 'V projection', dim: 16 },
    { key: 'attnOut', label: 'Attention output', dim: 16 },
    { key: 'postResidual1', label: 'After residual 1', dim: 16 },
    { key: 'postNorm2', label: 'After pre-MLP RMSNorm', dim: 16 },
    { key: 'mlpHidden', label: 'MLP hidden (64-dim)', dim: 64 },
    { key: 'mlpActivated', label: 'After ReLU²', dim: 64 },
    { key: 'mlpOut', label: 'MLP output', dim: 16 },
    { key: 'postResidual2', label: 'Final hidden state', dim: 16 },
  ];

  const html = sections.map(({ key, label }) => {
    const values = intermediates[key];
    if (!values) return '';
    const maxAbs = Math.max(...values.map(Math.abs), 0.001);
    const cells = values.map(v => {
      const norm = v / maxAbs;
      const r = norm > 0 ? Math.round(59 + norm * 196) : Math.round(59);
      const g = norm > 0 ? Math.round(130 + norm * 125) : Math.round(68);
      const b = norm > 0 ? Math.round(246) : Math.round(68 + Math.abs(norm) * 178);
      const bg = `rgba(${r}, ${g}, ${b}, ${Math.abs(norm) * 0.4 + 0.1})`;
      return `<div class="val-cell" style="background:${bg}" title="${v.toFixed(6)}">${v.toFixed(3)}</div>`;
    }).join('');

    return `<div class="data-viewer" style="margin-bottom:0.5rem">
      <div class="vector-label">${label} [${values.length}]</div>
      <div class="values-grid">${cells}</div>
    </div>`;
  }).filter(Boolean).join('');

  content.innerHTML = html || '<p style="color:var(--text-dim);font-size:0.85rem;padding:0.5rem">No intermediate data available.</p>';
}

async function generate(vocab, temperature) {
  if (generating) return;
  generating = true;

  const btn = document.getElementById('btn-generate');
  const output = document.getElementById('generated-output');
  const probTitle = document.getElementById('prob-column-title');
  btn.disabled = true;
  output.innerHTML = '<span class="cursor"></span>';
  probTitle.textContent = `Probabilities (T=${temperature.toFixed(1)})`;

  const bos = vocab.bos;
  const chars = vocab.chars;
  const keys = Array.from({ length: N_LAYER }, () => []);
  const values = Array.from({ length: N_LAYER }, () => []);

  let tokenId = bos;
  const tokens = [bos]; // include BOS in history
  const generatedTokens = [];
  const allStepAttn = [];

  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const delay = reduceMotion ? 0 : 200;

  for (let pos = 0; pos < BLOCK_SIZE; pos++) {
    const result = gptForward(tokenId, pos, keys, values, { intermediates: true });
    const { logits, attentionWeights, intermediates } = result;
    allStepAttn.push(attentionWeights);

    // Update raw logits display
    updateProbBars('logit-bars-container', logits, -1, 'logit');

    // Apply temperature and show probabilities
    const scaled = logits.map(l => l / temperature);
    const probs = softmax(scaled);

    // Sample
    tokenId = sampleFrom(probs);

    // Update probability bars
    updateProbBars('prob-bars-container', Array.from(probs), tokenId, 'prob');

    // Update heatmaps
    updateHeatmaps(allStepAttn, pos + 1);

    // Update intermediate viewer (last step's data)
    renderIntermediateViewer(intermediates);

    if (tokenId === bos) break;

    generatedTokens.push(tokenId);
    tokens.push(tokenId);

    // Update token history
    renderTokenHistory(tokens, vocab);

    // Show token in output
    output.innerHTML = generatedTokens.map(t => chars[t]).join('') + '<span class="cursor"></span>';

    if (delay > 0) {
      await new Promise(r => setTimeout(r, delay));
    }
  }

  // Final state
  output.innerHTML = generatedTokens.map(t => chars[t]).join('') || '<span style="color:var(--text-dim)">(empty)</span>';

  btn.disabled = false;
  generating = false;
}

export function initInference({ vocab }) {
  renderBars('logit-bars-container', vocab);
  renderBars('prob-bars-container', vocab);
  renderTokenHistory([], vocab);
  renderIntermediateViewer(null);

  const tempSlider = document.getElementById('temp-slider');
  const tempValue = document.getElementById('temp-value');
  const btnGenerate = document.getElementById('btn-generate');

  tempSlider.addEventListener('input', () => {
    tempValue.textContent = (parseInt(tempSlider.value) / 10).toFixed(1);
  });

  btnGenerate.addEventListener('click', () => {
    const temp = parseInt(tempSlider.value) / 10;
    generate(vocab, temp);
  });

  // Collapsible intermediate viewer toggle
  const interToggle = document.getElementById('infer-inter-toggle');
  const interContent = document.getElementById('infer-inter-content');
  const toggleHandler = () => {
    const expanded = interToggle.getAttribute('aria-expanded') === 'true';
    interToggle.setAttribute('aria-expanded', !expanded);
    interContent.classList.toggle('open');
  };
  interToggle.addEventListener('click', toggleHandler);
  interToggle.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggleHandler(); }
  });
}
