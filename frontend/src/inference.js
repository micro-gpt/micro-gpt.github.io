/**
 * Inference playground — temperature control, token-by-token generation,
 * probability bars, attention heatmaps.
 */

import { gptForward, softmax, sampleFrom, N_LAYER, N_HEAD, BLOCK_SIZE } from './gpt.js';

let generating = false;

function renderProbBars(vocab) {
  const container = document.getElementById('prob-bars-container');
  const chars = [...vocab.chars, 'BOS'];

  container.innerHTML = chars.map((ch, i) => `
    <div class="prob-bar-row" data-token="${i}">
      <span class="token-label">${ch === ' ' ? '␣' : ch}</span>
      <div class="bar-track"><div class="bar-fill" style="width:0%"></div></div>
      <span class="prob-value">0%</span>
    </div>
  `).join('');
}

function updateProbBars(probs, selectedToken) {
  const rows = document.querySelectorAll('#prob-bars-container .prob-bar-row');

  // Sort by probability for display
  const indexed = Array.from(probs).map((p, i) => ({ p, i }));
  indexed.sort((a, b) => b.p - a.p);

  const container = document.getElementById('prob-bars-container');
  const sortedRows = indexed.map(({ i }) => rows[i]);
  sortedRows.forEach(row => container.appendChild(row));

  rows.forEach((row, i) => {
    const prob = probs[i];
    const fill = row.querySelector('.bar-fill');
    const val = row.querySelector('.prob-value');
    const pct = (prob * 100);

    fill.style.width = `${Math.min(pct, 100)}%`;

    if (i === selectedToken) {
      fill.classList.add('selected');
    } else {
      fill.classList.remove('selected');
    }

    val.textContent = pct < 1 ? `${pct.toFixed(1)}%` : `${pct.toFixed(0)}%`;
  });
}

function updateHeatmaps(allStepAttn, seqLen) {
  const canvases = document.querySelectorAll('#heatmap-container canvas');

  canvases.forEach((canvas, headIdx) => {
    canvas.width = seqLen;
    canvas.height = seqLen;
    const ctx = canvas.getContext('2d');

    for (let row = 0; row < seqLen; row++) {
      // allStepAttn[row] has N_HEAD entries; each entry is the attn weights for that step
      const stepAttn = allStepAttn[row];
      const headWeights = stepAttn ? stepAttn[headIdx] : null;

      for (let col = 0; col < seqLen; col++) {
        let val = 0;
        if (headWeights && col < headWeights.length) {
          val = headWeights[col];
        }

        // Blue-to-white color scale
        const r = Math.round(15 + val * 220);
        const g = Math.round(23 + val * 180);
        const b = Math.round(42 + val * 213);
        ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
        ctx.fillRect(col, row, 1, 1);
      }
    }
  });
}

async function generate(vocab, temperature) {
  if (generating) return;
  generating = true;

  const btn = document.getElementById('btn-generate');
  const output = document.getElementById('generated-output');
  btn.disabled = true;
  output.innerHTML = '<span class="cursor"></span>';

  const bos = vocab.bos;
  const chars = vocab.chars;
  const keys = Array.from({ length: N_LAYER }, () => []);
  const values = Array.from({ length: N_LAYER }, () => []);

  let tokenId = bos;
  const tokens = [];
  const allStepAttn = [];

  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const delay = reduceMotion ? 0 : 200;

  for (let pos = 0; pos < BLOCK_SIZE; pos++) {
    const { logits, attentionWeights } = gptForward(tokenId, pos, keys, values);
    allStepAttn.push(attentionWeights);

    // Apply temperature
    const scaled = logits.map(l => l / temperature);
    const probs = softmax(scaled);

    // Sample
    tokenId = sampleFrom(probs);

    // Update probability bars
    updateProbBars(probs, tokenId);

    // Update heatmaps
    updateHeatmaps(allStepAttn, pos + 1);

    if (tokenId === bos) break;

    tokens.push(tokenId);
    const ch = chars[tokenId];

    // Show token in output
    output.innerHTML = tokens.map(t => chars[t]).join('') + '<span class="cursor"></span>';

    if (delay > 0) {
      await new Promise(r => setTimeout(r, delay));
    }
  }

  // Final state — remove cursor
  output.innerHTML = tokens.map(t => chars[t]).join('') || '<span style="color:var(--text-dim)">(empty)</span>';

  btn.disabled = false;
  generating = false;
}

export function initInference({ vocab }) {
  renderProbBars(vocab);

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
}
