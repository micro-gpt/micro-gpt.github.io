/**
 * Inference section — temperature control, token-by-token generation,
 * raw logits + probabilities, interactive attention arc visualization,
 * collapsible intermediate values viewer.
 */

import { gptForward, softmax, sampleFrom, N_LAYER, N_HEAD, BLOCK_SIZE } from './gpt.js';
import { set } from './state.js';
import { t } from './content.js';
import { drawAttentionArcs } from './viz-utils.js';

const BLOCK_COLORS = {
  tokEmb: '#5B8DEF', posEmb: '#5B8DEF', combined: '#5B8DEF',
  postNorm0: '#9B7AEA', postNorm1: '#9B7AEA', postNorm2: '#9B7AEA',
  q: '#22D3EE', k: '#22D3EE', v: '#22D3EE', attnOut: '#22D3EE',
  postResidual1: '#6B7585', postResidual2: '#6B7585',
  mlpHidden: '#FB923C', mlpActivated: '#FB923C', mlpOut: '#FB923C',
};

let generating = false;
let storedAttn = [];
let storedTokens = [];
let selectedTokenIdx = -1;
let activeHead = 'all';
let storedVocab = null;
let lastLogits = null;

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
  container.classList.remove('prob-bars-empty');
  const rows = container.querySelectorAll('.prob-bar-row');

  const indexed = Array.from(values).map((v, i) => ({ v, i }));
  indexed.sort((a, b) => b.v - a.v);

  const sortedRows = indexed.map(({ i }) => rows[i]);
  sortedRows.forEach(row => container.appendChild(row));

  const maxAbs = Math.max(...values.map(Math.abs), 0.001);

  rows.forEach((row, i) => {
    const val = values[i];
    const fill = row.querySelector('.bar-fill');
    const valSpan = row.querySelector('.prob-value');

    fill.classList.remove('selected', 'logit-bar');

    if (cssClass === 'logit') {
      const pct = (Math.max(0, val) / maxAbs) * 100;
      fill.style.width = `${Math.min(pct, 100)}%`;
      fill.classList.add('logit-bar');
      valSpan.textContent = val.toFixed(2);
    } else {
      const pct = val * 100;
      fill.style.width = `${Math.min(pct, 100)}%`;
      valSpan.textContent = pct < 1 ? `${pct.toFixed(1)}%` : `${pct.toFixed(0)}%`;
    }

    if (i === selectedToken) {
      fill.classList.add('selected');
    }
  });
}

function renderAttnTokens() {
  const container = document.getElementById('attn-token-row');
  const hint = document.getElementById('attn-hint');

  if (storedTokens.length === 0) {
    container.innerHTML = '';
    hint.style.display = '';
    clearArcs();
    return;
  }

  hint.style.display = 'none';
  const chars = storedVocab.chars;

  container.innerHTML = storedTokens.map((t, i) => {
    const label = t === storedVocab.bos ? 'BOS' : (chars[t] === ' ' ? '␣' : chars[t]);
    const titleAttr = t === storedVocab.bos ? ' title="Beginning Of Sequence — a special token that signals the start of generation"' : '';
    const classes = ['attn-token'];
    if (i === selectedTokenIdx) classes.push('selected');
    if (i === storedTokens.length - 1 && generating) classes.push('current');
    return `<button class="${classes.join(' ')}" data-idx="${i}" aria-label="Token ${i}: ${label}"${titleAttr}>${label}</button>`;
  }).join('');

  const buttons = container.querySelectorAll('.attn-token');
  buttons.forEach(btn => {
    const idx = parseInt(btn.dataset.idx);

    btn.addEventListener('click', () => {
      if (generating) return;
      if (selectedTokenIdx === idx) {
        selectedTokenIdx = -1;
        btn.classList.remove('selected');
        clearArcs();
      } else {
        container.querySelectorAll('.attn-token').forEach(b => b.classList.remove('selected'));
        selectedTokenIdx = idx;
        btn.classList.add('selected');
        renderArcs(idx);
      }
    });

    btn.addEventListener('mouseenter', () => {
      if (generating || selectedTokenIdx !== -1) return;
      renderArcs(idx);
    });

    btn.addEventListener('mouseleave', () => {
      if (generating || selectedTokenIdx !== -1) return;
      clearArcs();
    });
  });

  if (selectedTokenIdx >= 0 && selectedTokenIdx < storedTokens.length) {
    renderArcs(selectedTokenIdx);
  }
}

function measureTokenPositions() {
  const container = document.getElementById('attn-token-row');
  const svg = document.getElementById('attn-arcs-svg');
  const svgRect = svg.getBoundingClientRect();
  const buttons = container.querySelectorAll('.attn-token');

  return Array.from(buttons).map(btn => {
    const rect = btn.getBoundingClientRect();
    return {
      cx: rect.left + rect.width / 2 - svgRect.left,
      top: rect.bottom - svgRect.top + 4,
    };
  });
}

function renderArcs(tokenIdx) {
  const svg = document.getElementById('attn-arcs-svg');

  if (tokenIdx < 0 || tokenIdx >= storedAttn.length) {
    svg.innerHTML = '';
    svg.style.height = '0';
    return;
  }

  const positions = measureTokenPositions();
  if (positions.length === 0) return;

  const tokenBtns = document.querySelectorAll('#attn-token-row .attn-token');
  drawAttentionArcs(svg, tokenIdx, storedAttn[tokenIdx], positions, {
    head: activeHead,
    showLabels: true,
    targetElements: tokenBtns,
  });
}

function clearArcs() {
  const svg = document.getElementById('attn-arcs-svg');
  svg.innerHTML = '';
  svg.style.height = '0';
  document.querySelectorAll('#attn-token-row .attn-token').forEach(b => b.classList.remove('target'));
}

function renderIntermediateViewer(intermediates) {
  const content = document.getElementById('infer-inter-content');
  if (!intermediates) {
    content.innerHTML = '<p class="inter-empty">Run a generation step to see intermediate values.</p>';
    return;
  }

  const interKeys = [
    'tokEmb', 'posEmb', 'combined', 'postNorm0', 'postNorm1',
    'q', 'k', 'v', 'attnOut', 'postResidual1', 'postNorm2',
    'mlpHidden', 'mlpActivated', 'mlpOut', 'postResidual2',
  ];
  const sections = interKeys.map(key => ({ key, label: t('inter.' + key) }));

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

    const borderColor = BLOCK_COLORS[key] || 'var(--text-dim)';
    return `<div class="data-viewer inter-block" style="border-left-color:${borderColor}">
      <div class="vector-label">${label} [${values.length}]</div>
      <div class="values-grid">${cells}</div>
    </div>`;
  }).filter(Boolean).join('');

  content.innerHTML = html || '<p class="inter-empty">No intermediate data available.</p>';
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

  // Reset attention state
  storedAttn = [];
  storedTokens = [vocab.bos];
  selectedTokenIdx = 0;

  const bos = vocab.bos;
  const chars = vocab.chars;
  const keys = Array.from({ length: N_LAYER }, () => []);
  const values = Array.from({ length: N_LAYER }, () => []);

  let tokenId = bos;
  const generatedTokens = [];

  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const delay = reduceMotion ? 0 : 200;

  for (let pos = 0; pos < BLOCK_SIZE; pos++) {
    const result = gptForward(tokenId, pos, keys, values, { intermediates: true });
    const { logits, attentionWeights, intermediates } = result;

    storedAttn.push(attentionWeights);
    selectedTokenIdx = pos;

    lastLogits = Array.from(logits);
    updateProbBars('logit-bars-container', logits, -1, 'logit');

    const scaled = logits.map(l => l / temperature);
    const probs = softmax(scaled);
    tokenId = sampleFrom(probs);

    updateProbBars('prob-bars-container', Array.from(probs), tokenId, 'prob');
    renderIntermediateViewer(intermediates);
    renderAttnTokens();

    if (tokenId === bos) break;

    generatedTokens.push(tokenId);
    storedTokens.push(tokenId);

    output.innerHTML = generatedTokens.map((t, i) =>
      `<span class="gen-token" data-token="${t}" data-pos="${i}">${chars[t]}</span>`
    ).join('') + '<span class="cursor"></span>';

    if (delay > 0) {
      await new Promise(r => setTimeout(r, delay));
    }
  }

  // Final state — select last token, stop cursor blink
  selectedTokenIdx = storedTokens.length - 1;
  generating = false;
  const cursor = output.querySelector('.cursor');
  if (cursor) cursor.classList.add('done');
  renderAttnTokens();

  if (generatedTokens.length === 0) {
    output.innerHTML = '<span class="inter-empty">(empty)</span>';
  } else {
    output.innerHTML = generatedTokens.map((t, i) =>
      `<span class="gen-token" data-token="${t}" data-pos="${i}">${chars[t]}</span>`
    ).join('');

    // Add click handlers to scroll to architecture and show token's forward pass
    output.querySelectorAll('.gen-token').forEach(span => {
      span.addEventListener('click', () => {
        set('token', parseInt(span.dataset.token));
        set('position', parseInt(span.dataset.pos));
        set('genStep', { ts: Date.now(), char: span.textContent, pos: parseInt(span.dataset.pos) });
        document.getElementById('section-architecture').scrollIntoView({ behavior: 'smooth' });
      });
    });
  }

  btn.disabled = false;
}

export function initInference({ vocab }) {
  storedVocab = vocab;

  renderBars('logit-bars-container', vocab);
  renderBars('prob-bars-container', vocab);
  document.getElementById('logit-bars-container').classList.add('prob-bars-empty');
  document.getElementById('prob-bars-container').classList.add('prob-bars-empty');
  renderIntermediateViewer(null);

  const tempSlider = document.getElementById('temp-slider');
  const tempValue = document.getElementById('temp-value');
  const btnGenerate = document.getElementById('btn-generate');

  set('temperature', 0.5);

  const tempPreview = document.getElementById('temp-preview');

  function updateTempPreview(temp) {
    if (!lastLogits) {
      tempPreview.innerHTML = '';
      return;
    }
    const scaled = lastLogits.map(l => l / temp);
    const probs = softmax(scaled);

    // Sort by probability for display
    const indexed = Array.from(probs).map((p, i) => ({ p, i }));
    indexed.sort((a, b) => b.p - a.p);

    if (tempPreview.children.length !== probs.length) {
      tempPreview.innerHTML = indexed.map(() => '<div class="temp-preview-bar"></div>').join('');
    }
    const bars = tempPreview.children;
    for (let j = 0; j < indexed.length; j++) {
      const pct = indexed[j].p * 100;
      bars[j].style.height = `${Math.max(1, pct)}%`;
    }
  }

  tempSlider.addEventListener('input', () => {
    const temp = parseInt(tempSlider.value) / 10;
    set('temperature', temp);
    tempValue.textContent = temp.toFixed(1);
    updateTempPreview(temp);
  });

  btnGenerate.addEventListener('click', async () => {
    const temp = parseInt(tempSlider.value) / 10;
    await generate(vocab, temp);
    updateTempPreview(temp);
  });

  // Head selector toggle
  const headBtns = document.querySelectorAll('.attn-head-selector button');
  headBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      headBtns.forEach(b => b.setAttribute('aria-pressed', 'false'));
      btn.setAttribute('aria-pressed', 'true');
      activeHead = btn.dataset.head;
      if (selectedTokenIdx >= 0) {
        renderArcs(selectedTokenIdx);
      }
    });
  });

  // Re-render arcs on resize
  const resizeObserver = new ResizeObserver(() => {
    if (selectedTokenIdx >= 0 && storedAttn.length > 0) {
      renderArcs(selectedTokenIdx);
    }
  });
  resizeObserver.observe(document.getElementById('attn-viz-container'));

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

  // Auto-generate on init so cards are populated immediately
  const temp = parseInt(tempSlider.value) / 10;
  generate(vocab, temp).then(() => updateTempPreview(temp));
}
