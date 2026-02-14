/**
 * Inference section — temperature control, token-by-token generation,
 * raw logits + probabilities, interactive attention arc visualization,
 * collapsible intermediate values viewer.
 */

import { gptForward, softmax, sampleFrom, N_LAYER, N_HEAD, BLOCK_SIZE } from './gpt.js';
import { set } from './state.js';
import { t } from './content.js';

const SVG_NS = 'http://www.w3.org/2000/svg';
const HEAD_COLORS = ['var(--accent-blue)', 'var(--accent-purple)', 'var(--accent-green)', 'var(--accent-cyan)'];

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
  svg.innerHTML = '';

  if (tokenIdx < 0 || tokenIdx >= storedAttn.length) return;

  const positions = measureTokenPositions();
  if (positions.length === 0) return;

  const stepAttn = storedAttn[tokenIdx];
  if (!stepAttn) return;

  // Mark target tokens
  const tokenBtns = document.querySelectorAll('#attn-token-row .attn-token');
  tokenBtns.forEach(b => b.classList.remove('target'));

  const singleHead = activeHead !== 'all';
  const headList = singleHead ? [parseInt(activeHead)] : [0, 1, 2, 3];

  // Collect all arcs
  const arcs = [];
  for (const h of headList) {
    const weights = stepAttn[h];
    if (!weights) continue;
    for (let t = 0; t < weights.length; t++) {
      if (t === tokenIdx) continue; // skip self-attention
      if (weights[t] < 0.01) continue;
      arcs.push({ head: h, target: t, weight: weights[t] });
    }
  }

  // Sort thin-first so heavy arcs render on top
  arcs.sort((a, b) => a.weight - b.weight);

  const y = positions[0].top;
  let maxDepth = 0;

  for (const arc of arcs) {
    const srcPos = positions[tokenIdx];
    const tgtPos = positions[arc.target];
    if (!srcPos || !tgtPos) continue;

    // Mark target token
    if (tokenBtns[arc.target]) {
      tokenBtns[arc.target].classList.add('target');
    }

    const x1 = srcPos.cx;
    const x2 = tgtPos.cx;
    const dist = Math.abs(tokenIdx - arc.target);
    const depth = y + 12 + dist * 18;
    if (depth > maxDepth) maxDepth = depth;

    const strokeWidth = 1.5 + arc.weight * 4.5;
    const opacity = singleHead
      ? 0.3 + arc.weight * 0.65
      : 0.2 + arc.weight * 0.5;

    const path = document.createElementNS(SVG_NS, 'path');
    path.setAttribute('d', `M ${x1},${y} Q ${(x1 + x2) / 2},${depth} ${x2},${y}`);
    path.setAttribute('class', 'attn-arc');
    path.setAttribute('stroke', HEAD_COLORS[arc.head]);
    path.setAttribute('stroke-width', strokeWidth);
    path.setAttribute('opacity', opacity);
    svg.appendChild(path);

    // Weight label on significant arcs in single-head mode
    if (singleHead && arc.weight >= 0.1) {
      const label = document.createElementNS(SVG_NS, 'text');
      const midX = (x1 + x2) / 2;
      const midY = (y + depth) / 2 + 4;
      label.setAttribute('x', midX);
      label.setAttribute('y', midY);
      label.setAttribute('text-anchor', 'middle');
      label.setAttribute('class', 'attn-arc-label');
      label.textContent = `${Math.round(arc.weight * 100)}%`;
      svg.appendChild(label);
    }
  }

  // Set SVG height to fit arcs
  svg.style.height = maxDepth > 0 ? `${maxDepth + 8}px` : '0';
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
    content.innerHTML = '<p style="color:var(--text-dim);font-size:0.85rem;padding:0.5rem">Run a generation step to see intermediate values.</p>';
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
    return `<div class="data-viewer inter-block" style="border-left-color:${borderColor};margin-bottom:0.5rem">
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

  // Final state — select last token
  selectedTokenIdx = storedTokens.length - 1;
  generating = false;
  renderAttnTokens();

  if (generatedTokens.length === 0) {
    output.innerHTML = '<span style="color:var(--text-dim)">(empty)</span>';
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

  tempSlider.addEventListener('input', () => {
    const temp = parseInt(tempSlider.value) / 10;
    set('temperature', temp);
    tempValue.textContent = temp.toFixed(1);
  });

  btnGenerate.addEventListener('click', () => {
    const temp = parseInt(tempSlider.value) / 10;
    generate(vocab, temp);
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
}
