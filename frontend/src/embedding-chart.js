/**
 * Parallel coordinates embedding chart — shows all 16 dimensions of learned
 * character embeddings without dimensionality reduction.
 * Each polyline is one token (a–z + BOS). Current token highlighted.
 */

import { createChart, FONT_FAMILY, tooltipWrap } from './echarts-setup.js';
import { getStateDict, N_EMBD } from './gpt.js';
import { get, subscribe } from './state.js';

let chart = null;
let cleanupSubs = [];

export function initEmbeddingChart(el, vocab) {
  for (const unsub of cleanupSubs) unsub();
  cleanupSubs = [];

  chart = createChart(el);
  updateEmbedding(vocab);

  cleanupSubs.push(subscribe('weightsVersion', () => updateEmbedding(vocab)));
  cleanupSubs.push(subscribe('token', () => updateHighlight(vocab)));
}

function buildOption(vocab) {
  const sd = getStateDict();
  if (!sd || !sd.wte) return null;

  const wte = sd.wte;
  const currentToken = get('token');
  const labels = [...vocab.chars, 'BOS'];

  // Find per-dimension min/max for axis ranges
  const mins = new Array(N_EMBD).fill(Infinity);
  const maxs = new Array(N_EMBD).fill(-Infinity);
  for (const row of wte) {
    for (let d = 0; d < N_EMBD; d++) {
      if (row[d] < mins[d]) mins[d] = row[d];
      if (row[d] > maxs[d]) maxs[d] = row[d];
    }
  }

  // Parallel axis definitions — one per embedding dimension
  const parallelAxis = Array.from({ length: N_EMBD }, (_, d) => ({
    dim: d,
    name: `d${d}`,
    nameTextStyle: { fontSize: 10, color: '#6B7585', fontFamily: FONT_FAMILY },
    axisLine: { lineStyle: { color: '#2A3140' } },
    axisTick: { show: false },
    axisLabel: { show: false },
    min: mins[d],
    max: maxs[d],
  }));

  // Data: one row per token, values = 16-dim embedding vector
  const data = wte.map((row, i) => ({
    value: row,
    name: labels[i],
    lineStyle: {
      color: i === currentToken ? '#5B8DEF' : 'rgba(160, 170, 184, 0.25)',
      width: i === currentToken ? 3 : 1,
      opacity: i === currentToken ? 1 : 0.4,
    },
    emphasis: {
      lineStyle: { color: '#5B8DEF', width: 3, opacity: 1 },
    },
  }));

  // Move the current token to the end so it renders on top
  if (currentToken != null && currentToken < data.length) {
    const item = data.splice(currentToken, 1)[0];
    data.push(item);
  }

  return {
    parallelAxis,
    parallel: {
      left: 40,
      right: 40,
      top: 30,
      bottom: 40,
      parallelAxisDefault: {
        axisLine: { lineStyle: { color: '#2A3140' } },
        axisTick: { show: false },
        axisLabel: { show: false },
        nameGap: 20,
      },
    },
    tooltip: {
      trigger: 'item',
      formatter: (params) => {
        const vec = params.value;
        const preview = vec.slice(0, 8).map(v => v.toFixed(3)).join(', ');
        const rest = vec.slice(8).map(v => v.toFixed(3)).join(', ');
        return `<strong>${params.name}</strong><br/>` +
          tooltipWrap(`[${preview},<br/>&nbsp;${rest}]`);
      },
    },
    series: [{
      type: 'parallel',
      data,
      smooth: 0.3,
      lineStyle: { width: 1 },
      animationDuration: 600,
      animationEasing: 'cubicOut',
    }],
  };
}

function updateEmbedding(vocab) {
  if (!chart) return;
  const option = buildOption(vocab);
  if (option) chart.setOption(option, true);
}

function updateHighlight(vocab) {
  if (!chart) return;
  const option = buildOption(vocab);
  if (option) chart.setOption(option, true);
}
