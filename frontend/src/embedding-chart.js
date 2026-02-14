/**
 * 2D PCA embedding scatter — shows learned character embeddings projected to 2D.
 * Highlights current token, animates on checkpoint change.
 */

import { createChart, FONT_FAMILY } from './echarts-setup.js';
import { getStateDict } from './gpt.js';
import { get, subscribe } from './state.js';

let chart = null;
let cleanupSubs = [];

/** Simple 2-component PCA on an n×d matrix. Returns n×2 coordinates. */
function pca2d(matrix) {
  const n = matrix.length;
  const d = matrix[0].length;

  // Center the data
  const mean = new Array(d).fill(0);
  for (const row of matrix) for (let j = 0; j < d; j++) mean[j] += row[j];
  for (let j = 0; j < d; j++) mean[j] /= n;

  const centered = matrix.map(row => row.map((v, j) => v - mean[j]));

  // Covariance matrix (d×d)
  const cov = Array.from({ length: d }, () => new Array(d).fill(0));
  for (const row of centered) {
    for (let i = 0; i < d; i++) {
      for (let j = i; j < d; j++) {
        cov[i][j] += row[i] * row[j];
      }
    }
  }
  for (let i = 0; i < d; i++) {
    cov[i][i] /= n;
    for (let j = i + 1; j < d; j++) {
      cov[i][j] /= n;
      cov[j][i] = cov[i][j];
    }
  }

  // Power iteration to find top 2 eigenvectors
  function powerIteration(mat, deflateVec) {
    let vec = new Array(d);
    for (let i = 0; i < d; i++) vec[i] = Math.random() - 0.5;

    for (let iter = 0; iter < 50; iter++) {
      let result = new Array(d).fill(0);
      for (let i = 0; i < d; i++) {
        for (let j = 0; j < d; j++) result[i] += mat[i][j] * vec[j];
      }

      // Deflate
      if (deflateVec) {
        let dot = 0;
        for (let i = 0; i < d; i++) dot += result[i] * deflateVec[i];
        for (let i = 0; i < d; i++) result[i] -= dot * deflateVec[i];
      }

      let norm = 0;
      for (let i = 0; i < d; i++) norm += result[i] * result[i];
      norm = Math.sqrt(norm);
      if (norm < 1e-10) break;
      for (let i = 0; i < d; i++) vec[i] = result[i] / norm;
    }
    return vec;
  }

  const pc1 = powerIteration(cov, null);
  const pc2 = powerIteration(cov, pc1);

  return centered.map(row => [
    row.reduce((s, v, i) => s + v * pc1[i], 0),
    row.reduce((s, v, i) => s + v * pc2[i], 0),
  ]);
}

export function initEmbeddingChart(el, vocab) {
  for (const unsub of cleanupSubs) unsub();
  cleanupSubs = [];

  chart = createChart(el);
  updateEmbedding(vocab);

  cleanupSubs.push(subscribe('weightsVersion', () => updateEmbedding(vocab)));
  cleanupSubs.push(subscribe('token', () => updateHighlight(vocab)));
}

function updateEmbedding(vocab) {
  const sd = getStateDict();
  if (!sd || !sd.wte) return;

  const wte = sd.wte;
  const coords = pca2d(wte);
  const currentToken = get('token');

  const labels = [...vocab.chars, 'BOS'];

  const data = coords.map((pos, i) => ({
    value: [...pos, i],
    name: labels[i],
    itemStyle: {
      color: i === currentToken ? '#5B8DEF' : 'rgba(160, 170, 184, 0.6)',
    },
    symbolSize: i === currentToken ? 16 : 10,
  }));

  chart.setOption({
    tooltip: {
      trigger: 'item',
      formatter: (params) => {
        const idx = params.value[2];
        const vec = sd.wte[idx];
        const preview = vec.slice(0, 6).map(v => v.toFixed(3)).join(', ') + '...';
        return `<strong>${labels[idx]}</strong><br/>` +
               `<span style="font-family:monospace;font-size:11px">Embedding: [${preview}]</span>`;
      },
    },
    xAxis: {
      type: 'value',
      axisLabel: { show: false },
      axisTick: { show: false },
      splitLine: { lineStyle: { color: '#1E2433' } },
      name: 'PC1',
      nameTextStyle: { color: '#6B7585', fontSize: 11 },
    },
    yAxis: {
      type: 'value',
      axisLabel: { show: false },
      axisTick: { show: false },
      splitLine: { lineStyle: { color: '#1E2433' } },
      name: 'PC2',
      nameTextStyle: { color: '#6B7585', fontSize: 11 },
    },
    series: [{
      type: 'scatter',
      data,
      label: {
        show: true,
        position: 'right',
        formatter: '{b}',
        fontFamily: FONT_FAMILY,
        fontSize: 11,
        color: '#A0AAB8',
      },
      emphasis: {
        itemStyle: { borderColor: '#5B8DEF', borderWidth: 2 },
        label: { color: '#E8ECF1', fontWeight: 'bold' },
      },
      animationDuration: 800,
      animationEasing: 'cubicOut',
    }],
    grid: { left: 40, right: 40, top: 30, bottom: 40 },
  }, true);
}

function updateHighlight(vocab) {
  const sd = getStateDict();
  if (!sd || !sd.wte || !chart) return;

  const currentToken = get('token');
  const coords = pca2d(sd.wte);
  const labels = [...vocab.chars, 'BOS'];

  const data = coords.map((pos, i) => ({
    value: [...pos, i],
    name: labels[i],
    itemStyle: {
      color: i === currentToken ? '#5B8DEF' : 'rgba(160, 170, 184, 0.6)',
    },
    symbolSize: i === currentToken ? 16 : 10,
  }));

  chart.setOption({
    series: [{ data }],
  });
}
