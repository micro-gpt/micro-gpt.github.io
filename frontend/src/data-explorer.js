/**
 * Data Explorer section — training data browser + ECharts charts.
 * Searchable name list, bigram heatmap, length distribution, character position heatmap.
 */

import { createChart, FONT_FAMILY } from './echarts-setup.js';

let allNames = [];
let filteredNames = [];
let displayedCount = 50;
let bigramChart = null;
let lengthChart = null;
let positionChart = null;

export async function initDataExplorer() {
  const res = await fetch('/data/training-docs.json');
  if (!res.ok) return;
  allNames = await res.json();
  filteredNames = allNames;

  renderStats(allNames);
  renderNameList();
  initBigramChart(allNames);
  initLengthChart(allNames);
  initPositionChart(allNames);

  const searchInput = document.getElementById('data-search-input');
  searchInput.addEventListener('input', () => {
    const query = searchInput.value.toLowerCase().trim();
    filteredNames = query ? allNames.filter(n => n.toLowerCase().startsWith(query)) : allNames;
    displayedCount = 50;
    renderStats(filteredNames);
    renderNameList();
  });
}

function renderStats(names) {
  const container = document.getElementById('data-stats');
  const uniqueChars = new Set(names.join('')).size;
  const avgLen = names.length > 0 ? (names.reduce((s, n) => s + n.length, 0) / names.length).toFixed(1) : 0;
  container.innerHTML = `
    <span class="stat"><strong>${names.length.toLocaleString()}</strong> names</span>
    <span class="stat"><strong>${avgLen}</strong> avg length</span>
    <span class="stat"><strong>${uniqueChars}</strong> unique chars</span>
  `;
}

function renderNameList() {
  const container = document.getElementById('data-name-list');
  const visible = filteredNames.slice(0, displayedCount);

  let html = '<div class="data-name-grid">';
  html += visible.map(n => `<span>${escHtml(n)}</span>`).join('');
  html += '</div>';

  if (displayedCount < filteredNames.length) {
    const remaining = filteredNames.length - displayedCount;
    html += `<button class="btn btn-secondary btn-sm data-load-more" id="data-load-more">Show more (${remaining.toLocaleString()} remaining)</button>`;
  }

  container.innerHTML = html;

  const loadMore = document.getElementById('data-load-more');
  if (loadMore) {
    loadMore.addEventListener('click', () => {
      displayedCount += 100;
      renderNameList();
    });
  }
}

function escHtml(s) {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

// --- Bigram Frequency Heatmap ---
function initBigramChart(names) {
  const el = document.getElementById('data-bigram-chart');
  bigramChart = createChart(el);

  const chars = 'abcdefghijklmnopqrstuvwxyz'.split('');
  const counts = {};
  for (const name of names) {
    const lower = name.toLowerCase();
    for (let i = 0; i < lower.length - 1; i++) {
      const bigram = lower[i] + lower[i + 1];
      counts[bigram] = (counts[bigram] || 0) + 1;
    }
  }

  const data = [];
  let maxCount = 0;
  for (let y = 0; y < chars.length; y++) {
    for (let x = 0; x < chars.length; x++) {
      const bigram = chars[y] + chars[x];
      const count = counts[bigram] || 0;
      data.push([x, y, count]);
      if (count > maxCount) maxCount = count;
    }
  }

  bigramChart.setOption({
    tooltip: {
      formatter: (params) => {
        const [x, y, count] = params.value;
        return `<strong>"${chars[y]}${chars[x]}"</strong> appears <strong>${count.toLocaleString()}</strong> times`;
      },
    },
    xAxis: {
      type: 'category',
      data: chars,
      splitArea: { show: true, areaStyle: { color: ['rgba(0,0,0,0)', 'rgba(255,255,255,0.02)'] } },
      axisLabel: { fontFamily: 'monospace', fontSize: 11, color: '#6B7585' },
    },
    yAxis: {
      type: 'category',
      data: chars,
      splitArea: { show: true, areaStyle: { color: ['rgba(0,0,0,0)', 'rgba(255,255,255,0.02)'] } },
      axisLabel: { fontFamily: 'monospace', fontSize: 11, color: '#6B7585' },
    },
    visualMap: {
      min: 0,
      max: maxCount,
      calculable: true,
      orient: 'horizontal',
      left: 'center',
      bottom: 0,
      inRange: {
        color: ['#141926', '#1a2744', '#234580', '#2d63b8', '#5B8DEF', '#9BC2FF'],
      },
      textStyle: { color: '#6B7585', fontFamily: FONT_FAMILY },
    },
    series: [{
      type: 'heatmap',
      data,
      label: { show: false },
      emphasis: {
        itemStyle: { shadowBlur: 10, shadowColor: 'rgba(91, 141, 239, 0.5)' },
      },
    }],
    grid: { left: 30, right: 20, top: 10, bottom: 60 },
  });
}

// --- Name Length Distribution ---
function initLengthChart(names) {
  const el = document.getElementById('data-length-chart');
  lengthChart = createChart(el);

  const lengthCounts = {};
  const lengthExamples = {};
  for (const name of names) {
    const len = name.length;
    lengthCounts[len] = (lengthCounts[len] || 0) + 1;
    if (!lengthExamples[len]) lengthExamples[len] = [];
    if (lengthExamples[len].length < 3) lengthExamples[len].push(name);
  }

  const maxLen = Math.max(...Object.keys(lengthCounts).map(Number));
  const categories = [];
  const values = [];
  for (let i = 1; i <= maxLen; i++) {
    categories.push(String(i));
    values.push(lengthCounts[i] || 0);
  }

  lengthChart.setOption({
    tooltip: {
      trigger: 'axis',
      formatter: (params) => {
        const len = parseInt(params[0].name);
        const count = params[0].value;
        const examples = lengthExamples[len] || [];
        let html = `Length <strong>${len}</strong>: <strong>${count.toLocaleString()}</strong> names`;
        if (examples.length > 0) {
          html += `<br/><span style="color:#A0AAB8">${examples.join(', ')}</span>`;
        }
        return html;
      },
    },
    xAxis: {
      type: 'category',
      data: categories,
      name: 'Name length',
      nameTextStyle: { color: '#6B7585', fontSize: 11 },
      axisLabel: { fontFamily: 'monospace', color: '#6B7585' },
    },
    yAxis: {
      type: 'value',
      name: 'Count',
      nameTextStyle: { color: '#6B7585', fontSize: 11 },
      axisLabel: { color: '#6B7585' },
      splitLine: { lineStyle: { color: '#1E2433' } },
    },
    series: [{
      type: 'bar',
      data: values,
      itemStyle: {
        color: '#5B8DEF',
        borderRadius: [3, 3, 0, 0],
      },
      emphasis: {
        itemStyle: { color: '#7BA4F5' },
      },
    }],
    grid: { left: 50, right: 20, top: 30, bottom: 50 },
  });
}

// --- Character Position Heatmap ---
function initPositionChart(names) {
  const el = document.getElementById('data-position-chart');
  positionChart = createChart(el);

  const maxPos = 8;
  const chars = 'abcdefghijklmnopqrstuvwxyz'.split('');
  const counts = {};
  let maxCount = 0;

  for (const name of names) {
    const lower = name.toLowerCase();
    for (let pos = 0; pos < Math.min(lower.length, maxPos); pos++) {
      const ch = lower[pos];
      if (ch >= 'a' && ch <= 'z') {
        const key = `${pos}-${ch}`;
        counts[key] = (counts[key] || 0) + 1;
        if (counts[key] > maxCount) maxCount = counts[key];
      }
    }
  }

  const data = [];
  for (let x = 0; x < maxPos; x++) {
    for (let y = 0; y < chars.length; y++) {
      const key = `${x}-${chars[y]}`;
      data.push([x, y, counts[key] || 0]);
    }
  }

  positionChart.setOption({
    tooltip: {
      formatter: (params) => {
        const [pos, charIdx, count] = params.value;
        return `<strong>"${chars[charIdx]}"</strong> at position <strong>${pos}</strong>: <strong>${count.toLocaleString()}</strong> times`;
      },
    },
    xAxis: {
      type: 'category',
      data: Array.from({ length: maxPos }, (_, i) => String(i)),
      name: 'Position',
      nameTextStyle: { color: '#6B7585', fontSize: 11 },
      axisLabel: { fontFamily: 'monospace', fontSize: 11, color: '#6B7585' },
    },
    yAxis: {
      type: 'category',
      data: chars,
      axisLabel: { fontFamily: 'monospace', fontSize: 11, color: '#6B7585' },
    },
    visualMap: {
      min: 0,
      max: maxCount,
      calculable: true,
      orient: 'horizontal',
      left: 'center',
      bottom: 0,
      inRange: {
        color: ['#141926', '#1a3040', '#1a5040', '#2a7050', '#4ADE80', '#A5F3C0'],
      },
      textStyle: { color: '#6B7585', fontFamily: FONT_FAMILY },
    },
    series: [{
      type: 'heatmap',
      data,
      label: { show: false },
      emphasis: {
        itemStyle: { shadowBlur: 10, shadowColor: 'rgba(74, 222, 128, 0.5)' },
      },
    }],
    grid: { left: 30, right: 20, top: 10, bottom: 60 },
  });
}
