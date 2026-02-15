/**
 * Signal magnitude bar chart — shows L2 norm of activations at each
 * stage of the forward pass, revealing where the network amplifies
 * or attenuates the signal.
 */

import { createChart, FONT_FAMILY, monoTooltip } from './echarts-setup.js';

let chart = null;

const LAYERS = [
  { key: 'tokEmb',        label: 'Token Embed',  color: '#5B8DEF' },
  { key: 'posEmb',        label: 'Pos Embed',    color: '#5B8DEF' },
  { key: 'combined',      label: 'Combined',     color: '#5B8DEF' },
  { key: 'postNorm0',     label: 'RMSNorm\u2080',color: '#9B7AEA' },
  { key: 'postNorm1',     label: 'RMSNorm\u2081',color: '#9B7AEA' },
  { key: 'attnOut',       label: 'Attention',    color: '#22D3EE' },
  { key: 'postResidual1', label: 'Residual\u2081',color: '#6B7585' },
  { key: 'postNorm2',     label: 'RMSNorm\u2082',color: '#9B7AEA' },
  { key: 'mlpOut',        label: 'MLP',          color: '#FB923C' },
  { key: 'postResidual2', label: 'Residual\u2082',color: '#6B7585' },
  { key: 'logits',        label: 'LM Head',      color: '#4ADE80' },
];

function l2Norm(vec) {
  if (!vec) return 0;
  return Math.sqrt(vec.reduce((s, v) => s + v * v, 0));
}

export function initFlowChart(el) {
  if (chart) chart.dispose();
  chart = createChart(el);
  updateFlowChart(null);
}

export function updateFlowChart(intermediates) {
  if (!chart) return;

  const labels = LAYERS.map(l => l.label);
  const values = LAYERS.map(l => ({
    value: intermediates ? l2Norm(intermediates[l.key]) : 0,
    itemStyle: { color: l.color },
  }));

  chart.setOption({
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'shadow' },
      formatter: (params) => {
        const p = params[0];
        return `<strong>${p.name}</strong><br/>` +
               monoTooltip('L2 norm', p.value.toFixed(2));
      },
    },
    xAxis: {
      type: 'value',
      name: 'L2 Norm',
      nameTextStyle: { color: '#6B7585', fontSize: 11 },
      axisLabel: { color: '#6B7585' },
      splitLine: { lineStyle: { color: '#1E2433' } },
    },
    yAxis: {
      type: 'category',
      data: labels,
      inverse: true,
      axisLabel: { fontFamily: FONT_FAMILY, fontSize: 11, color: '#A0AAB8' },
      axisTick: { show: false },
      axisLine: { show: false },
    },
    series: [{
      type: 'bar',
      data: values,
      barWidth: '60%',
      emphasis: {
        itemStyle: { shadowBlur: 8, shadowColor: 'rgba(91, 141, 239, 0.3)' },
      },
      animationDuration: 600,
      animationEasing: 'cubicOut',
    }],
    grid: { left: 90, right: 30, top: 10, bottom: 30 },
  }, true);
}

export function disposeFlowChart() {
  if (chart) {
    chart.dispose();
    chart = null;
  }
}
