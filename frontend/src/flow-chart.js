/**
 * Sankey-style data flow diagram using ECharts.
 * Shows activation magnitudes flowing through the network layers.
 */

import { createChart, FONT_FAMILY, monoTooltip } from './echarts-setup.js';

let chart = null;

const NODE_COLORS = {
  'Token Embed': '#5B8DEF',
  'Pos Embed': '#5B8DEF',
  'Combined': '#5B8DEF',
  'RMSNorm₀': '#9B7AEA',
  'RMSNorm₁': '#9B7AEA',
  'Attention': '#22D3EE',
  'Residual₁': '#6B7585',
  'RMSNorm₂': '#9B7AEA',
  'MLP': '#FB923C',
  'Residual₂': '#6B7585',
  'LM Head': '#4ADE80',
  'Softmax': '#4ADE80',
};

function l2Norm(vec) {
  if (!vec) return 0;
  return Math.sqrt(vec.reduce((s, v) => s + v * v, 0));
}

const FLOW_PAIRS = [
  ['Token Embed', 'Combined', 'tokEmb'],
  ['Pos Embed', 'Combined', 'posEmb'],
  ['Combined', 'RMSNorm₀', 'combined'],
  ['RMSNorm₀', 'RMSNorm₁', 'postNorm0'],
  ['RMSNorm₁', 'Attention', 'postNorm1'],
  ['Attention', 'Residual₁', 'attnOut'],
  ['Residual₁', 'RMSNorm₂', 'postResidual1'],
  ['RMSNorm₂', 'MLP', 'postNorm2'],
  ['MLP', 'Residual₂', 'mlpOut'],
  ['Residual₂', 'LM Head', 'postResidual2'],
  ['LM Head', 'Softmax', 'logits'],
];

function buildSankeyData(intermediates) {
  const nodes = Object.keys(NODE_COLORS).map(name => ({
    name,
    itemStyle: { color: NODE_COLORS[name], borderColor: 'rgba(0,0,0,0.3)' },
  }));

  // Uniform link values — Sankey is a flow diagram, not a magnitude chart.
  // L2 norms shown in tooltip only.
  const links = FLOW_PAIRS.map(([source, target, key]) => ({
    source,
    target,
    value: 1,
    magnitude: intermediates ? l2Norm(intermediates[key]) : 0,
  }));

  return { nodes, links };
}

export function initFlowChart(el) {
  if (chart) chart.dispose();
  chart = createChart(el);
  updateFlowChart(null);
}

export function updateFlowChart(intermediates) {
  if (!chart) return;

  const { nodes, links } = buildSankeyData(intermediates);

  chart.setOption({
    tooltip: {
      trigger: 'item',
      formatter: (params) => {
        if (params.dataType === 'node') {
          return `<strong>${params.name}</strong>`;
        }
        return `${params.data.source} \u2192 ${params.data.target}<br/>` +
               monoTooltip('L2 norm', params.data.magnitude.toFixed(2));
      },
    },
    series: [{
      type: 'sankey',
      data: nodes,
      links,
      orient: 'vertical',
      nodeAlign: 'justify',
      layoutIterations: 32,
      nodeWidth: 20,
      nodeGap: 12,
      label: {
        show: true,
        position: 'right',
        fontFamily: FONT_FAMILY,
        fontSize: 11,
        color: '#A0AAB8',
      },
      lineStyle: {
        color: 'gradient',
        curveness: 0.5,
        opacity: 0.4,
      },
      emphasis: {
        lineStyle: { opacity: 0.7 },
      },
      animationDuration: 600,
      animationEasing: 'cubicOut',
    }],
  }, true);
}

export function disposeFlowChart() {
  if (chart) {
    chart.dispose();
    chart = null;
  }
}
