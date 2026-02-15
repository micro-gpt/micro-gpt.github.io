/**
 * Shared ECharts configuration — tree-shakeable imports, dark theme, init helper.
 */

import { use, init } from 'echarts/core';
import { HeatmapChart, BarChart, SankeyChart, ParallelChart, LineChart } from 'echarts/charts';
import {
  TooltipComponent,
  VisualMapComponent,
  GridComponent,
  DataZoomComponent,
  LegendComponent,
  ParallelComponent,
  MarkPointComponent,
  MarkLineComponent,
} from 'echarts/components';
import { CanvasRenderer } from 'echarts/renderers';

use([
  HeatmapChart,
  BarChart,
  SankeyChart,
  ParallelChart,
  LineChart,
  TooltipComponent,
  VisualMapComponent,
  GridComponent,
  DataZoomComponent,
  LegendComponent,
  ParallelComponent,
  MarkPointComponent,
  MarkLineComponent,
  CanvasRenderer,
]);

const FONT_FAMILY = '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, Cantarell, sans-serif';

const DARK_THEME = {
  backgroundColor: 'transparent',
  textStyle: { fontFamily: FONT_FAMILY, color: '#A0AAB8' },
  title: { textStyle: { color: '#E8ECF1', fontFamily: FONT_FAMILY } },
  legend: { textStyle: { color: '#A0AAB8' } },
  tooltip: {
    backgroundColor: 'rgba(20, 25, 38, 0.92)',
    borderColor: 'rgba(255, 255, 255, 0.06)',
    textStyle: { color: '#E8ECF1', fontFamily: FONT_FAMILY, fontSize: 12 },
    extraCssText: 'backdrop-filter: blur(12px); border-radius: 8px; box-shadow: 0 2px 12px rgba(0,0,0,0.25);',
  },
  categoryAxis: {
    axisLine: { lineStyle: { color: '#2A3140' } },
    axisTick: { lineStyle: { color: '#2A3140' } },
    axisLabel: { color: '#6B7585', fontFamily: FONT_FAMILY },
    splitLine: { lineStyle: { color: '#1E2433' } },
  },
  valueAxis: {
    axisLine: { lineStyle: { color: '#2A3140' } },
    axisTick: { lineStyle: { color: '#2A3140' } },
    axisLabel: { color: '#6B7585', fontFamily: FONT_FAMILY },
    splitLine: { lineStyle: { color: '#1E2433' } },
  },
};

/** Initialize an ECharts instance on a DOM element with the dark theme. */
export function createChart(el) {
  const chart = init(el, DARK_THEME, { renderer: 'canvas' });

  const ro = new ResizeObserver(() => chart.resize());
  ro.observe(el);

  return chart;
}

/** Wrap tooltip HTML in monospace styling (ECharts tooltips require inline styles). */
export function tooltipWrap(html) {
  return `<span style="font-family:monospace;font-size:12px">${html}</span>`;
}

/** Format a label: value tooltip in monospace. */
export function monoTooltip(label, value) {
  return tooltipWrap(`${label}: <strong>${value}</strong>`);
}

export { FONT_FAMILY };
