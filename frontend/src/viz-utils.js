/**
 * Shared visualization utilities — canvas heatmaps and attention arc SVG rendering.
 * Used by training.js, architecture.js, and inference.js to eliminate code duplication.
 */

const SVG_NS = 'http://www.w3.org/2000/svg';
const DEFAULT_HEAD_COLORS = ['var(--accent-blue)', 'var(--accent-purple)', 'var(--accent-green)', 'var(--accent-cyan)'];

/**
 * Draw a diverging blue/red heatmap onto a canvas element.
 * Sets canvas width/height to match matrix dimensions (pixelated rendering).
 */
export function drawCanvasHeatmap(canvas, matrix) {
  const rows = matrix.length;
  const cols = matrix[0].length;
  canvas.width = cols;
  canvas.height = rows;
  const ctx = canvas.getContext('2d');

  let absMax = 0;
  for (const row of matrix) for (const v of row) absMax = Math.max(absMax, Math.abs(v));
  if (absMax < 0.001) absMax = 0.001;

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const v = matrix[r][c] / absMax;
      if (v >= 0) {
        ctx.fillStyle = `rgb(${Math.round(15 + (1 - v) * 30)}, ${Math.round(23 + (1 - v) * 30)}, ${Math.round(80 + v * 175)})`;
      } else {
        const a = Math.abs(v);
        ctx.fillStyle = `rgb(${Math.round(80 + a * 175)}, ${Math.round(23 + (1 - a) * 30)}, ${Math.round(15 + (1 - a) * 30)})`;
      }
      ctx.fillRect(c, r, 1, 1);
    }
  }
}

/**
 * Draw attention arcs as quadratic bezier SVG paths.
 * @param {SVGElement} svg - Target SVG element (will be cleared)
 * @param {number} sourceIdx - Index of the source token
 * @param {Array} stepAttn - Attention weights per head: stepAttn[head][targetPos]
 * @param {Array} positions - Measured positions: [{cx, top}, ...]
 * @param {Object} opts
 * @param {string} opts.head - 'all' | 'avg' | '0'-'3'
 * @param {Array} opts.headColors - CSS color strings per head
 * @param {boolean} opts.showLabels - Show weight % labels on arcs
 * @param {NodeList|Array} opts.targetElements - Elements to mark with .target class
 */
export function drawAttentionArcs(svg, sourceIdx, stepAttn, positions, opts = {}) {
  const {
    head = 'all',
    headColors = DEFAULT_HEAD_COLORS,
    showLabels = false,
    targetElements = null,
  } = opts;

  svg.innerHTML = '';

  if (!stepAttn || positions.length === 0) {
    svg.style.height = '0';
    return;
  }

  if (targetElements) {
    targetElements.forEach(el => el.classList.remove('target'));
  }

  const singleHead = head !== 'all' && head !== 'avg';
  const headList = singleHead ? [parseInt(head)] : [0, 1, 2, 3];

  const arcs = [];
  for (const h of headList) {
    const weights = stepAttn[h];
    if (!weights) continue;
    for (let t = 0; t < weights.length; t++) {
      if (t === sourceIdx) continue;
      if (weights[t] < 0.01) continue;
      arcs.push({ head: h, target: t, weight: weights[t] });
    }
  }

  // Sort thin-first so heavy arcs render on top
  arcs.sort((a, b) => a.weight - b.weight);

  const y = positions[0]?.top || 0;
  let maxDepth = 0;

  for (const arc of arcs) {
    const srcPos = positions[sourceIdx];
    const tgtPos = positions[arc.target];
    if (!srcPos || !tgtPos) continue;

    if (targetElements?.[arc.target]) {
      targetElements[arc.target].classList.add('target');
    }

    const x1 = srcPos.cx;
    const x2 = tgtPos.cx;
    const dist = Math.abs(sourceIdx - arc.target);
    const depth = y + 12 + dist * 18;
    if (depth > maxDepth) maxDepth = depth;

    const strokeWidth = 1.5 + arc.weight * 4.5;
    const opacity = singleHead ? 0.3 + arc.weight * 0.65 : 0.2 + arc.weight * 0.5;

    const path = document.createElementNS(SVG_NS, 'path');
    path.setAttribute('d', `M ${x1},${y} Q ${(x1 + x2) / 2},${depth} ${x2},${y}`);
    path.setAttribute('class', 'attn-arc');
    path.setAttribute('stroke', headColors[arc.head]);
    path.setAttribute('stroke-width', strokeWidth);
    path.setAttribute('opacity', opacity);
    svg.appendChild(path);

    if (showLabels && singleHead && arc.weight >= 0.1) {
      const label = document.createElementNS(SVG_NS, 'text');
      label.setAttribute('x', (x1 + x2) / 2);
      label.setAttribute('y', (y + depth) / 2 + 4);
      label.setAttribute('text-anchor', 'middle');
      label.setAttribute('class', 'attn-arc-label');
      label.textContent = `${Math.round(arc.weight * 100)}%`;
      svg.appendChild(label);
    }
  }

  svg.style.height = maxDepth > 0 ? `${maxDepth + 8}px` : '0';
}
