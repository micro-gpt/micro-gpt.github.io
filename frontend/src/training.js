/**
 * Training visualization — SVG loss curve with animated draw-on,
 * cosine LR curve, range slider scrubber, checkpoint samples.
 */

const SVG_NS = 'http://www.w3.org/2000/svg';

function createTrainingSVG(trainingLog) {
  const w = 800;
  const h = 300;
  const padL = 55;
  const padR = 55;
  const padT = 20;
  const padB = 40;
  const plotW = w - padL - padR;
  const plotH = h - padT - padB;

  const maxStep = trainingLog.length;
  const maxLoss = Math.ceil(Math.max(...trainingLog.map(d => d.loss)));
  const maxLR = Math.max(...trainingLog.map(d => d.lr));

  const xScale = (step) => padL + ((step - 1) / (maxStep - 1)) * plotW;
  const yLoss = (loss) => padT + (1 - loss / maxLoss) * plotH;
  const yLR = (lr) => padT + (1 - lr / maxLR) * plotH;

  const svg = document.createElementNS(SVG_NS, 'svg');
  svg.setAttribute('viewBox', `0 0 ${w} ${h}`);
  svg.setAttribute('width', '100%');
  svg.setAttribute('height', h);
  svg.setAttribute('aria-label', `Training loss curve over ${maxStep} steps, decreasing from ${maxLoss.toFixed(1)} to ${trainingLog[trainingLog.length - 1].loss.toFixed(2)}`);
  svg.setAttribute('role', 'img');
  svg.style.maxWidth = `${w}px`;

  // Grid lines
  for (let i = 0; i <= 4; i++) {
    const yVal = (i / 4) * maxLoss;
    const y = yLoss(yVal);
    const line = document.createElementNS(SVG_NS, 'line');
    line.setAttribute('x1', padL);
    line.setAttribute('y1', y);
    line.setAttribute('x2', w - padR);
    line.setAttribute('y2', y);
    line.setAttribute('stroke', '#1e293b');
    line.setAttribute('stroke-width', '1');
    svg.appendChild(line);

    const label = document.createElementNS(SVG_NS, 'text');
    label.setAttribute('x', padL - 8);
    label.setAttribute('y', y + 4);
    label.setAttribute('text-anchor', 'end');
    label.setAttribute('fill', '#64748b');
    label.setAttribute('font-size', '11');
    label.setAttribute('font-family', '-apple-system, BlinkMacSystemFont, sans-serif');
    label.textContent = yVal.toFixed(1);
    svg.appendChild(label);
  }

  // X axis labels
  for (let s = 0; s <= 500; s += 100) {
    if (s === 0) continue;
    const x = xScale(s);
    const label = document.createElementNS(SVG_NS, 'text');
    label.setAttribute('x', x);
    label.setAttribute('y', h - 8);
    label.setAttribute('text-anchor', 'middle');
    label.setAttribute('fill', '#64748b');
    label.setAttribute('font-size', '11');
    label.setAttribute('font-family', '-apple-system, BlinkMacSystemFont, sans-serif');
    label.textContent = s;
    svg.appendChild(label);
  }

  // Axis labels
  const lossLabel = document.createElementNS(SVG_NS, 'text');
  lossLabel.setAttribute('x', 14);
  lossLabel.setAttribute('y', padT + plotH / 2);
  lossLabel.setAttribute('text-anchor', 'middle');
  lossLabel.setAttribute('transform', `rotate(-90, 14, ${padT + plotH / 2})`);
  lossLabel.setAttribute('fill', '#3b82f6');
  lossLabel.setAttribute('font-size', '12');
  lossLabel.setAttribute('font-weight', '600');
  lossLabel.setAttribute('font-family', '-apple-system, BlinkMacSystemFont, sans-serif');
  lossLabel.textContent = 'Loss';
  svg.appendChild(lossLabel);

  const lrLabel = document.createElementNS(SVG_NS, 'text');
  lrLabel.setAttribute('x', w - 14);
  lrLabel.setAttribute('y', padT + plotH / 2);
  lrLabel.setAttribute('text-anchor', 'middle');
  lrLabel.setAttribute('transform', `rotate(90, ${w - 14}, ${padT + plotH / 2})`);
  lrLabel.setAttribute('fill', '#8b5cf6');
  lrLabel.setAttribute('font-size', '12');
  lrLabel.setAttribute('font-weight', '600');
  lrLabel.setAttribute('font-family', '-apple-system, BlinkMacSystemFont, sans-serif');
  lrLabel.textContent = 'Learning Rate';
  svg.appendChild(lrLabel);

  // LR curve path (purple, behind loss)
  let lrPathD = '';
  trainingLog.forEach((d, i) => {
    const x = xScale(d.step);
    const y = yLR(d.lr);
    lrPathD += i === 0 ? `M${x},${y}` : `L${x},${y}`;
  });

  const lrPath = document.createElementNS(SVG_NS, 'path');
  lrPath.setAttribute('d', lrPathD);
  lrPath.setAttribute('fill', 'none');
  lrPath.setAttribute('stroke', '#8b5cf6');
  lrPath.setAttribute('stroke-width', '1.5');
  lrPath.setAttribute('opacity', '0.5');
  svg.appendChild(lrPath);

  // Loss curve path (blue)
  let lossPathD = '';
  trainingLog.forEach((d, i) => {
    const x = xScale(d.step);
    const y = yLoss(d.loss);
    lossPathD += i === 0 ? `M${x},${y}` : `L${x},${y}`;
  });

  const lossPath = document.createElementNS(SVG_NS, 'path');
  lossPath.setAttribute('d', lossPathD);
  lossPath.setAttribute('fill', 'none');
  lossPath.setAttribute('stroke', '#3b82f6');
  lossPath.setAttribute('stroke-width', '2');
  svg.appendChild(lossPath);

  // Animate draw-on
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  if (!reduceMotion) {
    const totalLength = lossPath.getTotalLength();
    lossPath.style.strokeDasharray = totalLength;
    lossPath.style.strokeDashoffset = totalLength;
    lossPath.style.transition = 'stroke-dashoffset 2s ease-out';
    requestAnimationFrame(() => {
      lossPath.style.strokeDashoffset = '0';
    });

    const lrLength = lrPath.getTotalLength();
    lrPath.style.strokeDasharray = lrLength;
    lrPath.style.strokeDashoffset = lrLength;
    lrPath.style.transition = 'stroke-dashoffset 2s ease-out';
    requestAnimationFrame(() => {
      lrPath.style.strokeDashoffset = '0';
    });
  }

  // Scrubber indicator line
  const scrubLine = document.createElementNS(SVG_NS, 'line');
  scrubLine.setAttribute('y1', padT);
  scrubLine.setAttribute('y2', padT + plotH);
  scrubLine.setAttribute('stroke', '#f1f5f9');
  scrubLine.setAttribute('stroke-width', '1');
  scrubLine.setAttribute('opacity', '0.4');
  scrubLine.setAttribute('stroke-dasharray', '4 3');
  scrubLine.id = 'scrub-line';
  svg.appendChild(scrubLine);

  // Scrubber dot on loss curve
  const scrubDot = document.createElementNS(SVG_NS, 'circle');
  scrubDot.setAttribute('r', '5');
  scrubDot.setAttribute('fill', '#3b82f6');
  scrubDot.setAttribute('stroke', '#f1f5f9');
  scrubDot.setAttribute('stroke-width', '2');
  scrubDot.id = 'scrub-dot';
  svg.appendChild(scrubDot);

  return { svg, xScale, yLoss };
}

function renderCheckpoints(checkpoints, activeStep) {
  const container = document.getElementById('checkpoint-container');
  const steps = Object.keys(checkpoints).map(Number).sort((a, b) => a - b);

  container.innerHTML = steps.map(step => {
    const names = checkpoints[String(step)];
    const isActive = step <= activeStep;
    const isCurrent = step === steps.reduce((prev, s) => s <= activeStep ? s : prev, steps[0]);
    return `
      <div class="checkpoint-card ${isCurrent ? 'active' : ''}" ${!isActive ? 'style="opacity:0.4"' : ''}>
        <div class="step-label">Step ${step}</div>
        <div class="names">${names.join('\n')}</div>
      </div>
    `;
  }).join('');
}

export function initTraining({ trainingLog, checkpoints }) {
  const container = document.getElementById('training-svg-container');
  const { svg, xScale, yLoss } = createTrainingSVG(trainingLog);
  container.innerHTML = '';
  container.appendChild(svg);

  const slider = document.getElementById('step-slider');
  const stepValue = document.getElementById('step-value');

  function updateScrubber(step) {
    const entry = trainingLog[step - 1];
    const x = xScale(entry.step);
    const y = yLoss(entry.loss);

    const scrubLine = document.getElementById('scrub-line');
    const scrubDot = document.getElementById('scrub-dot');
    scrubLine.setAttribute('x1', x);
    scrubLine.setAttribute('x2', x);
    scrubDot.setAttribute('cx', x);
    scrubDot.setAttribute('cy', y);

    stepValue.textContent = step;
    renderCheckpoints(checkpoints, step);
  }

  slider.addEventListener('input', () => {
    updateScrubber(parseInt(slider.value) + 1);
  });

  // Initialize at final step
  updateScrubber(500);
}
