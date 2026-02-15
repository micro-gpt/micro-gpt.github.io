/**
 * Custom tooltip system — replaces browser title tooltips with styled popups.
 * Intercepts [title] attributes, moves content to data-tooltip, and shows
 * a positioned tooltip on hover after a 200ms delay.
 */

let tooltipEl = null;
let hideTimeout = null;
let showTimeout = null;

function createTooltipEl() {
  const el = document.createElement('div');
  el.className = 'custom-tooltip';
  el.setAttribute('role', 'tooltip');
  el.hidden = true;
  document.body.appendChild(el);
  return el;
}

function show(target) {
  const text = target.getAttribute('data-tooltip');
  if (!text) return;

  if (!tooltipEl) tooltipEl = createTooltipEl();
  tooltipEl.textContent = text;
  tooltipEl.hidden = false;

  // Position relative to target
  const rect = target.getBoundingClientRect();
  const ttRect = tooltipEl.getBoundingClientRect();
  const pad = 8;

  let top = rect.top - ttRect.height - pad;
  let left = rect.left + rect.width / 2 - ttRect.width / 2;

  // Flip below if no room above
  if (top < pad) {
    top = rect.bottom + pad;
  }

  // Clamp horizontal
  left = Math.max(pad, Math.min(left, window.innerWidth - ttRect.width - pad));

  tooltipEl.style.top = `${top + window.scrollY}px`;
  tooltipEl.style.left = `${left}px`;
}

function hide() {
  if (tooltipEl) tooltipEl.hidden = true;
}

export function initTooltips() {
  // Convert existing title attributes to data-tooltip on first encounter
  document.addEventListener('pointerenter', (e) => {
    if (!(e.target instanceof Element)) return;
    const target = e.target.closest('[title], [data-tooltip]');
    if (!target) return;

    // Move title to data-tooltip (prevents native tooltip)
    if (target.hasAttribute('title') && !target.hasAttribute('data-tooltip')) {
      target.setAttribute('data-tooltip', target.getAttribute('title'));
      target.removeAttribute('title');
    }

    clearTimeout(hideTimeout);
    clearTimeout(showTimeout);
    showTimeout = setTimeout(() => show(target), 200);
  }, true);

  document.addEventListener('pointerleave', (e) => {
    if (!(e.target instanceof Element)) return;
    const target = e.target.closest('[data-tooltip]');
    if (!target) return;
    clearTimeout(showTimeout);
    hideTimeout = setTimeout(hide, 100);
  }, true);

  // Hide on scroll/click
  document.addEventListener('scroll', () => {
    clearTimeout(showTimeout);
    hide();
  }, { passive: true });

  document.addEventListener('pointerdown', () => {
    clearTimeout(showTimeout);
    hide();
  });
}
