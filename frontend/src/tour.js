/**
 * Guided walkthrough — opt-in tour that steps through key UI sections.
 */

const STEPS = [
  {
    target: '#btn-intro-generate',
    text: 'This tiny GPT model (4,064 parameters) learned to generate human-like names. Click Generate to see it in action.',
    label: 'Step 1 of 6',
  },
  {
    target: '#section-architecture .input-selector',
    text: 'Choose any input token and position. The forward pass runs instantly through every layer of the network.',
    label: 'Step 2 of 6',
    scrollTo: '#section-architecture',
  },
  {
    target: '#arch-svg-container',
    text: 'The architecture diagram shows live data inside each block. Click any block to see its intermediate values in detail.',
    label: 'Step 3 of 6',
  },
  {
    target: '#section-training .viz-container',
    text: 'Watch how the model learned over 500 training steps. Drag the slider or click checkpoint markers on the loss curve.',
    label: 'Step 4 of 6',
    scrollTo: '#section-training',
  },
  {
    target: '#weight-inspector',
    text: 'The weight filmstrip shows how every parameter matrix evolved during training. Toggle Diff to highlight changes.',
    label: 'Step 5 of 6',
  },
  {
    target: '#section-generation .controls-row',
    text: 'Generate names with adjustable temperature. Click any generated token to trace its forward pass in the architecture section.',
    label: 'Step 6 of 6',
    scrollTo: '#section-generation',
  },
];

let currentStep = 0;
let active = false;

function positionCard(targetRect) {
  const card = document.getElementById('tour-card');
  const cardRect = card.getBoundingClientRect();
  const padding = 16;

  // Default: below the target
  let top = targetRect.bottom + padding;
  let left = targetRect.left + targetRect.width / 2 - cardRect.width / 2;

  // Clamp horizontal
  left = Math.max(padding, Math.min(left, window.innerWidth - cardRect.width - padding));

  // If below would overflow, place above
  if (top + cardRect.height > window.innerHeight - padding) {
    top = targetRect.top - cardRect.height - padding;
  }

  // If above would overflow, place below anyway but clamp
  if (top < padding) {
    top = padding;
  }

  card.style.top = `${top}px`;
  card.style.left = `${left}px`;
}

function showStep(index) {
  currentStep = index;
  const step = STEPS[index];

  const overlay = document.getElementById('tour-overlay');
  const spotlight = document.getElementById('tour-spotlight');
  const card = document.getElementById('tour-card');
  const stepLabel = document.getElementById('tour-step-label');
  const text = document.getElementById('tour-text');
  const backBtn = document.getElementById('tour-back');

  // Scroll to section if needed
  if (step.scrollTo) {
    const scrollTarget = document.querySelector(step.scrollTo);
    if (scrollTarget) {
      scrollTarget.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }

  // Wait for scroll to settle
  setTimeout(() => {
    const target = document.querySelector(step.target);
    if (!target) return;

    const rect = target.getBoundingClientRect();
    const pad = 8;

    spotlight.style.top = `${rect.top - pad}px`;
    spotlight.style.left = `${rect.left - pad}px`;
    spotlight.style.width = `${rect.width + pad * 2}px`;
    spotlight.style.height = `${rect.height + pad * 2}px`;

    stepLabel.textContent = step.label;
    text.textContent = step.text;
    backBtn.disabled = index === 0;

    const nextBtn = document.getElementById('tour-next');
    nextBtn.textContent = index === STEPS.length - 1 ? 'Finish' : 'Next';

    overlay.hidden = false;
    positionCard(rect);
  }, step.scrollTo ? 500 : 50);
}

function endTour() {
  active = false;
  document.getElementById('tour-overlay').hidden = true;
  localStorage.setItem('tourComplete', 'true');
}

export function initTour() {
  const btn = document.getElementById('btn-tour');
  if (!btn) return;

  // Hide tour button if already completed
  if (localStorage.getItem('tourComplete') === 'true') {
    btn.textContent = 'Retake the tour';
  }

  btn.addEventListener('click', () => {
    active = true;
    currentStep = 0;
    showStep(0);
  });

  document.getElementById('tour-next').addEventListener('click', () => {
    if (currentStep >= STEPS.length - 1) {
      endTour();
    } else {
      showStep(currentStep + 1);
    }
  });

  document.getElementById('tour-back').addEventListener('click', () => {
    if (currentStep > 0) showStep(currentStep - 1);
  });

  document.getElementById('tour-exit').addEventListener('click', endTour);

  // Close on Escape
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && active) endTour();
  });

  // Reposition on resize
  window.addEventListener('resize', () => {
    if (active) showStep(currentStep);
  });
}
