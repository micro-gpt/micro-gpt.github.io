import { loadWeights } from './gpt.js';
import { get, set, subscribe } from './state.js';

// Data cache
let data = null;

async function loadData() {
  if (data) return data;
  const [weights, trainingLog, checkpoints, vocab] = await Promise.all([
    fetch('/data/weights.json').then(r => r.json()),
    fetch('/data/training-log.json').then(r => r.json()),
    fetch('/data/checkpoints.json').then(r => r.json()),
    fetch('/data/vocab.json').then(r => r.json()),
  ]);

  loadWeights(weights, vocab.chars.length + 1);

  data = { weights, trainingLog, checkpoints, vocab };
  return data;
}

// Lazy section initializers
const initialized = {};
const sections = ['intro', 'architecture', 'training', 'generation'];

async function initSection(name) {
  if (initialized[name]) return;
  initialized[name] = true;

  const d = await loadData();

  switch (name) {
    case 'intro': {
      const { gptForward, softmax, sampleFrom, N_LAYER, BLOCK_SIZE } = await import('./gpt.js');
      const btn = document.getElementById('btn-intro-generate');
      const output = document.getElementById('intro-output');

      function generateName() {
        const { vocab } = d;
        const keys = Array.from({ length: N_LAYER }, () => []);
        const values = Array.from({ length: N_LAYER }, () => []);
        let tokenId = vocab.bos;
        const chars = [];

        for (let pos = 0; pos < BLOCK_SIZE; pos++) {
          const { logits } = gptForward(tokenId, pos, keys, values);
          const probs = softmax(logits.map(l => l / 0.5));
          tokenId = sampleFrom(probs);
          if (tokenId === vocab.bos) break;
          chars.push(vocab.chars[tokenId]);
        }

        output.textContent = chars.join('') || '(empty)';
      }

      btn.addEventListener('click', generateName);
      generateName();
      break;
    }
    case 'architecture': {
      const { initArchitecture } = await import('./architecture.js');
      initArchitecture(d);
      break;
    }
    case 'training': {
      const { initTraining } = await import('./training.js');
      initTraining(d, () => {
        initialized.architecture = false;
        set('weightsVersion', (get('weightsVersion') || 0) + 1);
      });
      break;
    }
    case 'generation': {
      const { initInference } = await import('./inference.js');
      initInference(d);
      break;
    }
  }
}

// --- Intersection Observers ---

const sectionEls = sections.map(name => document.getElementById(`section-${name}`));

// Lazy init: fire when section approaches viewport
const initObserver = new IntersectionObserver((entries) => {
  for (const entry of entries) {
    if (!entry.isIntersecting) continue;
    const name = entry.target.dataset.section;
    initObserver.unobserve(entry.target);
    initSection(name);
  }
}, { rootMargin: '200px', threshold: 0 });

sectionEls.forEach(el => initObserver.observe(el));

// Active section tracking
const navLinks = document.querySelectorAll('.mini-nav-link');
const trackThresholds = Array.from({ length: 11 }, (_, i) => i / 10);

const trackObserver = new IntersectionObserver((entries) => {
  for (const entry of entries) entry.target.__ratio = entry.intersectionRatio;
  // Find best
  let best = null;
  let bestRatio = -1;
  for (const el of sectionEls) {
    const ratio = el.__ratio ?? 0;
    if (ratio > bestRatio) { bestRatio = ratio; best = el; }
  }
  if (best) set('activeSection', best.dataset.section);
}, { threshold: trackThresholds });

sectionEls.forEach(el => trackObserver.observe(el));

// Nav highlighting
subscribe('activeSection', (name) => {
  navLinks.forEach(link => {
    const isActive = link.dataset.section === name;
    link.classList.toggle('active', isActive);
    link.setAttribute('aria-current', isActive);
  });
});

// Entrance animations (skip for reduced motion)
const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

if (!prefersReducedMotion) {
  const animObserver = new IntersectionObserver((entries) => {
    for (const entry of entries) {
      if (!entry.isIntersecting) continue;
      entry.target.classList.add('visible');
      animObserver.unobserve(entry.target);
    }
  }, { threshold: 0.05 });

  sectionEls.forEach(el => {
    if (!el.classList.contains('visible')) animObserver.observe(el);
  });
}

// Nav click handling
navLinks.forEach(link => {
  link.addEventListener('click', (e) => {
    e.preventDefault();
    const name = link.dataset.section;
    const target = document.getElementById(`section-${name}`);
    target.scrollIntoView({ behavior: prefersReducedMotion ? 'auto' : 'smooth' });
    initSection(name);
  });
});

// Keyboard: 1-4 jump to sections
document.addEventListener('keydown', (e) => {
  const tag = document.activeElement?.tagName;
  if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;

  const idx = parseInt(e.key) - 1;
  if (idx >= 0 && idx < sections.length) {
    const target = sectionEls[idx];
    target.scrollIntoView({ behavior: prefersReducedMotion ? 'auto' : 'smooth' });
    initSection(sections[idx]);
  }
});

// Training callback: re-init architecture when weights change
subscribe('weightsVersion', () => {
  const archEl = document.getElementById('section-architecture');
  // If architecture is near viewport, init immediately; otherwise re-observe
  const rect = archEl.getBoundingClientRect();
  const inView = rect.bottom > -200 && rect.top < window.innerHeight + 200;
  if (inView) {
    initSection('architecture');
  } else {
    initObserver.observe(archEl);
  }
});

// Init
set('activeSection', 'intro');
initSection('intro');
