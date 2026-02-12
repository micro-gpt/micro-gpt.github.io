import { loadWeights } from './gpt.js';

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

async function initSection(name) {
  if (initialized[name]) return;
  initialized[name] = true;

  const d = await loadData();

  switch (name) {
    case 'architecture': {
      const { initArchitecture } = await import('./architecture.js');
      initArchitecture(d);
      break;
    }
    case 'training': {
      const { initTraining } = await import('./training.js');
      initTraining(d, () => {
        // Weights updated from training — reset architecture section for fresh data
        initialized.architecture = false;
      });
      break;
    }
    case 'inference': {
      const { initInference } = await import('./inference.js');
      initInference(d);
      break;
    }
  }
}

// Tab navigation
const tabs = document.querySelectorAll('.tab-btn');
const panels = document.querySelectorAll('.tab-panel');

function switchTab(tabId) {
  const sectionName = tabId.replace('tab-', '');

  tabs.forEach(tab => {
    const selected = tab.id === tabId;
    tab.setAttribute('aria-selected', selected);
  });

  panels.forEach(panel => {
    const visible = panel.id === `panel-${sectionName}`;
    panel.setAttribute('aria-hidden', !visible);
  });

  initSection(sectionName);
}

tabs.forEach(tab => {
  tab.addEventListener('click', () => switchTab(tab.id));
});

// Keyboard navigation for tabs
document.querySelector('.tab-nav').addEventListener('keydown', (e) => {
  const tabList = [...tabs];
  const current = tabList.findIndex(t => t.getAttribute('aria-selected') === 'true');

  let next;
  if (e.key === 'ArrowRight') next = (current + 1) % tabList.length;
  else if (e.key === 'ArrowLeft') next = (current - 1 + tabList.length) % tabList.length;
  else if (e.key === 'Home') next = 0;
  else if (e.key === 'End') next = tabList.length - 1;
  else return;

  e.preventDefault();
  tabList[next].focus();
  switchTab(tabList[next].id);
});

// Init first section
initSection('architecture');
