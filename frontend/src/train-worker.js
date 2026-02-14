/**
 * Web Worker: runs the full microGPT training loop (500 steps) in-browser.
 * Posts messages: { type: 'step', step, loss, lr }
 *                 { type: 'checkpoint', step, names }
 *                 { type: 'weights', weights: Float64Array }
 *                 { type: 'done' }
 *
 * Expects message: { type: 'start', docs: string[] }
 *   docs = shuffled training documents (name strings)
 */

import { Value } from './value.js';

// --- Config ---
const N_EMBD = 16;
const N_HEAD = 4;
const N_LAYER = 1;
const BLOCK_SIZE = 8;
const HEAD_DIM = N_EMBD / N_HEAD;
const NUM_STEPS = 500;
const LEARNING_RATE = 1e-2;
const BETA1 = 0.9;
const BETA2 = 0.95;
const EPS_ADAM = 1e-8;
const CHECKPOINT_STEPS = [1, 10, 50, 100, 200, 300, 400, 500];

// --- Seeded RNG (matches Python random.seed(42) sequence) ---
// Using a simple mulberry32 PRNG for reproducibility
function mulberry32(seed) {
  return function() {
    seed |= 0; seed = seed + 0x6D2B79F5 | 0;
    let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

let rng;

function rngGauss(mean, std) {
  // Box-Muller transform
  const u1 = rng();
  const u2 = rng();
  const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  return mean + std * z;
}

function rngChoices(weights) {
  let total = 0;
  for (let i = 0; i < weights.length; i++) total += weights[i];
  let r = rng() * total;
  for (let i = 0; i < weights.length; i++) {
    r -= weights[i];
    if (r <= 0) return i;
  }
  return weights.length - 1;
}

// --- Model Functions (using Value autograd) ---
function matrix(nout, nin, std = 0.02) {
  const m = [];
  for (let r = 0; r < nout; r++) {
    const row = [];
    for (let c = 0; c < nin; c++) {
      row.push(new Value(rngGauss(0, std)));
    }
    m.push(row);
  }
  return m;
}

function linear(x, w) {
  return w.map(wo => {
    let s = wo[0].mul(x[0]);
    for (let i = 1; i < x.length; i++) {
      s = s.add(wo[i].mul(x[i]));
    }
    return s;
  });
}

function softmax(logits) {
  let maxVal = -Infinity;
  for (const v of logits) if (v.data > maxVal) maxVal = v.data;
  const exps = logits.map(v => v.sub(maxVal).exp());
  let total = exps[0];
  for (let i = 1; i < exps.length; i++) total = total.add(exps[i]);
  return exps.map(e => e.div(total));
}

function rmsnorm(x) {
  let ms = x[0].mul(x[0]);
  for (let i = 1; i < x.length; i++) ms = ms.add(x[i].mul(x[i]));
  ms = ms.mul(1 / x.length);
  const scale = ms.add(1e-5).pow(-0.5);
  return x.map(xi => xi.mul(scale));
}

function gpt(tokenId, posId, keys, values, stateDict) {
  const tokEmb = stateDict.wte[tokenId];
  const posEmb = stateDict.wpe[posId];
  let x = tokEmb.map((t, i) => t.add(posEmb[i]));
  x = rmsnorm(x);

  for (let li = 0; li < N_LAYER; li++) {
    const xResidual = x;
    x = rmsnorm(x);
    const q = linear(x, stateDict[`layer${li}.attn_wq`]);
    const k = linear(x, stateDict[`layer${li}.attn_wk`]);
    const v = linear(x, stateDict[`layer${li}.attn_wv`]);
    keys[li].push(k);
    values[li].push(v);

    const xAttn = [];
    for (let h = 0; h < N_HEAD; h++) {
      const hs = h * HEAD_DIM;
      const qH = q.slice(hs, hs + HEAD_DIM);
      const kH = keys[li].map(ki => ki.slice(hs, hs + HEAD_DIM));
      const vH = values[li].map(vi => vi.slice(hs, hs + HEAD_DIM));

      const attnLogits = kH.map(kht => {
        let dot = qH[0].mul(kht[0]);
        for (let j = 1; j < HEAD_DIM; j++) dot = dot.add(qH[j].mul(kht[j]));
        return dot.mul(1 / Math.sqrt(HEAD_DIM));
      });

      const attnWeights = softmax(attnLogits);

      for (let j = 0; j < HEAD_DIM; j++) {
        let val = attnWeights[0].mul(vH[0][j]);
        for (let t = 1; t < vH.length; t++) val = val.add(attnWeights[t].mul(vH[t][j]));
        xAttn.push(val);
      }
    }

    x = linear(xAttn, stateDict[`layer${li}.attn_wo`]);
    x = x.map((a, i) => a.add(xResidual[i]));

    const xResidual2 = x;
    x = rmsnorm(x);
    x = linear(x, stateDict[`layer${li}.mlp_fc1`]);
    x = x.map(xi => xi.relu().pow(2));
    x = linear(x, stateDict[`layer${li}.mlp_fc2`]);
    x = x.map((a, i) => a.add(xResidual2[i]));
  }

  return linear(x, stateDict.lm_head);
}

// --- Inference (plain numbers, no autograd) ---
function inferLinear(x, w) {
  return w.map(wo => {
    let s = 0;
    for (let i = 0; i < x.length; i++) s += wo[i] * x[i];
    return s;
  });
}

function inferSoftmax(logits) {
  let max = -Infinity;
  for (const v of logits) if (v > max) max = v;
  const exps = logits.map(v => Math.exp(v - max));
  const total = exps.reduce((a, b) => a + b, 0);
  return exps.map(e => e / total);
}

function inferRmsnorm(x) {
  let ms = 0;
  for (const xi of x) ms += xi * xi;
  ms /= x.length;
  const scale = 1 / Math.sqrt(ms + 1e-5);
  return x.map(xi => xi * scale);
}

function generateName(stateDict, uchars, vocabSize, temperature) {
  const BOS = vocabSize - 1;
  const keys = Array.from({ length: N_LAYER }, () => []);
  const vals = Array.from({ length: N_LAYER }, () => []);
  let tokenId = BOS;
  const sample = [];

  // Extract plain number weights
  const sd = {};
  for (const key of Object.keys(stateDict)) {
    sd[key] = stateDict[key].map(row => row.map(v => v.data));
  }

  for (let pos = 0; pos < BLOCK_SIZE; pos++) {
    const tokEmb = sd.wte[tokenId];
    const posEmb = sd.wpe[pos];
    let x = tokEmb.map((t, i) => t + posEmb[i]);
    x = inferRmsnorm(x);

    for (let li = 0; li < N_LAYER; li++) {
      const xRes = x;
      x = inferRmsnorm(x);
      const q = inferLinear(x, sd[`layer${li}.attn_wq`]);
      const k = inferLinear(x, sd[`layer${li}.attn_wk`]);
      const v = inferLinear(x, sd[`layer${li}.attn_wv`]);
      keys[li].push(k);
      vals[li].push(v);

      const xAttn = [];
      for (let h = 0; h < N_HEAD; h++) {
        const hs = h * HEAD_DIM;
        const qH = q.slice(hs, hs + HEAD_DIM);
        const kH = keys[li].map(ki => ki.slice(hs, hs + HEAD_DIM));
        const vH = vals[li].map(vi => vi.slice(hs, hs + HEAD_DIM));
        const al = kH.map(kht => {
          let dot = 0;
          for (let j = 0; j < HEAD_DIM; j++) dot += qH[j] * kht[j];
          return dot / Math.sqrt(HEAD_DIM);
        });
        const aw = inferSoftmax(al);
        for (let j = 0; j < HEAD_DIM; j++) {
          let val = 0;
          for (let t = 0; t < vH.length; t++) val += aw[t] * vH[t][j];
          xAttn.push(val);
        }
      }
      x = inferLinear(xAttn, sd[`layer${li}.attn_wo`]);
      x = x.map((a, i) => a + xRes[i]);

      const xRes2 = x;
      x = inferRmsnorm(x);
      x = inferLinear(x, sd[`layer${li}.mlp_fc1`]);
      x = x.map(xi => { const r = Math.max(0, xi); return r * r; });
      x = inferLinear(x, sd[`layer${li}.mlp_fc2`]);
      x = x.map((a, i) => a + xRes2[i]);
    }

    const logits = inferLinear(x, sd.lm_head);
    const scaled = logits.map(l => l / temperature);
    const probs = inferSoftmax(scaled);
    tokenId = rngChoices(probs);
    if (tokenId === BOS) break;
    sample.push(uchars[tokenId]);
  }

  return sample.join('');
}

// --- Weight export order ---
const WEIGHT_ORDER = [
  'wte', 'wpe', 'lm_head',
  'layer0.attn_wq', 'layer0.attn_wk', 'layer0.attn_wv', 'layer0.attn_wo',
  'layer0.mlp_fc1', 'layer0.mlp_fc2',
];

function exportWeights(stateDict) {
  const flat = [];
  for (const key of WEIGHT_ORDER) {
    for (const row of stateDict[key]) {
      for (const v of row) {
        flat.push(v.data);
      }
    }
  }
  return new Float64Array(flat);
}

// --- Main training loop ---
self.onmessage = function(e) {
  if (e.data.type !== 'start') return;

  const { docs } = e.data;
  rng = mulberry32(42);

  // Tokenizer
  const uchars = [...new Set(docs.join(''))].sort();
  const BOS = uchars.length;
  const vocabSize = uchars.length + 1;

  // Initialize parameters
  const stateDict = {
    wte: matrix(vocabSize, N_EMBD),
    wpe: matrix(BLOCK_SIZE, N_EMBD),
    lm_head: matrix(vocabSize, N_EMBD),
  };
  for (let i = 0; i < N_LAYER; i++) {
    stateDict[`layer${i}.attn_wq`] = matrix(N_EMBD, N_EMBD);
    stateDict[`layer${i}.attn_wk`] = matrix(N_EMBD, N_EMBD);
    stateDict[`layer${i}.attn_wv`] = matrix(N_EMBD, N_EMBD);
    stateDict[`layer${i}.attn_wo`] = matrix(N_EMBD, N_EMBD, 0);
    stateDict[`layer${i}.mlp_fc1`] = matrix(4 * N_EMBD, N_EMBD);
    stateDict[`layer${i}.mlp_fc2`] = matrix(N_EMBD, 4 * N_EMBD, 0);
  }

  const params = [];
  for (const key of WEIGHT_ORDER) {
    for (const row of stateDict[key]) {
      for (const v of row) params.push(v);
    }
  }

  // Adam buffers
  const m = new Float64Array(params.length);
  const v = new Float64Array(params.length);

  // Docs arrive pre-shuffled (same order as Python's random.shuffle with seed 42)
  for (let step = 0; step < NUM_STEPS; step++) {
    const doc = docs[step % docs.length];
    const tokens = [BOS, ...doc.split('').map(ch => uchars.indexOf(ch)), BOS];
    const n = Math.min(BLOCK_SIZE, tokens.length - 1);

    const keys = Array.from({ length: N_LAYER }, () => []);
    const vals = Array.from({ length: N_LAYER }, () => []);
    const losses = [];

    for (let pos = 0; pos < n; pos++) {
      const tokenId = tokens[pos];
      const targetId = tokens[pos + 1];
      const logits = gpt(tokenId, pos, keys, vals, stateDict);
      const probs = softmax(logits);
      const lossT = probs[targetId].log().neg();
      losses.push(lossT);
    }

    let loss = losses[0];
    for (let i = 1; i < losses.length; i++) loss = loss.add(losses[i]);
    loss = loss.mul(1 / n);

    loss.backward();

    // Adam update with cosine LR
    const lrT = LEARNING_RATE * 0.5 * (1 + Math.cos(Math.PI * step / NUM_STEPS));
    for (let i = 0; i < params.length; i++) {
      const p = params[i];
      m[i] = BETA1 * m[i] + (1 - BETA1) * p.grad;
      v[i] = BETA2 * v[i] + (1 - BETA2) * p.grad * p.grad;
      const mHat = m[i] / (1 - BETA1 ** (step + 1));
      const vHat = v[i] / (1 - BETA2 ** (step + 1));
      p.data -= lrT * mHat / (Math.sqrt(vHat) + EPS_ADAM);
      p.grad = 0;
    }

    const stepNum = step + 1;
    self.postMessage({ type: 'step', step: stepNum, loss: loss.data, lr: lrT });

    if (CHECKPOINT_STEPS.includes(stepNum)) {
      const names = [];
      for (let i = 0; i < 8; i++) {
        names.push(generateName(stateDict, uchars, vocabSize, 0.5));
      }
      self.postMessage({ type: 'checkpoint', step: stepNum, names });
      self.postMessage({ type: 'checkpoint-weights', step: stepNum, weights: exportWeights(stateDict) });
    }
  }

  // Export final weights
  self.postMessage({ type: 'weights', weights: exportWeights(stateDict) });
  self.postMessage({ type: 'done' });
};
