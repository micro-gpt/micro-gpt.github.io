/**
 * JS port of microGPT forward pass — inference only, no autograd.
 * Uses plain number arrays. Returns logits + attention weights.
 */

const N_EMBD = 16;
const N_HEAD = 4;
const N_LAYER = 1;
const BLOCK_SIZE = 8;
const HEAD_DIM = N_EMBD / N_HEAD;

// Weight matrices extracted from flat array
let weights = null;
let stateDict = null;

function extractMatrix(flat, offset, rows, cols) {
  const mat = [];
  for (let r = 0; r < rows; r++) {
    mat.push(flat.slice(offset + r * cols, offset + (r + 1) * cols));
  }
  return { data: mat, size: rows * cols };
}

export function loadWeights(flatWeights, vocabSize) {
  weights = flatWeights;
  let offset = 0;
  const sd = {};

  // Must match weight_order in export_data.py
  const shapes = [
    ['wte', vocabSize, N_EMBD],
    ['wpe', BLOCK_SIZE, N_EMBD],
    ['lm_head', vocabSize, N_EMBD],
    ['layer0.attn_wq', N_EMBD, N_EMBD],
    ['layer0.attn_wk', N_EMBD, N_EMBD],
    ['layer0.attn_wv', N_EMBD, N_EMBD],
    ['layer0.attn_wo', N_EMBD, N_EMBD],
    ['layer0.mlp_fc1', 4 * N_EMBD, N_EMBD],
    ['layer0.mlp_fc2', N_EMBD, 4 * N_EMBD],
  ];

  for (const [name, rows, cols] of shapes) {
    const { data, size } = extractMatrix(flatWeights, offset, rows, cols);
    sd[name] = data;
    offset += size;
  }

  stateDict = sd;
}

function linear(x, w) {
  return w.map(wo => {
    let sum = 0;
    for (let i = 0; i < x.length; i++) sum += wo[i] * x[i];
    return sum;
  });
}

export function softmax(logits) {
  let maxVal = -Infinity;
  for (let i = 0; i < logits.length; i++) {
    if (logits[i] > maxVal) maxVal = logits[i];
  }
  const exps = new Float64Array(logits.length);
  let total = 0;
  for (let i = 0; i < logits.length; i++) {
    exps[i] = Math.exp(logits[i] - maxVal);
    total += exps[i];
  }
  const probs = new Float64Array(logits.length);
  for (let i = 0; i < logits.length; i++) {
    probs[i] = exps[i] / total;
  }
  return probs;
}

function rmsnorm(x) {
  let ms = 0;
  for (let i = 0; i < x.length; i++) ms += x[i] * x[i];
  ms /= x.length;
  const scale = 1 / Math.sqrt(ms + 1e-5);
  return x.map(xi => xi * scale);
}

/**
 * Forward pass. Returns { logits, attentionWeights }.
 * attentionWeights[head] = Float64Array of attention weights for this step.
 * keys/values are KV caches — pass in [] for each layer to start fresh.
 */
export function gptForward(tokenId, posId, keys, values) {
  const sd = stateDict;

  // Embeddings
  const tokEmb = sd.wte[tokenId];
  const posEmb = sd.wpe[posId];
  let x = tokEmb.map((t, i) => t + posEmb[i]);
  x = rmsnorm(x);

  const allAttnWeights = [];

  for (let li = 0; li < N_LAYER; li++) {
    const xResidual1 = x;
    x = rmsnorm(x);
    const q = linear(x, sd[`layer${li}.attn_wq`]);
    const k = linear(x, sd[`layer${li}.attn_wk`]);
    const v = linear(x, sd[`layer${li}.attn_wv`]);
    keys[li].push(k);
    values[li].push(v);

    const xAttn = [];
    for (let h = 0; h < N_HEAD; h++) {
      const hs = h * HEAD_DIM;
      const qH = q.slice(hs, hs + HEAD_DIM);
      const kH = keys[li].map(ki => ki.slice(hs, hs + HEAD_DIM));
      const vH = values[li].map(vi => vi.slice(hs, hs + HEAD_DIM));

      const attnLogits = kH.map(kht => {
        let dot = 0;
        for (let j = 0; j < HEAD_DIM; j++) dot += qH[j] * kht[j];
        return dot / Math.sqrt(HEAD_DIM);
      });

      const attnWeights = softmax(attnLogits);
      allAttnWeights.push(Array.from(attnWeights));

      for (let j = 0; j < HEAD_DIM; j++) {
        let val = 0;
        for (let t = 0; t < vH.length; t++) val += attnWeights[t] * vH[t][j];
        xAttn.push(val);
      }
    }

    x = linear(xAttn, sd[`layer${li}.attn_wo`]);
    x = x.map((a, i) => a + xResidual1[i]);

    // MLP
    const xResidual2 = x;
    x = rmsnorm(x);
    x = linear(x, sd[`layer${li}.mlp_fc1`]);
    x = x.map(xi => { const r = Math.max(0, xi); return r * r; }); // ReLU²
    x = linear(x, sd[`layer${li}.mlp_fc2`]);
    x = x.map((a, i) => a + xResidual2[i]);
  }

  const logits = linear(x, sd.lm_head);
  return { logits, attentionWeights: allAttnWeights };
}

/**
 * Sample from probability distribution (weighted random choice).
 */
export function sampleFrom(probs) {
  let r = Math.random();
  for (let i = 0; i < probs.length; i++) {
    r -= probs[i];
    if (r <= 0) return i;
  }
  return probs.length - 1;
}

export { N_EMBD, N_HEAD, N_LAYER, BLOCK_SIZE, HEAD_DIM };
