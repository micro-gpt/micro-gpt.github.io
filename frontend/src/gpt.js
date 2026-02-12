/**
 * JS port of microGPT forward pass — inference only, no autograd.
 * Uses plain number arrays. Returns logits + attention weights.
 * With { intermediates: true }, captures all intermediate values at each stage.
 */

const N_EMBD = 16;
const N_HEAD = 4;
const N_LAYER = 1;
const BLOCK_SIZE = 8;
const HEAD_DIM = N_EMBD / N_HEAD;

let stateDict = null;

function extractMatrix(flat, offset, rows, cols) {
  const mat = [];
  for (let r = 0; r < rows; r++) {
    mat.push(flat.slice(offset + r * cols, offset + (r + 1) * cols));
  }
  return { data: mat, size: rows * cols };
}

export function loadWeights(flatWeights, vocabSize) {
  let offset = 0;
  const sd = {};

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

export function getStateDict() {
  return stateDict;
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
 * Forward pass. Returns { logits, attentionWeights, intermediates? }.
 * When opts.intermediates is true, returns all intermediate values at each stage.
 */
export function gptForward(tokenId, posId, keys, values, opts) {
  const sd = stateDict;
  const capture = opts && opts.intermediates;
  const inter = capture ? {} : null;

  // Embeddings
  const tokEmb = Array.from(sd.wte[tokenId]);
  const posEmb = Array.from(sd.wpe[posId]);
  let x = tokEmb.map((t, i) => t + posEmb[i]);

  if (capture) {
    inter.tokEmb = tokEmb;
    inter.posEmb = posEmb;
    inter.combined = Array.from(x);
  }

  x = rmsnorm(x);
  if (capture) inter.postNorm0 = Array.from(x);

  const allAttnWeights = [];
  const allAttnLogits = [];

  for (let li = 0; li < N_LAYER; li++) {
    const xResidual1 = x;
    x = rmsnorm(x);
    if (capture) inter.postNorm1 = Array.from(x);

    const q = linear(x, sd[`layer${li}.attn_wq`]);
    const k = linear(x, sd[`layer${li}.attn_wk`]);
    const v = linear(x, sd[`layer${li}.attn_wv`]);
    keys[li].push(k);
    values[li].push(v);

    if (capture) {
      inter.q = Array.from(q);
      inter.k = Array.from(k);
      inter.v = Array.from(v);
    }

    const xAttn = [];
    const headAttnLogits = [];
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

      headAttnLogits.push(Array.from(attnLogits));
      const attnWeights = softmax(attnLogits);
      allAttnWeights.push(Array.from(attnWeights));

      for (let j = 0; j < HEAD_DIM; j++) {
        let val = 0;
        for (let t = 0; t < vH.length; t++) val += attnWeights[t] * vH[t][j];
        xAttn.push(val);
      }
    }

    if (capture) inter.attnLogits = headAttnLogits;
    allAttnLogits.push(headAttnLogits);

    x = linear(xAttn, sd[`layer${li}.attn_wo`]);
    if (capture) inter.attnOut = Array.from(x);

    x = x.map((a, i) => a + xResidual1[i]);
    if (capture) inter.postResidual1 = Array.from(x);

    // MLP
    const xResidual2 = x;
    x = rmsnorm(x);
    if (capture) inter.postNorm2 = Array.from(x);

    const mlpHidden = linear(x, sd[`layer${li}.mlp_fc1`]);
    if (capture) inter.mlpHidden = Array.from(mlpHidden);

    const mlpActivated = mlpHidden.map(xi => { const r = Math.max(0, xi); return r * r; });
    if (capture) inter.mlpActivated = Array.from(mlpActivated);

    x = linear(mlpActivated, sd[`layer${li}.mlp_fc2`]);
    if (capture) inter.mlpOut = Array.from(x);

    x = x.map((a, i) => a + xResidual2[i]);
    if (capture) inter.postResidual2 = Array.from(x);
  }

  const logits = linear(x, sd.lm_head);

  const result = { logits, attentionWeights: allAttnWeights };
  if (capture) {
    inter.logits = Array.from(logits);
    result.intermediates = inter;
  }
  return result;
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
