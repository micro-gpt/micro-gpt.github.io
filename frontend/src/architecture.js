/**
 * Architecture section — scrollytelling block explorer.
 * Sticky SVG diagram on left, scrollable narrative blocks on right.
 * Scroll position drives SVG block highlighting via IntersectionObserver.
 */

import { gptForward, softmax, N_LAYER } from './gpt.js';
import { get, set, subscribe } from './state.js';
import { t } from './content.js';

const BLOCKS = [
  { id: 'tok-embed', label: 'Token Embed', color: '#5B8DEF', dimOut: '16-dim', interKey: 'tokEmb', lines: [109, 109] },
  { id: 'pos-embed', label: '+ Pos Embed', color: '#5B8DEF', dimOut: '16-dim', interKey: 'combined', lines: [110, 111] },
  { id: 'rmsnorm0', label: 'RMSNorm', color: '#9B7AEA', dimOut: '16-dim', interKey: 'postNorm0', lines: [103, 106] },
  { id: 'rmsnorm1', label: 'RMSNorm', color: '#9B7AEA', dimOut: '16-dim', interKey: 'postNorm1', lines: [103, 106] },
  { id: 'attention', label: 'Multi-Head Attention', color: '#22D3EE', dimOut: '16-dim', interKey: 'attnOut', wide: true, lines: [118, 133] },
  { id: 'residual1', label: '+ Residual', color: '#6B7585', dimOut: '16-dim', interKey: 'postResidual1', lines: [134, 134] },
  { id: 'rmsnorm2', label: 'RMSNorm', color: '#9B7AEA', dimOut: '16-dim', interKey: 'postNorm2', lines: [103, 106] },
  { id: 'mlp', label: 'MLP (ReLU²)', color: '#FB923C', dimOut: '16-dim', interKey: 'mlpOut', wide: true, lines: [138, 140] },
  { id: 'residual2', label: '+ Residual', color: '#6B7585', dimOut: '16-dim', interKey: 'postResidual2', lines: [141, 141] },
  { id: 'lm-head', label: 'LM Head', color: '#4ADE80', dimOut: '27 logits', interKey: 'logits', lines: [143, 143] },
  { id: 'softmax', label: 'Softmax', color: '#4ADE80', dimOut: '27 probs', interKey: 'probs', lines: [97, 101] },
];

// Detailed intermediate data labels per block
const BLOCK_DETAILS = {
  'tok-embed': { keys: [{ key: 'tokEmb', label: 'wte[token_id]', dim: 16 }] },
  'pos-embed': { keys: [{ key: 'posEmb', label: 'wpe[pos_id]', dim: 16 }, { key: 'combined', label: 'tok + pos', dim: 16 }] },
  'rmsnorm0': { keys: [{ key: 'postNorm0', label: 'rmsnorm(x)', dim: 16 }] },
  'rmsnorm1': { keys: [{ key: 'postNorm1', label: 'rmsnorm(x)', dim: 16 }] },
  'attention': { keys: [
    { key: 'q', label: 'Q projection', dim: 16 },
    { key: 'k', label: 'K projection', dim: 16 },
    { key: 'v', label: 'V projection', dim: 16 },
    { key: 'attnOut', label: 'Attention output (after Wo)', dim: 16 },
  ]},
  'residual1': { keys: [{ key: 'postResidual1', label: 'x + x_residual', dim: 16 }] },
  'rmsnorm2': { keys: [{ key: 'postNorm2', label: 'rmsnorm(x)', dim: 16 }] },
  'mlp': { keys: [
    { key: 'mlpHidden', label: 'fc1 output (pre-activation)', dim: 64 },
    { key: 'mlpActivated', label: 'After ReLU²', dim: 64 },
    { key: 'mlpOut', label: 'fc2 output', dim: 16 },
  ]},
  'residual2': { keys: [{ key: 'postResidual2', label: 'x + x_residual', dim: 16 }] },
  'lm-head': { keys: [{ key: 'logits', label: 'Linear → 27 logits', dim: 27 }] },
  'softmax': { keys: [{ key: 'probs', label: 'Probabilities', dim: 27 }] },
};

const SVG_NS = 'http://www.w3.org/2000/svg';
const HEAD_COLORS = ['var(--accent-blue)', 'var(--accent-purple)', 'var(--accent-green)', 'var(--accent-cyan)'];

// --- Full microgpt.py source (embedded) ---
const MICROGPT_SOURCE = `"""
The most atomic way to train and inference a GPT in pure, dependency-free Python.
This file is the complete algorithm.
Everything else is just efficiency.

@karpathy
"""

import os       # os.path.exists
import math     # math.log, math.exp
import random   # random.seed, random.choices, random.gauss, random.shuffle
random.seed(42) # Let there be order among chaos

# Let there be an input dataset \`docs\`: list[str] of documents (e.g. a dataset of names)
if not os.path.exists('input.txt'):
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')
docs = [l.strip() for l in open('input.txt').read().strip().split('\\n') if l.strip()] # list[str] of documents
random.shuffle(docs)
print(f"num docs: {len(docs)}")

# Let there be a Tokenizer to translate strings to discrete symbols and back
uchars = sorted(set(''.join(docs))) # unique characters in the dataset become token ids 0..n-1
BOS = len(uchars) # token id for the special Beginning of Sequence (BOS) token
vocab_size = len(uchars) + 1 # total number of unique tokens, +1 is for BOS
print(f"vocab size: {vocab_size}")

# Let there be an Autograd to apply the chain rule recursively across a computation graph
class Value:
    """Stores a single scalar value and its gradient, as a node in a computation graph."""

    def __init__(self, data, children=(), local_grads=()):
        self.data = data                # scalar value of this node calculated during forward pass
        self.grad = 0                   # derivative of the loss w.r.t. this node, calculated in backward pass
        self._children = children       # children of this node in the computation graph
        self._local_grads = local_grads # local derivative of this node w.r.t. its children

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other): return Value(self.data**other, (self,), (other * self.data**(other-1),))
    def log(self): return Value(math.log(self.data), (self,), (1/self.data,))
    def exp(self): return Value(math.exp(self.data), (self,), (math.exp(self.data),))
    def relu(self): return Value(max(0, self.data), (self,), (float(self.data > 0),))
    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad

# Initialize the parameters, to store the knowledge of the model.
n_embd = 16     # embedding dimension
n_head = 4      # number of attention heads
n_layer = 1     # number of layers
block_size = 8  # maximum sequence length
head_dim = n_embd // n_head # dimension of each head
matrix = lambda nout, nin, std=0.02: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]
state_dict = {'wte': matrix(vocab_size, n_embd), 'wpe': matrix(block_size, n_embd), 'lm_head': matrix(vocab_size, n_embd)}
for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd, std=0)
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd, std=0)
params = [p for mat in state_dict.values() for row in mat for p in row] # flatten params into a single list[Value]
print(f"num params: {len(params)}")

# Define the model architecture: a stateless function mapping token sequence and parameters to logits over what comes next.
# Follow GPT-2, blessed among the GPTs, with minor differences: layernorm -> rmsnorm, no biases, GeLU -> ReLU^2
def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]

def softmax(logits):
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]

def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]

def gpt(token_id, pos_id, keys, values):
    tok_emb = state_dict['wte'][token_id] # token embedding
    pos_emb = state_dict['wpe'][pos_id] # position embedding
    x = [t + p for t, p in zip(tok_emb, pos_emb)] # joint token and position embedding
    x = rmsnorm(x)

    for li in range(n_layer):
        # 1) Multi-head attention block
        x_residual = x
        x = rmsnorm(x)
        q = linear(x, state_dict[f'layer{li}.attn_wq'])
        k = linear(x, state_dict[f'layer{li}.attn_wk'])
        v = linear(x, state_dict[f'layer{li}.attn_wv'])
        keys[li].append(k)
        values[li].append(v)
        x_attn = []
        for h in range(n_head):
            hs = h * head_dim
            q_h = q[hs:hs+head_dim]
            k_h = [ki[hs:hs+head_dim] for ki in keys[li]]
            v_h = [vi[hs:hs+head_dim] for vi in values[li]]
            attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5 for t in range(len(k_h))]
            attn_weights = softmax(attn_logits)
            head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h))) for j in range(head_dim)]
            x_attn.extend(head_out)
        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
        x = [a + b for a, b in zip(x, x_residual)]
        # 2) MLP block
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
        x = [xi.relu() ** 2 for xi in x]
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_residual)]

    logits = linear(x, state_dict['lm_head'])
    return logits

# Let there be Adam, the blessed optimizer and its buffers
learning_rate, beta1, beta2, eps_adam = 1e-2, 0.9, 0.95, 1e-8
m = [0.0] * len(params) # first moment buffer
v = [0.0] * len(params) # second moment buffer

# Repeat in sequence
num_steps = 500 # number of training steps
for step in range(num_steps):

    # Take single document, tokenize it, surround it with BOS special token on both sides
    doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n = min(block_size, len(tokens) - 1)

    # Forward the token sequence through the model, building up the computation graph all the way to the loss.
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    losses = []
    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax(logits)
        loss_t = -probs[target_id].log()
        losses.append(loss_t)
    loss = (1 / n) * sum(losses) # final average loss over the document sequence. May yours be low.

    # Backward the loss, calculating the gradients with respect to all model parameters.
    loss.backward()

    # Adam optimizer update: update the model parameters based on the corresponding gradients.
    lr_t = learning_rate * 0.5 * (1 + math.cos(math.pi * step / num_steps)) # cosine learning rate decay
    for i, p in enumerate(params):
        m[i] = beta1 * m[i] + (1 - beta1) * p.grad
        v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
        m_hat = m[i] / (1 - beta1 ** (step + 1))
        v_hat = v[i] / (1 - beta2 ** (step + 1))
        p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
        p.grad = 0

    print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f}")

# Inference: may the model babble back to us
temperature = 0.5 # in (0, 1], control the "creativity" of generated text, low to high
print("\\n--- inference ---")
for sample_idx in range(20):
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    token_id = BOS
    sample = []
    for pos_id in range(block_size):
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax([l / temperature for l in logits])
        token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        if token_id == BOS:
            break
        sample.append(uchars[token_id])
    print(f"sample {sample_idx+1:2d}: {''.join(sample)}")`;

// Single-pass Python syntax highlighter (tokenizer approach avoids nested span bugs)
const PY_KEYWORDS = new Set(['def', 'class', 'return', 'for', 'in', 'if', 'not', 'else', 'import', 'from', 'lambda', 'and', 'or', 'is', 'as', 'with', 'while', 'break', 'elif']);
const PY_BUILTINS = new Set(['print', 'len', 'range', 'max', 'min', 'sum', 'set', 'sorted', 'zip', 'enumerate', 'open', 'float', 'int', 'isinstance']);

function escHtml(s) {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function highlightLine(line) {
  const out = [];
  let i = 0;

  while (i < line.length) {
    // Comment — consumes rest of line
    if (line[i] === '#') {
      out.push(`<span class="comment">${escHtml(line.slice(i))}</span>`);
      break;
    }

    // Triple-quoted string
    if (line.slice(i, i + 3) === '"""' || line.slice(i, i + 3) === "'''") {
      const q3 = line.slice(i, i + 3);
      let end = line.indexOf(q3, i + 3);
      if (end === -1) {
        out.push(`<span class="string">${escHtml(line.slice(i))}</span>`);
        break;
      }
      out.push(`<span class="string">${escHtml(line.slice(i, end + 3))}</span>`);
      i = end + 3;
      continue;
    }

    // f-string prefix
    if (line[i] === 'f' && i + 1 < line.length && (line[i + 1] === '"' || line[i + 1] === "'")) {
      const quote = line[i + 1];
      let j = i + 2;
      while (j < line.length && line[j] !== quote) {
        if (line[j] === '\\') j++;
        j++;
      }
      out.push(`<span class="string">${escHtml(line.slice(i, j + 1))}</span>`);
      i = j + 1;
      continue;
    }

    // Single/double-quoted string
    if (line[i] === '"' || line[i] === "'") {
      const quote = line[i];
      let j = i + 1;
      while (j < line.length && line[j] !== quote) {
        if (line[j] === '\\') j++;
        j++;
      }
      out.push(`<span class="string">${escHtml(line.slice(i, j + 1))}</span>`);
      i = j + 1;
      continue;
    }

    // Word (keyword, builtin, or plain identifier)
    if (/[a-zA-Z_]/.test(line[i])) {
      let j = i;
      while (j < line.length && /[a-zA-Z0-9_]/.test(line[j])) j++;
      const word = line.slice(i, j);
      if (PY_KEYWORDS.has(word)) {
        out.push(`<span class="keyword">${word}</span>`);
      } else if (PY_BUILTINS.has(word)) {
        out.push(`<span class="function">${word}</span>`);
      } else {
        out.push(escHtml(word));
      }
      i = j;
      continue;
    }

    // Number (digits, decimals, scientific notation)
    if (/[0-9]/.test(line[i])) {
      let j = i;
      while (j < line.length && /[0-9.eE]/.test(line[j])) j++;
      // Include sign after e/E
      if (j > i && /[eE]/.test(line[j - 1]) && j < line.length && /[+-]/.test(line[j])) j++;
      while (j < line.length && /[0-9]/.test(line[j])) j++;
      out.push(`<span class="number">${escHtml(line.slice(i, j))}</span>`);
      i = j;
      continue;
    }

    // Any other character
    out.push(escHtml(line[i]));
    i++;
  }

  return out.join('');
}

function highlightPython(code) {
  const lines = code.split('\n');
  // Track multi-line triple-quoted strings
  let inTriple = false;
  let tripleQuote = '';

  return lines.map((line, i) => {
    let content;

    if (inTriple) {
      const end = line.indexOf(tripleQuote);
      if (end === -1) {
        content = `<span class="string">${escHtml(line)}</span>`;
      } else {
        content = `<span class="string">${escHtml(line.slice(0, end + 3))}</span>${highlightLine(line.slice(end + 3))}`;
        inTriple = false;
      }
    } else {
      // Check if this line starts a multi-line triple-quoted string
      const dq = line.indexOf('"""');
      const sq = line.indexOf("'''");
      let tqPos = -1;
      if (dq >= 0 && (sq < 0 || dq <= sq)) { tqPos = dq; tripleQuote = '"""'; }
      else if (sq >= 0) { tqPos = sq; tripleQuote = "'''"; }

      if (tqPos >= 0) {
        const closePos = line.indexOf(tripleQuote, tqPos + 3);
        if (closePos === -1) {
          // Opens but doesn't close on this line
          content = highlightLine(line.slice(0, tqPos)) + `<span class="string">${escHtml(line.slice(tqPos))}</span>`;
          inTriple = true;
        } else {
          content = highlightLine(line);
        }
      } else {
        content = highlightLine(line);
      }
    }

    return { num: i + 1, content };
  });
}

// Render full source into the collapsible panel
function renderFullSource() {
  const panel = document.getElementById('arch-full-source-code');
  const highlighted = highlightPython(MICROGPT_SOURCE);
  panel.innerHTML = highlighted.map(({ num, content }) =>
    `<div class="line" data-line="${num}"><span class="line-num">${num}</span><span class="line-content">${content}</span></div>`
  ).join('');
}

// Highlight lines in the full source panel (when collapsible is open)
function highlightFullSourceLines(startLine, endLine) {
  const panel = document.getElementById('arch-full-source-code');
  const lines = panel.querySelectorAll('.line');
  lines.forEach(line => line.classList.remove('highlighted'));

  for (let i = startLine - 1; i < endLine && i < lines.length; i++) {
    lines[i].classList.add('highlighted');
  }

  const targetLine = lines[startLine - 1];
  if (targetLine) {
    targetLine.scrollIntoView({ behavior: 'auto', block: 'center' });
  }
}

// Render a code snippet (extracted lines from MICROGPT_SOURCE) into the detail panel
function renderCodeSnippet(startLine, endLine) {
  const allLines = MICROGPT_SOURCE.split('\n');
  const snippetLines = allLines.slice(startLine - 1, endLine);
  const highlighted = snippetLines.map((line, i) => {
    const num = startLine + i;
    return { num, content: highlightLine(line) };
  });
  return `<div class="code-panel arch-code-snippet">${highlighted.map(({ num, content }) =>
    `<div class="line"><span class="line-num">${num}</span><span class="line-content">${content}</span></div>`
  ).join('')}</div>`;
}

// Render data bars for a vector of values (blue positive, orange negative)
function renderDataBars(values, label) {
  const maxAbs = Math.max(...values.map(Math.abs), 0.001);
  const bars = values.map((v, i) => {
    const heightPct = (Math.abs(v) / maxAbs) * 100;
    const colorClass = v >= 0 ? 'positive' : 'negative';
    return `<div class="arch-bar ${colorClass}" data-index="${i}" title="[${i}] ${v.toFixed(6)}" style="height:${heightPct}%"></div>`;
  }).join('');
  return `<div class="arch-data-section"><span class="arch-data-label">${escHtml(label)} [${values.length}]</span><div class="arch-data-bars">${bars}</div></div>`;
}

// Render probability bars for softmax output (horizontal, sorted, top 10)
function renderProbBars(probs, vocab) {
  const chars = vocab.chars;
  const indexed = probs.map((p, i) => ({ p, i }));
  indexed.sort((a, b) => b.p - a.p);
  const top = indexed.slice(0, 10);
  const maxP = top[0].p;

  const rows = top.map(({ p, i }) => {
    const label = i === vocab.bos ? 'BOS' : escHtml(chars[i]);
    const widthPct = (p / maxP) * 100;
    return `<div class="arch-prob-row">
      <span class="arch-prob-token">${label}</span>
      <div class="arch-prob-track"><div class="arch-prob-fill" style="width:${widthPct}%"></div></div>
      <span class="arch-prob-value">${(p * 100).toFixed(1)}%</span>
    </div>`;
  }).join('');

  return `<div class="arch-data-section"><span class="arch-data-label">Top 10 token probabilities</span><div class="arch-prob-bars">${rows}</div></div>`;
}

// Render attention weight summary (per-head bars)
function renderAttnSummary(attnWeights, posId) {
  if (!attnWeights || attnWeights.length === 0) return '';
  const rows = attnWeights.map((weights, h) => {
    const segments = weights.map((w, p) => {
      const pct = (w * 100);
      const label = pct >= 10 ? `<span class="arch-attn-pct">${Math.round(pct)}%</span>` : '';
      return `<div class="arch-attn-weight" style="flex:${Math.max(w, 0.02)}" title="pos ${p}: ${pct.toFixed(1)}%"><div class="arch-attn-bar" style="background:${HEAD_COLORS[h]};opacity:${0.3 + w * 0.7}"></div>${label}</div>`;
    }).join('');
    return `<div class="arch-attn-row"><span class="arch-attn-head" style="color:${HEAD_COLORS[h]}">H${h + 1}</span><div class="arch-attn-weights">${segments}</div></div>`;
  }).join('');
  return `<div class="arch-data-section"><span class="arch-data-label">Attention weights per head (positions 0–${posId})</span>${rows}</div>`;
}

// Render all 11 narrative blocks into the narrative container
function renderNarrativeBlocks(intermediates, vocab, attnWeights) {
  const container = document.getElementById('arch-narrative-container');
  container.innerHTML = BLOCKS.map((block, i) => {
    const html = [];
    html.push(`<h3 class="arch-detail-title">${escHtml(t(block.id + '.title'))}</h3>`);
    html.push(`<p class="arch-detail-desc">${escHtml(t(block.id + '.desc'))}</p>`);
    html.push(renderCodeSnippet(block.lines[0], block.lines[1]));
    html.push(`<div class="arch-narrative-data">${renderBlockData(i, intermediates, vocab, attnWeights)}</div>`);
    return `<div class="arch-narrative" data-block-index="${i}"><div class="card" style="--block-color: ${block.color}">${html.join('')}</div></div>`;
  }).join('');
}

// Render data visualization HTML for a single block
function renderBlockData(blockIndex, intermediates, vocab, attnWeights) {
  if (!intermediates) return '';
  const block = BLOCKS[blockIndex];
  const details = BLOCK_DETAILS[block.id];
  if (block.id === 'softmax') {
    return intermediates.probs ? renderProbBars(intermediates.probs, vocab) : '';
  }
  let html = details.keys.map(({ key, label }) => {
    const values = intermediates[key];
    return values ? renderDataBars(values, label) : '';
  }).join('');
  if (block.id === 'attention' && attnWeights) {
    const posId = parseInt(document.getElementById('arch-pos-select').value);
    html += renderAttnSummary(attnWeights, posId);
  }
  return html;
}

// Update only the data sections within narrative blocks (after forward pass recompute)
function updateNarrativeData(intermediates, vocab, attnWeights) {
  const container = document.getElementById('arch-narrative-container');
  container.querySelectorAll('.arch-narrative').forEach((el, i) => {
    const dataEl = el.querySelector('.arch-narrative-data');
    if (dataEl) dataEl.innerHTML = renderBlockData(i, intermediates, vocab, attnWeights);
  });
}

// Render step dots navigation
function renderStepDots(container, currentIndex) {
  container.innerHTML = BLOCKS.map((block, i) => {
    const stateClass = i === currentIndex ? 'current' : i < currentIndex ? 'completed' : 'upcoming';
    return `<button class="arch-dot ${stateClass}" data-index="${i}" aria-label="${block.label} (step ${i + 1} of ${BLOCKS.length})" title="${block.label}" style="--dot-color:${block.color}"></button>`;
  }).join('');
}

// Update SVG block visual states
function updateBlockStates(svg, currentIndex) {
  svg.querySelectorAll('.arch-block').forEach((g, i) => {
    const block = BLOCKS[i];
    const rect = g.querySelector('rect');

    g.classList.remove('current', 'completed', 'upcoming');
    g.removeAttribute('aria-current');

    if (i === currentIndex) {
      g.classList.add('current');
      g.setAttribute('aria-current', 'step');
      rect.setAttribute('fill', block.color + '50');
      rect.setAttribute('stroke-width', '3');
      g.style.opacity = '1';
    } else if (i < currentIndex) {
      g.classList.add('completed');
      rect.setAttribute('fill', block.color + '35');
      rect.setAttribute('stroke-width', '1.5');
      g.style.opacity = '1';
    } else {
      g.classList.add('upcoming');
      rect.setAttribute('fill', block.color + '15');
      rect.setAttribute('stroke-width', '1.5');
      g.style.opacity = '0.5';
    }
  });
}

// Create architecture SVG
function createSVG() {
  const blockW = 200;
  const blockWWide = 260;
  const blockH = 44;
  const gap = 16;
  const padX = 60;
  const padY = 30;

  const totalH = BLOCKS.length * (blockH + gap) - gap + padY * 2;
  const totalW = blockWWide + padX * 2 + 120;

  const svg = document.createElementNS(SVG_NS, 'svg');
  svg.setAttribute('viewBox', `0 0 ${totalW} ${totalH}`);
  svg.setAttribute('width', '100%');
  svg.setAttribute('aria-label', 'GPT architecture flow diagram');
  svg.setAttribute('role', 'img');
  svg.style.maxWidth = `${totalW}px`;

  const centerX = totalW / 2 - 30;

  BLOCKS.forEach((block, i) => {
    const y = padY + i * (blockH + gap);
    const w = block.wide ? blockWWide : blockW;
    const x = centerX - w / 2;

    // Connection line
    if (i < BLOCKS.length - 1) {
      const line = document.createElementNS(SVG_NS, 'line');
      line.setAttribute('x1', centerX);
      line.setAttribute('y1', y + blockH);
      line.setAttribute('x2', centerX);
      line.setAttribute('y2', y + blockH + gap);
      line.setAttribute('stroke', '#2A3140');
      line.setAttribute('stroke-width', '2');
      line.setAttribute('stroke-dasharray', '4 3');
      svg.appendChild(line);
    }

    // Block group
    const g = document.createElementNS(SVG_NS, 'g');
    g.setAttribute('class', 'arch-block');
    g.setAttribute('data-block', block.id);
    g.setAttribute('data-index', i);
    g.setAttribute('role', 'button');
    g.setAttribute('tabindex', '0');
    g.setAttribute('aria-label', `${block.label}: ${block.dimOut}`);

    // SVG tooltip
    const titleEl = document.createElementNS(SVG_NS, 'title');
    titleEl.textContent = t(block.id + '.tooltip');
    g.appendChild(titleEl);

    const rect = document.createElementNS(SVG_NS, 'rect');
    rect.setAttribute('x', x);
    rect.setAttribute('y', y);
    rect.setAttribute('width', w);
    rect.setAttribute('height', blockH);
    rect.setAttribute('rx', '8');
    rect.setAttribute('fill', block.color + '20');
    rect.setAttribute('stroke', block.color);
    rect.setAttribute('stroke-width', '1.5');
    g.appendChild(rect);

    const text = document.createElementNS(SVG_NS, 'text');
    text.setAttribute('x', centerX);
    text.setAttribute('y', y + blockH / 2 + 1);
    text.setAttribute('text-anchor', 'middle');
    text.setAttribute('dominant-baseline', 'middle');
    text.setAttribute('fill', '#E8ECF1');
    text.setAttribute('font-size', '13');
    text.setAttribute('font-weight', '600');
    text.setAttribute('font-family', '-apple-system, BlinkMacSystemFont, sans-serif');
    text.textContent = block.label;
    g.appendChild(text);

    // Dimension label on right
    const dimText = document.createElementNS(SVG_NS, 'text');
    dimText.setAttribute('x', x + w + 12);
    dimText.setAttribute('y', y + blockH / 2 + 1);
    dimText.setAttribute('dominant-baseline', 'middle');
    dimText.setAttribute('class', 'dim-label');
    dimText.textContent = block.dimOut;
    g.appendChild(dimText);

    svg.appendChild(g);
  });

  return svg;
}

let cleanupSubs = [];

export function initArchitecture({ vocab }) {
  // Clean up previous subscriptions on re-init
  for (const unsub of cleanupSubs) unsub();
  cleanupSubs = [];

  const container = document.getElementById('arch-svg-container');
  const narrativeContainer = document.getElementById('arch-narrative-container');
  const tokenSelect = document.getElementById('arch-token-select');
  const posSelect = document.getElementById('arch-pos-select');
  const btnRun = document.getElementById('btn-run-forward');
  const dotsContainer = document.getElementById('arch-step-dots');
  const btnBack = document.getElementById('arch-btn-back');
  const btnNext = document.getElementById('arch-btn-next');

  // Populate token selector
  const chars = vocab.chars;
  const options = chars.map((ch, i) => `<option value="${i}">${ch}</option>`).join('');
  tokenSelect.innerHTML = options + `<option value="${vocab.bos}" title="Beginning Of Sequence — a special token that signals the start of generation">BOS</option>`;

  // Create SVG diagram
  const svg = createSVG();
  container.innerHTML = '';
  container.appendChild(svg);

  // Render full source (collapsible, initially collapsed)
  renderFullSource();

  // State
  let currentIntermediates = null;
  let currentAttnWeights = null;
  let currentIndex = 0;

  // Compute a forward pass and return intermediates
  function computeForwardPass() {
    const tokenId = parseInt(tokenSelect.value);
    const posId = parseInt(posSelect.value);
    const keys = Array.from({ length: N_LAYER }, () => []);
    const values = Array.from({ length: N_LAYER }, () => []);
    const result = gptForward(tokenId, posId, keys, values, { intermediates: true });
    currentIntermediates = result.intermediates;
    currentIntermediates.probs = Array.from(softmax(result.logits));
    currentAttnWeights = result.attentionWeights;
  }

  // Highlight a block by index (updates SVG, dots, source, active narrative)
  function highlightBlock(index) {
    currentIndex = index;
    set('currentBlock', index);

    // Update SVG visual states
    updateBlockStates(svg, currentIndex);

    // Update step dots
    renderStepDots(dotsContainer, currentIndex);
    attachDotListeners();

    // Update button states
    btnBack.disabled = currentIndex === 0;
    btnNext.disabled = currentIndex === BLOCKS.length - 1;

    // Update active narrative block
    narrativeContainer.querySelectorAll('.arch-narrative').forEach((el, i) => {
      el.classList.toggle('active', i === currentIndex);
    });

    // Update full source highlighting (if collapsible is open)
    const block = BLOCKS[currentIndex];
    const fullSourceContent = document.getElementById('arch-full-source-content');
    if (fullSourceContent.classList.contains('open')) {
      highlightFullSourceLines(block.lines[0], block.lines[1]);
    }
  }

  // Scroll to a narrative block and highlight it
  function scrollToBlock(index) {
    const narrativeEl = narrativeContainer.querySelector(`[data-block-index="${index}"]`);
    if (narrativeEl) narrativeEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
    highlightBlock(index);
  }

  function attachDotListeners() {
    dotsContainer.querySelectorAll('.arch-dot').forEach(dot => {
      dot.addEventListener('click', () => {
        scrollToBlock(parseInt(dot.dataset.index));
      });
    });
  }

  // Navigation — scroll to prev/next narrative block
  btnBack.addEventListener('click', () => {
    if (currentIndex > 0) scrollToBlock(currentIndex - 1);
  });

  btnNext.addEventListener('click', () => {
    if (currentIndex < BLOCKS.length - 1) scrollToBlock(currentIndex + 1);
  });

  // Run Forward Pass: recompute + update all narrative data with visual feedback
  btnRun.addEventListener('click', () => {
    computeForwardPass();
    updateNarrativeData(currentIntermediates, vocab, currentAttnWeights);

    // Flash data sections
    narrativeContainer.querySelectorAll('.arch-narrative-data').forEach(el => {
      el.classList.remove('flash');
      el.offsetWidth; // force reflow to re-trigger animation
      el.classList.add('flash');
    });

    // Button feedback
    const originalText = btnRun.textContent;
    btnRun.textContent = 'Computed ✓';
    btnRun.disabled = true;
    setTimeout(() => {
      btnRun.textContent = originalText;
      btnRun.disabled = false;
    }, 800);
  });

  // Flag to prevent self-triggering when architecture's own selects call set()
  let externalUpdate = false;

  // Context indicator element
  const genContext = document.getElementById('arch-gen-context');

  // Auto-update on input change (clear gen context on manual changes)
  tokenSelect.addEventListener('change', () => {
    if (!externalUpdate) {
      set('token', parseInt(tokenSelect.value));
      genContext.hidden = true;
    }
    computeForwardPass();
    updateNarrativeData(currentIntermediates, vocab, currentAttnWeights);
  });
  posSelect.addEventListener('change', () => {
    if (!externalUpdate) {
      set('position', parseInt(posSelect.value));
      genContext.hidden = true;
    }
    computeForwardPass();
    updateNarrativeData(currentIntermediates, vocab, currentAttnWeights);
  });

  // SVG block click → scroll to narrative
  svg.querySelectorAll('.arch-block').forEach(g => {
    const handler = () => scrollToBlock(parseInt(g.dataset.index));
    g.addEventListener('click', handler);
    g.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); handler(); }
    });
  });

  // Full source collapsible toggle
  const fullSourceToggle = document.getElementById('arch-full-source-toggle');
  const fullSourceContent = document.getElementById('arch-full-source-content');
  function toggleFullSource() {
    const isOpen = fullSourceContent.classList.toggle('open');
    fullSourceToggle.setAttribute('aria-expanded', isOpen);
    if (isOpen) {
      const block = BLOCKS[currentIndex];
      highlightFullSourceLines(block.lines[0], block.lines[1]);
    }
  }
  fullSourceToggle.addEventListener('click', toggleFullSource);
  fullSourceToggle.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggleFullSource(); }
  });

  // Subscribe to external token/position changes (e.g. from generation section)
  cleanupSubs.push(subscribe('token', (val) => {
    if (val == null) return;
    externalUpdate = true;
    tokenSelect.value = String(val);
    tokenSelect.dispatchEvent(new Event('change'));
    externalUpdate = false;
  }));
  cleanupSubs.push(subscribe('position', (val) => {
    if (val == null) return;
    externalUpdate = true;
    posSelect.value = String(val);
    posSelect.dispatchEvent(new Event('change'));
    externalUpdate = false;
  }));

  // Pick up current state values (may have been set before architecture init)
  const initToken = get('token');
  const initPos = get('position');
  const useToken = initToken != null ? initToken : vocab.bos;
  const usePos = initPos != null ? initPos : 0;

  // Run initial forward pass
  tokenSelect.value = String(useToken);
  set('token', useToken);
  set('position', usePos);
  set('currentBlock', 0);
  computeForwardPass();

  // Render all narrative blocks with data
  renderNarrativeBlocks(currentIntermediates, vocab, currentAttnWeights);

  // Set up scroll observer — highlights block when it enters top 40% of viewport
  let scrollObserver = null;
  function setupScrollObserver() {
    if (scrollObserver) scrollObserver.disconnect();
    scrollObserver = new IntersectionObserver((entries) => {
      for (const entry of entries) {
        if (entry.isIntersecting) {
          highlightBlock(parseInt(entry.target.dataset.blockIndex));
        }
      }
    }, { rootMargin: '0px 0px -60% 0px', threshold: 0 });
    narrativeContainer.querySelectorAll('.arch-narrative').forEach(el => {
      scrollObserver.observe(el);
    });
  }
  setupScrollObserver();

  // ELI5 mode: update tooltips in-place, re-render narrative blocks
  cleanupSubs.push(subscribe('eli5', () => {
    svg.querySelectorAll('.arch-block').forEach(g => {
      const titleEl = g.querySelector('title');
      if (titleEl) titleEl.textContent = t(g.dataset.block + '.tooltip');
    });
    renderNarrativeBlocks(currentIntermediates, vocab, currentAttnWeights);
    setupScrollObserver();
    highlightBlock(currentIndex);
  }));

  // Forward pass replay animation (triggered by genStep)
  let animating = false;

  async function animateForwardPass() {
    if (animating) return;
    animating = true;
    const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const stepDelay = reduceMotion ? 0 : 120;

    for (let i = 0; i < BLOCKS.length; i++) {
      highlightBlock(i);
      const el = narrativeContainer.querySelector(`[data-block-index="${i}"]`);
      if (el) el.scrollIntoView({ behavior: reduceMotion ? 'auto' : 'smooth', block: 'nearest' });
      if (stepDelay > 0 && i < BLOCKS.length - 1) {
        await new Promise(r => setTimeout(r, stepDelay));
      }
    }
    animating = false;
  }

  cleanupSubs.push(subscribe('genStep', (step) => {
    if (!step) return;
    genContext.textContent = `Viewing: token '${step.char}' at position ${step.pos}`;
    genContext.hidden = false;
    requestAnimationFrame(() => animateForwardPass());
  }));

  // Arrow key navigation for architecture blocks
  function handleArrowKeys(e) {
    if (get('activeSection') !== 'architecture') return;
    const tag = document.activeElement?.tagName;
    if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;
    if (e.key === 'ArrowLeft' && currentIndex > 0) {
      e.preventDefault();
      scrollToBlock(currentIndex - 1);
    } else if (e.key === 'ArrowRight' && currentIndex < BLOCKS.length - 1) {
      e.preventDefault();
      scrollToBlock(currentIndex + 1);
    }
  }
  document.addEventListener('keydown', handleArrowKeys);
  cleanupSubs.push(() => document.removeEventListener('keydown', handleArrowKeys));

  // Start at block 0
  highlightBlock(0);
}
