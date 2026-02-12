/**
 * Architecture section — live data viewer + full source panel.
 * Click any block → shows actual numeric values from a real forward pass.
 * Full microgpt.py source with line highlighting per block.
 */

import { gptForward, N_LAYER } from './gpt.js';

const BLOCKS = [
  { id: 'tok-embed', label: 'Token Embed', color: '#3b82f6', dimOut: '16-dim', interKey: 'tokEmb', lines: [109, 109] },
  { id: 'pos-embed', label: '+ Pos Embed', color: '#3b82f6', dimOut: '16-dim', interKey: 'combined', lines: [110, 111] },
  { id: 'rmsnorm0', label: 'RMSNorm', color: '#8b5cf6', dimOut: '16-dim', interKey: 'postNorm0', lines: [103, 106] },
  { id: 'rmsnorm1', label: 'RMSNorm', color: '#8b5cf6', dimOut: '16-dim', interKey: 'postNorm1', lines: [103, 106] },
  { id: 'attention', label: 'Multi-Head Attention', color: '#06b6d4', dimOut: '16-dim', interKey: 'attnOut', wide: true, lines: [118, 133] },
  { id: 'residual1', label: '+ Residual', color: '#64748b', dimOut: '16-dim', interKey: 'postResidual1', lines: [134, 134] },
  { id: 'rmsnorm2', label: 'RMSNorm', color: '#8b5cf6', dimOut: '16-dim', interKey: 'postNorm2', lines: [103, 106] },
  { id: 'mlp', label: 'MLP (ReLU²)', color: '#f97316', dimOut: '16-dim', interKey: 'mlpOut', wide: true, lines: [138, 140] },
  { id: 'residual2', label: '+ Residual', color: '#64748b', dimOut: '16-dim', interKey: 'postResidual2', lines: [141, 141] },
  { id: 'lm-head', label: 'LM Head', color: '#22c55e', dimOut: '27 logits', interKey: 'logits', lines: [143, 143] },
  { id: 'softmax', label: 'Softmax', color: '#22c55e', dimOut: '27 probs', interKey: null, lines: [97, 101] },
];

// Detailed intermediate data labels (shown when a block is clicked)
const BLOCK_DETAILS = {
  'tok-embed': { title: 'Token Embedding', keys: [{ key: 'tokEmb', label: 'wte[token_id]', dim: 16 }] },
  'pos-embed': { title: 'Position Embedding', keys: [{ key: 'posEmb', label: 'wpe[pos_id]', dim: 16 }, { key: 'combined', label: 'tok + pos', dim: 16 }] },
  'rmsnorm0': { title: 'RMSNorm (initial)', keys: [{ key: 'postNorm0', label: 'rmsnorm(x)', dim: 16 }] },
  'rmsnorm1': { title: 'RMSNorm (pre-attention)', keys: [{ key: 'postNorm1', label: 'rmsnorm(x)', dim: 16 }] },
  'attention': { title: 'Multi-Head Attention', keys: [
    { key: 'q', label: 'Q projection', dim: 16 },
    { key: 'k', label: 'K projection', dim: 16 },
    { key: 'v', label: 'V projection', dim: 16 },
    { key: 'attnOut', label: 'Attention output (after Wo)', dim: 16 },
  ]},
  'residual1': { title: 'Residual Connection 1', keys: [{ key: 'postResidual1', label: 'x + x_residual', dim: 16 }] },
  'rmsnorm2': { title: 'RMSNorm (pre-MLP)', keys: [{ key: 'postNorm2', label: 'rmsnorm(x)', dim: 16 }] },
  'mlp': { title: 'MLP (ReLU²)', keys: [
    { key: 'mlpHidden', label: 'fc1 output (pre-activation)', dim: 64 },
    { key: 'mlpActivated', label: 'After ReLU²', dim: 64 },
    { key: 'mlpOut', label: 'fc2 output', dim: 16 },
  ]},
  'residual2': { title: 'Residual Connection 2', keys: [{ key: 'postResidual2', label: 'x + x_residual', dim: 16 }] },
  'lm-head': { title: 'LM Head', keys: [{ key: 'logits', label: 'Linear → 27 logits', dim: 27 }] },
  'softmax': { title: 'Softmax', keys: [] },
};

const SVG_NS = 'http://www.w3.org/2000/svg';

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

// Syntax highlighting for Python
function highlightPython(code) {
  const lines = code.split('\n');
  return lines.map((line, i) => {
    let content = line
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');

    // Comments
    content = content.replace(/(#.*)$/gm, '<span class="comment">$1</span>');
    // Strings (triple-quoted handled by line-level, single/double quoted)
    content = content.replace(/(&quot;|&#x27;|"|')((?:(?!\1).)*?)\1/g, '<span class="string">$1$2$1</span>');
    content = content.replace(/(f)(&#x27;|"|')/g, '<span class="string">$1$2');
    // Keywords
    content = content.replace(/\b(def|class|return|for|in|if|not|else|import|from|lambda|and|or|is|as|with|while|break|elif)\b/g, '<span class="keyword">$1</span>');
    // Numbers
    content = content.replace(/\b(\d+\.?\d*(?:e[+-]?\d+)?)\b/g, '<span class="number">$1</span>');
    // Built-in functions
    content = content.replace(/\b(print|len|range|max|min|sum|set|sorted|zip|enumerate|open|float|int|isinstance)\b/g, '<span class="function">$1</span>');

    return { num: i + 1, content };
  });
}

// Render the source panel
function renderSourcePanel() {
  const panel = document.getElementById('source-code-panel');
  const highlighted = highlightPython(MICROGPT_SOURCE);
  panel.innerHTML = highlighted.map(({ num, content }) =>
    `<div class="line" data-line="${num}"><span class="line-num">${num}</span><span class="line-content">${content}</span></div>`
  ).join('');
}

// Highlight specific line range in source
function highlightSourceLines(startLine, endLine) {
  const panel = document.getElementById('source-code-panel');
  const lines = panel.querySelectorAll('.line');
  lines.forEach(line => line.classList.remove('highlighted'));

  for (let i = startLine - 1; i < endLine && i < lines.length; i++) {
    lines[i].classList.add('highlighted');
  }

  // Scroll to the highlighted lines
  const targetLine = lines[startLine - 1];
  if (targetLine) {
    targetLine.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }

  // Expand the source panel if collapsed
  const toggle = document.getElementById('source-toggle');
  const content = document.getElementById('source-panel-content');
  if (toggle.getAttribute('aria-expanded') === 'false') {
    toggle.setAttribute('aria-expanded', 'true');
    content.classList.add('open');
  }
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
  svg.setAttribute('height', totalH);
  svg.setAttribute('aria-label', 'GPT architecture flow diagram');
  svg.setAttribute('role', 'img');
  svg.style.maxWidth = `${totalW}px`;

  const centerX = totalW / 2 - 30;

  const blockPositions = [];
  BLOCKS.forEach((block, i) => {
    const y = padY + i * (blockH + gap);
    const w = block.wide ? blockWWide : blockW;
    const x = centerX - w / 2;
    blockPositions.push({ x, y, w, h: blockH, block });

    // Connection line
    if (i < BLOCKS.length - 1) {
      const line = document.createElementNS(SVG_NS, 'line');
      line.setAttribute('x1', centerX);
      line.setAttribute('y1', y + blockH);
      line.setAttribute('x2', centerX);
      line.setAttribute('y2', y + blockH + gap);
      line.setAttribute('stroke', '#334155');
      line.setAttribute('stroke-width', '2');
      line.setAttribute('stroke-dasharray', '4 3');
      svg.appendChild(line);
    }

    // Block group
    const g = document.createElementNS(SVG_NS, 'g');
    g.setAttribute('class', 'arch-block');
    g.setAttribute('data-block', block.id);
    g.setAttribute('role', 'button');
    g.setAttribute('tabindex', '0');
    g.setAttribute('aria-label', `${block.label}: ${block.dimOut}`);

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
    text.setAttribute('fill', '#f1f5f9');
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

  return { svg, blockPositions };
}

// Render a vector as colored cells
function renderVectorDisplay(values, label) {
  const maxAbs = Math.max(...values.map(Math.abs), 0.001);
  const html = [`<div class="vector-label">${label} [${values.length}]</div><div class="values-grid">`];
  for (const v of values) {
    const norm = v / maxAbs;
    const r = norm > 0 ? Math.round(59 + norm * 196) : Math.round(59);
    const g = norm > 0 ? Math.round(130 + norm * 125) : Math.round(68);
    const b = norm > 0 ? Math.round(246) : Math.round(68 + Math.abs(norm) * 178);
    const bg = `rgba(${r}, ${g}, ${b}, ${Math.abs(norm) * 0.4 + 0.1})`;
    html.push(`<div class="val-cell" style="background:${bg}" title="${v.toFixed(6)}">${v.toFixed(3)}</div>`);
  }
  html.push('</div>');
  return html.join('');
}

// Show data for a specific block
function showBlockData(blockId, intermediates) {
  const card = document.getElementById('arch-data-card');
  const title = document.getElementById('arch-data-title');
  const content = document.getElementById('arch-data-content');
  const details = BLOCK_DETAILS[blockId];

  if (!details || !intermediates) {
    card.style.display = 'none';
    return;
  }

  title.textContent = details.title;
  const html = [];

  for (const { key, label, dim } of details.keys) {
    const values = intermediates[key];
    if (values) {
      html.push(`<div class="data-viewer" style="margin-bottom:0.5rem">${renderVectorDisplay(values, label)}</div>`);
    }
  }

  // Show attention logits for attention block
  if (blockId === 'attention' && intermediates.attnLogits) {
    html.push('<div class="data-viewer" style="margin-bottom:0.5rem">');
    html.push('<h4>Attention logits (per head, pre-softmax)</h4>');
    intermediates.attnLogits.forEach((headLogits, h) => {
      html.push(`<div class="vector-label">Head ${h} [${headLogits.length}]</div>`);
      html.push('<div class="values-grid">');
      for (const v of headLogits) {
        html.push(`<div class="val-cell" title="${v.toFixed(6)}">${v.toFixed(3)}</div>`);
      }
      html.push('</div>');
    });
    html.push('</div>');
  }

  if (html.length === 0) {
    html.push('<p style="color:var(--text-dim);font-size:0.85rem">No intermediate data for this block. Run a forward pass first.</p>');
  }

  content.innerHTML = html.join('');
  card.style.display = 'block';
}

export function initArchitecture({ vocab }) {
  const container = document.getElementById('arch-svg-container');
  const tokenSelect = document.getElementById('arch-token-select');
  const posSelect = document.getElementById('arch-pos-select');
  const btnRun = document.getElementById('btn-run-forward');

  // Populate token selector
  const chars = vocab.chars;
  const options = chars.map((ch, i) => `<option value="${i}">${ch}</option>`).join('');
  tokenSelect.innerHTML = options + `<option value="${vocab.bos}">BOS</option>`;

  // Create SVG diagram
  const { svg, blockPositions } = createSVG();
  container.innerHTML = '';
  container.appendChild(svg);

  // Render source panel
  renderSourcePanel();

  // Source panel toggle
  const sourceToggle = document.getElementById('source-toggle');
  const sourceContent = document.getElementById('source-panel-content');
  const toggleHandler = () => {
    const expanded = sourceToggle.getAttribute('aria-expanded') === 'true';
    sourceToggle.setAttribute('aria-expanded', !expanded);
    sourceContent.classList.toggle('open');
  };
  sourceToggle.addEventListener('click', toggleHandler);
  sourceToggle.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggleHandler(); }
  });

  // State
  let currentIntermediates = null;
  let activeBlockId = null;

  // Run forward pass and populate all blocks with real data
  function runForwardPass() {
    const tokenId = parseInt(tokenSelect.value);
    const posId = parseInt(posSelect.value);
    const keys = Array.from({ length: N_LAYER }, () => []);
    const values = Array.from({ length: N_LAYER }, () => []);

    const result = gptForward(tokenId, posId, keys, values, { intermediates: true });
    currentIntermediates = result.intermediates;

    // Highlight blocks that have data
    svg.querySelectorAll('.arch-block').forEach(g => {
      const blockId = g.getAttribute('data-block');
      const block = BLOCKS.find(b => b.id === blockId);
      if (block && block.interKey && currentIntermediates[block.interKey]) {
        g.querySelector('rect').style.fill = block.color + '40';
      }
    });

    // If a block is selected, update its display
    if (activeBlockId) {
      showBlockData(activeBlockId, currentIntermediates);
    }
  }

  btnRun.addEventListener('click', runForwardPass);

  // Block click handlers
  svg.querySelectorAll('.arch-block').forEach(g => {
    const handler = () => {
      const blockId = g.getAttribute('data-block');
      activeBlockId = blockId;

      // Visual highlight
      svg.querySelectorAll('.arch-block').forEach(b => b.classList.remove('active'));
      g.classList.add('active');

      // Show data
      showBlockData(blockId, currentIntermediates);

      // Highlight source lines
      const block = BLOCKS.find(b => b.id === blockId);
      if (block && block.lines) {
        highlightSourceLines(block.lines[0], block.lines[1]);
      }
    };

    g.addEventListener('click', handler);
    g.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); handler(); }
    });
  });

  // Run initial forward pass with BOS at position 0
  tokenSelect.value = String(vocab.bos);
  runForwardPass();
}
