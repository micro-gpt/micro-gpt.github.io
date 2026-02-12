/**
 * Architecture flow diagram — vertical SVG showing the forward pass.
 * Blocks are clickable to show corresponding Python source.
 */

const BLOCKS = [
  { id: 'tok-embed', label: 'Token Embed', color: '#3b82f6', dimIn: 'token id', dimOut: '16-dim' },
  { id: 'pos-embed', label: '+ Pos Embed', color: '#3b82f6', dimIn: 'pos id', dimOut: '16-dim' },
  { id: 'rmsnorm1', label: 'RMSNorm', color: '#8b5cf6', dimIn: '16-dim', dimOut: '16-dim' },
  { id: 'attention', label: 'Multi-Head Attention', color: '#06b6d4', dimIn: '16-dim', dimOut: '16-dim', wide: true },
  { id: 'residual1', label: '+ Residual', color: '#64748b', dimIn: '16-dim', dimOut: '16-dim' },
  { id: 'rmsnorm2', label: 'RMSNorm', color: '#8b5cf6', dimIn: '16-dim', dimOut: '16-dim' },
  { id: 'mlp', label: 'MLP (ReLU²)', color: '#f97316', dimIn: '16-dim → 64-dim → 16-dim', dimOut: '16-dim', wide: true },
  { id: 'residual2', label: '+ Residual', color: '#64748b', dimIn: '16-dim', dimOut: '16-dim' },
  { id: 'lm-head', label: 'LM Head', color: '#22c55e', dimIn: '16-dim', dimOut: '27 logits' },
  { id: 'softmax', label: 'Softmax', color: '#22c55e', dimIn: '27 logits', dimOut: '27 probs' },
];

const CODE_SNIPPETS = {
  'tok-embed': {
    title: 'Token Embedding',
    code: `tok_emb = state_dict['wte'][token_id]  <span class="comment"># [vocab_size, 16] lookup</span>`,
  },
  'pos-embed': {
    title: 'Position Embedding',
    code: `pos_emb = state_dict['wpe'][pos_id]  <span class="comment"># [block_size, 16] lookup</span>
x = [t + p <span class="keyword">for</span> t, p <span class="keyword">in</span> <span class="function">zip</span>(tok_emb, pos_emb)]`,
  },
  'rmsnorm1': {
    title: 'RMSNorm',
    code: `<span class="keyword">def</span> <span class="function">rmsnorm</span>(x):
    ms = <span class="function">sum</span>(xi * xi <span class="keyword">for</span> xi <span class="keyword">in</span> x) / <span class="function">len</span>(x)
    scale = (ms + <span class="number">1e-5</span>) ** <span class="number">-0.5</span>
    <span class="keyword">return</span> [xi * scale <span class="keyword">for</span> xi <span class="keyword">in</span> x]`,
  },
  'rmsnorm2': {
    title: 'RMSNorm',
    code: `<span class="keyword">def</span> <span class="function">rmsnorm</span>(x):
    ms = <span class="function">sum</span>(xi * xi <span class="keyword">for</span> xi <span class="keyword">in</span> x) / <span class="function">len</span>(x)
    scale = (ms + <span class="number">1e-5</span>) ** <span class="number">-0.5</span>
    <span class="keyword">return</span> [xi * scale <span class="keyword">for</span> xi <span class="keyword">in</span> x]`,
  },
  'attention': {
    title: 'Multi-Head Attention (4 heads, head_dim=4)',
    code: `q = <span class="function">linear</span>(x, state_dict[<span class="string">'layer0.attn_wq'</span>])
k = <span class="function">linear</span>(x, state_dict[<span class="string">'layer0.attn_wk'</span>])
v = <span class="function">linear</span>(x, state_dict[<span class="string">'layer0.attn_wv'</span>])
keys[li].<span class="function">append</span>(k)
values[li].<span class="function">append</span>(v)

<span class="keyword">for</span> h <span class="keyword">in</span> <span class="function">range</span>(n_head):  <span class="comment"># 4 heads</span>
    q_h = q[hs:hs+head_dim]   <span class="comment"># slice 4-dim</span>
    attn_logits = [<span class="function">sum</span>(q_h[j] * k_h[t][j] ...) / head_dim**<span class="number">0.5</span> ...]
    attn_weights = <span class="function">softmax</span>(attn_logits)
    head_out = [<span class="function">sum</span>(attn_weights[t] * v_h[t][j] ...) ...]

x = <span class="function">linear</span>(x_attn, state_dict[<span class="string">'layer0.attn_wo'</span>])`,
  },
  'residual1': {
    title: 'Residual Connection',
    code: `x = [a + b <span class="keyword">for</span> a, b <span class="keyword">in</span> <span class="function">zip</span>(x, x_residual)]`,
  },
  'residual2': {
    title: 'Residual Connection',
    code: `x = [a + b <span class="keyword">for</span> a, b <span class="keyword">in</span> <span class="function">zip</span>(x, x_residual)]`,
  },
  'mlp': {
    title: 'MLP with ReLU² activation',
    code: `x = <span class="function">linear</span>(x, state_dict[<span class="string">'layer0.mlp_fc1'</span>])  <span class="comment"># 16 → 64</span>
x = [xi.<span class="function">relu</span>() ** <span class="number">2</span> <span class="keyword">for</span> xi <span class="keyword">in</span> x]           <span class="comment"># ReLU²</span>
x = <span class="function">linear</span>(x, state_dict[<span class="string">'layer0.mlp_fc2'</span>])  <span class="comment"># 64 → 16</span>`,
  },
  'lm-head': {
    title: 'Language Model Head',
    code: `logits = <span class="function">linear</span>(x, state_dict[<span class="string">'lm_head'</span>])  <span class="comment"># 16 → 27</span>`,
  },
  'softmax': {
    title: 'Softmax',
    code: `<span class="keyword">def</span> <span class="function">softmax</span>(logits):
    max_val = <span class="function">max</span>(val.data <span class="keyword">for</span> val <span class="keyword">in</span> logits)
    exps = [(val - max_val).<span class="function">exp</span>() <span class="keyword">for</span> val <span class="keyword">in</span> logits]
    total = <span class="function">sum</span>(exps)
    <span class="keyword">return</span> [e / total <span class="keyword">for</span> e <span class="keyword">in</span> exps]`,
  },
};

const SVG_NS = 'http://www.w3.org/2000/svg';

function createSVG() {
  const blockW = 200;
  const blockWWide = 260;
  const blockH = 44;
  const gap = 16;
  const padX = 60;
  const padY = 30;

  const totalH = BLOCKS.length * (blockH + gap) - gap + padY * 2;
  const totalW = blockWWide + padX * 2 + 120; // extra for dim labels

  const svg = document.createElementNS(SVG_NS, 'svg');
  svg.setAttribute('viewBox', `0 0 ${totalW} ${totalH}`);
  svg.setAttribute('width', '100%');
  svg.setAttribute('height', totalH);
  svg.setAttribute('aria-label', 'GPT architecture flow diagram');
  svg.setAttribute('role', 'img');
  svg.style.maxWidth = `${totalW}px`;

  const centerX = totalW / 2 - 30; // offset left for dimension labels on right

  // Draw connection lines + blocks
  const blockPositions = [];
  BLOCKS.forEach((block, i) => {
    const y = padY + i * (blockH + gap);
    const w = block.wide ? blockWWide : blockW;
    const x = centerX - w / 2;
    blockPositions.push({ x, y, w, h: blockH, block });

    // Connection line to next block
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

  // Glowing dot for animation (hidden initially)
  const dot = document.createElementNS(SVG_NS, 'circle');
  dot.setAttribute('r', '6');
  dot.setAttribute('fill', '#60a5fa');
  dot.setAttribute('filter', 'url(#glow)');
  dot.style.display = 'none';
  dot.id = 'anim-dot';

  // Glow filter
  const defs = document.createElementNS(SVG_NS, 'defs');
  const filter = document.createElementNS(SVG_NS, 'filter');
  filter.setAttribute('id', 'glow');
  filter.setAttribute('x', '-50%');
  filter.setAttribute('y', '-50%');
  filter.setAttribute('width', '200%');
  filter.setAttribute('height', '200%');
  const blur = document.createElementNS(SVG_NS, 'feGaussianBlur');
  blur.setAttribute('stdDeviation', '4');
  blur.setAttribute('result', 'blur');
  const merge = document.createElementNS(SVG_NS, 'feMerge');
  const m1 = document.createElementNS(SVG_NS, 'feMergeNode');
  m1.setAttribute('in', 'blur');
  const m2 = document.createElementNS(SVG_NS, 'feMergeNode');
  m2.setAttribute('in', 'SourceGraphic');
  merge.appendChild(m1);
  merge.appendChild(m2);
  filter.appendChild(blur);
  filter.appendChild(merge);
  defs.appendChild(filter);
  svg.appendChild(defs);
  svg.appendChild(dot);

  return { svg, blockPositions, dot };
}

function animateForwardPass(svg, blockPositions, dot) {
  const btn = document.getElementById('btn-animate');
  btn.disabled = true;

  const blocks = svg.querySelectorAll('.arch-block');
  blocks.forEach(b => {
    b.querySelector('rect').style.fill = '';
  });

  dot.style.display = 'block';
  const centerX = blockPositions[0].x + blockPositions[0].w / 2;
  let step = 0;

  // Check reduced motion preference
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const stepDelay = reduceMotion ? 10 : 300;

  function nextStep() {
    if (step > 0) {
      // Highlight previous block
      const prevBlock = blocks[step - 1];
      const prevColor = BLOCKS[step - 1].color;
      prevBlock.querySelector('rect').style.fill = prevColor + '40';
    }

    if (step >= blockPositions.length) {
      dot.style.display = 'none';
      btn.disabled = false;
      return;
    }

    const pos = blockPositions[step];
    dot.setAttribute('cx', centerX);
    dot.setAttribute('cy', pos.y + pos.h / 2);
    dot.setAttribute('fill', pos.block.color);

    step++;
    setTimeout(nextStep, stepDelay);
  }

  nextStep();
}

export function initArchitecture() {
  const container = document.getElementById('arch-svg-container');
  const codeCard = document.getElementById('code-card');
  const codeTitle = document.getElementById('code-title');
  const codeContent = document.getElementById('code-content');

  const { svg, blockPositions, dot } = createSVG();
  container.innerHTML = '';
  container.appendChild(svg);

  // Click handlers for blocks
  svg.querySelectorAll('.arch-block').forEach(g => {
    const handler = () => {
      const blockId = g.getAttribute('data-block');
      const snippet = CODE_SNIPPETS[blockId];
      if (!snippet) return;

      codeTitle.textContent = snippet.title;
      codeContent.innerHTML = snippet.code;
      codeCard.style.display = 'block';

      // Highlight active block
      svg.querySelectorAll('.arch-block rect').forEach(r => {
        r.style.strokeWidth = '';
      });
      g.querySelector('rect').style.strokeWidth = '3';
    };

    g.addEventListener('click', handler);
    g.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        handler();
      }
    });
  });

  // Animate button
  document.getElementById('btn-animate').addEventListener('click', () => {
    animateForwardPass(svg, blockPositions, dot);
  });
}
