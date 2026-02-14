/**
 * Dual content maps for ELI5 mode.
 * t(key) returns plain-language text when eli5 is active, technical text otherwise.
 */

import { get } from './state.js';

const TECHNICAL = {
  // Architecture block descriptions
  'tok-embed.desc': 'Look up the learned 16-dimensional vector for this token',
  'pos-embed.desc': 'Add position information so the model knows token order',
  'rmsnorm0.desc': 'Normalize the vector magnitude to stabilize training',
  'rmsnorm1.desc': 'Normalize before attention to keep gradients stable',
  'attention.desc': '4 heads independently compute attention weights, then combine',
  'residual1.desc': 'Add the original input back, preserving earlier information',
  'rmsnorm2.desc': 'Normalize before the MLP feed-forward layer',
  'mlp.desc': 'Two linear layers with squared ReLU: expand to 64-dim, compress back',
  'residual2.desc': 'Add MLP input back for a second residual connection',
  'lm-head.desc': 'Project 16-dim hidden state to 27 logits, one per token',
  'softmax.desc': 'Convert raw logits into a probability distribution',

  // Architecture block titles
  'tok-embed.title': 'Token Embedding',
  'pos-embed.title': 'Position Embedding',
  'rmsnorm0.title': 'RMSNorm (initial)',
  'rmsnorm1.title': 'RMSNorm (pre-attention)',
  'attention.title': 'Multi-Head Attention',
  'residual1.title': 'Residual Connection 1',
  'rmsnorm2.title': 'RMSNorm (pre-MLP)',
  'mlp.title': 'MLP (ReLU\u00B2)',
  'residual2.title': 'Residual Connection 2',
  'lm-head.title': 'LM Head',
  'softmax.title': 'Softmax',

  // Architecture block tooltips
  'tok-embed.tooltip': 'Look up the learned vector representation for this token',
  'pos-embed.tooltip': 'Add position information so the model knows token order',
  'rmsnorm0.tooltip': 'Root Mean Square Normalization \u2014 stabilizes values before processing',
  'rmsnorm1.tooltip': 'Root Mean Square Normalization \u2014 stabilizes values before attention',
  'attention.tooltip': 'Each head learns different relationships between tokens, then results are combined',
  'residual1.tooltip': 'Skip connection \u2014 adds original input back to preserve earlier information',
  'rmsnorm2.tooltip': 'Root Mean Square Normalization \u2014 stabilizes values before the MLP',
  'mlp.tooltip': 'Feed-forward network with squared ReLU activation: expand to 64-dim, compress back to 16-dim',
  'residual2.tooltip': 'Skip connection \u2014 adds MLP input back for a second residual path',
  'lm-head.tooltip': 'Final linear layer projecting hidden state to one score per vocabulary token',
  'softmax.tooltip': 'Converts raw scores into a probability distribution that sums to 1',

  // Section headings
  'heading.forwardPass': 'Forward Pass',
  'heading.tokenProbs': 'Token Probabilities',
  'heading.attnExplorer': 'Attention Explorer',
  'heading.intermediateValues': 'Intermediate Values',
  'heading.weightInspector': 'Weight Inspector',
  'heading.trainingProgress': 'Training Progress',
  'heading.nameGenerator': 'Name Generator',

  // Inference intermediate labels
  'inter.tokEmb': 'Token Embedding',
  'inter.posEmb': 'Position Embedding',
  'inter.combined': 'Combined (tok + pos)',
  'inter.postNorm0': 'After initial RMSNorm',
  'inter.postNorm1': 'After pre-attention RMSNorm',
  'inter.q': 'Q projection',
  'inter.k': 'K projection',
  'inter.v': 'V projection',
  'inter.attnOut': 'Attention output',
  'inter.postResidual1': 'After residual 1',
  'inter.postNorm2': 'After pre-MLP RMSNorm',
  'inter.mlpHidden': 'MLP hidden (64-dim)',
  'inter.mlpActivated': 'After ReLU\u00B2',
  'inter.mlpOut': 'MLP output',
  'inter.postResidual2': 'Final hidden state',

  // Training weight labels
  'weight.wte': 'Token Embeddings (wte)',
  'weight.wpe': 'Position Embeddings (wpe)',
  'weight.attn_wq': 'Attention Wq',
  'weight.attn_wk': 'Attention Wk',
  'weight.attn_wv': 'Attention Wv',
  'weight.attn_wo': 'Attention Wo',
  'weight.lm_head': 'LM Head',

  // Tooltips
  'tooltip.temperature': 'Controls randomness: lower = more predictable, higher = more creative',
  'tooltip.logits': 'Raw scores before normalization \u2014 higher means the model thinks that token is more likely',
  'tooltip.bos': 'Beginning Of Sequence \u2014 a special token that signals the start of generation',
};

const ELI5 = {
  // Architecture block descriptions
  'tok-embed.desc': 'Find this letter\'s unique fingerprint in the model\'s dictionary',
  'pos-embed.desc': 'Stamp the position number onto the fingerprint so the model knows where this letter sits',
  'rmsnorm0.desc': 'Shrink or stretch the numbers so none get too big or too small',
  'rmsnorm1.desc': 'Re-balance the numbers before looking at which letters are related',
  'attention.desc': '4 little readers each look back at earlier letters to decide what matters, then pool their notes',
  'residual1.desc': 'Mix in the original signal so the model doesn\'t forget what it started with',
  'rmsnorm2.desc': 'Re-balance the numbers one more time before the thinking step',
  'mlp.desc': 'A two-step thinking layer: spread the information out wide, then squeeze it back down',
  'residual2.desc': 'Mix in the signal from before thinking, giving the model a second safety net',
  'lm-head.desc': 'Score every possible next letter \u2014 higher score means the model likes it more',
  'softmax.desc': 'Turn the scores into percentages that add up to 100%',

  // Architecture block titles
  'tok-embed.title': 'Letter Lookup',
  'pos-embed.title': 'Position Stamp',
  'rmsnorm0.title': 'Balancing (first)',
  'rmsnorm1.title': 'Balancing (before reading)',
  'attention.title': 'Reading Back',
  'residual1.title': 'Safety Net 1',
  'rmsnorm2.title': 'Balancing (before thinking)',
  'mlp.title': 'Thinking Layer',
  'residual2.title': 'Safety Net 2',
  'lm-head.title': 'Letter Scoring',
  'softmax.title': 'Percentages',

  // Architecture block tooltips
  'tok-embed.tooltip': 'Find this letter\'s unique fingerprint in the dictionary',
  'pos-embed.tooltip': 'Stamp the position number so the model knows where the letter sits',
  'rmsnorm0.tooltip': 'Re-balance the numbers so none are too big or too small',
  'rmsnorm1.tooltip': 'Re-balance before looking at which letters are related',
  'attention.tooltip': '4 little readers look back at earlier letters and pool their notes',
  'residual1.tooltip': 'Mix in the original signal so the model doesn\'t forget',
  'rmsnorm2.tooltip': 'Re-balance before the thinking step',
  'mlp.tooltip': 'Spread information out wide, think about it, then squeeze it back down',
  'residual2.tooltip': 'A second safety net \u2014 mix in what came before thinking',
  'lm-head.tooltip': 'Score every possible next letter',
  'softmax.tooltip': 'Turn scores into percentages that add up to 100%',

  // Section headings
  'heading.forwardPass': 'How the Model Reads',
  'heading.tokenProbs': 'Which Letter Next?',
  'heading.attnExplorer': 'What the Model Pays Attention To',
  'heading.intermediateValues': 'Numbers at Each Step',
  'heading.weightInspector': 'The Model\'s Learned Knowledge',
  'heading.trainingProgress': 'How the Model Learned',

  // Inference intermediate labels
  'inter.tokEmb': 'Letter fingerprint',
  'inter.posEmb': 'Position stamp',
  'inter.combined': 'Fingerprint + position combined',
  'inter.postNorm0': 'After first balancing',
  'inter.postNorm1': 'After balancing (before reading)',
  'inter.q': 'Question ("what am I looking for?")',
  'inter.k': 'Key ("what do I contain?")',
  'inter.v': 'Value ("what can I offer?")',
  'inter.attnOut': 'After reading back',
  'inter.postResidual1': 'After safety net 1',
  'inter.postNorm2': 'After balancing (before thinking)',
  'inter.mlpHidden': 'Spread out wide (64 slots)',
  'inter.mlpActivated': 'After throwing away negatives',
  'inter.mlpOut': 'After thinking',
  'inter.postResidual2': 'Final result for this letter',

  // Training weight labels
  'weight.wte': 'Letter Fingerprints',
  'weight.wpe': 'Position Stamps',
  'weight.attn_wq': 'Question Weights',
  'weight.attn_wk': 'Key Weights',
  'weight.attn_wv': 'Value Weights',
  'weight.attn_wo': 'Output Mixer',
  'weight.lm_head': 'Letter Scorer',

  // Tooltips
  'tooltip.temperature': 'Controls how creative the model is: low = safe guesses, high = surprising choices',
  'tooltip.logits': 'Raw scores for each letter \u2014 bigger means the model likes that letter more',
  'tooltip.bos': 'A special "start" signal that tells the model to begin generating',
};

export function t(key) {
  if (get('eli5') && key in ELI5) return ELI5[key];
  return TECHNICAL[key];
}
