# microGPT Vision Document

## What This App Is

An **explorable explanation** — an interactive article that teaches how a GPT works by letting you explore a 199-line, 4064-parameter model running live in the browser. Not a dashboard, not a learning platform — a reactive document in Bret Victor's tradition.

**North star examples:** Transformer Explainer (Georgia Tech), TensorFlow Playground, Distill.pub articles, Jay Alammar's Illustrated Transformer.

**Two audiences, one app:** A global ELI5 toggle switches between plain-language analogies for learners and raw math/code for practitioners.

---

## Narrative Structure (replaces tabs)

Scrolling sections with a sticky mini-nav. Each section: brief narrative, interactive visualization, optional depth. The user scrolls through a story but can stop and play at any point.

```
[Sticky mini-nav: . Intro  . Architecture  . Training  . Generation]

Section 1: INTRO / HOOK
  "This model learned to generate names. Try it."
  [Generate button -> live output, immediate payoff]
  [Stats: 4,064 params, 199 lines, runs in your browser]

Section 2: ARCHITECTURE - "How it thinks"
  Scrollytelling: sticky diagram on left, narrative scrolls on right.
  As narrative advances, blocks light up one by one.
  User can also click blocks, use stepper, or free-explore.
  ELI5: "Each block transforms the input step by step, like an assembly line"
  Technical: Code snippets, data bars, intermediate values

Section 3: TRAINING - "How it learns"
  Loss curve with scrubber. Checkpoint samples.
  ELI5: "The model starts guessing randomly, then gradually gets better"
  Technical: Adam optimizer, gradient flow, weight evolution

Section 4: GENERATION - "Watch it think"
  Token-by-token generation with attention arcs.
  Linked back to architecture: clicking a generation step
  highlights which blocks processed it.
  ELI5: "At each step, the model considers all previous letters to predict the next one"
  Technical: Temperature, logits, softmax, sampling
```

---

## Core Design Principles

Drawn from research into explorable explanations, data visualization tools, developer tools, and Apple Design Award winners.

### 1. Linked Views Everywhere
Hover a token in the diagram -> highlight it in code, data, and probabilities simultaneously. Currently Architecture and Inference are completely disconnected. Everything should be connected:

```
Token selected in Architecture <-> Same token context in Generation
Training step scrubbed         <-> Weight heatmaps update
                               <-> Architecture uses those weights
                               <-> Generation uses those weights
Block clicked in Architecture  <-> Full source highlights
                               <-> Generation step highlights same block
Color encoding                 <-> Consistent across all sections
```

### 2. Start Small, Build Big
Teach embedding, attention, MLP in isolation before the full forward pass. The step-through explorer is a step toward this.

### 3. Details-on-Demand Across Abstraction Levels
High-level flow by default, drill into matrix operations when wanted. Never dump everything at once.

### 4. Consistent Semantic Color Encoding
Architecture diagram uses meaningful colors (blue=embedding, purple=norm, cyan=attention, orange=MLP, green=output). Use these colors consistently across ALL views. Add a legend.

### 5. Tooltips That Add Information
A bar tooltip shouldn't just say "0.34" — it should say which dimension, what it means.

### 6. Keyboard-First Navigation
Arrow keys to step through blocks, number keys to switch sections. Mouse adds detail; keyboard drives.

### 7. Animation for Causality, Not Decoration
Animate the forward pass to show data flowing through layers. Don't animate UI chrome.

### 8. Interaction Quality Over Quantity
Fewer controls that do exactly what you expect.

---

## ELI5 System

A global toggle that changes three layers:

| Layer | ELI5 Mode | Technical Mode |
|-------|-----------|----------------|
| Text | Plain language, analogies, no jargon | Formal descriptions, math notation, code references |
| Visualizations | Simplified (fewer bars, labeled axes, guided annotations) | Full detail (all dimensions, raw values, hover tooltips) |
| Code | Hidden or shown as "peek behind the curtain" | Integrated with each block, line-highlighted |

Implementation: dual content stored as a content map keyed by mode. Toggle swaps which content renders. Not two separate apps — one app, two lenses.

---

## Technical Architecture

### Current (isolated modules)
```
main.js -> lazy-loads independent modules
  |-- architecture.js (own state, own DOM)
  |-- training.js     (own state, own DOM)
  |-- inference.js    (own state, own DOM)
```

### Proposed (shared state)
```
main.js -> manages shared state + scroll sections
  |-- state.js        (shared reactive state store)
  |-- architecture.js (reads/writes shared state)
  |-- training.js     (reads/writes shared state)
  |-- inference.js    (reads/writes shared state)
  |-- eli5.js         (content maps for both modes)
```

The shared state store: simple pub/sub object holding selected token/position, current block, training step, ELI5 mode, model weights reference, current section.

### What Gets Preserved
- `gpt.js` — forward pass implementation, no changes
- SVG architecture diagram — evolves but same foundation
- Training web worker — good pattern
- Loss curve, attention arcs — good visualizations
- Design system (tokens, colors, typography) — mostly intact
- Python syntax highlighter — reuse as-is

### What Gets Rebuilt
- Tab navigation -> scroll sections with intersection observer + sticky mini-nav
- Three isolated modules -> modules that read/write shared state
- All descriptive text -> dual ELI5/Technical content
- Architecture right panel -> scrollytelling narrative panel
- Inference as isolated tab -> "Generation" section linked to architecture
- Empty/initial states -> purposeful defaults that teach

---

## Phased Roadmap

Each phase is shippable — the app works at the end of every phase.

| Phase | Focus | Outcome |
|-------|-------|---------|
| 0 | Vision alignment | ✅ This document. Agree on direction before code. |
| 1 | Shared state + scroll layout | ✅ Replaced tabs with scroll sections. Sticky mini-nav. IntersectionObserver drives active state. |
| 2 | Architecture scrollytelling | ✅ Sticky diagram on left, all 11 narrative blocks scroll on right. Scroll position drives SVG highlighting via IntersectionObserver. SVG/dot/nav clicks smooth-scroll to blocks. |
| 3 | Linked views | ✅ Token/position shared across sections. Training step affects weights everywhere. Consistent color encoding. |
| 4 | ELI5 system | ✅ Content maps, toggle, dual rendering. |
| 5 | Generation integration | ✅ Forward pass replay animation, context indicator, per-head attention weight summary in architecture. |
| 6 | Polish | ✅ Arrow key block nav, mobile nav gradient fade, color legend, typography hierarchy, fixed weight inspector affordance, accessibility improvements. |

---

## UX Audit Findings (current state)

### High Priority
- ~~**Step dots are 12px** — far below 44px touch target~~ — fixed: dots now 44px
- ~~**Back/Next buttons use btn-sm (36px)** — also below 44px~~ — fixed: min-height 44px
- ~~**Inference shows 54 empty probability bars before generation** — looks broken~~ — fixed: empty state message
- ~~**Mobile: tapping a diagram block updates detail panel below the fold**~~ — resolved: narrative blocks always visible, click scrolls to block
- **No context for jargon** (Forward Pass, BOS, Temperature, RMSNorm, Logits) — partially addressed via ELI5 toggle and tooltips

### Medium Priority
- ~~**"Run Forward Pass" gives no visual confirmation** it did anything~~ — fixed: shows "Computed ✓"
- ~~**Weight inspector heatmaps have cursor:pointer + "Click to inspect" title** but no click handler~~ — fixed: removed broken affordance
- **Training step slider doesn't affect weight heatmaps** — breaks time-travel metaphor — partially fixed: checkpoint weights load on slider change
- **Card borders at 8% opacity nearly invisible** — cards blend into background
- ~~**No tab overflow gradient fade** on mobile~~ — fixed: gradient fade on overflowing mini-nav
- ~~**Architecture-to-Inference disconnect**~~ — fixed: shared state + genStep replay animation

### Lower Priority
- ~~**h2 (1.1rem) vs h3 (0.95rem)** — barely perceptible hierarchy difference~~ — fixed: h2 bumped to 1.2rem
- ~~**Architecture detail descriptions at 0.85rem --text-muted** — feel secondary~~ — fixed: 0.9rem, text-secondary
- ~~**Data bar labels at 0.7rem** — very small for critical information~~ — fixed: 0.8rem, text-muted
- ~~**No color legend** for architecture diagram~~ — fixed: semantic color legend added
- ~~**Tab panel exit has no animation**~~ — N/A: tabs replaced with scroll sections

### What Works Well
- Design token system is comprehensive and consistent
- Architecture step-through with linked SVG + detail + source highlighting
- Training step slider linked to loss curve and checkpoint cards
- prefers-reduced-motion properly disables all animations
- Semantic color encoding in architecture (blue=embed, purple=norm, cyan=attention, orange=MLP, green=output)
- Loading skeletons for initial content
- ~~Architecture welcome state with clear call to action~~ — removed: scrollytelling always starts at block 0 with data
- Monospace font stack and tabular-nums on numeric displays
- Dark mode as primary canvas with gradient accents
