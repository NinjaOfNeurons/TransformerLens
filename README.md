# ⚡ Transformer Lens

An interactive GPT-2 visualizer that goes beyond next-token prediction. Built as a learning tool for understanding transformers from the inside out.

## What it shows

### ① Tokenization layer
See exactly how GPT-2 splits your text into subword tokens with their IDs. Most people don't realize `"unbelievable"` becomes `["un", "bel", "iev", "able"]`. This is the first thing the model actually sees.

### ② Token coloring — two modes
- **Surprise mode** (green → red): How surprised was the model to see each token? Green = expected, red = shocking. Low probability = high surprise.
- **Entropy mode** (blue intensity): How *uncertain* was the model at each position? High entropy = many equally likely options. Low entropy = model was confident. A completely different signal.

### ③ Top-10 candidates
At every token position, see the 10 tokens GPT-2 was considering and their probabilities. The actual token is highlighted with ✓. Click any position to expand.

## Phase roadmap

- [x] Phase 1 — Tokenization + Surprise + Entropy + Top-K (this)
- [ ] Phase 2 — Attention head viewer per token
- [ ] Phase 2 — Layer-by-layer residual stream PCA
- [ ] Phase 2 — Model size comparison (GPT-2 S/M/L/XL)
- [ ] Phase 3 — Prompt perturbation explorer (causal tracing lite)

## Running locally

```bash
pip install -r requirements.txt
python app.py
```

## Stack
- GPT-2 (124M) via HuggingFace Transformers
- Gradio 4.x
- PyTorch
- Deployed on HuggingFace Spaces (Docker)
