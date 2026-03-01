# ⚡ Transformer Lens

<img width="1312" height="691" alt="image" src="https://github.com/user-attachments/assets/8b211f67-51b1-4f2b-941b-ad34bc79290d" />


Built this from scratch to understand how GPT-2 thinks — token by token.

## What it does
Type any text and see:
- **Tokenization** — how GPT-2 actually splits your words into subword tokens
- **Surprise coloring** — green means expected, red means the model was shocked
- **Probability** — exact likelihood the model assigned to each token given prior context

## What I learned building this

**Probability ≠ Entropy.**
`of` had higher probability than `Paris` (16% vs 3%) but `Paris` had higher entropy. That means the model was more focused when predicting `of`, but at the `Paris` position it had many equally reasonable options spread around — other cities, descriptions, articles. Two completely different signals.

**Capitalization changes token identity.**
`paris` (lowercase) splits into two tokens: `par` + `is`. `Paris` is one token. The model reasons about them through completely different representational pathways.

**GPT-2 is not a knowledge base.**
After `"The capital of France is"`, the top prediction is `the` (8.46%), not `Paris` (3.22% at #5). It's pattern matching from training distribution, not factual lookup.

## Stack
- GPT-2 (124M) via HuggingFace Transformers
- PyTorch
- Gradio

## Run locally
```bash
pip install gradio transformers torch
python app.py
```

## What's next
- Top-K candidates panel
- Entropy color mode
- Model comparison (GPT-2 small → medium → large)
