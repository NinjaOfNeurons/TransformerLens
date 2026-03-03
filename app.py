import gradio as gr
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── Load model once ──────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained("gpt2-xl")
model = AutoModelForCausalLM.from_pretrained("gpt2-xl")
model.eval()

# ── Helpers ──────────────────────────────────────────────────────────────────

def surprise_to_color(prob: float) -> str:
    """Green (expected) → Yellow → Red (surprising). prob in [0,1]."""
    # low prob = high surprise = red
    r = int(255 * (1 - prob))
    g = int(255 * prob)
    b = 60
    return f"rgb({r},{g},{b})"

def entropy_color(ent: float, max_ent: float) -> str:
    """Blue tones: bright = high entropy (uncertain), dim = low (confident)."""
    ratio = ent / max_ent if max_ent > 0 else 0
    v = int(80 + 175 * ratio)
    return f"rgb(30,{v},{255})"

def compute_token_stats(text: str):
    """
    Returns per-token stats:
      - token_str, token_id
      - prob_of_actual  (how likely was THIS token given prior context)
      - entropy         (how uncertain was the model at this position)
      - top_k           (list of (token_str, prob) for top 5 candidates)
    """
    if not text.strip():
        return []

    ids = tokenizer.encode(text, return_tensors="pt")  # (1, T)
    T = ids.shape[1]

    results = []

    with torch.no_grad():
        outputs = model(ids)
        logits = outputs.logits[0]  # (T, vocab)

    for i in range(T):
        tok_id = ids[0][i].item()
        tok_str = tokenizer.decode([tok_id])

        if i == 0:
            # No prior context for first token
            results.append({
                "token_str": tok_str,
                "token_id": tok_id,
                "prob": None,
                "entropy": None,
                "top_k": []
            })
            continue

        # Logits at position i-1 predict position i
        logit_prev = logits[i - 1]
        probs = F.softmax(logit_prev, dim=-1)

        prob_actual = probs[tok_id].item()
        ent = -torch.sum(probs * torch.log(probs + 1e-10)).item()

        top5 = torch.topk(probs, 10)
        top_k = [
            (tokenizer.decode([top5.indices[j].item()]), top5.values[j].item())
            for j in range(10)
        ]

        results.append({
            "token_str": tok_str,
            "token_id": tok_id,
            "prob": prob_actual,
            "entropy": ent,
            "top_k": top_k
        })

    return results


def build_token_html(stats, view_mode):
    """Build the colored token strip HTML."""
    if not stats:
        return ""

    max_ent = max((s["entropy"] for s in stats if s["entropy"] is not None), default=1)

    html = "<div style='display:flex;flex-wrap:wrap;gap:4px;align-items:flex-end;margin:12px 0;'>"
    for s in stats:
        tok = s["token_str"].replace("<", "&lt;").replace(">", "&gt;")

        if s["prob"] is None:
            bg = "#444"
            label = "start"
        elif view_mode == "Surprise":
            bg = surprise_to_color(s["prob"])
            label = f"{s['prob']:.1%}"
        else:  # Entropy
            bg = entropy_color(s["entropy"], max_ent)
            label = f"H={s['entropy']:.2f}"

        html += f"""
        <div style='
            display:inline-flex;flex-direction:column;align-items:center;
            background:{bg};
            border-radius:6px;padding:6px 10px 4px;
            font-family:"Courier New",monospace;
            cursor:default;
            transition:transform 0.15s;
        ' title='{label}'>
            <span style='color:#fff;font-size:1em;font-weight:600;white-space:pre;'>{tok if tok.strip() else "·"}</span>
            <span style='color:rgba(255,255,255,0.75);font-size:0.62em;margin-top:3px;'>{label}</span>
        </div>"""

    html += "</div>"
    return html


def build_topk_html(stats):
    """Build the top-K candidates table for each token position."""
    if not stats:
        return ""

    html = "<div style='font-family:monospace;'>"
    for i, s in enumerate(stats):
        if not s["top_k"]:
            continue
        tok = s["token_str"].replace("<", "&lt;")
        html += f"<details style='margin:6px 0;border:1px solid #333;border-radius:6px;padding:4px 10px;background:#111;'>"
        html += f"<summary style='cursor:pointer;color:#eee;font-size:0.9em;'>Position {i}: <b style='color:#7ef'>\"{tok}\"</b></summary>"
        html += "<table style='width:100%;margin-top:8px;border-collapse:collapse;font-size:0.85em;'>"
        html += "<tr style='color:#aaa;'><th style='text-align:left;padding:2px 8px;'>Rank</th><th style='text-align:left;padding:2px 8px;'>Token</th><th style='text-align:left;padding:2px 8px;'>Prob</th><th style='padding:2px 8px;'></th></tr>"
        for rank, (t, p) in enumerate(s["top_k"]):
            t_disp = t.replace("<", "&lt;")
            bar_w = int(p * 200)
            color = "#4ade80" if rank == 0 else ("#facc15" if rank < 3 else "#f87171")
            is_actual = (t == s["token_str"])
            row_bg = "background:#1a2a1a;" if is_actual else ""
            html += f"<tr style='{row_bg}'>"
            html += f"<td style='padding:3px 8px;color:#888;'>#{rank+1}</td>"
            html += f"<td style='padding:3px 8px;color:#fff;font-weight:{'700' if is_actual else '400'};'>{t_disp}{'  ✓' if is_actual else ''}</td>"
            html += f"<td style='padding:3px 8px;color:{color};'>{p:.2%}</td>"
            html += f"<td style='padding:3px 8px;'><div style='height:8px;width:{bar_w}px;background:{color};border-radius:4px;opacity:0.7;'></div></td>"
            html += "</tr>"
        html += "</table></details>"
    html += "</div>"
    return html


def build_tokenization_html(stats):
    """Show raw tokenization: token → ID mapping."""
    if not stats:
        return ""

    html = "<div style='font-family:monospace;display:flex;flex-wrap:wrap;gap:6px;margin:10px 0;'>"
    colors = ["#6366f1","#ec4899","#14b8a6","#f59e0b","#84cc16","#06b6d4","#a78bfa","#fb923c"]
    for i, s in enumerate(stats):
        tok = s["token_str"].replace("<", "&lt;").replace(">","&gt;")
        c = colors[i % len(colors)]
        html += f"""
        <div style='border:2px solid {c};border-radius:6px;padding:5px 10px;text-align:center;background:#0a0a0a;'>
            <div style='color:{c};font-size:0.72em;margin-bottom:3px;opacity:0.8;'>id:{s["token_id"]}</div>
            <div style='color:#fff;font-size:0.95em;font-weight:600;white-space:pre;'>{tok if tok.strip() else "⎵"}</div>
        </div>"""
    html += "</div>"
    return html


def build_entropy_summary(stats):
    """Mini summary bar of entropy across the sequence."""
    filtered = [s for s in stats if s["entropy"] is not None]
    if not filtered:
        return ""
    max_ent = max(s["entropy"] for s in filtered)
    avg_ent = np.mean([s["entropy"] for s in filtered])

    html = f"""
    <div style='font-family:monospace;font-size:0.85em;color:#aaa;padding:8px 0;'>
        <span style='margin-right:16px;'>📊 Avg entropy: <b style='color:#7ef'>{avg_ent:.3f}</b></span>
        <span style='margin-right:16px;'>📈 Max entropy: <b style='color:#f87'>{max_ent:.3f}</b></span>
        <span>🔢 Tokens: <b style='color:#afa'>{len(stats)}</b></span>
    </div>"""
    return html


# ── Main inference function ──────────────────────────────────────────────────

def analyze(text, view_mode):
    if not text.strip():
        return "", "", "", ""

    stats = compute_token_stats(text)

    token_html = build_token_html(stats, view_mode)
    tokenization_html = build_tokenization_html(stats)
    topk_html = build_topk_html(stats)
    summary_html = build_entropy_summary(stats)

    return token_html, tokenization_html, topk_html, summary_html


# ── UI ────────────────────────────────────────────────────────────────────────

CSS = """
body, .gradio-container { background: #080808 !important; }
.gradio-container { font-family: 'Courier New', monospace !important; }
#title { text-align:center; padding: 24px 0 8px; }
#title h1 { font-size: 2em; color:#fff; letter-spacing:0.05em; }
#title p  { color:#666; font-size:0.85em; margin-top:4px; }
.section-label { color:#555; font-size:0.75em; text-transform:uppercase; letter-spacing:0.1em; margin: 16px 0 4px; }
textarea, input { background:#111 !important; color:#eee !important; border:1px solid #333 !important; }
.gr-button { background:#1a1a2e !important; color:#7ef !important; border:1px solid #333 !important; }
.gr-button:hover { background:#2a2a4e !important; }
details summary::-webkit-details-marker { color: #7ef; }
"""

EXAMPLES = [
    "The transformer architecture was introduced in",
    "To be or not to be, that is the",
    "In machine learning, gradient descent is used to",
    "The quick brown fox jumps over the",
    "import torch\nimport torch.nn as",
]

with gr.Blocks() as demo:

    gr.HTML("""
    <div id='title'>
        <h1>⚡ Transformer Lens</h1>
        <p>GPT-2 · Next token prediction · Tokenization · Surprise · Entropy · Top-K candidates</p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=4):
            text_input = gr.Textbox(
                label="Input text",
                placeholder="Type anything and watch the model think...",
                value="The transformer architecture was introduced in",
                lines=3,
            )
        with gr.Column(scale=1):
            view_mode = gr.Radio(
                choices=["Surprise", "Entropy"],
                value="Surprise",
                label="Color mode",
            )
            run_btn = gr.Button("Analyze →", variant="primary")

    gr.Examples(examples=EXAMPLES, inputs=text_input, label="Try these")

    gr.HTML("<div class='section-label'>① Tokenization — how GPT-2 sees your text</div>")
    tokenization_out = gr.HTML()

    gr.HTML("<div class='section-label'>② Token coloring — surprise (green=expected, red=surprising) or entropy (blue intensity = uncertainty)</div>")
    token_out = gr.HTML()

    summary_out = gr.HTML()

    gr.HTML("<div class='section-label'>③ Top-10 candidates — what the model considered at each position (click to expand)</div>")
    topk_out = gr.HTML()

    # Wire up
    run_btn.click(analyze, inputs=[text_input, view_mode], outputs=[token_out, tokenization_out, topk_out, summary_out])
    text_input.submit(analyze, inputs=[text_input, view_mode], outputs=[token_out, tokenization_out, topk_out, summary_out])
    view_mode.change(analyze, inputs=[text_input, view_mode], outputs=[token_out, tokenization_out, topk_out, summary_out])

    gr.HTML("""
    <div style='text-align:center;color:#333;font-size:0.75em;padding:20px 0;font-family:monospace;'>
        Phase 1 · built on GPT-2 (124M) · next: attention heads, residual stream PCA, model comparison
    </div>
    """)

if __name__ == "__main__":
    demo.launch(css=CSS, theme=gr.themes.Base())