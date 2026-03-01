import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
model.eval()

def prob_to_color(prob):
    r = int(255 * (1 - prob))
    g = int(255 * prob)
    return f"rgb({r},{g},60)"

def analyze(text):
    ids_tensor = tokenizer.encode(text, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(ids_tensor)
        logits = outputs.logits[0]
        T = ids_tensor.shape[1]

        # collect stats for every token
        stats = []
        for i in range(T):
            token_id = ids_tensor[0][i].item()
            token_str = tokenizer.decode([token_id])

            if i == 0:
                stats.append({"token": token_str, "prob": None, "entropy": None})
                continue

            prob = F.softmax(logits[i-1], dim=0)
            actual_prob = prob[token_id].item()
            entropy = -torch.sum(prob * torch.log(prob + 1e-10)).item()

            stats.append({"token": token_str, "prob": actual_prob, "entropy": entropy})

    # build html
    html = ""
    for s in stats:
        token = s["token"]
        if s["prob"] is None:
            html += f"<span style='background:#444; color:white; padding:5px; margin:3px; border-radius:4px;'>{token} start</span>"
        else:
            color = prob_to_color(s["prob"])
            html += f"<span style='background:{color}; color:white; padding:5px; margin:3px; border-radius:4px;'>{token} {s['prob']:.2%}</span>"
    return html

gr.Interface(
    fn=analyze,
    inputs=gr.Textbox(label="Input text"),
    outputs=gr.HTML(label="Output")
).launch()




"""
Discussion:

is     → prob: 0.1217 | entropy: 3.089   ← most confident
of     → prob: 0.1642 | entropy: 4.266
France → prob: 0.0065 | entropy: 5.602
Paris  → prob: 0.0322 | entropy: 5.998
capital→ prob: 0.0001 | entropy: 8.822   ← most uncertain
Notice something interesting — of has higher probability than Paris (16% vs 3%) but Paris has higher entropy (5.998 vs 4.266).
That means at the of position, the model was fairly focused — it strongly expected of after "capital". But at the Paris position, even though Paris was the top pick, the model had many other equally reasonable options spread around — cities, descriptions, articles.
Probability = how likely was THIS token
Entropy = how spread out was the entire distribution

"""