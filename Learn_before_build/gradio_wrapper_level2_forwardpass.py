import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
model.eval()

def analyze(text):
    ids = tokenizer.encode(text)
    ids_tensor = tokenizer.encode(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(ids_tensor)
        logits = outputs.logits[0]  # shape: (T, vocab_size)
        T = ids_tensor.shape[1]
        for i in range(1, T):
            logits_prev = logits[i-1]       # predicts position i
            prob = F.softmax(logits_prev, dim=0)
            actual_token_id = ids_tensor[0][i].item()
            actual_prob = prob[actual_token_id].item()
            entropy = -torch.sum(prob * torch.log(prob + 1e-10)).item()
            top_k = torch.topk(prob, k=10)
            top_k_tokens = [tokenizer.decode([idx.item()]) for idx in top_k.indices]
            top_k_probs = [f"{v.item():.2%}" for v in top_k.values]

            print(f"token: {tokenizer.decode([actual_token_id])} → prob: {actual_prob:.4f} | entropy: {entropy:.3f}")
            print(f"top 10: {list(zip(top_k_tokens, top_k_probs))}")
            print("---")

            tokens = [tokenizer.decode([id]) for id in ids]
    
    result = list(zip(tokens, ids)) # zipping  token and ids in 1 single variable 
    return str(result)

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