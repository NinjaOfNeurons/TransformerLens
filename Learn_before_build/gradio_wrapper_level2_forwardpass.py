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
            print(f"token: {tokenizer.decode([actual_token_id])} → prob: {actual_prob:.4f}")
            tokens = [tokenizer.decode([id]) for id in ids]
    
    result = list(zip(tokens, ids)) # zipping  token and ids in 1 single variable 
    return str(result)

gr.Interface(
    fn=analyze,
    inputs=gr.Textbox(label="Input text"),
    outputs=gr.HTML(label="Output")
).launch()