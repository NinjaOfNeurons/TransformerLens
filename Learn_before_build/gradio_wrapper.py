import gradio as gr
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")  # runs once

def analyze(text):
    ids = tokenizer.encode(text)
    tokens = [tokenizer.decode([id]) for id in ids]
    

    result = list(zip(tokens, ids)) # zipping  token and ids in 1 single variable 
    return str(result)

gr.Interface(
    fn=analyze,
    inputs=gr.Textbox(label="Input text"),
    outputs=gr.HTML(label="Output")
).launch()