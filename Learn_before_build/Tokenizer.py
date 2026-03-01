from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("gpt2")

text = "capital of france is paris"


ids = tokenizer.encode(text)

print(ids)

tokens = [tokenizer.decode([id]) for id in ids]

print(tokens)
