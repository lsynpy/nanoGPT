import torch
from transformers import GPT2Tokenizer

from model import GPT

# -----------------------------------------------------------------------------
prompts = [
    "Who are you",
    "Mr Potter",
    "Alan Turing theorized that computers would one day become",
]
batch_size = len(prompts)
max_new_tokens = 50  # number of tokens generated in each sample
device = "cpu"
# -----------------------------------------------------------------------------


# disable dropout when inference
model: GPT = GPT.from_pretrained("gpt2", dict(dropout=0.0))

model.eval()
model.to(device)
model = torch.compile(model)

# hf transformer tokenizer
tokenizer: GPT2Tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
# TODO: how to handle the padding token gracefully
x = tokenizer.batch_encode_plus(prompts, padding=True, return_tensors="pt")["input_ids"]

y = model.generate(x, max_new_tokens)
for i in range(batch_size):
    print(f"\n--------------> batch {i}\n")
    print(tokenizer.decode(y[i].tolist()))
