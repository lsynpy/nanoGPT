"""
Sample from a trained model
"""

from contextlib import nullcontext

import torch

from model import GPT

# -----------------------------------------------------------------------------
prompts = ["Who are you", "Mr Potter", "Well done"]
batch_size = 3
max_new_tokens = 50  # number of tokens generated in each sample
temperature = 0.8  # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200  # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = "cpu"
dtype = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast
ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[dtype]
ctx = (
    nullcontext()
    if device_type == "cpu"
    else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
)


# disable dropout when inference
model: GPT = GPT.from_pretrained("gpt2", dict(dropout=0.0))

model.eval()
model.to(device)
model = torch.compile(model)

# tiktoken tokenizer
# import tiktoken
# tokenizer = tiktoken.get_encoding("gpt2")
# start_ids = [tokenizer.encode(x, allowed_special={"<|endoftext|>"}) for x in prompts]
# x = torch.tensor(start_ids, dtype=torch.long, device=device)

# hf transformer tokenizer
from transformers import GPT2Tokenizer

tokenizer: GPT2Tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
x = tokenizer(prompts, padding=True, return_tensors="pt").input_ids

y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
for i in range(batch_size):
    print(f"\n--------------> batch {i}\n")
    print(tokenizer.decode(y[i].tolist()))
