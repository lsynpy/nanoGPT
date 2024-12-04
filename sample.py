"""
Sample from a trained model
"""

from contextlib import nullcontext

import tiktoken
import torch

from model import GPT

# -----------------------------------------------------------------------------
init_from = "gpt2"  # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = "out"  # ignored if init_from is not 'resume'
start = ["I", "Mr", "well"]
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


model: GPT = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
model = torch.compile(model)

encoder = tiktoken.get_encoding("gpt2")

# encode the beginning of the prompt
start_ids = [encoder.encode(x, allowed_special={"<|endoftext|>"}) for x in start]
x = torch.tensor(start_ids, dtype=torch.long, device=device)

# run generation
with torch.no_grad():
    with ctx:
        y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
        for i in range(batch_size):
            print(f"--------------> sample {i}")
            print(encoder.decode(y[i].tolist()))
