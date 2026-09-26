import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import gzip
import random
import tqdm
import numpy as np

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from accelerate import Accelerator

from fire import Fire

from log_depth_recurrent_modeling import ARGRC

# helpers

def cycle(loader):
    while True:
        for data in loader:
            yield data

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return "".join(list(map(decode_token, tokens)))

# train

def main(
    multi_head = False,
    heads = 8,
    dim_head = None,
    dim = 512,
    depth = 1,
    seq_len = 512,
    batch_size = 4,
    grad_accum_every = 4,
    learning_rate = 1e-4,
    num_batches = int(1e5),
    validate_every = 100,
    generate_every = 500,
    prime_length = 128,
    generate_length = 512
):
    accelerator = Accelerator()

    # the AR-GRC char language model

    model_kwargs = dict(
        num_tokens = 256,
        dim = dim,
        depth = depth,
        max_seq_len = seq_len,
        shift_tokens = True
    )

    if multi_head:
        from log_depth_recurrent_modeling.multi_head_ar_grc import MultiHeadARGRC

        model = MultiHeadARGRC(**model_kwargs, heads = heads, dim_head = dim_head)
        accelerator.print(f"training multi-head ar-grc with {heads} heads")
    else:
        model = ARGRC(**model_kwargs)
        accelerator.print("training ar-grc")

    # prepare enwik8 data

    with gzip.open("./data/enwik8.gz") as file:
        data = np.frombuffer(file.read(int(95e6)), dtype = np.uint8).copy()
        np_train, np_valid = np.split(data, [int(90e6)])
        data_train, data_val = torch.from_numpy(np_train), torch.from_numpy(np_valid)

    class TextSamplerDataset(Dataset):
        def __init__(self, data, seq_len):
            super().__init__()
            self.data = data
            self.seq_len = seq_len

        def __len__(self):
            return self.data.size(0) // self.seq_len

        def __getitem__(self, index):
            rand_start = torch.randint(0, self.data.size(0) - self.seq_len, (1,))
            full_seq = self.data[rand_start : rand_start + self.seq_len + 1].long()
            return full_seq

    train_dataset = TextSamplerDataset(data_train, seq_len)
    val_dataset = TextSamplerDataset(data_val, seq_len)
    train_loader = DataLoader(train_dataset, batch_size = batch_size)
    val_loader = DataLoader(val_dataset, batch_size = batch_size)

    # optimizer

    optim = Adam(model.parameters(), lr = learning_rate)

    model, optim, train_loader, val_loader = accelerator.prepare(
        model, optim, train_loader, val_loader
    )

    train_loader = cycle(train_loader)
    val_loader = cycle(val_loader)

    # training

    for i in tqdm.tqdm(range(num_batches), mininterval = 10.0, desc = "training"):
        model.train()

        for _ in range(grad_accum_every):
            data = next(train_loader)

            loss = model(data, return_loss = True)

            accelerator.backward(loss / grad_accum_every)

        accelerator.print(f"training loss: {loss.item():.3f}")

        accelerator.clip_grad_norm_(model.parameters(), 0.5)

        optim.step()
        optim.zero_grad()

        if validate_every > 0 and i % validate_every == 0:
            model.eval()
            with torch.no_grad():
                valid_data = next(val_loader)

                loss = model(valid_data, return_loss = True)
                accelerator.print(f"validation loss: {loss.item():.3f}")

        if generate_every > 0 and i % generate_every == 0:
            model.eval()

            inp = random.choice(val_dataset)[:prime_length]
            inp = inp.to(accelerator.device)

            prime = decode_tokens(inp)
            accelerator.print(f"INPUT: {prime}")

            prompt = inp[None, ...]

            sampled = accelerator.unwrap_model(model).generate(prompt, generate_length)

            base_decode_output = decode_tokens(sampled[0])

            accelerator.print(f"\nOUTPUT: {base_decode_output}")

if __name__ == '__main__':
    Fire(main)
