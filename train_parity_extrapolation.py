import math
import torch
from torch.optim import AdamW

from log_depth_recurrent_modeling import AutoregressiveGatedRecursiveCell

# seed

torch.manual_seed(42)

# constants

TRAIN_SEQ_LEN = 16
TEST_SEQ_LENS = (16, 32, 64, 128, 256, 512, 1024)

BATCH_SIZE = 64
NUM_STEPS = 200
LEARNING_RATE = 3e-3
DIM = 32

# model

model = AutoregressiveGatedRecursiveCell(
    num_tokens = 2,
    dim_embed = DIM,
    dim = DIM
)

opt = AdamW(model.parameters(), lr = LEARNING_RATE, weight_decay = 1e-4)

# dataset helper

def generate_parity_batch(batch_size, seq_len):
    bits = torch.randint(0, 2, (batch_size, seq_len))
    labels = (torch.cumsum(bits, dim = -1) % 2).long()
    return bits, labels

# train on short sequences

print(f"training parity task on sequence length {TRAIN_SEQ_LEN}...\n")

model.train()

for step in range(1, NUM_STEPS + 1):
    bits, labels = generate_parity_batch(BATCH_SIZE, TRAIN_SEQ_LEN)

    loss = model(bits, labels = labels)
    loss.backward()

    opt.step()
    opt.zero_grad()

    if step % 50 == 0:
        print(f"step {step:3d} | loss: {loss.item():.4f}")

# evaluate length extrapolation, no window size override needed

print("\nevaluating length extrapolation...\n")

model.eval()

with torch.no_grad():
    print(f"{'seq len':<8} | {'depth':<6} | {'token acc':<10} | {'seq acc':<10}")
    print("-" * 42)

    for seq_len in TEST_SEQ_LENS:
        depth = math.ceil(math.log2(seq_len))
        bits, labels = generate_parity_batch(100, seq_len)

        preds = model(bits).argmax(dim = -1)

        token_acc = (preds == labels).float().mean().item()
        seq_acc = ((preds == labels).all(dim = -1)).float().mean().item()

        print(f"{seq_len:<8} | {depth:<6} | {token_acc * 100:6.2f}%   | {seq_acc * 100:6.2f}%")
