import math
import torch
from torch.optim import AdamW

from log_depth_recurrent_modeling import AutoregressiveGatedRecursiveCell

# seed

torch.manual_seed(42)

# constants

TRAIN_WINDOW_SIZE = 16
TEST_WINDOW_SIZES = (16, 32, 64, 128, 256, 512, 1024)

BATCH_SIZE = 64
NUM_STEPS = 200
LEARNING_RATE = 3e-3
DIM = 32

# model

model = AutoregressiveGatedRecursiveCell(
    num_tokens = 2,
    dim_embed = DIM,
    dim = DIM,
    window_size = TRAIN_WINDOW_SIZE
)

opt = AdamW(model.parameters(), lr = LEARNING_RATE, weight_decay = 1e-4)

# dataset helper

def generate_parity_batch(batch_size, seq_len):
    bits = torch.randint(0, 2, (batch_size, seq_len))
    labels = (torch.cumsum(bits, dim = -1) % 2).long()
    return bits, labels

# train on window size 16

print(f"training parity task on window size {TRAIN_WINDOW_SIZE}...\n")

model.train()

for step in range(1, NUM_STEPS + 1):
    bits, labels = generate_parity_batch(BATCH_SIZE, TRAIN_WINDOW_SIZE)

    loss = model(bits, labels = labels)
    loss.backward()

    opt.step()
    opt.zero_grad()

    if step % 50 == 0:
        print(f"step {step:3d} | loss: {loss.item():.4f}")

# evaluate length extrapolation

print(f"\nevaluating length extrapolation...\n")

model.eval()

with torch.no_grad():
    # 1. show failure when window_size is not overridden on length 32

    bits_32, labels_32 = generate_parity_batch(100, 32)
    preds_default = model(bits_32).argmax(dim = -1)

    acc_w1 = (preds_default[:, :16] == labels_32[:, :16]).float().mean().item()
    acc_w2 = (preds_default[:, 16:] == labels_32[:, 16:]).float().mean().item()

    print(f"without override (chunked by train window size {TRAIN_WINDOW_SIZE}):")
    print(f"  window 1 (tokens  0..15) accuracy: {acc_w1 * 100:.1f}%")
    print(f"  window 2 (tokens 16..31) accuracy: {acc_w2 * 100:.1f}% (breaks down due to no cross-window communication)\n")

    # 2. extrapolate across window sizes by overriding window_size on forward

    print(f"{'window':<8} | {'depth':<6} | {'token acc':<10} | {'seq acc':<10}")
    print("-" * 42)

    for window_size in TEST_WINDOW_SIZES:
        depth = int(math.log2(window_size))
        bits, labels = generate_parity_batch(100, window_size)

        logits = model(bits, window_size = window_size)
        preds = logits.argmax(dim = -1)

        token_acc = (preds == labels).float().mean().item()
        seq_acc = ((preds == labels).all(dim = -1)).float().mean().item()

        print(f"{window_size:<8} | {depth:<6} | {token_acc * 100:6.2f}%   | {seq_acc * 100:6.2f}%")
