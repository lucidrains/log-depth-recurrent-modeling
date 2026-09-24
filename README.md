<img src="./ldrlm-fig1.png" width="350px"></img>

## Log Depth Recurrent Modeling (wip)

Explorations into the [Log Depth Recurrent Modeling](https://arxiv.org/abs/2609.28212) proposed by Yiqin Wang of Imperial College London

`AutoregressiveGatedRecursiveCell` scans a sequence in log depth by padding it to the nearest power of two and running a Blelloch scan over the resulting balanced binary tree, giving it length extrapolation for free. Setting `max_seq_len` (a power of two) pins the window size instead, padding and folding longer sequences into independent windows processed in parallel.

## Usage

Train on the parity task at sequence length 16, then extrapolate to unseen sequence lengths

```python
import torch
from log_depth_recurrent_modeling import ARGRC

torch.manual_seed(42)

model = ARGRC(num_tokens = 2, dim_embed = 32, dim = 32)
opt = torch.optim.AdamW(model.parameters(), lr = 3e-3)

def parity_batch(seq_len, batch_size = 64):
    bits = torch.randint(0, 2, (batch_size, seq_len))
    return bits, bits.cumsum(dim = -1) % 2

# train on sequences of length 16

for step in range(1, 201):
    bits, labels = parity_batch(16)

    loss = model(bits, labels = labels)
    loss.backward()
    opt.step()
    opt.zero_grad()

    if step % 50 == 0:
        print(f"step {step:3d} | loss: {loss.item():.4f}")

# extrapolate to unseen sequence lengths

for seq_len in (16, 64, 256, 1024):
    bits, labels = parity_batch(seq_len, 100)
    acc = (model(bits).argmax(dim = -1) == labels).float().mean().item()
    print(f"seq len {seq_len:4d} | accuracy: {acc * 100:.1f}%")
```

```
step  50 | loss: 0.6559
step 100 | loss: 0.0003
step 150 | loss: 0.0000
step 200 | loss: 0.0000
seq len   16 | accuracy: 100.0%
seq len   64 | accuracy: 100.0%
seq len  256 | accuracy: 100.0%
seq len 1024 | accuracy: 100.0%
```

## Citations

```bibtex
@misc{wang2026logdepthrecurrentlanguagemodeling,
    title    = {Log-Depth Recurrent Language Modeling},
    author   = {Yiqin Wang and Nuri Cingillioglu and Charles Pert},
    year     = {2026},
    eprint   = {2609.28212},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG},
    url      = {https://arxiv.org/abs/2609.28212},
}
```

```bibtex
@misc{shen2019orderedmemory,
    title   = {Ordered Memory},
    author  = {Yikang Shen and Shawn Tan and Arian Hosseini and Zhouhan Lin and Alessandro Sordoni and Aaron Courville},
    year    = {2019},
    eprint  = {1910.13466},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG},
    url     = {https://arxiv.org/abs/1910.13466},
}
```

```bibtex
@misc{pert2026lengthgeneralizationlogdepthrecurrent,
    title     = {Length Generalization with Log-Depth Recurrent Units},
    author    = {Charles Pert and Dalal Alrajeh and Alessandra Russo},
    year      = {2026},
    eprint    = {2605.26035},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG},
    url       = {https://arxiv.org/abs/2605.26035},
}
```
