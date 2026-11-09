# Research Diary — Structural Pruning Project

Newest entries at the top.

---

## 2026-11-09 — Week 1, Day 1: Setup + Tensors

**Did**
- set up VS Code + Jupyter on my Mac (pruning-env, PyTorch 2.15, MPS available)
- Connected the project folder to the GitHub repo
- Learned tensors and shapes; removed rows and columns from a tensor

**Results**
- [3, 4] → remove row 1 → [2, 4]
- [4, 6] → keep columns [0, 2, 4, 5] → [4, 4]

**Problems and fixes**
- Kernel not found → registered it with `python -m ipykernel install`
- `torch._version_` typo → correct is `torch.__version__`
- Can't put `:` inside a list → use `A[:, keep]`

**Learned**
- Pruning = removing rows / slices from weight tensors
- Conv weight [16, 3, 5, 5] = 16 filters, each 3 × 5 × 5 (first number = output)

**Next**
- Day 2: layer shapes and why layers are tied (Conv → BN → Conv)

**AI use**
- Used Claude to guide setup and explain tensor shapes