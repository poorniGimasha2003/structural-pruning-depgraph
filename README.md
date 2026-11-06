# Systematic Evaluation and Extension of Dependency Graph-Based Structural Pruning

**CO3554 Data Management Project** · Department of Computer Engineering, University of Sri Jayewardenepura

| | |
|---|---|
| **Student** | Pathirage L.P.D.P.G (23/ENG/091) |
| **Supervisor** | Sanka Mohottala |
| **External supervisors** | Mr. Asiri Gawesha (SLIIT), Dr. Dinuka Sahabandu (University of Washington) |
| **Status** | Proposal stage - started October 2026 |

## About the project

Structural pruning makes deep neural networks smaller and faster by removing whole channels, neurons and attention heads. **DepGraph** (Fang et al., CVPR 2023), implemented in the open-source [Torch-Pruning](https://github.com/VainF/Torch-Pruning) library, claims to prune almost any architecture automatically. Outside image models, this claim has been tested on very few models.

This project will:

1. Build an automated **test harness** that applies Torch-Pruning to 30+ CNN, Transformer, RNN and GNN models and records whether pruning succeeds, crashes, or silently gives wrong results.
2. Produce a **support map** showing which layers and model families work, and why failures happen.
3. **Extend the library** with pruning support for at least one under-served layer (e.g. GRU, stacked LSTM, graph layers).
4. Run a **fair benchmark** of pruning methods on non-image models (accuracy, model size, MACs, latency).
5. Store all results in a **database** and show them in an interactive **dashboard**.

## Repository structure

```
harness/        Automated test harness
models/         Scripts that load the models under test
experiments/    Benchmark experiments and their results
dashboard/      Results dashboard
data/samples/   Small dataset samples (full datasets are downloaded, not committed)
docs/           Proposal, slides and project notes
notebooks/      Learning and exploration notebooks
```

## Milestones

| ID | Week | Milestone | Status |
|---|---|---|---|
| M1 | 0 | Project proposal submitted | ✅ |
| M2 | 6 | Background study done; DepGraph result reproduced on CIFAR-10 | ⬜ |
| M3 | 13 | Test harness complete for ≥ 30 models | ⬜ |
| M4 | 17 | Support map and failure analysis complete | ⬜ |
| M5 | 24 | Library extension implemented, tested, pull request opened | ⬜ |
| M6 | 31 | Benchmark experiments complete; dashboard working | ⬜ |
| M7 | 35 | Research paper submitted; code released | ⬜ |

## Setup

Setup and installation instructions will be added as the code is developed.

## Datasets

The project uses public benchmark datasets: CIFAR-10/100, ImageNet-1k (validation), AG News, PPI and Cora. Full datasets are not stored in this repository; download instructions and small samples will be added in `data/`.

## Use of generative AI

AI tools were used during this project. Every use is recorded in [`docs/AI_USAGE.md`](docs/AI_USAGE.md).

## References

1. G. Fang, X. Ma, M. Song, M. B. Mi, and X. Wang, "DepGraph: Towards any structural pruning," CVPR, 2023.
2. G. Fang, "Torch-Pruning," GitHub repository, 2023. https://github.com/VainF/Torch-Pruning
