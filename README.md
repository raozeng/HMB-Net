# HMB-Net: Synergizing Latent Sequential Dynamics and Local Motifs via State Space Models for Robust RNA Methylation Prediction** is an interpretable hybrid deep learning framework for accurate site-level identification of RNA modifications (m6A, m5C, and m1A) across species. It overcomes the bottleneck of simultaneously mapping vast long-range nucleotide dependencies and microscopic local structural motifs. By synergizing the linear sequence dynamics of lower-complexity **State Space Models (Mamba)** with the localized gating mechanisms of **Bi-directional LSTMs (Bi-LSTMs)**, HMB-Net leverages a specialized "Latent-to-Motif" sequential hierarchical architecture to outclass orthodox parallel topologies and computationally heavy attention layers.


## Introduction
**HMB-Net** is an interpretable hybrid deep learning framework for accurate site-level identification of RNA modifications (m6A, m5C, and m1A) across species. It overcomes the bottleneck of simultaneously mapping vast long-range nucleotide dependencies and microscopic local structural motifs. By synergizing the linear sequence dynamics of lower-complexity **State Space Models (Mamba)** with the localized gating mechanisms of **Bi-directional LSTMs (Bi-LSTMs)**, HMB-Net leverages a specialized "Latent-to-Motif" sequential hierarchical architecture to outclass orthodox parallel topologies and computationally heavy attention layers.

## Repository Structure
- `run.py`: The unified execution portal.
- `model.py`: Model architecture blueprint containing HMB-Net, specific ablation components (BiLSTMOnly, MambaOnly, R-HMB-Net), and a suite of baseline implementations.
- `train.py` / `test.py`: Modular training (K-Fold CV) and isolated testing engines.
- `dataset.py`: Memory-efficient FASTA processors and k-mer encoders.

## Environment Installation
Ensure a properly configured CUDA-enabled Python environment (Python 3.8+ recommended). Note that Mamba is extensively reliant on PyTorch CUDA extensions.

```bash
pip install -r requirements.txt
```

*(Note: Installing `mamba-ssm` might require the `causal-conv1d` dependency depending on your specific CUDA toolkit environment.)*

## Quick Start

### 1. Training the Architecture
Launch a full 5-fold cross-validation loop. Supported models include `hybrid` (HMB-Net), `bilstm`, `mamba`, `reverse` (R-HMB-Net), and `transformer`.

```bash
python run.py \
    --mode train \
    --model_type hybrid \
    --train_data path/to/train_dataset.fasta \
    --epochs 30 \
    --batch_size 64
```
*Trained model checkpoints and raw metrics log strings will be isolated within the `train_model` output directory.*

### 2. Independent Testing
Load the heaviest model dictionary and deploy it onto blind sequences.

```bash
python run.py \
    --mode test \
    --model_type hybrid \
    --test_data path/to/independent_test_dataset.fasta
```
*Predictions map to internal evaluation scripts that emit exhaustive ACC, Sn, Sp, F1, MCC, and ROC-AUC summaries into `test_result/`.*

## License & Citation
If this repository accelerates your epitranscriptomic exploration or benchmark engineering, please kindly consider referencing our original work.
