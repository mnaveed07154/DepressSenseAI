# DepressSenseAI

## A Speech-Based Depression Detection Framework Using VoxDepressNet with Domain-Invariant and Multi-Task Learning

DepressSenseAI is a deep learning framework for speech-based depression detection and severity estimation using hybrid acoustic modeling, self-supervised speech representations, temporal attention modeling, and domain-invariant adversarial learning.

The framework is designed to operate across heterogeneous public speech datasets and supports:

- Binary depression screening
- Depression severity estimation
- Cross-dataset generalization
- Domain-invariant representation learning
- Attention-based temporal aggregation
- Multi-task learning

---

# Features

## Core Capabilities

- Self-supervised speech embeddings using wav2vec2 / HuBERT / WavLM
- Acoustic and prosodic feature extraction
- Hybrid gated feature fusion
- Bi-GRU / Transformer temporal modeling
- Attention pooling for session-level embedding
- Multi-task learning
- Domain-adversarial learning using Gradient Reversal Layer (GRL)
- Cross-dataset evaluation
- Ablation studies
- Reproducible training pipeline

---

# Project Structure

```text
DepressSenseAI/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── metadata/
│
├── src/
│   ├── config.py
│   ├── preprocess.py
│   ├── feature_extraction.py
│   ├── dataset.py
│   ├── train.py
│   ├── evaluate.py
│   ├── cross_dataset_eval.py
│   ├── ablation.py
│   ├── utils.py
│   │
│   └── models/
│       ├── gradient_reversal.py
│       ├── attention_pooling.py
│       ├── fusion.py
│       └── vox_depress_net.py
│
├── notebooks/
├── results/
├── requirements.txt
├── README.md
└── main.py
```

---

# Datasets

The framework supports the following public datasets:

| Dataset | Language | Task |
|----------|----------|------|
| ADC | Italian | Depression classification |
| EATD-Corpus | Mandarin | Depression classification |
| PDCH | Mandarin | Depression severity estimation |

---

# Dataset Preparation

Place datasets inside:

```text
data/raw/
```

Recommended structure:

```text
data/raw/
├── ADC/
├── EATD/
└── PDCH/
```

Metadata CSV format:

```csv
audio_path,label,severity,domain,subject_id
sample.wav,1,18,0,S001
```

Where:

- `label` → 0 = non-depressed, 1 = depressed
- `severity` → HAMD-17 score
- `domain`
  - 0 = ADC
  - 1 = EATD
  - 2 = PDCH

---

# Installation

## Clone Repository

```bash
git clone https://github.com/mnaveed07154/DepressSenseAI
cd DepressSenseAI
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Required Libraries

Main dependencies include:

- torch
- torchaudio
- transformers
- librosa
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn
- opensmile
- praat-parselmouth
- tqdm

---

# Audio Preprocessing

The preprocessing pipeline performs:

- Resampling to 16 kHz
- Mono conversion
- Loudness normalization
- Voice Activity Detection (VAD)
- Fixed-length segmentation

Run preprocessing:

```bash
python src/preprocess.py
```

---

# Training

Train VoxDepressNet:

```bash
python src/train.py
```

Training includes:

- Depression classification
- Severity estimation
- Domain-adversarial training
- Attention-based temporal aggregation

---

# Evaluation

Evaluate trained model:

```bash
python src/evaluate.py
```

Outputs:

- Accuracy
- F1-score
- UAR
- AUC
- MAE
- RMSE
- Pearson correlation
- CCC

---

# Cross-Dataset Evaluation

Run domain generalization experiments:

```bash
python src/cross_dataset_eval.py
```

This evaluates:

- Train on ADC → Test on EATD
- Train on EATD → Test on PDCH
- Leave-one-dataset-out evaluation

---

# Ablation Studies

Run ablation experiments:

```bash
python src/ablation.py
```

Available ablations:

- Without SSL branch
- Without acoustic branch
- Without GRL
- Without attention pooling
- Without severity head
- Without gated fusion

---

# Model Architecture

## VoxDepressNet Pipeline

```text
Speech Audio
    ↓
Audio Preprocessing
    ↓
Speech Segmentation
    ↓
SSL Encoder + Acoustic Features
    ↓
Gated Feature Fusion
    ↓
Bi-GRU / Transformer Encoder
    ↓
Attention Pooling
    ↓
Session Embedding
    ↓
├── Depression Classification Head
├── Severity Estimation Head
└── Domain Classifier (GRL)
```

---

# Training Objective

The total loss is:

```text
L_total = L_cls + αL_severity + βL_domain
```

Where:

- `L_cls` → Depression classification loss
- `L_severity` → Severity estimation loss
- `L_domain` → Domain adversarial loss

---

# Results Directory

Generated outputs are stored in:

```text
results/
```

Includes:

- checkpoints/
- metrics/
- plots/
- logs/

---

# Reproducibility

The framework ensures reproducibility through:

- Fixed random seeds
- Subject-independent splits
- Standardized preprocessing
- Version-controlled experiments

---

# Hardware Requirements

| Component | Requirement |
|-----------|-------------|
| GPU | NVIDIA RTX 3060 or higher |
| VRAM | 8 GB minimum |
| RAM | 16 GB minimum |
| Python | 3.10+ |

---

# Citation

Citation
If you use this repository, please cite:
@article{DepressSenseAI2026,
title={DepressSenseAI: A Speech-Based Depression Detection Framework Using VoxDepressNet
with Domain-Invariant and Multi-Task Learning},
author={Author Name},
journal={SCI Journal},
year={2026}
}

# License

This project is intended for research purposes.

---
