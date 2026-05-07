DepressSenseAI
DepressSenseAI: A Speech-Based Depression Detection Framework Using VoxDepressNet with Domain-Invariant and Multi-Task Learning
DepressSenseAI is a deep learning framework for speech-based depression detection and severity estimation using hybrid acoustic modeling, self-supervised speech representations, temporal attention modeling, and domain-invariant adversarial learning.
The framework is designed to operate across heterogeneous public speech datasets and supports:
•	Binary depression screening
•	Depression severity estimation
•	Cross-dataset generalization
•	Domain-invariant representation learning
•	Attention-based temporal aggregation
•	Multi-task learning
________________________________________
Features
Core Capabilities
•	Self-supervised speech embeddings using wav2vec2/HuBERT/WavLM
•	Acoustic and prosodic feature extraction
•	Hybrid gated feature fusion
•	Bi-GRU / Transformer temporal modeling
•	Attention pooling for session-level embedding
•	Multi-task learning
•	Domain-adversarial learning using Gradient Reversal Layer (GRL)
•	Cross-dataset evaluation
•	Ablation studies
•	Reproducible training pipeline
________________________________________
Project Structure
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
________________________________________
Datasets
The framework supports the following public datasets:
Dataset	Language	Task
ADC	Italian	Depression classification
EATD-Corpus	Mandarin	Depression classification
PDCH	Mandarin	Depression severity estimation
________________________________________
Dataset Preparation
Place datasets inside:
data/raw/
Recommended structure:
data/raw/
├── ADC/
├── EATD/
└── PDCH/
Metadata CSV format:
audio_path,label,severity,domain,subject_id
Example:
sample.wav,1,18,0,S001
Where:
•	label → 0 = non-depressed, 1 = depressed
•	severity → HAMD-17 score
•	domain
o	0 = ADC
o	1 = EATD
o	2 = PDCH
________________________________________
Installation
Clone Repository
git clone https://github.com/yourusername/DepressSenseAI.git
cd DepressSenseAI
________________________________________
Create Virtual Environment
Windows
python -m venv venv
venv\Scripts\activate
Linux / Mac
python3 -m venv venv
source venv/bin/activate
________________________________________
Install Dependencies
pip install -r requirements.txt
________________________________________
Required Libraries
Main dependencies include:
torch
torchaudio
transformers
librosa
numpy
pandas
scikit-learn
matplotlib
seaborn
opensmile
praat-parselmouth
tqdm
________________________________________
Audio Preprocessing
The preprocessing pipeline performs:
•	Resampling to 16 kHz
•	Mono conversion
•	Loudness normalization
•	Voice Activity Detection (VAD)
•	Fixed-length segmentation
Run preprocessing:
python src/preprocess.py
________________________________________
Training
Train VoxDepressNet:
python src/train.py
Training includes:
•	Depression classification
•	Severity estimation
•	Domain-adversarial training
•	Attention-based temporal aggregation
________________________________________
Evaluation
Evaluate trained model:
python src/evaluate.py
Outputs:
•	Accuracy
•	F1-score
•	UAR
•	AUC
•	MAE
•	RMSE
•	Pearson correlation
•	CCC
________________________________________
Cross-Dataset Evaluation
Run domain generalization experiments:
python src/cross_dataset_eval.py
This evaluates:
•	Train on ADC → Test on EATD
•	Train on EATD → Test on PDCH
•	Leave-one-dataset-out evaluation
________________________________________
Ablation Studies
Run ablation experiments:
python src/ablation.py
Available ablations:
•	Without SSL branch
•	Without acoustic branch
•	Without GRL
•	Without attention pooling
•	Without severity head
•	Without gated fusion
________________________________________
Model Architecture
VoxDepressNet Pipeline
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
________________________________________
Training Objective
The total loss is:
L_total = L_cls + αL_severity + βL_domain
Where:
•	L_cls → Depression classification loss
•	L_severity → Severity estimation loss
•	L_domain → Domain adversarial loss
________________________________________
Results Directory
Generated outputs are stored in:
results/
Includes:
checkpoints/
metrics/
plots/
logs/
________________________________________
Example Outputs
Generated files include:
trained_model.pt
classification_report.csv
severity_metrics.csv
confusion_matrix.png
roc_curve.png
attention_weights.csv
cross_dataset_results.csv
________________________________________
Reproducibility
The framework ensures reproducibility through:
•	Fixed random seeds
•	Subject-independent splits
•	Standardized preprocessing
•	Version-controlled experiments
________________________________________
Hardware Requirements
Recommended:
Component	Requirement
GPU	NVIDIA RTX 3060 or higher
VRAM	8 GB minimum
RAM	16 GB minimum
Python	3.10+
________________________________________
Citation
If you use this repository, please cite:
@article{DepressSenseAI2026,
  title={DepressSenseAI: A Speech-Based Depression Detection Framework Using VoxDepressNet with Domain-Invariant and Multi-Task Learning},
  author={Author Name},
  journal={SCI Journal},
  year={2026}
}
________________________________________
License
This project is intended for academic and research purposes.
________________________________________
Contact
For research collaborations, issues, or improvements:
Author: Your Name
Email: your_email@example.com

