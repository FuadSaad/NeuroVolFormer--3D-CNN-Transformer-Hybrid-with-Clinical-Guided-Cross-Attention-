# NeuroVolFormer: 3D-CNN-Transformer-Hybrid with Clinical-Guided Cross-Attention & NeuroGAT

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyTorch%20Geometric-2.3%2B-3C2179.svg)](https://pyg.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Official Implementation** for the multimodal Alzheimer's Disease Neuroimaging Initiative (ADNI) benchmark paper:  
> *"NeuroVolFormer: 3D-CNN-Transformer-Hybrid with Clinical-Guided Cross-Attention and Population Graph Attention Networks for Early Alzheimer's Diagnosis, Continuous Cognitive Severity, and 24-Month Progression Trajectory Prognosis"*.

---

## 📌 Clinical Motivation & Abstract

Accurate differentiation of Alzheimer's Disease (AD), Cognitively Normal (CN), Early Mild Cognitive Impairment (EMCI), and Late Mild Cognitive Impairment (LMCI) remains one of the hardest transitional challenges in computational neurology. Standard deep learning and machine learning models struggle with the **transitional boundary between EMCI and LMCI** and suffer from majority class bias.

**NeuroGAT** introduces a patient-level population graph framework that synergistically fuses:
1. **3D Structural T1-weighted MRI** (DenseNet121 deep representations)
2. **High-Dimensional Radiomics** (68 pyradiomics shape, texture, and intensity descriptors)
3. **Clinical Phenotyping & Cognitive Scales** (MMSE, CDRSB, Logical Memory, Age, Gender, Education)

---

## 🚀 Key Methodological Innovations

1. **Dual-Head Multi-Task Learning**:
   Jointly predicts 4-class diagnostic staging and continuous cognitive impairment severity ($\lambda_{\text{cog}} = 0.1$), establishing a regularized, clinically grounded latent manifold.
2. **Multi-Scale Edge-Attentive Graph Topology**:
   Constructs population graphs over multi-scale neighborhood radii ($K = [3, 5, 10]$) with dynamic cross-attention fusion.
3. **Fisher-Consistent Logit-Adjusted Loss (NeurIPS 2020)**:
   Mathematically enforces class margins $\tau \log \pi_y$ directly derived from Bayesian Decision Theory to boost minority transitional LMCI recall without synthetic sample corruption.
4. **Effective Number of Samples Weighting (CVPR 2019)**:
   Information-theoretic volume discounting ($\beta = 0.9999$) to balance gradient updates across imbalanced clinical classes.
5. **DropEdge Graph Regularization (ICLR 2020)**:
   Prevents GNN over-smoothing across deep multi-scale layers via dynamic 15% edge dropping during training.
6. **24-Month MCI Progression Prognosis Engine**:
   Computes a continuous non-linear hazard trajectory index classifying patients into Low (<25%), Moderate (25-60%), and Rapid Conversion (>60%) clinical strata.

---

## 📊 Benchmark Results (5-Fold Stratified Cross-Validation)

### Proposed NeuroGAT vs. Traditional ML Baselines (Pure Imaging Features)

| Model Architecture | Input Modality | Overall Accuracy | Macro Avg F1 | LMCI Recall | DeLong $p$-value |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Support Vector Machine (RBF) | Imaging (100-D) | 49.11% | 46.20% | 34.20% | $< 10^{-5}$ |
| Random Forest (500 Trees) | Imaging (100-D) | 50.19% | 47.85% | 36.50% | $< 10^{-5}$ |
| Multi-Layer Perceptron (MLP) | Imaging (100-D) | 41.19% | 38.90% | 28.10% | $< 10^{-5}$ |
| Regularized XGBoost | Imaging (100-D) | 48.35% | 45.10% | 32.70% | $< 10^{-5}$ |
| **Proposed NeuroGAT (Ours)** | **Multimodal Graph** | **82.29%** | **78.43%** | **63.33%** | **$0.00144^{**}$** |

*Proposed NeuroGAT outperforms the strongest baseline (Random Forest) by **+32.10%** and XGBoost by **+33.95%** ($p = 0.00144$).*

### 95% Non-Parametric Bootstrap Confidence Intervals ($B = 1,000$)
- **Overall Accuracy**: **82.29%** [95% CI: **81.03%** – **83.46%**]
- **Macro Average F1**: **78.43%** [95% CI: **76.95%** – **79.88%**]
- **Model Calibration (ECE)**: Top-Label ECE = **4.12%** | Multi-Class Brier Score = **0.2415**

---

## 📂 Repository Architecture

```text
├── Section_01_Setup_Configuration.py   # Global configuration, seeds, hyperparameters, paths
├── Section_02_EDA_Visualization.py     # Exploratory data analysis, demographic distributions
├── Section_03_Preprocessing.py         # 3D MRI skull-stripping, resampling, clinical normalization
├── Section_04_Dataset_DataLoader.py    # Stratified 5-fold patient data splitters & loaders
├── Section_04B_Feature_Extraction.py   # 3D DenseNet + PyRadiomics feature extraction pipeline
├── Section_05_Model_Architecture.py    # NeuroGAT model, GATConv, multi-scale graph builder
├── Section_06_Training_Engine.py       # Dual-task trainer, Logit-Adjusted Focal Loss, DropEdge
├── Section_07_Evaluation_Metrics.py    # ROC-AUC, DeLong tests, Bootstrap CI, ECE, Fairness
├── Section_08_Ablation_Study.py        # 11-point ablation suite (modality, topology, loss)
├── Section_09_Explainable_AI.py        # Population attention heatmaps, Grad-CAM, SHAP, LIME
├── Section_10_ML_Baselines.py          # Fair benchmark suite (SVM, RF, MLP, XGBoost)
├── Section_11_Clinical_Inference_App.py# Gradio clinical web application & batch CSV processor
├── run_pipeline.py                     # Master command-line orchestrator
├── verify_environment.py               # Hardware, CUDA, and environment health diagnostic
└── requirements.txt                    # Pinned Python package dependencies
```

---

## ⚡ Quick Start

### 1. Environment Setup
```bash
# Clone repository
git clone https://github.com/your-username/neurogat-adni.git
cd neurogat-adni

# Install dependencies
pip install -r requirements.txt

# Run hardware & environment diagnostic
python verify_environment.py
```

### 2. Running Pipeline Stages
Use the unified `run_pipeline.py` orchestrator:

```bash
# 1. Train NeuroGAT with 5-Fold Cross-Validation
python run_pipeline.py --mode train --epochs 150

# 2. Generate Master Evaluation (ROC, DeLong, 95% CI, Calibration, Fairness)
python run_pipeline.py --mode evaluate

# 3. Run Fair Traditional ML Baselines
python run_pipeline.py --mode baselines

# 4. Run Comprehensive 11-Point Ablation Study
python run_pipeline.py --mode ablation

# 5. Generate Explainable AI Visualizations (Attention, Saliency, SHAP)
python run_pipeline.py --mode xai

# 6. Launch Interactive Clinical Decision Support Web Application
python run_pipeline.py --mode app
```

---

## 🩺 Clinical Decision Support App (Gradio)

NeuroGAT includes an interactive clinical web interface allowing neurologists to:
- Input patient demographic & cognitive metrics (MMSE, CDRSB, Logical Memory).
- Upload pre-extracted 3D MRI representations.
- Dynamically project the patient into the ADNI population graph cohort.
- View real-time diagnostic probabilities, continuous cognitive deficit regression, and 24-month progression hazard strata.

---

## 📚 Academic Citations & References

If you find this codebase or methodology helpful in your research, please cite:

```bibtex
@inproceedings{menon2020long,
  title={Long-tail learning via logit adjustment},
  author={Menon, Aditya Krishna and Jayasumana, Sadeep and Rawat, Ankit Singh and Jain, Himanshu and Veit, Andreas and Kumar, Sanjiv},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  volume={33},
  pages={21994--22004},
  year={2020}
}

@inproceedings{cui2019class,
  title={Class-balanced loss based on effective number of samples},
  author={Cui, Yin and Jia, Menglin and Lin, Tsung-Yi and Song, Yang and Belongie, Serge},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={9268--9277},
  year={2019}
}

@inproceedings{rong2020dropedge,
  title={DropEdge: Towards Deep Graph Convolutional Networks on Large Graphs},
  author={Rong, Yu and Huang, Wenbing and Xu, Tingyang and Huang, Junzhou},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2020}
}

@article{delong1988comparing,
  title={Comparing the areas under two or more correlated receiver operating characteristic curves: a nonparametric approach},
  author={DeLong, Elizabeth R and DeLong, David M and Clarke-Pearson, Daniel L},
  journal={Biometrics},
  pages={837--845},
  year={1988}
}
```

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
