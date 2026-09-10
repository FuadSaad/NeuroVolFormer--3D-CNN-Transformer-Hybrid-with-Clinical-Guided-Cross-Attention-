#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          SECTION 1: SETUP & CONFIGURATION - NeuroVolFormer                 ║
║  3D CNN-Transformer Hybrid with Clinical-Guided Cross-Attention            ║
║  for Multi-Class Alzheimer's Disease Classification                        ║
║                                                                            ║
║  Author: Md Fuad Hossain Saad                                              ║
║  Dataset: ADNI (Alzheimer's Disease Neuroimaging Initiative)                ║
║  GPU: Kaggle T4 × 2                                                        ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ═══════════════════════════════════════════════════════════════════
# Uncomment if running directly in a Kaggle notebook cell:
# !pip install -q monai nibabel torchio grad-cam shap gradio torch_geometric
# !pip install -q git+https://github.com/AIM-Harvard/pyradiomics.git

# ═══════════════════════════════════════════════════════════════════
# 1.2 All Imports
# ═══════════════════════════════════════════════════════════════════
import os
import sys

# Ensure UTF-8 output on Windows terminals
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import gc
import glob
import json
import time
import random
import warnings
import itertools
from pathlib import Path
from datetime import datetime
from collections import Counter, OrderedDict
from typing import Dict, List, Tuple, Optional, Any, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from PIL import Image

# ── Scientific Computing ──
from scipy import ndimage, stats
from scipy.ndimage import zoom
from skimage import filters, morphology, measure

# ── Deep Learning (PyTorch) ──
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR

# ── Graph Neural Networks ──
try:
    import torch_geometric
    from torch_geometric.data import Data
    from torch_geometric.nn import GATConv
except ImportError:
    if os.path.exists('/kaggle'):
        try:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "torch_geometric"])
            import torch_geometric
            from torch_geometric.data import Data
            from torch_geometric.nn import GATConv
        except Exception:
            torch_geometric = None
    else:
        torch_geometric = None
        Data = object
        GATConv = object

try:
    import nibabel as nib
except ImportError:
    if os.path.exists('/kaggle'):
        try:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "nibabel"])
            import nibabel as nib
        except Exception:
            nib = None
    else:
        nib = None

# ── Scikit-Learn ──
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, auc, confusion_matrix,
    classification_report, precision_recall_curve, average_precision_score,
    cohen_kappa_score
)
from sklearn.manifold import TSNE

# ── Progress Bars & Math ──
import math
from tqdm import tqdm

# ── Suppress Warnings ──
warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════════════════
# 1.2.5 Dynamic Path Resolution (For Robust Kaggle/Local Execution)
# ═══════════════════════════════════════════════════════════════════

def find_kaggle_path(patterns: Union[str, List[str]], is_dir: bool = False, default: str = "") -> str:
    """
    Robust dynamic path locator for Kaggle and local environments.
    Supports multiple pattern aliases, handles Kaggle slugs, spaces, dashes, commas,
    and performs recursive detection in /kaggle/input and current directory.
    """
    if isinstance(patterns, str):
        pat_list = [patterns]
    else:
        pat_list = list(patterns)

    # 1. If default exists on disk right now, use it directly
    if default and os.path.exists(default):
        return default.replace('\\', '/')

    def _clean(s: str) -> str:
        return s.lower().replace(' ', '').replace('_', '').replace('-', '').replace(',', '').replace('.', '')

    clean_pats = [_clean(p) for p in pat_list]

    # 2. Check local directory candidates
    for base in ['.', '..', 'data']:
        if os.path.exists(base):
            try:
                for entry in os.listdir(base):
                    full_p = os.path.join(base, entry)
                    if (is_dir and os.path.isdir(full_p)) or (not is_dir and os.path.isfile(full_p)):
                        clean_entry = _clean(entry)
                        for cp in clean_pats:
                            if cp in clean_entry or clean_entry in cp:
                                return os.path.abspath(full_p).replace('\\', '/')
            except Exception:
                pass

    # 3. If running on Kaggle, search /kaggle/input thoroughly
    search_dir = '/kaggle/input'
    if os.path.exists(search_dir):
        # 3a. Exact or substring match in targets
        for root, dirs, files in os.walk(search_dir):
            targets = dirs if is_dir else files
            for t in targets:
                clean_t = _clean(t)
                for cp in clean_pats:
                    if cp == clean_t or (len(cp) >= 3 and (cp in clean_t or clean_t in cp)):
                        return os.path.join(root, t).replace('\\', '/')

        # 3b. For directories, check if root folder itself is the dataset
        if is_dir:
            for root, dirs, files in os.walk(search_dir):
                clean_root = _clean(os.path.basename(root))
                for cp in clean_pats:
                    if cp == clean_root or (len(cp) >= 3 and cp in clean_root):
                        return root.replace('\\', '/')

    return default.replace('\\', '/')


# ═══════════════════════════════════════════════════════════════════
# 1.3 Configuration
# ═══════════════════════════════════════════════════════════════════

class Config:
    """Central configuration for the entire NeuroVolFormer pipeline."""

    # ── Project ──
    PROJECT_NAME = "NeuroVolFormer_3D"
    VERSION = "2.0.0"

    # ── Random Seed ──
    SEED = 42

    # ── Class Definitions ──
    NUM_CLASSES = 4
    CLASS_NAMES = ['AD', 'CN', 'EMCI', 'LMCI']
    CLASS_TO_IDX = {'AD': 0, 'CN': 1, 'EMCI': 2, 'LMCI': 3}
    IDX_TO_CLASS = {0: 'AD', 1: 'CN', 2: 'EMCI', 3: 'LMCI'}
    CLASS_COLORS = {'AD': '#e74c3c', 'CN': '#2ecc71', 'EMCI': '#f39c12', 'LMCI': '#9b59b6'}
    CLASS_COLORS_LIST = ['#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']

    # ── Input Dimensions ──
    INPUT_SIZE = (128, 128, 128)  # 3D volume size
    IN_CHANNELS = 1               # Grayscale MRI

    # ── 3D CNN Encoder (NeuroVolFormer Spatial Backbone) ──
    CNN_CHANNELS = [32, 64, 128, 256] # Multi-stage hierarchical 3D feature representation
    CNN_DROPOUT = 0.2                 # Spatial dropout for CNN stages

    # ── Transformer & Cross-Attention (NeuroVolFormer Hybrid) ──
    PATCH_SIZE = (16, 16, 16)         # 3D volumetric tokenization patch size
    D_MODEL = 256                     # Latent embedding dimension
    N_HEADS = 4                       # Multi-head attention heads
    N_LAYERS = 4                      # Transformer encoder depth
    FFN_DIM = 512                     # Feed-forward hidden dimension
    TRANSFORMER_DROPOUT = 0.1         # Self/Cross-attention dropout

    # ── 3D Feature Extraction (Pre-trained + Radiomics) ──
    DEEP_FEATURE_DIM = 1024       # Dimension from 3D MONAI DenseNet121
    PCA_DIM = 64                  # Preserves 64 principal components of 3D spatial variance
    RADIOMICS_FEATURE_DIM = 68    # Standard PyRadiomics texture feature count

    # ── Graph Neural Network (NeuroGAT A* Edition) ──
    KNN_K = 5                     # Meso-scale neighborhood
    KNN_K_LIST = [3, 5, 10]       # Multi-scale neighborhood (Micro, Meso, Macro)
    GAT_HIDDEN_DIM = 128          # Hidden dimensions in GATv2 layers
    GAT_HEADS = 4                 # Multi-head attention heads in GATv2
    GAT_DROPOUT = 0.15            # Calibrated dropout for graph representation (relaxed from 0.35 to prevent underfitting)
    L2_REGULARIZATION = 1e-4      # Weight decay for GAT (relaxed from 1e-3)
    AUX_COG_WEIGHT = 0.02         # Balanced auxiliary cognitive loss weight (prevents 1D regression from dominating 4-class manifold)

    # ── Cohort Participant-Level Independence ──
    ONE_SCAN_PER_SUBJECT = True   # The one-scan-per-subject protocol eliminates within-subject repeated-measure dependence.

    # ── Clinical Demographics & Feature Schema Provenance (Zero Target Leakage) ──
    CLINICAL_DEMOGRAPHICS = ['AGE', 'EDUCATION', 'GENDER', 'GDS_TOTAL', 'BP_Systolic', 'Pulse']
    DIAGNOSTIC_PROXIES = ['CDRSB', 'MMSE', 'LogMem_Delayed', 'LogMem_Immediate']
    INCLUDE_DIAGNOSTIC_PROXIES = False  # False: eliminates circular diagnostic proxy leakage
    CLINICAL_FEATURES = CLINICAL_DEMOGRAPHICS if not INCLUDE_DIAGNOSTIC_PROXIES else (DIAGNOSTIC_PROXIES + CLINICAL_DEMOGRAPHICS)
    CLINICAL_DIM = len(CLINICAL_FEATURES)

    FEATURE_SCHEMA = {
        'deep_mri': '3D DenseNet-121 / Native 3D CNN (1024-D -> PCA 64-D)',
        'radiomics': 'PyRadiomics Handcrafted Morphological & Texture Features (68-D)',
        'demographics': CLINICAL_DEMOGRAPHICS
    }
    FORBIDDEN_GRAPH_VARIABLES = {'MMSE', 'CDRSB', 'LogMem_Delayed', 'LogMem_Immediate', 'DX', 'DX_bl'}

    # ── Classifier Head ──
    CLASSIFIER_DROPOUT = 0.20     # Calibrated dropout (relaxed from 0.45 to prevent underfitting on 472 train nodes)

    # ── Training Hyperparameters (GNN is Transductive Full-Batch) ──
    BATCH_SIZE = 1                # GNN processes the entire graph as a single batch
    GRAD_ACCUM_STEPS = 1          # Gradient accumulation steps
    EPOCHS = 200                  # Maximum training epochs
    PATIENCE = 40                 # Early stopping patience
    MONITOR_METRIC = 'val_loss'   # Monitor validation loss strictly

    # ── Optimizer & Anti-Overfitting Learning Rate Scheduler ──
    LEARNING_RATE = 5e-4          # Optimal learning rate for AdamW
    WEIGHT_DECAY = 1e-4           # Calibrated weight decay (1e-4 avoids stifling capacity)
    BETAS = (0.9, 0.999)
    LR_SCHEDULER_TYPE = 'CosineAnnealingWarmRestarts'  # Cyclic exploration prevents premature stagnation
    LR_PLATEAU_FACTOR = 0.7       # Smooth decay if ReduceLROnPlateau selected
    LR_PLATEAU_PATIENCE = 12      # Sufficient patience before reducing LR
    WARMUP_EPOCHS = 10
    T_0 = 25                      # Period for CosineAnnealingWarmRestarts
    T_MULT = 2

    # ── Loss & Cost-Sensitive Learning (Targeting 85-88% with Calibrated LMCI F1) ──
    LABEL_SMOOTHING = 0.05        # Label smoothing for focal loss
    FOCAL_GAMMA = 1.0             # Balanced focusing parameter (1.0 prevents over-suppression of gradients)
    USE_FOCAL_LOSS = True         # True=ClassBalancedFocalLoss
    CUSTOM_CLASS_WEIGHTS = [1.0, 1.0, 1.0, 1.0]  # Neutral base weights
    USE_CUSTOM_CLASS_WEIGHTS = False             # False: rely on mathematically pure Effective Number of Samples (Cui et al., CVPR 2019)
    USE_COST_SENSITIVE_LOSS = False              # False: avoid artificial bias that induces high LMCI false alarms
    COST_EMCI_LMCI_PENALTY = 1.0                 # Neutral penalty
    FAIR_BASELINE_MODE = True                    # Symmetrical evaluation: Baselines & NeuroGAT receive identical features

    # ── Logit Adjustment (NeurIPS 2020) & Effective Number of Samples (CVPR 2019) ──
    USE_LOGIT_ADJUSTMENT = False                 # Disabled when Effective Samples is active to prevent over-adjustment
    LOGIT_ADJUST_TAU = 0.1                       # Mild temperature scaling if enabled
    USE_EFFECTIVE_NUM_SAMPLES = True             # Information-theoretic sample weighting (Cui et al., CVPR 2019)
    EFFECTIVE_NUM_BETA = 0.999                   # Beta=0.999 yields smooth ~1.56x LMCI weight perfectly fitting 2:1 ratio

    # ── Graph Regularization & Publication Rigor (Q1 Upgrades) ──
    USE_DROPEDGE = True                          # Graph data augmentation & over-smoothing prevention
    DROPEDGE_RATE = 0.05                         # Calibrated DropEdge (relaxed from 0.15 to preserve small-graph connectivity)
    BOOTSTRAP_ITERATIONS = 1000                  # 1,000 resamplings for 95% Confidence Intervals
    MCNEMAR_CORRECTION = 'holm-bonferroni'       # Stepwise family-wise error rate control
    GENERATE_LATEX_TABLES = True                 # Export camera-ready booktabs .tex tables

    # ── Data Splitting ──
    N_FOLDS = 5
    TEST_SIZE = 0.2

    # ── DataLoader ──
    NUM_WORKERS = 4
    PIN_MEMORY = True

    # ── Checkpointing ──
    CHECKPOINT_INTERVAL = 5  # Save every N epochs

    # ── Kaggle Dataset Paths (Auto-resolves across all Kaggle naming conventions) ──
    DATA_PATHS = {
        'AD': find_kaggle_path(
            ['MPRAGE ad', 'ad-mprage', 'AD MPRAGE', 'MPRAGE_ad'],
            is_dir=True,
            default='/kaggle/input/ad-mprage/MPRAGE ad'
        ),
        'CN': find_kaggle_path(
            ['MPRAGE cn', 'cn-mprage', 'CN MPRAGE', 'MPRAGE_cn'],
            is_dir=True,
            default='/kaggle/input/cn-mprage/MPRAGE cn'
        ),
        'EMCI': find_kaggle_path(
            ['MPRAGE EMCI', 'emci-mprage', 'EMCI MPRAGE', 'MPRAGE_EMCI'],
            is_dir=True,
            default='/kaggle/input/emci-mprage/MPRAGE EMCI'
        ),
        'LMCI': find_kaggle_path(
            ['MPRAGE LMCI', 'lmci-mprage', 'LMCI MPRAGE', 'MPRAGE_LMCI'],
            is_dir=True,
            default='/kaggle/input/lmci-mprage/MPRAGE LMCI'
        ),
    }
    CSV_PATHS = {
        'AD': find_kaggle_path(
            ['MPRAGE_ad', 'MPRAGE_ad_7_01_2026.csv', 'ad_mprage.csv', 'ad.csv'],
            is_dir=False,
            default='/kaggle/input/ad-mprage/MPRAGE_ad_7_01_2026.csv'
        ),
        'CN': find_kaggle_path(
            ['MPRAGE_cn', 'MPRAGE_cn_7_06_2026.csv', 'cn_mprage.csv', 'cn.csv'],
            is_dir=False,
            default='/kaggle/input/cn-mprage/MPRAGE_cn_7_06_2026.csv'
        ),
        'EMCI': find_kaggle_path(
            ['MPRAGE_EMCI', 'MPRAGE_EMCI_7_01_2026.csv', 'emci_mprage.csv', 'emci.csv'],
            is_dir=False,
            default='/kaggle/input/emci-mprage/MPRAGE_EMCI_7_01_2026.csv'
        ),
        'LMCI': find_kaggle_path(
            ['MPRAGE_LMCI', 'MPRAGE_LMCI_6_30_2026.csv', 'lmci_mprage.csv', 'lmci.csv'],
            is_dir=False,
            default='/kaggle/input/lmci-mprage/MPRAGE_LMCI_6_30_2026.csv'
        ),
    }
    CLINICAL_CSV = find_kaggle_path(
        ['adnicorefeatures', 'adnicorefeatures.csv', 'adniclinical', 'ADNI,clinical-Data', 'clinical'],
        is_dir=False,
        default='/kaggle/input/adniclinical-data/adnicorefeatures.csv'
    )

    # ── Output Paths (Auto-detect Kaggle vs Local environment) ──
    IS_KAGGLE = os.path.exists('/kaggle')
    BASE_OUTPUT = '/kaggle/working' if IS_KAGGLE else '.'
    CHECKPOINT_DIR = os.path.join(BASE_OUTPUT, 'checkpoints').replace('\\', '/')
    OUTPUT_DIR = os.path.join(BASE_OUTPUT, 'outputs').replace('\\', '/')
    PREPROCESSED_DIR = os.path.join(BASE_OUTPUT, 'preprocessed').replace('\\', '/')
    FIGURES_DIR = os.path.join(BASE_OUTPUT, 'outputs/figures').replace('\\', '/')

    # ── Pipeline Execution & Clean-Slate Control ──
    CLEAN_OUTPUTS_ON_START = True          # Auto-clean previous run checkpoints, figures & metrics for fresh run
    PRESERVE_EXTRACTED_FEATURES = True     # True: preserve node_features.npy, node_labels.npy, splits.pt (~3h runtime saved)
                                           # False: full nuclear purge of all preprocessed data as well

    # ── Visualization ──
    FIG_DPI = 300
    FIG_FORMAT = 'png'
    SEABORN_STYLE = 'whitegrid'
    FONT_SIZE = 12


# ═══════════════════════════════════════════════════════════════════
# 1.4 Seed Everything
# ═══════════════════════════════════════════════════════════════════

def seed_everything(seed: int = Config.SEED) -> None:
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"✅ Random seed set to {seed}")


# ═══════════════════════════════════════════════════════════════════
# 1.5 Device Setup
# ═══════════════════════════════════════════════════════════════════

def setup_device() -> torch.device:
    """Detect and configure GPU devices."""
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        print(f"✅ CUDA Available: {torch.cuda.is_available()}")
        print(f"✅ Number of GPUs: {n_gpus}")
        for i in range(n_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_mem = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"   GPU {i}: {gpu_name} ({gpu_mem:.1f} GB)")
        device = torch.device('cuda')
    else:
        print("⚠️  No GPU detected, using CPU (training will be very slow!)")
        device = torch.device('cpu')
    return device


# ═══════════════════════════════════════════════════════════════════
# 1.6 Create Output Directories
# ═══════════════════════════════════════════════════════════════════

def create_directories() -> None:
    """Create all necessary output directories."""
    dirs = [
        Config.CHECKPOINT_DIR,
        Config.OUTPUT_DIR,
        Config.PREPROCESSED_DIR,
        Config.FIGURES_DIR,
        os.path.join(Config.OUTPUT_DIR, 'xai'),
        os.path.join(Config.OUTPUT_DIR, 'models'),
        os.path.join(Config.OUTPUT_DIR, 'results'),
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print(f"✅ Output directories created")


# ═══════════════════════════════════════════════════════════════════
# 1.6.5 Clean Previous Run Artifacts (Fresh Run Initializer)
# ═══════════════════════════════════════════════════════════════════

def clean_previous_run_artifacts(preserve_extracted_features: bool = True) -> None:
    """
    Purge previous run checkpoints, figures, logs, and evaluation metrics to ensure
    a completely clean-slate execution.
    
    Args:
        preserve_extracted_features (bool): If True, preserves node_features.npy, 
            node_labels.npy, processed_files.csv, radiomics_features.csv, and splits.pt 
            to prevent re-running the 3-4 hour 3D feature extraction step.
            If False, performs a full nuclear purge of everything.
    """
    import shutil

    print("\n" + "=" * 70)
    print("  🧹 Clean-Slate Execution Initializer: Purging Old Run Artifacts")
    print("=" * 70)

    # 1. Flush GPU & System RAM
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
        print("  🧹 Flushed GPU CUDA memory cache & Python garbage collection.")

    cleaned_items = []

    # 2. Checkpoints directory
    if os.path.exists(Config.CHECKPOINT_DIR):
        try:
            shutil.rmtree(Config.CHECKPOINT_DIR)
            cleaned_items.append(f"Wiped checkpoints directory: {Config.CHECKPOINT_DIR}")
        except Exception as e:
            print(f"  ⚠️ Could not remove {Config.CHECKPOINT_DIR}: {e}")

    # 3. Figures directory
    if os.path.exists(Config.FIGURES_DIR):
        try:
            shutil.rmtree(Config.FIGURES_DIR)
            cleaned_items.append(f"Wiped figures directory: {Config.FIGURES_DIR}")
        except Exception as e:
            print(f"  ⚠️ Could not remove {Config.FIGURES_DIR}: {e}")

    # 4. Outputs subdirectories: xai, models, results
    for sub in ['xai', 'models', 'results']:
        sub_path = os.path.join(Config.OUTPUT_DIR, sub).replace('\\', '/')
        if os.path.exists(sub_path):
            try:
                shutil.rmtree(sub_path)
                cleaned_items.append(f"Wiped output subdirectory: {sub_path}")
            except Exception:
                pass

    # 5. Stale evaluation metrics, tables, and caches in BASE_OUTPUT and OUTPUT_DIR
    target_patterns = [
        os.path.join(Config.BASE_OUTPUT, '*.pt'),
        os.path.join(Config.BASE_OUTPUT, '*.tex'),
        os.path.join(Config.BASE_OUTPUT, '*.png'),
        os.path.join(Config.OUTPUT_DIR, '*.csv'),
        os.path.join(Config.OUTPUT_DIR, '*.tex'),
        os.path.join(Config.OUTPUT_DIR, '*.pt'),
        os.path.join(Config.OUTPUT_DIR, '*.npy'),
        os.path.join(Config.OUTPUT_DIR, '*.txt'),
    ]
    for pattern in target_patterns:
        for f in glob.glob(pattern):
            fname = os.path.basename(f)
            # Guard against deleting essential preprocessed feature files
            if preserve_extracted_features and ('node_features' in fname or 'splits.pt' in fname or 'node_labels' in fname):
                continue
            try:
                os.remove(f)
                cleaned_items.append(f"Removed stale file: {fname}")
            except Exception:
                pass

    # 6. Preprocessed data handling
    if not preserve_extracted_features and os.path.exists(Config.PREPROCESSED_DIR):
        try:
            shutil.rmtree(Config.PREPROCESSED_DIR)
            cleaned_items.append(f"Wiped preprocessed features directory: {Config.PREPROCESSED_DIR}")
        except Exception:
            pass
    elif preserve_extracted_features and os.path.exists(Config.PREPROCESSED_DIR):
        features_found = [f for f in os.listdir(Config.PREPROCESSED_DIR) if f.endswith(('.npy', '.csv', '.pt'))]
        if features_found:
            print(f"  💾 Preserved {len(features_found)} preprocessed feature file(s) in {Config.PREPROCESSED_DIR}:")
            for feat in features_found:
                print(f"     - {feat}")
            print("     (Saves ~3-4 hours of 3D feature extraction runtime!)")

    print(f"  ✅ Cleanup complete! Purged {len(cleaned_items)} old artifact(s)/director(y/ies).")
    print("  🚀 Fresh environment initialized for new run.")
    print("=" * 70 + "\n")


# ═══════════════════════════════════════════════════════════════════
# 1.7 Matplotlib & Seaborn Configuration
# ═══════════════════════════════════════════════════════════════════

def setup_plotting() -> None:
    """Configure matplotlib and seaborn for publication-quality plots."""
    sns.set_style(Config.SEABORN_STYLE)
    plt.rcParams.update({
        'font.size': Config.FONT_SIZE,
        'font.family': 'sans-serif',
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
        'figure.titlesize': 18,
        'figure.dpi': Config.FIG_DPI,
        'savefig.dpi': Config.FIG_DPI,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
    })
    print("✅ Plotting configuration set")


# ═══════════════════════════════════════════════════════════════════
# 1.8 Utility Functions
# ═══════════════════════════════════════════════════════════════════

def print_config(config: Config) -> None:
    """Print all configuration parameters in a formatted table."""
    print("\n" + "=" * 70)
    print(f"  {config.PROJECT_NAME} v{config.VERSION} - Configuration")
    print("=" * 70)
    categories = {
        'Model': ['NUM_CLASSES', 'INPUT_SIZE', 'IN_CHANNELS'],
        'CNN Encoder': ['CNN_CHANNELS', 'CNN_DROPOUT'],
        'Transformer': ['PATCH_SIZE', 'D_MODEL', 'N_HEADS', 'N_LAYERS', 'FFN_DIM', 'TRANSFORMER_DROPOUT'],
        'Population Graph': ['KNN_K_LIST', 'GAT_HIDDEN_DIM', 'GAT_HEADS', 'GAT_DROPOUT', 'USE_DROPEDGE'],
        'Clinical': ['CLINICAL_DIM', 'INCLUDE_DIAGNOSTIC_PROXIES'],
        'Training': ['BATCH_SIZE', 'GRAD_ACCUM_STEPS', 'EPOCHS', 'PATIENCE', 'MONITOR_METRIC', 'LEARNING_RATE', 'WEIGHT_DECAY', 'LR_SCHEDULER_TYPE'],
        'Loss & Balancing': ['LABEL_SMOOTHING', 'FOCAL_GAMMA', 'USE_LOGIT_ADJUSTMENT', 'USE_EFFECTIVE_NUM_SAMPLES', 'COST_EMCI_LMCI_PENALTY'],
        'Data': ['N_FOLDS', 'TEST_SIZE', 'SEED', 'ONE_SCAN_PER_SUBJECT'],
        'Pipeline & Clean-Slate': ['CLEAN_OUTPUTS_ON_START', 'PRESERVE_EXTRACTED_FEATURES'],
    }
    # Enforce Anti-Leakage Feature Provenance Check
    if not getattr(config, 'INCLUDE_DIAGNOSTIC_PROXIES', False):
        forbidden = set(config.CLINICAL_FEATURES) & config.FORBIDDEN_GRAPH_VARIABLES
        if forbidden:
            raise ValueError(f"CRITICAL LEAKAGE DETECTED: Forbidden variables in clinical input: {forbidden}")
        print("🛡️  Feature Provenance Verified: Zero Diagnostic Proxies in Clinical Manifold.")

    for cat_name, params in categories.items():
        print(f"\n  [{cat_name}]")
        for p in params:
            val = getattr(config, p, 'N/A')
            print(f"    {p:<25s} = {val}")
    print("\n" + "=" * 70)


def get_memory_usage() -> str:
    """Get current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        return f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
    return "No GPU"


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def verify_dataset_paths() -> None:
    """Verifies that all dataset directories and CSV files exist and prints a diagnostic report."""
    print("\n" + "=" * 70)
    print("  🔍 Kaggle Dataset Path Resolution & Diagnostic Report")
    print("=" * 70)

    all_found = True
    print("\n  [3D MRI Volume Directories]")
    for group, path in Config.DATA_PATHS.items():
        exists = os.path.exists(path)
        status = "✅ FOUND" if exists else "⚠️ NOT FOUND"
        if not exists:
            all_found = False
        print(f"    {group:<6s}: {path} [{status}]")

    print("\n  [Metadata CSV Files]")
    for group, path in Config.CSV_PATHS.items():
        exists = os.path.exists(path)
        status = "✅ FOUND" if exists else "⚠️ NOT FOUND"
        if not exists:
            all_found = False
        print(f"    {group:<6s}: {path} [{status}]")

    clin_exists = os.path.exists(Config.CLINICAL_CSV)
    clin_status = "✅ FOUND" if clin_exists else "⚠️ NOT FOUND"
    if not clin_exists:
        all_found = False
    print(f"\n  Clinical CSV: {Config.CLINICAL_CSV} [{clin_status}]")

    print("-" * 70)
    if all_found:
        print("🎉 ALL DATASET PATHS VERIFIED & ACCESSIBLE!")
    elif not os.path.exists('/kaggle/input'):
        print("💻 Environment Detected: Local PC (Windows).")
        print("ℹ️  Metadata CSV files are verified above.")
        print("ℹ️  3D MRI volume directories point to canonical Kaggle paths (/kaggle/input/...).")
        print("    When executed on Kaggle, the datasets mounted in /kaggle/input will be auto-detected.")
    else:
        print("ℹ️ Note: Paths not immediately found will be searched dynamically under /kaggle/input at runtime.")
    print("=" * 70 + "\n")


# ═══════════════════════════════════════════════════════════════════
# 1.9 Initialize Everything
# ═══════════════════════════════════════════════════════════════════

print("╔══════════════════════════════════════════════════════════════╗")
print("║     NeuroGAT 3D: Multimodal Graph Attention Network         ║")
print("║     Alzheimer's Disease Classification from MPRAGE MRI      ║")
print("╚══════════════════════════════════════════════════════════════╝")
print()

seed_everything()
DEVICE = setup_device()

# Clean-slate execution: purge previous artifacts if configured
if Config.CLEAN_OUTPUTS_ON_START:
    clean_previous_run_artifacts(preserve_extracted_features=Config.PRESERVE_EXTRACTED_FEATURES)

create_directories()
setup_plotting()
print_config(Config)
verify_dataset_paths()

print(f"\n✅ Section 1 Complete - All systems initialized!")
print(f"   PyTorch version: {torch.__version__}")
print(f"   Device: {DEVICE}")
print(f"   {get_memory_usage()}")

