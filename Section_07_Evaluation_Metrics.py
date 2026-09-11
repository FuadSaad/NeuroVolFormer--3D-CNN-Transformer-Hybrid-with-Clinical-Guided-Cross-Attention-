#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          SECTION 7: EVALUATION METRICS & VISUALIZATION (NeuroGAT)            ║
║  A* / Q1 Publication Figures:                                                ║
║    1. Confusion Matrix (Normalized & Raw Counts)                             ║
║    2. Multi-Class ROC-AUC Analysis                                           ║
║    3. DeLong's Non-Parametric Statistical Significance Tests (p-values)      ║
║    4. 24-Month MCI Conversion Prognosis Risk Distribution                    ║
║    5. High-Resolution t-SNE Manifold Node Embeddings                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import glob
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from scipy import stats
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    accuracy_score, precision_recall_fscore_support
)
from sklearn.manifold import TSNE

if 'Config' not in globals() and 'Config' not in locals():
    try:
        from Section_01_Setup_Configuration import Config
    except (ImportError, ModuleNotFoundError):
        class Config:
            pass

_defaults = {
    'SEED': 42,
    'NUM_CLASSES': 4,
    'CLASS_NAMES': ['AD', 'CN', 'EMCI', 'LMCI'],
    'CLASS_TO_IDX': {'AD': 0, 'CN': 1, 'EMCI': 2, 'LMCI': 3},
    'IDX_TO_CLASS': {0: 'AD', 1: 'CN', 2: 'EMCI', 3: 'LMCI'},
    'CLASS_COLORS': {'AD': '#e74c3c', 'CN': '#2ecc71', 'EMCI': '#3498db', 'LMCI': '#e67e22'},
    'CLASS_COLORS_LIST': ['#e74c3c', '#2ecc71', '#3498db', '#e67e22'],
    'KNN_K': 5,
    'KNN_K_LIST': [3, 5, 10],
    'GAT_HIDDEN_DIM': 128,
    'GAT_HEADS': 4,
    'OUTPUT_DIR': '/kaggle/working/outputs' if os.path.exists('/kaggle') else './outputs',
    'CHECKPOINT_DIR': '/kaggle/working/checkpoints' if os.path.exists('/kaggle') else './checkpoints',
    'FIGURES_DIR': '/kaggle/working/outputs/figures' if os.path.exists('/kaggle') else './outputs/figures',
    'PREPROCESSED_DIR': '/kaggle/working/preprocessed' if os.path.exists('/kaggle') else './preprocessed',
}
for _k, _v in _defaults.items():
    if not hasattr(Config, _k):
        setattr(Config, _k, _v)

try:
    from Section_05_Model_Architecture import NeuroGAT, build_population_graph, build_multiscale_population_graph
except (ImportError, ModuleNotFoundError):
    pass

# ═══════════════════════════════════════════════════════════════════
# 7.1 DeLong's Non-Parametric ROC Statistical Significance Test
# ═══════════════════════════════════════════════════════════════════

def compute_midrank(x: np.ndarray) -> np.ndarray:
    """Computes midranks for DeLong's algorithm."""
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2

def delong_roc_variance_and_covariance(ground_truth: np.ndarray, preds_a: np.ndarray, preds_b: np.ndarray) -> Tuple[float, float, float, float, float]:
    """
    Computes exact DeLong AUCs, individual variances, and paired covariance S_12
    according to DeLong, DeLong, & Clarke-Pearson (Biometrics 1988, 44(3):837-845).
    """
    pos_idx = np.where(ground_truth == 1)[0]
    neg_idx = np.where(ground_truth == 0)[0]
    m = len(pos_idx)
    n = len(neg_idx)

    if m == 0 or n == 0:
        return 0.5, 0.5, 1e-6, 1e-6, 0.0

    def _get_v_components(preds):
        pos_preds = preds[pos_idx]
        neg_preds = preds[neg_idx]
        all_preds = np.concatenate([pos_preds, neg_preds])
        all_ranks = compute_midrank(all_preds)
        pos_ranks = all_ranks[:m]
        neg_ranks = all_ranks[m:]
        auc_val = (pos_ranks.sum() - m * (m + 1) / 2.0) / (m * n)
        v10 = (pos_ranks - compute_midrank(pos_preds)) / n
        v01 = 1.0 - (neg_ranks - compute_midrank(neg_preds)) / m
        return auc_val, v10, v01

    auc_a, v10_a, v01_a = _get_v_components(preds_a)
    auc_b, v10_b, v01_b = _get_v_components(preds_b)

    # Covariances for positive cases (m) and negative cases (n)
    if m > 1:
        s10_a = np.var(v10_a, ddof=1)
        s10_b = np.var(v10_b, ddof=1)
        cov_10 = np.cov(v10_a, v10_b, ddof=1)[0, 1]
    else:
        s10_a, s10_b, cov_10 = 0.0, 0.0, 0.0

    if n > 1:
        s01_a = np.var(v01_a, ddof=1)
        s01_b = np.var(v01_b, ddof=1)
        cov_01 = np.cov(v01_a, v01_b, ddof=1)[0, 1]
    else:
        s01_a, s01_b, cov_01 = 0.0, 0.0, 0.0

    var_a = (s10_a / m) + (s01_a / n)
    var_b = (s10_b / m) + (s01_b / n)
    cov_ab = (cov_10 / m) + (cov_01 / n)

    return float(auc_a), float(auc_b), float(var_a), float(var_b), float(cov_ab)


def delong_roc_test(ground_truth: np.ndarray, preds_a: np.ndarray, preds_b: np.ndarray) -> Tuple[float, float, float, float]:
    """
    True Paired DeLong Test with exact bivariate covariance S_12 (DeLong et al., 1988).
    Var(theta_A - theta_B) = Var(theta_A) + Var(theta_B) - 2 * Cov(theta_A, theta_B).
    """
    auc_a, auc_b, var_a, var_b, cov_ab = delong_roc_variance_and_covariance(ground_truth, preds_a, preds_b)
    paired_var = var_a + var_b - 2.0 * cov_ab
    sigma = np.sqrt(max(paired_var, 1e-12))
    diff = auc_a - auc_b
    z_stat = diff / sigma
    p_val = 2.0 * (1.0 - stats.norm.cdf(abs(z_stat)))
    return auc_a, auc_b, float(z_stat), float(p_val)

def _load_or_generate_independent_baseline_probs(y_true: np.ndarray, num_classes: int = 4) -> Optional[np.ndarray]:
    """
    Loads independently trained baseline test predictions generated by Section 10 (ML Baselines).
    Evaluation module strictly evaluates and does not perform dynamic baseline retraining.
    """
    n_samples = len(y_true)

    # Check potential serialized locations from Section 10
    candidate_paths = [
        os.path.join(Config.OUTPUT_DIR, 'baseline_predictions', 'random_forest_test_probs.npy'),
        os.path.join(Config.OUTPUT_DIR, 'baseline_test_probs.npy'),
        os.path.join(Config.OUTPUT_DIR, 'baseline_predictions', 'svm_rbf_test_probs.npy'),
    ]

    for p in candidate_paths:
        if os.path.exists(p):
            try:
                base_probs = np.load(p)
                if base_probs.ndim == 2 and base_probs.shape == (n_samples, num_classes):
                    print(f"📦 Loaded independent benchmark predictions from: {p}")
                    return base_probs
            except Exception as e:
                print(f"ℹ️ Could not load baseline from {p}: {e}")

    print("⚠️ Baseline test probabilities not found in outputs/baseline_predictions/ or outputs/baseline_test_probs.npy.")
    print("   Please execute Section 10 (ML Baselines) first to train and serialize independent benchmarks.")
    return None


def holm_bonferroni_correction(p_values: List[float]) -> List[float]:
    """
    Stepwise Holm-Bonferroni FWER adjustment for multiple hypothesis testing.
    Controls family-wise error rate across 4 One-vs-Rest DeLong ROC tests:
      p_(1) <= p_(2) <= ... <= p_(M)
      p_adj_(i) = min(1.0, max_{j<=i} (M - j + 1) * p_(j))
    """
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adj = [0.0] * m
    running_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        val = (m - rank) * p
        running_max = max(running_max, val)
        adj[orig_idx] = min(1.0, running_max)
    return adj


def run_delong_significance_analysis(y_true: np.ndarray, probs_gnn: np.ndarray, output_csv: Optional[str] = None) -> pd.DataFrame:
    """
    Executes DeLong statistical significance tests across all 4 diagnostic classes with
    step-down Holm-Bonferroni correction for multiple hypothesis testing (Q1 Gold Standard).
    """
    print("\n" + "="*70)
    print("  🔬 DeLong's Statistical Significance Test (ROC-AUC Comparison with Holm-Bonferroni)")
    print("="*70)

    n_classes = Config.NUM_CLASSES
    baseline_all = _load_or_generate_independent_baseline_probs(y_true, n_classes)
    if baseline_all is None:
        print("ℹ️ DeLong analysis safely skipped: No independent baseline predictions found to compare against.")
        print("   Run Section 10 (ML Baselines) to generate independent test benchmark predictions.")
        return pd.DataFrame()

    assert len(y_true) == len(probs_gnn) == len(baseline_all), (
        f"Sample size mismatch: y_true ({len(y_true)}), NeuroGAT ({len(probs_gnn)}), Baseline ({len(baseline_all)})"
    )

    raw_results = []
    raw_p_values = []

    for i in range(n_classes):
        class_name = Config.CLASS_NAMES[i]
        bin_true = (y_true == i).astype(int)
        gnn_scores = probs_gnn[:, i]
        baseline_scores = baseline_all[:, i]

        # Verify predictions are not identical clones (Critique 15 & Priority 3)
        if not np.allclose(baseline_scores, baseline_scores[0]):
            assert not np.allclose(gnn_scores, baseline_scores), (
                f"CRITICAL STATISTICAL ERROR: NeuroGAT scores and Baseline scores are identical for class {class_name}! "
                "DeLong paired test requires independent predictions."
            )

        auc_gnn, auc_base, z, p = delong_roc_test(bin_true, gnn_scores, baseline_scores)
        raw_p_values.append(p)
        raw_results.append({
            'Target Class': class_name,
            'NeuroGAT AUROC': auc_gnn,
            'Standard Baseline AUROC': auc_base,
            'DeLong Z-Score': z,
            'Raw p-Value': p
        })

    # Apply Stepwise Holm-Bonferroni Correction across the 4 One-vs-Rest tests
    adj_p_values = holm_bonferroni_correction(raw_p_values)

    results = []
    for i, res in enumerate(raw_results):
        adj_p = adj_p_values[i]
        sig = "*** (p < 0.001)" if adj_p < 0.001 else ("** (p < 0.01)" if adj_p < 0.01 else ("* (p < 0.05)" if adj_p < 0.05 else "NS"))
        results.append({
            'Target Class': res['Target Class'],
            'NeuroGAT AUROC': f"{res['NeuroGAT AUROC']:.4f}",
            'Standard Baseline AUROC': f"{res['Standard Baseline AUROC']:.4f}",
            'DeLong Z-Score': f"{res['DeLong Z-Score']:.3f}",
            'Raw p-Value': f"{res['Raw p-Value']:.4e}",
            'Holm-Adjusted p-Value': f"{adj_p:.4e}",
            'Significance (Holm-adj)': sig
        })
        print(f"   [{res['Target Class']}] NeuroGAT: {res['NeuroGAT AUROC']:.4f} vs Baseline: {res['Standard Baseline AUROC']:.4f} | Z={res['DeLong Z-Score']:.2f}, Raw p={res['Raw p-Value']:.4e}, Holm-adj p={adj_p:.4e} {sig}")

    df = pd.DataFrame(results)
    if output_csv is None:
        output_csv = os.path.join(Config.OUTPUT_DIR, 'delong_significance_test.csv')
    df.to_csv(output_csv, index=False)
    print(f"✅ DeLong significance test (Holm-corrected) saved to {output_csv}\n")
    return df


# ═══════════════════════════════════════════════════════════════════
# 7.2 Confusion Matrix and Multi-Class ROC Curves
# ═══════════════════════════════════════════════════════════════════

def plot_confusion_matrix(all_labels: np.ndarray, all_preds: np.ndarray):
    print("📈 Generating High-Res Publication Confusion Matrix...")
    cm = confusion_matrix(all_labels, all_preds)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # Raw Counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=Config.CLASS_NAMES, yticklabels=Config.CLASS_NAMES,
                annot_kws={'size': 14, 'weight': 'bold'})
    ax1.set_title('NeuroGAT Confusion Matrix (Raw Counts)', fontsize=15, fontweight='bold', pad=15)
    ax1.set_ylabel('Ground Truth', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Predicted Label', fontsize=13, fontweight='bold')

    # Normalized Percentage
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Greens', ax=ax2,
                xticklabels=Config.CLASS_NAMES, yticklabels=Config.CLASS_NAMES,
                annot_kws={'size': 14, 'weight': 'bold'})
    ax2.set_title('NeuroGAT Confusion Matrix (Normalized %)', fontsize=15, fontweight='bold', pad=15)
    ax2.set_ylabel('Ground Truth', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Predicted Label', fontsize=13, fontweight='bold')

    plt.tight_layout()
    os.makedirs(Config.FIGURES_DIR, exist_ok=True)
    plt.savefig(os.path.join(Config.FIGURES_DIR, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_multiclass_roc(all_labels: np.ndarray, all_probs: np.ndarray):
    print("📈 Generating Multi-class ROC-AUC Curve...")
    n_classes = getattr(Config, 'NUM_CLASSES', 4)
    class_names = getattr(Config, 'CLASS_NAMES', ['AD', 'CN', 'EMCI', 'LMCI'])
    colors = getattr(Config, 'CLASS_COLORS_LIST', ['#e74c3c', '#2ecc71', '#3498db', '#e67e22'])
    figures_dir = getattr(Config, 'FIGURES_DIR', '/kaggle/working/outputs/figures')
    os.makedirs(figures_dir, exist_ok=True)

    y_true_bin = np.zeros((len(all_labels), n_classes))
    for i, label in enumerate(all_labels):
        y_true_bin[i, label] = 1

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], all_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(10, 8))

    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2.5,
                 label=f'{class_names[i]} (AUROC = {roc_auc[i]:.3f})')

    # Micro-average AUROC
    fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), all_probs.ravel())
    auc_micro = auc(fpr_micro, tpr_micro)
    plt.plot(fpr_micro, tpr_micro, color='navy', linestyle=':', lw=2.5,
             label=f'Micro-Average (AUROC = {auc_micro:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.7)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=13, fontweight='bold')
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=13, fontweight='bold')
    plt.title('Multi-Class ROC-AUC Trajectories (NeuroGAT A* Edition)', fontsize=15, fontweight='bold', pad=15)
    plt.legend(loc="lower right", prop={'size': 12, 'weight': 'bold'}, frameon=True)
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.savefig(os.path.join(figures_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


# ═══════════════════════════════════════════════════════════════════
# 7.3 24-Month MCI Conversion Prognosis Risk Distribution Plot
# ═══════════════════════════════════════════════════════════════════

def plot_mci_risk_distribution(all_labels: np.ndarray, risk_scores: np.ndarray):
    """
    Plots the distribution of empirical disease severity and transition vulnerability scores
    stratified by diagnostic cohort (Issue 45: empirical staging, not longitudinal hazard).
    """
    print("📈 Generating MCI Transition Vulnerability Distribution...")
    idx_to_class = getattr(Config, 'IDX_TO_CLASS', {0: 'AD', 1: 'CN', 2: 'EMCI', 3: 'LMCI'})
    class_colors = getattr(Config, 'CLASS_COLORS', {'AD': '#e74c3c', 'CN': '#2ecc71', 'EMCI': '#3498db', 'LMCI': '#e67e22'})
    figures_dir = getattr(Config, 'FIGURES_DIR', '/kaggle/working/outputs/figures')
    os.makedirs(figures_dir, exist_ok=True)

    df = pd.DataFrame({
        'Diagnosis': [idx_to_class.get(l, f'Class_{l}') for l in all_labels],
        'Transition Vulnerability (%)': risk_scores
    })

    plt.figure(figsize=(11, 6))
    order = [c for c in ['CN', 'EMCI', 'LMCI', 'AD'] if c in df['Diagnosis'].values]
    palette = [class_colors.get(c, '#333333') for c in order]

    sns.violinplot(
        x='Diagnosis', y='Transition Vulnerability (%)', data=df,
        order=order, palette=palette, inner='quartile', cut=0
    )
    sns.stripplot(
        x='Diagnosis', y='Transition Vulnerability (%)', data=df,
        order=order, color='black', alpha=0.2, jitter=0.2, size=3
    )

    plt.axhline(25.0, color='green', linestyle='--', alpha=0.7, label='Low Progression Vulnerability (<25%)')
    plt.axhline(60.0, color='red', linestyle='--', alpha=0.7, label='Elevated Progression Vulnerability (>60%)')

    plt.title('Disease Severity & MCI Transition Vulnerability Stratified by Cohort', fontsize=15, fontweight='bold', pad=15)
    plt.xlabel('Diagnostic Subgroup', fontsize=13, fontweight='bold')
    plt.ylabel('Empirical Transition Vulnerability Score (%)', fontsize=13, fontweight='bold')
    plt.legend(loc='upper left', prop={'size': 11, 'weight': 'bold'})
    plt.grid(True, linestyle='--', alpha=0.4)

    plt.savefig(os.path.join(figures_dir, 'mci_transition_vulnerability_distribution.png'), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


# ═══════════════════════════════════════════════════════════════════
# 7.4 t-SNE Manifold Embeddings
# ═══════════════════════════════════════════════════════════════════

def plot_tsne_embeddings(features_path: str, labels_path: str):
    print("📈 Generating t-SNE Node Embeddings Visualization...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_dir = getattr(Config, 'CHECKPOINT_DIR', '/kaggle/working/checkpoints')
    figures_dir = getattr(Config, 'FIGURES_DIR', '/kaggle/working/outputs/figures')
    os.makedirs(figures_dir, exist_ok=True)

    try:
        features = np.load(features_path)
        labels = np.load(labels_path)
    except FileNotFoundError:
        print("❌ Error: features.npy or labels.npy not found for t-SNE.")
        return

    ckpt_path = os.path.join(checkpoint_dir, 'neurogat_best_model.pt')
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(checkpoint_dir, 'fold0_best.pt')
    if not os.path.exists(ckpt_path):
        print("❌ Error: Best model checkpoint not found. Run training first.")
        return

    # Build Multi-Scale Graph
    knn_k = getattr(Config, 'KNN_K', 5)
    graph = build_multiscale_population_graph(features, k_list=[3, knn_k, 10]).to(device)
    graph.y = torch.tensor(labels, dtype=torch.long).to(device)

    # Load Model
    model = NeuroGAT(in_channels=graph.x.shape[1]).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    with torch.no_grad():
        x = graph.x
        edge_index = graph.edge_index
        edge_attr = getattr(graph, 'edge_attr', None)

        # Cross-modal fusion
        x = model.fusion(x)
        # GAT Layer 1
        res_x1 = model.res1(x)
        x1 = model.gat1(x, edge_index, edge_attr=edge_attr)
        x = model.bn1(torch.nn.functional.leaky_relu(x1 + res_x1, 0.2))
        # GAT Layer 2
        res_x2 = model.res2(x)
        x2 = model.gat2(x, edge_index, edge_attr=edge_attr)
        embeddings = model.bn2(torch.nn.functional.leaky_relu(x2 + res_x2, 0.2)).cpu().numpy()

    print("🧠 Running t-SNE Manifold Projection...")
    tsne = TSNE(n_components=2, random_state=getattr(Config, 'SEED', 42), perplexity=30)
    tsne_results = tsne.fit_transform(embeddings)

    idx_to_class = getattr(Config, 'IDX_TO_CLASS', {0: 'AD', 1: 'CN', 2: 'EMCI', 3: 'LMCI'})
    class_colors = getattr(Config, 'CLASS_COLORS', {'AD': '#e74c3c', 'CN': '#2ecc71', 'EMCI': '#3498db', 'LMCI': '#e67e22'})

    df_tsne = pd.DataFrame()
    df_tsne['tsne-2d-one'] = tsne_results[:, 0]
    df_tsne['tsne-2d-two'] = tsne_results[:, 1]
    df_tsne['Diagnosis'] = [idx_to_class.get(l, f'Class_{l}') for l in labels]

    plt.figure(figsize=(12, 10))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="Diagnosis",
        palette=class_colors,
        data=df_tsne,
        legend="full",
        alpha=0.8,
        s=90
    )
    plt.title('t-SNE Visualization of Latent Graph Embeddings (NeuroGAT)', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('t-SNE Dimension 1', fontsize=13, fontweight='bold')
    plt.ylabel('t-SNE Dimension 2', fontsize=13, fontweight='bold')
    plt.legend(prop={'size': 12, 'weight': 'bold'})
    plt.grid(True, linestyle='--', alpha=0.4)

    plt.savefig(os.path.join(figures_dir, 'tsne_embeddings.png'), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


# ═══════════════════════════════════════════════════════════════════
# 7.5 Master Evaluation Pipeline
# ═══════════════════════════════════════════════════════════════════

def generate_full_evaluation(features_path: str, labels_path: str):
    print("\n" + "="*70)
    print("  📊 SECTION 7: MASTER EVALUATION SUITE (A* / Q1 Submission)")
    print("="*70)

    final_test_path = os.path.join(Config.OUTPUT_DIR, 'results', 'final_test_predictions.pt')
    results_path = os.path.join(Config.OUTPUT_DIR, 'cv_gnn_results.pt')

    if os.path.exists(final_test_path):
        print(f"📦 Loading definitive held-out test evaluation from: {final_test_path}")
        test_data = torch.load(final_test_path, weights_only=False)
        all_labels = np.array(test_data['labels'])
        all_preds = np.array(test_data['preds'])
        all_probs = np.array(test_data['probs'])
        if 'test_risk_scores' in test_data and len(test_data['test_risk_scores']) == len(all_labels):
            all_risks = np.array(test_data['test_risk_scores'])
        else:
            from Section_06_Training_Engine import compute_mci_conversion_risk
            all_risks, _ = compute_mci_conversion_risk(all_probs)
        results = torch.load(results_path, weights_only=False) if os.path.exists(results_path) else []
    elif os.path.exists(results_path):
        results = torch.load(results_path, weights_only=False)
        all_labels = []
        all_preds = []
        all_probs = []
        all_risks = []
        for res in results:
            all_labels.extend(res['labels'])
            all_preds.extend(res['preds'])
            all_probs.extend(res['probs'])
            if 'test_risk_scores' in res:
                all_risks.extend(res['test_risk_scores'])
            elif 'val_risk_scores' in res:
                all_risks.extend(res['val_risk_scores'])
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        if len(all_risks) == len(all_labels):
            all_risks = np.array(all_risks)
        else:
            from Section_06_Training_Engine import compute_mci_conversion_risk
            all_risks, _ = compute_mci_conversion_risk(all_probs)
    else:
        print(f"❌ Error: Neither {final_test_path} nor {results_path} found. Please run training (Section 06).")
        return

    # 1. Classification Report
    print("\n📝 Independent Held-Out Test Cohort Evaluation:")
    report = classification_report(all_labels, all_preds, target_names=Config.CLASS_NAMES)
    print(report)

    report_dict = classification_report(all_labels, all_preds, target_names=Config.CLASS_NAMES, output_dict=True)
    df_report = pd.DataFrame(report_dict).transpose()
    try:
        from IPython.display import display
        display(df_report)
    except Exception:
        print(df_report.to_string())
    df_report.to_csv(os.path.join(Config.OUTPUT_DIR, 'classification_report.csv'))

    # 2. 95% Bootstrap Confidence Intervals (Q1 Publication Requirement)
    n_bootstraps = getattr(Config, 'BOOTSTRAP_ITERATIONS', 1000)
    ci_results = compute_bootstrap_ci(all_labels, all_preds, all_probs, n_bootstraps=n_bootstraps)

    # 2b. Per-Class Clinical Diagnostic Utility Matrix (Sensitivity, Specificity, PPV, NPV, Balanced Acc, DOR)
    compute_clinical_diagnostic_matrix(all_labels, all_preds, Config.OUTPUT_DIR)

    # 2c. 5-Fold Soft Probability Ensemble Test Evaluation (Variance Reduction)
    if results and len(results) >= 2:
        evaluate_multifold_soft_ensemble(results)

    # 3. Export Camera-Ready LaTeX Tables
    if getattr(Config, 'GENERATE_LATEX_TABLES', True):
        export_evaluation_to_latex(ci_results, df_report, Config.OUTPUT_DIR, all_labels)

    # 4. DeLong Statistical Significance Tests (True Paired DeLong with Covariance)
    run_delong_significance_analysis(all_labels, all_probs)

    # 5. Model Calibration & Reliability Diagrams (Guo et al., ICML 2017)
    ece, brier, class_ece = compute_expected_calibration_error(all_labels, all_probs)
    print(f"\n🎯 Raw Model Calibration : Top-Label ECE = {ece*100:.2f}% | Multi-Class Brier Score = {brier:.4f}")
    for c_name, c_ece in class_ece.items():
        print(f"   • {c_name:<4} OVR ECE   : {c_ece*100:.2f}%")

    # Apply Frozen Temperature Scaling Calibration (fitted on OOF dev logits, never re-fitted on test labels)
    scaler_path = os.path.join(Config.OUTPUT_DIR, 'results', 'temperature_scaler.pt')
    if os.path.exists(scaler_path):
        try:
            scaler_ckpt = torch.load(scaler_path, weights_only=False)
            optimal_t = float(scaler_ckpt.get('temperature', 1.0))
            temp_scaler = TemperatureScaling(temperature=optimal_t)
            logits_approx = np.log(np.clip(all_probs, 1e-12, 1.0))
            calibrated_probs = temp_scaler.calibrate(logits_approx)
            cal_ece, cal_brier, _ = compute_expected_calibration_error(all_labels, calibrated_probs)
            print(f"🌡️ Evaluated Frozen Temperature Scaling (T* = {optimal_t:.3f}):")
            print(f"   • Calibrated Top-Label ECE: {cal_ece*100:.2f}% (ECE Improvement: {(ece - cal_ece)*100:.2f}%)")
            print(f"   • Calibrated Brier Score   : {cal_brier:.4f}")
        except Exception as e:
            print(f"ℹ️ Temperature scaling loading note: {e}")
            calibrated_probs = all_probs
    else:
        print("ℹ️ Frozen temperature scaler not found; evaluating uncalibrated probabilities.")
        calibrated_probs = all_probs

    plot_reliability_diagram(all_labels, calibrated_probs, Config.OUTPUT_DIR)

    # 6. Demographic Fairness & Subgroup Performance Audit (CONSORT-AI / Lancet)
    audit_demographic_fairness(all_labels, all_preds, all_probs, Config.OUTPUT_DIR)

    # 7. Standard Publication Figures
    plot_confusion_matrix(all_labels, all_preds)
    plot_multiclass_roc(all_labels, all_probs)
    plot_mci_risk_distribution(all_labels, all_risks)
    plot_tsne_embeddings(features_path, labels_path)

    print("\n✅ All A* Publication Figures, LaTeX Tables, Calibration & Fairness Audits generated in outputs/")


# ═══════════════════════════════════════════════════════════════════
# 7.6 95% Bootstrap Confidence Intervals & LaTeX Table Generator
# ═══════════════════════════════════════════════════════════════════

def compute_bootstrap_ci(
    all_labels: np.ndarray,
    all_preds: np.ndarray,
    all_probs: np.ndarray,
    n_bootstraps: int = 1000,
    confidence_level: float = 0.95
) -> Dict[str, Any]:
    """
    Computes non-parametric 95% Confidence Intervals via Bootstrapping (1,000 resamplings).
    Strict requirement for Q1 medical journals (IEEE TMI, The Lancet Digital Health).
    """
    print(f"\n📊 Computing {int(confidence_level*100)}% Bootstrap Confidence Intervals ({n_bootstraps} iterations)...")
    np.random.seed(getattr(Config, 'SEED', 42))
    n_samples = len(all_labels)
    alpha = (1.0 - confidence_level) / 2.0

    boot_accs = []
    boot_macro_f1 = []
    boot_weighted_f1 = []
    boot_macro_auc = []
    class_names = getattr(Config, 'CLASS_NAMES', ['AD', 'CN', 'EMCI', 'LMCI'])
    boot_class_metrics = {c: {'precision': [], 'recall': [], 'f1': [], 'auc': []} for c in class_names}
    has_probs = all_probs is not None and len(all_probs) == n_samples
    from sklearn.metrics import roc_auc_score

    # Class-Stratified Bootstrap Resampling: resample within each diagnostic class with replacement
    # to guarantee all 4 classes are strictly preserved in every bootstrap replication (Issue 34)
    class_indices = [np.where(all_labels == c)[0] for c in range(len(class_names))]

    for _ in range(n_bootstraps):
        boot_idx_list = []
        for c_idx in class_indices:
            if len(c_idx) > 0:
                boot_idx_list.append(np.random.choice(c_idx, size=len(c_idx), replace=True))
        indices = np.concatenate(boot_idx_list)
        np.random.shuffle(indices)
        b_true = all_labels[indices]
        b_pred = all_preds[indices]

        boot_accs.append(accuracy_score(b_true, b_pred))
        _, _, m_f1, _ = precision_recall_fscore_support(b_true, b_pred, average='macro', zero_division=0)
        _, _, w_f1, _ = precision_recall_fscore_support(b_true, b_pred, average='weighted', zero_division=0)
        boot_macro_f1.append(m_f1)
        boot_weighted_f1.append(w_f1)

        p_cls, r_cls, f_cls, _ = precision_recall_fscore_support(
            b_true, b_pred, labels=list(range(len(class_names))), average=None, zero_division=0
        )
        for idx, c in enumerate(class_names):
            boot_class_metrics[c]['precision'].append(p_cls[idx])
            boot_class_metrics[c]['recall'].append(r_cls[idx])
            boot_class_metrics[c]['f1'].append(f_cls[idx])

        if has_probs:
            b_prob = all_probs[indices]
            try:
                m_auc = roc_auc_score(b_true, b_prob, multi_class='ovr', average='macro')
                boot_macro_auc.append(m_auc)
            except Exception:
                pass
            for idx, c in enumerate(class_names):
                b_bin = (b_true == idx).astype(int)
                if len(np.unique(b_bin)) == 2:
                    try:
                        c_auc = roc_auc_score(b_bin, b_prob[:, idx])
                        boot_class_metrics[c]['auc'].append(c_auc)
                    except Exception:
                        pass

    def get_ci(arr):
        if len(arr) == 0:
            return 0.0, 0.0, 0.0
        low = np.percentile(arr, alpha * 100)
        high = np.percentile(arr, (1.0 - alpha) * 100)
        mean = float(np.mean(arr))
        return mean, float(low), float(high)

    ci_results = {
        'accuracy': get_ci(boot_accs),
        'macro_f1': get_ci(boot_macro_f1),
        'weighted_f1': get_ci(boot_weighted_f1),
        'macro_auc': get_ci(boot_macro_auc) if len(boot_macro_auc) > 10 else (0.0, 0.0, 0.0),
        'classes': {}
    }

    print(f"   • Overall Accuracy : {ci_results['accuracy'][0]*100:.2f}% [95% CI: {ci_results['accuracy'][1]*100:.2f}% - {ci_results['accuracy'][2]*100:.2f}%]")
    print(f"   • Macro Avg F1     : {ci_results['macro_f1'][0]*100:.2f}% [95% CI: {ci_results['macro_f1'][1]*100:.2f}% - {ci_results['macro_f1'][2]*100:.2f}%]")
    print(f"   • Weighted Avg F1  : {ci_results['weighted_f1'][0]*100:.2f}% [95% CI: {ci_results['weighted_f1'][1]*100:.2f}% - {ci_results['weighted_f1'][2]*100:.2f}%]")
    if ci_results['macro_auc'][0] > 0:
        print(f"   • Macro Avg AUROC  : {ci_results['macro_auc'][0]*100:.2f}% [95% CI: {ci_results['macro_auc'][1]*100:.2f}% - {ci_results['macro_auc'][2]*100:.2f}%]")

    for c in class_names:
        p_ci = get_ci(boot_class_metrics[c]['precision'])
        r_ci = get_ci(boot_class_metrics[c]['recall'])
        f_ci = get_ci(boot_class_metrics[c]['f1'])
        a_ci = get_ci(boot_class_metrics[c]['auc'])
        ci_results['classes'][c] = {'precision': p_ci, 'recall': r_ci, 'f1': f_ci, 'auc': a_ci}
        auc_str = f" | AUROC: {a_ci[0]*100:.2f}% [{a_ci[1]*100:.2f}% - {a_ci[2]*100:.2f}%]" if a_ci[0] > 0 else ""
        print(f"   • {c:<4} F1: {f_ci[0]*100:.2f}% [{f_ci[1]*100:.2f}% - {f_ci[2]*100:.2f}%] | Recall: {r_ci[0]*100:.2f}% [{r_ci[1]*100:.2f}% - {r_ci[2]*100:.2f}%]{auc_str}")

    # Save CSV
    ci_rows = []
    for c in class_names:
        row_dict = {
            'Class': c,
            'Precision': f"{ci_results['classes'][c]['precision'][0]*100:.2f}% [{ci_results['classes'][c]['precision'][1]*100:.2f}%, {ci_results['classes'][c]['precision'][2]*100:.2f}%]",
            'Recall': f"{ci_results['classes'][c]['recall'][0]*100:.2f}% [{ci_results['classes'][c]['recall'][1]*100:.2f}%, {ci_results['classes'][c]['recall'][2]*100:.2f}%]",
            'F1-Score': f"{ci_results['classes'][c]['f1'][0]*100:.2f}% [{ci_results['classes'][c]['f1'][1]*100:.2f}%, {ci_results['classes'][c]['f1'][2]*100:.2f}%]",
        }
        if ci_results['classes'][c]['auc'][0] > 0:
            row_dict['AUROC'] = f"{ci_results['classes'][c]['auc'][0]*100:.2f}% [{ci_results['classes'][c]['auc'][1]*100:.2f}%, {ci_results['classes'][c]['auc'][2]*100:.2f}%]"
        ci_rows.append(row_dict)
    df_ci = pd.DataFrame(ci_rows)
    df_ci.to_csv(os.path.join(Config.OUTPUT_DIR, 'metrics_95_ci.csv'), index=False)
    return ci_results


# ═══════════════════════════════════════════════════════════════════
# 7.6B Clinical Diagnostic Utility Matrix (Sensitivity, Specificity, NPV)
# ═══════════════════════════════════════════════════════════════════

def compute_clinical_diagnostic_matrix(
    all_labels: np.ndarray,
    all_preds: np.ndarray,
    output_dir: str
) -> pd.DataFrame:
    """
    Computes per-class One-vs-Rest (OvR) clinical diagnostic utility metrics:
    Sensitivity, Specificity, PPV (Precision), NPV, Balanced Accuracy, and Class-wise OvR DOR.
    Essential for Q1 / A* medical AI journal peer review.
    """
    cm = confusion_matrix(all_labels, all_preds)
    class_names = getattr(Config, 'CLASS_NAMES', ['AD', 'CN', 'EMCI', 'LMCI'])
    rows = []

    for i, c_name in enumerate(class_names):
        tp = int(cm[i, i])
        fn = int(cm[i, :].sum() - tp)
        fp = int(cm[:, i].sum() - tp)
        tn = int(cm.sum() - (tp + fn + fp))

        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        ppv = tp / max(tp + fp, 1)
        npv = tn / max(tn + fn, 1)
        bal_acc = (sens + spec) / 2.0

        # Haldane-Anscombe correction (Anscombe 1956, Haldane 1955)
        # Adds 0.5 to all 4 confusion cells if any cell is 0, eliminating division by zero or infinite DOR
        if tp == 0 or tn == 0 or fp == 0 or fn == 0:
            dor = ((tp + 0.5) * (tn + 0.5)) / ((fp + 0.5) * (fn + 0.5))
        else:
            dor = (tp * tn) / (fp * fn)

        rows.append({
            'Diagnostic Class': c_name,
            'Sensitivity (%)': round(sens * 100, 2),
            'Specificity (%)': round(spec * 100, 2),
            'PPV / Precision (%)': round(ppv * 100, 2),
            'NPV (%)': round(npv * 100, 2),
            'Balanced Accuracy (%)': round(bal_acc * 100, 2),
            'OvR Diagnostic Odds Ratio': round(dor, 2)
        })

    df_diag = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, 'table_clinical_diagnostic_metrics.csv')
    df_diag.to_csv(csv_path, index=False)

    print("\n🏥 Per-Class One-vs-Rest Clinical Diagnostic Utility Matrix (IEEE TMI / MedIA Standard):")
    print(df_diag.to_string(index=False))

    zero_sens_classes = [r['Diagnostic Class'] for r in rows if r['Sensitivity (%)'] == 0.0]
    if zero_sens_classes:
        print(f"   ⚠️ Methodological Defense Caveat: Diagnostic class(es) {zero_sens_classes} exhibited 0.0% Sensitivity.")
        print("      Reported DOR relies on Haldane-Anscombe (+0.5) continuity correction and must NOT be interpreted as true clinical efficacy.")

    if getattr(Config, 'GENERATE_LATEX_TABLES', True):
        tex_path = os.path.join(output_dir, 'table_clinical_diagnostic_metrics.tex')
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\small",
            r"\caption{Per-Class One-vs-Rest (OvR) Clinical Diagnostic Utility Matrix for NeuroGAT under 5-Fold Cross-Validation.}",
            r"\label{tab:clinical_diagnostic_metrics}",
            r"\begin{tabular}{lcccccc}",
            r"\toprule",
            r"\textbf{Class} & \textbf{Sensitivity (\%)} & \textbf{Specificity (\%)} & \textbf{PPV (\%)} & \textbf{NPV (\%)} & \textbf{Balanced Acc (\%)} & \textbf{OvR DOR} \\",
            r"\midrule"
        ]
        for _, r in df_diag.iterrows():
            lines.append(f"{r['Diagnostic Class']} & {r['Sensitivity (%)']:.2f} & {r['Specificity (%)']:.2f} & {r['PPV / Precision (%)']:.2f} & {r['NPV (%)']:.2f} & {r['Balanced Accuracy (%)']:.2f} & {r['OvR Diagnostic Odds Ratio']:.2f} \\\\")
        lines.extend([
            r"\bottomrule",
            r"\multicolumn{7}{l}{\footnotesize \textit{Note:} Haldane-Anscombe (+0.5) correction applied. Zero-sensitivity classes must not be construed as clinical efficacy.} \\",
            r"\end{tabular}",
            r"\end{table}"
        ])
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"📄 Camera-Ready Diagnostic Utility LaTeX Table exported to: {tex_path}")

    return df_diag


def evaluate_multifold_soft_ensemble(cv_results: List[Dict[str, Any]]) -> Optional[Dict[str, float]]:
    """
    Computes 5-Fold Soft Probability Ensemble Test Performance.
    Ensemble averaging across the 5 independent fold models reduces variance and boosts overall accuracy.
    """
    if not cv_results or len(cv_results) < 2:
        return None

    try:
        probs_list = [res['probs'] for res in cv_results if 'probs' in res]
        first_len = len(probs_list[0])
        if not all(len(p) == first_len for p in probs_list):
            return None

        ensemble_probs = np.mean(probs_list, axis=0)
        true_labels = cv_results[0]['labels']
        ensemble_preds = np.argmax(ensemble_probs, axis=1)

        ens_acc = accuracy_score(true_labels, ensemble_preds) * 100
        prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(true_labels, ensemble_preds, average='macro', zero_division=0)
        prec_wt, rec_wt, f1_wt, _ = precision_recall_fscore_support(true_labels, ensemble_preds, average='weighted', zero_division=0)

        print("\n" + "="*70)
        print("  🌟 5-FOLD SOFT PROBABILITY ENSEMBLE EVALUATION (Deployed Model)")
        print("="*70)
        print(f"   • Ensemble Test Accuracy : {ens_acc:.2f}% (Reduced model variance)")
        print(f"   • Ensemble Macro F1      : {f1_macro*100:.2f}%")
        print(f"   • Ensemble Weighted F1   : {f1_wt*100:.2f}%")

        class_names = getattr(Config, 'CLASS_NAMES', ['AD', 'CN', 'EMCI', 'LMCI'])
        p_c, r_c, f_c, _ = precision_recall_fscore_support(true_labels, ensemble_preds, labels=list(range(len(class_names))), average=None, zero_division=0)
        for i, c in enumerate(class_names):
            print(f"     - {c:<4} -> Precision: {p_c[i]*100:.2f}%, Recall: {r_c[i]*100:.2f}%, F1: {f_c[i]*100:.2f}%")
        print("="*70 + "\n")

        return {
            'ensemble_acc': ens_acc,
            'ensemble_macro_f1': f1_macro * 100,
            'ensemble_weighted_f1': f1_wt * 100
        }
    except Exception as e:
        print(f"⚠️ Ensemble evaluation note: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════
# 7.7 Model Calibration (Temperature Scaling, ECE & Reliability Diagrams)
# ═══════════════════════════════════════════════════════════════════

class TemperatureScaling:
    """
    Post-hoc Temperature Scaling Probability Calibration (Guo et al., ICML 2017).
    Optimizes a single scalar parameter T > 0 on validation set logits using L-BFGS to minimize NLL.
    """
    def __init__(self):
        self.temperature = 1.0

    def fit(self, val_logits: np.ndarray, val_labels: np.ndarray) -> float:
        """Find optimal temperature T on validation partition."""
        logits_t = torch.tensor(val_logits, dtype=torch.float32)
        labels_t = torch.tensor(val_labels, dtype=torch.long)
        temp = torch.nn.Parameter(torch.ones(1) * 1.5)
        optimizer = torch.optim.LBFGS([temp], lr=0.01, max_iter=50)

        def _eval():
            optimizer.zero_grad()
            t_clamped = torch.clamp(temp, min=0.01, max=10.0)
            scaled_logits = logits_t / t_clamped
            loss = torch.nn.functional.cross_entropy(scaled_logits, labels_t)
            loss.backward()
            return loss

        try:
            optimizer.step(_eval)
            self.temperature = float(torch.clamp(temp, min=0.01, max=10.0).detach().item())
        except Exception:
            self.temperature = 1.0
        return self.temperature

    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        """Calibrate logits with frozen temperature T."""
        scaled = logits / max(self.temperature, 0.01)
        exp_scaled = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
        return exp_scaled / np.sum(exp_scaled, axis=1, keepdims=True)


def compute_expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> Tuple[float, float, Dict[str, float]]:
    """
    Computes Expected Calibration Error (ECE) and Brier Score (Guo et al., ICML 2017).
    A vital requirement for trustworthy clinical decision support systems.
    """
    n_classes = y_prob.shape[1]
    class_names = getattr(Config, 'CLASS_NAMES', ['AD', 'CN', 'EMCI', 'LMCI'])
    class_ece = {}
    bin_boundaries = np.linspace(0, 1, n_bins + 1)

    # Top-label overall ECE
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = (predictions == y_true).astype(float)

    ece = 0.0
    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_conf_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_conf_in_bin - accuracy_in_bin) * prop_in_bin

    # Per-class One-vs-Rest ECE
    for c in range(n_classes):
        c_true = (y_true == c).astype(float)
        c_prob = y_prob[:, c]
        c_ece = 0.0
        for i in range(n_bins):
            bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
            in_bin = (c_prob > bin_lower) & (c_prob <= bin_upper)
            prop = np.mean(in_bin)
            if prop > 0:
                acc = np.mean(c_true[in_bin])
                conf = np.mean(c_prob[in_bin])
                c_ece += np.abs(conf - acc) * prop
        c_name = class_names[c] if c < len(class_names) else f"Class_{c}"
        class_ece[c_name] = c_ece

    # Multiclass Brier Score
    y_true_onehot = np.zeros_like(y_prob)
    for i, t in enumerate(y_true):
        y_true_onehot[i, t] = 1.0
    brier_score = float(np.mean(np.sum((y_prob - y_true_onehot) ** 2, axis=1)))

    return float(ece), brier_score, class_ece


def plot_reliability_diagram(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_dir: str,
    n_bins: int = 10
):
    """
    Plots multi-class reliability diagrams (calibration curves) with confidence distributions.
    Standard requirement in medical AI for demonstrating probability calibration.
    """
    figures_dir = os.path.join(output_dir, 'figures') if not output_dir.endswith('figures') else output_dir
    os.makedirs(figures_dir, exist_ok=True)

    n_classes = y_prob.shape[1]
    class_names = getattr(Config, 'CLASS_NAMES', ['AD', 'CN', 'EMCI', 'LMCI'])
    class_colors = getattr(Config, 'CLASS_COLORS_LIST', ['#e74c3c', '#2ecc71', '#3498db', '#e67e22'])

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centers = 0.5 * (bin_boundaries[:-1] + bin_boundaries[1:])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: Top-label Overall Calibration
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = (predictions == y_true).astype(float)

    bin_accs = []
    bin_confs = []
    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        if np.sum(in_bin) > 0:
            bin_accs.append(np.mean(accuracies[in_bin]))
            bin_confs.append(np.mean(confidences[in_bin]))
        else:
            bin_accs.append(0.0)
            bin_confs.append(bin_centers[i])

    axes[0].plot([0, 1], [0, 1], 'k--', lw=2, label='Perfect Calibration')
    axes[0].bar(bin_centers, bin_accs, width=1.0/n_bins, alpha=0.5, color='#2980b9', edgecolor='black', label='Empirical Accuracy')
    axes[0].plot(bin_confs, bin_accs, 'ro-', lw=2, label='NeuroGAT Calibration Curve')
    axes[0].set_title('Overall Multi-Class Reliability Diagram', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Mean Predicted Confidence', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Observed Empirical Accuracy', fontsize=11, fontweight='bold')
    axes[0].set_xlim([0, 1])
    axes[0].set_ylim([0, 1])
    axes[0].legend(loc='upper left', fontsize=10)
    axes[0].grid(True, linestyle='--', alpha=0.5)

    # Panel 2: Per-Class Calibration Curves
    axes[1].plot([0, 1], [0, 1], 'k--', lw=2, label='Perfect Calibration')
    for c in range(n_classes):
        c_true = (y_true == c).astype(float)
        c_prob = y_prob[:, c]
        c_accs = []
        c_confs = []
        for i in range(n_bins):
            bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
            in_bin = (c_prob > bin_lower) & (c_prob <= bin_upper)
            if np.sum(in_bin) > 0:
                c_accs.append(np.mean(c_true[in_bin]))
                c_confs.append(np.mean(c_prob[in_bin]))
        if len(c_confs) > 1:
            axes[1].plot(c_confs, c_accs, 'o-', lw=2, color=class_colors[c % len(class_colors)],
                         label=f'{class_names[c]}')

    axes[1].set_title('Per-Class Diagnostic Reliability Curves', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Predicted Class Probability', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Empirical Class Frequency', fontsize=11, fontweight='bold')
    axes[1].set_xlim([0, 1])
    axes[1].set_ylim([0, 1])
    axes[1].legend(loc='upper left', fontsize=10)
    axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    save_path = os.path.join(figures_dir, 'reliability_diagram_calibration.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"📈 Reliability Diagram (Calibration Curve) saved to: {save_path}")


def audit_demographic_fairness(
    all_labels: np.ndarray,
    all_preds: np.ndarray,
    all_probs: np.ndarray,
    output_dir: str
) -> pd.DataFrame:
    """
    Conducts demographic fairness and subgroup performance audit across Gender and Age.
    Essential for clinical reporting standards (CONSORT-AI / SPIRIT-AI / Lancet Digital Health).
    """
    print("\n⚖️ Conducting Demographic Fairness & Subgroup Audit...")
    figures_dir = os.path.join(output_dir, 'figures') if not output_dir.endswith('figures') else output_dir
    os.makedirs(figures_dir, exist_ok=True)

    metadata_path = os.path.join(output_dir, 'preprocessed_metadata.csv')
    if not os.path.exists(metadata_path):
        metadata_path = os.path.join(getattr(Config, 'OUTPUT_DIR', './outputs'), 'preprocessed_metadata.csv')

    df_meta = pd.read_csv(metadata_path) if os.path.exists(metadata_path) else None
    subgroup_data = []

    splits_path = os.path.join(output_dir, 'results', 'splits.pt')
    test_idx = None
    if os.path.exists(splits_path):
        try:
            splits = torch.load(splits_path, weights_only=False)
            test_idx = splits.get('test_indices', None)
        except Exception:
            test_idx = None

    if df_meta is not None and ('PTGENDER' in df_meta.columns or 'GENDER' in df_meta.columns):
        gender_col = 'PTGENDER' if 'PTGENDER' in df_meta.columns else 'GENDER'
        age_col = 'AGE' if 'AGE' in df_meta.columns else None
        if test_idx is not None and max(test_idx) < len(df_meta):
            genders = df_meta[gender_col].iloc[test_idx].values
            ages = df_meta[age_col].iloc[test_idx].values if age_col else None
        elif len(df_meta) == len(all_labels):
            genders = df_meta[gender_col].values
            ages = df_meta[age_col].values if age_col else None
        else:
            print("⚠️ Demographic metadata size does not match test cohort. Skipping demographic fairness audit.")
            return None
    else:
        print("ℹ️ Demographic metadata (preprocessed_metadata.csv) not available. Skipping fairness audit.")
        return None

    # Evaluate Gender Subgroups
    for g_val in ['Male', 'Female']:
        mask = (genders == g_val)
        if np.sum(mask) > 0:
            acc = accuracy_score(all_labels[mask], all_preds[mask])
            _, _, f1_macro, _ = precision_recall_fscore_support(all_labels[mask], all_preds[mask], average='macro', zero_division=0)
            subgroup_data.append({
                'Subgroup Category': 'Sex / Gender',
                'Subgroup': g_val,
                'N': int(np.sum(mask)),
                'Accuracy (%)': round(acc * 100, 2),
                'Macro F1 (%)': round(f1_macro * 100, 2)
            })

    # Evaluate Age Subgroups (<75 vs >=75)
    if ages is not None:
        mask_younger = (ages < 75.0)
        mask_older = (ages >= 75.0)
        for label, m in [('< 75 Years', mask_younger), ('≥ 75 Years', mask_older)]:
            if np.sum(m) > 0:
                acc = accuracy_score(all_labels[m], all_preds[m])
                _, _, f1_macro, _ = precision_recall_fscore_support(all_labels[m], all_preds[m], average='macro', zero_division=0)
                subgroup_data.append({
                    'Subgroup Category': 'Age Cohort',
                    'Subgroup': label,
                    'N': int(np.sum(m)),
                    'Accuracy (%)': round(acc * 100, 2),
                    'Macro F1 (%)': round(f1_macro * 100, 2)
                })

    df_fairness = pd.DataFrame(subgroup_data)
    csv_path = os.path.join(output_dir, 'demographic_fairness_audit.csv')
    df_fairness.to_csv(csv_path, index=False)
    print(df_fairness.to_string(index=False))
    print(f"📄 Demographic Fairness Audit saved to: {csv_path}")

    # Plot Demographic Fairness Comparison
    plt.figure(figsize=(9, 5))
    sns.barplot(data=df_fairness, x='Subgroup', y='Accuracy (%)', hue='Subgroup Category', palette='Blues_d')
    plt.ylim([0, 100])
    plt.title('Demographic Fairness Audit: Subgroup Accuracy Invariance', fontsize=13, fontweight='bold')
    plt.ylabel('Test Accuracy (%)', fontsize=11, fontweight='bold')
    plt.xlabel('Patient Cohort Subgroup', fontsize=11, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    for p in plt.gca().patches:
        height = p.get_height()
        if height > 0:
            plt.gca().annotate(f'{height:.1f}%',
                (p.get_x() + p.get_width() / 2., height / 2.),
                ha='center', va='center', fontsize=11, color='white', fontweight='bold'
            )
    plot_path = os.path.join(figures_dir, 'demographic_fairness_audit.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    return df_fairness


def export_evaluation_to_latex(ci_results: Dict[str, Any], df_report: pd.DataFrame, output_dir: str, all_labels: np.ndarray):
    """Exports camera-ready booktabs LaTeX table for IEEE/Springer papers."""
    table_path = os.path.join(output_dir, 'table_evaluation_metrics_ci.tex')
    has_auc = 'macro_auc' in ci_results and ci_results['macro_auc'][0] > 0

    if has_auc:
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\small",
            r"\caption{NeuroGAT 5-Fold Cross-Validation Diagnostic Performance with 95\% Non-Parametric Bootstrap Confidence Intervals ($B=1,000$).}",
            r"\label{tab:neurogat_metrics_ci}",
            r"\begin{tabular}{lccccc}",
            r"\toprule",
            r"\textbf{Diagnostic Class} & \textbf{Precision [95\% CI]} & \textbf{Recall / Sensitivity [95\% CI]} & \textbf{F1-Score [95\% CI]} & \textbf{AUROC [95\% CI]} & \textbf{Support} \\",
            r"\midrule"
        ]
        class_names = getattr(Config, 'CLASS_NAMES', ['AD', 'CN', 'EMCI', 'LMCI'])
        for c in class_names:
            p_m, p_l, p_h = ci_results['classes'][c]['precision']
            r_m, r_l, r_h = ci_results['classes'][c]['recall']
            f_m, f_l, f_h = ci_results['classes'][c]['f1']
            a_m, a_l, a_h = ci_results['classes'][c].get('auc', (0.0, 0.0, 0.0))
            supp = int(df_report.loc[c, 'support']) if c in df_report.index else 0
            p_str = f"{p_m*100:.1f} [{p_l*100:.1f}, {p_h*100:.1f}]"
            r_str = f"{r_m*100:.1f} [{r_l*100:.1f}, {r_h*100:.1f}]"
            f_str = f"{f_m*100:.1f} [{f_l*100:.1f}, {f_h*100:.1f}]"
            a_str = f"{a_m*100:.1f} [{a_l*100:.1f}, {a_h*100:.1f}]"
            lines.append(f"{c} & {p_str} & {r_str} & {f_str} & {a_str} & {supp:,} \\\\")

        lines.append(r"\midrule")
        acc_m, acc_l, acc_h = ci_results['accuracy']
        mf1_m, mf1_l, mf1_h = ci_results['macro_f1']
        mauc_m, mauc_l, mauc_h = ci_results['macro_auc']
        lines.append(r"\textbf{Overall Accuracy} & \multicolumn{4}{c}{\textbf{" + f"{acc_m*100:.2f}\\% [{acc_l*100:.2f}\\%, {acc_h*100:.2f}\\%]" + r"}} & " + f"{len(all_labels):,}" + r" \\")
        lines.append(r"\textbf{Macro Average F1} & \multicolumn{4}{c}{\textbf{" + f"{mf1_m*100:.2f}\\% [{mf1_l*100:.2f}\\%, {mf1_h*100:.2f}\\%]" + r"}} & - \\")
        lines.append(r"\textbf{Macro Average AUROC} & \multicolumn{4}{c}{\textbf{" + f"{mauc_m*100:.2f}\\% [{mauc_l*100:.2f}\\%, {mauc_h*100:.2f}\\%]" + r"}} & - \\")
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")
    else:
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\small",
            r"\caption{NeuroGAT 5-Fold Cross-Validation Diagnostic Performance with 95\% Non-Parametric Bootstrap Confidence Intervals ($B=1,000$).}",
            r"\label{tab:neurogat_metrics_ci}",
            r"\begin{tabular}{lcccc}",
            r"\toprule",
            r"\textbf{Diagnostic Class} & \textbf{Precision [95\% CI]} & \textbf{Recall / Sensitivity [95\% CI]} & \textbf{F1-Score [95\% CI]} & \textbf{Support} \\",
            r"\midrule"
        ]
        class_names = getattr(Config, 'CLASS_NAMES', ['AD', 'CN', 'EMCI', 'LMCI'])
        for c in class_names:
            p_m, p_l, p_h = ci_results['classes'][c]['precision']
            r_m, r_l, r_h = ci_results['classes'][c]['recall']
            f_m, f_l, f_h = ci_results['classes'][c]['f1']
            supp = int(df_report.loc[c, 'support']) if c in df_report.index else 0
            p_str = f"{p_m*100:.1f} [{p_l*100:.1f}, {p_h*100:.1f}]"
            r_str = f"{r_m*100:.1f} [{r_l*100:.1f}, {r_h*100:.1f}]"
            f_str = f"{f_m*100:.1f} [{f_l*100:.1f}, {f_h*100:.1f}]"
            lines.append(f"{c} & {p_str} & {r_str} & {f_str} & {supp:,} \\\\")

        lines.append(r"\midrule")
        acc_m, acc_l, acc_h = ci_results['accuracy']
        mf1_m, mf1_l, mf1_h = ci_results['macro_f1']
        lines.append(r"\textbf{Overall Accuracy} & \multicolumn{3}{c}{\textbf{" + f"{acc_m*100:.2f}\\% [{acc_l*100:.2f}\\%, {acc_h*100:.2f}\\%]" + r"}} & " + f"{len(all_labels):,}" + r" \\")
        lines.append(r"\textbf{Macro Average F1} & \multicolumn{3}{c}{\textbf{" + f"{mf1_m*100:.2f}\\% [{mf1_l*100:.2f}\\%, {mf1_h*100:.2f}\\%]" + r"}} & - \\")
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")

    with open(table_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"📄 Camera-Ready LaTeX Evaluation Table exported to: {table_path}")


if __name__ == "__main__":
    try:
        if os.path.exists(os.path.join(Config.OUTPUT_DIR, 'cv_gnn_results.pt')):
            f_path = os.path.join(Config.OUTPUT_DIR, 'node_features.npy')
            l_path = os.path.join(Config.OUTPUT_DIR, 'node_labels.npy')
            generate_full_evaluation(f_path, l_path)
        else:
            print("Section 07 (Evaluation) Loaded. Run Section 06 first to evaluate.")
    except Exception as e:
        print(f"⚠️ Could not auto-run Section 07: {e}")

