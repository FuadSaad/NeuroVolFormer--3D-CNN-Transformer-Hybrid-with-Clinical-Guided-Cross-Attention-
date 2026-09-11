#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          SECTION 10: TRADITIONAL MACHINE LEARNING BASELINES (Q1 Edition)     ║
║  Compares NeuroGAT against SVM, Random Forest, MLP, and Regularized XGBoost  ║
║  Includes:                                                                   ║
║    1. Patient-Level 5-Fold Cross-Validation on Symmetrical Multimodal Space  ║
║    2. Confusion Matrices & ROC Curves for every baseline                     ║
║    3. Edwards' McNemar Significance Tests with Holm-Bonferroni Adjustment    ║
║    4. Master Benchmark Comparison Table                                      ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
from typing import List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from scipy.stats import ttest_rel
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, precision_recall_fscore_support
import xgboost as xgb

try:
    from Section_01_Setup_Configuration import Config
except (ImportError, ModuleNotFoundError):
    pass

# ═══════════════════════════════════════════════════════════════════
# 10.1 Plotting Functions
# ═══════════════════════════════════════════════════════════════════

def plot_model_confusion_matrix(all_labels, all_preds, model_name):
    print(f"📈 Generating Confusion Matrix for {model_name}...")
    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=Config.CLASS_NAMES,
                yticklabels=Config.CLASS_NAMES,
                annot_kws={'size': 14, 'weight': 'bold'})

    plt.title(f'{model_name} Confusion Matrix', fontsize=14, fontweight='bold', pad=15)
    plt.ylabel('True Class', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Class', fontsize=12, fontweight='bold')
    plt.xticks(fontsize=11, fontweight='bold')
    plt.yticks(fontsize=11, fontweight='bold', rotation=0)

    filename = f'confusion_matrix_{model_name.replace(" ", "_").replace("(", "").replace(")", "")}.png'
    plt.savefig(os.path.join(Config.FIGURES_DIR, filename), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_model_roc_curve(all_labels, all_probs, model_name):
    print(f"📈 Generating Multi-class ROC Curve for {model_name}...")
    n_classes = Config.NUM_CLASSES

    y_true_bin = np.zeros((len(all_labels), n_classes))
    for i, label in enumerate(all_labels):
        y_true_bin[i, label] = 1

    fpr = dict(); tpr = dict(); roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], all_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(8, 6))
    colors = Config.CLASS_COLORS_LIST

    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'{Config.CLASS_NAMES[i]} (AUC = {roc_auc[i]:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    plt.title(f'Multi-class ROC Curve ({model_name})', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", prop={'size': 10, 'weight': 'bold'})

    filename = f'roc_curve_{model_name.replace(" ", "_").replace("(", "").replace(")", "")}.png'
    plt.savefig(os.path.join(Config.FIGURES_DIR, filename), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


def compute_mcnemar_test(y_true: np.ndarray, preds_proposed: np.ndarray, preds_baseline: np.ndarray) -> Tuple[float, float]:
    """
    Computes Edwards' continuity-corrected McNemar's test comparing paired predictions on the held-out test cohort.
    b: Proposed model correct, Baseline incorrect
    c: Proposed model incorrect, Baseline correct
    chi2 = (|b - c| - 1)^2 / (b + c)
    """
    assert len(y_true) == len(preds_proposed) == len(preds_baseline), (
        f"McNemar test sample size mismatch: y_true={len(y_true)}, proposed={len(preds_proposed)}, baseline={len(preds_baseline)}"
    )
    from scipy.stats import chi2
    correct_p = (preds_proposed == y_true)
    correct_b = (preds_baseline == y_true)

    b = int(np.sum(correct_p & ~correct_b))
    c = int(np.sum(~correct_p & correct_b))

    if b + c == 0 or abs(b - c) <= 1.0:
        return 0.0, 1.0

    stat = (abs(b - c) - 1.0)**2 / (b + c)
    p_val = float(1.0 - chi2.cdf(stat, df=1))
    return float(stat), float(p_val)


def holm_bonferroni_correction(p_values: List[float]) -> List[float]:
    """Stepwise Holm-Bonferroni FWER adjustment for multiple hypothesis testing."""
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adj = [0.0] * m
    running_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        val = (m - rank) * p
        running_max = max(running_max, val)
        adj[orig_idx] = min(1.0, running_max)
    return adj


# ═══════════════════════════════════════════════════════════════════
# 10.2 ML Baselines Cross-Validation Engine
# ═══════════════════════════════════════════════════════════════════

def run_ml_baselines(features_path: str, labels_path: str):
    print("\n" + "="*70)
    print("  🚀 SECTION 10: TRADITIONAL ML BASELINE COMPARISON (Q1 Edition)")
    print("="*70)

    os.makedirs(Config.FIGURES_DIR, exist_ok=True)

    # 1. Load Data
    X_raw = np.load(features_path)
    y = np.load(labels_path)

    # Clean NaN / Inf
    X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)

    # Load Patient Splits
    splits_path = os.path.join(Config.OUTPUT_DIR, 'results', 'splits.pt')
    if not os.path.exists(splits_path):
        print("❌ Error: splits.pt not found. Run Section 04 first.")
        return

    splits_data = torch.load(splits_path, weights_only=False)
    fold_splits = splits_data['fold_splits']
    test_indices = splits_data['test_indices']
    train_val_indices = [i for i in range(len(X_raw)) if i not in set(test_indices)]
    n_dev = len(train_val_indices)
    n_test = len(test_indices)
    print(f"👥 Cohort Partitions: {n_dev} Development participants (80%) | {n_test} Held-Out Test participants (20%).")

    pca_dim = getattr(Config, 'PCA_DIM', 64)

    print("🛡️ Benchmark Protocol: All models receive the same subject-level input modalities and leakage-free")
    print("   feature information. Graph-based models additionally exploit inter-subject relational structure,")
    print("   while conventional baselines operate on the corresponding tabular feature representation.")

    # Define Standardized, Well-Regularized Baseline Models with Compact Controlled Grids
    models = {
        'SVM (RBF)': SVC(kernel='rbf', C=1.0, probability=True, random_state=Config.SEED),
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=5, random_state=Config.SEED, n_jobs=-1),
        'MLP (Deep NN)': MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, early_stopping=True, random_state=Config.SEED),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.5,
            reg_lambda=1.0,
            eval_metric='mlogloss',
            random_state=Config.SEED
        )
    }

    # Load NeuroGAT results for statistical comparison
    final_test_path = os.path.join(Config.OUTPUT_DIR, 'results', 'final_test_predictions.pt')
    gnn_results_path = os.path.join(Config.OUTPUT_DIR, 'cv_gnn_results.pt')
    gnn_preds = None
    gnn_true = None
    gnn_fold_accs = []

    if os.path.exists(gnn_results_path):
        try:
            gnn_results = torch.load(gnn_results_path, weights_only=False)
            gnn_fold_accs = [res.get('val_acc', res.get('best_val_acc', None)) for res in gnn_results]
            gnn_fold_accs = [a for a in gnn_fold_accs if a is not None]
        except Exception:
            pass

    if os.path.exists(final_test_path):
        try:
            t_data = torch.load(final_test_path, weights_only=False)
            gnn_preds = np.array(t_data['preds'])
            gnn_true = np.array(t_data['labels'])
        except Exception:
            pass

    master_table = []
    baseline_evals = []
    mcnemar_raw_p = []

    base_pred_dir = os.path.join(Config.OUTPUT_DIR, 'baseline_predictions')
    os.makedirs(base_pred_dir, exist_ok=True)

    print("\n🔄 Running 5-Fold Patient-Level CV with Strict Out-of-Fold Validation...\n")

    for model_name, model in models.items():
        print(f"==================================================")
        print(f"  ▶️ Training {model_name}...")
        print(f"==================================================")

        all_val_labels = []
        all_val_preds = []
        all_val_probs = []
        fold_accs = []
        fold_evs = []

        # ── Stage 1: 5-Fold Cross-Validation on Development Cohort ──
        for fold_idx, (train_idx, val_idx) in enumerate(fold_splits):
            # Strict Fold-Wise Preprocessing (Fit on train_idx, Transform val_idx)
            if X_raw.shape[1] >= 1024:
                pca = PCA(n_components=pca_dim, random_state=Config.SEED)
                X_train_deep = pca.fit_transform(X_raw[train_idx, :1024])
                X_val_deep = pca.transform(X_raw[val_idx, :1024])
                ev = float(pca.explained_variance_ratio_.sum() * 100)
                fold_evs.append(ev)

                X_train = np.concatenate([X_train_deep, X_raw[train_idx, 1024:]], axis=1)
                X_val = np.concatenate([X_val_deep, X_raw[val_idx, 1024:]], axis=1)
            else:
                X_train = X_raw[train_idx].copy()
                X_val = X_raw[val_idx].copy()

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
            y_train = y[train_idx]
            y_val = y[val_idx]

            model.fit(X_train, y_train)
            preds_val = model.predict(X_val)
            probs_val = model.predict_proba(X_val)

            acc = accuracy_score(y_val, preds_val)
            fold_accs.append(acc)

            all_val_labels.extend(y_val)
            all_val_preds.extend(preds_val)
            all_val_probs.extend(probs_val)

            ev_str = f" | PCA EV: {fold_evs[-1]:.1f}%" if fold_evs else ""
            print(f"   Fold {fold_idx + 1} Val Acc: {acc * 100:.2f}%{ev_str}")

        all_val_labels = np.array(all_val_labels)
        all_val_preds = np.array(all_val_preds)
        all_val_probs = np.array(all_val_probs)

        mean_cv_acc = np.mean(fold_accs) * 100
        std_cv_acc = np.std(fold_accs) * 100
        prec, rec, f1, _ = precision_recall_fscore_support(all_val_labels, all_val_preds, average='macro', zero_division=0)
        print(f"📊 {model_name} 5-Fold CV Mean: {mean_cv_acc:.2f}% ± {std_cv_acc:.2f}% | Macro F1: {f1 * 100:.2f}%")

        # ── Stage 2 & 3: Retrain on 100% Development Cohort & Evaluate Once on Held-Out Test ──
        print(f"   Retraining {model_name} on 100% Development Cohort (N={n_dev})...")
        if X_raw.shape[1] >= 1024:
            final_pca_base = PCA(n_components=pca_dim, random_state=Config.SEED)
            X_dev_deep = final_pca_base.fit_transform(X_raw[train_val_indices, :1024])
            X_test_deep = final_pca_base.transform(X_raw[test_indices, :1024])
            X_dev = np.concatenate([X_dev_deep, X_raw[train_val_indices, 1024:]], axis=1)
            X_test = np.concatenate([X_test_deep, X_raw[test_indices, 1024:]], axis=1)
        else:
            X_dev = X_raw[train_val_indices].copy()
            X_test = X_raw[test_indices].copy()

        scaler_base = StandardScaler()
        X_dev = scaler_base.fit_transform(X_dev)
        X_test = scaler_base.transform(X_test)

        model.fit(X_dev, y[train_val_indices])
        test_preds = model.predict(X_test)
        test_probs = model.predict_proba(X_test)
        test_acc = accuracy_score(y[test_indices], test_preds) * 100
        print(f"🎯 {model_name} Held-Out Test Acc (Test-Once, N={n_test}): {test_acc:.2f}%")

        # Serialize predictions for DeLong testing & independent verification
        model_slug = model_name.lower().replace(' ', '_').replace('(', '').replace(')', '')
        np.save(os.path.join(base_pred_dir, f'{model_slug}_test_probs.npy'), test_probs)
        np.save(os.path.join(base_pred_dir, f'{model_slug}_test_preds.npy'), test_preds)
        if 'Random Forest' in model_name:
            np.save(os.path.join(Config.OUTPUT_DIR, 'baseline_test_probs.npy'), test_probs)

        # Edwards' continuity-corrected McNemar test on held-out test predictions
        p_mcnemar = None
        if gnn_preds is not None and len(gnn_preds) == len(test_preds):
            _, p_mcnemar = compute_mcnemar_test(y[test_indices], gnn_preds, test_preds)

        mcnemar_raw_p.append(p_mcnemar)
        baseline_evals.append({
            'Category': 'Machine Learning',
            'Baseline Model': model_name,
            'Accuracy': f"{mean_cv_acc:.2f}% ± {std_cv_acc:.2f}%",
            'Precision': f"{prec * 100:.2f}%",
            'Recall': f"{rec * 100:.2f}%",
            'F1-Score': f"{f1 * 100:.2f}%",
            'Test Acc': f"{test_acc:.2f}%",
            'raw_p': p_mcnemar
        })

        plot_model_confusion_matrix(all_val_labels, all_val_preds, model_name)
        plot_model_roc_curve(all_val_labels, all_val_probs, model_name)

    # Apply Holm-Bonferroni Correction if paired tests were valid
    valid_p = [p for p in mcnemar_raw_p if p is not None]
    if len(valid_p) == len(mcnemar_raw_p):
        adj_p_values = holm_bonferroni_correction(mcnemar_raw_p)
        for i, b_eval in enumerate(baseline_evals):
            adj_p = adj_p_values[i]
            if adj_p < 0.001:
                sig_text = f"McNemar p < 0.001 (Sig, Holm-adj)"
            elif adj_p < 0.05:
                sig_text = f"McNemar p = {adj_p:.4f} (Sig, Holm-adj)"
            else:
                sig_text = f"McNemar p = {adj_p:.4f} (NS, Holm-adj)"
            b_eval['Significance vs Proposed'] = sig_text
            del b_eval['raw_p']
            master_table.append(b_eval)
    else:
        for b_eval in baseline_evals:
            b_eval['Significance vs Proposed'] = "N/A (Paired test predictions required)"
            del b_eval['raw_p']
            master_table.append(b_eval)

    # Append Proposed NeuroGAT using actual evaluated metrics (Zero Fabrication)
    if gnn_true is not None and gnn_preds is not None and len(gnn_true) == len(gnn_preds):
        real_acc = float(accuracy_score(gnn_true, gnn_preds) * 100)
        prec, rec, f1, _ = precision_recall_fscore_support(gnn_true, gnn_preds, average='macro', zero_division=0)
        master_table.append({
            'Category': 'Graph Neural Network',
            'Baseline Model': 'NeuroGAT (Proposed)',
            'CV Acc': f"{np.mean(gnn_fold_accs)*100:.2f}%" if len(gnn_fold_accs) > 0 else "N/A",
            'Test Acc': f"{real_acc:.2f}%",
            'Accuracy': f"{real_acc:.2f}%",
            'Precision': f"{prec * 100:.2f}%",
            'Recall': f"{rec * 100:.2f}%",
            'F1-Score': f"{f1 * 100:.2f}%",
            'Significance vs Proposed': 'Reference Baseline (N/A)'
        })
    elif len(gnn_fold_accs) > 0:
        mean_gnn_acc = float(np.mean(gnn_fold_accs) * 100)
        master_table.append({
            'Category': 'Graph Neural Network',
            'Baseline Model': 'NeuroGAT (Proposed)',
            'CV Acc': f"{mean_gnn_acc:.2f}%",
            'Test Acc': "Pending Evaluation",
            'Accuracy': f"{mean_gnn_acc:.2f}%",
            'Precision': "N/A",
            'Recall': "N/A",
            'F1-Score': "N/A",
            'Significance vs Proposed': 'Reference Baseline (N/A)'
        })

    # Save and Display Master Table
    df = pd.DataFrame(master_table)
    df.to_csv(os.path.join(Config.OUTPUT_DIR, 'ml_baselines_master_table.csv'), index=False)

    if getattr(Config, 'GENERATE_LATEX_TABLES', True):
        export_baseline_table_to_latex(df, Config.OUTPUT_DIR)

    print("\n" + "="*70)
    print("  🏆 MASTER MACHINE LEARNING BASELINE COMPARISON TABLE")
    print("="*70)
    try:
        from IPython.display import display
        display(df)
    except ImportError:
        print(df.to_string(index=False))

    return df


def export_baseline_table_to_latex(df_baselines: pd.DataFrame, output_dir: str):
    """Exports publication-grade LaTeX booktabs table for baseline comparisons."""
    table_path = os.path.join(output_dir, 'table_ml_baselines.tex')
    lines = [
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{Comprehensive Benchmark Comparison of Traditional Machine Learning Baselines vs. Proposed NeuroGAT under Patient-Level 5-Fold Cross-Validation.}",
        r"\label{tab:ml_baselines_comparison}",
        r"\begin{tabular}{llcccccc}",
        r"\toprule",
        r"\textbf{Paradigm} & \textbf{Model} & \textbf{CV Acc (\%)} & \textbf{Test Acc (\%)} & \textbf{Precision (\%)} & \textbf{Recall (\%)} & \textbf{F1-Score (\%)} & \textbf{Significance vs. Proposed} \\",
        r"\midrule"
    ]
    for _, row in df_baselines.iterrows():
        is_proposed = "Proposed" in str(row['Baseline Model'])
        cv_str = str(row.get('CV Acc', row.get('Accuracy', 'N/A')))
        test_str = str(row.get('Test Acc', 'N/A'))
        prec_str = str(row.get('Precision', 'N/A'))
        rec_str = str(row.get('Recall', 'N/A'))
        f1_str = str(row.get('F1-Score', 'N/A'))
        sig_str = str(row.get('Significance vs Proposed', 'N/A'))
        if is_proposed:
            lines.append(r"\midrule")
            lines.append(
                r"\textbf{" + str(row['Category']) + r"} & \textbf{" + str(row['Baseline Model']) +
                r"} & \textbf{" + cv_str + r"} & \textbf{" + test_str +
                r"} & \textbf{" + prec_str +
                r"} & \textbf{" + rec_str + r"} & \textbf{" + f1_str +
                r"} & \textbf{" + sig_str + r"} \\"
            )
        else:
            lines.append(f"{row['Category']} & {row['Baseline Model']} & {cv_str} & {test_str} & {prec_str} & {rec_str} & {f1_str} & {sig_str} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    with open(table_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"📄 Camera-Ready LaTeX Baselines Table exported to: {table_path}")


if __name__ == "__main__":
    f_path = os.path.join(Config.OUTPUT_DIR, 'node_features.npy')
    l_path = os.path.join(Config.OUTPUT_DIR, 'node_labels.npy')
    if os.path.exists(f_path) and os.path.exists(l_path):
        run_ml_baselines(f_path, l_path)
    else:
        print("Section 10 Loaded. Run Section 04/06 first.")
