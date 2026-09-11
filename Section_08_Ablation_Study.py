#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          SECTION 8: AUTOMATED ABLATION STUDIES (Q1 / A* Journal Edition)     ║
║  Generates:                                                                  ║
║    1. Progressive Input Modality Ablation (Deep, Handcrafted, Clinical)      ║
║    2. Graph Architecture Ablation (GCN, GraphSAGE, NeuroGAT)                 ║
║    3. Full A-to-Z Metrics (CM, ROC, Learning Curves, t-SNE, Master Table)    ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_fscore_support
from sklearn.manifold import TSNE

try:
    import torch_geometric.nn as geom_nn
except ImportError:
    pass

try:
    from Section_01_Setup_Configuration import Config
    from Section_05_Model_Architecture import build_population_graph, build_multiscale_population_graph, NeuroGAT
    from Section_06_Training_Engine import GNNTrainer
except (ImportError, ModuleNotFoundError):
    pass

# ═══════════════════════════════════════════════════════════════════
# 8.1 Visualizations Helper Functions
# ═══════════════════════════════════════════════════════════════════

def plot_experiment_confusion_matrix(all_labels, all_preds, exp_name):
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=Config.CLASS_NAMES,
                yticklabels=Config.CLASS_NAMES,
                annot_kws={'size': 14, 'weight': 'bold'})
    plt.title(f'{exp_name} Confusion Matrix', fontsize=14, fontweight='bold', pad=15)
    plt.ylabel('True Class', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Class', fontsize=12, fontweight='bold')
    filename = f'cm_ablation_{exp_name.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "plus")}.png'
    plt.savefig(os.path.join(Config.FIGURES_DIR, filename), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_experiment_roc_curve(all_labels, all_probs, exp_name):
    all_labels = np.asarray(all_labels)
    all_probs = np.asarray(all_probs)
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
    plt.title(f'Multi-class ROC Curve ({exp_name})', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", prop={'size': 10, 'weight': 'bold'})
    filename = f'roc_ablation_{exp_name.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "plus")}.png'
    plt.savefig(os.path.join(Config.FIGURES_DIR, filename), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_experiment_learning_curves(history, exp_name):
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(14, 5))

    # Loss Curve
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss', lw=2)
    plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', lw=2)
    plt.title(f'{exp_name} Loss Curve (Fold 1)', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12, fontweight='bold')
    plt.ylabel('Loss', fontsize=12, fontweight='bold')
    plt.legend(); plt.grid(True, linestyle='--', alpha=0.7)

    # Accuracy Curve
    plt.subplot(1, 2, 2)
    plt.plot(epochs, [x*100 for x in history['train_acc']], 'b-', label='Train Acc', lw=2)
    plt.plot(epochs, [x*100 for x in history['val_acc']], 'r-', label='Validation Acc', lw=2)
    plt.title(f'{exp_name} Accuracy Curve (Fold 1)', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12, fontweight='bold')
    plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    plt.legend(); plt.grid(True, linestyle='--', alpha=0.7)

    filename = f'learning_curve_ablation_{exp_name.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "plus")}.png'
    plt.savefig(os.path.join(Config.FIGURES_DIR, filename), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_experiment_tsne(model, graph_data, exp_name, device):
    model.eval()
    with torch.no_grad():
        x = graph_data.x
        edge_index = graph_data.edge_index
        edge_attr = getattr(graph_data, 'edge_attr', None)

        if hasattr(model, 'gat1'):
            x = model.fusion(x)
            res_x1 = model.res1(x)
            x1 = model.gat1(x, edge_index, edge_attr=edge_attr)
            x = model.bn1(F.leaky_relu(x1 + res_x1, 0.2))
            res_x2 = model.res2(x)
            x2 = model.gat2(x, edge_index, edge_attr=edge_attr)
            embeddings = model.bn2(F.leaky_relu(x2 + res_x2, 0.2)).cpu().numpy()
        else:
            x = F.leaky_relu(model.conv1(x, edge_index), 0.2)
            x = F.leaky_relu(model.conv2(x, edge_index), 0.2)
            embeddings = x.cpu().numpy()

    tsne = TSNE(n_components=2, random_state=Config.SEED, perplexity=30)
    tsne_results = tsne.fit_transform(embeddings)
    labels = graph_data.y.cpu().numpy()

    df_tsne = pd.DataFrame()
    df_tsne['tsne-2d-one'] = tsne_results[:, 0]
    df_tsne['tsne-2d-two'] = tsne_results[:, 1]
    df_tsne['Diagnosis'] = [Config.IDX_TO_CLASS[l] for l in labels]

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x="tsne-2d-one", y="tsne-2d-two", hue="Diagnosis",
                    palette=Config.CLASS_COLORS, data=df_tsne, legend="full", alpha=0.8, s=60)
    plt.title(f't-SNE Visualization ({exp_name})', fontsize=14, fontweight='bold')
    plt.legend(prop={'size': 10, 'weight': 'bold'})
    filename = f'tsne_ablation_{exp_name.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "plus")}.png'
    plt.savefig(os.path.join(Config.FIGURES_DIR, filename), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


# ═══════════════════════════════════════════════════════════════════
# 8.2 Baseline Graph Architectures
# ═══════════════════════════════════════════════════════════════════

class NeuroGCN(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        hidden = Config.GAT_HIDDEN_DIM
        self.conv1 = geom_nn.GCNConv(in_channels, hidden * Config.GAT_HEADS)
        self.conv2 = geom_nn.GCNConv(hidden * Config.GAT_HEADS, hidden)
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(Config.CLASSIFIER_DROPOUT),
            nn.Linear(hidden // 2, Config.NUM_CLASSES)
        )
    def forward(self, x, edge_index, edge_attr=None, return_aux=False):
        x = F.leaky_relu(self.conv1(x, edge_index), 0.2)
        x = F.dropout(x, p=Config.GAT_DROPOUT, training=self.training)
        x = F.leaky_relu(self.conv2(x, edge_index), 0.2)
        logits = self.classifier(x)
        if return_aux:
            return logits, torch.zeros((x.size(0), 1), device=x.device)
        return logits

class NeuroSAGE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        hidden = Config.GAT_HIDDEN_DIM
        self.conv1 = geom_nn.SAGEConv(in_channels, hidden * Config.GAT_HEADS)
        self.conv2 = geom_nn.SAGEConv(hidden * Config.GAT_HEADS, hidden)
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(Config.CLASSIFIER_DROPOUT),
            nn.Linear(hidden // 2, Config.NUM_CLASSES)
        )
    def forward(self, x, edge_index, edge_attr=None, return_aux=False):
        x = F.leaky_relu(self.conv1(x, edge_index), 0.2)
        x = F.dropout(x, p=Config.GAT_DROPOUT, training=self.training)
        x = F.leaky_relu(self.conv2(x, edge_index), 0.2)
        logits = self.classifier(x)
        if return_aux:
            return logits, torch.zeros((x.size(0), 1), device=x.device)
        return logits


# ═══════════════════════════════════════════════════════════════════
# 8.3 Feature Ablation Masking (Rigorous Progressive Logic)
# ═══════════════════════════════════════════════════════════════════

def get_ablation_features(raw_features: np.ndarray, modality: str) -> np.ndarray:
    """
    Carefully isolates specific modalities to benchmark their isolated and combined contributions.
    """
    features = raw_features.copy()
    deep_dim = Config.DEEP_FEATURE_DIM
    radiomics_dim = getattr(Config, 'RADIOMICS_FEATURE_DIM', 68)

    if modality == "Deep Only":
        # Zero out Handcrafted & Clinical features (keep ONLY Deep features)
        features[:, deep_dim:] = 0.0
    elif modality == "Handcrafted Only":
        # Zero out Deep & Clinical features (keep ONLY Radiomics)
        features[:, :deep_dim] = 0.0
        features[:, (deep_dim + radiomics_dim):] = 0.0
    elif modality == "Deep + Handcrafted":
        # Zero out Clinical features (keep Deep + Radiomics)
        features[:, (deep_dim + radiomics_dim):] = 0.0
    elif modality == "All (Proposed)":
        # Keep all features intact
        pass

    return features


# ═══════════════════════════════════════════════════════════════════
# 8.4 Automated Ablation Suite
# ═══════════════════════════════════════════════════════════════════

def run_ablation_studies(features_path: str, labels_path: str):
    print("\n" + "="*70)
    print("  🧪 SECTION 8: COMPREHENSIVE ABLATION STUDIES (Q1 / A* Edition)")
    print("="*70)

    os.makedirs(Config.FIGURES_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    raw_features = np.load(features_path)
    labels = np.load(labels_path)

    splits_path = os.path.join(Config.OUTPUT_DIR, 'results', 'splits.pt')
    splits_data = torch.load(splits_path, weights_only=False)
    test_indices = splits_data['test_indices']
    fold_splits = splits_data['fold_splits']

    experiments = [
        # 1. Modality Contribution
        {"Category": "Input Modality", "Name": "Deep Features Only", "Model": "GAT", "Modality": "Deep Only", "MultiScale": True},
        {"Category": "Input Modality", "Name": "Deep + Handcrafted", "Model": "GAT", "Modality": "Deep + Handcrafted", "MultiScale": True},
        {"Category": "Input Modality", "Name": "Deep + Handcrafted + Clinical", "Model": "GAT", "Modality": "All (Proposed)", "MultiScale": True},

        # 2. Graph Backbone Architecture
        {"Category": "Graph Backbone", "Name": "GCN Baseline", "Model": "GCN", "Modality": "All (Proposed)", "MultiScale": True},
        {"Category": "Graph Backbone", "Name": "GraphSAGE Baseline", "Model": "SAGE", "Modality": "All (Proposed)", "MultiScale": True},
        {"Category": "Graph Backbone", "Name": "NeuroGAT (Proposed)", "Model": "GAT", "Modality": "All (Proposed)", "MultiScale": True},

        # 3. Neighborhood Topology & Connectivity
        {"Category": "Topology Scale", "Name": "Single-Scale Graph (K=5)", "Model": "GAT", "Modality": "All (Proposed)", "MultiScale": False},
        {"Category": "Topology Scale", "Name": "Multi-Scale Graph (K=[3,5,10], Proposed)", "Model": "GAT", "Modality": "All (Proposed)", "MultiScale": True},

        # 4. Auxiliary Task & Regularization
        {"Category": "Multi-Task Learning", "Name": "Without Auxiliary Cognitive Task", "Model": "GAT", "Modality": "All (Proposed)", "MultiScale": True, "lambda_cog": 0.0},
        {"Category": "Graph Regularization", "Name": "Without DropEdge (p=0.0)", "Model": "GAT", "Modality": "All (Proposed)", "MultiScale": True, "use_dropedge": False},
        {"Category": "Loss Calibration", "Name": "Standard Cross-Entropy Loss", "Model": "GAT", "Modality": "All (Proposed)", "MultiScale": True, "standard_ce": True},
    ]

    master_table = []

    for exp in experiments:
        print(f"\n==================================================")
        print(f"▶️ Running Experiment: {exp['Category']} -> {exp['Name']}")
        print(f"==================================================")

        ablated_features = get_ablation_features(raw_features, exp['Modality'])
        fold_accs = []
        fold_f1s = []
        all_labels = []
        all_preds = []
        all_probs = []
        history_fold1 = None
        best_model = None
        last_graph = None

        for fold_idx, (train_idx, val_idx) in enumerate(fold_splits):
            # Strict fold-wise preprocessing & graph construction inside fold loop (Point 4)
            if exp.get('MultiScale', True):
                k_list = getattr(Config, 'KNN_K_LIST', [3, getattr(Config, 'KNN_K', 5), 10])
                graph_data = build_multiscale_population_graph(ablated_features, k_list=k_list, train_indices=train_idx).to(device)
            else:
                graph_data = build_population_graph(ablated_features, k=getattr(Config, 'KNN_K', 5), train_indices=train_idx).to(device)
            graph_data.y = torch.tensor(labels, dtype=torch.long).to(device)

            graph_data.train_mask = torch.zeros(graph_data.num_nodes, dtype=torch.bool).to(device)
            graph_data.val_mask = torch.zeros(graph_data.num_nodes, dtype=torch.bool).to(device)
            graph_data.test_mask = torch.zeros(graph_data.num_nodes, dtype=torch.bool).to(device)

            graph_data.train_mask[train_idx] = True
            graph_data.val_mask[val_idx] = True
            graph_data.test_mask[test_indices] = True

            trainer = GNNTrainer(graph_data, fold_idx, device)
            trainer._plot_learning_curve = lambda: None  # Suppress per-fold auto plot

            if 'lambda_cog' in exp:
                trainer.lambda_cog = exp['lambda_cog']
            if 'use_dropedge' in exp:
                trainer.use_dropedge = exp['use_dropedge']
            if exp.get('standard_ce', False):
                trainer.criterion = torch.nn.CrossEntropyLoss()

            if exp['Model'] == "GCN":
                trainer.model = NeuroGCN(in_channels=graph_data.x.shape[1]).to(device)
            elif exp['Model'] == "SAGE":
                trainer.model = NeuroSAGE(in_channels=graph_data.x.shape[1]).to(device)

            trainer.optimizer = torch.optim.AdamW(
                trainer.model.parameters(),
                lr=Config.LEARNING_RATE,
                weight_decay=Config.L2_REGULARIZATION
            )
            trainer.scheduler = CosineAnnealingWarmRestarts(
                trainer.optimizer,
                T_0=getattr(Config, 'T_0', 25),
                T_mult=getattr(Config, 'T_MULT', 2),
                eta_min=1e-5
            )

            fold_res = trainer.fit()
            f_acc = fold_res.get('val_acc', fold_res.get('best_val_acc', 0.0)) * 100
            _, _, f_f1, _ = precision_recall_fscore_support(fold_res['val_labels'], fold_res['val_preds'], average='macro', zero_division=0)
            fold_accs.append(f_acc)
            fold_f1s.append(f_f1 * 100)
            print(f"   Fold {fold_idx+1} Val Acc: {f_acc:.2f}% | Val Macro F1: {f_f1*100:.2f}%")

            if fold_idx == 0:
                history_fold1 = fold_res['history']

            all_labels.extend(fold_res['val_labels'])
            all_preds.extend(fold_res['val_preds'])
            all_probs.extend(fold_res['val_probs'])
            best_model = trainer.model
            last_graph = graph_data

        mean_acc = float(np.mean(fold_accs))
        std_acc = float(np.std(fold_accs))
        mean_f1 = float(np.mean(fold_f1s))
        std_f1 = float(np.std(fold_f1s))

        master_table.append({
            'Category': exp['Category'],
            'Ablation Model': exp['Name'],
            'Accuracy': f"{mean_acc:.2f} ± {std_acc:.2f}%",
            'Macro F1': f"{mean_f1:.2f} ± {std_f1:.2f}%"
        })

        print(f"✅ {exp['Name']} Development 5-Fold: Acc = {mean_acc:.2f} ± {std_acc:.2f}% | Macro F1 = {mean_f1:.2f} ± {std_f1:.2f}%")

        # Plot Figures for each Ablation
        plot_experiment_learning_curves(history_fold1, exp['Name'])
        plot_experiment_confusion_matrix(all_labels, all_preds, exp['Name'])
        plot_experiment_roc_curve(all_labels, all_probs, exp['Name'])
        plot_experiment_tsne(best_model, last_graph, exp['Name'], device)

    # Display and Save Master Ablation Table
    df = pd.DataFrame(master_table)
    df.to_csv(os.path.join(Config.OUTPUT_DIR, 'ablation_master_table.csv'), index=False)

    if getattr(Config, 'GENERATE_LATEX_TABLES', True):
        export_ablation_table_to_latex(df, Config.OUTPUT_DIR)

    print("\n" + "="*70)
    print("  🏆 MASTER ABLATION STUDY RESULTS TABLE")
    print("="*70)
    try:
        from IPython.display import display
        display(df)
    except ImportError:
        print(df.to_string(index=False))

    return df


def export_ablation_table_to_latex(df_ablation: pd.DataFrame, output_dir: str):
    """Exports camera-ready booktabs LaTeX table for ablation studies."""
    table_path = os.path.join(output_dir, 'table_ablation_study.tex')
    lines = [
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{Comprehensive 10-Point Ablation Study on Modality Combinations and Graph Architectures.}",
        r"\label{tab:ablation_study}",
        r"\begin{tabular}{llcc}",
        r"\toprule",
        r"\textbf{Category} & \textbf{Experimental Variant} & \textbf{Accuracy (\%)} & \textbf{Macro F1 (\%)} \\",
        r"\midrule"
    ]
    curr_cat = None
    for _, row in df_ablation.iterrows():
        cat = str(row['Category'])
        if curr_cat is not None and cat != curr_cat:
            lines.append(r"\midrule")
        curr_cat = cat
        name = str(row['Ablation Model'])
        is_full = "Full" in name or "Proposed" in name or "NeuroGAT" in name
        acc_val = str(row['Accuracy'])
        f1_val = str(row['Macro F1'])
        if is_full:
            lines.append(
                r"\textbf{" + cat + r"} & \textbf{" + name +
                r"} & \textbf{" + acc_val +
                r"} & \textbf{" + f1_val + r"} \\"
            )
        else:
            lines.append(f"{cat} & {name} & {acc_val} & {f1_val} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    with open(table_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"📄 Camera-Ready LaTeX Ablation Table exported to: {table_path}")


if __name__ == "__main__":
    f_path = os.path.join(Config.OUTPUT_DIR, 'node_features.npy')
    l_path = os.path.join(Config.OUTPUT_DIR, 'node_labels.npy')
    if os.path.exists(f_path) and os.path.exists(l_path):
        run_ablation_studies(f_path, l_path)
    else:
        print("Section 08 Loaded. Run Section 04/06 first.")
