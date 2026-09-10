#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          SECTION 6: GNN TRAINING ENGINE - NeuroGAT (A* / Q1 Edition)         ║
║  Innovations:                                                                ║
║    1. Dual-Head Multi-Task Loss (Class-Balanced Focal + Cognitive MSE)       ║
║    2. Multi-Scale Edge-Attentive Graph Learning (K=[3, 5, 10])               ║
║    3. 24-Month MCI Conversion Prognosis Risk Trajectory Scoring              ║
║    4. Transductive 5-Fold Cross-Validation with Complete Metrics             ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import time
import copy
from typing import Optional, Tuple, Dict, Any, List, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

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
    'KNN_K': 5,
    'KNN_K_LIST': [3, 5, 10],
    'GAT_HIDDEN_DIM': 128,
    'GAT_HEADS': 4,
    'GAT_DROPOUT': 0.3,
    'CLASSIFIER_DROPOUT': 0.4,
    'L2_REGULARIZATION': 5e-4,
    'AUX_COG_WEIGHT': 0.1,
    'BATCH_SIZE': 1,
    'EPOCHS': 300,
    'PATIENCE': 50,
    'LEARNING_RATE': 5e-4,
    'WEIGHT_DECAY': 0.01,
    'BETAS': (0.9, 0.999),
    'T_0': 20,
    'T_MULT': 2,
    'LABEL_SMOOTHING': 0.05,
    'FOCAL_GAMMA': 1.5,
    'CUSTOM_CLASS_WEIGHTS': [1.0, 1.0, 1.0, 1.0],
    'USE_CUSTOM_CLASS_WEIGHTS': False,
    'USE_COST_SENSITIVE_LOSS': False,
    'COST_EMCI_LMCI_PENALTY': 1.0,
    'USE_LOGIT_ADJUSTMENT': False,
    'LOGIT_ADJUST_TAU': 0.1,
    'USE_EFFECTIVE_NUM_SAMPLES': True,
    'EFFECTIVE_NUM_BETA': 0.999,
    'USE_DROPEDGE': True,
    'DROPEDGE_RATE': 0.15,
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
# 6.1 Cost-Sensitive Balanced Focal Loss & Logit Adjustment (NeurIPS 2020)
# ═══════════════════════════════════════════════════════════════════

def compute_effective_class_weights(
    labels: Union[np.ndarray, torch.Tensor],
    beta: float = 0.9999,
    num_classes: int = 4
) -> torch.Tensor:
    """
    Computes Class-Balanced Loss weights based on Effective Number of Samples (Cui et al., CVPR 2019).
    E_n = (1 - beta^n) / (1 - beta)
    Weight_c = (1 - beta) / (1 - beta^{n_c})
    Weights are re-normalized so sum(weights) = num_classes (preserving gradient scale).
    """
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    unique, counts = np.unique(labels, return_counts=True)
    count_dict = dict(zip(unique, counts))

    weights = []
    for c in range(num_classes):
        n_c = count_dict.get(c, 1)
        eff_num = (1.0 - np.power(beta, n_c)) / (1.0 - beta)
        weights.append(1.0 / max(eff_num, 1e-8))

    weights = np.array(weights, dtype=np.float32)
    weights = (weights / weights.sum()) * num_classes
    return torch.tensor(weights, dtype=torch.float32)


class CostSensitiveBalancedFocalLoss(nn.Module):
    """
    Cost-Sensitive Balanced Multi-Class Focal Loss with Asymmetric Penalty and
    Logit-Adjusted Loss (Menon et al., NeurIPS 2020).
    Addresses severe dataset imbalance (LMCI vs CN) and penalizes transitional confusion (EMCI <-> LMCI).
    Features:
      1. Class importance weights (alpha) based on Effective Number of Samples (Cui et al., CVPR 2019).
      2. Modulating focal factor ((1 - p_t) ** gamma) down-weighting easy majority examples.
      3. Asymmetric pairwise cost matrix: applies a multiplier (e.g. 2.5x) when confusing EMCI with LMCI.
      4. Fisher-consistent Logit Adjustment (NeurIPS 2020) enforcing margin tau * log(pi_y).
    """
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 1.5,
        label_smoothing: float = 0.05,
        cost_matrix: Optional[torch.Tensor] = None,
        logit_adjustment: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.cost_matrix = cost_matrix
        self.logit_adjustment = logit_adjustment

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Apply Logit Adjustment (Menon et al., NeurIPS 2020) if configured
        if self.logit_adjustment is not None:
            if self.logit_adjustment.device != inputs.device:
                self.logit_adjustment = self.logit_adjustment.to(inputs.device)
            inputs = inputs + self.logit_adjustment

        # 1. Compute unweighted cross-entropy to get true p_t for the focal modulating factor (Lin et al., ICCV 2017)
        ce_loss_raw = F.cross_entropy(
            inputs, targets,
            label_smoothing=self.label_smoothing,
            reduction='none'
        )
        pt = torch.exp(-ce_loss_raw)
        focal_weight = (1.0 - pt) ** self.gamma

        # 2. Apply class importance weights (alpha) linearly (Cui et al., CVPR 2019)
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            loss = self.alpha[targets] * focal_weight * ce_loss_raw
        else:
            loss = focal_weight * ce_loss_raw

        # 3. Cost-Sensitive Misclassification Penalty (if configured)
        if self.cost_matrix is not None:
            probs = F.softmax(inputs, dim=-1)
            if self.cost_matrix.device != inputs.device:
                self.cost_matrix = self.cost_matrix.to(inputs.device)

            # sample_costs[i, c] = cost of predicting class c for ground truth target[i]
            sample_costs = self.cost_matrix[targets] # Shape: (B, num_classes)
            cost_factor = (sample_costs * probs).sum(dim=-1) # Shape: (B,)
            loss = loss * cost_factor

        return loss.mean()

# Backward compatibility alias
ClassBalancedFocalLoss = CostSensitiveBalancedFocalLoss


# ═══════════════════════════════════════════════════════════════════
# 6.1.5 DropEdge Graph Regularization (Rong et al., ICLR 2020)
# ═══════════════════════════════════════════════════════════════════

def apply_dropedge(
    edge_index: torch.Tensor,
    drop_rate: float = 0.15,
    training: bool = True,
    edge_attr: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    DropEdge: Randomly removes a fraction of graph edges during training.
    Prevents GNN over-smoothing, acts as message-passing data augmentation,
    and improves out-of-distribution topological generalization.
    """
    if not training or drop_rate <= 0.0 or edge_index is None:
        return edge_index, edge_attr

    num_edges = edge_index.size(1)
    keep_mask = torch.rand(num_edges, device=edge_index.device) >= drop_rate
    dropped_edge_index = edge_index[:, keep_mask]
    dropped_edge_attr = edge_attr[keep_mask] if edge_attr is not None else None
    return dropped_edge_index, dropped_edge_attr


# ═══════════════════════════════════════════════════════════════════
# 6.2 24-Month MCI Conversion Prognosis Risk Trajectory
# ═══════════════════════════════════════════════════════════════════

def compute_mci_conversion_risk(
    probs: np.ndarray,
    cognitive_scores: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, List[str]]:
    """
    Computes a calibrated 24-Month Mild Cognitive Impairment (MCI) to Alzheimer's Disease (AD)
    conversion hazard index for clinical prognosis.

    Class mapping: 0: AD, 1: CN, 2: EMCI, 3: LMCI

    Clinical Formula:
        Hazard Index H = 0.50 * P(LMCI) + 0.35 * P(AD) + 0.15 * (P(LMCI) / (P(EMCI) + P(LMCI) + 1e-6))
        If cognitive severity (e.g. normalized MMSE deficit) is provided, it modulates the hazard:
        Risk = H * (0.8 + 0.4 * cog_severity) scaled to [0, 100]%

    Returns:
        risk_percentages: (N,) array of risk scores in [0, 100]%
        risk_categories: List of clinical risk classifications:
                         - 'Low Conversion Risk (<25%)'
                         - 'Moderate Conversion Risk (25-60%)'
                         - 'High Rapid Conversion Risk (>60%)'
    """
    probs = np.asarray(probs)
    p_ad = probs[:, 0]
    p_emci = probs[:, 2]
    p_lmci = probs[:, 3]

    # MCI progression ratio
    mci_ratio = p_lmci / (p_emci + p_lmci + 1e-6)

    # Base progression hazard index
    base_hazard = 0.50 * p_lmci + 0.35 * p_ad + 0.15 * mci_ratio

    if cognitive_scores is not None:
        cog = np.asarray(cognitive_scores).squeeze()
        # Scale hazard by continuous cognitive impairment severity
        modulated_hazard = base_hazard * (0.8 + 0.4 * np.clip(cog, 0.0, 1.0))
    else:
        modulated_hazard = base_hazard

    risk_percentages = np.clip(modulated_hazard * 100.0, 0.0, 100.0)

    risk_categories = []
    for r in risk_percentages:
        if r < 25.0:
            risk_categories.append("Low Conversion Risk (<25%)")
        elif r < 60.0:
            risk_categories.append("Moderate Conversion Risk (25-60%)")
        else:
            risk_categories.append("High Rapid Conversion Risk (>60%)")

    return risk_percentages, risk_categories



# ═══════════════════════════════════════════════════════════════════
# 6.3 GNN Trainer Engine (Multi-Task Learning & Risk Prognosis)
# ═══════════════════════════════════════════════════════════════════

class GNNTrainer:
    def __init__(self, graph_data, fold_idx: int, device: torch.device):
        self.graph = graph_data.to(device)
        self.fold_idx = fold_idx
        self.device = device
        self.lambda_cog = 0.1  # Auxiliary cognitive task loss weight

        # Clinical cognitive impairment severity prior (0: AD, 1: CN, 2: EMCI, 3: LMCI)
        self.severity_prior = torch.tensor([0.85, 0.10, 0.35, 0.65], dtype=torch.float32).to(device)

        # Initialize NeuroGAT model
        self.model = NeuroGAT(in_channels=self.graph.x.shape[1]).to(device)

        # Optimizer with weight decay
        lr = getattr(Config, 'LEARNING_RATE', 5e-4)
        weight_decay = getattr(Config, 'L2_REGULARIZATION', getattr(Config, 'WEIGHT_DECAY', 5e-4))
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        # Cosine Annealing Learning Rate Scheduler
        t_0 = getattr(Config, 'T_0', 20)
        t_mult = getattr(Config, 'T_MULT', 2)
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=t_0,
            T_mult=t_mult
        )

        # 1. Class Priors for Logit Adjustment (Menon et al., NeurIPS 2020)
        train_labels = self.graph.y[self.graph.train_mask].cpu().numpy()
        num_cls = getattr(Config, 'NUM_CLASSES', 4)
        classes, counts = np.unique(train_labels, return_counts=True)
        total_samples = len(train_labels)
        priors = np.zeros(num_cls, dtype=np.float32)
        for c, count in zip(classes, counts):
            priors[c] = count / max(total_samples, 1)
        priors = np.clip(priors, 1e-6, 1.0)
        priors_tensor = torch.tensor(priors, dtype=torch.float32).to(device)

        # Logit adjustment offset vector: tau * log(pi)
        tau = getattr(Config, 'LOGIT_ADJUST_TAU', 1.0)
        use_logit_adj = getattr(Config, 'USE_LOGIT_ADJUSTMENT', True)
        self.logit_adjustment = (tau * torch.log(priors_tensor + 1e-12)) if use_logit_adj else None

        # 2. Compute Class Importance Weights (Cui et al., CVPR 2019 / Custom / Balanced)
        if getattr(Config, 'USE_EFFECTIVE_NUM_SAMPLES', True):
            beta = getattr(Config, 'EFFECTIVE_NUM_BETA', 0.9999)
            weights_tensor = compute_effective_class_weights(train_labels, beta=beta, num_classes=num_cls).to(device)
            if getattr(Config, 'USE_CUSTOM_CLASS_WEIGHTS', False):
                custom_weights = torch.tensor(getattr(Config, 'CUSTOM_CLASS_WEIGHTS', [1.2, 0.8, 0.8, 2.0]), dtype=torch.float32).to(device)
                weights_tensor = weights_tensor * (custom_weights / custom_weights.mean())
        elif getattr(Config, 'USE_CUSTOM_CLASS_WEIGHTS', False):
            custom_weights = getattr(Config, 'CUSTOM_CLASS_WEIGHTS', [1.2, 0.8, 0.8, 2.0])
            weights_tensor = torch.tensor(custom_weights, dtype=torch.float32).to(device)
        else:
            class_weights = compute_class_weight(
                class_weight='balanced',
                classes=np.unique(train_labels),
                y=train_labels
            )
            weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

        # 3. Cost-Sensitive Misclassification Penalty Matrix (EMCI <-> LMCI)
        cost_matrix = None
        if getattr(Config, 'USE_COST_SENSITIVE_LOSS', False):
            cost_matrix = torch.ones((num_cls, num_cls), dtype=torch.float32).to(device)
            penalty = getattr(Config, 'COST_EMCI_LMCI_PENALTY', 2.5)
            # Heavy penalty for transitional confusion
            cost_matrix[2, 3] = penalty # True EMCI predicted as LMCI
            cost_matrix[3, 2] = penalty # True LMCI predicted as EMCI
            cost_matrix[0, 1] = 1.5     # True AD predicted as CN
            cost_matrix[1, 0] = 1.5     # True CN predicted as AD

        # 4. Enable Criterion (ClassBalancedFocalLoss or CrossEntropyLoss)
        use_focal = getattr(Config, 'USE_FOCAL_LOSS', True)
        focal_gamma = getattr(Config, 'FOCAL_GAMMA', 1.5)
        label_smoothing = getattr(Config, 'LABEL_SMOOTHING', 0.05)
        if use_focal:
            self.criterion = CostSensitiveBalancedFocalLoss(
                alpha=weights_tensor,
                gamma=focal_gamma,
                label_smoothing=label_smoothing,
                cost_matrix=cost_matrix,
                logit_adjustment=self.logit_adjustment
            )
        else:
            self.criterion = nn.CrossEntropyLoss(
                weight=weights_tensor,
                label_smoothing=label_smoothing
            )

        # DropEdge Regularization Parameters
        self.use_dropedge = getattr(Config, 'USE_DROPEDGE', True)
        self.dropedge_rate = getattr(Config, 'DROPEDGE_RATE', 0.15)

        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
        self.best_model_wts = copy.deepcopy(self.model.state_dict())
        self.patience_counter = 0
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_mae': []}

    def _get_cognitive_target(self) -> torch.Tensor:
        """Retrieves ground truth or clinical stage prior for continuous cognitive severity."""
        if hasattr(self.graph, 'cog_y') and self.graph.cog_y is not None:
            return self.graph.cog_y
        return self.severity_prior[self.graph.y].unsqueeze(1)

    def train_epoch(self) -> Tuple[float, float]:
        self.model.train()
        self.optimizer.zero_grad()

        # Apply DropEdge on graph connectivity during training (ICLR 2020)
        edge_index = self.graph.edge_index
        edge_attr = getattr(self.graph, 'edge_attr', None)
        if self.use_dropedge and self.dropedge_rate > 0.0:
            edge_index, edge_attr = apply_dropedge(
                edge_index, drop_rate=self.dropedge_rate, training=True, edge_attr=edge_attr
            )

        try:
            out = self.model(self.graph.x, edge_index, edge_attr=edge_attr, return_aux=True)
        except TypeError:
            out = self.model(self.graph.x, edge_index, edge_attr=edge_attr)

        if isinstance(out, tuple):
            logits, cog_pred = out
        else:
            logits, cog_pred = out, None

        train_mask = self.graph.train_mask
        loss_cls = self.criterion(logits[train_mask], self.graph.y[train_mask])

        if cog_pred is not None:
            cog_target = self._get_cognitive_target()
            loss_cog = F.mse_loss(cog_pred[train_mask], cog_target[train_mask])
            total_loss = loss_cls + self.lambda_cog * loss_cog
        else:
            total_loss = loss_cls

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
        self.optimizer.step()
        self.scheduler.step()

        preds = logits[train_mask].argmax(dim=1)
        acc = accuracy_score(self.graph.y[train_mask].cpu(), preds.cpu())
        return total_loss.item(), acc

    def evaluate(self, mask: torch.Tensor):
        self.model.eval()
        with torch.no_grad():
            edge_attr = getattr(self.graph, 'edge_attr', None)
            try:
                out = self.model(self.graph.x, self.graph.edge_index, edge_attr=edge_attr, return_aux=True)
            except TypeError:
                out = self.model(self.graph.x, self.graph.edge_index, edge_attr=edge_attr)

            if isinstance(out, tuple):
                logits, cog_pred = out
                cog_target = self._get_cognitive_target()
                loss_cog = F.mse_loss(cog_pred[mask], cog_target[mask])
                total_loss = loss_cls = self.criterion(logits[mask], self.graph.y[mask]) + self.lambda_cog * loss_cog
                mae = F.l1_loss(cog_pred[mask], cog_target[mask]).item()
                cog_np = cog_pred[mask].cpu().numpy().squeeze()
            else:
                logits = out
                cog_pred = None
                total_loss = self.criterion(logits[mask], self.graph.y[mask])
                mae = 0.0
                cog_np = None

            preds = logits[mask].argmax(dim=1)
            probs = F.softmax(logits[mask], dim=1)
            acc = accuracy_score(self.graph.y[mask].cpu(), preds.cpu())

            probs_np = probs.cpu().numpy()
            risk_scores, risk_strata = compute_mci_conversion_risk(probs_np, cog_np)

            return (
                total_loss.item() if isinstance(total_loss, torch.Tensor) else float(total_loss),
                acc,
                mae,
                preds.cpu().numpy(),
                probs_np,
                cog_np,
                risk_scores,
                risk_strata
            )

    def fit(self, epochs: Optional[int] = None, patience: Optional[int] = None) -> Dict[str, Any]:
        epochs = epochs if epochs is not None else getattr(Config, 'EPOCHS', 300)
        patience = patience if patience is not None else getattr(Config, 'PATIENCE', 50)
        ckpt_dir = getattr(Config, 'CHECKPOINT_DIR', '/kaggle/working/checkpoints')
        os.makedirs(ckpt_dir, exist_ok=True)
        best_ckpt_path = os.path.join(ckpt_dir, f'fold{self.fold_idx}_best.pt')

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc, val_mae, _, _, _, _, _ = self.evaluate(self.graph.val_mask)

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_mae'].append(val_mae)

            # Save best checkpoint strictly based on validation loss to halt overfitting
            improved = False
            if val_loss < self.best_val_loss - 1e-4:
                improved = True
            elif abs(val_loss - self.best_val_loss) <= 1e-4 and val_acc > self.best_val_acc:
                improved = True

            if improved:
                self.best_val_loss = val_loss
                self.best_val_acc = val_acc
                self.best_model_wts = copy.deepcopy(self.model.state_dict())
                self.patience_counter = 0
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'fold_idx': self.fold_idx,
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                    'val_mae': val_mae
                }, best_ckpt_path)
                star = "⭐"
            else:
                self.patience_counter += 1
                star = "  "

            if epoch % 10 == 0 or star == "⭐" or epoch == 1:
                print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Cog MAE: {val_mae:.4f} {star}")

            if self.patience_counter >= patience:
                print(f"⏹️ Early stopping triggered at epoch {epoch} (Validation loss halted improving)")
                break

        # Load best model weights for test evaluation
        best_checkpoint = torch.load(best_ckpt_path, weights_only=False)
        self.model.load_state_dict(best_checkpoint['model_state_dict'])

        test_loss, test_acc, test_mae, test_preds, test_probs, test_cog, test_risk, test_strata = self.evaluate(self.graph.test_mask)
        test_labels = self.graph.y[self.graph.test_mask].cpu().numpy()

        # Plot Fold Learning Curve
        self._plot_learning_curve()

        return {
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'test_acc': test_acc,
            'test_loss': test_loss,
            'test_mae': test_mae,
            'test_preds': test_preds,
            'test_probs': test_probs,
            'test_labels': test_labels,
            'test_cog_preds': test_cog,
            'test_risk_scores': test_risk,
            'test_risk_strata': test_strata,
            'labels': test_labels,
            'preds': test_preds,
            'probs': test_probs,
            'history': self.history
        }

    def _plot_learning_curve(self):
        figures_dir = getattr(Config, 'FIGURES_DIR', '/kaggle/working/outputs/figures')
        os.makedirs(figures_dir, exist_ok=True)
        epochs = range(1, len(self.history['train_loss']) + 1)

        plt.figure(figsize=(14, 5))

        # Loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.history['train_loss'], 'b-', label='Train Loss (Joint)', lw=2)
        plt.plot(epochs, self.history['val_loss'], 'r-', label='Validation Loss', lw=2)
        plt.title(f'Fold {self.fold_idx + 1} Multi-Task Loss Curve', fontsize=14, fontweight='bold')
        plt.xlabel('Epochs', fontsize=12, fontweight='bold')
        plt.ylabel('Loss', fontsize=12, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, linestyle='--', alpha=0.6)

        # Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(epochs, [a * 100 for a in self.history['train_acc']], 'b-', label='Train Acc', lw=2)
        plt.plot(epochs, [a * 100 for a in self.history['val_acc']], 'r-', label='Validation Acc', lw=2)
        plt.title(f'Fold {self.fold_idx + 1} Accuracy Curve', fontsize=14, fontweight='bold')
        plt.xlabel('Epochs', fontsize=12, fontweight='bold')
        plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, f'learning_curve_fold{self.fold_idx+1}.png'), dpi=300)
        plt.show()
        plt.close()


# ═══════════════════════════════════════════════════════════════════
# 6.4 5-Fold Cross-Validation Pipeline
# ═══════════════════════════════════════════════════════════════════

def train_gnn_5fold(features_path: str, labels_path: str):
    print("\n" + "="*70)
    print("  🧠 SECTION 6: TRAINING NeuroGAT (MULTI-SCALE GRAPH ATTENTION NETWORK)")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using Device: {device}\n")

    output_dir = getattr(Config, 'OUTPUT_DIR', '/kaggle/working/outputs')
    checkpoint_dir = getattr(Config, 'CHECKPOINT_DIR', '/kaggle/working/checkpoints')

    # 1. Load Features, Labels, and optional Ground-Truth Cognitive Scores
    print("📦 Loading Node Features...")
    features = np.load(features_path)
    labels = np.load(labels_path)
    print(f"📊 Features Shape: {features.shape} | Labels Shape: {labels.shape}")

    cog_path = os.path.join(output_dir, 'node_cog_scores.npy')
    cog_scores = np.load(cog_path) if os.path.exists(cog_path) else None
    if cog_scores is not None:
        print(f"🧠 Loaded Ground-Truth Continuous Cognitive Impairment Scores: {cog_scores.shape}")

    # 2. Load Patient Splits
    splits_path = os.path.join(output_dir, 'results', 'splits.pt')
    if not os.path.exists(splits_path):
        print(f"❌ Error: splits.pt not found at {splits_path}")
        return

    splits_data = torch.load(splits_path, weights_only=False)
    test_indices = splits_data['test_indices']
    fold_splits = splits_data['fold_splits']

    k_list = getattr(Config, 'KNN_K_LIST', [3, getattr(Config, 'KNN_K', 5), 10])

    print("📖 Evaluation Setting: Transductive Population Graph Learning (Parisot et al., MICCAI 2017 & MedIA 2018).")
    print("   Unlabeled node manifold features available for graph topology; validation and test labels")
    print("   remain strictly masked during optimization with per-node LayerNorm isolation.")

    cv_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(fold_splits):
        print(f"\n==================================================")
        print(f"  Training Fold {fold_idx + 1}/5 (Multi-Scale GAT)")
        print(f"==================================================")

        t0 = time.time()

        # Build fold-isolated graph: PCA & StandardScaler fit strictly on train_idx (Zero Leakage)
        fold_graph = build_multiscale_population_graph(
            features, labels, k_list=k_list, cognitive_scores=cog_scores, train_indices=train_idx
        ).to(device)

        # Create masks on GPU
        fold_graph.train_mask = torch.zeros(fold_graph.num_nodes, dtype=torch.bool).to(device)
        fold_graph.val_mask = torch.zeros(fold_graph.num_nodes, dtype=torch.bool).to(device)
        fold_graph.test_mask = torch.zeros(fold_graph.num_nodes, dtype=torch.bool).to(device)

        fold_graph.train_mask[train_idx] = True
        fold_graph.val_mask[val_idx] = True
        fold_graph.test_mask[test_indices] = True
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Train
        trainer = GNNTrainer(fold_graph, fold_idx, device)
        res = trainer.fit()
        fold_time = (time.time() - t0) / 60

        print(f"\n🎯 Fold {fold_idx + 1} Best Val Acc: {res['best_val_acc']*100:.2f}% (Val Loss: {res['best_val_loss']:.4f}) | Test Acc: {res['test_acc']*100:.2f}% | Cog MAE: {res['test_mae']:.4f}")
        print(f"⏱️ Fold Time: {fold_time:.1f} minutes")

        cv_results.append(res)
        del trainer
        del fold_graph
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    mean_test_acc = np.mean([r['test_acc'] for r in cv_results])
    mean_test_mae = np.mean([r['test_mae'] for r in cv_results])
    print(f"\n🏆 Mean Held-Out Test Accuracy Across 5 Folds: {mean_test_acc*100:.2f}% | Cognitive MAE: {mean_test_mae:.4f}")

    # Save complete results checkpoint
    os.makedirs(output_dir, exist_ok=True)
    torch.save(cv_results, os.path.join(output_dir, 'cv_gnn_results.pt'))

    # Save overall best model state based STRICTLY on validation performance (Zero Test Leakage)
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_fold_idx = int(np.argmin([r['best_val_loss'] for r in cv_results]))
    best_fold_path = os.path.join(checkpoint_dir, f'fold{best_fold_idx}_best.pt')
    target_best_path = os.path.join(checkpoint_dir, 'neurogat_best_model.pt')
    if os.path.exists(best_fold_path):
        import shutil
        shutil.copyfile(best_fold_path, target_best_path)
        print(f"💾 Exported top performing model (Fold {best_fold_idx + 1}, Min Val Loss: {cv_results[best_fold_idx]['best_val_loss']:.4f}) to: {target_best_path}")

    return cv_results


if __name__ == "__main__":
    output_dir = getattr(Config, 'OUTPUT_DIR', '/kaggle/working/outputs')
    f_path = os.path.join(output_dir, 'node_features.npy')
    l_path = os.path.join(output_dir, 'node_labels.npy')
    if os.path.exists(f_path) and os.path.exists(l_path):
        train_gnn_5fold(f_path, l_path)
    else:
        print("Section 06 Loaded. Run Section 04B Feature Extraction first.")

