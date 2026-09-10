#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║      SECTION 5: NeuroGAT MODEL ARCHITECTURE (A* / Q1 Journal Edition)        ║
║  Innovations:                                                                ║
║    1. Multi-Scale Population Graph (Micro K=3, Meso K=5, Macro K=10)         ║
║    2. Cross-Modal Attention Fusion (Clinical Queries on Imaging Features)    ║
║    3. Multi-Head Graph Attention Network (GATv2 with Residual Connections)   ║
║    4. Dual-Head Multi-Task Learning (4-Class Diagnosis + Cognitive Score)    ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import math
from typing import Optional, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import torch_geometric.nn as geom_nn
    from torch_geometric.data import Data
except ImportError:
    print("Please install torch_geometric: pip install torch_geometric")
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

if 'Config' not in globals() and 'Config' not in locals():
    try:
        from Section_01_Setup_Configuration import Config
    except Exception:
        class Config:
            pass

_defaults = {
    'GAT_HIDDEN_DIM': 128, 'GAT_HEADS': 4, 'GAT_DROPOUT': 0.3,
    'CLASSIFIER_DROPOUT': 0.4, 'NUM_CLASSES': 4, 'KNN_K': 5,
    'DEEP_FEATURE_DIM': 1024, 'RADIOMICS_FEATURE_DIM': 68, 'CLINICAL_DIM': 10,
    'EPOCHS': 300, 'PATIENCE': 50, 'LEARNING_RATE': 5e-4, 'WEIGHT_DECAY': 0.01,
    'OUTPUT_DIR': '/kaggle/working/outputs' if os.path.exists('/kaggle') else './outputs',
    'CHECKPOINT_DIR': '/kaggle/working/checkpoints' if os.path.exists('/kaggle') else './checkpoints',
    'FIGURES_DIR': '/kaggle/working/outputs/figures' if os.path.exists('/kaggle') else './outputs/figures',
    'PREPROCESSED_DIR': '/kaggle/working/preprocessed' if os.path.exists('/kaggle') else './preprocessed',
    'CLASS_NAMES': ['AD', 'CN', 'EMCI', 'LMCI']
}
for _k, _v in _defaults.items():
    if not hasattr(Config, _k):
        setattr(Config, _k, _v)

# ═══════════════════════════════════════════════════════════════════
# 5.1 Multi-Scale Population Graph Construction
# ═══════════════════════════════════════════════════════════════════

def build_multiscale_population_graph(
    features: np.ndarray,
    labels: np.ndarray,
    k_list: List[int] = [3, 5, 10],
    cognitive_scores: Optional[np.ndarray] = None,
    train_indices: Optional[Union[List[int], np.ndarray]] = None
) -> Data:
    """
    Constructs a Multi-Scale Population Graph (PyG Data) integrating micro-, meso-,
    and macro-scale phenotypic patient manifolds under Transductive Graph Learning 
    (Parisot et al., MICCAI 2017 & Medical Image Analysis 2018).

    Args:
        features: (N, D) multimodal patient feature matrix (Deep + Radiomics + Clinical).
        labels: (N,) diagnostic class labels (0: AD, 1: CN, 2: EMCI, 3: LMCI).
        k_list: List of neighbor scales (default: [3, 5, 10]).
        cognitive_scores: (N,) continuous cognitive scores (MMSE or CDR-SB), if available.
        train_indices: Optional indices of training nodes. When provided, PCA and 
                       StandardScaler are strictly fit on train_indices and then used to 
                       transform validation and test nodes (Zero Distribution Leakage).

    Returns:
        torch_geometric.data.Data object containing x, edge_index, edge_attr, y, and optional cog_y.
    """
    print(f"🔗 Building Multi-Scale Population Graph (Scales K={k_list}) for {features.shape[0]} nodes...")

    # 1. Clean NaN / Inf values
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    # 2. Dimensionality Reduction for Deep Features if raw 1024-D
    if features.shape[1] >= 1024:
        deep_features = features[:, :1024]
        handcrafted = features[:, 1024:]

        print("🧠 Applying PCA to Deep Features (1024 -> 32 dims)...")
        pca = PCA(n_components=32, random_state=42)
        if train_indices is not None and len(train_indices) > 0:
            print("🛡️  Fold Isolation: Fitting PCA strictly on training fold indices...")
            pca.fit(deep_features[train_indices])
            deep_features_pca = pca.transform(deep_features)
        else:
            deep_features_pca = pca.fit_transform(deep_features)

        features = np.concatenate([deep_features_pca, handcrafted], axis=1)
        print(f"📊 New Feature Shape after PCA Fusion: {features.shape}")

    # 3. Standardize features (Fold-Wise Strict Isolation)
    scaler = StandardScaler()
    if train_indices is not None and len(train_indices) > 0:
        print("🛡️  Fold Isolation: Fitting StandardScaler strictly on training fold indices...")
        scaler.fit(features[train_indices])
        features_norm = scaler.transform(features)
    else:
        features_norm = scaler.fit_transform(features)

    # 4. Multi-Scale Affinity Graph Construction (Strict Unsupervised Phenotypic Manifold)
    # Zero Leakage Guard: Edges are computed purely from unsupervised cosine feature distances.
    # Diagnostic labels and MMSE cognitive scores NEVER participate in graph topology construction.
    print("🛡️  Topology Guard: Constructing edges purely from unsupervised feature affinity (Zero Label / Cognitive Score Leakage).")
    max_k = max(k_list)
    knn = NearestNeighbors(n_neighbors=max_k + 1, metric='cosine')
    knn.fit(features_norm)
    distances, indices = knn.kneighbors(features_norm)

    edge_dict = {}  # Map (src, dst) -> aggregated similarity across scales
    scale_weights = {k: 1.0 / len(k_list) for k in k_list}

    N = features_norm.shape[0]
    for i in range(N):
        for k in k_list:
            for j in range(1, k + 1):
                neighbor = indices[i, j]
                sim = max(0.0, 1.0 - distances[i, j])

                # Symmetrical directed edge pair
                for edge in [(i, neighbor), (neighbor, i)]:
                    if edge not in edge_dict:
                        edge_dict[edge] = 0.0
                    edge_dict[edge] += scale_weights[k] * sim

    edge_src = []
    edge_dst = []
    edge_weights = []

    for (src, dst), weight in edge_dict.items():
        edge_src.append(src)
        edge_dst.append(dst)
        edge_weights.append(weight)

    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    edge_weight = torch.tensor(edge_weights, dtype=torch.float32).unsqueeze(1)  # Shape: (E, 1)
    x = torch.tensor(features_norm, dtype=torch.float32)

    graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight)

    # Attach diagnostic labels strictly as downstream supervised training target
    if labels is not None:
        graph_data.y = torch.tensor(labels, dtype=torch.long)

    # Attach continuous cognitive scores strictly as auxiliary multi-task regression target
    if cognitive_scores is not None:
        graph_data.cog_y = torch.tensor(cognitive_scores, dtype=torch.float32).unsqueeze(1)

    print(f"✅ Multi-Scale Graph constructed: {graph_data.num_nodes} nodes, {graph_data.num_edges} edges.")
    return graph_data


def build_population_graph(features: np.ndarray, labels: np.ndarray, k: int = 5, train_indices: Optional[Any] = None) -> Data:
    """Backward-compatible wrapper defaulting to multi-scale population graph."""
    return build_multiscale_population_graph(features, labels, k_list=[3, k, 10], train_indices=train_indices)



# ═══════════════════════════════════════════════════════════════════
# 5.2 Cross-Modal Attention Fusion Module
# ═══════════════════════════════════════════════════════════════════

class CrossModalAttentionFusion(nn.Module):
    """
    Genuine Tri-Modal Multi-Head Cross-Attention Layer (A* / Q1 Architecture).

    Each patient's multimodal inputs (Deep 3D CNN, 3D Radiomics Texture, Clinical Biomarkers)
    are projected into a shared latent metric space as distinct modality tokens and cross-attend
    to one another via multi-head attention without inter-subject data leakage.
    """
    def __init__(
        self,
        feature_dim: int,
        deep_dim: int = 32,
        radio_dim: int = 68,
        clin_dim: int = 10,
        embed_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.2
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.embed_dim = embed_dim

        # Determine exact modality dimensions
        if feature_dim >= 110:
            self.deep_dim = deep_dim
            self.radio_dim = radio_dim
            self.clin_dim = feature_dim - (deep_dim + radio_dim)
        elif feature_dim >= 100:
            self.deep_dim = min(deep_dim, feature_dim)
            self.radio_dim = feature_dim - self.deep_dim
            self.clin_dim = 0
        else:
            self.deep_dim = feature_dim // 2
            self.radio_dim = feature_dim - self.deep_dim
            self.clin_dim = 0

        self.num_modalities = 3 if self.clin_dim > 0 else 2

        self.proj_deep = nn.Sequential(
            nn.Linear(self.deep_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )
        self.proj_radio = nn.Sequential(
            nn.Linear(self.radio_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )
        if self.clin_dim > 0:
            self.proj_clin = nn.Sequential(
                nn.Linear(self.clin_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU()
            )
        else:
            self.proj_clin = None

        # Learnable modality position embeddings
        self.modality_emb = nn.Parameter(torch.randn(1, self.num_modalities, embed_dim) * 0.02)

        # Multi-Head Attention across the modalities
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

        # Project back to original feature dimension with residual shortcut
        self.out_proj = nn.Linear(self.num_modalities * embed_dim, feature_dim)
        self.final_norm = nn.LayerNorm(feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N = x.size(0)

        # Partition modalities for each patient
        x_deep = x[:, :self.deep_dim]
        x_radio = x[:, self.deep_dim:self.deep_dim + self.radio_dim]
        h_deep = self.proj_deep(x_deep).unsqueeze(1)
        h_radio = self.proj_radio(x_radio).unsqueeze(1)

        if self.proj_clin is not None and self.clin_dim > 0:
            x_clin = x[:, self.deep_dim + self.radio_dim:self.deep_dim + self.radio_dim + self.clin_dim]
            h_clin = self.proj_clin(x_clin).unsqueeze(1)
            tokens = torch.cat([h_deep, h_radio, h_clin], dim=1) + self.modality_emb
        else:
            tokens = torch.cat([h_deep, h_radio], dim=1) + self.modality_emb

        # Cross-modal multi-head attention within each patient's representation
        attn_out, _ = self.mha(tokens, tokens, tokens)
        tokens = self.norm1(tokens + attn_out)

        # Feed-Forward refinement
        tokens = self.norm2(tokens + self.ffn(tokens))

        # Flatten the tokens: (N, num_modalities * embed_dim)
        flat_fused = tokens.reshape(N, self.num_modalities * self.embed_dim)

        # Residual fusion to input
        out = self.final_norm(x + self.out_proj(flat_fused))
        return out


# ═══════════════════════════════════════════════════════════════════
# 5.3 NeuroGAT Model Architecture (Dual-Head Multi-Task GATv2)
# ═══════════════════════════════════════════════════════════════════

class NeuroGAT(nn.Module):
    """
    NeuroGAT: Multimodal Multi-Task Graph Attention Network for Alzheimer's Disease Diagnosis.

    A* / Q1 Architecture:
      - Cross-Modal Multimodal Feature Refinement (Self/Cross Attention)
      - Dual-Layer Multi-Head Graph Attention (GATv2 with dynamic edge-affinity weights)
      - Dual Residual Skip Projections + Batch Normalization
      - Multi-Task Prediction Heads:
          1. 4-Class Diagnostic Classifier Head (AD, CN, EMCI, LMCI)
          2. Auxiliary Continuous Cognitive Score Regressor (MMSE / CDR-SB severity)
    """
    def __init__(self, in_channels: int, num_classes: Optional[int] = None):
        super().__init__()
        num_classes = num_classes if num_classes is not None else getattr(Config, 'NUM_CLASSES', 4)

        hidden_dim = getattr(Config, 'GAT_HIDDEN_DIM', 128)
        heads = getattr(Config, 'GAT_HEADS', 4)
        self.gat_dropout = getattr(Config, 'GAT_DROPOUT', 0.3)
        self.classifier_dropout = getattr(Config, 'CLASSIFIER_DROPOUT', 0.4)

        # 1. Cross-Modal Fusion
        self.fusion = CrossModalAttentionFusion(feature_dim=in_channels, num_heads=2, dropout=0.2)

        # 2. Layer 1: Multi-Head GATv2
        self.gat1 = geom_nn.GATv2Conv(
            in_channels=in_channels,
            out_channels=hidden_dim,
            heads=heads,
            dropout=self.gat_dropout,
            edge_dim=1,
            concat=True  # Output dim = hidden_dim * heads
        )

        # 3. Layer 2: Single-Head GATv2
        self.gat2 = geom_nn.GATv2Conv(
            in_channels=hidden_dim * heads,
            out_channels=hidden_dim,
            heads=1,
            dropout=self.gat_dropout,
            edge_dim=1,
            concat=False  # Output dim = hidden_dim
        )

        # Residual shortcut projections
        self.res1 = nn.Linear(in_channels, hidden_dim * heads)
        self.res2 = nn.Linear(hidden_dim * heads, hidden_dim)

        # Layer Normalization for stable convergence with ZERO cross-node distribution leakage
        self.bn1 = nn.LayerNorm(hidden_dim * heads)
        self.bn2 = nn.LayerNorm(hidden_dim)

        # 4a. Task 1 Head: Diagnostic Classifier (4 Classes)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(self.classifier_dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

        # 4b. Task 2 Head: Auxiliary Cognitive Score Regressor (MMSE/CDR-SB)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Normalized continuous severity score in [0, 1]
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        return_aux: bool = False
    ):
        """
        Forward pass.

        Args:
            x: Node features (N, in_channels)
            edge_index: Graph connectivity (2, E)
            edge_attr: Edge weights/attributes (E, 1)
            return_aux: If True, returns (logits, cognitive_prediction) tuple.
                        If False, returns logits only (backward compatibility).
        """
        # 1. Cross-Modal Attention Enhancement
        x = self.fusion(x)

        # 2. GATv2 Layer 1 with Edge Attention & Residual
        res_x1 = self.res1(x)
        x1 = self.gat1(x, edge_index, edge_attr=edge_attr)
        x = self.bn1(F.leaky_relu(x1 + res_x1, 0.2))
        x = F.dropout(x, p=self.gat_dropout, training=self.training)

        # 3. GATv2 Layer 2 with Edge Attention & Residual
        res_x2 = self.res2(x)
        x2 = self.gat2(x, edge_index, edge_attr=edge_attr)
        embeddings = self.bn2(F.leaky_relu(x2 + res_x2, 0.2))

        # 4. Prediction Heads
        logits = self.classifier(embeddings)

        if return_aux:
            cog_pred = self.regressor(embeddings)
            return logits, cog_pred

        return logits

    def get_attention_weights(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extracts multi-head attention weights from GATv2 layers for model interpretability (XAI).

        Returns:
            edge_idx1: Edge connectivity tensor for layer 1 (2, E)
            alpha1: Attention weights for layer 1 averaged across heads (E,)
            edge_idx2: Edge connectivity tensor for layer 2 (2, E)
            alpha2: Attention weights for layer 2 (E,)
        """
        self.eval()
        with torch.no_grad():
            x_fused = self.fusion(x)

            # Layer 1 Attention Weights
            x1, (edge_idx1, a1) = self.gat1(
                x_fused, edge_index, edge_attr=edge_attr, return_attention_weights=True
            )
            alpha1 = a1.mean(dim=1) if a1.dim() > 1 else a1.squeeze()

            res_x1 = self.res1(x_fused)
            x_inter = self.bn1(F.leaky_relu(x1 + res_x1, 0.2))

            # Layer 2 Attention Weights
            x2, (edge_idx2, a2) = self.gat2(
                x_inter, edge_index, edge_attr=edge_attr, return_attention_weights=True
            )
            alpha2 = a2.mean(dim=1) if a2.dim() > 1 else a2.squeeze()

        return edge_idx1, alpha1, edge_idx2, alpha2


# ═══════════════════════════════════════════════════════════════════
# Test Block / Sanity Check
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("🧠 NeuroGAT Q1 Multi-Scale & Multi-Task Architecture Initialized.")

    try:
        dummy_feat = np.random.randn(100, 1092)
        dummy_labels = np.random.randint(0, 4, 100)
        dummy_mmse = np.random.uniform(0.0, 1.0, 100)

        graph = build_multiscale_population_graph(
            dummy_feat, dummy_labels, k_list=[3, 5, 10], cognitive_scores=dummy_mmse
        )

        model = NeuroGAT(in_channels=graph.x.shape[1])
        model.eval()
        with torch.no_grad():
            # Test standard forward
            out_logits = model(graph.x, graph.edge_index, edge_attr=graph.edge_attr)
            print(f"✅ Standard Forward Succeeded! Output Shape: {out_logits.shape} (Expected: 100, 4)")

            # Test multi-task forward
            logits, cog_pred = model(graph.x, graph.edge_index, edge_attr=graph.edge_attr, return_aux=True)
            print(f"✅ Multi-Task Forward Succeeded! Logits: {logits.shape}, Cognitive: {cog_pred.shape}")
    except Exception as e:
        print(f"❌ Error during sanity check: {e}")

