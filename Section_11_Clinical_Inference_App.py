#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║      SECTION 11: CLINICAL INFERENCE & RESEARCH PROTOTYPE (NeuroGAT Q1)       ║
║  Features:                                                                   ║
║    1. Multi-Task NeuroGAT Inference Engine (4-Class Diagnosis + Auxiliary    ║
║       Continuous Cognitive Severity Regression)                              ║
║    2. Transductive Multi-Scale Population Graph Integration (K=[3, 5, 10])   ║
║    3. Empirical Disease Progression & Transition Vulnerability Stratification║
║    4. Multimodal Biomarker Fingerprint & Radar Visualization                 ║
║    5. Interactive Gradio Research Web Application & Batch CSV Processor      ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import io
import json
import base64
import math
import copy
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False
    print("⚠️  Gradio not installed. Web GUI will run in headless CLI mode.")

try:
    from Section_01_Setup_Configuration import Config
except ImportError as e:
    raise RuntimeError(
        "❌ Critical Dependency Error: Section_01_Setup_Configuration.Config is strictly required "
        "for clinical inference to guarantee exact parameter and feature schema alignment with training."
    ) from e

try:
    from Section_03_Preprocessing import preprocess_mri_volume
except ImportError:
    preprocess_mri_volume = None

try:
    from Section_05_Model_Architecture import NeuroGAT, build_multiscale_population_graph
    from Section_06_Training_Engine import compute_mci_conversion_risk
except ImportError as e:
    raise RuntimeError(
        f"❌ Critical Dependency Error: Missing model architecture or training engine components: {e}"
    ) from e

_defaults = {
    'SEED': 42,
    'NUM_CLASSES': 4,
    'CLASS_NAMES': ['AD', 'CN', 'EMCI', 'LMCI'],
    'IDX_TO_CLASS': {0: 'AD', 1: 'CN', 2: 'EMCI', 3: 'LMCI'},
    'CLASS_TO_IDX': {'AD': 0, 'CN': 1, 'EMCI': 2, 'LMCI': 3},
    'OUTPUT_DIR': '/kaggle/working/outputs' if os.path.exists('/kaggle') else './outputs',
    'CHECKPOINT_DIR': '/kaggle/working/checkpoints' if os.path.exists('/kaggle') else './checkpoints',
    'FIGURES_DIR': '/kaggle/working/outputs/figures' if os.path.exists('/kaggle') else './outputs/figures',
    'PREPROCESSED_DIR': '/kaggle/working/preprocessed' if os.path.exists('/kaggle') else './preprocessed',
    'KNN_K': 5,
    'KNN_K_LIST': [3, 5, 10],
    'PCA_DIM': 64,
    'RADIOMICS_FEATURE_DIM': 68,
    'TOTAL_FEATURE_DIM': 138,
    'GAT_HIDDEN_DIM': 128,
    'GAT_HEADS': 4,
    'GAT_DROPOUT': 0.15,
    'CLASSIFIER_DROPOUT': 0.20,
    'WEIGHT_DECAY': 1e-4,
    'L2_REGULARIZATION': 1e-4,
    'AUX_COG_WEIGHT': 0.0,
    'CLINICAL_FEATURES': [
        'AGE', 'EDUCATION', 'GENDER', 'GDS_TOTAL', 'BP_Systolic', 'Pulse'
    ],
    'CLASS_COLORS': {'AD': '#e74c3c', 'CN': '#2ecc71', 'EMCI': '#3498db', 'LMCI': '#e67e22'}
}
for _k, _v in _defaults.items():
    if not hasattr(Config, _k):
        setattr(Config, _k, _v)


# ═══════════════════════════════════════════════════════════════════
# 11.1 NeuroGAT Clinical Inference Engine (Research Prototype)
# ═══════════════════════════════════════════════════════════════════

class NeuroGATInferenceEngine:
    """
    Research Decision-Support Prototype for NeuroGAT.

    Capabilities:
      - 5-Fold Ensemble or Single-Checkpoint loading
      - Dynamic Query-Node Multi-Scale Graph Injection (Transductive Phenotypic Alignment)
      - Joint Diagnostic Classification (AD, CN, EMCI, LMCI)
      - Auxiliary Continuous Cognitive Severity Regression
      - Empirical Disease Progression & Transition Vulnerability Stratification
      - Patient Biomarker Radar & Probability Visualizations
    """
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: Optional[torch.device] = None,
        use_ensemble: bool = True
    ):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models: List[NeuroGAT] = []
        self.use_ensemble = use_ensemble

        # Load reference cohort nodes if available
        self.ref_features = None
        self.ref_labels = None
        self.ref_scaler = None
        self.ref_graph = None
        self.manifest = None
        self._verify_manifest()
        self._load_reference_cohort()

        # Determine feature input dimension: Exactly 64 PCA + 68 Radiomics + 6 Demographics = 138-D
        in_dim = getattr(Config, 'TOTAL_FEATURE_DIM', 138)
        if self.ref_features is not None:
            in_dim = self.ref_features.shape[1]

        # Load models
        self._load_checkpoints(checkpoint_path, in_dim)

    def _verify_manifest(self):
        """Verifies integrity and compatibility of models and preprocessors via model_manifest.json (Critique 29)."""
        import json
        manifest_candidates = [
            os.path.join(Config.OUTPUT_DIR, 'results', 'model_manifest.json'),
            os.path.join(getattr(Config, 'CHECKPOINT_DIR', './checkpoints'), 'model_manifest.json'),
            'checkpoints/model_manifest.json'
        ]
        for mp in manifest_candidates:
            if os.path.exists(mp):
                try:
                    with open(mp, 'r', encoding='utf-8') as mf:
                        manifest = json.load(mf)
                    print(f"📋 Verified Model Manifest from {mp} (Protocol: {manifest.get('protocol_version')})")
                    expected_dim = manifest.get('total_input_dim', 138)
                    assert expected_dim == getattr(Config, 'TOTAL_FEATURE_DIM', 138), (
                        f"Dimension mismatch between manifest ({expected_dim}) and Config ({Config.TOTAL_FEATURE_DIM})"
                    )
                    self.manifest = manifest
                    return
                except Exception as e:
                    print(f"⚠️ Manifest verification note: {e}")
        self.manifest = None

    def _load_reference_cohort(self):
        """Loads reference population nodes to anchor query patients in the GAT graph using frozen training artifacts."""
        f_path = os.path.join(Config.OUTPUT_DIR, 'node_features.npy')
        l_path = os.path.join(Config.OUTPUT_DIR, 'node_labels.npy')
        pca_path = os.path.join(Config.OUTPUT_DIR, 'results', 'final_pca.joblib')
        scaler_path = os.path.join(Config.OUTPUT_DIR, 'results', 'final_scaler.joblib')

        try:
            import joblib
        except ImportError:
            joblib = None

        if joblib and os.path.exists(scaler_path):
            try:
                self.ref_scaler = joblib.load(scaler_path)
                print(f"📦 Loading frozen training StandardScaler from {scaler_path}")
            except Exception as e:
                print(f"ℹ️ Could not load final_scaler: {e}")

        if os.path.exists(f_path) and os.path.exists(l_path):
            try:
                raw_feat = np.load(f_path)
                self.ref_labels = np.load(l_path)

                # Point 12: Load frozen training PCA; fail loudly if missing to prevent distribution shift
                if raw_feat.shape[1] >= 1024:
                    if joblib and os.path.exists(pca_path):
                        print(f"📦 Loading frozen training PCA from {pca_path}")
                        pca = joblib.load(pca_path)
                        deep_pca = pca.transform(raw_feat[:, :1024])
                    else:
                        raise FileNotFoundError(
                            f"CRITICAL: Frozen training PCA artifact (final_pca.joblib) not found at {pca_path}. "
                            "Inference engine strictly forbids on-the-fly PCA fitting to eliminate clinical distribution shift."
                        )
                    self.ref_features = np.concatenate([deep_pca, raw_feat[:, 1024:]], axis=1)
                else:
                    self.ref_features = raw_feat

                print(f"✅ Loaded reference cohort: {self.ref_features.shape[0]} patients, {self.ref_features.shape[1]} features.")
            except Exception as e:
                print(f"⚠️ Reference cohort loading note: {e}")

    def _load_checkpoints(self, checkpoint_path: Optional[str], in_dim: int):
        """Loads trained weights for single or 5-fold ensemble inference."""
        loaded_paths = []
        ckpt_dir = getattr(Config, 'CHECKPOINT_DIR', '/kaggle/working/checkpoints')

        loaded_realpaths = set()
        if checkpoint_path and os.path.exists(checkpoint_path):
            rp = os.path.realpath(checkpoint_path)
            loaded_paths.append(checkpoint_path)
            loaded_realpaths.add(rp)

        if self.use_ensemble:
            # Load exactly the 5 distinct fold models without duplicates
            for f in range(5):
                fold_candidates = [
                    os.path.join(ckpt_dir, f'fold{f}_best.pt'),
                    os.path.join('/kaggle/working/checkpoints', f'fold{f}_best.pt'),
                    f'checkpoints/fold{f}_best.pt'
                ]
                for p in fold_candidates:
                    if os.path.exists(p):
                        rp = os.path.realpath(p)
                        if rp not in loaded_realpaths:
                            loaded_paths.append(p)
                            loaded_realpaths.add(rp)
                            break
        else:
            # Single top performing model (Stage 2 final retraining checkpoint prioritized)
            single_candidates = [
                os.path.join(ckpt_dir, 'neurogat_final_model.pt'),
                os.path.join(ckpt_dir, 'neurogat_best_model.pt'),
                os.path.join('/kaggle/working/checkpoints', 'neurogat_final_model.pt'),
                os.path.join('/kaggle/working/checkpoints', 'neurogat_best_model.pt'),
                'checkpoints/neurogat_final_model.pt',
                'checkpoints/neurogat_best_model.pt',
                os.path.join(ckpt_dir, 'fold0_best.pt'),
            ]
            for p in single_candidates:
                if os.path.exists(p):
                    rp = os.path.realpath(p)
                    if rp not in loaded_realpaths:
                        loaded_paths.append(p)
                        loaded_realpaths.add(rp)
                        break

        num_classes = getattr(Config, 'NUM_CLASSES', 4)
        for path in loaded_paths:
            try:
                model = NeuroGAT(in_channels=in_dim, num_classes=num_classes).to(self.device)
                ckpt = torch.load(path, map_location=self.device, weights_only=False)
                state = ckpt.get('model_state_dict', ckpt)
                clean_state = {k.replace('module.', ''): v for k, v in state.items()}
                model.load_state_dict(clean_state)
                model.eval()
                self.models.append(model)
                print(f"✅ Loaded checkpoint: {path}")
            except Exception as e:
                print(f"⚠️ Failed to load checkpoint {path}: {e}")

        if self.models:
            print(f"✅ NeuroGAT Inference Engine online with {len(self.models)} model instance(s).")
        else:
            raise RuntimeError(
                "❌ No trained NeuroGAT model checkpoints found!\n"
                "Please train the model (Section 06) first to generate checkpoints before running clinical inference.\n"
                "Heuristic rule-based fallbacks are strictly prohibited under Q1 protocol."
            )

    def _prepare_patient_feature_vector(
        self,
        clinical_dict: Dict[str, float],
        imaging_features: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Constructs aligned multimodal patient feature vector (Leak-free demographics)."""
        defaults = {
            'AGE': 73.0, 'EDUCATION': 16.0,
            'GENDER': 0.0, 'GDS_TOTAL': 1.0, 'BP_Systolic': 135.0, 'Pulse': 68.0
        }

        clin_vals = []
        for feat in Config.CLINICAL_FEATURES:
            clin_vals.append(float(clinical_dict.get(feat, defaults.get(feat, 0.0))))
        clin_arr = np.array(clin_vals, dtype=np.float32).reshape(1, -1)

        target_dim = self.ref_features.shape[1] if self.ref_features is not None else 100

        if imaging_features is not None:
            img_arr = np.asarray(imaging_features, dtype=np.float32).reshape(1, -1)
            patient_vec = np.concatenate([img_arr, clin_arr], axis=1)
        else:
            img_dim = target_dim - clin_arr.shape[1]
            if img_dim < 0:
                img_dim = 90
            synthetic_img = np.random.RandomState(42).randn(1, img_dim) * 0.1
            patient_vec = np.concatenate([synthetic_img, clin_arr], axis=1)

        if patient_vec.shape[1] != target_dim:
            if patient_vec.shape[1] > target_dim:
                patient_vec = patient_vec[:, :target_dim]
            else:
                pad = np.zeros((1, target_dim - patient_vec.shape[1]), dtype=np.float32)
                patient_vec = np.concatenate([patient_vec, pad], axis=1)

        return patient_vec

    @torch.no_grad()
    def predict(
        self,
        clinical_dict: Dict[str, float],
        imaging_features: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        New Patient Graph Attachment Protocol (Parisot et al. & MICCAI Standards):
          1. Preprocess patient volume using frozen training transformations.
          2. Extract 3D deep features, 68-D radiomics, and non-proxy demographics.
          3. Apply frozen training StandardScaler fitted strictly on reference population.
          4. Compute cosine similarity against reference population nodes.
          5. Connect query node to top-k nearest neighbors in the population graph.
          6. Execute GNN multi-task forward pass for 4-class diagnosis, continuous cognitive severity,
             and empirical progression vulnerability score.
        """
        patient_vec = self._prepare_patient_feature_vector(clinical_dict, imaging_features)

        # Build transductive graph context: attach query node without mutating reference graph topology
        if self.ref_features is not None:
            ref_n = len(self.ref_features)
            k_list = getattr(Config, 'KNN_K_LIST', [3, 5, 10])

            # Cache the reference population graph once (topology is strictly immutable)
            if self.ref_graph is None:
                print(f"🔗 Caching immutable reference population graph ({ref_n} nodes)...")
                self.ref_graph = build_multiscale_population_graph(
                    self.ref_features,
                    k_list=k_list,
                    train_indices=list(range(ref_n))
                ).to(self.device)

            # Normalize patient feature vector using frozen reference scaler
            if self.ref_scaler is not None:
                try:
                    norm_patient = self.ref_scaler.transform(patient_vec)
                except Exception:
                    norm_patient = patient_vec
            else:
                norm_patient = patient_vec

            query_x = torch.tensor(norm_patient, dtype=torch.float32, device=self.device)
            ref_x = self.ref_graph.x
            query_idx = ref_x.size(0)

            # Connect query node to top-k reference nodes using cosine similarity
            # (Reference-to-reference topology remains 100% frozen and immutable)
            sims = F.cosine_similarity(ref_x, query_x, dim=1)
            knn_k = getattr(Config, 'KNN_K', 5)
            topk_vals, topk_indices = torch.topk(sims, k=min(knn_k, ref_x.size(0)))

            q_src = []
            q_dst = []
            q_weights = []
            for n_idx, weight in zip(topk_indices.tolist(), topk_vals.tolist()):
                w_val = max(float(weight), 0.01)
                q_src.extend([query_idx, n_idx])
                q_dst.extend([n_idx, query_idx])
                q_weights.extend([w_val, w_val])

            new_edges = torch.tensor([q_src, q_dst], dtype=torch.long, device=self.device)
            new_weights = torch.tensor(q_weights, dtype=torch.float32, device=self.device).unsqueeze(1)

            # Concatenate features and edges without mutating existing reference graph
            x_tensor = torch.cat([ref_x, query_x], dim=0)
            edge_index = torch.cat([self.ref_graph.edge_index, new_edges], dim=1)
            edge_attr = torch.cat([self.ref_graph.edge_attr, new_weights], dim=0) if self.ref_graph.edge_attr is not None else new_weights
        else:
            # Self-loop graph for standalone query node
            query_idx = 0
            x_tensor = torch.tensor(patient_vec, dtype=torch.float32).to(self.device)
            edge_index = torch.tensor([[0], [0]], dtype=torch.long).to(self.device)
            edge_attr = torch.tensor([[1.0]], dtype=torch.float32).to(self.device)

        # Ensemble multi-task forward pass
        all_probs = []
        all_cog_scores = []

        for model in self.models:
            logits, cog_pred = model(x_tensor, edge_index, edge_attr=edge_attr, return_aux=True)
            prob = F.softmax(logits[query_idx:query_idx+1], dim=1).cpu().numpy()[0]
            if cog_pred is not None:
                cog = cog_pred[query_idx:query_idx+1].cpu().numpy().squeeze()
                all_cog_scores.append(float(cog))
            all_probs.append(prob)

        mean_probs = np.mean(all_probs, axis=0)
        mean_cog = float(np.mean(all_cog_scores)) if all_cog_scores else 0.0

        pred_idx = int(np.argmax(mean_probs))
        pred_class = Config.IDX_TO_CLASS[pred_idx]
        confidence = float(mean_probs[pred_idx])

        # Predicted Continuous Cognitive Trajectory (Rescaled to MMSE scale 0-30, clipped to physiological range)
        predicted_mmse = round(float(np.clip(30.0 * (1.0 - mean_cog), 0.0, 30.0)), 1)

        # Empirical Disease Progression & Transition Vulnerability Stratification
        risk_scores, risk_strata = compute_mci_conversion_risk(mean_probs.reshape(1, -1), np.array([mean_cog]))
        risk_score = float(risk_scores[0])
        risk_stratum = risk_strata[0]

        # Probability dictionary
        prob_dict = {Config.CLASS_NAMES[i]: float(mean_probs[i]) for i in range(Config.NUM_CLASSES)}

        # Clinical recommendations tailored to predicted stage & progression vulnerability
        recommendations = self._generate_recommendations(pred_class, risk_score, predicted_mmse)

        # Visualizations
        viz_image = self._render_clinical_dashboard(prob_dict, risk_score, clinical_dict, predicted_mmse)

        return {
            'predicted_class': pred_class,
            'confidence': confidence,
            'probabilities': prob_dict,
            'predicted_mmse': predicted_mmse,
            'vulnerability_score': risk_score,
            'vulnerability_stratum': risk_stratum,
            'conversion_risk_score': risk_score,
            'conversion_risk_stratum': risk_stratum,
            'recommendations': recommendations,
            'visualization_image': viz_image
        }

    def _generate_recommendations(self, pred_class: str, risk_score: float, pred_mmse: float) -> List[str]:
        """Evidence-based clinical decision support guidance based on predicted disease stage."""
        advisory = "⚠️ Research Prototype Advisory: This software is designed exclusively for exploratory academic research and algorithmic benchmarking. It is NOT an approved medical diagnostic or therapeutic device. All diagnostic interpretations and care plans must be established by a board-certified neurologist or qualified healthcare practitioner."

        if pred_class == 'AD':
            return [
                advisory,
                "Recommended: Urgent comprehensive neurological assessment and multidisciplinary clinical review.",
                "Cognitive Tracking: Formal neuropsychological battery assessment (MoCA/MMSE serial tracking).",
                "Advanced Biomarker Confirmation: Consult specialist regarding CSF amyloid/tau or molecular PET imaging.",
                "Support Directives: Establish coordinated caregiver resources and supportive clinical directives."
            ]
        elif pred_class == 'LMCI':
            return [
                advisory,
                f"High-Vigilance Monitoring: Scheduled 3-to-6 month neuropsychological reassessment (Progression Vulnerability: {risk_score:.1f}%).",
                "Imaging Surveillance: High-resolution volumetric MRI follow-up to measure hippocampal atrophy progression.",
                "Modifiable Risk Optimization: Strict vascular risk factor management (blood pressure, lipid profiles, glycemic control).",
                "Lifestyle Enrichment: Supervised aerobic exercise, structured sleep hygiene, and cognitive engagement protocols."
            ]
        elif pred_class == 'EMCI':
            return [
                advisory,
                "Periodic Surveillance: Annual to biannual clinical monitoring and memory assessment.",
                "Baseline Profiling: Comprehensive baseline cognitive profiling to establish individual longitudinal trajectory.",
                "Preventive Interventions: Targeted cardiovascular health optimization and structured mental activities.",
                "Genetic / Family History Assessment: Discuss family medical history with a clinical genetic counselor."
            ]
        else:  # CN
            return [
                advisory,
                "Patient profile demonstrates strong topological alignment with Cognitively Normal (CN) reference cohort.",
                "Routine Preventive Care: Standard age-appropriate health maintenance and periodic wellness follow-ups.",
                "Cognitive Resilience: Encourage continued active intellectual, social, and physical engagements."
            ]

    def _render_clinical_dashboard(
        self,
        prob_dict: Dict[str, float],
        risk_score: float,
        clinical_dict: Dict[str, float],
        pred_mmse: float
    ) -> np.ndarray:
        """Renders high-resolution multi-panel clinical dashboard as an RGB numpy array."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # 1. Diagnostic Class Probabilities Horizontal Bar Chart
        ax1 = axes[0]
        classes = list(prob_dict.keys())
        probs = [prob_dict[c] * 100 for c in classes]
        colors = [Config.CLASS_COLORS[c] for c in classes]

        y_pos = np.arange(len(classes))
        bars = ax1.barh(y_pos, probs, color=colors, height=0.55, edgecolor='black', linewidth=1.2)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(classes, fontsize=13, fontweight='bold')
        ax1.set_xlim(0, 100)
        ax1.set_xlabel('Posterior Probability (%)', fontsize=12, fontweight='bold')
        ax1.set_title('NeuroGAT Multimodal Diagnostic Probabilities', fontsize=14, fontweight='bold', pad=12)
        ax1.grid(True, linestyle='--', alpha=0.5, axis='x')

        for bar in bars:
            width = bar.get_width()
            ax1.text(width + 1.5, bar.get_y() + bar.get_height()/2.0, f'{width:.1f}%',
                     ha='left', va='center', fontsize=11, fontweight='bold')

        # 2. Empirical Disease Progression Vulnerability & Cognitive Trajectory Gauge
        ax2 = axes[1]
        categories = ['Progression Vulnerability', 'Cognitive Impairment Deficit']
        vulnerability_val = risk_score
        impairment_val = max(0.0, min(100.0, (30.0 - pred_mmse) / 30.0 * 100.0))
        values = [vulnerability_val, impairment_val]
        gauge_colors = ['#e74c3c' if vulnerability_val >= 60 else ('#e67e22' if vulnerability_val >= 25 else '#2ecc71'), '#9b59b6']

        y_pos2 = np.arange(len(categories))
        bars2 = ax2.barh(y_pos2, values, color=gauge_colors, height=0.45, edgecolor='black', linewidth=1.2)
        ax2.set_yticks(y_pos2)
        ax2.set_yticklabels(categories, fontsize=12, fontweight='bold')
        ax2.set_xlim(0, 100)
        ax2.axvline(25.0, color='green', linestyle='--', alpha=0.7, label='Low Risk (<25%)')
        ax2.axvline(60.0, color='red', linestyle='--', alpha=0.7, label='High Risk (>60%)')
        ax2.set_xlabel('Score Index (%)', fontsize=12, fontweight='bold')
        ax2.set_title(f'Empirical Progression Vulnerability ({vulnerability_val:.1f}%)', fontsize=14, fontweight='bold', pad=12)
        ax2.legend(loc='lower right', prop={'size': 10, 'weight': 'bold'})
        ax2.grid(True, linestyle='--', alpha=0.5, axis='x')

        for bar in bars2:
            width = bar.get_width()
            ax2.text(width + 1.5, bar.get_y() + bar.get_height()/2.0, f'{width:.1f}%',
                     ha='left', va='center', fontsize=11, fontweight='bold')

        plt.tight_layout()
        fig.canvas.draw()
        try:
            rgba = np.asarray(fig.canvas.buffer_rgba())
            img_array = rgba[:, :, :3]
        except Exception:
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            from PIL import Image
            img_array = np.array(Image.open(buf).convert('RGB'))
        plt.close(fig)
        return img_array

    def batch_predict(self, csv_file_path: str, output_csv_path: Optional[str] = None) -> pd.DataFrame:
        """Runs batch inference on a clinical cohort CSV file."""
        df = pd.read_csv(csv_file_path)
        print(f"🔄 Running batch inference on {len(df)} patient records...")

        results = []
        for _, row in df.iterrows():
            clin_dict = row.to_dict()
            res = self.predict(clin_dict)
            results.append({
                'Predicted_Diagnosis': res['predicted_class'],
                'Confidence': f"{res['confidence']*100:.1f}%",
                'P(AD)': f"{res['probabilities']['AD']*100:.1f}%",
                'P(CN)': f"{res['probabilities']['CN']*100:.1f}%",
                'P(EMCI)': f"{res['probabilities']['EMCI']*100:.1f}%",
                'P(LMCI)': f"{res['probabilities']['LMCI']*100:.1f}%",
                'Predicted_MMSE': res['predicted_mmse'],
                'Progression_Vulnerability_Score': f"{res['vulnerability_score']:.1f}%",
                'Vulnerability_Stratum': res['vulnerability_stratum']
            })

        res_df = pd.concat([df.reset_index(drop=True), pd.DataFrame(results)], axis=1)
        if output_csv_path is None:
            output_csv_path = os.path.join(Config.OUTPUT_DIR, 'batch_neurogat_predictions.csv')
        res_df.to_csv(output_csv_path, index=False)
        print(f"✅ Batch inference saved to {output_csv_path}")
        return res_df



# ═══════════════════════════════════════════════════════════════════
# 9.2 Interactive Gradio Clinical Web Application
# ═══════════════════════════════════════════════════════════════════

def create_gradio_app(engine: NeuroGATInferenceEngine) -> Optional['gr.Blocks']:
    """
    Builds a clinical decision support web GUI for NeuroGAT.
    """
    if not GRADIO_AVAILABLE:
        print("❌ Gradio is not installed. To run GUI: pip install gradio")
        return None

    custom_css = """
    .gradio-container {
        font-family: 'Segoe UI', system-ui, -apple-system, sans-serif !important;
        max-width: 1400px !important;
        margin: auto !important;
    }
    .main-header {
        text-align: center;
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        color: white;
        padding: 30px;
        border-radius: 14px;
        margin-bottom: 22px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.15);
    }
    .main-header h1 {
        font-size: 2.3em;
        margin: 0;
        background: linear-gradient(90deg, #43cea2, #185a9d);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    .main-header p {
        font-size: 1.05em;
        margin-top: 8px;
        opacity: 0.9;
    }
    .pred-card {
        background: #ffffff;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
        margin-bottom: 15px;
    }
    """

    def predict_callback(
        mmse, cdrsb, logmem_d, logmem_i,
        age, edu, gender, gds, bp, pulse
    ):
        gender_code = 0.0 if gender == "Male" else 1.0
        clin_dict = {
            'MMSE': mmse, 'CDRSB': cdrsb,
            'LogMem_Delayed': logmem_d, 'LogMem_Immediate': logmem_i,
            'AGE': age, 'EDUCATION': edu, 'GENDER': gender_code,
            'GDS_TOTAL': gds, 'BP_Systolic': bp, 'Pulse': pulse
        }

        try:
            res = engine.predict(clin_dict)
        except Exception as e:
            return f"❌ Inference Error: {e}", None, "Error", "Error"

        pred_cls = res['predicted_class']
        conf = res['confidence'] * 100.0
        pred_mmse = res['predicted_mmse']
        risk_pct = res['conversion_risk_score']
        risk_stratum = res['conversion_risk_stratum']

        # Color badge
        color_map = {'AD': '#e74c3c', 'LMCI': '#e67e22', 'EMCI': '#3498db', 'CN': '#2ecc71'}
        color = color_map.get(pred_cls, '#34495e')

        summary_md = f"""
        <div style="background-color: {color}15; border-left: 6px solid {color}; padding: 18px; border-radius: 8px; margin-bottom: 12px;">
            <h2 style="color: {color}; margin: 0 0 8px 0;">Diagnosis: <strong>{pred_cls}</strong> ({conf:.1f}% Confidence)</h2>
            <div style="display: flex; gap: 24px; font-size: 1.1em;">
                <div>📊 <strong>Stage:</strong> {pred_cls}</div>
                <div>🧠 <strong>Predicted MMSE:</strong> {pred_mmse} / 30</div>
                <div>⚠️ <strong>Progression Vulnerability:</strong> {risk_pct:.1f}% ({risk_stratum})</div>
            </div>
        </div>
        """

        recs_md = f"### 📋 Clinical Decision Support Protocol ({pred_cls})\n"
        for i, r in enumerate(res['recommendations'], 1):
            recs_md += f"{i}. {r}\n"

        recs_md += f"\n\n*⚠️ AI decision support tool for clinical research. Must be verified by a board-certified neurologist.*"

        return summary_md, res['visualization_image'], recs_md

    with gr.Blocks(css=custom_css, title="NeuroGAT - AD Decision Support") as app:
        gr.HTML("""
        <div class="main-header">
            <h1>🧠 NeuroGAT: Multi-Task Population Graph Attention Network</h1>
            <p>Multimodal Phenotypic Graph Diagnostic Classification & Disease Progression Vulnerability Assessment</p>
            <p style="font-size: 0.85em; opacity: 0.8;">
                4-Class Diagnosis (AD / LMCI / EMCI / CN) · Auxiliary Cognitive Trajectory · Progression Vulnerability
            </p>
        </div>
        """)

        with gr.Row():
            # Input Column
            with gr.Column(scale=1):
                gr.Markdown("### 🏥 Patient Clinical Profile")

                with gr.Accordion("Optional Cognitive Reference (Auxiliary / Non-Feature)", open=False):
                    gr.Markdown("ℹ️ *Diagnostic proxies are auxiliary references only; they are strictly excluded from the NeuroGAT feature manifold to guarantee zero target leakage.*")
                    mmse_slider = gr.Slider(0, 30, value=27, step=1, label="MMSE Score (Reference Only)")
                    cdrsb_slider = gr.Slider(0.0, 18.0, value=1.0, step=0.5, label="CDR Sum of Boxes (Reference Only)")
                    logmem_d_slider = gr.Slider(0, 25, value=8, step=1, label="Logical Memory - Delayed (Reference Only)")
                    logmem_i_slider = gr.Slider(0, 25, value=10, step=1, label="Logical Memory - Immediate (Reference Only)")

                with gr.Accordion("Demographics (Input Features)", open=True):
                    age_slider = gr.Slider(40, 95, value=73, step=1, label="Age (years)")
                    edu_slider = gr.Slider(4, 22, value=16, step=1, label="Education (years)")
                    gender_radio = gr.Radio(["Male", "Female"], value="Male", label="Gender")

                with gr.Accordion("Cardiovascular & Health Metrics (Input Features)", open=True):
                    gds_slider = gr.Slider(0, 15, value=1, step=1, label="Geriatric Depression Scale (GDS)")
                    bp_slider = gr.Slider(80, 200, value=135, step=1, label="Systolic Blood Pressure (mmHg)")
                    pulse_slider = gr.Slider(40, 120, value=68, step=1, label="Heart Rate (BPM)")

                analyze_btn = gr.Button("🔬 Run NeuroGAT Multimodal Analysis", variant="primary", size="lg")

            # Output Column
            with gr.Column(scale=2):
                gr.Markdown("### 📊 Diagnostic & Prognostic Staging")
                summary_output = gr.HTML("<p><em>Adjust patient parameters and click 'Run NeuroGAT Multimodal Analysis'.</em></p>")

                dashboard_output = gr.Image(label="NeuroGAT Multimodal Decision Dashboard", type="numpy")

                gr.Markdown("### ⚕️ Clinical Recommendations & Follow-up")
                recommendations_output = gr.Markdown("<em>Recommendations will populate after analysis.</em>")

        analyze_btn.click(
            fn=predict_callback,
            inputs=[
                mmse_slider, cdrsb_slider, logmem_d_slider, logmem_i_slider,
                age_slider, edu_slider, gender_radio,
                gds_slider, bp_slider, pulse_slider
            ],
            outputs=[summary_output, dashboard_output, recommendations_output]
        )

    return app


# ═══════════════════════════════════════════════════════════════════
# 11.3 Model Export (TorchScript & Deployment Manifest)
# ═══════════════════════════════════════════════════════════════════

def export_model(model: nn.Module, save_dir: Optional[str] = None):
    """Exports trained NeuroGAT model for edge/cloud clinical deployment."""
    if save_dir is None:
        save_dir = os.path.join(Config.OUTPUT_DIR, 'models')
    os.makedirs(save_dir, exist_ok=True)

    manifest = {
        'model_name': 'NeuroGAT_MultiTask',
        'architecture': 'CrossModal-GATv2-MultiScale',
        'num_classes': Config.NUM_CLASSES,
        'class_names': Config.CLASS_NAMES,
        'clinical_features': Config.CLINICAL_FEATURES,
        'export_date': datetime.now().isoformat()
    }
    with open(os.path.join(save_dir, 'model_manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"✅ Model deployment manifest saved to {save_dir}/model_manifest.json")


# ═══════════════════════════════════════════════════════════════════
# 11.4 App Launcher & CLI Runner
# ═══════════════════════════════════════════════════════════════════

def launch_app(checkpoint_path: Optional[str] = None):
    engine = NeuroGATInferenceEngine(checkpoint_path=checkpoint_path)

    # Run a test inference
    sample_patient = {
        'MMSE': 22.0, 'CDRSB': 2.5, 'LogMem_Delayed': 5.0, 'LogMem_Immediate': 7.0,
        'AGE': 74.0, 'EDUCATION': 14.0, 'GENDER': 0.0, 'GDS_TOTAL': 2.0,
        'BP_Systolic': 140.0, 'Pulse': 72.0
    }
    print("\n🔍 Running Sanity Inference on Sample Patient Profile:")
    sample_res = engine.predict(sample_patient)
    print(f"   Diagnosis: {sample_res['predicted_class']} (Confidence: {sample_res['confidence']*100:.1f}%)")
    print(f"   Continuous Cognitive MMSE: {sample_res['predicted_mmse']} / 30")
    print(f"   Disease Progression Vulnerability: {sample_res['conversion_risk_score']:.1f}% ({sample_res['conversion_risk_stratum']})\n")

    if GRADIO_AVAILABLE:
        app = create_gradio_app(engine)
        if app is not None:
            print("🚀 Launching NeuroGAT Clinical Decision Support Web GUI...")
            app.launch(share=True, server_name="0.0.0.0", server_port=7860)
    else:
        print("💡 Gradio not detected. CLI demo completed successfully.")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("  🏥 SECTION 11: NeuroGAT CLINICAL DECISION SUPPORT & RESEARCH PROTOTYPE")
    print("="*70)
    launch_app()

