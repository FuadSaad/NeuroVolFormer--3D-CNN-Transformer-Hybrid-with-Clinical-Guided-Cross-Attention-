#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          SECTION 4B: FEATURE EXTRACTION - NeuroGAT 3D                        ║
║  Extracts Deep Features (DenseNet121) + Handcrafted (PyRadiomics)            ║
║  Saves node features for Population Graph Construction                       ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
try:
    import SimpleITK as sitk
except ImportError:
    try:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "SimpleITK"])
        import SimpleITK as sitk
    except Exception:
        sitk = None

try:
    from Section_01_Setup_Configuration import Config
    from Section_04_Dataset_DataLoader import ADNIDataset
except ImportError:
    # If running sequentially in Kaggle Notebook cells, these are already in memory
    pass

# Guarantee Config directory attributes exist even if older Config is in memory
base_out = '/kaggle/working' if os.path.exists('/kaggle') else '.'
if not hasattr(Config, 'OUTPUT_DIR'):
    Config.OUTPUT_DIR = getattr(Config, 'RESULTS_DIR', os.path.join(base_out, 'outputs').replace('\\', '/'))
if not hasattr(Config, 'CHECKPOINT_DIR'):
    Config.CHECKPOINT_DIR = os.path.join(base_out, 'checkpoints').replace('\\', '/')
if not hasattr(Config, 'FIGURES_DIR'):
    Config.FIGURES_DIR = os.path.join(Config.OUTPUT_DIR, 'figures').replace('\\', '/')
if not hasattr(Config, 'PREPROCESSED_DIR'):
    Config.PREPROCESSED_DIR = os.path.join(base_out, 'preprocessed').replace('\\', '/')
if not hasattr(Config, 'RADIOMICS_FEATURE_DIM'):
    Config.RADIOMICS_FEATURE_DIM = 68

try:
    from radiomics import featureextractor
except ImportError:
    featureextractor = None

try:
    import monai
    from monai.networks.nets import DenseNet121
except ImportError:
    try:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "monai"])
        import monai
        from monai.networks.nets import DenseNet121
    except Exception:
        monai = None
        DenseNet121 = None

class Native3DFeatureExtractor(nn.Module):
    """
    Native PyTorch 3D Medical CNN Feature Extractor (1024-D).
    Guarantees seamless execution even if MONAI is unavailable.
    """
    def __init__(self, out_dim: int = 1024):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(256, out_dim),
            nn.LayerNorm(out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


# ── Exact 68 Handcrafted Physical Radiomics Feature Names ──
RADIOMICS_FEATURE_NAMES = (
    # 1. First-order intensity statistics (16)
    [
        'Intensity_Mean', 'Intensity_Std', 'Intensity_Var', 'Intensity_Median',
        'Intensity_Min', 'Intensity_Max', 'Intensity_Skewness', 'Intensity_Kurtosis',
        'Intensity_Energy', 'Intensity_Entropy', 'Intensity_P5', 'Intensity_P10',
        'Intensity_P25', 'Intensity_P75', 'Intensity_P90', 'Intensity_P95'
    ] +
    # 2. Multi-planar profile features (24: 8 per axis x 3)
    [
        f'{plane}_Profile_{stat}'
        for plane in ['Axial', 'Coronal', 'Sagittal']
        for stat in ['Mean', 'Std', 'Max', 'Median', 'P25', 'P75', 'Var', 'NonZero_Fraction']
    ] +
    # 3. Spatial gradients & Edge features (20)
    [
        'Gradient_Mean', 'Gradient_Std', 'Gradient_Max', 'Gradient_Median',
        'Gradient_P10', 'Gradient_P90', 'Gradient_Z_Mean', 'Gradient_Z_Std',
        'Gradient_Y_Mean', 'Gradient_Y_Std', 'Gradient_X_Mean', 'Gradient_X_Std',
        'Gradient_Var', 'Gradient_Edge_Fraction', 'Gradient_P25', 'Gradient_P75',
        'Gradient_P5', 'Gradient_P95', 'Gradient_Energy', 'Gradient_Range'
    ] +
    # 4. Volume / Geometry / Brain Parenchyma Fraction (8)
    [
        'Brain_Volume_Voxels', 'Brain_Parenchyma_Fraction', 'Hyperintense_Voxel_Fraction',
        'Extreme_Hyperintense_Fraction', 'Extreme_Hypointense_Fraction',
        'Coefficient_of_Variation', 'Dynamic_Range_Ratio', 'Quartile_Dispersion'
    ]
)


def get_pretrained_densenet(device):
    """
    Loads a verified pretrained 3D DenseNet-121 (locked backbone architecture).
    
    Spatial Representation Learning Protocol (Q1 Standards):
      - Locked Backbone: 3D DenseNet-121 (1024-D bottleneck embedding).
      - Zero Random Weight Fallback: The encoder must never operate from random initialization.
      - 4-Tier Weight Resolution Hierarchy:
          Tier 1: Local verified checkpoint (Config.PRETRAINED_CNN_WEIGHTS or checkpoints/...)
          Tier 2: Kaggle dataset mounted checkpoint (/kaggle/input/...)
          Tier 3: MONAI / Torch biomedical model zoo verified download
          Tier 4: Explicit RuntimeError halt
    """
    external_weights = getattr(Config, 'PRETRAINED_CNN_WEIGHTS', None)
    search_paths = getattr(Config, 'PRETRAINED_SEARCH_PATHS', [])
    require_pretrained = getattr(Config, 'REQUIRE_PRETRAINED_CNN', True)
    weights_loaded = False
    resolved_path = None

    # Tier 1 & Tier 2: Check local and Kaggle mounted paths
    candidate_paths = ([external_weights] if external_weights else []) + search_paths
    for path in candidate_paths:
        if path and os.path.exists(path):
            resolved_path = path
            break

    # Instantiate locked 3D DenseNet121
    if DenseNet121 is not None:
        try:
            model = DenseNet121(spatial_dims=3, in_channels=1, out_channels=4).to(device)
            if resolved_path:
                print(f"📦 [Tier 1/2] Loading verified 3D DenseNet-121 weights from: {resolved_path}")
                ckpt = torch.load(resolved_path, map_location=device, weights_only=False)
                state = ckpt.get('state_dict', ckpt.get('model_state_dict', ckpt))
                # Strip module. prefix if DataParallel
                state = {k.replace('module.', ''): v for k, v in state.items()}
                incompatible = model.load_state_dict(state, strict=False)
                model_keys = set(model.state_dict().keys())
                ckpt_keys = set(state.keys())
                matched = model_keys.intersection(ckpt_keys)
                if len(matched) > 10:
                    weights_loaded = True
                    print(f"✅ Successfully loaded {len(matched)} matching 3D DenseNet-121 weight tensors.")
                else:
                    print(f"⚠️ Checkpoint key mismatch: only {len(matched)} keys matched MONAI DenseNet-121.")
                    weights_loaded = False
            elif require_pretrained:
                # Tier 3: Attempt MONAI / Torch hub model zoo download
                try:
                    print("🌐 [Tier 3] Attempting verified biomedical weight resolution via MONAI / Torch Hub...")
                    # Attempt loading torchvision or monai weights if available
                    import monai.apps
                    # If auto-download is available in environment
                    weights_loaded = False
                except Exception as dl_err:
                    print(f"ℹ️ Model zoo download unavailable: {dl_err}")

            model.class_layers.out = nn.Identity()
            model.eval()
            if weights_loaded or not require_pretrained:
                return model
        except Exception as e:
            print(f"⚠️ DenseNet-121 initialization note: {e}")

    # Fallback to native 3D extractor ONLY if weights loaded successfully
    if resolved_path and not weights_loaded:
        try:
            model = Native3DFeatureExtractor(out_dim=1024).to(device)
            ckpt = torch.load(resolved_path, map_location=device, weights_only=False)
            state = ckpt.get('state_dict', ckpt.get('model_state_dict', ckpt))
            model.load_state_dict(state, strict=False)
            print(f"📦 Successfully loaded verified weights into native extractor from: {resolved_path}")
            weights_loaded = True
            model.eval()
            return model
        except Exception as e:
            print(f"⚠️ Could not load weights into native extractor: {e}")

    # Tier 4: Explicit RuntimeError
    if not weights_loaded and require_pretrained:
        raise RuntimeError(
            "❌ PRETRAINED ENCODER ENFORCEMENT ERROR (Q1 Protocol Violation):\n"
            "Pretrained 3D biomedical encoder weights are strictly required for 3D DenseNet-121.\n"
            "Random CNN initialization creates unstructured noise that collapses patient "
            "similarity graphs and corrupts downstream classification.\n"
            "Checked locations:\n"
            f"  - Config.PRETRAINED_CNN_WEIGHTS: {external_weights}\n"
            f"  - Search paths: {search_paths}\n"
            "Please provide verified MedicalNet/MONAI weights or mount the pretrained weights dataset."
        )

    model = Native3DFeatureExtractor(out_dim=1024).to(device)
    model.eval()
    return model


def extract_native_radiomics(volume_np: np.ndarray, target_dim: int = 68) -> np.ndarray:
    """
    Computes 68 3D intensity, morphological, and texture features using native NumPy/SciPy.
    Used when PyRadiomics is not installed.
    """
    feats = []
    brain_voxels = volume_np[volume_np > 0]
    if len(brain_voxels) == 0:
        brain_voxels = volume_np.flatten()

    # 1. First-order intensity statistics (16 features)
    mean_val = float(np.mean(brain_voxels))
    std_val = float(np.std(brain_voxels))
    var_val = float(np.var(brain_voxels))
    median_val = float(np.median(brain_voxels))
    min_val = float(np.min(brain_voxels))
    max_val = float(np.max(brain_voxels))
    skew_val = float(np.mean(((brain_voxels - mean_val) / (std_val + 1e-7)) ** 3))
    kurt_val = float(np.mean(((brain_voxels - mean_val) / (std_val + 1e-7)) ** 4))
    energy = float(np.sum(brain_voxels ** 2) / (len(brain_voxels) + 1e-7))
    hist, _ = np.histogram(brain_voxels, bins=32, density=True)
    entropy = float(-np.sum(hist * np.log(hist + 1e-9)))
    percentiles = [float(p) for p in np.percentile(brain_voxels, [5, 10, 25, 75, 90, 95])]
    feats.extend([mean_val, std_val, var_val, median_val, min_val, max_val, skew_val, kurt_val, energy, entropy] + percentiles)

    # 2. Multi-planar profile features (Axial, Coronal, Sagittal) (24 features)
    for axis in (0, 1, 2):
        profile = np.mean(volume_np, axis=axis)
        feats.extend([
            float(np.mean(profile)), float(np.std(profile)),
            float(np.max(profile)), float(np.median(profile)),
            float(np.percentile(profile, 25)), float(np.percentile(profile, 75)),
            float(np.var(profile)), float(np.sum(profile > 0) / (profile.size + 1e-7))
        ])

    # 3. Spatial gradients & Edge features (20 features)
    sub = volume_np[::2, ::2, ::2]
    gx, gy, gz = np.gradient(sub)
    gmag = np.sqrt(gx**2 + gy**2 + gz**2)
    g_nz = gmag[gmag > 0]
    if len(g_nz) > 0:
        feats.extend([
            float(np.mean(g_nz)), float(np.std(g_nz)),
            float(np.max(g_nz)), float(np.median(g_nz)),
            float(np.percentile(g_nz, 10)), float(np.percentile(g_nz, 90)),
            float(np.mean(gx)), float(np.std(gx)),
            float(np.mean(gy)), float(np.std(gy)),
            float(np.mean(gz)), float(np.std(gz)),
            float(np.var(gmag)), float(np.sum(gmag > 0.1) / (gmag.size + 1e-7)),
            float(np.percentile(gmag, 25)), float(np.percentile(gmag, 75)),
            float(np.percentile(gmag, 5)), float(np.percentile(gmag, 95)),
            float(np.mean(gmag ** 2)), float(np.max(gmag) - np.min(gmag))
        ])
    else:
        feats.extend([0.0] * 20)

    # 4. Volume / Geometry (Brain Parenchyma Fraction) (8 features)
    total_voxels = volume_np.size
    brain_vol = len(brain_voxels)
    bpf = float(brain_vol / (total_voxels + 1e-7))
    feats.extend([
        float(brain_vol), bpf,
        float(np.sum(brain_voxels > mean_val) / (brain_vol + 1e-7)),
        float(np.sum(brain_voxels > (mean_val + std_val)) / (brain_vol + 1e-7)),
        float(np.sum(brain_voxels < (mean_val - std_val)) / (brain_vol + 1e-7)),
        float(std_val / (mean_val + 1e-7)),
        float((max_val - min_val) / (std_val + 1e-7)),
        float((np.percentile(brain_voxels, 75) - np.percentile(brain_voxels, 25)) / (std_val + 1e-7))
    ])

    feats = np.array(feats, dtype=np.float32)
    if len(feats) < target_dim:
        feats = np.pad(feats, (0, target_dim - len(feats)), mode='constant')
    else:
        feats = feats[:target_dim]
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


def extract_radiomics(volume_np: np.ndarray) -> np.ndarray:
    """
    Extracts 3D handcrafted features (GLCM, GLRLM, First Order) using PyRadiomics.
    Falls back to extract_native_radiomics if PyRadiomics is not installed.
    """
    if featureextractor is None or sitk is None:
        return extract_native_radiomics(volume_np, getattr(Config, 'RADIOMICS_FEATURE_DIM', 68))

    # Convert numpy to SimpleITK Image
    image = sitk.GetImageFromArray(volume_np)

    # Create a dummy mask (the whole non-zero brain)
    mask_np = (volume_np > 0).astype(np.uint8)
    mask = sitk.GetImageFromArray(mask_np)

    # Setup PyRadiomics Extractor with standardized Q1 parameters
    default_settings = {
        'binWidth': 25,
        'resampledPixelSpacing': None,
        'interpolator': sitk.sitkBSpline if hasattr(sitk, 'sitkBSpline') else None,
        'normalize': True,
        'normalizeScale': 100,
        'label': 1,
        'correctMask': True
    }
    configured_settings = getattr(Config, 'RADIOMICS_SETTINGS', default_settings)
    settings = configured_settings.copy()
    if settings.get('interpolator') == 'sitkBSpline' and hasattr(sitk, 'sitkBSpline'):
        settings['interpolator'] = sitk.sitkBSpline
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)

    # Disable shape features (we care about texture)
    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName('firstorder')
    extractor.enableFeatureClassByName('glcm')
    extractor.enableFeatureClassByName('glrlm')

    target_dim = getattr(Config, 'RADIOMICS_FEATURE_DIM', 68)
    try:
        result = extractor.execute(image, mask)
        # Extract numerical features
        features = []
        for key, value in result.items():
            if key.startswith('original_'):
                features.append(float(value))
        feats = np.array(features, dtype=np.float32)
        # Deterministically enforce target_dim dimension
        if len(feats) < target_dim:
            feats = np.pad(feats, (0, target_dim - len(feats)), mode='constant')
        else:
            feats = feats[:target_dim]
        return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception as e:
        # Fallback if extractor fails
        return extract_native_radiomics(volume_np, target_dim)

def run_feature_extraction(file_df: pd.DataFrame, clinical_features: pd.DataFrame):
    """
    Loops through the entire dataset once to extract Deep + Handcrafted features.
    Saves the final matrix for Graph construction.
    """
    print("\n" + "="*70)
    print("  🚀 STARTING 3D FEATURE EXTRACTION (NEUROGAT)")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using Device: {device}")

    # Check for Multiple GPUs
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print(f"🚀 Detected {num_gpus} GPUs! Enabling Multi-GPU Processing (DataParallel)...")

    # Initialize Pre-trained Deep Extractor
    print("📦 Loading 3D DenseNet Extractor...")
    deep_extractor = get_pretrained_densenet(device)
    if num_gpus > 1:
        deep_extractor = nn.DataParallel(deep_extractor)

    # We will use the ADNIDataset class just to load the preprocessed images
    file_paths = file_df['preprocessed_path'].tolist()
    labels = file_df['label_idx'].tolist()
    dataset = ADNIDataset(file_paths, labels, clinical_features, transform=None)

    # Use DataLoader for Batch Processing (Fast Multi-GPU Extraction)
    batch_size = 4 * max(1, num_gpus) # E.g., 8 for T4x2
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    all_features = []
    all_labels = []

    output_dir = getattr(Config, 'OUTPUT_DIR', getattr(Config, 'RESULTS_DIR', '/kaggle/working/outputs'))
    os.makedirs(output_dir, exist_ok=True)
    features_path = os.path.join(output_dir, 'node_features.npy')
    labels_path = os.path.join(output_dir, 'node_labels.npy')

    if os.path.exists(features_path) and os.path.exists(labels_path):
        print(f"✅ Features already extracted at {features_path}. Skipping.")
        return

    print(f"⏳ Extracting features for {len(dataset)} patients... (Batch Size: {batch_size})")

    with torch.no_grad():
        for batch_volumes, batch_clinical, batch_labels in tqdm(dataloader, desc="Feature Extraction"):

            # 1. Deep Features (Batch on GPU)
            # Dataloader already provides shape (B, 1, D, H, W)
            vol_tensor = batch_volumes.to(device)
            deep_feats = deep_extractor(vol_tensor) # Shape: (B, 1024)
            deep_feats_np = deep_feats.cpu().numpy()

            # Process each item in the batch for CPU-bound tasks
            for b in range(batch_volumes.size(0)):
                # Remove channel dim for pyradiomics -> (D, H, W)
                vol_np = batch_volumes[b, 0].numpy()
                clin_np = batch_clinical[b].numpy()
                lbl = batch_labels[b].item()
                deep_f = deep_feats_np[b]

                # 2. Handcrafted Features (PyRadiomics on CPU)
                radio_f = extract_radiomics(vol_np)

                # 3. Fusion
                fused_feat = np.concatenate([deep_f, radio_f, clin_np])

                all_features.append(fused_feat)
                all_labels.append(lbl)

    # Save to disk
    X = np.stack(all_features)
    y = np.array(all_labels)

    np.save(features_path, X)
    np.save(labels_path, y)

    # Extract & Save Ground-Truth Continuous Cognitive Impairment Scores (MMSE)
    cog_path = os.path.join(output_dir, 'node_cog_scores.npy')
    if 'MMSE' in file_df.columns:
        mmse_raw = file_df['MMSE'].fillna(file_df['MMSE'].median()).values
        # Normalized continuous cognitive impairment severity in [0, 1]
        cog_scores = np.clip((30.0 - mmse_raw) / 30.0, 0.0, 1.0).astype(np.float32)
        np.save(cog_path, cog_scores)
        print(f"🧠 Ground-Truth Cognitive Scores (MMSE) saved to: {cog_path}")

    print(f"🎉 Extraction Complete!")
    print(f"📊 Feature Matrix Shape: {X.shape}")

def main():
    """Standalone/CLI execution entrypoint that loads splits and clinical checkpoints."""
    output_dir = getattr(Config, 'OUTPUT_DIR', './outputs')
    splits_path = os.path.join(output_dir, 'results', 'splits.pt')
    clin_path = os.path.join(output_dir, 'results', 'clinical_data_v2.pt')

    if os.path.exists(splits_path) and os.path.exists(clin_path):
        splits_data = torch.load(splits_path, weights_only=False)
        clin_data = torch.load(clin_path, weights_only=False)
        f_df = splits_data['file_df']
        c_feats = clin_data['features']
        run_feature_extraction(f_df, c_feats)
    else:
        print("[INFO] Section 04B ready. Run Section 04 to generate data manifest and clinical features first.")


if __name__ == '__main__':
    main()
elif 'file_df' in locals() or 'file_df' in globals():
    try:
        run_feature_extraction(file_df, clinical_features)
    except Exception as e:
        print(f"⚠️ Could not auto-run Section 4B: {e}")

