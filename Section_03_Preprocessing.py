#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║      SECTION 3: 3D MRI PREPROCESSING PIPELINE                              ║
║      Skull Stripping, Registration, Normalization, Resizing                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import glob
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy import ndimage
from tqdm import tqdm

try:
    import nibabel as nib
except ImportError:
    nib = None

try:
    from skimage import filters
except ImportError:
    filters = None

try:
    from Section_01_Setup_Configuration import Config
except (ImportError, ModuleNotFoundError):
    pass

# ═══════════════════════════════════════════════════════════════════
# 3.1 Discover NIfTI Files
# ═══════════════════════════════════════════════════════════════════

def discover_nifti_files(data_paths: dict, csv_paths: dict) -> pd.DataFrame:
    """
    Discover all NIfTI files and match them with CSV metadata.
    Returns a DataFrame with columns: [nifti_path, subject_id, image_id, label, label_idx]
    """
    all_records = []

    for group, data_dir in data_paths.items():
        print(f"\n🔍 Scanning {group} directory: {data_dir}")

        # Find all NIfTI files
        nifti_files = glob.glob(os.path.join(data_dir, '**', '*.nii'), recursive=True)
        nifti_files += glob.glob(os.path.join(data_dir, '**', '*.nii.gz'), recursive=True)

        if not nifti_files:
            print(f"  ⚠️  No NIfTI files found in {data_dir}")
            continue

        print(f"  Found {len(nifti_files)} NIfTI files")

        # Load corresponding CSV for metadata
        csv_path = csv_paths.get(group, '')
        csv_data = {}
        if os.path.exists(csv_path):
            csv_df = pd.read_csv(csv_path)
            for _, row in csv_df.iterrows():
                img_id = str(row['Image Data ID']).strip()
                csv_data[img_id] = {
                    'subject_id': row['Subject'],
                    'sex': row.get('Sex', ''),
                    'age': row.get('Age', 0),
                    'visit': row.get('Visit', ''),
                }

        for nifti_path in nifti_files:
            # Extract Image Data ID from path (e.g., I241350)
            path_parts = Path(nifti_path).parts
            image_id = None
            subject_id = None

            for part in path_parts:
                if part.startswith('I') and part[1:].isdigit():
                    image_id = part
                # Subject ID pattern: XXX_S_XXXX
                if '_S_' in part and len(part) >= 9:
                    subject_id = part

            # Also try extracting from filename
            filename = os.path.basename(nifti_path)
            if image_id is None:
                for part in filename.split('_'):
                    if part.startswith('I') and part[1:].replace('.nii', '').replace('.gz', '').isdigit():
                        image_id = part.replace('.nii', '').replace('.gz', '')

            if subject_id is None:
                # Try to find subject ID from parent directories
                for part in path_parts:
                    if '_S_' in part:
                        subject_id = part
                        break

            record = {
                'nifti_path': nifti_path,
                'subject_id': subject_id or 'unknown',
                'image_id': image_id or 'unknown',
                'label': group,
                'label_idx': Config.CLASS_TO_IDX[group],
            }

            # Add CSV metadata if available
            if image_id and image_id in csv_data:
                record.update(csv_data[image_id])

            all_records.append(record)

    df = pd.DataFrame(all_records)
    print(f"\n✅ Total NIfTI files discovered: {len(df)}")
    print(f"   Per class: {dict(df['label'].value_counts())}")
    print(f"   Unique subjects: {df['subject_id'].nunique()}")

    return df


# ═══════════════════════════════════════════════════════════════════
# 3.2 NIfTI Loading
# ═══════════════════════════════════════════════════════════════════

def load_nifti(path: str) -> Tuple[np.ndarray, Any]:
    """Load a NIfTI file and return the volume data and affine matrix."""
    img = nib.load(path)
    data = img.get_fdata().astype(np.float32)
    affine = img.affine
    return data, affine


# ═══════════════════════════════════════════════════════════════════
# 3.3 Reorientation to RAS+
# ═══════════════════════════════════════════════════════════════════

def reorient_to_ras(img_data: np.ndarray, affine: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reorient a 3D volume to RAS+ (Right-Anterior-Superior) orientation.
    Uses nibabel's orientation utilities.
    """
    img = nib.Nifti1Image(img_data, affine)
    # Get current orientation
    orig_ornt = nib.io_orientation(affine)
    # Target RAS+ orientation
    ras_ornt = nib.orientations.axcodes2ornt(('R', 'A', 'S'))
    # Compute transformation
    transform = nib.orientations.ornt_transform(orig_ornt, ras_ornt)
    # Apply
    reoriented = nib.orientations.apply_orientation(img_data, transform)
    new_affine = affine.copy()  # Simplified - ideally update affine too
    return reoriented, new_affine


# ═══════════════════════════════════════════════════════════════════
# 3.4 Intensity Clipping (Replaces Brittle Skull Stripping)
# ═══════════════════════════════════════════════════════════════════

def remove_background_and_clip(volume: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Robust intensity clipping and background removal.
    Instead of brittle morphological skull stripping which can destroy the cortex,
    we rely on the model's attention mechanism to ignore the skull.
    We clip intensities to the 1st-99th percentile to remove bright noise/fat,
    and mask out the background.
    """
    # Create basic background mask (Otsu) without morphology
    pos_voxels = volume[volume > 0]
    if len(pos_voxels) == 0:
        return volume, np.ones_like(volume, dtype=bool)

    if filters is not None:
        try:
            threshold = float(filters.threshold_otsu(pos_voxels))
            binary_mask = volume > (threshold * 0.2)  # Low threshold just to capture the head
        except Exception:
            threshold = float(np.percentile(pos_voxels, 15))
            binary_mask = volume > threshold
    else:
        threshold = float(np.percentile(pos_voxels, 15))
        binary_mask = volume > threshold

    # Clip extreme intensities (remove bright artifacts)
    non_zero = volume[binary_mask]
    if len(non_zero) > 0:
        p1, p99 = np.percentile(non_zero, (1, 99))
        volume = np.clip(volume, p1, p99)

    # Apply mask to strictly zero out the background
    volume = volume * binary_mask.astype(np.float32)

    return volume, binary_mask


# ═══════════════════════════════════════════════════════════════════
# 3.5 Min-Max Normalization (Fixes Gray Background)
# ═══════════════════════════════════════════════════════════════════

def normalize_intensity(volume: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Min-Max scaling [0, 1] for brain pixels.
    Background remains perfectly 0.0 (Black).
    """
    brain_voxels = volume[mask > 0]

    if len(brain_voxels) == 0:
        return volume

    v_min, v_max = brain_voxels.min(), brain_voxels.max()
    normalized = np.zeros_like(volume, dtype=np.float32)

    if v_max > v_min:
        normalized[mask > 0] = (volume[mask > 0] - v_min) / (v_max - v_min)

    return normalized


# ═══════════════════════════════════════════════════════════════════
# 3.6 Resize Volume
# ═══════════════════════════════════════════════════════════════════

def resize_volume(volume: np.ndarray, target_size: Tuple[int, int, int] = (128, 128, 128)) -> np.ndarray:
    """
    Resize a 3D volume to target size using GPU-accelerated PyTorch trilinear interpolation.
    Massively faster than scipy.ndimage.zoom for large datasets.
    """
    current_shape = volume.shape[:3]

    if current_shape == target_size:
        return volume

    import torch
    import torch.nn.functional as F

    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Convert numpy array to PyTorch tensor and add batch+channel dims: (1, 1, D, H, W)
    vol_tensor = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float32)

    # Interpolate using GPU (Trilinear)
    with torch.no_grad():
        resized_tensor = F.interpolate(vol_tensor, size=target_size, mode='trilinear', align_corners=False)

    # Move back to CPU and return as numpy array
    resized = resized_tensor.squeeze().cpu().numpy()

    return resized.astype(np.float32)



# ═══════════════════════════════════════════════════════════════════
# 3.7 Crop to Brain Bounding Box
# ═══════════════════════════════════════════════════════════════════

def crop_to_brain(volume: np.ndarray, mask: np.ndarray, margin: int = 5) -> np.ndarray:
    """Crop volume to tight bounding box around the brain with margin."""
    # Find bounding box of the brain mask
    coords = np.where(mask > 0)
    if len(coords[0]) == 0:
        return volume

    min_coords = [max(0, c.min() - margin) for c in coords]
    max_coords = [min(s, c.max() + margin + 1) for c, s in zip(coords, volume.shape)]

    cropped = volume[
        min_coords[0]:max_coords[0],
        min_coords[1]:max_coords[1],
        min_coords[2]:max_coords[2]
    ]

    return cropped


# ═══════════════════════════════════════════════════════════════════
# 3.8 Complete Preprocessing Pipeline
# ═══════════════════════════════════════════════════════════════════

def preprocess_single_volume(
    nifti_path: str,
    target_size: Tuple[int, int, int] = (128, 128, 128)
) -> Optional[np.ndarray]:
    """
    Full preprocessing pipeline for a single 3D MRI volume.

    Steps:
    1. Load NIfTI
    2. Reorient to RAS+
    3. Intensity Clipping & Background Removal
    4. Crop to brain bounding box
    5. Resize to target size (GPU Accelerated)
    6. Min-Max Normalization

    Returns: preprocessed volume (target_size) or None if failed.
    """
    try:
        # Step 1: Load
        data, affine = load_nifti(nifti_path)

        # Handle 4D data (take first volume)
        if data.ndim == 4:
            data = data[:, :, :, 0]

        if data.ndim != 3:
            print(f"  ⚠️  Unexpected dimensions: {data.shape}")
            return None

        # Step 2: Reorient to RAS+
        data, affine = reorient_to_ras(data, affine)

        # Step 3: Intensity Clipping (robust background removal)
        data, brain_mask = remove_background_and_clip(data)

        # Step 4: Crop to brain bounding box
        data = crop_to_brain(data, brain_mask)

        # Step 5: Resize to target size
        data = resize_volume(data, target_size)

        # Step 6: Min-Max Normalization (Keeps background 0)
        # We need a new mask since the volume was cropped and resized
        new_mask = data > 0
        data = normalize_intensity(data, new_mask)

        # Clip extreme values
        data = np.clip(data, -5, 5)

        return data

    except Exception as e:
        print(f"  ❌ Error processing {nifti_path}: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════
# 3.9 Batch Preprocessing Pipeline
# ═══════════════════════════════════════════════════════════════════

class PreprocessingPipeline:
    """
    Batch preprocessing pipeline that processes all NIfTI files
    and saves preprocessed volumes as .npy files for fast loading.
    """

    def __init__(self, output_dir: str = Config.PREPROCESSED_DIR,
                 target_size: Tuple[int, int, int] = Config.INPUT_SIZE):
        self.output_dir = output_dir
        self.target_size = target_size
        os.makedirs(output_dir, exist_ok=True)

    def process_all(self, file_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process all NIfTI files in the DataFrame.
        Adds 'preprocessed_path' column and 'preprocess_success' column.
        """
        preprocessed_paths = []
        successes = []

        print(f"\n🧠 Starting preprocessing of {len(file_df)} volumes...")
        print(f"   Target size: {self.target_size}")
        print(f"   Output directory: {self.output_dir}")

        start_time = time.time()

        for idx, row in tqdm(file_df.iterrows(), total=len(file_df),
                              desc="Preprocessing MRI volumes"):
            nifti_path = row['nifti_path']
            image_id = row.get('image_id', f'vol_{idx}')
            label = row['label']

            # Output path
            out_filename = f"{label}_{image_id}.npz"
            out_path = os.path.join(self.output_dir, out_filename)

            # Skip if already preprocessed
            if os.path.exists(out_path):
                preprocessed_paths.append(out_path)
                successes.append(True)
                continue

            # Process
            volume = preprocess_single_volume(nifti_path, self.target_size)

            if volume is not None:
                try:
                    # Save as compressed float16 (massively reduces disk usage from 8MB to <1MB per volume)
                    np.savez_compressed(out_path, volume=volume.astype(np.float16))
                    preprocessed_paths.append(out_path)
                    successes.append(True)
                except OSError as e:
                    print(f"\n  ❌ Disk full or write error: {e}")
                    print(f"  ⚠️  Stopping preprocessing at volume {idx}. "
                          f"Successfully saved {sum(successes)} volumes so far.")
                    preprocessed_paths.append(None)
                    successes.append(False)
                    break
            else:
                preprocessed_paths.append(None)
                successes.append(False)

        # Pad lists if stopped early
        while len(preprocessed_paths) < len(file_df):
            preprocessed_paths.append(None)
            successes.append(False)

        elapsed = time.time() - start_time
        file_df = file_df.copy()
        file_df['preprocessed_path'] = preprocessed_paths
        file_df['preprocess_success'] = successes

        n_success = sum(successes)
        n_fail = len(successes) - n_success

        print(f"\n✅ Preprocessing Complete!")
        print(f"   Successful: {n_success}/{len(file_df)}")
        print(f"   Failed: {n_fail}")
        print(f"   Time: {format_time(elapsed)}")
        print(f"   Avg per volume: {elapsed/max(len(file_df),1):.1f}s")

        return file_df

    def get_preprocessed_stats(self, file_df: pd.DataFrame) -> None:
        """Print statistics of preprocessed data."""
        successful = file_df[file_df['preprocess_success'] == True]
        print(f"\n📊 Preprocessed Data Statistics:")
        print(f"   Total volumes: {len(successful)}")
        print(f"   Per class:")
        for group in Config.CLASS_NAMES:
            n = len(successful[successful['label'] == group])
            print(f"     {group}: {n}")


# ═══════════════════════════════════════════════════════════════════
# 3.10 Before/After Visualization
# ═══════════════════════════════════════════════════════════════════

def visualize_preprocessing(nifti_path: str) -> None:
    """Show before and after preprocessing for a single volume."""
    # Load original
    orig_data, affine = load_nifti(nifti_path)
    if orig_data.ndim == 4:
        orig_data = orig_data[:, :, :, 0]

    # Preprocess
    processed = preprocess_single_volume(nifti_path)

    if processed is None:
        print("❌ Preprocessing failed for this volume")
        return

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    views = ['Sagittal', 'Coronal', 'Axial']

    # Original (top row)
    orig_mid = [s // 2 for s in orig_data.shape[:3]]
    orig_slices = [orig_data[orig_mid[0], :, :], orig_data[:, orig_mid[1], :], orig_data[:, :, orig_mid[2]]]

    for col, (sl, view) in enumerate(zip(orig_slices, views)):
        axes[0, col].imshow(np.rot90(sl), cmap='gray', aspect='auto')
        axes[0, col].set_title(f'Original - {view}', fontsize=13, fontweight='bold')
        axes[0, col].axis('off')

    # Preprocessed (bottom row)
    proc_mid = [s // 2 for s in processed.shape[:3]]
    proc_slices = [processed[proc_mid[0], :, :], processed[:, proc_mid[1], :], processed[:, :, proc_mid[2]]]

    for col, (sl, view) in enumerate(zip(proc_slices, views)):
        axes[1, col].imshow(np.rot90(sl), cmap='gray', aspect='auto')
        axes[1, col].set_title(f'Preprocessed - {view}', fontsize=13, fontweight='bold', color='green')
        axes[1, col].axis('off')

    plt.suptitle('3D MRI Preprocessing: Before vs After', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(Config.FIGURES_DIR, 'fig_preprocessing_comparison.png'))
    plt.show()
    print("✅ Preprocessing visualization saved")


# ═══════════════════════════════════════════════════════════════════
# 3.11 Run Preprocessing
# ═══════════════════════════════════════════════════════════════════

def run_preprocessing():
    """Execute the complete preprocessing pipeline or load from checkpoint if exists."""
    checkpoint_path = os.path.join(Config.OUTPUT_DIR, 'processed_files.csv')
    if os.path.exists(checkpoint_path):
        print(f"📂 Found existing preprocessed files manifest. Loading: {checkpoint_path}")
        processed_df = pd.read_csv(checkpoint_path)
        print(f"✅ Loaded preprocessed manifest: {len(processed_df)} rows")
        return processed_df

    print("\n" + "=" * 70)
    print("  RUNNING 3D MRI PREPROCESSING PIPELINE")
    print("=" * 70)

    # Discover files
    file_df = discover_nifti_files(Config.DATA_PATHS, Config.CSV_PATHS)

    # Visualize one sample before/after
    if len(file_df) > 0:
        sample_path = file_df.iloc[0]['nifti_path']
        print(f"\n📸 Visualizing preprocessing on sample: {os.path.basename(sample_path)}")
        visualize_preprocessing(sample_path)

    # Run batch preprocessing
    pipeline = PreprocessingPipeline()
    processed_df = pipeline.process_all(file_df)
    pipeline.get_preprocessed_stats(processed_df)

    # Save processed dataframe for later use
    processed_df.to_csv(checkpoint_path, index=False)
    print(f"✅ File manifest saved to {checkpoint_path}")

    print(f"\n✅ Section 3 Complete - All volumes preprocessed!")
    return processed_df

# Run preprocessing
processed_df = run_preprocessing()

# Entrypoint alias for run_pipeline
main = run_preprocessing
