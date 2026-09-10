#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║      SECTION 4: DATASET, DATALOADER & PATIENT-LEVEL SPLITTING              ║
║      3D Augmentation, Stratified CV, Class Imbalance Handling              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ═══════════════════════════════════════════════════════════════════
# 4.1 3D Data Augmentation
# ═══════════════════════════════════════════════════════════════════

class RandomFlip3D:
    """Random left-right flip (anatomically valid for brain MRI)."""
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, volume: np.ndarray) -> np.ndarray:
        if random.random() < self.p:
            volume = np.flip(volume, axis=0).copy()  # Left-right flip
        return volume


class RandomRotation3D:
    """Random rotation along all 3 axes."""
    def __init__(self, max_angle: float = 10.0, p: float = 0.5):
        self.max_angle = max_angle
        self.p = p

    def __call__(self, volume: np.ndarray) -> np.ndarray:
        if random.random() < self.p:
            angles = [random.uniform(-self.max_angle, self.max_angle) for _ in range(3)]
            # Rotate along each axis
            for axis_pair, angle in zip([(1, 2), (0, 2), (0, 1)], angles):
                volume = ndimage.rotate(volume, angle, axes=axis_pair,
                                         reshape=False, order=1, mode='constant', cval=0)
        return volume


class RandomNoise3D:
    """Add random Gaussian noise."""
    def __init__(self, std_range: Tuple[float, float] = (0.0, 0.03), p: float = 0.5):
        self.std_range = std_range
        self.p = p

    def __call__(self, volume: np.ndarray) -> np.ndarray:
        if random.random() < self.p:
            std = random.uniform(*self.std_range)
            noise = np.random.normal(0, std, volume.shape).astype(np.float32)
            volume = volume + noise
        return volume


class RandomIntensityShift3D:
    """Random intensity shift and scale."""
    def __init__(self, shift_range: float = 0.1, scale_range: Tuple[float, float] = (0.9, 1.1), p: float = 0.5):
        self.shift_range = shift_range
        self.scale_range = scale_range
        self.p = p

    def __call__(self, volume: np.ndarray) -> np.ndarray:
        if random.random() < self.p:
            shift = random.uniform(-self.shift_range, self.shift_range)
            scale = random.uniform(*self.scale_range)
            volume = volume * scale + shift
        return volume


class RandomCutout3D:
    """Randomly zero out a cubic region (3D CutOut)."""
    def __init__(self, max_size: int = 20, p: float = 0.3):
        self.max_size = max_size
        self.p = p

    def __call__(self, volume: np.ndarray) -> np.ndarray:
        if random.random() < self.p:
            size = random.randint(5, self.max_size)
            h, w, d = volume.shape[:3]
            x = random.randint(0, max(0, h - size))
            y = random.randint(0, max(0, w - size))
            z = random.randint(0, max(0, d - size))
            volume = volume.copy()
            volume[x:x+size, y:y+size, z:z+size] = 0
        return volume


class Compose3D:
    """Compose multiple 3D augmentations."""
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, volume: np.ndarray) -> np.ndarray:
        for t in self.transforms:
            volume = t(volume)
        return volume


def get_train_transforms() -> Compose3D:
    """Get training augmentation pipeline."""
    return Compose3D([
        RandomFlip3D(p=0.5),
        RandomRotation3D(max_angle=10.0, p=0.5),
        RandomNoise3D(std_range=(0.0, 0.03), p=0.4),
        RandomIntensityShift3D(shift_range=0.1, p=0.4),
        RandomCutout3D(max_size=20, p=0.3),
    ])


def get_val_transforms() -> None:
    """No augmentation for validation/test."""
    return None


# ═══════════════════════════════════════════════════════════════════
# 4.2 ADNI Dataset Class
# ═══════════════════════════════════════════════════════════════════

class ADNIDataset(Dataset):
    """
    PyTorch Dataset for ADNI 3D MRI data with clinical features.

    Args:
        file_paths: List of paths to preprocessed .npy files
        labels: List of integer labels (0=AD, 1=CN, 2=EMCI, 3=LMCI)
        clinical_features: numpy array of shape (N, 10) or None
        transform: 3D augmentation transforms
    """

    def __init__(
        self,
        file_paths: List[str],
        labels: List[int],
        clinical_features: Optional[np.ndarray] = None,
        transform: Optional[Compose3D] = None
    ):
        self.file_paths = file_paths
        self.labels = labels
        self.clinical_features = clinical_features
        self.transform = transform

        assert len(file_paths) == len(labels), \
            f"Mismatch: {len(file_paths)} paths vs {len(labels)} labels"
        if clinical_features is not None:
            assert len(clinical_features) == len(labels), \
                f"Mismatch: {len(clinical_features)} clinical vs {len(labels)} labels"

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        # Load preprocessed volume
        try:
            # Load from compressed .npz file
            data = np.load(self.file_paths[idx])
            volume = data['volume'].astype(np.float32)
        except Exception as e:
            # Explicit warning log for transparency and data integrity tracking
            print(f"⚠️ Warning: Could not load volume at index {idx} ({self.file_paths[idx]}): {e}. Supplying zero-volume fallback.")
            volume = np.zeros(Config.INPUT_SIZE, dtype=np.float32)

        # Apply augmentation
        if self.transform is not None:
            volume = self.transform(volume)

        # Convert to tensor: add channel dimension (1, D, H, W)
        volume_tensor = torch.from_numpy(volume).unsqueeze(0).float()

        # Clinical features
        if self.clinical_features is not None:
            clinical_tensor = torch.from_numpy(
                self.clinical_features[idx].astype(np.float32)
            )
        else:
            clinical_tensor = torch.zeros(Config.CLINICAL_DIM, dtype=torch.float32)

        # Label
        label = self.labels[idx]

        return volume_tensor, clinical_tensor, label


# ═══════════════════════════════════════════════════════════════════
# 4.3 Patient-Level Stratified Splitting
# ═══════════════════════════════════════════════════════════════════

def get_patient_level_split(
    file_df: pd.DataFrame,
    n_folds: int = Config.N_FOLDS,
    test_size: float = Config.TEST_SIZE,
    seed: int = Config.SEED
) -> Tuple[List[int], List[Tuple[List[int], List[int]]]]:
    """
    Split data at the patient (subject) level to prevent data leakage, or load existing splits.

    Returns:
        test_indices: indices for held-out test set
        fold_splits: list of (train_indices, val_indices) for each fold
    """
    checkpoint_path = os.path.join(Config.OUTPUT_DIR, 'results', 'splits.pt')
    if os.path.exists(checkpoint_path):
        print(f"📂 Found existing patient-level splits. Loading: {checkpoint_path}")
        splits_data = torch.load(checkpoint_path, weights_only=False)
        print(f"✅ Loaded splits: Test set size = {len(splits_data['test_indices'])} images")
        return splits_data['test_indices'], splits_data['fold_splits']

    # Get unique subjects with their labels
    subject_df = file_df.drop_duplicates('subject_id')[['subject_id', 'label_idx']].reset_index(drop=True)
    subjects = subject_df['subject_id'].values
    subject_labels = subject_df['label_idx'].values

    # Step 1: Split off 20% subjects as held-out test
    train_val_subjects, test_subjects, train_val_labels, test_labels = train_test_split(
        subjects, subject_labels, test_size=test_size,
        stratify=subject_labels, random_state=seed
    )

    # Get image indices for test set
    test_indices = file_df[file_df['subject_id'].isin(test_subjects)].index.tolist()

    print(f"\n📊 Patient-Level Data Split:")
    print(f"   Total subjects: {len(subjects)}")
    print(f"   Train/Val subjects: {len(train_val_subjects)} ({100*(1-test_size):.0f}%)")
    print(f"   Test subjects: {len(test_subjects)} ({100*test_size:.0f}%)")
    print(f"   Test images: {len(test_indices)}")

    # Step 2: 5-Fold Stratified CV on remaining subjects
    train_val_df = file_df[file_df['subject_id'].isin(train_val_subjects)].reset_index(drop=True)
    train_val_subject_df = train_val_df.drop_duplicates('subject_id').reset_index(drop=True)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    fold_splits = []
    for fold_idx, (train_subj_idx, val_subj_idx) in enumerate(
        skf.split(train_val_subject_df, train_val_subject_df['label_idx'])
    ):
        train_subjects_fold = train_val_subject_df.iloc[train_subj_idx]['subject_id'].values
        val_subjects_fold = train_val_subject_df.iloc[val_subj_idx]['subject_id'].values

        # Map back to image indices in original file_df
        train_img_indices = file_df[
            file_df['subject_id'].isin(train_subjects_fold)
        ].index.tolist()
        val_img_indices = file_df[
            file_df['subject_id'].isin(val_subjects_fold)
        ].index.tolist()

        fold_splits.append((train_img_indices, val_img_indices))

        print(f"   Fold {fold_idx+1}: Train={len(train_img_indices)} images "
              f"({len(train_subjects_fold)} subjects), "
              f"Val={len(val_img_indices)} images ({len(val_subjects_fold)} subjects)")

        # Verify no subject overlap
        overlap = set(train_subjects_fold) & set(val_subjects_fold)
        assert len(overlap) == 0, f"❌ Data leakage! {len(overlap)} subjects in both train and val!"

    # Verify test-train/val no overlap
    test_set = set(test_subjects)
    trainval_set = set(train_val_subjects)
    assert len(test_set & trainval_set) == 0, "❌ Data leakage between test and train/val!"

    print(f"\n✅ No data leakage detected — all splits are patient-level clean!")

    # Save checkpoint (includes file_df for downstream XAI and Grad-CAM)
    os.makedirs(os.path.join(Config.OUTPUT_DIR, 'results'), exist_ok=True)
    torch.save({'test_indices': test_indices, 'fold_splits': fold_splits, 'file_df': file_df}, checkpoint_path)
    print(f"💾 Saved patient-level splits checkpoint to: {checkpoint_path}")

    return test_indices, fold_splits


# ═══════════════════════════════════════════════════════════════════
# 4.4 Clinical Feature Standardization
# ═══════════════════════════════════════════════════════════════════

def prepare_clinical_features(
    file_df: pd.DataFrame,
    train_indices: Optional[List[int]] = None
) -> Tuple[np.ndarray, Optional[StandardScaler]]:
    """
    Prepare and standardize clinical demographic features without data leakage.
    Imputation medians and scaling parameters are strictly computed from train_indices.
    """
    checkpoint_path = os.path.join(Config.OUTPUT_DIR, 'results', 'clinical_data_v2.pt')
    if os.path.exists(checkpoint_path):
        try:
            data = torch.load(checkpoint_path, weights_only=False)
            if data['features'].shape[1] == Config.CLINICAL_DIM:
                print(f"📂 Found compatible clinical features checkpoint. Loading: {checkpoint_path}")
                return data['features'], data['scaler']
            else:
                print(f"ℹ️ Cached clinical features dimension ({data['features'].shape[1]}) differs from Config.CLINICAL_DIM ({Config.CLINICAL_DIM}). Recomputing fresh...")
        except Exception:
            pass

    clinical_cols = Config.CLINICAL_FEATURES
    available_cols = [c for c in clinical_cols if c in file_df.columns]

    # Programmatic Guard: Verify no forbidden diagnostic proxies exist in available_cols
    forbidden_vars = getattr(Config, 'FORBIDDEN_GRAPH_VARIABLES', {'MMSE', 'CDRSB', 'LogMem_Delayed', 'LogMem_Immediate', 'DX', 'DX_bl'})
    if not getattr(Config, 'INCLUDE_DIAGNOSTIC_PROXIES', False):
        leak_detected = set(available_cols) & forbidden_vars
        if leak_detected:
            raise ValueError(f"CRITICAL LEAKAGE DETECTED: Forbidden variables {leak_detected} present in clinical feature inputs!")

    if len(available_cols) == 0:
        print("⚠️  No clinical features available, returning zeros")
        features = np.zeros((len(file_df), Config.CLINICAL_DIM), dtype=np.float32)
        return features, None

    # Extract features, fill missing values strictly using training set medians to prevent leakage
    features_df = file_df[available_cols].copy()
    if train_indices is not None and len(train_indices) > 0:
        train_df = features_df.iloc[train_indices]
        for col in available_cols:
            med_val = train_df[col].median()
            features_df[col] = features_df[col].fillna(med_val)
    else:
        for col in available_cols:
            features_df[col] = features_df[col].fillna(features_df[col].median())

    features = features_df.values.astype(np.float32)

    # Pad to CLINICAL_DIM if fewer columns available
    if features.shape[1] < Config.CLINICAL_DIM:
        padding = np.zeros((features.shape[0], Config.CLINICAL_DIM - features.shape[1]), dtype=np.float32)
        features = np.hstack([features, padding])

    # Fit scaler strictly on training split
    scaler = None
    if train_indices is not None and len(train_indices) > 0:
        scaler = StandardScaler()
        scaler.fit(features[train_indices])
        features = scaler.transform(features)

    print(f"✅ Clinical features prepared: {features.shape} (Leak-free)")
    print(f"   Active demographic/clinical features: {available_cols}")

    # Save checkpoint
    os.makedirs(os.path.join(Config.OUTPUT_DIR, 'results'), exist_ok=True)
    torch.save({'features': features.astype(np.float32), 'scaler': scaler}, checkpoint_path)
    print(f"💾 Saved clinical features checkpoint to: {checkpoint_path}")

    return features.astype(np.float32), scaler


# ═══════════════════════════════════════════════════════════════════
# 4.5 Create DataLoaders
# ═══════════════════════════════════════════════════════════════════

def get_class_weights(labels: List[int]) -> torch.Tensor:
    """Compute class weights for imbalanced data: w_c = N_total / (K * N_c)."""
    counter = Counter(labels)
    n_total = len(labels)
    n_classes = len(counter)
    weights = torch.zeros(n_classes)
    for cls_idx in range(n_classes):
        n_cls = counter.get(cls_idx, 1)
        weights[cls_idx] = n_total / (n_classes * n_cls)
    return weights


def get_dataloaders(
    file_df: pd.DataFrame,
    clinical_features: np.ndarray,
    train_indices: List[int],
    val_indices: List[int],
    test_indices: List[int],
    batch_size: int = Config.BATCH_SIZE,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for train, validation, and test sets.
    Uses WeightedRandomSampler for training to handle class imbalance.
    """
    # Extract data for each split
    all_paths = file_df['preprocessed_path'].tolist()
    all_labels = file_df['label_idx'].tolist()

    # Training set
    train_paths = [all_paths[i] for i in train_indices]
    train_labels = [all_labels[i] for i in train_indices]
    train_clinical = clinical_features[train_indices]

    # Validation set
    val_paths = [all_paths[i] for i in val_indices]
    val_labels = [all_labels[i] for i in val_indices]
    val_clinical = clinical_features[val_indices]

    # Test set
    test_paths = [all_paths[i] for i in test_indices]
    test_labels = [all_labels[i] for i in test_indices]
    test_clinical = clinical_features[test_indices]

    # Create datasets
    train_dataset = ADNIDataset(train_paths, train_labels, train_clinical,
                                 transform=get_train_transforms())
    val_dataset = ADNIDataset(val_paths, val_labels, val_clinical, transform=None)
    test_dataset = ADNIDataset(test_paths, test_labels, test_clinical, transform=None)

    # WeightedRandomSampler for training
    class_counts = Counter(train_labels)
    sample_weights = [1.0 / class_counts[label] for label in train_labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler,
        num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY
    )

    print(f"\n✅ DataLoaders created:")
    print(f"   Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"   Val:   {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"   Test:  {len(test_dataset)} samples, {len(test_loader)} batches")
    print(f"   Train class distribution: {dict(class_counts)}")

    return train_loader, val_loader, test_loader


# ═══════════════════════════════════════════════════════════════════
# 4.6 RUN: Split Data, Prepare Clinical Features & Visualize
# ═══════════════════════════════════════════════════════════════════

def run_section_4(processed_df: pd.DataFrame) -> Tuple[List[int], List[Tuple[List[int], List[int]]], np.ndarray]:
    """Execute Section 4: Data splitting, clinical feature preparation, and visualization."""
    print("\n" + "=" * 70)
    print("  SECTION 4: DATA SPLITTING & PREPARATION")
    print("=" * 70)

    # Filter to only successfully preprocessed files
    file_df = processed_df[processed_df['preprocess_success'] == True].reset_index(drop=True)
    print(f"\n📊 Working with {len(file_df)} successfully preprocessed volumes")

    # ── MERGE CLINICAL FEATURES ──
    clinical_csv = Config.CLINICAL_CSV
    if clinical_csv and os.path.exists(clinical_csv):
        clin_df = pd.read_csv(clinical_csv)

        # Drop overlapping columns from clin_df (like 'label') to prevent '_x' and '_y' suffixes
        overlapping_cols = [c for c in clin_df.columns if c in file_df.columns and c != 'PTID']
        if overlapping_cols:
            clin_df = clin_df.drop(columns=overlapping_cols)

        # Merge on subject_id == PTID
        file_df = pd.merge(file_df, clin_df, left_on='subject_id', right_on='PTID', how='left')
        print(f"📂 Merged clinical features from {os.path.basename(clinical_csv)}")
    else:
        print(f"⚠️ Clinical CSV not found at {clinical_csv}! Clinical features will be missing.")

    # Step 0: Participant-Level Cohort De-duplication (Chronological Baseline Selection)
    if getattr(Config, 'ONE_SCAN_PER_SUBJECT', True):
        initial_count = len(file_df)

        # 1. Identify visit code column ('VISCODE', 'VISCODE2', 'VISIT')
        viscode_col = next((c for c in ['VISCODE', 'VISCODE2', 'VISIT', 'Visit'] if c in file_df.columns), None)
        # 2. Identify chronological scan/exam date column ('EXAMDATE', 'Acq Date', 'SCAN_DATE')
        date_col = next((c for c in ['EXAMDATE', 'Acq Date', 'SCAN_DATE', 'ScanDate', 'Study Date'] if c in file_df.columns), None)

        has_baseline_codes = False
        if viscode_col:
            # Issue 38: Strictly match genuine baseline visits ('bl', 'baseline', 'm00'); do NOT match screening ('sc')
            is_bl_mask = file_df[viscode_col].astype(str).str.strip().str.lower().isin(['bl', 'baseline', 'm00'])
            if is_bl_mask.any():
                has_baseline_codes = True
                file_df['__is_bl__'] = is_bl_mask

        if has_baseline_codes:
            sort_cols = ['__is_bl__']
            ascending_order = [False]
            if date_col:
                file_df['__parsed_date__'] = pd.to_datetime(file_df[date_col], errors='coerce')
                sort_cols.append('__parsed_date__')
                ascending_order.append(True)
            file_df = file_df.sort_values(sort_cols, ascending=ascending_order)
            file_df = file_df.drop_duplicates('subject_id', keep='first').drop(columns=['__is_bl__'])
            if '__parsed_date__' in file_df.columns:
                file_df = file_df.drop(columns=['__parsed_date__'])
            file_df = file_df.reset_index(drop=True)
            method_desc = f"explicit visit code ('{viscode_col}') prioritized"
        elif date_col:
            file_df['__parsed_date__'] = pd.to_datetime(file_df[date_col], errors='coerce')
            file_df = file_df.sort_values(['subject_id', '__parsed_date__'], ascending=[True, True])
            file_df = file_df.drop_duplicates('subject_id', keep='first').drop(columns=['__parsed_date__']).reset_index(drop=True)
            method_desc = f"earliest acquisition date ('{date_col}')"
        else:
            sort_col = 'image_id' if 'image_id' in file_df.columns else ('file_name' if 'file_name' in file_df.columns else 'subject_id')
            file_df = file_df.sort_values(sort_col).drop_duplicates('subject_id', keep='first').reset_index(drop=True)
            method_desc = f"subject indexing fallback"

        print("\n" + "-" * 50)
        print("  Step 0: Participant-Level Cohort De-duplication (Chronological Baseline)")
        print("-" * 50)
        print(f"👥 Cohort Filtered: Exactly 1 baseline scan per participant via {method_desc}.")
        print(f"   Refined from {initial_count} longitudinal scans to {len(file_df)} unique participants.")
        print(f"   The one-scan-per-subject protocol eliminates within-subject repeated-measure dependence.")

    # Step 1: Patient-level split
    print("\n" + "-" * 50)
    print("  Step 1: Patient-Level Stratified Split")
    print("-" * 50)
    test_indices, fold_splits = get_patient_level_split(file_df)

    # Step 2: Clinical feature preparation
    print("\n" + "-" * 50)
    print("  Step 2: Clinical Feature Standardization")
    print("-" * 50)
    # Fit strictly on development cohort (train + val) without touching held-out test cohort
    train_val_indices = [i for i in range(len(file_df)) if i not in set(test_indices)]
    clinical_features, scaler = prepare_clinical_features(file_df, train_val_indices)

    # Step 3: Visualize splits
    print("\n" + "-" * 50)
    print("  Step 3: Visualizing Data Splits")
    print("-" * 50)

    # 3a. Split distribution table
    print("\n📋 Dataset Split Summary:")
    print(f"{'Set':<15} {'Samples':>10} {'Percentage':>12}")
    print("-" * 40)
    total = len(file_df)
    test_count = len(test_indices)
    print(f"{'Test (Held-out)':<15} {test_count:>10} {100*test_count/total:>11.1f}%")
    for fold_idx, (train_idx, val_idx) in enumerate(fold_splits):
        print(f"{'Fold '+str(fold_idx+1)+' Train':<15} {len(train_idx):>10} {100*len(train_idx)/total:>11.1f}%")
        print(f"{'Fold '+str(fold_idx+1)+' Val':<15} {len(val_idx):>10} {100*len(val_idx)/total:>11.1f}%")

    # 3b. Per-class distribution in test set
    print("\n📋 Test Set Class Distribution:")
    test_labels = [file_df.iloc[i]['label'] for i in test_indices]
    for cls in Config.CLASS_NAMES:
        n = test_labels.count(cls)
        print(f"   {cls}: {n} ({100*n/len(test_labels):.1f}%)")

    # 3c. Figure: Split distribution bar chart
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Plot 1: Overall split sizes
    fold_train_sizes = [len(fold_splits[i][0]) for i in range(len(fold_splits))]
    fold_val_sizes = [len(fold_splits[i][1]) for i in range(len(fold_splits))]
    ax = axes[0]
    bars_data = {'Test Set': test_count,
                 'Avg Train (per fold)': int(np.mean(fold_train_sizes)),
                 'Avg Val (per fold)': int(np.mean(fold_val_sizes))}
    colors = ['#e74c3c', '#2ecc71', '#3498db']
    bars = ax.bar(bars_data.keys(), bars_data.values(), color=colors, edgecolor='white', linewidth=2)
    for bar, val in zip(bars, bars_data.values()):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 10,
                f'{val}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    ax.set_title('Dataset Split Sizes', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of Samples', fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Plot 2: Class distribution per split (test vs train fold 1)
    ax = axes[1]
    train_labels_f1 = [file_df.iloc[i]['label'] for i in fold_splits[0][0]]
    val_labels_f1 = [file_df.iloc[i]['label'] for i in fold_splits[0][1]]
    x = np.arange(len(Config.CLASS_NAMES))
    width = 0.25
    train_counts = [train_labels_f1.count(c) for c in Config.CLASS_NAMES]
    val_counts = [val_labels_f1.count(c) for c in Config.CLASS_NAMES]
    test_counts = [test_labels.count(c) for c in Config.CLASS_NAMES]
    ax.bar(x - width, train_counts, width, label='Train (Fold 1)', color='#2ecc71', alpha=0.8)
    ax.bar(x, val_counts, width, label='Val (Fold 1)', color='#3498db', alpha=0.8)
    ax.bar(x + width, test_counts, width, label='Test', color='#e74c3c', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(Config.CLASS_NAMES, fontsize=11)
    ax.set_title('Class Distribution per Split (Fold 1)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12)
    ax.legend(fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Plot 3: Clinical feature heatmap (mean per class)
    ax = axes[2]
    available_cols = [c for c in Config.CLINICAL_FEATURES if c in file_df.columns]
    if len(available_cols) > 0 and clinical_features is not None:
        class_means = []
        for cls_idx in range(Config.NUM_CLASSES):
            mask = file_df['label_idx'] == cls_idx
            if mask.sum() > 0:
                class_means.append(clinical_features[mask.values].mean(axis=0)[:len(available_cols)])
            else:
                class_means.append(np.zeros(len(available_cols)))
        class_means = np.array(class_means)
        import seaborn as sns
        sns.heatmap(class_means, annot=True, fmt='.2f', cmap='RdYlBu_r',
                    xticklabels=[c[:10] for c in available_cols],
                    yticklabels=Config.CLASS_NAMES, ax=ax, cbar_kws={'shrink': 0.8})
        ax.set_title('Standardized Clinical Features\n(Mean per Class)', fontsize=14, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'No clinical features\navailable', ha='center', va='center',
                fontsize=14, transform=ax.transAxes)
        ax.set_title('Clinical Features', fontsize=14, fontweight='bold')

    plt.suptitle('Section 4: Data Splitting & Feature Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    os.makedirs(Config.FIGURES_DIR, exist_ok=True)
    plt.savefig(os.path.join(Config.FIGURES_DIR, 'fig_section4_splits.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print("\n✅ Figure saved: fig_section4_splits.png")

    # 3d. Augmentation demo
    print("\n" + "-" * 50)
    print("  Step 4: 3D Augmentation Demo")
    print("-" * 50)
    try:
        sample_path = file_df.iloc[0]['preprocessed_path']
        data = np.load(sample_path)
        sample_vol = data['volume'].astype(np.float32)

        transforms = get_train_transforms()
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        mid = sample_vol.shape[2] // 2

        axes[0, 0].imshow(np.rot90(sample_vol[:, :, mid]), cmap='gray')
        axes[0, 0].set_title('Original', fontsize=13, fontweight='bold')
        axes[0, 0].axis('off')
        axes[1, 0].imshow(np.rot90(sample_vol[sample_vol.shape[0]//2, :, :]), cmap='gray')
        axes[1, 0].set_title('Original (Sagittal)', fontsize=13, fontweight='bold')
        axes[1, 0].axis('off')

        for i in range(1, 4):
            aug_vol = transforms(sample_vol.copy())
            axes[0, i].imshow(np.rot90(aug_vol[:, :, mid]), cmap='gray')
            axes[0, i].set_title(f'Augmented #{i} (Axial)', fontsize=13, fontweight='bold')
            axes[0, i].axis('off')
            axes[1, i].imshow(np.rot90(aug_vol[aug_vol.shape[0]//2, :, :]), cmap='gray')
            axes[1, i].set_title(f'Augmented #{i} (Sagittal)', fontsize=13, fontweight='bold')
            axes[1, i].axis('off')

        plt.suptitle('3D Data Augmentation Demo: Original vs Augmented Samples', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(Config.FIGURES_DIR, 'fig_augmentation_demo.png'), dpi=300, bbox_inches='tight')
        plt.show()
        print("✅ Figure saved: fig_augmentation_demo.png")
    except Exception as e:
        print(f"⚠️ Augmentation demo skipped: {e}")

    # Summary
    print("\n" + "=" * 70)
    print("  SECTION 4 SUMMARY")
    print("=" * 70)
    print(f"  ✅ Total samples:      {len(file_df)}")
    print(f"  ✅ Test set:           {test_count} samples")
    print(f"  ✅ Cross-validation:   {len(fold_splits)} folds")
    print(f"  ✅ Clinical features:  {clinical_features.shape[1]} dimensions")
    print(f"  ✅ Augmentations:      5 types (Flip, Rotate, Noise, Intensity, CutOut)")
    print(f"  ✅ Class weights:      Computed via inverse frequency")
    print("\n✅ Section 4 Complete - Dataset & DataLoader ready!")

    return file_df, test_indices, fold_splits, clinical_features


# Execute Section 4
file_df, test_indices, fold_splits, clinical_features = run_section_4(processed_df)
