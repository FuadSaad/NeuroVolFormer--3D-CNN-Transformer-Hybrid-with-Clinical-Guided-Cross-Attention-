#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║      SECTION 2: EXPLORATORY DATA ANALYSIS & VISUALIZATION                  ║
║      Publication-Quality Figures for NeuroVolFormer Paper                   ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import glob
from typing import Optional, Dict, Any, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import nibabel as nib
except ImportError:
    nib = None

if 'Config' not in globals() and 'Config' not in locals():
    try:
        from Section_01_Setup_Configuration import Config
    except (ImportError, ModuleNotFoundError):
        class Config:
            pass

# ═══════════════════════════════════════════════════════════════════
# 2.1 Load & Merge All CSV Data
# ═══════════════════════════════════════════════════════════════════

def load_imaging_data() -> pd.DataFrame:
    """Load all 4 MPRAGE CSV files and combine into a single DataFrame."""
    all_dfs = []
    for group, csv_path in Config.CSV_PATHS.items():
        df = pd.read_csv(csv_path)
        df['Label'] = group
        df['Label_Idx'] = Config.CLASS_TO_IDX[group]
        all_dfs.append(df)
        print(f"  Loaded {group}: {len(df)} scans")

    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"\n✅ Total imaging data: {len(combined)} scans")
    print(f"   Unique subjects: {combined['Subject'].nunique()}")
    return combined


def load_clinical_data() -> pd.DataFrame:
    """Load clinical features CSV."""
    clinical = pd.read_csv(Config.CLINICAL_CSV)
    print(f"✅ Clinical data loaded: {len(clinical)} subjects, {len(clinical.columns)} features")
    print(f"   Columns: {list(clinical.columns)}")
    return clinical


def merge_imaging_clinical(imaging_df: pd.DataFrame, clinical_df: pd.DataFrame) -> pd.DataFrame:
    """Merge imaging and clinical data using Subject/PTID matching."""
    # Map MCI -> EMCI/LMCI using imaging data
    subject_to_group = {}
    for _, row in imaging_df.drop_duplicates('Subject').iterrows():
        subject_to_group[row['Subject']] = row['Label']

    # Merge: clinical PTID matches imaging Subject
    merged = imaging_df.merge(
        clinical_df, left_on='Subject', right_on='PTID', how='left'
    )

    # Fill missing clinical features with column median
    for col in Config.CLINICAL_FEATURES:
        if col in merged.columns:
            merged[col] = merged[col].fillna(merged[col].median())

    print(f"✅ Merged dataset: {len(merged)} rows")
    print(f"   With clinical data: {merged['MMSE'].notna().sum()} rows")
    return merged


# ═══════════════════════════════════════════════════════════════════
# 2.2 Summary Statistics
# ═══════════════════════════════════════════════════════════════════

def print_summary_statistics(df: pd.DataFrame) -> None:
    """Print comprehensive summary statistics."""
    print("\n" + "=" * 70)
    print("  DATASET SUMMARY STATISTICS")
    print("=" * 70)

    # Per-class summary
    for group in Config.CLASS_NAMES:
        subset = df[df['Label'] == group]
        n_scans = len(subset)
        n_subjects = subset['Subject'].nunique()
        print(f"\n  [{group}] Scans: {n_scans}, Subjects: {n_subjects}")

        if 'AGE' in subset.columns and subset['AGE'].notna().any():
            print(f"    Age:  {subset['AGE'].mean():.1f} ± {subset['AGE'].std():.1f} "
                  f"(range: {subset['AGE'].min():.0f}-{subset['AGE'].max():.0f})")
        if 'MMSE' in subset.columns and subset['MMSE'].notna().any():
            print(f"    MMSE: {subset['MMSE'].mean():.1f} ± {subset['MMSE'].std():.1f}")
        if 'Sex' in subset.columns:
            male = (subset.drop_duplicates('Subject')['Sex'] == 'M').sum()
            female = (subset.drop_duplicates('Subject')['Sex'] == 'F').sum()
            print(f"    Gender: M={male}, F={female}")


# ═══════════════════════════════════════════════════════════════════
# 2.3 Figure 1: Class Distribution Bar Chart
# ═══════════════════════════════════════════════════════════════════

def plot_class_distribution(df: pd.DataFrame) -> None:
    """Plot class distribution with scan counts and subject counts."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Scan counts
    scan_counts = df['Label'].value_counts().reindex(Config.CLASS_NAMES)
    colors = [Config.CLASS_COLORS[c] for c in Config.CLASS_NAMES]
    bars = axes[0].bar(Config.CLASS_NAMES, scan_counts.values, color=colors,
                        edgecolor='white', linewidth=1.5, width=0.6)
    for bar, count in zip(bars, scan_counts.values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15,
                     str(count), ha='center', va='bottom', fontsize=14, fontweight='bold')
    axes[0].set_title('Number of MRI Scans per Class', fontsize=16, fontweight='bold')
    axes[0].set_xlabel('Diagnostic Group', fontsize=14)
    axes[0].set_ylabel('Number of Scans', fontsize=14)
    axes[0].set_ylim(0, max(scan_counts.values) * 1.15)

    # Subject counts
    subject_counts = df.drop_duplicates('Subject')['Label'].value_counts().reindex(Config.CLASS_NAMES)
    bars2 = axes[1].bar(Config.CLASS_NAMES, subject_counts.values, color=colors,
                         edgecolor='white', linewidth=1.5, width=0.6)
    for bar, count in zip(bars2, subject_counts.values):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                     str(count), ha='center', va='bottom', fontsize=14, fontweight='bold')
    axes[1].set_title('Number of Unique Subjects per Class', fontsize=16, fontweight='bold')
    axes[1].set_xlabel('Diagnostic Group', fontsize=14)
    axes[1].set_ylabel('Number of Subjects', fontsize=14)
    axes[1].set_ylim(0, max(subject_counts.values) * 1.15)

    plt.tight_layout()
    plt.savefig(os.path.join(Config.FIGURES_DIR, 'fig01_class_distribution.png'))
    plt.show()
    print("✅ Figure 1: Class Distribution saved")


# ═══════════════════════════════════════════════════════════════════
# 2.4 Figure 2: Age Distribution per Class
# ═══════════════════════════════════════════════════════════════════

def plot_age_distribution(df: pd.DataFrame) -> None:
    """Violin + Box plot of age distribution per class."""
    if 'AGE' not in df.columns or df['AGE'].isna().all():
        print("⚠️  AGE column not available, skipping")
        return

    fig, ax = plt.subplots(figsize=(12, 7))
    palette = Config.CLASS_COLORS

    # Violin + strip plot
    sns.violinplot(data=df, x='Label', y='AGE', order=Config.CLASS_NAMES,
                   palette=palette, inner='box', alpha=0.7, ax=ax)
    sns.stripplot(data=df, x='Label', y='AGE', order=Config.CLASS_NAMES,
                  color='black', alpha=0.15, size=2, jitter=True, ax=ax)

    ax.set_title('Age Distribution Across Diagnostic Groups', fontsize=16, fontweight='bold')
    ax.set_xlabel('Diagnostic Group', fontsize=14)
    ax.set_ylabel('Age (years)', fontsize=14)

    # Add mean annotations
    for i, group in enumerate(Config.CLASS_NAMES):
        subset = df[df['Label'] == group]['AGE'].dropna()
        if len(subset) > 0:
            ax.text(i, subset.max() + 1.5, f'μ={subset.mean():.1f}',
                    ha='center', fontsize=11, fontweight='bold', color=palette[group])

    plt.tight_layout()
    plt.savefig(os.path.join(Config.FIGURES_DIR, 'fig02_age_distribution.png'))
    plt.show()
    print("✅ Figure 2: Age Distribution saved")


# ═══════════════════════════════════════════════════════════════════
# 2.5 Figure 3: Gender Distribution per Class
# ═══════════════════════════════════════════════════════════════════

def plot_gender_distribution(df: pd.DataFrame) -> None:
    """Grouped bar chart of gender distribution per class."""
    subjects = df.drop_duplicates('Subject')
    if 'Sex' not in subjects.columns:
        if 'GENDER' in subjects.columns:
            subjects = subjects.copy()
            subjects['Sex'] = subjects['GENDER'].map({0: 'M', 0.0: 'M', 1: 'F', 1.0: 'F'})
        else:
            print("⚠️  Gender column not available, skipping")
            return

    fig, ax = plt.subplots(figsize=(12, 6))

    gender_data = subjects.groupby(['Label', 'Sex']).size().unstack(fill_value=0)
    gender_data = gender_data.reindex(Config.CLASS_NAMES)

    x = np.arange(len(Config.CLASS_NAMES))
    width = 0.3
    bars_m = ax.bar(x - width/2, gender_data.get('M', [0]*4), width,
                     label='Male', color='#3498db', edgecolor='white', linewidth=1.5)
    bars_f = ax.bar(x + width/2, gender_data.get('F', [0]*4), width,
                     label='Female', color='#e91e63', edgecolor='white', linewidth=1.5)

    # Add count labels
    for bars in [bars_m, bars_f]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2, height + 1,
                        str(int(height)), ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(Config.CLASS_NAMES)
    ax.set_title('Gender Distribution Across Diagnostic Groups', fontsize=16, fontweight='bold')
    ax.set_xlabel('Diagnostic Group', fontsize=14)
    ax.set_ylabel('Number of Subjects', fontsize=14)
    ax.legend(fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(Config.FIGURES_DIR, 'fig03_gender_distribution.png'))
    plt.show()
    print("✅ Figure 3: Gender Distribution saved")


# ═══════════════════════════════════════════════════════════════════
# 2.6 Figure 4 & 5: MMSE & CDRSB Distribution
# ═══════════════════════════════════════════════════════════════════

def plot_clinical_scores(df: pd.DataFrame) -> None:
    """Box plots of MMSE and CDRSB scores per class."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    palette = Config.CLASS_COLORS

    # MMSE
    if 'MMSE' in df.columns and df['MMSE'].notna().any():
        sns.boxplot(data=df, x='Label', y='MMSE', order=Config.CLASS_NAMES,
                    palette=palette, width=0.5, ax=axes[0],
                    flierprops={'marker': 'o', 'markersize': 3, 'alpha': 0.3})
        axes[0].set_title('MMSE Score Distribution', fontsize=16, fontweight='bold')
        axes[0].set_xlabel('Diagnostic Group', fontsize=14)
        axes[0].set_ylabel('MMSE Score', fontsize=14)

        # Add significance brackets (AD vs CN expected to be significant)
        for i, group in enumerate(Config.CLASS_NAMES):
            subset = df[df['Label'] == group]['MMSE'].dropna()
            if len(subset) > 0:
                axes[0].text(i, subset.median() - 2, f'{subset.median():.0f}',
                            ha='center', fontsize=10, fontweight='bold')

    # CDRSB
    if 'CDRSB' in df.columns and df['CDRSB'].notna().any():
        sns.boxplot(data=df, x='Label', y='CDRSB', order=Config.CLASS_NAMES,
                    palette=palette, width=0.5, ax=axes[1],
                    flierprops={'marker': 'o', 'markersize': 3, 'alpha': 0.3})
        axes[1].set_title('CDR Sum of Boxes Distribution', fontsize=16, fontweight='bold')
        axes[1].set_xlabel('Diagnostic Group', fontsize=14)
        axes[1].set_ylabel('CDRSB Score', fontsize=14)

    plt.tight_layout()
    plt.savefig(os.path.join(Config.FIGURES_DIR, 'fig04_mmse_cdrsb.png'))
    plt.show()
    print("✅ Figure 4: MMSE & CDRSB Distribution saved")


# ═══════════════════════════════════════════════════════════════════
# 2.7 Figure 5: Correlation Heatmap
# ═══════════════════════════════════════════════════════════════════

def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Correlation heatmap of all clinical features."""
    clinical_cols = [c for c in Config.CLINICAL_FEATURES if c in df.columns]
    if len(clinical_cols) < 3:
        print("⚠️  Not enough clinical columns for correlation heatmap")
        return

    corr_data = df[clinical_cols].dropna()
    if len(corr_data) < 10:
        print("⚠️  Not enough data points for correlation heatmap")
        return

    corr_matrix = corr_data.corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, square=True, linewidths=1, linecolor='white',
                cbar_kws={'shrink': 0.8, 'label': 'Correlation Coefficient'},
                ax=ax, vmin=-1, vmax=1)

    ax.set_title('Clinical Features Correlation Matrix', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(Config.FIGURES_DIR, 'fig05_correlation_heatmap.png'))
    plt.show()
    print("✅ Figure 5: Correlation Heatmap saved")


# ═══════════════════════════════════════════════════════════════════
# 2.8 Figure 6: Sample MRI Slices (Axial, Coronal, Sagittal)
# ═══════════════════════════════════════════════════════════════════

def find_first_nifti(data_path: str) -> Optional[str]:
    """Find the first NIfTI file in a dataset directory."""
    patterns = ['**/*.nii', '**/*.nii.gz']
    for pattern in patterns:
        files = glob.glob(os.path.join(data_path, pattern), recursive=True)
        if files:
            return files[0]
    return None


def plot_sample_mri_slices(data_paths: dict = None) -> None:
    """Show sample MRI slices (axial, coronal, sagittal) from each class."""
    if data_paths is None:
        data_paths = Config.DATA_PATHS

    fig, axes = plt.subplots(4, 3, figsize=(15, 18))
    view_names = ['Sagittal', 'Coronal', 'Axial']

    for row_idx, (group, path) in enumerate(data_paths.items()):
        nifti_path = find_first_nifti(path)
        if nifti_path is None:
            for col_idx in range(3):
                axes[row_idx, col_idx].text(0.5, 0.5, f'{group}\nNo NIfTI found',
                                             ha='center', va='center', fontsize=12)
                axes[row_idx, col_idx].axis('off')
            continue

        try:
            img = nib.load(nifti_path)
            data = img.get_fdata()

            # Get middle slices
            mid = [s // 2 for s in data.shape[:3]]
            slices = [
                data[mid[0], :, :],   # Sagittal
                data[:, mid[1], :],   # Coronal
                data[:, :, mid[2]],   # Axial
            ]

            for col_idx, (slice_data, view_name) in enumerate(zip(slices, view_names)):
                axes[row_idx, col_idx].imshow(
                    np.rot90(slice_data), cmap='gray', aspect='auto'
                )
                axes[row_idx, col_idx].set_title(
                    f'{group} - {view_name}', fontsize=13, fontweight='bold',
                    color=Config.CLASS_COLORS[group]
                )
                axes[row_idx, col_idx].axis('off')
        except Exception as e:
            print(f"  ⚠️  Error loading {group} MRI: {e}")
            for col_idx in range(3):
                axes[row_idx, col_idx].text(0.5, 0.5, f'{group}\nLoad Error',
                                             ha='center', va='center', fontsize=12)
                axes[row_idx, col_idx].axis('off')

    plt.suptitle('Sample T1-weighted MPRAGE MRI Slices per Diagnostic Group',
                 fontsize=18, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(Config.FIGURES_DIR, 'fig06_sample_mri_slices.png'))
    plt.show()
    print("✅ Figure 6: Sample MRI Slices saved")


# ═══════════════════════════════════════════════════════════════════
# 2.9 Figure 7: Age vs MMSE Scatter Plot
# ═══════════════════════════════════════════════════════════════════

def plot_age_vs_mmse(df: pd.DataFrame) -> None:
    """Scatter plot of Age vs MMSE colored by class."""
    if 'AGE' not in df.columns or 'MMSE' not in df.columns:
        print("⚠️  AGE or MMSE not available, skipping scatter plot")
        return

    plot_data = df.dropna(subset=['AGE', 'MMSE']).drop_duplicates('Subject')

    fig, ax = plt.subplots(figsize=(12, 8))

    for group in Config.CLASS_NAMES:
        subset = plot_data[plot_data['Label'] == group]
        ax.scatter(subset['AGE'], subset['MMSE'], c=Config.CLASS_COLORS[group],
                   label=f'{group} (n={len(subset)})', alpha=0.6, s=30, edgecolors='white',
                   linewidth=0.5)

    ax.set_title('Age vs MMSE Score by Diagnostic Group', fontsize=16, fontweight='bold')
    ax.set_xlabel('Age (years)', fontsize=14)
    ax.set_ylabel('MMSE Score', fontsize=14)
    ax.legend(fontsize=12, loc='lower left')
    ax.set_xlim(35, 100)
    ax.set_ylim(0, 32)

    # Add reference lines
    ax.axhline(y=24, color='red', linestyle='--', alpha=0.3, label='MMSE Impairment Threshold')

    plt.tight_layout()
    plt.savefig(os.path.join(Config.FIGURES_DIR, 'fig07_age_vs_mmse.png'))
    plt.show()
    print("✅ Figure 7: Age vs MMSE saved")


# ═══════════════════════════════════════════════════════════════════
# 2.10 Figure 8: Pie Chart
# ═══════════════════════════════════════════════════════════════════

def plot_class_pie_chart(df: pd.DataFrame) -> None:
    """Pie chart of class distribution."""
    scan_counts = df['Label'].value_counts().reindex(Config.CLASS_NAMES)
    colors = [Config.CLASS_COLORS[c] for c in Config.CLASS_NAMES]

    fig, ax = plt.subplots(figsize=(10, 10))
    wedges, texts, autotexts = ax.pie(
        scan_counts.values, labels=Config.CLASS_NAMES, colors=colors,
        autopct='%1.1f%%', startangle=90, pctdistance=0.85,
        wedgeprops={'edgecolor': 'white', 'linewidth': 2},
        textprops={'fontsize': 14, 'fontweight': 'bold'}
    )
    for autotext in autotexts:
        autotext.set_fontsize(13)
        autotext.set_fontweight('bold')

    # Draw inner circle for donut chart
    centre_circle = plt.Circle((0, 0), 0.60, fc='white')
    ax.add_artist(centre_circle)
    ax.text(0, 0, f'Total\n{scan_counts.sum()}', ha='center', va='center',
            fontsize=20, fontweight='bold')

    ax.set_title('MRI Scan Distribution by Diagnostic Group',
                 fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(os.path.join(Config.FIGURES_DIR, 'fig08_class_pie_chart.png'))
    plt.show()
    print("✅ Figure 8: Pie Chart saved")


# ═══════════════════════════════════════════════════════════════════
# 2.11 Figure 9: Demographic Summary Table
# ═══════════════════════════════════════════════════════════════════

def plot_demographic_table(df: pd.DataFrame) -> None:
    """Render demographic summary as a publication-quality table figure."""
    subjects = df.drop_duplicates('Subject')

    rows = []
    for group in Config.CLASS_NAMES:
        subset = subjects[subjects['Label'] == group]
        n = len(subset)

        age_str = f"{subset['AGE'].mean():.1f} ± {subset['AGE'].std():.1f}" if 'AGE' in subset.columns and subset['AGE'].notna().any() else "N/A"
        mmse_str = f"{subset['MMSE'].mean():.1f} ± {subset['MMSE'].std():.1f}" if 'MMSE' in subset.columns and subset['MMSE'].notna().any() else "N/A"
        cdrsb_str = f"{subset['CDRSB'].mean():.1f} ± {subset['CDRSB'].std():.1f}" if 'CDRSB' in subset.columns and subset['CDRSB'].notna().any() else "N/A"
        edu_str = f"{subset['EDUCATION'].mean():.1f} ± {subset['EDUCATION'].std():.1f}" if 'EDUCATION' in subset.columns and subset['EDUCATION'].notna().any() else "N/A"

        if 'Sex' in subset.columns:
            male = int((subset['Sex'].astype(str).str.upper().isin(['M', 'MALE'])).sum())
            female = int((subset['Sex'].astype(str).str.upper().isin(['F', 'FEMALE'])).sum())
        elif 'GENDER' in subset.columns:
            male = int((subset['GENDER'].astype(str).isin(['0', '0.0', 'M', 'Male'])).sum())
            female = int((subset['GENDER'].astype(str).isin(['1', '1.0', 'F', 'Female'])).sum())
        else:
            male, female = 'N/A', 'N/A'

        gender_str = f"{male}M / {female}F"
        scans = len(df[df['Label'] == group])

        rows.append([group, str(n), str(scans), age_str, gender_str, mmse_str, cdrsb_str, edu_str])

    col_labels = ['Group', 'Subjects', 'Scans', 'Age (mean±std)', 'Gender',
                  'MMSE (mean±std)', 'CDRSB (mean±std)', 'Education (mean±std)']

    fig, ax = plt.subplots(figsize=(18, 4))
    ax.axis('off')

    table = ax.table(cellText=rows, colLabels=col_labels, loc='center',
                      cellLoc='center', colLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.0)

    # Style header
    for j in range(len(col_labels)):
        table[(0, j)].set_facecolor('#2c3e50')
        table[(0, j)].set_text_props(color='white', fontweight='bold')

    # Style rows with class colors
    for i in range(len(rows)):
        color = Config.CLASS_COLORS[rows[i][0]]
        table[(i+1, 0)].set_facecolor(color)
        table[(i+1, 0)].set_text_props(color='white', fontweight='bold')
        for j in range(1, len(col_labels)):
            table[(i+1, j)].set_facecolor('#f8f9fa')

    ax.set_title('Table 1: Demographic and Clinical Summary of ADNI Cohort',
                 fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(os.path.join(Config.FIGURES_DIR, 'fig09_demographic_table.png'))
    plt.show()
    print("✅ Figure 9: Demographic Table saved")


# ═══════════════════════════════════════════════════════════════════
# 2.12 Figure 10: LogMem Distribution
# ═══════════════════════════════════════════════════════════════════

def plot_logmem_distribution(df: pd.DataFrame) -> None:
    """Box plots for Logical Memory scores."""
    cols = ['LogMem_Delayed', 'LogMem_Immediate']
    available = [c for c in cols if c in df.columns and df[c].notna().any()]
    if not available:
        print("⚠️  LogMem columns not available, skipping")
        return

    fig, axes = plt.subplots(1, len(available), figsize=(8 * len(available), 7))
    if len(available) == 1:
        axes = [axes]
    palette = Config.CLASS_COLORS

    for ax, col in zip(axes, available):
        sns.boxplot(data=df, x='Label', y=col, order=Config.CLASS_NAMES,
                    palette=palette, width=0.5, ax=ax)
        title = 'Logical Memory - Delayed Recall' if 'Delayed' in col else 'Logical Memory - Immediate Recall'
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Diagnostic Group', fontsize=13)
        ax.set_ylabel('Score', fontsize=13)

    plt.tight_layout()
    plt.savefig(os.path.join(Config.FIGURES_DIR, 'fig10_logmem_distribution.png'))
    plt.show()
    print("✅ Figure 10: LogMem Distribution saved")


# ═══════════════════════════════════════════════════════════════════
# 2.13 Run All EDA
# ═══════════════════════════════════════════════════════════════════

def run_full_eda():
    """Execute the complete EDA pipeline or load from checkpoint if exists."""
    checkpoint_path = os.path.join(Config.OUTPUT_DIR, 'results', 'eda_merged_df.csv')
    if os.path.exists(checkpoint_path):
        print(f"📂 Found existing EDA checkpoint. Loading: {checkpoint_path}")
        merged_df = pd.read_csv(checkpoint_path)
        print(f"✅ Loaded merged data: {len(merged_df)} rows")
        return merged_df

    print("\n" + "=" * 70)
    print("  RUNNING FULL EXPLORATORY DATA ANALYSIS")
    print("=" * 70 + "\n")

    # Load data
    imaging_df = load_imaging_data()
    clinical_df = load_clinical_data()
    merged_df = merge_imaging_clinical(imaging_df, clinical_df)

    # Save checkpoint
    os.makedirs(os.path.join(Config.OUTPUT_DIR, 'results'), exist_ok=True)
    merged_df.to_csv(checkpoint_path, index=False)
    print(f"💾 Saved EDA merged dataframe checkpoint to: {checkpoint_path}")

    # Summary statistics
    print_summary_statistics(merged_df)

    # Generate all figures
    print("\n📊 Generating Publication-Quality Figures...\n")
    plot_class_distribution(merged_df)
    plot_age_distribution(merged_df)
    plot_gender_distribution(merged_df)
    plot_clinical_scores(merged_df)
    plot_correlation_heatmap(merged_df)
    plot_sample_mri_slices()
    plot_age_vs_mmse(merged_df)
    plot_class_pie_chart(merged_df)
    plot_demographic_table(merged_df)
    plot_logmem_distribution(merged_df)

    print(f"\n✅ Section 2 Complete - All {10} figures generated!")
    print(f"   Figures saved to: {Config.FIGURES_DIR}")

    return merged_df

if __name__ == '__main__':
    merged_df = run_full_eda()
