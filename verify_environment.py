#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║        ENVIRONMENT & SYSTEM DIAGNOSTIC RUNNER - NeuroGAT (A* Edition)        ║
║  Verifies Python, CUDA/GPU, PyTorch, PyG, MONAI, Radiomics, and Tooling     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import platform

# Ensure UTF-8 output on Windows terminals
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def test_package(pkg_name, import_name=None, critical=True):
    import_name = import_name or pkg_name
    try:
        mod = __import__(import_name)
        ver = getattr(mod, '__version__', 'Installed')
        return True, ver, None
    except Exception as e:
        return False, None, str(e)

def main():
    print("=" * 75)
    print("  [NeuroGAT Clinical Diagnostic System: Environment Health Check]")
    print("=" * 75)
    print(f"• Operating System : {platform.system()} {platform.release()} ({platform.architecture()[0]})")
    print(f"• Python Version   : {platform.python_version()} ({sys.executable})")

    # 1. PyTorch & CUDA Check
    print("\n[1/5] Checking PyTorch & Hardware Acceleration...")
    torch_ok, torch_ver, torch_err = test_package('torch')
    if torch_ok:
        import torch
        cuda_avail = torch.cuda.is_available()
        print(f"  [OK] PyTorch Version : {torch_ver}")
        if cuda_avail:
            device_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"  [OK] CUDA Hardware   : AVAILABLE ({device_name})")
            print(f"  [OK] CUDA Memory     : {vram_gb:.2f} GB VRAM")
            print(f"  [OK] cuDNN Enabled   : {torch.backends.cudnn.enabled} (v{torch.backends.cudnn.version()})")
        else:
            print("  [WARN] CUDA Hardware : NOT DETECTED (Running in CPU Fallback Mode)")
    else:
        print(f"  [FAIL] PyTorch Error : {torch_err}")

    # 2. Graph Neural Networks (PyG)
    print("\n[2/5] Checking Graph Neural Networks (PyTorch Geometric)...")
    pyg_ok, pyg_ver, pyg_err = test_package('torch_geometric')
    if pyg_ok:
        print(f"  [OK] torch_geometric : {pyg_ver}")
        try:
            from torch_geometric.nn import GATConv, GCNConv
            print("  [OK] Core GNN Ops    : GATConv, GCNConv functional")
        except Exception as e:
            print(f"  [WARN] GNN Ops       : {e}")
    else:
        print(f"  [FAIL] torch_geometric : Missing ({pyg_err})")

    # 3. Medical Imaging & Radiomics
    print("\n[3/5] Checking 3D Medical Imaging & Radiomics Backbones...")
    packages_med = [
        ('nibabel', 'nibabel', True),
        ('monai', 'monai', False),
        ('SimpleITK', 'SimpleITK', False),
        ('pyradiomics', 'radiomics', False)
    ]
    for pkg, imp, crit in packages_med:
        ok, ver, err = test_package(pkg, imp, crit)
        status = "[OK]  " if ok else ("[FAIL]" if crit else "[WARN]")
        detail = ver if ok else f"Missing / Fallback Available ({err})"
        print(f"  {status} {pkg:<15}: {detail}")

    # 4. Machine Learning & Statistics
    print("\n[4/5] Checking Core Scientific & Traditional ML Libraries...")
    packages_ml = [
        ('numpy', 'numpy', True),
        ('scipy', 'scipy', True),
        ('pandas', 'pandas', True),
        ('scikit-learn', 'sklearn', True),
        ('xgboost', 'xgboost', True),
        ('matplotlib', 'matplotlib', True),
        ('seaborn', 'seaborn', True),
        ('shap', 'shap', False),
        ('lime', 'lime', False),
        ('gradio', 'gradio', False),
    ]
    for pkg, imp, crit in packages_ml:
        ok, ver, err = test_package(pkg, imp, crit)
        status = "[OK]  " if ok else ("[FAIL]" if crit else "[WARN]")
        detail = ver if ok else f"Optional / Fallback Available ({err})"
        print(f"  {status} {pkg:<15}: {detail}")

    # 5. Local Workspace & Artifact Verification
    print("\n[5/5] Checking Workspace Directory Structure...")
    paths_to_check = [
        ('Data / Codebase Root', '.'),
        ('Output Directory', './outputs'),
        ('Checkpoints', './checkpoints'),
        ('Figures Directory', './outputs/figures'),
        ('Preprocessed Directory', './preprocessed')
    ]
    for name, path in paths_to_check:
        exists = os.path.exists(path)
        status = "[OK] Found" if exists else "[INFO] Will auto-create"
        print(f"  • {name:<25} [{path}]: {status}")

    print("\n" + "=" * 75)
    print("  Environment Verification Complete!")
    print("  To install missing optional dependencies, run:")
    print("     pip install -r requirements.txt")
    print("=" * 75)

if __name__ == "__main__":
    main()
