#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          MASTER PIPELINE ORCHESTRATOR - NeuroGAT (A* / Q1 Edition)           ║
║  Unified Command-Line Interface for Training, Evaluation, Ablation, and App  ║
╚══════════════════════════════════════════════════════════════════════════════╝

Usage Examples:
    python run_pipeline.py --mode verify
    python run_pipeline.py --mode train --epochs 100
    python run_pipeline.py --mode evaluate
    python run_pipeline.py --mode baselines
    python run_pipeline.py --mode ablation
    python run_pipeline.py --mode xai
    python run_pipeline.py --mode app
    python run_pipeline.py --mode all
"""

import os
import sys
import argparse
import time

# Ensure UTF-8 output on Windows terminals
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def print_banner():
    banner = """
===========================================================================
  NeuroGAT: Dual-Head Multi-Scale Graph Attention Network for AD
  A* / Q1 Medical AI Clinical Decision Support Pipeline
===========================================================================
"""
    print(banner)

def get_data_paths():
    from Section_01_Setup_Configuration import Config
    out_dir = getattr(Config, 'OUTPUT_DIR', './outputs')
    f_path = os.path.join(out_dir, 'node_features.npy')
    l_path = os.path.join(out_dir, 'node_labels.npy')
    return f_path, l_path, out_dir

def run_verify():
    print("▶️ Step: Environment & System Diagnostic Verification...")
    import verify_environment
    verify_environment.main()

def run_preprocess():
    print("▶️ Step: Preprocessing ADNI MRI Scans & Clinical Tabular Data...")
    import Section_03_Preprocessing
    Section_03_Preprocessing.main()

def run_feature_extraction():
    print("▶️ Step: Extracting Multimodal Features (DenseNet121 + Radiomics + Clinical)...")
    import Section_04B_Feature_Extraction
    # If main or extractor exists
    if hasattr(Section_04B_Feature_Extraction, 'main'):
        Section_04B_Feature_Extraction.main()
    else:
        print("[INFO] Section 04B loaded. Execute via dataloader.")

def run_train(epochs=None):
    print("▶️ Step: 5-Fold Stratified NeuroGAT Population Graph Training...")
    from Section_01_Setup_Configuration import Config
    if epochs is not None:
        Config.EPOCHS = epochs
    f_path, l_path, _ = get_data_paths()
    if not os.path.exists(f_path) or not os.path.exists(l_path):
        print(f"[ERROR] Features or labels not found at {f_path}. Please extract features first.")
        sys.exit(1)
    from Section_06_Training_Engine import train_gnn_5fold
    train_gnn_5fold(f_path, l_path)

def run_evaluate():
    print("▶️ Step: Master Evaluation Suite (ROC-AUC, DeLong, Calibration, Fairness)...")
    f_path, l_path, _ = get_data_paths()
    from Section_07_Evaluation_Metrics import generate_full_evaluation
    generate_full_evaluation(f_path, l_path)

def run_baselines():
    print("▶️ Step: Fair Traditional ML Benchmarks (SVM, RF, MLP, XGBoost)...")
    f_path, l_path, _ = get_data_paths()
    from Section_10_ML_Baselines import run_ml_baselines
    run_ml_baselines(f_path, l_path)

def run_ablation():
    print("▶️ Step: Comprehensive 11-Point Ablation Study...")
    f_path, l_path, _ = get_data_paths()
    from Section_08_Ablation_Study import run_ablation_studies
    run_ablation_studies(f_path, l_path)

def run_xai():
    print("▶️ Step: Comprehensive Explainable AI (Attention Flow, Grad-CAM, Saliency, SHAP)...")
    f_path, l_path, _ = get_data_paths()
    import Section_09_Explainable_AI
    if hasattr(Section_09_Explainable_AI, 'run_full_xai'):
        Section_09_Explainable_AI.run_full_xai(f_path, l_path)
    else:
        print("[INFO] Section 09 loaded. Running XAI routines...")

def run_app():
    print("▶️ Step: Launching Interactive Gradio Clinical Decision Support Application...")
    import Section_11_Clinical_Inference_App
    if hasattr(Section_11_Clinical_Inference_App, 'launch_app'):
        Section_11_Clinical_Inference_App.launch_app()
    elif hasattr(Section_11_Clinical_Inference_App, 'main'):
        Section_11_Clinical_Inference_App.main()
    else:
        print("[INFO] Section 11 loaded.")

def run_all(epochs=None):
    print("🚀 Running End-to-End NeuroGAT Research Pipeline...")
    t0 = time.time()
    run_verify()
    run_train(epochs=epochs)
    run_evaluate()
    run_baselines()
    run_ablation()
    run_xai()
    total_min = (time.time() - t0) / 60
    print(f"\n🎉 Entire NeuroGAT Research Pipeline finished in {total_min:.1f} minutes!")

def main():
    print_banner()
    parser = argparse.ArgumentParser(description="NeuroGAT Master Research & Clinical Pipeline")
    parser.add_argument(
        '--mode',
        type=str,
        default='verify',
        choices=['verify', 'preprocess', 'extract', 'train', 'evaluate', 'baselines', 'ablation', 'xai', 'app', 'all'],
        help='Pipeline stage to execute'
    )
    parser.add_argument('--epochs', type=int, default=None, help='Override training epochs')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'], help='Computation device')

    args = parser.parse_args()

    if args.device != 'auto':
        import torch
        os.environ['CUDA_VISIBLE_DEVICES'] = '0' if args.device == 'cuda' and torch.cuda.is_available() else '-1'

    mode_map = {
        'verify': run_verify,
        'preprocess': run_preprocess,
        'extract': run_feature_extraction,
        'train': lambda: run_train(args.epochs),
        'evaluate': run_evaluate,
        'baselines': run_baselines,
        'ablation': run_ablation,
        'xai': run_xai,
        'app': run_app,
        'all': lambda: run_all(args.epochs),
    }

    action = mode_map.get(args.mode)
    if action:
        action()
    else:
        print(f"[ERROR] Unknown mode: {args.mode}")
        parser.print_help()

if __name__ == "__main__":
    main()
