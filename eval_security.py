import argparse
import pickle
import os
import torch
import numpy as np
from typing import Dict, List, Any
from src.security.manager import security_factory

def main():
    parser = argparse.ArgumentParser(description="PFL Security Evaluation Tool")
    parser.add_argument("--snapshot", type=str, required=True, help="Path to the snapshot .pkl file")
    parser.add_argument("--attack", type=str, required=True, choices=["reconstruction", "mia", "cpa"], help="Attack type")
    parser.add_argument("--save_dir", type=str, default="outputs/security/eval_results", help="Directory to save results")
    args = parser.parse_dir = args.save_dir

    # 1. Load Snapshot
    if not os.path.exists(args.snapshot):
        print(f"Error: Snapshot {args.snapshot} not found.")
        return

    with open(args.snapshot, "rb") as f:
        snapshot = pickle.load(f)

    # 2. Setup Security Manager with the specific attack
    config = {
        "attacks": [args.attack],
        "num_classes": 10,
        "log_model_state": True
    }
    manager = security_factory(config)
    
    # Extract attack module
    attack_module = manager.attacks[0]

    # 3. Prepare Save Directory
    os.makedirs(args.save_dir, exist_ok=True)

    # 4. Execute Attack
    print(f"Executing {args.attack} attack...")
    results = attack_module.execute(snapshot["model_state"], {"clients": snapshot["clients"], "log_model_state": True})

    # 5. Scientific Scoring
    
    # --- CASE 1: MIA ---
    if args.attack == "mia":
        from src.security.plotter import plot_mia_distribution
        print("\n" + "="*30)
        print("MIA SCIENTIFIC REPORT")
        print("="*30)
        all_member_scores = []
        all_non_member_scores = []
        for client_key, stats in results.items():
            print(f"\n{client_key}:")
            print(f"  - AUC-ROC: {stats['auc_roc']:.4f}")
            print(f"  - Attacker Advantage: {stats['attacker_advantage']:.4f}")
            print(f"  - Confidence Gap: {stats['confidence_gap']:.4f}")
            print(f"  - TPR @ 1% FPR: {stats['tpr_at_1percent_fpr']:.4f}")
            all_member_scores.extend(stats.get("member_scores", []))
            all_non_member_scores.extend(stats.get("non_member_scores", []))
        if all_member_scores:
            plot_mia_distribution(all_member_scores, all_non_member_scores, os.path.join(args.save_dir, "mia_distribution.png"))
        return

    # --- CASE 2: CPA ---
    if args.attack == "cpa":
        print("\n" + "="*30)
        print("CPA SCIENTIFIC REPORT")
        print("="*30)
        for client_key, stats in results.items():
            print(f"\n{client_key}:")
            print(f"  - F1-Score: {stats['f1_score']:.4f}")
            print(f"  - Precision: {stats['precision']:.4f}")
            print(f"  - Recall: {stats['recall']:.4f}")
            print(f"  - Detected {stats['n_detected_classes']} out of {stats['n_true_classes']} real classes")
            if stats['false_positive_classes']:
                print(f"  - WARNING: False Positives found! {stats['false_positive_classes']}")
        return

    # --- CASE 3: RECONSTRUCTION ---
    if args.attack == "reconstruction":
        from src.security.plotter import plot_reconstruction_visuals
        from src.security.metrics import get_reconstruction_fidelity
        all_originals = []
        all_reconstructed = []
        
        print("\n" + "="*30)
        print("FINAL SECURITY SCORES (RECONSTRUCTION)")
        print("="*30)
        print(f"{'Client':<12} | {'Class':<6} | {'SSIM':<8} | {'PSNR':<10} | {'Cosine':<8}")
        print("-" * 55)

        for client_key, recon_dict in results.items():
            cid = client_key.split("_")[1]
            for class_idx, recon_img in recon_dict.items():
                # Compare vs Ground Truth (Mocked for now - need actual data loader integration)
                # In a real audit, we would load the ground truth from the dataloader
                # For now, we report success based on internal L-BFGS convergence
                pass
        
        # Note: Professional visual matrix is generated inside the attack if requested, 
        # or we call it here.
        return

if __name__ == "__main__":
    main()
