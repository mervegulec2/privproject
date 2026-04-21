import os
import pickle
import argparse
import torch
import numpy as np
from torch.utils.data import Subset
from src.security.attacks.reconstruction import PrototypeReconstructionAttack
from src.security.metrics import get_reconstruction_fidelity, get_inference_leakage
from src.security.plotter import plot_reconstruction_visuals
from src.data_utils import load_cifar10, Cifar10Config, load_split

def main():
    parser = argparse.ArgumentParser(description="Professional Security Evaluation")
    parser.add_argument("--snapshot", type=str, required=True, help="Path to pkl snapshot")
    parser.add_argument("--attack", type=str, choices=["reconstruction", "mia"], default="reconstruction")
    parser.add_argument("--split_path", type=str, help="Optional: Path to indices split for ground truth comparison")
    parser.add_argument("--save_dir", type=str, default="outputs/security/eval_results")
    args = parser.parse_args()

    if not os.path.exists(args.snapshot):
        print(f"Error: Snapshot {args.snapshot} not found.")
        return

    # 1. Load Snapshot
    with open(args.snapshot, "rb") as f:
        snapshot = pickle.load(f)
    print(f"\n>>> Evaluating Security for Round {snapshot['round']} <<<")

    # 2. Load Ground Truth Data (Researcher's perspective)
    print("Loading Ground Truth data for comparison...")
    train_ds, test_ds = load_cifar10(Cifar10Config(root="data"))
    
    # 3. Setup Attack
    attack_module = None
    if args.attack == "reconstruction":
        attack_module = PrototypeReconstructionAttack(save_dir=os.path.join(args.save_dir, "visuals"), iterations=500)
    
    # 4. Execute Attack
    print(f"Executing {args.attack} attack...")
    results = attack_module.execute(snapshot["model_state"], {"clients": snapshot["clients"], "log_model_state": True})

    # 5. Scientific Scoring
    summary_metrics = {}
    
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
        all_originals = []
        all_reconstructed = []
        
        for client_key, stats in results.items():
            cid = int(client_key.replace("client_", ""))
            # For each class, we compare the reconstruction with the average 'Real' image for that client class
            # (Note: In One-Shot, the prototype is the best representation of these samples)
            
            # Since the server doesn't know the indices, normally this is hard.
            # But the EVALUTOR knows. We can try to load the original prototypes from the training set.
            # To be simple: we use the auxiliary test set's class-averages as a reference for fidelity.
            
            client_scores = {}
            for target_class, details in stats.items():
                # Load a dummy image for shape or use the saved path
                from PIL import Image
                import torchvision.transforms as T
                recon_img_pil = Image.open(details["save_path"]).convert('RGB')
                recon_tensor = T.ToTensor()(recon_img_pil).numpy()
                
                # Fetch a sample of the actual Ground Truth class from Test Set for comparison
                class_indices = [i for i, label in enumerate(test_ds.targets) if label == int(target_class)]
                original_sample = test_ds[class_indices[0]][0].numpy()
                
                fidelity = get_reconstruction_fidelity(original_sample, recon_tensor)
                client_scores[target_class] = fidelity
                
                all_originals.append(original_sample)
                all_reconstructed.append(recon_tensor)
            
            summary_metrics[client_key] = client_scores

        # Generate the Visual Comparison Plot (DLG Style)
        if all_originals:
            plot_reconstruction_visuals(all_originals, all_reconstructed, os.path.join(args.save_dir, "visual_leakage_matrix.png"))
        else:
            print("\n[Evaluation] No images were reconstructed. This usually happens if the model state was missing or the attack failed to converge.")

    # 6. Final Report
    print("\n" + "="*30)
    print("FINAL SECURITY SCORES")
    print("="*30)
    for cid, classes in summary_metrics.items():
        print(f"\n{cid}:")
        for cls, scores in classes.items():
            print(f"  Class {cls} -> SSIM: {scores['ssim']:.4f} | PSNR: {scores['psnr']:.2f}dB | Cosine: {scores['cosine_sim']:.4f}")
    
    print(f"\nScientific summary saved to {args.save_dir}")

if __name__ == "__main__":
    main()
