import argparse
import pickle
import os
import torch
import numpy as np
from typing import Dict, List, Any
from src.security.manager import security_factory
from src.data_utils import load_cifar10, Cifar10Config, load_split
from PIL import Image

def main():
    parser = argparse.ArgumentParser(description="PFL Security Evaluation Tool")
    parser.add_argument("--snapshot", type=str, required=True, help="Path to the snapshot .pkl file")
    parser.add_argument("--attack", type=str, required=True, choices=["reconstruction", "mia", "cpa"], help="Attack type")
    parser.add_argument("--save_dir", type=str, default="outputs/security/eval_results", help="Directory to save results")
    parser.add_argument("--split_path", type=str, default="outputs/splits/cifar10_dirichlet_a0.1_s42_c10.npy", help="Path to the client dataset splits")
    parser.add_argument("--limit_clients", type=int, default=0, help="Limit number of clients to attack (0 for all)")
    parser.add_argument("--limit_classes", type=int, default=0, help="Limit number of classes per client to attack (0 for all)")
    args = parser.parse_args()

    # 1. Load Snapshot
    if not os.path.exists(args.snapshot):
        print(f"Error: Snapshot {args.snapshot} not found.")
        return

    try:
        with open(args.snapshot, "rb") as f:
            snapshot = pickle.load(f)
    except Exception as e:
        print(f"Error loading snapshot: {e}")
        return

    # 2. Extract and format client data from snapshot
    clients_list = []
    if "prototypes" in snapshot:
        # Standard PFL snapshot format: {cid: {class_id: proto}}
        for cid, protos in snapshot["prototypes"].items():
            counts = snapshot.get("counts", {}).get(cid, {})
            clients_list.append({
                "cid": cid,
                "protos": protos,
                "counts": counts
            })
    elif "clients" in snapshot:
        # Alternative format if already list-based
        clients_list = snapshot["clients"]
    
    if not clients_list:
        print("Error: No client data found in snapshot.")
        return

    # Filter if limits are set
    if args.limit_clients > 0:
        clients_list = clients_list[:args.limit_clients]
    
    if args.limit_classes > 0:
        for client in clients_list:
            keys = list(client["protos"].keys())[:args.limit_classes]
            client["protos"] = {k: client["protos"][k] for k in keys}
            client["counts"] = {k: client["counts"][k] for k in keys}

    # 2. Setup Security Manager
    config = {"attacks": [args.attack], "num_classes": 10, "log_model_state": True}
    manager = security_factory(config)
    attack_module = manager.attacks[0]

    os.makedirs(args.save_dir, exist_ok=True)

    # 4. Execute Attack
    print(f"Executing {args.attack} attack...")
    results = attack_module.execute(snapshot.get("model_state"), {
        "clients": clients_list, 
        "log_model_state": True,
        "split_path": args.split_path
    })

    # 5. Scientific Scoring
    
    if args.attack == "mia":
        from src.security.plotter import plot_mia_distribution
        print("\n" + "="*30 + "\nMIA SCIENTIFIC REPORT\n" + "="*30)
        all_m, all_nm = [], []
        for ck, stats in results.items():
            print(f"\n{ck}:")
            for k, v in stats.items():
                if k not in ["member_scores", "non_member_scores"]: print(f"  - {k}: {v:.4f}")
            all_m.extend(stats.get("member_scores", []))
            all_nm.extend(stats.get("non_member_scores", []))
        if all_m: plot_mia_distribution(all_m, all_nm, os.path.join(args.save_dir, "mia_distribution.png"))
        return

    if args.attack == "cpa":
        print("\n" + "="*30 + "\nCPA SCIENTIFIC REPORT\n" + "="*30)
        for ck, stats in results.items():
            print(f"\n{ck}:\n  - F1-Score: {stats.get('f1_score', 0):.4f}\n  - Precision: {stats.get('precision',0):.4f}\n  - Recall: {stats.get('recall',0):.4f}")
        return

    if args.attack == "reconstruction":
        from src.security.metrics import get_reconstruction_fidelity
        from src.security.plotter import plot_reconstruction_visuals

        # Robust Split Loading
        split_path = args.split_path
        if not os.path.exists(split_path):
            print(f"Error: Split file {split_path} not found. Please run PFL first.")
            return
        
        train_ds, _ = load_cifar10(Cifar10Config(root="data"))
        splits = load_split(split_path)
        
        print("\n" + "="*30 + "\nFINAL SECURITY SCORES (RECONSTRUCTION)\n" + "="*30)
        comparison_data = []

        for client_key, recon_dict in results.items():
            try:
                cid = int(client_key.split("_")[1])
            except (IndexError, ValueError): continue
            
            print(f"\n{client_key}:")
            if cid not in splits:
                print(f"  [Warning] Client {cid} missing from splits.")
                continue
            
            member_indices = splits[cid]
            for class_idx, info in recon_dict.items():
                target_class = int(class_idx)
                recon_path = info["save_path"]
                
                if not os.path.exists(recon_path): continue
                
                # 1. Load Reconstructed Image
                recon_img = np.array(Image.open(recon_path)).astype(np.float32) / 255.0

                # 2. Find Real Images Centroid
                class_indices = [i for i in member_indices if int(train_ds.targets[i]) == target_class]
                if not class_indices: continue
                
                real_imgs = []
                for i in class_indices:
                    img_np = np.array(train_ds[i][0]).transpose(1, 2, 0)
                    # Min-Max Normalize to [0, 1] for visual fidelity check
                    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
                    real_imgs.append(img_np)
                
                mean_real_img = np.mean(real_imgs, axis=0)
                
                # 3. Fidelity
                metrics = get_reconstruction_fidelity(mean_real_img, recon_img)
                print(f"  Class {target_class} -> SSIM: {metrics['ssim']:.4f} | PSNR: {metrics['psnr']:.2f}dB | Cosine: {metrics['cosine_sim']:.4f}")
                
                comparison_data.append({
                    "original": mean_real_img,
                    "reconstructed": recon_img,
                    "title": f"C{cid} Class {target_class}"
                })

        if comparison_data:
            plot_reconstruction_visuals(comparison_data, os.path.join(args.save_dir, "visual_leakage_matrix.png"))
        return

if __name__ == "__main__":
    main()
