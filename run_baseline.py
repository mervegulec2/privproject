import os
import torch
import csv
from torch.utils.data import DataLoader, Subset

from src.data_utils import load_cifar10, Cifar10Config, DirichletSplitConfig, dirichlet_split_indices, save_split, load_split
from src.models import ResNet18Cifar
from src.train_utils_baseline import TrainConfig, train_one_client, evaluate_accuracy, set_seed
from src.eval.test_sets import (
    create_local_aware_indices,
    create_local_proportional_indices
)
import numpy as np
def run_baseline_experiment(seed: int, alpha: float, num_clients: int, epochs: int = 5):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n>>> Baseline simulation: Seed={seed}, Alpha={alpha}, Clients={num_clients}, Epochs={epochs}, Device={device} <<<")
    set_seed(seed)
    
    # 1. Data Setup (Pre-load to memory if possible, but standard is fine)
    train_ds, test_ds = load_cifar10(Cifar10Config(root="data"))
    split_dir = "outputs/splits"
    os.makedirs(split_dir, exist_ok=True)
    split_path = os.path.join(split_dir, f"cifar10_dirichlet_a{alpha}_s{seed}_c{num_clients}.npy")
    
    if os.path.exists(split_path):
        split = load_split(split_path)
    else:
        print(f"Generating new split for seed {seed}...")
        cfg = DirichletSplitConfig(num_clients=num_clients, alpha=alpha, seed=seed)
        split = dirichlet_split_indices(train_ds, cfg)
        save_split(split, split_path)
    
    # 2. Server Weight Initialization
    global_model = ResNet18Cifar(num_classes=10)
    global_weights = global_model.state_dict()
    
    results = []
    
    # 3. Independent Client Training (Sequential to save memory)
    # Optimization: One model reused to save memory allocation time
    client_model = ResNet18Cifar(num_classes=10).to(device)
    
    for cid in range(num_clients):
        print(f"\n[Client {cid+1}/{num_clients} Training]")
        train_idx = split[cid]
        
        # Optimization: use num_workers if on Linux/Mac, but 0 is safer for Windows scripts
        # pin_memory helps with GPU speed
        train_loader = DataLoader(
            Subset(train_ds, train_idx), 
            batch_size=64, 
            shuffle=True, 
            num_workers=0, 
            pin_memory=(device=="cuda")
        )
        test_loader = DataLoader(
            test_ds, 
            batch_size=256, 
            shuffle=False, 
            num_workers=0, 
            pin_memory=(device=="cuda")
        )
        
        # Reset to global server weights
        client_model.load_state_dict(global_weights)
        
        cfg = TrainConfig(epochs=epochs, device=device)
        train_one_client(client_model, train_loader, cfg)
        
        # Get seen classes
        if hasattr(train_ds, "targets"):
            train_targets = np.array(train_ds.targets)
        else:
            train_targets = np.array([train_ds[i][1] for i in range(len(train_ds))])
        
        client_labels = train_targets[train_idx]
        seen_classes = set(np.unique(client_labels))
        
        # 1. Global Standard (Eşit Dağılım / Orijinal CIFAR-10)
        acc_global = evaluate_accuracy(client_model, test_loader, device=device)
        print(f"  Client {cid} Global Acc: {acc_global:.4f}")
        
        # 2. Local Proportional (Eğitimdeki Veri Oranıyla)
        local_prop_idx = create_local_proportional_indices(test_ds, train_ds, train_idx, seed=seed)
        local_prop_loader = DataLoader(Subset(test_ds, local_prop_idx), batch_size=256, shuffle=False)
        acc_local_prop = evaluate_accuracy(client_model, local_prop_loader, device=device)
        print(f"  Client {cid} Local Proportional Acc: {acc_local_prop:.4f}")
        
        # 3. Local-Aware Full (Görülen Tüm Classların Tüm Verisi)
        local_aware_idx = create_local_aware_indices(test_ds, seen_classes)
        local_aware_loader = DataLoader(Subset(test_ds, local_aware_idx), batch_size=256, shuffle=False)
        acc_local_aware = evaluate_accuracy(client_model, local_aware_loader, device=device)
        print(f"  Client {cid} Local-Aware Full Acc: {acc_local_aware:.4f}")
        
        # Ortalama Accuracy (İsteğe Bağlı: Bu 3 testin client bazlı ortalaması)
        avg_acc = (acc_global + acc_local_prop + acc_local_aware) / 3.0
        
        results.append({
            "seed": seed,
            "alpha": alpha,
            "client_id": cid,
            "acc_global": acc_global,
            "acc_local_prop": acc_local_prop,
            "acc_local_aware": acc_local_aware,
            "avg_tests_acc": avg_acc,
            "num_samples": len(train_idx)
        })
    return results

def main():
    # Parameters updated to 10 clients as per latest instruction
    seeds = [42]
    alpha = 0.1
    num_clients = 10
    epochs = 5
    
    all_results = []
    for seed in seeds:
        res = run_baseline_experiment(seed, alpha, num_clients=num_clients, epochs=epochs)
        all_results.extend(res)
        
    # Save Results
    os.makedirs("outputs/metrics", exist_ok=True)
    csv_path = "outputs/metrics/baseline_results.csv"
    if all_results:
        keys = all_results[0].keys()
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_results)
    
    # Summary report
    print("\n" + "="*80)
    print("           EXPERIMENT SUMMARY")
    print("="*80)
    
    total_accs = {"global": 0, "local_prop": 0, "local_aware": 0, "avg_tests": 0}
    
    for s in seeds:
        seed_results = [r for r in all_results if r["seed"] == s]
        if not seed_results: continue
        
        avg_global = sum(r["acc_global"] for r in seed_results) / len(seed_results)
        avg_local_prop = sum(r["acc_local_prop"] for r in seed_results) / len(seed_results)
        avg_local_aware = sum(r["acc_local_aware"] for r in seed_results) / len(seed_results)
        avg_tests = sum(r["avg_tests_acc"] for r in seed_results) / len(seed_results)
        
        total_accs["global"] += avg_global
        total_accs["local_prop"] += avg_local_prop
        total_accs["local_aware"] += avg_local_aware
        total_accs["avg_tests"] += avg_tests
        
        print(f"Seed {s:3d}: Global = {avg_global:.4f} | Local-Prop = {avg_local_prop:.4f} | Local-Aware = {avg_local_aware:.4f} | Overall Client Avg = {avg_tests:.4f}")
        
    print("-" * 80)
    print(f"OVERALL MEAN (across all clients):")
    print(f"  1. Global Standard:      {total_accs['global'] / len(seeds):.4f}")
    print(f"  2. Local Proportional:   {total_accs['local_prop'] / len(seeds):.4f}")
    print(f"  3. Local-Aware Full:     {total_accs['local_aware'] / len(seeds):.4f}")
    print(f"  -> Average of 3 Tests:   {total_accs['avg_tests'] / len(seeds):.4f}")
    print("="*80)

if __name__ == "__main__":
    main()
