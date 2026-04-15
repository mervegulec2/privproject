import os
import torch
import csv
from torch.utils.data import DataLoader, Subset

from src.data_utils import load_cifar10, Cifar10Config, DirichletSplitConfig, dirichlet_split_indices, save_split, load_split
from src.models import ResNet18Cifar
from src.train_utils import TrainConfig, train_one_client, evaluate_accuracy, set_seed

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
        
        acc = evaluate_accuracy(client_model, test_loader, device=device)
        print(f"  Client {cid} Test Accuracy: {acc:.4f}")
        
        results.append({
            "seed": seed,
            "alpha": alpha,
            "client_id": cid,
            "accuracy": acc,
            "num_samples": len(train_idx)
        })
    return results

def main():
    # Parameters updated to 10 clients as per latest instruction
    seeds = [42, 123, 999]
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
    print("\n" + "="*40)
    print("           EXPERIMENT SUMMARY")
    print("="*40)
    seed_accs = {}
    for r in all_results:
        s = r["seed"]
        if s not in seed_accs: seed_accs[s] = []
        seed_accs[s].append(r["accuracy"])
        
    total_avg = 0
    for s, accs in seed_accs.items():
        avg = sum(accs) / len(accs)
        total_avg += avg
        print(f"Seed {s:3d}: Average Accuracy = {avg:.4f}")
    print("-" * 40)
    print(f"OVERALL MEAN ACCURACY (10 CLIENTS): {total_avg / len(seed_accs):.4f}")
    print("="*40)

if __name__ == "__main__":
    main()
