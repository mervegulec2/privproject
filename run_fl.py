import os
# Ensure we are running the correct FL script
print("="*50)
print("CRITICAL: RUNNING PROTOTYPE FL SYSTEM (2 ROUNDS)")
print("="*50)

import torch
import flwr as fl
import numpy as np
from torch.utils.data import DataLoader, Subset
from typing import Dict, List, Tuple

from src.data_utils import load_cifar10, Cifar10Config, DirichletSplitConfig, dirichlet_split_indices, load_split, save_split
from src.models import ResNet18Cifar
from src.train_utils import TrainConfig, train_local_proto, compute_prototypes, evaluate_accuracy, set_seed
from src.aggregation import PrototypeStrategy
from src.eval.test_sets import (
    create_local_aware_indices,
    create_local_proportional_indices
)
import pickle
from src.utils.logging_utils import save_summary_json, setup_metrics_dir, clear_metrics

# 1. Define the Flower Client
class FlowerPrototypeClient(fl.client.NumPyClient):
    def __init__(self, cid: int, train_ds, test_ds, split, cfg: TrainConfig):
        self.cid = cid
        self.train_ds = train_ds
        self.test_ds = test_ds
        self.split_indices = split[cid]
        self.cfg = cfg
        
        self.device = cfg.device
        self.model = ResNet18Cifar(num_classes=10).to(self.device)

    def get_parameters(self, config):
        # 1. Send our trained model weights back to the server
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        """Local Training with Prototypes + Global Model Weights."""
        server_round = config.get("server_round", "?")
        print(f"  --> [Round {server_round}] Client {self.cid} starting training...")
        
        # 0. Load the Global Model weights that the server sent us
        if parameters is not None and len(parameters) > 0:
            params_dict = zip(self.model.state_dict().keys(), parameters)
            state_dict = {k: torch.tensor(v) for k, v in params_dict}
            self.model.load_state_dict(state_dict, strict=True)
        
        # 1. Unpack global prototypes from bytes
        global_protos = {}
        if "protos_bytes" in config:
            global_protos = pickle.loads(config["protos_bytes"])

        # 2. Local Training (Alignment Loss lambda=0.05)
        train_loader = DataLoader(Subset(self.train_ds, self.split_indices), batch_size=self.cfg.batch_size, shuffle=True)
        train_local_proto(self.model, train_loader, global_protos, self.cfg, lambda_p=0.05)

        # 3. Compute local prototypes
        local_protos = compute_prototypes(self.model, train_loader, self.device)
        
        # 4. Local Evaluation (3 Test Sets)
        
        # Get seen classes
        if hasattr(self.train_ds, "targets"):
            train_targets = np.array(self.train_ds.targets)
        else:
            train_targets = np.array([self.train_ds[i][1] for i in range(len(self.train_ds))])
        
        client_labels = train_targets[self.split_indices]
        seen_classes = set(np.unique(client_labels))
        
        # Test 1: Global Standard
        test_loader = DataLoader(self.test_ds, batch_size=256, shuffle=False)
        acc_global = evaluate_accuracy(self.model, test_loader, self.device)
        
        # Test 2: Local Proportional
        local_prop_idx = create_local_proportional_indices(self.test_ds, self.train_ds, self.split_indices, seed=42)
        local_prop_loader = DataLoader(Subset(self.test_ds, local_prop_idx), batch_size=256, shuffle=False)
        acc_local_prop = evaluate_accuracy(self.model, local_prop_loader, self.device)
        
        # Test 3: Local-Aware Full
        local_aware_idx = create_local_aware_indices(self.test_ds, seen_classes)
        local_aware_loader = DataLoader(Subset(self.test_ds, local_aware_idx), batch_size=256, shuffle=False)
        acc_local_aware = evaluate_accuracy(self.model, local_aware_loader, self.device)
        
        avg_tests_acc = (acc_global + acc_local_prop + acc_local_aware) / 3.0
        
        print(f"      Client {self.cid} Eval -> Global: {acc_global:.4f} | Local-Prop: {acc_local_prop:.4f} | Local-Aware: {acc_local_aware:.4f} | Avg: {avg_tests_acc:.4f}")

        # 6. Send results (Prototypes serialized as bytes in metrics)
        metrics = {
            "accuracy": float(acc_global), # Standard tracking
            "acc_global": float(acc_global),
            "acc_local_prop": float(acc_local_prop),
            "acc_local_aware": float(acc_local_aware),
            "avg_tests_acc": float(avg_tests_acc),
            "cid": int(self.cid),
            "protos_bytes": pickle.dumps(local_protos)
        }

        return self.get_parameters(config={}), len(self.split_indices), metrics

    def evaluate(self, parameters, config):
        """Local Evaluation."""
        # Note: We already evaluate in fit(), but Flower requires this for the evaluate round.
        # To avoid ZeroDivisionError, we return the test set size.
        return 0.0, len(self.test_ds), {}

# 2. Main Simulation
def main():
    num_clients = 10
    num_rounds = 2
    alpha = 0.1
    seed = 42
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    set_seed(seed)
    train_ds, test_ds = load_cifar10(Cifar10Config(root="data"))
    
    # Load Split
    split_path = f"outputs/splits/cifar10_dirichlet_a{alpha}_s{seed}_c{num_clients}.npy"
    if not os.path.exists(split_path):
        cfg_s = DirichletSplitConfig(num_clients=num_clients, alpha=alpha, seed=seed)
        split = dirichlet_split_indices(train_ds, cfg_s)
        save_split(split, split_path)
    else:
        split = load_split(split_path)

    cfg_train = TrainConfig(epochs=5, device=device)

    def client_fn(cid: str) -> FlowerPrototypeClient:
        return FlowerPrototypeClient(int(cid), train_ds, test_ds, split, cfg_train)

    strategy = PrototypeStrategy(
        num_classes=10,
        min_fit_clients=num_clients,
        min_available_clients=num_clients,
        fraction_evaluate=0.0,  # Disable separate evaluation round to avoid crashes (we evaluate in fit)
    )

    # 0. Prep (Clear old results)
    clear_metrics()
    setup_metrics_dir()

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.5 if torch.cuda.is_available() else 0}, 
    )

    # 3. Save Final Summary
    save_summary_json(history.__dict__)
    
    print("\n" + "="*50)
    print("SIMULATION COMPLETE")
    print(f"Results saved to: outputs/metrics/simulation_results.csv")
    print(f"Summary saved to: outputs/metrics/utility_summary.json")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
