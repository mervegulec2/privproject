import os
# Ensure we are running the correct FL script
print("="*50)
print("CRITICAL: RUNNING PROTOTYPE FL SYSTEM (20 ROUNDS)")
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
import pickle

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
        
        # Load local model state if it exists (Persistence across rounds)
        self.model_path = f"outputs/models/client_{cid}.pth"
        os.makedirs("outputs/models", exist_ok=True)
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))

    def get_parameters(self, config):
        return [] # No weight sharing

    def fit(self, parameters, config):
        """Local Training with Prototypes."""
        server_round = config.get("server_round", "?")
        print(f"  --> [Round {server_round}] Client {self.cid} starting training...")
        
        # 1. Unpack global prototypes from bytes
        global_protos = {}
        if "protos_bytes" in config:
            global_protos = pickle.loads(config["protos_bytes"])

        # 2. Local Training (Alignment Loss lambda=0.05)
        train_loader = DataLoader(Subset(self.train_ds, self.split_indices), batch_size=self.cfg.batch_size, shuffle=True)
        train_local_proto(self.model, train_loader, global_protos, self.cfg, lambda_p=0.05)

        # 3. Compute local prototypes
        local_protos = compute_prototypes(self.model, train_loader, self.device)
        
        # 4. Local Evaluation
        test_loader = DataLoader(self.test_ds, batch_size=256, shuffle=False)
        acc = evaluate_accuracy(self.model, test_loader, self.device)
        print(f"      Client {self.cid} Accuracy: {acc:.4f}")

        # 5. Save local model state
        torch.save(self.model.state_dict(), self.model_path)

        # 6. Send results (Prototypes serialized as bytes in metrics)
        metrics = {
            "accuracy": acc,
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

    cfg_train = TrainConfig(epochs=1, device=device)

    def client_fn(cid: str) -> FlowerPrototypeClient:
        return FlowerPrototypeClient(int(cid), train_ds, test_ds, split, cfg_train)

    strategy = PrototypeStrategy(
        num_classes=10,
        min_fit_clients=num_clients,
        min_available_clients=num_clients,
        fraction_evaluate=0.0,  # Disable separate evaluation round to avoid crashes (we evaluate in fit)
    )

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.5 if torch.cuda.is_available() else 0}, # Lowered GPU resource to allow some concurrency
    )

if __name__ == "__main__":
    main()
