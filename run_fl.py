import os
import torch
import flwr as fl
import numpy as np
from torch.utils.data import DataLoader, Subset
from typing import Dict, List, Tuple

from src.data_utils import load_cifar10, Cifar10Config, DirichletSplitConfig, dirichlet_split_indices, load_split
from src.models import ResNet18Cifar
from src.train_utils import TrainConfig, train_local_proto, compute_prototypes, evaluate_accuracy, set_seed
from src.aggregation import PrototypeStrategy

# 1. Define the Flower Client
class FlowerPrototypeClient(fl.client.NumPyClient):
    def __init__(self, cid: int, train_ds, test_ds, split, cfg: TrainConfig):
        self.cid = cid
        self.train_ds = train_ds
        self.test_ds = test_ds
        self.split_indices = split[cid]
        self.cfg = cfg
        
        # Persistent local model
        self.model = ResNet18Cifar(num_classes=10).to(cfg.device)
        self.device = cfg.device

    def get_parameters(self, config):
        # We don't share parameters, so we can return empty or dummy
        return []

    def fit(self, parameters, config):
        """Local Training with Prototypes."""
        # 1. Unpack global prototypes from config
        global_protos = {}
        for key, val in config.items():
            if key.startswith("proto_"):
                class_id = int(key.split("_")[1])
                global_protos[class_id] = np.array(val, dtype=np.float32)

        # 2. Train locally with Alignment Loss
        train_loader = DataLoader(
            Subset(self.train_ds, self.split_indices), 
            batch_size=self.cfg.batch_size, 
            shuffle=True
        )
        train_local_proto(self.model, train_loader, global_protos, self.cfg, lambda_p=0.05)

        # 3. Compute local prototypes to send to server
        local_protos = compute_prototypes(self.model, train_loader, self.device)
        
        # 4. Evaluate locally
        test_loader = DataLoader(self.test_ds, batch_size=256, shuffle=False)
        acc = evaluate_accuracy(self.model, test_loader, self.device)

        # 5. Pack prototypes and accuracy into metrics
        metrics = {"accuracy": acc}
        for c, vec in local_protos.items():
            metrics[f"proto_{c}"] = vec.tolist() # Send as list for JSON-like metric transport

        # Return dummy parameters, sample size, and metrics
        return self.get_parameters(config={}), len(self.split_indices), metrics

    def evaluate(self, parameters, config):
        """Flower's global evaluation (optional here as we do it in fit)."""
        test_loader = DataLoader(self.test_ds, batch_size=256, shuffle=False)
        acc = evaluate_accuracy(self.model, test_loader, self.device)
        return float(acc), len(self.test_ds), {"accuracy": float(acc)}

# 2. Simulation Orchestrator
def main():
    # Parameters
    num_clients = 10
    num_rounds = 20
    alpha = 0.1
    seed = 42
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    set_seed(seed)
    train_ds, test_ds = load_cifar10(Cifar10Config(root="data"))
    
    # Load Split
    split_path = f"outputs/splits/cifar10_dirichlet_a{alpha}_s{seed}_c{num_clients}.npy"
    if not os.path.exists(split_path):
        # Fallback if not generated
        cfg_s = DirichletSplitConfig(num_clients=num_clients, alpha=alpha, seed=seed)
        split = dirichlet_split_indices(train_ds, cfg_s)
    else:
        split = load_split(split_path)

    cfg_train = TrainConfig(epochs=2, device=device)

    # 3. Client Manager for Simulation
    def client_fn(cid: str) -> FlowerPrototypeClient:
        return FlowerPrototypeClient(int(cid), train_ds, test_ds, split, cfg_train)

    # 4. Custom Strategy
    strategy = PrototypeStrategy(
        num_classes=10,
        min_fit_clients=num_clients,
        min_available_clients=num_clients,
    )

    # 5. Start Simulation
    print(f"\n>>> Starting Flower Prototype-FL Simulation (Alpha={alpha}, Rounds={num_rounds}) <<<")
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.2 if torch.cuda.is_available() else 0},
    )

if __name__ == "__main__":
    main()
