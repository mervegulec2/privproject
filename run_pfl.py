import os
import time
import multiprocessing as mp
import torch
import csv
import numpy as np
import pickle
from torch.utils.data import DataLoader, Subset
from typing import Dict
from tqdm import tqdm
from src.data_utils import load_cifar10, Cifar10Config, DirichletSplitConfig, dirichlet_split_indices, load_split, get_seen_classes
from src.models import ResNet18Cifar
from src.train_utils import TrainConfig, train_local_proto, compute_prototypes, evaluate_accuracy, set_seed, compute_class_weights
from src.aggregation import PrototypeStrategy
from src.eval.test_sets import (
    create_local_proportional_indices,
    create_local_aware_indices
)

# Flower is optional in this file: USE_FLOWER=1 enables gRPC server+client mode (no Ray).
try:
    import flwr as fl  # type: ignore
except Exception:  # pragma: no cover
    fl = None  # type: ignore


def _select_device() -> str:
    """Pick device, default CPU for multi-process safety unless explicitly requested."""
    requested = os.environ.get("DEVICE", "")
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

# 1. Define the local client (no Ray/Flower dependency required)
class LocalPrototypeClient:
    def __init__(self, cid: int, train_ds, test_sets: Dict[str, Subset], split, cfg: TrainConfig):
        self.cid = cid
        self.train_ds = train_ds
        self.test_sets = test_sets
        self.split_indices = split[cid]
        max_samples = int(os.environ.get("MAX_SAMPLES_PER_CLIENT", "0"))
        if max_samples > 0 and len(self.split_indices) > max_samples:
            # Keep it deterministic for quick smoke tests
            self.split_indices = self.split_indices[:max_samples]
        self.cfg = cfg
        
        # Persistent local model
        self.model = ResNet18Cifar(num_classes=10).to(cfg.device)
        self.device = cfg.device

    def fit(self, global_protos: Dict[int, np.ndarray]) -> Dict[str, object]:
        """Local training with prototype alignment + report local prototypes and accuracy."""
        train_loader = DataLoader(
            Subset(self.train_ds, self.split_indices), 
            batch_size=self.cfg.batch_size, 
            shuffle=True
        )
        
        class_weights = compute_class_weights(self.train_ds, self.split_indices, num_classes=10, device=self.device)
        
        progress = os.environ.get("PROGRESS", "0") != "0"
        lambda_p = float(os.environ.get("LD", os.environ.get("LAMBDA_P", "0.1")))
        train_local_proto(self.model, train_loader, global_protos, self.cfg, lambda_p=lambda_p, progress=progress, cid=self.cid, class_weights=class_weights)

        local_protos, local_counts = compute_prototypes(self.model, train_loader, self.device)
        
        # Evaluate on all provided test sets
        accuracies = {}
        for name, subset in self.test_sets.items():
            if len(subset) == 0:
                accuracies[name] = 0.0
                continue
            test_loader = DataLoader(subset, batch_size=256, shuffle=False)
            accuracies[name] = float(evaluate_accuracy(self.model, test_loader, self.device))

        return {"n": len(self.split_indices), "accuracies": accuracies, "protos": local_protos, "counts": local_counts}


class FlowerPrototypeClient(fl.client.NumPyClient if fl is not None else object):  # type: ignore[misc]
    """Prototype-only Flower client: shares prototypes, keeps all weights local."""

    def __init__(self, cid: int, train_ds, test_sets: Dict[str, Subset], split_indices: np.ndarray, cfg: TrainConfig):
        self.cid = cid
        self.train_ds = train_ds
        self.test_sets = test_sets
        self.split_indices = split_indices
        max_samples = int(os.environ.get("MAX_SAMPLES_PER_CLIENT", "0"))
        if max_samples > 0 and len(self.split_indices) > max_samples:
            self.split_indices = self.split_indices[:max_samples]
        self.cfg = cfg

        self.model = ResNet18Cifar(num_classes=10).to(cfg.device)
        self.device = cfg.device

        # Persist client model across simulated rounds
        self.model_dir = "outputs/client_models"
        os.makedirs(self.model_dir, exist_ok=True)
        self.model_path = os.path.join(self.model_dir, f"client_{self.cid}.pt")
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device, weights_only=True))

    def get_parameters(self, config):
        return []

    def fit(self, parameters, config):
        # Unpack global prototypes from server config (using pickle).
        global_protos: Dict[int, np.ndarray] = {}
        if "protos_bytes" in config:
            global_protos = pickle.loads(config["protos_bytes"])

        train_loader = DataLoader(
            Subset(self.train_ds, self.split_indices),
            batch_size=self.cfg.batch_size,
            shuffle=True,
        )

        class_weights = compute_class_weights(self.train_ds, self.split_indices, num_classes=10, device=self.device)

        progress = os.environ.get("PROGRESS", "0") != "0"
        lambda_p = float(os.environ.get("LD", os.environ.get("LAMBDA_P", "0.1")))
        train_local_proto(self.model, train_loader, global_protos, self.cfg, lambda_p=lambda_p, progress=progress, cid=self.cid, class_weights=class_weights)
        
        # Save model after training
        torch.save(self.model.state_dict(), self.model_path)

        local_protos, local_counts = compute_prototypes(self.model, train_loader, self.device)

        # Evaluate on all provided test sets
        accuracies = {}
        for name, subset in self.test_sets.items():
            if len(subset) == 0:
                accuracies[name] = 0.0
                continue
            test_loader = DataLoader(subset, batch_size=256, shuffle=False)
            accuracies[name] = float(evaluate_accuracy(self.model, test_loader, self.device))

        # Metrics for server: send serialized bytes for prototypes and counts
        metrics: Dict[str, object] = {f"acc_{name}": acc for name, acc in accuracies.items()}
        metrics["accuracy"] = accuracies["local"] # Standard fallback
        metrics["cid"] = self.cid
        metrics["protos_bytes"] = pickle.dumps(local_protos)
        metrics["counts_bytes"] = pickle.dumps(local_counts)

        return [], len(self.split_indices), metrics


def _flower_server_proc(server_address: str, num_rounds: int, num_clients: int) -> None:
    if fl is None:
        raise RuntimeError("flwr is not installed but USE_FLOWER=1 was set.")
    strategy = PrototypeStrategy(
        num_classes=10,
        min_fit_clients=num_clients,
        min_available_clients=num_clients,
    )
    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )


def _flower_client_proc(server_address: str, cid: int, split_path: str, cfg_dict: dict) -> None:
    if fl is None:
        raise RuntimeError("flwr is not installed but USE_FLOWER=1 was set.")
    cfg = TrainConfig(**cfg_dict)
    train_ds, test_ds = load_cifar10(Cifar10Config(root="data"))
    split = load_split(split_path)
    seed = int(os.environ.get("SEED", "42"))

    # Prepare Test Sets (Same logic as main)
    seen_classes = get_seen_classes(train_ds, split[cid])
    lp_indices = create_local_proportional_indices(test_ds, train_ds, split[cid], total_test_samples=1000, seed=seed)
    la_indices = create_local_aware_indices(test_ds, seen_classes)

    test_sets = {
        "global": test_ds,
        "local_proportional": Subset(test_ds, lp_indices),
        "local": Subset(test_ds, la_indices),
    }

    client = FlowerPrototypeClient(cid, train_ds, test_sets, split[cid], cfg)
    fl.client.start_numpy_client(server_address=server_address, client=client)


def aggregate_prototypes(client_proto_dicts: list[Dict[int, np.ndarray]], client_count_dicts: list[Dict[int, int]], num_classes: int = 10) -> Dict[int, np.ndarray]:
    """Weighted mean-aggregate class-wise prototypes across clients (Eq 6)."""
    global_protos: Dict[int, np.ndarray] = {}
    for c in range(num_classes):
        c_protos = []
        c_counts = []
        for p_dict, c_dict in zip(client_proto_dicts, client_count_dicts):
            if c in p_dict and c in c_dict:
                c_protos.append(p_dict[c])
                c_counts.append(c_dict[c])
        if c_protos:
            total = sum(c_counts)
            weighted_sum = sum(p * cnt for p, cnt in zip(c_protos, c_counts))
            global_protos[c] = (weighted_sum / total).astype(np.float32)
    return global_protos

# 2. Simulation Orchestrator
def main():
    # Parameters
    num_clients = int(os.environ.get("NUM_CLIENTS", "10"))
    num_rounds = int(os.environ.get("NUM_ROUNDS", "2"))
    alpha = float(os.environ.get("ALPHA", "0.1"))
    seed = int(os.environ.get("SEED", "42"))
    device = _select_device()
    
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

    # FedProto-style local training: clients keep their full model local,
    # and only share class-wise prototypes. For stable prototypes, we train the backbone too.
    epochs = int(os.environ.get("EPOCHS", "5"))
    train_backbone = os.environ.get("TRAIN_BACKBONE", "1") != "0"
    train_head = os.environ.get("TRAIN_HEAD", "1") != "0"
    cfg_train = TrainConfig(epochs=epochs, device=device, train_backbone=train_backbone, train_head=train_head)

    use_flower = True  # Changed from os.environ.get("USE_FLOWER", "0") == "1"
    
    if use_flower:
        if fl is None:
            raise RuntimeError("USE_FLOWER=1 but flwr import failed. Install flwr in the active venv.")

        server_address = os.environ.get("SERVER_ADDRESS", "127.0.0.1:8080")

        print(
            f"\n>>> Flower Simulation (Virtual Client Engine) | clients={num_clients} rounds={num_rounds} epochs={epochs} LD={os.environ.get('LD', os.environ.get('LAMBDA_P', '0.1'))} device={cfg_train.device} <<<",
            flush=True,
        )

        def client_fn(cid: str) -> fl.client.Client:
            cid_int = int(cid)
            seen_classes = get_seen_classes(train_ds, split[cid_int])
            
            lp_indices = create_local_proportional_indices(test_ds, train_ds, split[cid_int], total_test_samples=1000, seed=seed)
            la_indices = create_local_aware_indices(test_ds, seen_classes)
            
            test_sets = {
                "global": test_ds,
                "local_proportional": Subset(test_ds, lp_indices),
                "local": Subset(test_ds, la_indices)
            }
            return FlowerPrototypeClient(cid_int, train_ds, test_sets, split[cid_int], cfg_train).to_client()
            
        strategy = PrototypeStrategy(
            fraction_fit=1.0,
            fraction_evaluate=0.0,
            min_fit_clients=num_clients,
            min_evaluate_clients=0,
            min_available_clients=num_clients,
            num_classes=10
        )
        
        # Determine client resources for Ray
        client_resources = {"num_cpus": 1, "num_gpus": 0.0}
        if cfg_train.device and "cuda" in str(cfg_train.device):
            # Assign fractional GPU so multiple virtual clients can run concurrently on same GPU
            client_resources["num_gpus"] = float(os.environ.get("RAY_NUM_GPUS", "0.2"))

        # Start Ray-based Flower Simulation
        fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=num_clients,
            config=fl.server.ServerConfig(num_rounds=num_rounds),
            strategy=strategy,
            client_resources=client_resources
        )
        return

    clients = []
    for cid in range(num_clients):
        seen_classes = get_seen_classes(train_ds, split[cid])
        
        lp_indices = create_local_proportional_indices(test_ds, train_ds, split[cid], total_test_samples=1000, seed=seed)
        la_indices = create_local_aware_indices(test_ds, seen_classes)
        
        client_test_sets = {
            "global": test_ds,
            "local_proportional": Subset(test_ds, lp_indices),
            "local": Subset(test_ds, la_indices)
        }
        clients.append(LocalPrototypeClient(cid, train_ds, client_test_sets, split, cfg_train))

    # Local prototype-FL loop (no Ray needed)
    global_prototypes: Dict[int, np.ndarray] = {}
    print(
        f"\n>>> Starting Local Prototype-FL Simulation (Alpha={alpha}, Rounds={num_rounds}, Clients={num_clients}, Epochs={epochs}, LD={os.environ.get('LD', os.environ.get('LAMBDA_P', '0.1'))}) <<<",
        flush=True,
    )
    progress = os.environ.get("PROGRESS", "1") != "0"
    round_iter = range(1, num_rounds + 1)
    if progress:
        round_iter = tqdm(round_iter, desc="Rounds")

    for r in round_iter:
        round_proto_dicts = []
        round_count_dicts = []
        round_metrics = {
            "global": [],
            "local_proportional": [],
            "local": []
        }
        
        client_iter = clients
        if progress:
            client_iter = tqdm(clients, desc=f"Round {r}/{num_rounds} clients", leave=False)
        for client in client_iter:
            out = client.fit(global_prototypes)
            round_proto_dicts.append(out["protos"])
            round_count_dicts.append(out["counts"])
            for name, acc in out["accuracies"].items():
                round_metrics[name].append(acc)
            
            if progress and hasattr(client_iter, "set_postfix_str"):
                client_iter.set_postfix_str(f"local_acc={out['accuracies']['local']:.3f}")

        global_prototypes = aggregate_prototypes(round_proto_dicts, round_count_dicts, num_classes=10)
        
        avgs = {name: float(np.mean(accs)) if accs else 0.0 for name, accs in round_metrics.items()}
        
        # 1. Detailed Terminal Output
        print(f"[Round {r}] Avg Accs -> Global: {avgs['global']:.4f} | Local-Prop: {avgs['local_proportional']:.4f} | Local: {avgs['local']:.4f}", flush=True)

        # 2. Save to CSV Output File
        log_dir = "outputs/metrics"
        os.makedirs(log_dir, exist_ok=True)
        csv_path = os.path.join(log_dir, "simulation_results.csv")
        file_exists = os.path.isfile(csv_path)
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                headers = ["round", "avg_global", "avg_local_proportional", "avg_local"]
                writer.writerow(headers)
            writer.writerow([r, avgs["global"], avgs["local_proportional"], avgs["local"]])

        # 3. Save final local prototypes and models at the last round
        if r == num_rounds:
            model_dir = "outputs/client_models"
            os.makedirs(model_dir, exist_ok=True)
            for client in clients:
                torch.save(client.model.state_dict(), os.path.join(model_dir, f"client_{client.cid}.pt"))
            
            with open(os.path.join(log_dir, "final_local_protos.pkl"), "wb") as f:
                pickle.dump(round_proto_dicts, f)

if __name__ == "__main__":
    main()
