import os
import time
import multiprocessing as mp
import torch
import csv
import numpy as np
from torch.utils.data import DataLoader, Subset
from typing import Dict
from tqdm import tqdm

from src.data_utils import load_cifar10, Cifar10Config, DirichletSplitConfig, dirichlet_split_indices, load_split, save_split
from src.models import ResNet18Cifar
from src.train_utils import TrainConfig, train_local_proto, compute_prototypes, evaluate_accuracy, set_seed
from src.aggregation import PrototypeStrategy

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
    def __init__(self, cid: int, train_ds, test_ds, split, cfg: TrainConfig):
        self.cid = cid
        self.train_ds = train_ds
        self.test_ds = test_ds
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
        progress = os.environ.get("PROGRESS", "1") != "0"
        # FedProto uses a prototype-loss weight "ld". Their CIFAR-10 example uses ld=0.1.
        lambda_p = float(os.environ.get("LD", os.environ.get("LAMBDA_P", "0.1")))
        train_local_proto(self.model, train_loader, global_protos, self.cfg, lambda_p=lambda_p, progress=progress)

        # Compute local prototypes to send to server
        local_protos = compute_prototypes(self.model, train_loader, self.device)
        
        # Evaluate on the shared test set (utility)
        max_test = int(os.environ.get("MAX_TEST_SAMPLES", "0"))
        test_ds = self.test_ds
        if max_test > 0 and len(test_ds) > max_test:
            test_ds = Subset(test_ds, list(range(max_test)))
        test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)
        acc = evaluate_accuracy(self.model, test_loader, self.device)

        return {"n": len(self.split_indices), "accuracy": float(acc), "protos": local_protos}


class FlowerPrototypeClient(fl.client.NumPyClient if fl is not None else object):  # type: ignore[misc]
    """Prototype-only Flower client: shares prototypes, keeps all weights local."""

    def __init__(self, cid: int, train_ds, test_ds, split_indices: np.ndarray, cfg: TrainConfig):
        self.cid = cid
        self.train_ds = train_ds
        self.test_ds = test_ds
        self.split_indices = split_indices
        max_samples = int(os.environ.get("MAX_SAMPLES_PER_CLIENT", "0"))
        if max_samples > 0 and len(self.split_indices) > max_samples:
            self.split_indices = self.split_indices[:max_samples]
        self.cfg = cfg

        self.model = ResNet18Cifar(num_classes=10).to(cfg.device)
        self.device = cfg.device

    def get_parameters(self, config):
        return []

    def fit(self, parameters, config):
        # Unpack global prototypes from server config.
        global_protos: Dict[int, np.ndarray] = {}
        for key, val in config.items():
            if key.startswith("proto_"):
                class_id = int(key.split("_")[1])
                global_protos[class_id] = np.array(val, dtype=np.float32)

        train_loader = DataLoader(
            Subset(self.train_ds, self.split_indices),
            batch_size=self.cfg.batch_size,
            shuffle=True,
        )

        progress = os.environ.get("PROGRESS", "1") != "0"
        lambda_p = float(os.environ.get("LD", os.environ.get("LAMBDA_P", "0.1")))
        train_local_proto(self.model, train_loader, global_protos, self.cfg, lambda_p=lambda_p, progress=progress)

        local_protos = compute_prototypes(self.model, train_loader, self.device)

        # Evaluate (utility).
        max_test = int(os.environ.get("MAX_TEST_SAMPLES", "0"))
        test_ds = self.test_ds
        if max_test > 0 and len(test_ds) > max_test:
            test_ds = Subset(test_ds, list(range(max_test)))
        test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)
        acc = float(evaluate_accuracy(self.model, test_loader, self.device))

        metrics: Dict[str, object] = {"accuracy": acc}
        for c, vec in local_protos.items():
            metrics[f"proto_{c}"] = vec.tolist()

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
    client = FlowerPrototypeClient(cid, train_ds, test_ds, split[cid], cfg)
    fl.client.start_numpy_client(server_address=server_address, client=client)


def aggregate_prototypes(client_proto_dicts: list[Dict[int, np.ndarray]], num_classes: int = 10) -> Dict[int, np.ndarray]:
    """Mean-aggregate class-wise prototypes across clients (only for classes present)."""
    global_protos: Dict[int, np.ndarray] = {}
    for c in range(num_classes):
        class_protos = [p[c] for p in client_proto_dicts if c in p]
        if class_protos:
            global_protos[c] = np.mean(class_protos, axis=0).astype(np.float32)
    return global_protos

# 2. Simulation Orchestrator
def main():
    # Parameters
    num_clients = int(os.environ.get("NUM_CLIENTS", "10"))
    num_rounds = int(os.environ.get("NUM_ROUNDS", "20"))
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

    use_flower = os.environ.get("USE_FLOWER", "0") == "1"
    if use_flower:
        if fl is None:
            raise RuntimeError("USE_FLOWER=1 but flwr import failed. Install flwr in the active venv.")

        # For multi-process Flower mode, default to CPU unless DEVICE is explicitly set.
        # (Users can override with DEVICE=cuda, but it may thrash/OOM.)
        if os.environ.get("DEVICE", "") == "":
            cfg_train = TrainConfig(
                epochs=epochs,
                device="cpu",
                train_backbone=train_backbone,
                train_head=train_head,
            )

        # Ensure split exists on disk so child processes can load it.
        if not os.path.exists(split_path):
            save_split(split, split_path)

        server_address = os.environ.get("SERVER_ADDRESS", "127.0.0.1:8080")

        mp.set_start_method("spawn", force=True)

        print(
            f"\n>>> Flower gRPC FedProto-style (no Ray) | addr={server_address} | clients={num_clients} rounds={num_rounds} epochs={epochs} LD={os.environ.get('LD', os.environ.get('LAMBDA_P', '0.1'))} device={cfg_train.device} <<<",
            flush=True,
        )

        server = mp.Process(target=_flower_server_proc, args=(server_address, num_rounds, num_clients), daemon=True)
        server.start()
        time.sleep(2.0)

        cfg_dict = {
            "epochs": cfg_train.epochs,
            "batch_size": cfg_train.batch_size,
            "lr": cfg_train.lr,
            "momentum": cfg_train.momentum,
            "weight_decay": cfg_train.weight_decay,
            "device": cfg_train.device,
            "train_backbone": cfg_train.train_backbone,
            "train_head": cfg_train.train_head,
        }

        procs: list[mp.Process] = []
        for cid in range(num_clients):
            p = mp.Process(target=_flower_client_proc, args=(server_address, cid, split_path, cfg_dict))
            p.start()
            procs.append(p)

        for p in procs:
            p.join()
        if server.is_alive():
            server.join(timeout=5)
        return

    # Local loop mode: Build persistent clients (clients keep their own heads/models)
    clients = [LocalPrototypeClient(cid, train_ds, test_ds, split, cfg_train) for cid in range(num_clients)]

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
        round_accs = []
        client_iter = clients
        if progress:
            client_iter = tqdm(clients, desc=f"Round {r}/{num_rounds} clients", leave=False)
        for client in client_iter:
            out = client.fit(global_prototypes)
            round_proto_dicts.append(out["protos"])
            round_accs.append(out["accuracy"])
            if progress and hasattr(client_iter, "set_postfix_str"):
                client_iter.set_postfix_str(f"last_acc={out['accuracy']:.3f}")

        global_prototypes = aggregate_prototypes(round_proto_dicts, num_classes=10)
        avg_acc = float(np.mean(round_accs)) if round_accs else 0.0
        
        # 1. Detailed Terminal Output (goes to simulation_log.txt via tee)
        acc_list_str = ", ".join([f"C{i}:{a:.4f}" for i, a in enumerate(round_accs)])
        print(f"[Round {r}] Avg Acc: {avg_acc:.4f} | Clients: {acc_list_str}", flush=True)

        # 2. Save to CSV Output File
        log_dir = "outputs/metrics"
        os.makedirs(log_dir, exist_ok=True)
        csv_path = os.path.join(log_dir, "simulation_results.csv")
        file_exists = os.path.isfile(csv_path)
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["round", "avg_accuracy"] + [f"client_{i}" for i in range(num_clients)])
            writer.writerow([r, avg_acc] + round_accs)

if __name__ == "__main__":
    main()
