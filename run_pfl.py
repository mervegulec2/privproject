import os
import shutil
import torch
import numpy as np
import pickle
from torch.utils.data import DataLoader, Subset

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from typing import Dict
from src.data_utils import (
    load_cifar10,
    Cifar10Config,
    DirichletSplitConfig,
    dirichlet_split_indices,
    load_split,
    get_seen_classes,
)
from src.models import ResNet18Cifar
from src.train_utils import TrainConfig, train_local_proto, compute_prototypes, evaluate_accuracy, set_seed, compute_class_weights
from src.aggregation import PrototypeStrategy
from src.security.manager import SecurityManager
from src.security.plotter import plot_accuracy_curves
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




class FlowerPrototypeClient(fl.client.NumPyClient if fl is not None else object):  # type: ignore[misc]
    """Prototype-only Flower client: shares prototypes, keeps all weights local."""

    def __init__(self, cid: int, train_ds, test_sets: Dict[str, Subset], split_indices: np.ndarray, cfg: TrainConfig, security_manager: SecurityManager = None):
        self.cid = cid
        self.train_ds = train_ds
        self.test_sets = test_sets
        self.split_indices = split_indices
        max_samples = int(os.environ.get("MAX_SAMPLES_PER_CLIENT", "0"))
        if max_samples > 0 and len(self.split_indices) > max_samples:
            self.split_indices = self.split_indices[:max_samples]
        self.cfg = cfg
        self.security_manager = security_manager

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

        use_cw = os.environ.get("USE_CLASS_WEIGHTS", "1") != "0"
        class_weights = (
            compute_class_weights(self.train_ds, self.split_indices, num_classes=10, device=self.device)
            if use_cw else None
        )

        progress = os.environ.get("PROGRESS", "0") != "0"
        lambda_p = float(os.environ.get("LD", os.environ.get("LAMBDA_P", "0.1")))
        train_local_proto(self.model, train_loader, global_protos, self.cfg, lambda_p=lambda_p, progress=progress, cid=self.cid, class_weights=class_weights)
        
        # Save model after training
        torch.save(self.model.state_dict(), self.model_path)

        local_protos, local_counts = compute_prototypes(self.model, train_loader, self.device)

        # Apply Defenses (Modular Hook)
        if self.security_manager:
            local_protos = self.security_manager.apply_defenses(local_protos, local_counts)

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


def _reset_ray_if_needed() -> None:
    try:
        import ray  # type: ignore
        if ray.is_initialized():
            ray.shutdown()
    except Exception:
        pass


def _clear_client_model_checkpoints() -> None:
    d = os.path.join("outputs", "client_models")
    if os.path.isdir(d):
        shutil.rmtree(d)
    os.makedirs(d, exist_ok=True)


def run_flower_experiment(
    *,
    exp_mode: str = "baseline",
    epochs: int,
    train_transform: str = "default",
    mixup_alpha: float = 0.0,
    swa_enabled: bool = False,
    swa_last_epochs: int = 2,
    num_clients: int,
    num_rounds: int,
    alpha: float,
    seed: int,
    device: str,
    train_ds,
    test_ds,
    split,
    metrics_csv_path: str,
    plot_path: str,
    # Explicit hyperparameter overrides (used by the sweep; env-vars are the fallback)
    lambda_p: float = None,
    use_class_weights: bool = None,
) -> None:
    """Single Flower Ray simulation with metrics/plots written to the given paths."""
    train_backbone = os.environ.get("TRAIN_BACKBONE", "1") != "0"
    train_head = os.environ.get("TRAIN_HEAD", "1") != "0"

    # Apply explicit overrides to env so FlowerPrototypeClient picks them up
    if lambda_p is not None:
        os.environ["LAMBDA_P"] = str(lambda_p)
        os.environ["LD"] = str(lambda_p)
    if use_class_weights is not None:
        os.environ["USE_CLASS_WEIGHTS"] = "1" if use_class_weights else "0"

    cfg_train = TrainConfig(
        epochs=epochs,
        device=device,
        train_backbone=train_backbone,
        train_head=train_head,
        mixup_alpha=mixup_alpha,
        swa_enabled=swa_enabled,
        swa_last_epochs=swa_last_epochs,
    )

    security_manager = SecurityManager(
        active_defenses=[],
        active_attacks=[],
        log_model_state=os.environ.get("SECURITY_LOGGING", "1") == "1",
    )

    if fl is None:
        raise RuntimeError("flwr is required. Install flwr in the active venv.")

    print(
        f"\n>>> Flower Simulation | exp_mode={exp_mode} | train_tf={train_transform} "
        f"mixup={mixup_alpha} swa={swa_enabled} (last_epochs={swa_last_epochs}) | "
        f"clients={num_clients} rounds={num_rounds} local_epochs={epochs} "
        f"LD={os.environ.get('LD', os.environ.get('LAMBDA_P', '0.1'))} device={cfg_train.device} | "
        f"metrics_csv={metrics_csv_path} <<<\n",
        flush=True,
    )

    def client_fn(cid: str) -> fl.client.Client:
        cid_int = int(cid)
        seen_classes = get_seen_classes(train_ds, split[cid_int])

        lp_indices = create_local_proportional_indices(
            test_ds, train_ds, split[cid_int], seed=seed
        )
        la_indices = create_local_aware_indices(test_ds, seen_classes)

        test_sets = {
            "global": test_ds,
            "local_proportional": Subset(test_ds, lp_indices),
            "local": Subset(test_ds, la_indices),
        }
        return FlowerPrototypeClient(
            cid_int, train_ds, test_sets, split[cid_int], cfg_train, security_manager
        ).to_client()

    strategy = PrototypeStrategy(
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=0,
        min_available_clients=num_clients,
        num_classes=10,
        security_manager=security_manager,
        metrics_csv_path=metrics_csv_path,
    )

    client_resources = {"num_cpus": 1, "num_gpus": 0.0}
    if cfg_train.device and "cuda" in str(cfg_train.device):
        client_resources["num_gpus"] = float(os.environ.get("RAY_NUM_GPUS", "0.2"))

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources=client_resources,
    )

    plot_accuracy_curves(metrics_csv_path, save_path=plot_path)


# ---------------------------------------------------------------------------
# Baseline entry point — single FL run with default settings.
# For performance experiments (epoch sweep / augment / swa) use:
#   python pfl_performance_experiments.py --exp-mode <epochs|augment|swa>
# ---------------------------------------------------------------------------
def main():
    num_clients = int(os.environ.get("NUM_CLIENTS", "10"))
    num_rounds = int(os.environ.get("NUM_ROUNDS", "2"))
    epochs = int(os.environ.get("EPOCHS", "5"))
    alpha = float(os.environ.get("ALPHA", "0.1"))
    seed = int(os.environ.get("SEED", "42"))
    device = _select_device()

    set_seed(seed)
    train_ds, test_ds = load_cifar10(Cifar10Config(root="data"))

    split_path = f"outputs/splits/cifar10_dirichlet_a{alpha}_s{seed}_c{num_clients}.npy"
    if not os.path.exists(split_path):
        cfg_s = DirichletSplitConfig(num_clients=num_clients, alpha=alpha, seed=seed)
        split = dirichlet_split_indices(train_ds, cfg_s)
    else:
        split = load_split(split_path)

    base = os.path.join("outputs", "metrics")
    os.makedirs(base, exist_ok=True)
    _reset_ray_if_needed()
    _clear_client_model_checkpoints()

    run_flower_experiment(
        epochs=epochs,
        num_clients=num_clients,
        num_rounds=num_rounds,
        alpha=alpha,
        seed=seed,
        device=device,
        train_ds=train_ds,
        test_ds=test_ds,
        split=split,
        metrics_csv_path=os.path.join(base, "simulation_results.csv"),
        plot_path=os.path.join(base, "accuracy_curve.png"),
    )


if __name__ == "__main__":
    main()
