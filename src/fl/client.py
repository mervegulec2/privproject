from __future__ import annotations

import os
import csv
import numpy as np
import flwr as fl
import torch
from torch.utils.data import DataLoader, Subset

from src.models.resnet_cifar import ResNet18Cifar
from src.eval.train_eval import TrainConfig, train_one_client, accuracy
from src.proto.compute import compute_class_prototypes


def _append_row_csv_atomic(csv_path: str, header: list[str], row: list):
    """Append one row to a CSV safely when multiple processes write concurrently (macOS/Linux)."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    with open(csv_path, "a", newline="") as f:
        # File lock (Unix/macOS)
        try:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        except Exception:
            pass

        # Write header once if file is empty
        if f.tell() == 0:
            csv.writer(f).writerow(header)

        csv.writer(f).writerow(row)

        try:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass


class OneShotClient(fl.client.NumPyClient):
    """
    One-shot client:
    - Receives global init weights
    - Trains locally (one time)
    - Computes class-wise prototypes (512-d)
    - Saves prototypes to disk as the shared artifact
    - Logs local utility (test accuracy) to CSV
    - Returns model weights to Flower (server ignores aggregation)
    """

    def __init__(self, cid: int, train_ds, test_ds, train_idx, cfg: TrainConfig):
        self.cid = cid
        self.cfg = cfg
        self.model = ResNet18Cifar(num_classes=10)

        self.train_loader = DataLoader(
            Subset(train_ds, train_idx),
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=0,
        )
        self.test_loader = DataLoader(
            test_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=0,
        )

    def get_parameters(self, config):
        return [val.detach().cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        state_dict = self.model.state_dict()
        keys = list(state_dict.keys())
        for k, p in zip(keys, parameters):
            state_dict[k] = torch.tensor(p)
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        # 1) Initialize with global params
        self.set_parameters(parameters)

        # Output dirs (configurable from server)
        proto_out_dir = config.get("proto_out_dir", "outputs/client_protos")
        metrics_out_dir = config.get("metrics_out_dir", "outputs/metrics")
        os.makedirs(proto_out_dir, exist_ok=True)
        os.makedirs(metrics_out_dir, exist_ok=True)

        # Run tags (for bookkeeping)
        alpha_tag = str(config.get("alpha", "unknown"))
        seed_tag = str(config.get("seed", "unknown"))

        # 2) Local train
        print(f"\n[Client {self.cid}] local training")
        train_one_client(self.model, self.train_loader, self.cfg)

        # 3) Utility: local test accuracy
        acc = accuracy(self.model, self.test_loader, device=self.cfg.device)
        print(f"[Client {self.cid}] local test accuracy: {acc:.4f}")

        # 4) Compute prototypes
        protos = compute_class_prototypes(
            self.model,
            self.train_loader,
            device=self.cfg.device,
            num_classes=10,
        )

        # 5) Save prototypes (artifact)
        class_ids = np.array(sorted(protos.keys()), dtype=np.int64)
        if len(class_ids) == 0:
            mat = np.zeros((0, 512), dtype=np.float32)
        else:
            mat = np.stack([protos[int(c)] for c in class_ids], axis=0).astype(np.float32)

        np.savez(
            os.path.join(proto_out_dir, f"client_{self.cid}_prototypes.npz"),
            class_ids=class_ids,
            prototypes=mat,
        )
        print(
            f"[Client {self.cid}] saved prototypes to {proto_out_dir}/client_{self.cid}_prototypes.npz "
            f"(num_classes={len(protos)})"
        )

        # 6) Log accuracy to CSV (atomic append)
        csv_path = os.path.join(metrics_out_dir, "client_accuracy.csv")
        header = ["alpha", "seed", "client_id", "local_test_accuracy", "num_train_samples", "num_proto_classes"]
        row = [alpha_tag, seed_tag, int(self.cid), float(acc), int(len(self.train_loader.dataset)), int(len(protos))]
        _append_row_csv_atomic(csv_path, header, row)

        # Return model weights to Flower (server ignores aggregation anyway)
        ndarrays = [val.detach().cpu().numpy() for _, val in self.model.state_dict().items()]
        num_examples = len(self.train_loader.dataset)
        metrics = {"local_acc": float(acc), "num_protos": int(len(protos))}
        return ndarrays, num_examples, metrics

    def evaluate(self, parameters, config):
        return 0.0, 0, {}