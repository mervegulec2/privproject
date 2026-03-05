from __future__ import annotations
import argparse
import flwr as fl

from src.data.cifar import Cifar10Config, load_cifar10
from src.data.split import load_split
from src.eval.train_eval import TrainConfig
from src.fl.client import OneShotClient
from src.utils.seed import set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cid", type=int, required=True)
    ap.add_argument("--split_path", type=str, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    set_seed(args.seed)

    train_ds, test_ds = load_cifar10(Cifar10Config(root="data"))
    split = load_split(args.split_path)
    train_idx = split[args.cid]

    # ✅ tuned config (better accuracy baseline under non-IID)
    cfg = TrainConfig(
        epochs=5,          # was 5
        batch_size=64,
        lr=0.01,            # was 0.1
        momentum=0.9,
        weight_decay=5e-4,
        device=args.device,
    )

    client = OneShotClient(args.cid, train_ds, test_ds, train_idx, cfg)

    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=client.to_client(),
    )


if __name__ == "__main__":
    main()