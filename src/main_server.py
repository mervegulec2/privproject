from __future__ import annotations
import flwr as fl

from src.models.resnet_cifar import ResNet18Cifar
from src.fl.strategy import NoAggStrategy


def get_initial_parameters() -> fl.common.Parameters:
    model = ResNet18Cifar(num_classes=10)
    ndarrays = [v.detach().cpu().numpy() for _, v in model.state_dict().items()]
    return fl.common.ndarrays_to_parameters(ndarrays)


def fit_config(server_round: int):
    return {"proto_out_dir": "outputs/client_protos"}


def main():
    init_params = get_initial_parameters()

    strategy = NoAggStrategy(
        initial_parameters=init_params,
        fraction_fit=1.0,
        min_fit_clients=15,          # ✅ debug: later set to 15
        min_available_clients=15,    # ✅ debug: later set to 15

        # ✅ IMPORTANT: disable evaluation to avoid ZeroDivisionError
        fraction_evaluate=0.0,
        min_evaluate_clients=0,

        on_fit_config_fn=fit_config,
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=1),
        strategy=strategy,
    )

    print("✅ Round finished. Prototypes should be in outputs/client_protos/")


if __name__ == "__main__":
    main()