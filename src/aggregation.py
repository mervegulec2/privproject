import numpy as np
import flwr as fl
from typing import List, Dict, Optional, Tuple, Union
import pickle
from flwr.common import Metrics, Scalar, Parameters, FitRes, FitIns, NDArrays, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy

class PrototypeStrategy(fl.server.strategy.FedAvg):
    """
    Custom Flower Strategy for Prototype FL.
    - Ignores model weights (no weight aggregation).
    - Aggregates class-wise prototypes collected from clients.
    """
    def __init__(self, num_classes: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.global_prototypes: Dict[int, np.ndarray] = {}

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Server aggregations:
        1. Ignore parameters (weights).
        2. Aggregate prototypes from metrics.
        """
        if not results:
            return None, {}

        # 1. Collect prototypes from all clients
        client_protos_list = []
        for _, fit_res in results:
            # We will send prototypes as flattened lists in metrics
            protos = self._unpack_prototypes(fit_res.metrics)
            client_protos_list.append(protos)

        # 2. Aggregate prototypes
        self.global_prototypes = self._aggregate_protos(client_protos_list)
        
        # 3. Report average accuracy across clients
        accuracies = [fit_res.metrics["accuracy"] for _, fit_res in results if "accuracy" in fit_res.metrics]
        avg_acc = sum(accuracies) / len(accuracies) if accuracies else 0.0
        
        print(f"\n[Round {server_round}] Server Aggregated {len(results)} clients. Avg Acc: {avg_acc:.4f}")

        # We return None for parameters because we don't aggregate weights
        return None, {"avg_accuracy": avg_acc}

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: fl.server.client_manager.ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Send the current global prototypes to clients via the config."""
        config = self._pack_prototypes(self.global_prototypes)
        config["server_round"] = server_round
        fit_ins = FitIns(parameters, config)
        clients = client_manager.sample(num_clients=self.min_fit_clients, min_num_clients=self.min_available_clients)
        return [(client, fit_ins) for client in clients]

    def _aggregate_protos(self, client_protos_list: List[Dict[int, np.ndarray]]) -> Dict[int, np.ndarray]:
        global_protos = {}
        for c in range(self.num_classes):
            class_protos = [p[c] for p in client_protos_list if c in p]
            if class_protos:
                global_protos[c] = np.mean(class_protos, axis=0)
        return global_protos

    def _pack_prototypes(self, protos: Dict[int, np.ndarray]) -> Dict[str, Scalar]:
        """Convert prototypes dict to bytes for Flower transmission."""
        if not protos:
            return {}
        # We use pickle to serialize the entire dictionary into a bytes object
        return {"protos_bytes": pickle.dumps(protos)}

    def _unpack_prototypes(self, metrics: Dict[str, Scalar]) -> Dict[int, np.ndarray]:
        """Reconstruct prototypes dict from Flower bytes."""
        if "protos_bytes" in metrics:
            return pickle.loads(metrics["protos_bytes"])
        return {}
