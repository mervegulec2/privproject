import numpy as np
import flwr as fl
from typing import List, Dict, Optional, Tuple, Union
from flwr.common import Metrics, Scalar, Parameters, FitRes, FitIns, NDArrays, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
import pickle

class PrototypeStrategy(fl.server.strategy.FedAvg):
    """
    Custom Flower Strategy for Prototype FL.
    - Ignores model weights (no weight aggregation).
    - Aggregates class-wise prototypes collected from clients.
    """
    def __init__(self, num_classes: int = 10, security_manager=None, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.security_manager = security_manager
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
            # Return empty parameters to keep Flower protocol happy
            return ndarrays_to_parameters([]), {}

        # 1. Collect prototypes and counts from all clients
        client_protos_list = []
        client_counts_list = []
        client_snapshots = []
        for _, fit_res in results:
            # We will send prototypes and counts as serialized bytes in metrics
            protos, counts = self._unpack_prototypes(fit_res.metrics)
            client_protos_list.append(protos)
            client_counts_list.append(counts)
            
            client_snapshots.append({
                "cid": fit_res.metrics.get("cid", "?"),
                "protos": protos,
                "counts": counts
            })

        # 2. Aggregate prototypes using weighted average (Eq 6)
        self.global_prototypes = self._aggregate_protos(client_protos_list, client_counts_list)
        
        # Curious Server Hook: Log Snapshot and Run Attacks
        if self.security_manager:
            # Note: In Flower simulation, we don't have direct access to client models easily at server.
            # But the server knows the architecture and initialization.
            # For now we skip model-state here or provide a way to pass the architecture.
            # In real FL, the backbone would be the one being shared or known.
            self.security_manager.log_and_attack(server_round, None, client_snapshots)
        
        # 3. Save Results (Harmonized with Local Loop)
        import os
        import csv
        accs_global = [fit_res.metrics["acc_global"] for _, fit_res in results if "acc_global" in fit_res.metrics]
        accs_local_prop = [fit_res.metrics["acc_local_proportional"] for _, fit_res in results if "acc_local_proportional" in fit_res.metrics]
        accs_local = [fit_res.metrics["acc_local"] for _, fit_res in results if "acc_local" in fit_res.metrics]
        
        log_dir = "outputs/metrics"
        os.makedirs(log_dir, exist_ok=True)
        csv_path = os.path.join(log_dir, "simulation_results.csv")
        
        avg_g = float(np.mean(accs_global)) if accs_global else 0.0
        avg_lp = float(np.mean(accs_local_prop)) if accs_local_prop else 0.0
        avg_l = float(np.mean(accs_local)) if accs_local else 0.0
        
        file_exists = os.path.isfile(csv_path)
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["round", "avg_global", "avg_local_proportional", "avg_local"])
            writer.writerow([server_round, avg_g, avg_lp, avg_l])

        print(f"| Round {server_round} | Avg Accs -> Global: {avg_g:.4f} | Local-Prop: {avg_lp:.4f} | Local: {avg_l:.4f}")

        # Return metrics for the Flower history object
        return ndarrays_to_parameters([]), {"avg_accuracy": avg_l}

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: fl.server.client_manager.ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Send the current global prototypes to clients via the config."""
        # In prototype-only FL, we may have empty parameters.
        if parameters is None:
            parameters = ndarrays_to_parameters([])
        config = self._pack_prototypes(self.global_prototypes)
        fit_ins = FitIns(parameters, config)
        clients = client_manager.sample(num_clients=self.min_fit_clients, min_num_clients=self.min_available_clients)
        return [(client, fit_ins) for client in clients]

    def _aggregate_protos(self, client_protos_list: List[Dict[int, np.ndarray]], client_counts_list: List[Dict[int, int]]) -> Dict[int, np.ndarray]:
        """Weighted average of class prototypes (Eq 6)."""
        global_protos = {}
        for c in range(self.num_classes):
            c_protos = []
            c_counts = []
            for protos, counts in zip(client_protos_list, client_counts_list):
                if c in protos and c in counts:
                    c_protos.append(protos[c])
                    c_counts.append(counts[c])
            
            if c_protos:
                total_count = sum(c_counts)
                weighted_sum = sum(p * cnt for p, cnt in zip(c_protos, c_counts))
                global_protos[c] = (weighted_sum / total_count).astype(np.float32)
        return global_protos

    def _pack_prototypes(self, protos: Dict[int, np.ndarray]) -> Dict[str, Scalar]:
        """Convert prototypes dict to bytes for Flower transmission."""
        if not protos:
            return {}
        return {"protos_bytes": pickle.dumps(protos)}

    def _unpack_prototypes(self, metrics: Dict[str, Scalar]) -> Tuple[Dict[int, np.ndarray], Dict[int, int]]:
        """Reconstruct prototypes and counts from Flower bytes."""
        protos = pickle.loads(metrics["protos_bytes"]) if "protos_bytes" in metrics else {}
        counts = pickle.loads(metrics["counts_bytes"]) if "counts_bytes" in metrics else {}
        return protos, counts
