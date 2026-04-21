import numpy as np
import flwr as fl
from typing import List, Dict, Optional, Tuple, Union
from flwr.common import Metrics, Scalar, Parameters, FitRes, FitIns, NDArrays, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
import pickle
import os

class PrototypeStrategy(fl.server.strategy.FedAvg):
    """
    Custom Flower Strategy for Prototype FL.
    - Ignores model weights (no weight aggregation).
    - Aggregates class-wise prototypes collected from clients.
    """
    def __init__(self, num_classes: int = 10, run_id: str = "", **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.global_prototypes: Dict[int, np.ndarray] = {}
        self.run_id = run_id

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
        for _, fit_res in results:
            # We will send prototypes and counts as serialized bytes in metrics
            protos, counts = self._unpack_prototypes(fit_res.metrics)
            client_protos_list.append(protos)
            client_counts_list.append(counts)

        # Log client uploads
        round_dir = os.path.abspath(os.path.join("runs", self.run_id, f"round_{server_round}"))
        os.makedirs(round_dir, exist_ok=True)
        os.makedirs(os.path.join(round_dir, "clients"), exist_ok=True)
        for i, (_, fit_res) in enumerate(results):
            protos, counts = self._unpack_prototypes(fit_res.metrics)
            cid = fit_res.metrics.get("cid", i)
            sent_classes = list(protos.keys())
            upload = {
                "client_id": cid,
                "round": server_round,
                "phase": "pre_upload",
                "sent_artifact_type": "classwise_proto",
                "sent_classes": sent_classes,
                "prototype_dict": protos,
                "class_counts": counts,
                "proto_dim": protos[sent_classes[0]].shape[0] if sent_classes else 0,
                "n_sent_classes": len(sent_classes)
            }
            with open(os.path.join(round_dir, "clients", f"client_{cid}_upload.pkl"), "wb") as f:
                pickle.dump(upload, f)

        # 2. Aggregate prototypes using weighted average (Eq 6)
        self.global_prototypes = self._aggregate_protos(client_protos_list, client_counts_list)
        
        # Log server artifact
        server_artifact = {
            "global_prototypes": self.global_prototypes,
            "server_reply": self.global_prototypes,
            "global_metrics": {"avg_accuracy": avg_l}
        }
        with open(os.path.join(round_dir, "server_artifact.pkl"), "wb") as f:
            pickle.dump(server_artifact, f)
        
        # 3. Report detailed accuracy across clients
        accs_global = []
        accs_local_prop = []
        accs_local = []
        
        # Prepare output text
        log_dir = "outputs/metrics"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "flower_results.txt")
        
        log_lines = []
        log_lines.append(f"\n[Round {server_round}] Server Aggregated {len(results)} clients.")
        
        for _, fit_res in results:
            metrics = fit_res.metrics
            cid = metrics.get("cid", "?")
            msg = f"  - Client {cid}: "
            
            if "acc_global" in metrics:
                accs_global.append(metrics["acc_global"])
                msg += f"Global: {metrics['acc_global']:.4f} "
            if "acc_local_proportional" in metrics:
                accs_local_prop.append(metrics["acc_local_proportional"])
                msg += f"Local-Prop: {metrics['acc_local_proportional']:.4f} "
            if "acc_local" in metrics:
                accs_local.append(metrics["acc_local"])
                msg += f"Local: {metrics['acc_local']:.4f}"
            log_lines.append(msg)
            
        avg_g = sum(accs_global) / len(accs_global) if accs_global else 0.0
        avg_lp = sum(accs_local_prop) / len(accs_local_prop) if accs_local_prop else 0.0
        avg_l = sum(accs_local) / len(accs_local) if accs_local else 0.0
        
        log_lines.append(f"| Avg Accs -> Global: {avg_g:.4f} | Local-Prop: {avg_lp:.4f} | Local: {avg_l:.4f}")

        log_str = "\n".join(log_lines)
        print(log_str) # Keep it in terminal in case user looks
        with open(log_file, "a") as f:
            f.write(log_str + "\n")

        # Return empty parameters because we don't aggregate weights
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
