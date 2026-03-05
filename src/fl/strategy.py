from __future__ import annotations
import flwr as fl


class NoAggStrategy(fl.server.strategy.FedAvg):
    """
    One-shot orchestration:
    - Clients train locally and return weights (to satisfy Flower)
    - Server ignores aggregation and keeps initial global weights
    - Prototypes are saved by clients to disk (outputs/client_protos)
    """

    def __init__(self, initial_parameters: fl.common.Parameters, **kwargs):
        super().__init__(initial_parameters=initial_parameters, **kwargs)

    def aggregate_fit(self, server_round, results, failures):
        # Log metrics from clients
        for client_proxy, fit_res in results:
            cid = client_proxy.cid
            metrics = fit_res.metrics or {}
            print(f"[Server] got fit from client={cid}, metrics={metrics}")

        # NO aggregation: return initial params unchanged
        return self.initial_parameters, {}