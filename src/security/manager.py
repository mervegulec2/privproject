import os
import pickle
import torch
import numpy as np
from typing import Dict, List, Any, Optional
from .base import BaseAttack, BaseDefense

class SecurityManager:
    """
    Orchestrates attacks and defenses in a pluggable way.
    Adheres to the Honest-but-Curious server threat model.
    """
    def __init__(
        self, 
        active_defenses: List[BaseDefense] = None,
        active_attacks: List[BaseAttack] = None,
        snapshot_dir: str = "outputs/security/snapshots",
        enable_logging: bool = True,
        log_model_state: bool = False,
    ):
        self.defenses = active_defenses or []
        self.attacks = active_attacks or []
        self.snapshot_dir = snapshot_dir
        self.enable_logging = enable_logging
        # Strict honest-but-curious default: server should NOT log or use any client-local model weights.
        # Keep this off unless you explicitly opt into a stronger proxy setting.
        self.log_model_state = bool(log_model_state)
        
        if self.enable_logging:
            os.makedirs(self.snapshot_dir, exist_ok=True)

    def apply_defenses(self, prototypes: Dict[int, np.ndarray], counts: Dict[int, int]) -> Dict[int, np.ndarray]:
        """Client-side: Apply all active defenses to the prototypes."""
        processed_protos = prototypes
        for defense in self.defenses:
            processed_protos = defense.apply(processed_protos, counts)
        return processed_protos

    def log_and_attack(self, round_idx: int, model: torch.nn.Module, client_data: List[Dict[str, Any]]):
        """
        Server-side (Curious Behavior): 
        1. Save snapshots of observed client prototypes and the model.
        2. Execute active attacks on the observed data.
        """
        if not self.enable_logging and not self.attacks:
            return

        # Adversary Knowledge Snapshot
        model_state = None
        if self.log_model_state and model is not None:
             model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        snapshot = {
            "round": round_idx,
            "model_state": model_state, # System Knowledge
            "clients": client_data # Observed from Clients (protos, counts)
        }

        if self.enable_logging:
            path = os.path.join(self.snapshot_dir, f"round_{round_idx}.pkl")
            with open(path, "wb") as f:
                pickle.dump(snapshot, f)
            # print(f"[SecurityManager] Snapshot saved for round {round_idx} at {path}")

        # Execute Attacks
        results = {}
        for attack in self.attacks:
            attack_name = attack.__class__.__name__
            # print(f"[SecurityManager] Executing attack: {attack_name}")
            results[attack_name] = attack.execute(
                snapshot["model_state"],
                {"clients": client_data, "log_model_state": self.log_model_state},
            )
        
        return results

def create_security_manager(config: Dict[str, Any]) -> SecurityManager:
    """Factory to create manager from a config dictionary."""
    defenses = []
    attacks = []
    
    # Placeholder for defenses
    # if "dp" in config.get("defenses", []):
    #     defenses.append(DifferentialPrivacyDefense(...))
    
    # Load Attacks
    active_attack_names = config.get("attacks", [])
    # Optional: allow enabling attacks without touching PFL pipeline code.
    # If config doesn't specify attacks, read comma-separated env var.
    if not active_attack_names:
        env_attacks = os.environ.get("SECURITY_ATTACKS", "").strip()
        if env_attacks:
            active_attack_names = [a.strip() for a in env_attacks.split(",") if a.strip()]

    if "reconstruction" in active_attack_names:
        from src.security.attacks.reconstruction import PrototypeReconstructionAttack
        attacks.append(PrototypeReconstructionAttack(
            iterations=500, # Can be moved to config later
            lr=1.0
        ))

    if "cpa" in active_attack_names:
        from src.security.attacks.class_inference.trivial_cpa import TrivialClassPresenceAttack
        attacks.append(TrivialClassPresenceAttack(num_classes=int(config.get("num_classes", 10))))

    if "mia_feasibility" in active_attack_names:
        from src.security.attacks.membership.mia_feasibility import MIAFeasibilityAttack
        attacks.append(MIAFeasibilityAttack())

    if "mia" in active_attack_names:
        from src.security.attacks.membership.mia_proto_scoring import PrototypeMIAAttack
        attacks.append(
            PrototypeMIAAttack(
                num_classes=int(config.get("num_classes", 10)),
                max_aux_samples=int(os.environ.get("MIA_MAX_AUX", "512")),
                scorer=os.environ.get("MIA_SCORER", "cosine"),
            )
        )
    
    return SecurityManager(
        active_defenses=defenses,
        active_attacks=attacks,
        enable_logging=config.get("enable_logging", True),
        snapshot_dir=config.get("snapshot_dir", "outputs/security/snapshots"),
        log_model_state=bool(config.get("log_model_state", os.environ.get("SECURITY_LOG_MODEL_STATE", "0") == "1")),
    )
