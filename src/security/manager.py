import os
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional

class SecurityManager:
    """
    Central orchestrator for all privacy/security modules in the PFL pipeline.
    Connects attacks and defenses to the main training loop.
    """
    def __init__(self, active_defenses: List[Any], active_attacks: List[Any], log_model_state: bool = False):
        self.defenses = active_defenses
        self.attacks = active_attacks
        self.log_model_state = log_model_state
        self.snapshot_dir = "outputs/security/snapshots"
        os.makedirs(self.snapshot_dir, exist_ok=True)

    def apply_defenses(self, prototypes: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        """Applies all registered privacy-preserving defenses (e.g., DP, Perturbation)."""
        perturbed_protos = prototypes
        for defense in self.defenses:
            perturbed_protos = defense.apply(perturbed_protos)
        return perturbed_protos

    def log_and_attack(self, round_idx: int, client_data: List[Dict[str, Any]], model: Optional[nn.Module] = None):
        """
        1. Captures a snapshot of the current global state and client transmissions.
        2. Optionally runs active attacks (Honest-but-Curious server perspective).
        """
        # Prepare snapshot
        snapshot = {
            "round": round_idx,
            "clients": client_data, # List of {cid, protos, counts}
            "model_state": model.state_dict() if (model and self.log_model_state) else None
        }

        # Save snapshot for post-hoc analysis (The 'Scientific Snapshot')
        snapshot_path = os.path.join(self.snapshot_dir, f"round_{round_idx}.pkl")
        import pickle
        with open(snapshot_path, "wb") as f:
            pickle.dump(snapshot, f)
        # print(f"[SecurityManager] Saved snapshot for round {round_idx} to {snapshot_path}")

        # Run Live Attacks
        attack_results = {}
        for attack in self.attacks:
            attack_name = attack.__class__.__name__
            # print(f"[SecurityManager] Executing {attack_name}...")
            results = attack.execute(snapshot["model_state"], {"clients": client_data, "log_model_state": self.log_model_state})
            attack_results[attack_name] = results
        
        return attack_results

def security_factory(config: Dict[str, Any]) -> SecurityManager:
    """
    Factory method to initialize the SecurityManager based on environment or config.
    """
    defenses = []
    # if config.get("defense_type") == "gaussian":
    #     from src.security.defenses.gaussian import GaussianNoiseDefense
    #     defenses.append(GaussianNoiseDefense(sigma=config.get("sigma", 0.01)))

    attacks = []
    active_attack_names = config.get("attacks", [])
    if not active_attack_names:
        env_attacks = os.environ.get("SECURITY_ATTACKS", "").strip()
        if env_attacks:
            active_attack_names = [a.strip() for a in env_attacks.split(",") if a.strip()]

    # Architecture Logic for Attacks
    if "reconstruction" in active_attack_names:
        from src.security.attacks.reconstruction import PrototypeReconstructionAttack
        attacks.append(PrototypeReconstructionAttack(iterations=500, lr=1.0))

    if "cpa" in active_attack_names:
        from src.security.attacks.trivial_cpa import TrivialClassPresenceAttack
        attacks.append(TrivialClassPresenceAttack(num_classes=int(config.get("num_classes", 10))))

    if "mia" in active_attack_names:
        from src.security.attacks.membership import ScientificMIAAuditor
        attacks.append(ScientificMIAAuditor(num_classes=int(config.get("num_classes", 10))))
    
    # Check for model state logging flag
    log_model_state = config.get("log_model_state", False)
    if os.environ.get("SECURITY_LOG_MODEL_STATE") == "1":
        log_model_state = True

    return SecurityManager(
        active_defenses=defenses,
        active_attacks=attacks,
        log_model_state=log_model_state
    )
