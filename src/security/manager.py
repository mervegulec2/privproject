import os
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional

class SecurityManager:
    """
    Central orchestrator for all privacy/security modules in the PFL pipeline.
    Connects attacks and defenses to the main training loop.
    """
    def __init__(self, active_defenses: List[Any], active_attacks: List[Any], log_model_state: bool = False, base_dir: str = "outputs"):
        self.defenses = active_defenses
        self.attacks = active_attacks
        self.log_model_state = log_model_state
        self.snapshot_dir = os.path.join(base_dir, "security/snapshots")
        os.makedirs(self.snapshot_dir, exist_ok=True)

    def apply_defenses(self, prototypes: Dict[int, torch.Tensor], counts: Dict[int, int] = None) -> Dict[int, torch.Tensor]:
        """Applies all registered privacy-preserving defenses and calculates distortion metrics."""
        from src.security.metrics import calculate_statistical_leakage
        import numpy as np
        
        perturbed_protos = prototypes
        self.defense_stats = {}
        
        for defense in self.defenses:
            original_protos = {c: p.clone() if hasattr(p, 'clone') else p.copy() for c, p in perturbed_protos.items()}
            perturbed_protos = defense.apply(perturbed_protos, counts=counts)
            
            # Calculate distortion for this defense layer
            kls, corrs = [], []
            for c in perturbed_protos:
                if c in original_protos:
                    # Convert to numpy for the metric function
                    o_np = original_protos[c].detach().cpu().numpy() if hasattr(original_protos[c], 'detach') else original_protos[c]
                    p_np = perturbed_protos[c].detach().cpu().numpy() if hasattr(perturbed_protos[c], 'detach') else perturbed_protos[c]
                    
                    stats = calculate_statistical_leakage(o_np, p_np)
                    kls.append(stats["kl_divergence"])
                    corrs.append(stats["correlation_leakage"])
            
            self.defense_stats[defense.__class__.__name__] = {
                "avg_kl": float(np.mean(kls)) if kls else 0.0,
                "avg_correlation": float(np.mean(corrs)) if corrs else 1.0
            }
            
        return perturbed_protos

    def log_and_attack(self, round_idx: int, client_data: List[Dict[str, Any]], model: Optional[nn.Module] = None):
        """
        1. Captures a snapshot of the current global state and client transmissions.
        2. Executes active attacks and prints a summary.
        """
        # Prepare snapshot
        snapshot = {
            "round": round_idx,
            "clients": client_data, # List of {cid, protos, counts}
            "model_state": model.state_dict() if (model and self.log_model_state) else None,
            "defense_stats": getattr(self, "defense_stats", {})
        }

        # Save snapshot
        snapshot_path = os.path.join(self.snapshot_dir, f"round_{round_idx}.pkl")
        import pickle
        with open(snapshot_path, "wb") as f:
            pickle.dump(snapshot, f)

        # Run Live Attacks
        attack_results = {}
        print(f"\n" + "="*60)
        print(f" SECURITY AUDIT | ROUND {round_idx}")
        print("="*60)
        
        if snapshot["defense_stats"]:
            print(f" [Defenses Active]")
            for d_name, d_stat in snapshot["defense_stats"].items():
                print(f"  - {d_name}: KL={d_stat['avg_kl']:.4f}, Corr={d_stat['avg_correlation']:.4f}")
        
        for attack in self.attacks:
            attack_name = attack.__class__.__name__
            results = attack.execute(snapshot["model_state"], {"clients": client_data, "log_model_state": self.log_model_state})
            attack_results[attack_name] = results
            
            # Print Summary for this attack
            print(f" [Attack: {attack_name}]")
            if "status" in results and results["status"] == "skipped":
                print(f"  - Status: Skipped ({results.get('reason', '')})")
            else:
                # Summarize client-level results
                avg_metrics = {}
                for cid_key, c_res in results.items():
                    if not isinstance(c_res, dict): continue
                    # Handle nested class-level results (like Reconstruction)
                    if any(isinstance(v, dict) for v in c_res.values()):
                        for cls_id, cls_res in c_res.items():
                            if not isinstance(cls_res, dict): continue
                            for m_name, m_val in cls_res.items():
                                if isinstance(m_val, (int, float)):
                                    avg_metrics[m_name] = avg_metrics.get(m_name, []) + [m_val]
                    else:
                        for m_name, m_val in c_res.items():
                            if isinstance(m_val, (int, float)):
                                avg_metrics[m_name] = avg_metrics.get(m_name, []) + [m_val]
                
                for m_name, m_vals in avg_metrics.items():
                    print(f"  - Avg {m_name}: {np.mean(m_vals):.4f}")
        
        print("="*60 + "\n")
        return attack_results

def security_factory(config: Dict[str, Any]) -> SecurityManager:
    """
    Factory method to initialize the SecurityManager based on environment or config.
    """
    defenses = []
    active_defense_names = config.get("defenses", [])
    if not active_defense_names:
        env_defenses = os.environ.get("SECURITY_DEFENSES", "").strip()
        if env_defenses:
            active_defense_names = [d.strip() for d in env_defenses.split(",") if d.strip()]

    if "gaussian_dp" in active_defense_names:
        from src.security.defenses.dp_gaussian import GaussianDPDefense
        clip_norm = float(os.environ.get("DP_CLIP_NORM", config.get("clip_norm", 1.0)))
        sigma = os.environ.get("DP_SIGMA")
        epsilon = os.environ.get("DP_EPSILON")
        delta = float(os.environ.get("DP_DELTA", 1e-5))
        sigma = float(sigma) if sigma else None
        epsilon = float(epsilon) if epsilon else None
        defenses.append(GaussianDPDefense(clip_norm=clip_norm, sigma=sigma, epsilon=epsilon, delta=delta))

    if "adaptive_gaussian_dp" in active_defense_names:
        from src.security.defenses.adaptive_gaussian import AdaptiveGaussianDPDefense
        clip_norm = float(os.environ.get("DP_CLIP_NORM", config.get("clip_norm", 1.0)))
        alpha = float(os.environ.get("ADAPTIVE_DP_ALPHA", 0.5))
        beta = float(os.environ.get("ADAPTIVE_DP_BETA", 1.0))
        sigma_max = float(os.environ.get("ADAPTIVE_DP_SIGMA_MAX", 1.0))
        defenses.append(AdaptiveGaussianDPDefense(
            clip_norm=clip_norm, alpha=alpha, beta=beta, sigma_max=sigma_max
        ))
    
    if "dummy_prototype" in active_defense_names:
        from src.security.defenses.dummy import DummyPrototypeDefense
        lambda_val = float(os.environ.get("DUMMY_LAMBDA", 0.7))
        tau = float(os.environ.get("DUMMY_TAU", 0.02))
        dummy_ratio = int(os.environ.get("DUMMY_RATIO", 1))
        defenses.append(DummyPrototypeDefense(
            lambda_val=lambda_val, tau=tau, dummy_ratio=dummy_ratio
        ))

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
        log_model_state=log_model_state,
        base_dir=config.get("base_dir", "outputs")
    )
