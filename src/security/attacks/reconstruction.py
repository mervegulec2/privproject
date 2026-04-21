import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List
import numpy as np
from torchvision.utils import save_image

from src.security.base import BaseAttack
from src.security.metrics import get_reconstruction_fidelity
from src.models import ResNet18Cifar

def _replace_activation(module: nn.Module):
    """
    From DLG Paper Section 4.1:
    'Two changes we have made to the models are replacing activation ReLU to Sigmoid 
     and removing strides, as our algorithm requires the model to be twice-differentiable.'
    We dynamically replace ReLU with Sigmoid for the server's local attack backbone.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.ReLU):
            setattr(module, name, nn.Sigmoid())
        else:
            _replace_activation(child)

class PrototypeReconstructionAttack(BaseAttack):
    """
    Implements the Deep Leakage from Gradients (DLG) logic (Section 3 of Zhu et al.),
    adapted for Prototype matching instead of gradient matching.
    
    Adversary Knowledge:
    - Target: Class-wise prototype embeddings
    - System: Global model weights, architecture, embedding dimension.
    """
    
    def __init__(self, 
                 image_shape=(1, 3, 32, 32), # Batch size 1, 3 channels, 32x32 for CIFAR
                 iterations=500, 
                 lr=1.0, 
                 save_dir="outputs/security/reconstructions",
                 tv_lambda=1e-3,
                 l2_lambda=1e-4):
        self.image_shape = image_shape
        self.iterations = iterations
        self.lr = lr
        self.save_dir = save_dir
        self.tv_lambda = tv_lambda
        self.l2_lambda = l2_lambda
        
        os.makedirs(self.save_dir, exist_ok=True)

    def total_variation_loss(self, img: torch.Tensor) -> torch.Tensor:
        """Standard Total Variation regularization to denoise generated images."""
        if len(img.shape) != 4:
            raise ValueError("Expected 4D tensor (B, C, H, W)")
        tv_h = torch.sum(torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]))
        tv_w = torch.sum(torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]))
        return tv_h + tv_w

    def execute(self, model_state: Dict[str, Any], shared_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the reconstruction attack on logged snapshots.
        """
        if "clients" not in shared_data:
            return {"status": "skipped", "reason": "Missing client data"}
            
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 1. Setup Server's Copy of the Model (System Knowledge)
        model = ResNet18Cifar(num_classes=10)
        
        if model_state is not None:
            # print("[Attack] Using specific model_state from snapshot.")
            model.load_state_dict(model_state)
        else:
            print("[Attack] Notice: No model_state in snapshot. Proceeding with Global Model Initialization.")

        # Apply DLG paper modifications (twice-differentiable requirement)
        _replace_activation(model)
        model.to(device)
        model.eval() 
        
        results = {}
        
        # 2. Iterate over clients and their observed prototypes (Observed Knowledge)
        for client_info in shared_data["clients"]:
            cid = client_info["cid"]
            protos = client_info["protos"]
            counts = client_info["counts"]
            
            client_results = {}
            for target_class, target_proto_np in protos.items():
                print(f"[Attack] Reconstructing Client {cid}, Class {target_class}...")
                
                target_proto = torch.tensor(target_proto_np, device=device, dtype=torch.float32)
                
                # Algorithm 1: Line 2 - Initialize dummy inputs from N(0, 1)
                dummy_data = torch.randn(self.image_shape, device=device).requires_grad_(True)
                
                # DLG Section 4: We use L-BFGS with learning rate 1
                optimizer = torch.optim.LBFGS([dummy_data], lr=self.lr)
                
                history = []
                
                def closure():
                    optimizer.zero_grad()
                    # Forward pass of dummy data
                    # In our system knowledge, model returns (logits, embeddings)
                    _, dummy_emb = model(dummy_data)
                    
                    # Our target is the embedding prototype, not the full model gradients
                    # Algorithm 1 Line 5: Distance D = || dummy - real ||^2
                    # Note: DLG matches \nabla W. Since we share features directly, we match features.
                    # This is feature inversion, geometrically identical to gradient matching on the last layer.
                    distance_loss = F.mse_loss(dummy_emb.mean(dim=0), target_proto)
                    
                    # Add standard regularizers for image recovery
                    tv_loss = self.tv_lambda * self.total_variation_loss(dummy_data)
                    l2_loss = self.l2_lambda * torch.norm(dummy_data, p=2)
                    
                    total_loss = distance_loss + tv_loss + l2_loss
                    total_loss.backward()
                    return total_loss
                
                # Algorithm 1: Line 3-7 (Optimization Loop)
                for it in range(self.iterations):
                    optimizer.step(closure)
                    
                    if it % 50 == 0:
                        current_loss = closure().item()
                        history.append(current_loss)
                
                # Save the reconstructed image
                recovered_img = dummy_data.clone().detach().cpu()
                # Normalize typical output to [0, 1] for viewing/saving
                recovered_img = (recovered_img - recovered_img.min()) / (recovered_img.max() - recovered_img.min() + 1e-8)
                
                save_path = os.path.join(self.save_dir, f"recon_c{cid}_class{target_class}.png")
                save_image(recovered_img, save_path)
                
                # We don't have the original image *here* to compute perfect PSNR/SSIM. 
                # In a real evaluation script, we would load the auxiliary/ground truth dataset.
                # For now, we report the final optimization distance distance.
                client_results[target_class] = {
                    "final_loss": history[-1] if history else 0.0,
                    "save_path": save_path
                }
            
            results[f"client_{cid}"] = client_results
            
        return results
