import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List
import numpy as np
from torchvision.utils import save_image

from src.security.base import BaseAttack
from src.models import ResNet18Cifar

class PrototypeReconstructionAttack(BaseAttack):
    """
    Scientific Prototype Reconstruction Attack (Feature Inversion).
    Target: Reconstruct pixel-level data from shared class prototypes.
    """
    
    def __init__(self, 
                 image_shape=(1, 3, 32, 32), 
                 iterations=500, 
                 lr=0.1, # Reduced LR for better stability
                 save_dir="outputs/security/reconstructions",
                 tv_lambda=1e-2):
        self.image_shape = image_shape
        self.iterations = iterations
        self.lr = lr
        self.save_dir = save_dir
        self.tv_lambda = tv_lambda
        os.makedirs(self.save_dir, exist_ok=True)

    def total_variation_loss(self, img: torch.Tensor) -> torch.Tensor:
        tv_h = torch.sum(torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]))
        tv_w = torch.sum(torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]))
        return tv_h + tv_w

    def execute(self, model_state: Dict[str, Any], shared_data: Dict[str, Any]) -> Dict[str, Any]:
        if "clients" not in shared_data:
            return {"status": "skipped", "reason": "Missing client data"}
            
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = ResNet18Cifar(num_classes=10)
        
        if model_state is not None:
            model.load_state_dict(model_state)
        
        model.to(device)
        model.eval() 
        
        results = {}
        for client_info in shared_data["clients"]:
            cid = client_info["cid"]
            protos = client_info["protos"]
            client_results = {}
            
            for target_class, target_proto_np in protos.items():
                print(f"[Attack] Reconstructing Client {cid}, Class {target_class}...")
                target_proto = torch.tensor(target_proto_np, device=device, dtype=torch.float32)
                
                # Initialize dummy data (0.5 start is often better than random for CIFAR)
                dummy_data = torch.full(self.image_shape, 0.5, device=device).requires_grad_(True)
                optimizer = torch.optim.LBFGS([dummy_data], lr=self.lr)
                
                for it in range(self.iterations):
                    def closure():
                        optimizer.zero_grad()
                        _, dummy_emb = model(dummy_data)
                        
                        # Distance matching in embedding space
                        dist_loss = F.mse_loss(dummy_emb.mean(dim=0), target_proto)
                        tv_loss = self.tv_lambda * self.total_variation_loss(dummy_data)
                        
                        total_loss = dist_loss + tv_loss
                        total_loss.backward()
                        return total_loss
                    
                    optimizer.step(closure)
                
                # Final post-processing
                with torch.no_grad():
                    recovered_img = dummy_data.clone().detach().cpu()
                    recovered_img = torch.clamp(recovered_img, 0, 1)
                
                save_path = os.path.join(self.save_dir, f"recon_c{cid}_class{target_class}.png")
                save_image(recovered_img, save_path)
                
                # --- NEW: Calculate Fidelity Metrics (Auditor perspective) ---
                from src.security.metrics import get_reconstruction_fidelity
                from src.data_utils import load_cifar10, Cifar10Config, load_split
                
                # Get ground truth mean image for this client/class
                train_ds, _ = load_cifar10(Cifar10Config(root="data"))
                splits = load_split("outputs/data/client_splits.json")
                fid_metrics = {"psnr": 0.0, "ssim": 0.0}
                
                if splits and str(cid) in splits:
                    indices = splits[str(cid)]
                    # Find samples of this class
                    class_indices = [i for i in indices if train_ds.targets[i] == int(target_class)]
                    if class_indices:
                        gt_imgs = torch.stack([train_ds[i][0] for i in class_indices])
                        gt_mean = gt_imgs.mean(dim=0)
                        fid_metrics = get_reconstruction_fidelity(gt_mean.numpy(), recovered_img.squeeze(0).numpy())

                client_results[int(target_class)] = {
                    "save_path": save_path,
                    "psnr": fid_metrics["psnr"],
                    "ssim": fid_metrics["ssim"],
                    "cosine_sim": fid_metrics.get("cosine_sim", 0.0)
                }
            
            results[f"client_{cid}"] = client_results
            
        return results
