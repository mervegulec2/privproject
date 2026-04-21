# Implementation Plan: Attacking and Defending Prototype-based PFL

This plan outlines the steps to evaluate the security of the current personalized Federated Learning (PFL) pipeline, where clients share class-wise prototypes with an honest-but-curious server.

## 1. Vulnerability Analysis
In the current setup (`FedProto`), clients send the mean of feature embeddings for each class:
$$\mu_{c,i} = \frac{1}{|D_{c,i}|} \sum_{x \in D_{c,i}} f_\theta(x)$$
An honest-but-curious server can:
- **Reconstruct** samples by inverting the mapping $f_\theta$.
- **Infer membership** by checking proximity to $\mu_{c,i}$.
- **Identify properties** of the local dataset via the distribution of prototypes.

## 2. Attack Implementation
We will implement a set of attacks in a new directory `src/security/attacks/`.

### A. Feature Inversion (Reconstruction)
- **Method**: Optimization-based. Given a prototype $P$, we solve:
  $$\min_x \|f_\theta(x) - P\|^2_2 + \lambda_{TV} \mathcal{R}_{TV}(x) + \lambda_{l2} \|x\|^2_2$$
- **Metric**: PSNR (Peak Signal-to-Noise Ratio) and SSIM (Structural Similarity Index) between reconstructed and original (average) images.

### B. Membership Inference Attack (MIA)
- **Method**: Distance-based. If $\|f_\theta(x) - \mu_{c,i}\| < \tau$, we predict $x \in D_{c,i}$.
- **Metric**: Precision, Recall, and AUC-ROC.

## 3. Defense Implementation
We will implement defenses in `src/security/defenses/`.

### A. Local Differential Privacy (LDP)
- **Method**: Add Gaussian noise to prototypes before transmission.
- **Sensitivity**: Estimated based on the embedding space radius or normalized embeddings.
- **Parameters**: $\epsilon, \delta$ privacy budget.

### B. Prototype Normalization/Clipping
- **Method**: Clip embedding norms before averaging to bound sensitivity for LDP.

## 4. Evaluation Workflow

### Phase 1: Attack without Defense
1. Run `run_pfl.py` and modify it briefly to export client prototypes and a few sample images per class for ground truth.
2. Run `attack_reconstruction.py` using the client's local backbone (assuming the server knows the architecture).
3. Run `attack_mia.py`.
4. Document results (Reconstructed images, MIA accuracy).

### Phase 2: Apply Defense
1. Add a `--dp_sigma` argument to `run_pfl.py`.
2. Update `FlowerPrototypeClient.fit` to add noise to `local_protos`:
   ```python
   if dp_sigma > 0:
       for c in local_protos:
           local_protos[c] += np.random.normal(0, dp_sigma, local_protos[c].shape)
   ```
3. Run simulation with various $\sigma$ values.

### Phase 3: Attack with Defense
1. Run the same attacks on the noisy prototypes.
2. Plot the **Privacy-Utility Trade-off**: Accuracy vs. Attack Success (e.g., PSNR or MIA AUC).

## 5. Proposed File Structure
- `src/security/`
    - `attacks/`
        - `reconstruction.py`: Optimization loop for inversion.
        - `membership_inference.py`: Score-based MIA.
    - `defenses/`
        - `privacy_utils.py`: Noise addition and clipping logic.
    - `eval_security.py`: Orchestrator for the security pipeline.
