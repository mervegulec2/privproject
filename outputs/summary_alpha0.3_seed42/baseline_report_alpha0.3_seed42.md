# Baseline Report (alpha=0.3, seed=42)

## Setup
- Dataset: CIFAR-10 (10 classes)
- #Clients: 15
- Non-IID split: Dirichlet alpha=0.3, seed=42
- Backbone: ResNet-18 (CIFAR)
- Shared artifact: class-wise prototypes (512-d, GAP before final FC)

## Utility (Local Test Accuracy)
Client accuracies (alpha=0.3, seed=42):
- client  0: 0.2495
- client  1: 0.2846
- client  2: 0.1653
- client  3: 0.3296
- client  4: 0.2755
- client  5: 0.2340
- client  6: 0.1995
- client  7: 0.1812
- client  8: 0.2179
- client  9: 0.2468
- client 10: 0.2806
- client 11: 0.3309
- client 12: 0.3892
- client 13: 0.2336
- client 14: 0.2921

**Mean:** 0.2607  |  **Std:** 0.0604  |  **Min:** 0.1653  |  **Max:** 0.3892

## Class Presence Summary
```text
Num classes per client: dirichlet = 0.3
  client  0: 10
  client  1: 9
  client  2: 9
  client  3: 9
  client  4: 10
  client  5: 8
  client  6: 8
  client  7: 8
  client  8: 10
  client  9: 9
  client 10: 8
  client 11: 9
  client 12: 10
  client 13: 10
  client 14: 10

Per-class presence across clients (#clients that sent that class):
  class 0: 12/15
  class 1: 14/15
  class 2: 15/15
  class 3: 12/15
  class 4: 14/15
  class 5: 14/15
  class 6: 14/15
  class 7: 15/15
  class 8: 13/15
  class 9: 14/15
```

## Presence Matrix (Client × Class)
```text
Presence matrix (1=sent proto, 0=not sent)
cid |  0  1  2  3  4  5  6  7  8  9 | #classes
----------------------------------------------
  0 | 1  1  1  1  1  1  1  1  1  1 | 10
  1 | 1  1  1  0  1  1  1  1  1  1 | 9
  2 | 1  1  1  1  1  1  1  1  0  1 | 9
  3 | 0  1  1  1  1  1  1  1  1  1 | 9
  4 | 1  1  1  1  1  1  1  1  1  1 | 10
  5 | 1  1  1  0  1  0  1  1  1  1 | 8
  6 | 0  1  1  1  1  1  1  1  1  0 | 8
  7 | 1  1  1  1  0  1  0  1  1  1 | 8
  8 | 1  1  1  1  1  1  1  1  1  1 | 10
  9 | 0  1  1  1  1  1  1  1  1  1 | 9
 10 | 1  1  1  0  1  1  1  1  0  1 | 8
 11 | 1  0  1  1  1  1  1  1  1  1 | 9
 12 | 1  1  1  1  1  1  1  1  1  1 | 10
 13 | 1  1  1  1  1  1  1  1  1  1 | 10
 14 | 1  1  1  1  1  1  1  1  1  1 | 10
```

## Global Prototype File Info
```text
global_prototypes_path: outputs/global_prototypes.npz
num_classes: 10
classes: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
one_proto_shape: (512,)
dtype: float32
```

## Client → Global Cosine Similarity (Heterogeneity)
```text
cid | mean_cos_to_global | min | max | #classes
 8  | 0.8430           | 0.7243 | 0.9684 | 10
13  | 0.8824           | 0.8389 | 0.9739 | 10
 6  | 0.9224           | 0.8195 | 0.9665 | 8
 0  | 0.9394           | 0.9038 | 0.9849 | 10
 5  | 0.9442           | 0.8896 | 0.9690 | 8
 9  | 0.9538           | 0.8651 | 0.9877 | 9
 1  | 0.9601           | 0.9443 | 0.9733 | 9
 4  | 0.9607           | 0.9324 | 0.9788 | 10
 7  | 0.9629           | 0.9377 | 0.9770 | 8
 2  | 0.9633           | 0.9216 | 0.9784 | 9
14  | 0.9641           | 0.9411 | 0.9847 | 10
12  | 0.9673           | 0.9535 | 0.9733 | 10
11  | 0.9740           | 0.9255 | 0.9860 | 9
10  | 0.9742           | 0.9681 | 0.9825 | 8
 3  | 0.9798           | 0.9537 | 0.9906 | 9
```

