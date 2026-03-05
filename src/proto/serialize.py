from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
import flwr as fl

def prototypes_to_parameters(protos: Dict[int, np.ndarray]) -> fl.common.Parameters:
    """
    Encode prototypes dict into Flower Parameters.
    We'll store:
    - an array of class_ids (int64)
    - a stacked matrix of prototypes (float32) shape [K,512]
    """
    class_ids = np.array(sorted(protos.keys()), dtype=np.int64)
    if len(class_ids) == 0:
        # no prototypes (edge case)
        mat = np.zeros((0, 512), dtype=np.float32)
    else:
        mat = np.stack([protos[int(c)] for c in class_ids], axis=0).astype(np.float32)

    tensors = [
        fl.common.ndarray_to_bytes(class_ids),
        fl.common.ndarray_to_bytes(mat),
    ]
    return fl.common.Parameters(tensors=tensors, tensor_type="numpy-bytes")

def parameters_to_prototypes(params: fl.common.Parameters) -> Dict[int, np.ndarray]:
    class_ids = fl.common.bytes_to_ndarray(params.tensors[0]).astype(np.int64)
    mat = fl.common.bytes_to_ndarray(params.tensors[1]).astype(np.float32)

    protos: Dict[int, np.ndarray] = {}
    for i, c in enumerate(class_ids):
        protos[int(c)] = mat[i]
    return protos