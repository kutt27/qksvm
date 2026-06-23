import numpy as np
from qiskit.quantum_info import Statevector


def kernel_value(feature_map_fn, x1, x2):
    state1 = Statevector.from_instruction(
        feature_map_fn(x1)
    )
    state2 = Statevector.from_instruction(
        feature_map_fn(x2)
    )

    overlap = np.vdot(
        state1.data,
        state2.data
    )
    return abs(overlap) ** 2


def kernel_matrix(feature_map_fn, X):
    n = len(X)
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            K[i, j] = kernel_value(
                feature_map_fn,
                X[i],
                X[j]
            )
    return K
