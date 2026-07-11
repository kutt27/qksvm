import numpy as np
from qiskit.quantum_info import Statevector


def kernel_value(feature_map_fn, x1, x2):
    """Compute the quantum kernel value K(x1, x2) = |<φ(x1)|φ(x2)>|²."""
    state1 = Statevector.from_instruction(feature_map_fn(x1))
    state2 = Statevector.from_instruction(feature_map_fn(x2))
    overlap = np.vdot(state1.data, state2.data)
    return abs(overlap) ** 2


def kernel_matrix(feature_map_fn, X1, X2=None):
    """
    Compute the quantum kernel matrix.

    Parameters
    ----------
    feature_map_fn : callable
        Function that maps feature values to a QuantumCircuit.
    X1 : ndarray of shape (n1, n_features)
        First dataset (training set).
    X2 : ndarray of shape (n2, n_features) or None
        Second dataset (test set).  If ``None``, the symmetric N×N
        kernel of ``X1`` is returned.

    Returns
    -------
    K : ndarray of shape (n1, n1) if X2 is None, else (n2, n1)
    """
    X1 = np.asarray(X1)

    if X2 is None:
        # Symmetric N×N kernel
        n = X1.shape[0]
        K = np.zeros((n, n))
        # Only compute upper triangle and copy to lower (symmetric)
        for i in range(n):
            for j in range(i, n):
                val = kernel_value(feature_map_fn, X1[i], X1[j])
                K[i, j] = val
                K[j, i] = val
        return K
    else:
        # Asymmetric M×N kernel — rows = X2 (test), cols = X1 (train)
        X2 = np.asarray(X2)
        n1 = X1.shape[0]
        n2 = X2.shape[0]
        K = np.zeros((n2, n1))
        for i in range(n2):
            for j in range(n1):
                K[i, j] = kernel_value(feature_map_fn, X2[i], X1[j])
        return K
