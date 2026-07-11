from qiskit import QuantumCircuit


def build_zz_feature_map(features, reps=1):
    """
    Build a ZZ feature map with data re-uploading.

    The circuit applies Hadamard gates on the first repetition, then
    alternates angle encoding (Rz) and entanglement (CNOT + Rz(ZZ))
    for ``reps`` repetitions.

    Parameters
    ----------
    features : list or array of shape (2,)
        The two classical feature values to encode.
    reps : int, default=1
        Number of times to repeat the encoding + entanglement block.
        Higher values create more expressive quantum states.
    """
    n_qubits = 2
    qc = QuantumCircuit(n_qubits)

    for rep in range(reps):
        # First repetition: initial superposition
        if rep == 0:
            qc.h(0)
            qc.h(1)

        # ── Angle encoding (data re-uploading) ──
        qc.rz(features[0], 0)
        qc.rz(features[1], 1)

        # ── Entanglement block (ZZ interaction) ──
        qc.cx(0, 1)
        qc.rz(features[0] * features[1], 1)
        qc.cx(0, 1)

    return qc

