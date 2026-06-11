from qiskit import QuantumCircuit

def angle_encode(features):
    """
    Encode classical features into qubit rotations.
    Example:
        [0.3, 1.2]
            ↓
        q0 -> RY(0.3)
        q1 -> RY(1.2)
    """
    n_qubits = len(features)
    circuit = QuantumCircuit(n_qubits)
    for qubit, value in enumerate(features):
        circuit.ry(value, qubit)
    return circuit 
