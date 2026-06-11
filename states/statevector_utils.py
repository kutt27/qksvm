from qiskit.quantum_info import Statevector

def get_statevector(circuit):
    return Statevector.from_instruction(circuit)
