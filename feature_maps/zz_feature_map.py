from qiskit import QuantumCircuit

def build_zz_feature_map(features):
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.h(1)
    qc.rz(features[0], 0)
    qc.rz(features[1], 1)
    qc.cx(0, 1)
    qc.rz(features[0] * features[1], 1)
    qc.cx(0, 1)
    return qc

