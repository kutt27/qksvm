from encoding.angle_encoding import angle_encode
from states.statevector_utils import get_statevector

x = [0.3, 1.2]

circuit = angle_encode(x)
state = get_statevector(circuit)
print(state)
