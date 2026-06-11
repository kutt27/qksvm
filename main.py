from encoding.angle_encoding import angle_encode
from states.statevector_utils import get_statevector
from feature_maps.zz_feature_map import build_zz_feature_map

x = [0.3, 1.2]

circuit = build_zz_feature_map(x)
print(circuit.draw())
