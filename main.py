from kernels.quantum_kernel import kernel_matrix
from feature_maps.zz_feature_map import build_zz_feature_map

X = [
    [0.3, 1.2],
    [0.4, 1.1],
    [2.7, 2.8]
]

K = kernel_matrix(
    build_zz_feature_map,
    X
)

print(K)
