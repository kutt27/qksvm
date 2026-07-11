"""
QKSVM — Improved Quantum Kernel SVM Demo

Uses:
  • Data re-uploading (multiple feature map repetitions) for richer
    quantum states.
  • Grid search over C (regularization) to find the best classifier
    on a validation split (no test set leakage).
  • A held-out test set for final evaluation.
"""

import numpy as np
from functools import partial
from sklearn.svm import SVC
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

from classifier.qksvm import QKSVM
from kernels.quantum_kernel import kernel_matrix
from feature_maps.zz_feature_map import build_zz_feature_map

# ── 1. Load data ─────────────────────────────────────────────────────
X, y = make_moons(n_samples=80, noise=0.15, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

print(f"Train set size:  {len(X_train)}")
print(f"Test set size:   {len(X_test)}")
print(f"Features:        {X_train.shape[1]}")
print()

# ── 2. Split train further into sub-train + validation for tuning ────
X_sub_train, X_val, y_sub_train, y_val = train_test_split(
    X_train, y_train, test_size=0.25, random_state=42
)

print(f"Sub-train: {len(X_sub_train)}   Validation: {len(X_val)}   Test: {len(X_test)}")
print()

# ── 3. Grid search reps & C ──────────────────────────────────────────
CANDIDATE_REPS = [1, 2, 3, 4]
CANDIDATE_C = [0.01, 0.1, 1, 10, 100]

print("=" * 60)
print("Grid search: reps × C")
print("=" * 60)

best_val_acc = -1
best_params = {}

for reps in CANDIDATE_REPS:
    feature_map_fn = partial(build_zz_feature_map, reps=reps)

    # Compute quantum kernels (expensive — done once per reps value)
    from utils.normalize import MinMaxScaler
    scaler = MinMaxScaler()
    X_sub_scaled = scaler.fit_transform(X_sub_train)
    X_val_scaled = scaler.transform(X_val)

    K_sub = kernel_matrix(feature_map_fn, X_sub_scaled)           # symmetric
    K_val = kernel_matrix(feature_map_fn, X_sub_scaled, X_val_scaled)  # cross

    for C in CANDIDATE_C:
        svc = SVC(kernel='precomputed', C=C)
        svc.fit(K_sub, y_sub_train)
        val_acc = np.mean(svc.predict(K_val) == y_val)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = {'reps': reps, 'C': C}

        print(f"  reps={reps}  C={C:<6g}  val_acc={val_acc:.2%}")

print()

# ── 4. Train final model with best params on full training set ────────
feature_map_fn = partial(build_zz_feature_map, reps=best_params['reps'])
qksvm = QKSVM(feature_map_fn, C=best_params['C'])
qksvm.fit(X_train, y_train)

train_acc = qksvm.score(X_train, y_train)
test_acc  = qksvm.score(X_test, y_test)

print("=" * 60)
print(f"🏆 Best: reps={best_params['reps']}, C={best_params['C']:.2g}")
print(f"   Train accuracy: {train_acc:.2%}")
print(f"   Test accuracy:  {test_acc:.2%}")
print("=" * 60)
