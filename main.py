"""
QKSVM — Quantum Kernel Support Vector Machine Demo

Trains a hybrid quantum-classical SVM on the sklearn make_moons dataset
and reports accuracy on a held-out test set.
"""

import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

from classifier.qksvm import QKSVM
from feature_maps.zz_feature_map import build_zz_feature_map

# ── 1. Load a toy binary classification dataset ──────────────────────
X, y = make_moons(n_samples=40, noise=0.15, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print(f"Train set size: {len(X_train)}")
print(f"Test set size:  {len(X_test)}")
print(f"Features:       {X_train.shape[1]}")
print()

# ── 2. Build and train the QKSVM ─────────────────────────────────────
qksvm = QKSVM(
    feature_map_fn=build_zz_feature_map,
    C=1.0,
)
qksvm.fit(X_train, y_train)

# ── 3. Evaluate ───────────────────────────────────────────────────────
train_acc = qksvm.score(X_train, y_train)
test_acc  = qksvm.score(X_test, y_test)

print(f"Train accuracy: {train_acc:.2%}")
print(f"Test accuracy:  {test_acc:.2%}")
print()

# ── 4. Predict on a few test samples ──────────────────────────────────
y_pred = qksvm.predict(X_test)
print("Predictions on test set:")
for i in range(min(5, len(X_test))):
    print(f"  X={X_test[i]}  true={y_test[i]}  pred={y_pred[i]}")
