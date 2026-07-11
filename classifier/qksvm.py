import numpy as np
from sklearn.svm import SVC

from kernels.quantum_kernel import kernel_matrix
from utils.normalize import MinMaxScaler


class QKSVM:
    """
    Quantum Kernel Support Vector Machine.

    A hybrid classifier that uses a quantum computer to compute the
    kernel matrix (pairwise similarity of data in Hilbert space) and
    feeds it into a classical SVM with ``kernel='precomputed'``.

    Parameters
    ----------
    feature_map_fn : callable
        A function that takes a list/array of feature values and returns a
        ``QuantumCircuit``.  Example: ``feature_maps.zz_feature_map.build_zz_feature_map``.
    C : float, default=1.0
        Regularization parameter for the SVM.
    **svm_kwargs
        Additional keyword arguments passed to ``sklearn.svm.SVC``.
    """

    def __init__(self, feature_map_fn, C=1.0, **svm_kwargs):
        self.feature_map_fn = feature_map_fn
        self.C = C
        self.svm_kwargs = svm_kwargs

        self.scaler_ = MinMaxScaler()
        self.X_fit_ = None
        self.svc_ = None

    def fit(self, X, y):
        """
        Fit the QKSVM model.

        1. Normalise features to [-π, π].
        2. Build the N×N quantum kernel matrix.
        3. Train a classical SVM with the precomputed kernel.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target labels.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        # 1. Normalise to [-π, π] so quantum gates operate on sensible angles
        X_scaled = self.scaler_.fit_transform(X)

        # 2. Quantum kernel matrix (symmetric N×N)
        K = kernel_matrix(self.feature_map_fn, X_scaled)

        # 3. Train classical SVM with precomputed kernel
        self.svc_ = SVC(
            kernel='precomputed',
            C=self.C,
            **self.svm_kwargs
        )
        self.svc_.fit(K, y)
        self.X_fit_ = X_scaled

        return self

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to predict.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        if self.svc_ is None:
            raise RuntimeError(
                "Classifier has not been fitted yet. Call .fit() first."
            )

        X = np.asarray(X, dtype=float)
        X_scaled = self.scaler_.transform(X)

        # K_test has shape (n_test, n_train) — compare each test point
        # against every training point, as required by sklearn's
        # precomputed kernel predict().
        K_test = kernel_matrix(self.feature_map_fn, self.X_fit_, X_scaled)

        return self.svc_.predict(K_test)

    def score(self, X, y):
        """
        Return the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.
        y : array-like of shape (n_samples,)
            True labels.

        Returns
        -------
        score : float
            Mean accuracy of ``self.predict(X)`` wrt. ``y``.
        """
        return np.mean(self.predict(X) == np.asarray(y))
