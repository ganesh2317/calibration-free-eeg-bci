"""Common Spatial Patterns (CSP) + Linear Discriminant Analysis (LDA) Baseline Model."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mne.decoding import CSP


class CSPLDABaseline(BaseEstimator, ClassifierMixin):
    """Classical BCI baseline: Common Spatial Patterns (CSP) spatial filtering + LDA classifier.

    Extracts spatial filters to maximize variance ratio between two motor imagery classes,
    followed by log-power variance feature extraction and Linear Discriminant Analysis.
    """

    def __init__(
        self,
        n_components: int = 4,
        reg: str = None,
        log: bool = True,
        solver: str = "lsqr",
        shrinkage: str = "auto",
    ):
        self.n_components = n_components
        self.reg = reg
        self.log = log
        self.solver = solver
        self.shrinkage = shrinkage

        self.csp = CSP(
            n_components=self.n_components,
            reg=self.reg,
            log=self.log,
            cov_est="concat",
            transform_into="average_power",
        )
        self.lda = LinearDiscriminantAnalysis(
            solver=self.solver,
            shrinkage=self.shrinkage,
        )
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit CSP spatial filters and LDA classifier on training fold.

        Args:
            X: Training EEG data of shape (n_trials, n_channels, n_samples).
            y: Training class labels (n_trials,).

        Returns:
            self
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)

        # Fit CSP spatial filters
        X_features = self.csp.fit_transform(X, y)

        # Fit LDA classifier on CSP log-variance features
        self.lda.fit(X_features, y)
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply fitted CSP spatial filters to extract log-variance features.

        Args:
            X: EEG data array of shape (n_trials, n_channels, n_samples).

        Returns:
            X_feat: Extracted CSP features of shape (n_trials, n_components).
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling transform.")
        X = np.asarray(X, dtype=np.float64)
        return self.csp.transform(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for input trials.

        Args:
            X: EEG data of shape (n_trials, n_channels, n_samples).

        Returns:
            y_pred: Predicted class labels (0 or 1).
        """
        X_feat = self.transform(X)
        return self.lda.predict(X_feat)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for input trials.

        Args:
            X: EEG data of shape (n_trials, n_channels, n_samples).

        Returns:
            probs: Predicted probabilities of shape (n_trials, 2).
        """
        X_feat = self.transform(X)
        return self.lda.predict_proba(X_feat)

    @property
    def spatial_filters(self) -> np.ndarray:
        """Return fitted spatial filter weights of shape (n_components, n_channels)."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted.")
        return self.csp.filters_[: self.n_components]
