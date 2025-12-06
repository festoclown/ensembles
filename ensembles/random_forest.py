import json
from pathlib import Path
from typing import Any

from sklearn.exceptions import NotFittedError
from sklearn.tree import DecisionTreeRegressor
import joblib
import numpy as np
import numpy.typing as npt

from .utils import ConvergenceHistory, rmsle, whether_to_stop


class RandomForestMSE:
    def __init__(
        self, n_estimators: int, tree_params: dict[str, Any] | None = None
    ) -> None:
        """
        Handmade random forest regressor.

        Classic ML algorithm that trains a set of independent tall decision
        trees and averages its predictions.
        Employs scikit-learn `DecisionTreeRegressor` under the hood.

        Args:
            n_estimators (int): Number of trees in the forest.
            tree_params (dict[str, Any] | None, optional):
            Parameters for sklearn trees. Defaults to None.
        """
        self.n_estimators = n_estimators
        if tree_params is None:
            tree_params = {}
        self.forest = [
            DecisionTreeRegressor(**tree_params) for _ in range(n_estimators)
        ]
        self._is_fitted = False
        self._fitting = False
        self._n_fitted_estimators = 0

    def _get_bootstrap_sample(self, X, y):
        indices = np.random.choice(X.shape[0], X.shape[0], replace=True)
        return X[indices], y[indices]

    def fit(
        self,
        X: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        X_val: npt.NDArray[np.float64] | None = None,
        y_val: npt.NDArray[np.float64] | None = None,
        trace: bool | None = None,
        patience: int | None = None,
    ) -> ConvergenceHistory | None:
        """
        Train an ensemble of trees on the provided data.

        Args:
            X (npt.NDArray[np.float64]):
                Objects features matrix, array of shape (n_objects, n_features).
            y (npt.NDArray[np.float64]):
                Regression labels, array of shape (n_objects,).
            X_val (npt.NDArray[np.float64] | None, optional):
                Validation set of objects,
                array of shape (n_val_objects, n_features).
                Defaults to None.
            y_val (npt.NDArray[np.float64] | None, optional):
                Validation set of labels, array of shape (n_val_objects,).
                Defaults to None.
            trace (bool | None, optional):
                Whether to calculate rmsle while training.
                True by default if validation data is provided. Defaults to None.
            patience (int | None, optional):
                Number of training steps without decreasing the train loss
                (or validation if provided), after which to stop training.
                Defaults to None.

        Returns:
            ConvergenceHistory | None: Instance of `ConvergenceHistory`
                if `trace=True` or if validation data is provided.
        """
        self._fitting = True
        validation = None
        if (X_val is not None and y_val is not None):
            validation = True

        if trace is None:
            trace = validation

        if trace:
            history: ConvergenceHistory = {
                "train": [],
                "val": None
            }
            if validation:
                history['val'] = []

        for i in range(self.n_estimators):

            X_bootstrap, y_bootstrap = self._get_bootstrap_sample(X, y)
            self.forest[i].fit(X_bootstrap,
                               y_bootstrap)
            self._n_fitted_estimators += 1

            if trace:
                y_pred_train = self.predict(X)
                train_loss = rmsle(y, y_pred_train)
                history['train'].append(train_loss)

                if validation:
                    y_pred_val = self.predict(X_val)
                    val_loss = rmsle(y_val, y_pred_val)
                    history['val'].append(val_loss)

                if patience is not None:
                    if whether_to_stop(history, patience):
                        self.n_estimators = self._n_fitted_estimators
                        break

        self._fitting = False
        self._is_fitted = True
        if trace:
            return history
        return None

    def predict(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Make prediction with ensemble of trees.

        All the trees make their own predictions which then are averaged.

        Args:
            X (npt.NDArray[np.float64]):
                Objects' features matrix, array of shape (n_objects, n_features).

        Returns:
            npt.NDArray[np.float64]: Predicted values, array of shape (n_objects,).
        """
        if not self._is_fitted and not self._fitting:
            raise NotFittedError('Model is not fitted')
        predictions = []
        if self._fitting:
            n_estimators = self._n_fitted_estimators
        else:
            n_estimators = self.n_estimators

        for i in range(n_estimators):
            yi_pred = self.forest[i].predict(X)
            predictions.append(yi_pred)

        return np.mean(np.array(predictions), axis=0)

    def dump(self, dirpath: str) -> None:
        """
        Save the trained model to the specified directory.

        Args:
            dirpath (str): Path to the directory where the model will be saved.
        """
        path = Path(dirpath)
        path.mkdir(parents=True, exist_ok=True)

        params = {"n_estimators": self.n_estimators}
        with (path / "params.json").open("w") as file:
            json.dump(params, file, indent=4)

        trees_path = path / "trees"
        trees_path.mkdir()
        for i, tree in enumerate(self.forest):
            joblib.dump(tree, trees_path / f"tree_{i:04d}.joblib")

    @classmethod
    def load(cls, dirpath: str) -> "RandomForestMSE":
        """
        Load a trained model from the specified directory.

        Args:
            dirpath (str): Path to the directory where the model is saved.

        Returns:
            RandomForestMSE: An instance of the loaded model.
        """
        with (Path(dirpath) / "params.json").open() as file:
            params = json.load(file)
        instance = cls(params["n_estimators"])

        trees_path = Path(dirpath) / "trees"

        instance.forest = [
            joblib.load(trees_path / f"tree_{i:04d}.joblib")
            for i in range(params["n_estimators"])
        ]

        instance._is_fitted = True
        instance._fitting = False
        return instance
