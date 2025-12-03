from typing import Any

import numpy as np
import numpy.typing as npt
import requests


from ensembles.backend.schemas import (
    ExperimentConfig,
    ConvergenceHistoryResponse
)


class Client:
    def __init__(self, base_url: str) -> None:
        """
        Initializes the Client with a base URL for the API.

        Args:
            base_url (str): The base URL of the API.
        """

        self.base_url = base_url
        self.session = requests.Session()

    def get_names(self) -> list[str]:
        """
        Retrieves the names of all existing experiments.

        Returns:
            list[str]: A list of experiment names.
        """

        response = self.session.get(f"{self.base_url}/existing_experiments/")
        response.raise_for_status()
        return response.json()["experiment_names"]

    def register_experiment(self, experiment_config, train_file) -> None:
        """
        Registers a new experiment with the given configuration and training data.

        Args:
            experiment_config (Any): The configuration for the experiment.
            train_file (Any): The training data file.
        """
        config_json = experiment_config.model_dump_json()
        files = {"train_file": train_file}
        data = {"config": config_json}
        response = self.session.post(f"{self.base_url}/register/",
                                     data=data,
                                     files=files)
        response.raise_for_status()

    def load_experiment_config(self, experiment_name) -> dict[str, Any]:
        """
        Loads the configuration of an existing experiment.

        Args:
            experiment_name (Any): The name of the experiment.

        Returns:
            ExperimentConfig: The configuration of the experiment.
        """
        response_str = f"{self.base_url}/experiment/{experiment_name}/config"
        response = self.session.get(response_str)
        response.raise_for_status()
        data = response.json()
        return ExperimentConfig(**data)

    def is_training_needed(self, experiment_name) -> bool:
        """
        Request info about was the model ever trained.

        Args:
            experiment_name (Any): The name of the experiment.

        Returns:
            bool: indicator was the model ever trained.
        """
        params = {"experiment_name": experiment_name}
        response = self.session.get(f"{self.base_url}/needs_training",
                                    params=params)
        response.raise_for_status()
        return response.json()["response"]

    def train_model(self, experiment_name) -> None:
        """
        Trains the model for the specified experiment.

        Args:
            experiment_name (Any): The name of the experiment.
        """

        data = {"experiment_name": experiment_name}
        response = self.session.post(f"{self.base_url}/train/",
                                     json=data)
        response.raise_for_status()

    def get_convergence_history(self, experiment_name) -> ConvergenceHistoryResponse:
        """
        Retrieves the convergence history of the specified experiment.

        Args:
            experiment_name (Any): The name of the experiment.

        Returns:
            ConvergenceHistory: The convergence history of the experiment.
        """
        response_str = f"{self.base_url}/convergence/{experiment_name}"
        response = self.session.get(response_str)
        response.raise_for_status()
        data = response.json()
        return ConvergenceHistoryResponse.model_validate(data)

    def predict(self, experiment_name, test_file) -> npt.NDArray[Any]:
        """
        Makes predictions using the trained model of the specified experiment.

        Args:
            experiment_name (Any): The name of the experiment.
            test_file (Any): The test data file.

        Returns:
            npt.NDArray[Any]: The predictions made by the model.
        """

        files = {"test_file": test_file}
        data = {"experiment_name": experiment_name}
        response = self.session.post(f"{self.base_url}/predict/",
                                     data=data,
                                     files=files)
        response.raise_for_status()
        predictions = response.json()["predictions"]
        return np.array(predictions)
