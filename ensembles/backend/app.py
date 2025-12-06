from pathlib import Path as FSPath

import shutil
import json
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .schemas import ExistingExperimentsResponse, ExperimentConfig
from ..boosting import GradientBoostingMSE
from ..random_forest import RandomForestMSE

app = FastAPI()


def get_runs_dir() -> FSPath:
    """
    Get path to runs directory

    Returns:
        FSPath: path to runs directory
    """
    return FSPath.cwd() / "runs"


@app.get("/existing_experiments/")
async def existing_experiments() -> ExistingExperimentsResponse:
    """
    Get information about existing experiments.

    This endpoint scans the directory where experiments are stored and returns a list of
    existing experiments along with their absolute paths. Each experiment is stored as
    a directory in the host filesystem.

    Returns:
        ExistingExperimentsResponse: A response containing the location of the experiments
        directory, absolute paths of the experiment directories, and the names of the experiments.
    """
    path = get_runs_dir()
    response = ExistingExperimentsResponse(location=path)
    if not path.exists():
        return response
    response.abs_paths = [obj for obj in path.iterdir() if obj.is_dir()]
    response.experiment_names = [filepath.stem
                                 for filepath in response.abs_paths]
    return response


@app.post("/register/")
async def register_experiment(
    config: str = Form(...),
    train_file: UploadFile = File(...)
):
    """
    Register a new experiment by saving its config and training data.

    Args:
        config (str): JSON string containing experiment configuration.
        train_file (UploadFile): CSV file with training data.

    Returns:
        dict: Confirmation message or experiment metadata.
    """
    try:
        config_obj = ExperimentConfig.model_validate_json(config)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid config: {e}") from e

    runs_dir = get_runs_dir()
    runs_dir.mkdir(exist_ok=True)

    experiment_dir = runs_dir / config_obj.name
    if experiment_dir.exists():
        ex_message = f"Experiment '{config_obj.name}' already exists"
        raise HTTPException(status_code=400,
                            detail=ex_message)

    experiment_dir.mkdir()

    train_path = experiment_dir / "train.csv"
    with open(train_path, "wb", encoding="utf-8") as f:
        shutil.copyfileobj(train_file.file, f)

    config_path = experiment_dir / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(config_obj.model_dump_json(indent=2))

    return {"message":
            f"Experiment '{config_obj.name}' registered successfully"}


@app.get("/experiment/{experiment_name}/config")
async def get_experiment_config(experiment_name: str = Path(...)):
    """
    Get config of a given experiment.

    Args:
        experiment_name (str): Name of the experiment directory.

    Returns:
        dict: Content of config.json for the experiment.
    """
    experiment_dir = get_runs_dir() / experiment_name
    if not experiment_dir.exists():
        raise HTTPException(status_code=404, detail="Experiment not found")

    config_path = experiment_dir / "config.json"
    if not config_path.exists():
        raise HTTPException(status_code=404, detail="Config not found")

    with open(config_path, encoding="utf-8") as f:
        return json.load(f)


@app.get("/needs_training")
async def needs_training(experiment_name: str):
    """
    Check if model training is needed for the experiment.

    Args:
        experiment_name (str): Name of the experiment directory.

    Returns:
        dict: {"response": bool} indicating if model is not yet trained.
    """
    experiment_dir = get_runs_dir() / experiment_name
    model_trained = (experiment_dir / "trees").exists()
    return {"response": not model_trained}


@app.post("/train/")
async def train_model(data: dict):
    """
    Train model for the specified experiment.

    Args:
        data (dict): Request payload containing "experiment_name".

    Returns:
        dict: Success message after model training and saving.
    """
    experiment_name = data["experiment_name"]
    experiment_dir = get_runs_dir() / experiment_name

    df = pd.read_csv(experiment_dir / "train.csv")
    with open(experiment_dir / "config.json", encoding="utf-8") as f:
        config = json.load(f)

    numeric_df = df.select_dtypes(include=['number'])

    X = numeric_df.drop(
        columns=[config["target_column"]]
        ).values.astype(np.float64)
    y = df[config["target_column"]].values.astype(np.float64)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    max_features = config["max_features"]
    if isinstance(max_features, str) and max_features == "all":
        max_features = X.shape[1]
    elif isinstance(max_features, str) and max_features in ("sqrt", "log2"):
        pass

    tree_params = {
        "max_depth": config["max_depth"],
        "max_features": max_features,
        "random_state": 42
    }

    if config["ml_model"] == "Random Forest":
        model = RandomForestMSE(n_estimators=config["n_estimators"],
                                tree_params=tree_params)
    elif config["ml_model"] == "Gradient Boosting":
        model = GradientBoostingMSE(
            n_estimators=config["n_estimators"],
            tree_params=tree_params,
            learning_rate=0.1
        )
    else:
        raise HTTPException(status_code=400, detail="Unknown model type")

    convergence_history = model.fit(
        X_train, y_train,
        X_val=X_val, y_val=y_val,
        trace=True,
        patience=None
    )

    model.dump(str(experiment_dir))
    with open(experiment_dir / "convergence.json", "w", encoding="utf-8") as f:

        history_to_save = {
            "train": convergence_history["train"],
            "val": convergence_history["val"] or []
        }
        json.dump(history_to_save, f)

    return {"message": "Model trained successfully"}


@app.get("/convergence/{experiment_name}")
async def get_convergence(experiment_name: str = Path(...)):
    """
    Get training convergence history for an experiment.

    Args:
        experiment_name (str): Name of the experiment with a trained model.
        test_file (UploadFile): CSV file with test data.

    Returns:
        dict: Train and validation loss history.
    """
    experiment_dir = get_runs_dir() / experiment_name
    convergence_path = experiment_dir / "convergence.json"
    if not convergence_path.exists():
        raise HTTPException(status_code=404,
                            detail="Convergence history not found")

    with open(convergence_path, encoding="utf-8") as f:
        data = json.load(f)

    return data


@app.post("/predict/")
async def predict(
    experiment_name: str = Form(...),
    test_file: UploadFile = File(...)
):
    """
    Make predictions using trained model on uploaded CSV.

    Args:
        experiment_name (str): Name of the experiment with a trained model.
        test_file (UploadFile): CSV file containing test data.

    Returns:
        dict: Model predictions for the test data.
    """
    experiment_dir = get_runs_dir() / experiment_name
    if not (experiment_dir / "trees").exists():
        raise HTTPException(status_code=400, detail="Model not trained yet")

    with open(experiment_dir / "config.json", encoding="utf-8") as f:
        config = json.load(f)

    if config["ml_model"] == "Random Forest":
        model = RandomForestMSE.load(str(experiment_dir))
    elif config["ml_model"] == "Gradient Boosting":
        model = GradientBoostingMSE.load(str(experiment_dir))
    else:
        raise HTTPException(status_code=400, detail="Unknown model type")

    test_df = pd.read_csv(test_file.file)
    numeric_df = test_df.select_dtypes(
        include=['number']
        ).drop(columns=[config["target_column"]],
               errors='ignore')
    X_test = numeric_df.values.astype(np.float64)

    predictions = model.predict(X_test).tolist()
    return {"predictions": predictions}
