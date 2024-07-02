from datetime import datetime
import os

EXPERIMENTS_ROOT = os.path.join(os.getcwd(), "experiments")
CONF_DIR_PATH = os.path.join(os.getcwd(), "conf")


def get_experiment_name():
    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d_%H-%M-%S")
    return current_time


def get_experiment_path(experiment_name) -> str:
    return os.path.join(EXPERIMENTS_ROOT, experiment_name)


def get_last_model_path(experiment_name) -> str:
    return os.path.join(
        get_experiment_path(experiment_name),
        f"model_last.pth"
    )


def get_best_model_path(experiment_name) -> str:
    return os.path.join(
        get_experiment_path(experiment_name),
        f"model_best.pth"
    )


def get_train_config_output_path(experiment_name) -> str:
    return os.path.join(
        get_experiment_path(experiment_name),
        "train.yaml"
    )


def get_train_config_path() -> str:
    return os.path.join(CONF_DIR_PATH, "train.yaml")


def get_eval_config_path() -> str:
    return os.path.join(CONF_DIR_PATH, "eval.yaml")


def get_visualize_config_path() -> str:
    return os.path.join(CONF_DIR_PATH, "visualize.yaml")


def get_eval_output_path(experiment_name: str) -> str:
    return os.path.join("results", experiment_name)
    

def get_plots_save_path(experiment_name: str) -> str:
    return os.path.join(get_eval_output_path(experiment_name=experiment_name), "images")

