import yaml
from pydantic import BaseModel
from typing import Literal, List, Optional, Tuple, Union
from pathlib import Path

from .paths import get_train_config_path, get_eval_config_path, get_train_config_output_path, get_experiment_name, get_visualize_config_path


class TrainConfig(BaseModel):
    batch_size: int
    upsampling_method: Literal["up_conv", "conv_up", "deconv"]
    backbone: Literal["resnet", "resnext", "vit"]
    rebalancing: bool
    experiment_name: str
    learning_rate: float
    pretrained: bool = True
    freeze_backbone: bool = False
    lambda_loss: float = 0.5
    num_workers: int = 10
    max_num_epoch: int = 1000
    num_iterations_per_epoch: int = 320_000
    lr_scheduler_step: int = 3
    early_stopping_patience: int = 5
    start_epoch: int = 0
    approach: Literal['classification', 'regression'] = 'classification'


class EvalConfig(BaseModel):
    batch_size: int
    experiment_name: str
    num_workers: int = 10


class VisualizeConfig(BaseModel):
    experiment_name: str
    temperature: float
    anealing_temperatures: Union[Tuple[float], List[float]] =  (1, .77, .58, .38, .29, .14, 0)
    indices: Optional[List[int]] = None
    num_samples: int = 1


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    

def load_train_config() -> TrainConfig:
    train_config_path = get_train_config_path()
    config_dict = load_config(train_config_path)
    if "experiment_name" not in config_dict:
        config_dict["experiment_name"] = get_experiment_name()
    config =  TrainConfig(**config_dict)
    save_train_config(config)
    return config


def load_eval_config() -> EvalConfig:
    config_path = get_eval_config_path()
    config_dict = load_config(config_path)
    return EvalConfig(**config_dict)


def load_visualize_config() -> VisualizeConfig:
    config_path = get_visualize_config_path()
    config_dict = load_config(config_path)
    return VisualizeConfig(**config_dict)


def save_train_config(config: TrainConfig) -> None:
    config_path = get_train_config_output_path(config.experiment_name)
    Path(config_path).parent.mkdir(exist_ok=True, parents=True)
    with open(config_path, "w") as f:
        yaml.safe_dump(config.dict(), f)
