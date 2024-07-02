import os
import torch

from models import ImageColorizerSE
from utils import get_last_model_path, get_best_model_path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_best_model(model: torch.nn.Module, experiment_name: str) -> None:
    model_path = get_best_model_path(experiment_name)
    torch.save(model, model_path)


def save_last_model(model: torch.nn.Module, experiment_name: str) -> None:
    model_path = get_last_model_path(experiment_name)
    torch.save(model, model_path)


def load_best_model(experiment_name: str) -> torch.nn.Module:
    model_path = get_best_model_path(experiment_name)
    model = torch.load(model_path, map_location=device)
    return model


def load_last_model(experiment_name: str) -> torch.nn.Module:
    model_path = get_last_model_path(experiment_name)
    model = torch.load(model_path, map_location=device)
    return model