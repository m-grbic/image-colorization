import sys
import os
import gc

import torch
import torch.optim as optim
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data import create_dataloader, TrainDataset, TrainRegressionDataset, load_metadata, SubsetRandomSampler
from models import ImageColorizerClassificator, ImageColorizerRegressor, save_last_model, save_best_model, load_last_state_dict
from loss import MultinomialCrossEntropyLoss, L2Loss
from utils import load_train_config, get_experiment_path
from torch_lr_finder import LRFinder
from pathlib import Path


def find_learning_rate(model, criterion, optimizer, train_dl):

    lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
    lr_finder.range_test(train_dl, end_lr=2, start_lr=1e-6, num_iter=100)
    ax, _ = lr_finder.plot()
    lr_finder.reset()
    output_dir_path = os.path.join("experiments", config.experiment_name)
    Path(output_dir_path).mkdir(parents=True, exist_ok=True)
    ax.figure.savefig(os.path.join(output_dir_path, "learning_rate_finder.png"))


def main():
    train_df, valid_df, test_df = load_metadata()

    del test_df
    gc.collect()

    if config.model.approach == 'classification':
        model = ImageColorizerClassificator(**config.model.get_init_model_dict())
        criterion = MultinomialCrossEntropyLoss(batch_size=config.batch_size, l=config.lambda_loss, use_weights=config.rebalancing)
        train_ds = TrainDataset(train_df, config.sigma_encoding)
        valid_ds = TrainDataset(valid_df, config.sigma_encoding)
    else:
        model = ImageColorizerRegressor(**config.model.get_init_model_dict())
        criterion = L2Loss()
        train_ds = TrainRegressionDataset(train_df, config.sigma_encoding)
        valid_ds = TrainRegressionDataset(valid_df, config.sigma_encoding)
    model.to(device)

    # Prepare DataLoaders
    sampler = SubsetRandomSampler(train_df, subset_size=config.num_iterations_per_epoch)

    train_dl = create_dataloader(train_ds, sampler=sampler, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    # Create scheduler
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    find_learning_rate(model, criterion, optimizer, train_dl)


if __name__ == "__main__":
    config = load_train_config()
    print(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device}")
    main()
