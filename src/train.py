import sys
import os
import gc
sys.path.insert(0, os.path.join(os.getcwd(), "src"))

import torch
import torch.optim as optim
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data import create_dataloader, TrainDataset, TrainRegressionDataset, load_metadata, SubsetRandomSampler
from models import ImageColorizerClassificator, ImageColorizerRegressor, save_last_model, save_best_model, load_last_state_dict
from loss import MultinomialCrossEntropyLoss, L2Loss
from utils import load_train_config, get_experiment_path


def train_loop(train_dl, val_dl, model, criterion, optimizer, scheduler) -> None:

    best_val_loss = None
    early_stopping_patience = 0

    for epoch in range(config.start_epoch, config.max_num_epoch):
        model.train()
        running_loss = 0.0
        summary_writer.add_scalar('LearningRate', optimizer.param_groups[0]["lr"], epoch)

        for idx, (x, y) in enumerate(tqdm(train_dl)):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            y_hat = model(x)

            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            summary_writer.add_scalar('Train/RunningLoss', loss.item(), epoch * len(train_dl) + idx)
        
        epoch_loss = running_loss / len(train_dl.dataset)
        summary_writer.add_scalar('Train/EpochLoss', epoch_loss, epoch)
        print(f'Epoch {epoch+1}/{config.max_num_epoch}, Loss: {epoch_loss:.8f}')

        # Call validation loop after each epoch
        val_loss = val_loop(val_dl, model, criterion, epoch)
        print(f'Epoch {epoch+1}/{config.max_num_epoch}, Validation Loss: {val_loss:.8f}')

        # Early stopping
        if best_val_loss is None or val_loss < best_val_loss:
            print('New best model found!')
            early_stopping_patience = 0
            best_val_loss = val_loss
            save_best_model(model, config.experiment_name)
        else:
            early_stopping_patience += 1
            if early_stopping_patience == config.early_stopping_patience:
                print(
                    f"Early stopping after no loss improvement in {config.early_stopping_patience}. Training finished"
                )
                return
            
        save_last_model(model, config.experiment_name)
        scheduler.step(val_loss)
        gc.collect()


def val_loop(val_dl, model, criterion, epoch):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():  # Disable gradient calculation
        for x, y in tqdm(val_dl):
            x, y = x.to(device), y.to(device)

            y_hat = model(x)
            loss = criterion(y_hat, y)

            running_loss += loss.item()

    epoch_loss = running_loss / len(val_dl.dataset)
    summary_writer.add_scalar('Valid/EpochLoss', epoch_loss, epoch)
    return epoch_loss


def main():
    train_df, valid_df, test_df = load_metadata()

    del test_df
    gc.collect()

    if config.model.approach == 'classification':
        model = ImageColorizerClassificator(**config.model.get_init_model_dict())
        criterion = MultinomialCrossEntropyLoss(batch_size=config.batch_size, l=config.lambda_loss, use_weights=config.rebalancing)
        train_ds, valid_ds = TrainDataset(train_df), TrainDataset(valid_df)
    else:
        model = ImageColorizerRegressor(**config.model.get_init_model_dict())
        criterion = L2Loss()
        train_ds, valid_ds = TrainRegressionDataset(train_df), TrainRegressionDataset(valid_df)
    model.to(device)

    # Prepare DataLoaders
    sampler = SubsetRandomSampler(train_df, subset_size=config.num_iterations_per_epoch)

    train_dl = create_dataloader(train_ds, sampler=sampler, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    valid_dl = create_dataloader(valid_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    # Load model weights
    if config.start_epoch > 0:
        print(f"Continuing training from epoch {config.start_epoch}")
        state_dict = load_last_state_dict(config.experiment_name)
        model.load_state_dict(state_dict=state_dict)

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=config.lr_scheduler_step, verbose=True)

    train_loop(train_dl=train_dl, val_dl=valid_dl, model=model, criterion=criterion, optimizer=optimizer, scheduler=scheduler)


if __name__ == "__main__":
    config = load_train_config()
    print(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    summary_writer = SummaryWriter(get_experiment_path(config.experiment_name))
    print(f"Using {device}")
    main()
    summary_writer.close()
