import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), "src"))

import torch
import torch.optim as optim
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data import create_dataloader, EvalDataset, load_metadata
from models import load_best_model
from loss import MultinomialCrossEntropyLoss
from utils import load_eval_config, get_experiment_path


def test_loop(test_dl, model, metrics):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():  # Disable gradient calculation
        for x, y in tqdm(test_dl):
            x, y = x.to(device), y.to(device)

            y_hat = model(x)
            # TODO ADD METRICS



def main():
    _, _, test_df = load_metadata()

    test_ds = EvalDataset(test_df)

    test_dl = create_dataloader(test_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    model = load_best_model(experiment_name=config.experiment_name)

    metrics = ""

    test_loop(test_dl=test_dl, model=model, metrics=metrics)


if __name__ == "__main__":
    config = load_eval_config()
    print(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device}")
    main()
