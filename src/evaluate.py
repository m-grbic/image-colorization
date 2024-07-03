import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), "src"))

import torch
from tqdm import tqdm

from data import create_dataloader, EvalDataset, load_metadata
from models import load_best_state_dict, ImageColorizerClassificator, ImageColorizerRegressor
from colorizer import ImageColorizer
from utils import load_eval_config
from metrics import MetricClaculator


def test_loop(test_dl: torch.utils.data.DataLoader, image_colorizer: ImageColorizer, metric_calculator: MetricClaculator):
    with torch.no_grad():  # Disable gradient calculation
        for x, y in tqdm(test_dl):
            x, y = x.to(device), y.to(device)

            # Predict ab values
            y_hat = image_colorizer(x) - 128

            # Calculate metrics
            metric_calculator(preds=y_hat, targets=y)

        metrics_df = metric_calculator.summary()
        print(metrics_df)
            

def main():
    _, _, test_df = load_metadata()

    test_df = test_df.iloc[:1000, :]

    test_ds = EvalDataset(test_df)

    test_dl = create_dataloader(test_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    # Load model
    state_dict = load_best_state_dict(config.experiment_name)
    if config.model.approach == "classification":
        model = ImageColorizerClassificator(**config.model.get_init_model_dict())
    else:
        model = ImageColorizerRegressor(**config.model.get_init_model_dict())
    model.load_state_dict(state_dict=state_dict)
    model.eval().to(device)

    image_colorizer = ImageColorizer(model=model, approach=config.model.approach, t=config.temperature)

    metric_calculator = MetricClaculator(metrics=config.metrics)

    test_loop(test_dl=test_dl, image_colorizer=image_colorizer, metric_calculator=metric_calculator)


if __name__ == "__main__":
    config = load_eval_config()
    print(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device}")
    main()
