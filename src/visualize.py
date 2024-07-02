import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), "src"))

import matplotlib.pyplot as plt
import numpy as np
import random
import torch
from data import VisualDataset, load_metadata
from models import load_best_state_dict, ImageColorizerClassificator, ImageColorizerRegressor
from colorizer import ImageColorizer, save_comparison_plot
from utils import load_visualize_config, get_plots_save_path
from pathlib import Path


def visualize_temperature_change(model, test_ds, indices,  temperatures = (1, .77, .58, .38, .29, .14, 0)) -> None:
    # Create a figure with two subplots
    fig, axes = plt.subplots(len(indices), len(temperatures), figsize=(12, 6))

    for it, t in enumerate(temperatures):
        image_colorizer = ImageColorizer(model, t)
        for i, idx in enumerate(indices):
            x, l, _ = test_ds[idx]
            ab_components = image_colorizer(x)
            reconstructed_rgb_image = image_colorizer.reconstruct_image(l, ab_components)

            axes[i, it].imshow(reconstructed_rgb_image)
            if i == 0:
                axes[i, it].set_title(f"T={t}")
            axes[i, it].axis('off')
    
    plt.tight_layout()
    plot_save_path = os.path.join(
            get_plots_save_path(config.experiment_name),
            f"temperatures.png"
        )
    Path(plot_save_path).parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(plot_save_path, bbox_inches='tight')


def main():

    _, _, test_df = load_metadata()

    test_ds = VisualDataset(test_df)

    state_dict = load_best_state_dict(config.experiment_name)
    if config.model.approach == "classification":
        model = ImageColorizerClassificator(**config.model.get_init_model_dict())
    else:
        model = ImageColorizerRegressor(**config.model.get_init_model_dict())
    model.load_state_dict(state_dict=state_dict)

    image_colorizer = ImageColorizer(model, config.temperature)

    indices = config.indices or random.sample(test_df.index.tolist(), k=config.num_samples)

    visualize_temperature_change(model, test_ds, random.sample(indices, k=2))

    for idx in indices:
        x, l, image_rgb = test_ds[idx]

        ab_components = image_colorizer(x)

        reconstructed_rgb_image = image_colorizer.reconstruct_image(l, ab_components)

        plot_save_path = os.path.join(
            get_plots_save_path(config.experiment_name),
            f"{idx}.png"
        )
        save_comparison_plot(image_rgb, reconstructed_rgb_image, plot_save_path)


if __name__ == "__main__":
    config = load_visualize_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    main()
    