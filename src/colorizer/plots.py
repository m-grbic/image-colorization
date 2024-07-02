import matplotlib.pyplot as plt
from pathlib import Path


def save_comparison_plot(image_rgb, image_rgb_hat, output_path: str):
    Path(output_path).parent.mkdir(exist_ok=True, parents=True)

    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(image_rgb)
    axes[0].set_title('Orginal image')
    axes[0].axis('off')

    axes[1].imshow(image_rgb_hat)
    axes[1].set_title('Reconstructed image')
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')


