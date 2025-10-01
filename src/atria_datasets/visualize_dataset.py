import fire
import matplotlib.pyplot as plt
from atria_core.logger import get_logger

from atria_datasets.core.dataset.atria_dataset import DatasetLoadingMode
from atria_datasets.core.storage.utilities import FileStorageType

logger = get_logger(__name__)


def visualize_samples(dataset, split_name="train", num_samples=16, grid_cols=4):
    """Visualize dataset samples in a grid layout."""
    split = getattr(dataset, split_name)

    # Calculate grid dimensions
    grid_rows = (num_samples + grid_cols - 1) // grid_cols

    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(15, 4 * grid_rows))
    if grid_rows == 1:
        axes = axes.reshape(1, -1)

    samples = []
    for i, sample in enumerate(split):
        if i >= num_samples:
            break
        sample.load()
        samples.append(sample)

    for i in range(grid_rows * grid_cols):
        row = i // grid_cols
        col = i % grid_cols
        ax = axes[row, col]

        if i < len(samples):
            sample = samples[i]

            # Handle different types of data (images, text, etc.)
            if hasattr(sample, "image") and sample.image is not None:
                # Display image
                ax.imshow(sample.image)
                ax.set_title(f"Sample {i}")
            elif hasattr(sample, "text") and sample.text is not None:
                # Display text
                ax.text(
                    0.5,
                    0.5,
                    str(sample.text)[:100] + "...",
                    ha="center",
                    va="center",
                    wrap=True,
                    fontsize=8,
                )
                ax.set_title(f"Sample {i}")
            else:
                # Display sample info
                sample_info = str(sample)[:200] + "..."
                ax.text(
                    0.5,
                    0.5,
                    sample_info,
                    ha="center",
                    va="center",
                    wrap=True,
                    fontsize=6,
                )
                ax.set_title(f"Sample {i}")
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_title("Empty")

        ax.axis("off")

    plt.tight_layout()
    plt.show()


def main(
    name: str,
    data_dir: str,
    access_token: str | None = None,
    overwrite_existing_cached: bool = False,
    num_processes: int = 8,
    upload_to_hub: bool = False,
    overwrite_in_hub: bool = True,
    visualize: bool = True,
    split_name: str = "train",
    num_samples: int = 16,
    grid_cols: int = 4,
):
    from atria_datasets import AtriaDataset

    dataset = AtriaDataset.load_from_registry(
        name=name,
        data_dir=data_dir + "/" + name.split("/")[0],
        overwrite_existing_cached=overwrite_existing_cached,
        access_token=access_token,
        num_processes=num_processes,
        enable_cached_splits=True,
        cached_storage_type=FileStorageType.MSGPACK,
        dataset_load_mode=DatasetLoadingMode.local_streaming,
    )
    logger.info(f"Loaded dataset:\n{dataset}")

    if visualize:
        logger.info(f"Visualizing {num_samples} samples from {split_name} split")
        visualize_samples(dataset, split_name, num_samples, grid_cols)

    if upload_to_hub:
        dataset.upload_to_hub(overwrite_existing=overwrite_in_hub)


if __name__ == "__main__":
    fire.Fire(main)
