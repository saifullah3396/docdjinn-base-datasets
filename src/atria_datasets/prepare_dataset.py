import fire
import matplotlib.pyplot as plt
from atria_core.logger import get_logger

from atria_datasets.core.dataset.atria_dataset import DatasetLoadingMode
from atria_datasets.core.storage.utilities import FileStorageType

logger = get_logger(__name__)


def visualize_samples(dataset, split_name="train", num_samples=16, grid_cols=4):
    """Visualize dataset samples in a grid layout."""
    from atria_core.types import DocumentInstance, ImageInstance

    split = getattr(dataset, split_name)

    # Calculate grid dimensions
    grid_rows = (num_samples + grid_cols - 1) // grid_cols

    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(12, 8))
    fig.subplots_adjust(hspace=0, wspace=0)
    if grid_rows == 1:
        axes = axes.reshape(1, -1)
    if grid_rows * grid_cols == 1:
        axes = [axes]

    samples: DocumentInstance | ImageInstance = []
    for i, sample in enumerate(split):
        if i >= num_samples:
            break
        sample: DocumentInstance | ImageInstance
        sample.load()
        samples.append(sample)

    for i in range(grid_rows * grid_cols):
        row = i // grid_cols
        col = i % grid_cols

        if grid_rows == 1:
            ax = axes[col] if grid_cols > 1 else axes[0]
        else:
            ax = axes[row, col]

        if i < len(samples):
            sample = samples[i]
            ax.imshow(sample.image.content)
            ax.set_title(f"ID: {sample.sample_id}", fontsize=10)

        ax.axis("off")

    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    plt.show()


def main(
    name: str,
    data_dir: str,
    cache_artifacts: bool = True,
    access_token: str | None = None,
    overwrite_existing_cached: bool = False,
    num_processes: int = 1,
    upload_to_hub: bool = False,
    overwrite_in_hub: bool = True,
    visualize: bool = False,
    visualized_split: str = "train",
    n_visualized_samples: int = 16,
    grid_cols: int = 4,
    print_samples: bool = True,
    max_samples: int | None = None,
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
        cache_artifacts=cache_artifacts,
        dataset_load_mode=DatasetLoadingMode.local_streaming,
        build_kwargs={
            "max_train_samples": max_samples,
            "max_test_samples": max_samples,
            "max_validation_samples": max_samples,
        },
    )
    logger.info(f"Loaded dataset:\n{dataset}")

    if print_samples:
        logger.info(f"Printing one sample from {name} dataset")
        for sample in dataset.train:
            logger.info(sample)
            break

    if visualize:
        logger.info(
            f"Visualizing {n_visualized_samples} samples from {visualized_split} split"
        )
        visualize_samples(dataset, visualized_split, n_visualized_samples, grid_cols)

    if upload_to_hub:
        dataset.upload_to_hub(overwrite_existing=overwrite_in_hub)


if __name__ == "__main__":
    fire.Fire(main)
