from typing import Any

from atria_core.types import ClassificationAnnotation, Image, ImageInstance, Label

from atria_datasets import DATASET, AtriaHuggingfaceImageDataset
from atria_datasets.core.dataset.atria_huggingface_dataset import (
    AtriaHuggingfaceDatasetConfig,
)


@DATASET.register(
    "mnist",
    configs=[
        AtriaHuggingfaceDatasetConfig(
            config_name="mnist", hf_repo="ylecun/mnist", hf_config_name="mnist"
        ),
        AtriaHuggingfaceDatasetConfig(
            config_name="mnist_1k",
            hf_repo="ylecun/mnist",
            hf_config_name="mnist",
            max_train_samples=1000,
            max_test_samples=1000,
            max_validation_samples=1000,
        ),
    ],
)
class MNIST(AtriaHuggingfaceImageDataset):
    def _input_transform(self, sample: dict[str, Any]) -> ImageInstance:
        return ImageInstance(
            image=Image(content=sample["image"]),
            annotations=[
                ClassificationAnnotation(
                    label=Label(
                        value=sample["label"],
                        name=self.metadata.dataset_labels.classification[
                            sample["label"]
                        ],
                    )
                )
            ],
        )
