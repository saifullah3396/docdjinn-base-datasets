from atria_core.types import (
    ClassificationAnnotation,
    DatasetLabels,
    DatasetMetadata,
    DatasetSplitType,
    Image,
    ImageInstance,
    Label,
)

from atria_datasets import DATASET
from atria_datasets.core.dataset.atria_dataset import (
    AtriaDatasetConfig,
    AtriaImageDataset,
)

_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


@DATASET.register(
    "cifar10",
    configs=[
        AtriaDatasetConfig(
            config_name="1k",
            max_train_samples=1000,
            max_test_samples=1000,
            max_validation_samples=1000,
        )
    ],
)
class Cifar10(AtriaImageDataset):
    def _custom_download(self, data_dir: str, access_token: str | None = None) -> None:
        from torchvision.datasets import CIFAR10

        CIFAR10(root=data_dir, train=True, download=True)
        CIFAR10(root=data_dir, train=False, download=True)

    def _metadata(self):
        return DatasetMetadata(
            description="CIFAR-10 dataset",
            dataset_labels=DatasetLabels(classification=_CLASSES),
        )

    def _available_splits(self) -> list[DatasetSplitType]:
        return [DatasetSplitType.train, DatasetSplitType.test]

    def _split_iterator(self, split: DatasetSplitType, data_dir: str):  # type: ignore
        from torchvision.datasets import CIFAR10

        return CIFAR10(
            root=data_dir, train=split == DatasetSplitType.train, download=False
        )

    def _input_transform(self, sample) -> ImageInstance:
        image_instance = ImageInstance(
            image=Image(content=sample[0]),
            annotations=[
                ClassificationAnnotation(
                    label=Label(value=sample[1], name=_CLASSES[sample[1]])
                )
            ],
        )
        return image_instance
