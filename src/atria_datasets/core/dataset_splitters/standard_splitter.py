"""
Dataset Splitter Module

This module defines the `StandardSplitter` class, which provides utilities for splitting
datasets into training and validation subsets. It supports both sequential and random
splitting strategies, with configurable options for shuffle, and split ratio.

Classes:
    - StandardSplitter: A class for splitting datasets into training and validation subsets.

Dependencies:
    - copy: For deep copying datasets.
    - typing: For type annotations.
    - torch.utils.data: For dataset splitting utilities.
    - atria_core.logger.logger: For logging utilities.
    - atria_datasets.core.datasets.atria_dataset: For the base dataset class.

Author: Your Name (your.email@example.com)
Date: 2025-04-07
Version: 1.0.0
License: MIT
"""

from typing import TYPE_CHECKING

from atria_core.utilities.repr import RepresentationMixin

if TYPE_CHECKING:
    from atria_datasets.core.dataset.split_iterator import SplitIterator


class StandardSplitter(RepresentationMixin):
    """
    A class for splitting datasets into training and validation subsets.

    This class provides methods for creating sequential and random splits of datasets.
    It supports configurable options for shuffle, and split ratio.

    Attributes:
        split_ratio (float): The ratio of the training split. Defaults to 0.8.
        shuffle (bool): Whether to shuffle the dataset before splitting. Defaults to True.
    """

    def __init__(self, split_ratio: float = 0.8, shuffle: bool = True):
        """
        Initializes the `StandardSplitter`.

        Args:
            split_ratio (float): The ratio of the training split. Defaults to 0.8.
            shuffle (bool): Whether to shuffle the dataset before splitting. Defaults to True.
        """
        self.split_ratio = split_ratio
        self.shuffle = shuffle

    def create_sequential_split(
        self, train: "SplitIterator"
    ) -> tuple["SplitIterator", "SplitIterator"]:
        """
        Creates a sequential split of the dataset.

        The dataset is split into training and validation subsets based on the split ratio,
        without shuffling.

        Args:
            train_dataset (AtriaDataset): The dataset to split.

        Returns:
            Tuple[AtriaDataset, AtriaDataset]: The training and validation subsets.
        """
        import copy

        dataset_size = len(train)
        split_point = int(dataset_size * round(self.split_ratio, 2))

        validation = copy.deepcopy(train)
        train.subset_indices = list(range(split_point))  # type: ignore
        validation.subset_indices = list(range(split_point, dataset_size))  # type: ignore
        return train, validation

    def create_random_split(
        self, train: "SplitIterator"
    ) -> tuple["SplitIterator", "SplitIterator"]:
        """
        Creates a random split of the dataset.

        The dataset is split into training and validation subsets based on the split ratio,
        with shuffling.

        Args:
            train_dataset (AtriaDataset): The dataset to split.

        Returns:
            Tuple[AtriaDataset, AtriaDataset]: The training and validation subsets.
        """
        import copy

        from sklearn.model_selection import train_test_split

        assert train is not None, (
            "The dataset must have a 'train' split defined for sequential splitting."
        )

        train_dataset_size = len(train)
        validation = copy.deepcopy(train)
        train_subset, validation_subset = train_test_split(
            list(range(train_dataset_size)), test_size=1 - self.split_ratio
        )
        train.subset_indices = list(train_subset.indices)  # type: ignore
        validation.subset_indices = list(validation_subset.indices)  # type: ignore
        return train, validation

    def __call__(
        self, train_split: "SplitIterator"
    ) -> tuple["SplitIterator", "SplitIterator"]:
        """
        Splits the dataset into training and validation subsets.

        The splitting strategy (sequential or random) is determined by the `shuffle` attribute.

        Args:
            train_dataset (AtriaDataset): The dataset to split.

        Returns:
            Tuple[AtriaDataset, AtriaDataset]: The training and validation subsets.

        Raises:
            AssertionError: If the dataset is not an instance of `AtriaDataset` or if the
                            dataset size is unknown (e.g., in iterable mode).
        """
        from atria_datasets.core.dataset.split_iterator import SplitIterator

        assert isinstance(train_split, SplitIterator), (
            "The dataset must be a PyTorch or Hugging Face dataset."
        )
        assert len(train_split) != "unknown", (
            "The dataset size is unknown. This means that the dataset is set up "
            "in iterable mode and splitting is not supported."
        )
        if self.shuffle:
            return self.create_random_split(train_split)
        else:
            return self.create_sequential_split(train_split)
