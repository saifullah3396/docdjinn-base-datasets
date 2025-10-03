"""
Atria Dataset Module

This module defines the `AtriaDataset` class, which serves as a base class for datasets
used in the Atria application. It provides functionality for managing dataset splits,
configurations, metadata, and runtime transformations.

Classes:
    - AtriaDataset: Base class for datasets in the Atria application.

Dependencies:
    - torch: For tensor operations.
    - pathlib.Path: For handling file paths.
    - typing: For type annotations and generic types.
    - atria_core.logger: For logging utilities.
    - atria_core.utilities.file: For resolving file paths.
    - atria_core.utilities.repr: For rich object representations.
    - atria_datasets.core.datasets.config: For dataset configuration classes.
    - atria_datasets.core.datasets.exceptions: For custom dataset-related exceptions.
    - atria_datasets.core.datasets.metadata: For dataset metadata management.
    - atria_datasets.core.datasets.splits: For dataset split management.
    - atria_datasets.core.storage.dataset_storage_manager: For dataset storage management.
    - atria_datasets.core.transforms.base: For runtime data transformations.

Author: Your Name (your.email@example.com)
Date: 2025-04-07
Version: 1.0.0
License: MIT
"""

from collections.abc import Callable
from typing import Any

from atria_core.logger import get_logger
from atria_core.types import BaseDataInstance

from atria_datasets.core.typing.common import T_BaseDataInstance

logger = get_logger(__name__)


class InstanceTransform:
    def __init__(
        self,
        data_model: T_BaseDataInstance,
        input_transform: Callable | None = None,
        output_transform: Callable | None = None,
        load_from_disk: bool = True,
    ):
        self._data_model = data_model
        self._input_transform = input_transform
        self._output_transform = output_transform
        self._load_from_disk = load_from_disk

    def __call__(self, index: int, data_instance: Any) -> BaseDataInstance:
        # apply input transformation
        if self._input_transform is not None:
            data_instance: BaseDataInstance = self._input_transform(data_instance)

        # assert that the transformed instance is of the expected data model type
        assert isinstance(data_instance, self._data_model), (
            f"self._input_transform(sample) should return {self._data_model}, but got {type(data_instance)}"
        )

        # apply index
        if data_instance.index is None:
            data_instance.index = index

        if self._load_from_disk:
            data_instance.load()

        # yield the transformed data instance if output transform is enabled
        if self._output_transform is not None:
            data_instance = self._output_transform(data_instance)
        return data_instance
