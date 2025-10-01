"""
Atria Hugging Face Dataset Module

This module defines the `AtriaHuggingfaceDataset` class, which extends the `AtriaDataset`
class to support datasets hosted on Hugging Face. It provides functionality for managing
dataset splits, configurations, metadata, and runtime transformations specific to Hugging
Face datasets.

Classes:
    - AtriaHuggingfaceDataset: A dataset class for Hugging Face datasets.

Dependencies:
    - datasets: For interacting with Hugging Face datasets.
    - pathlib.Path: For handling file paths.
    - typing: For type annotations and generic types.
    - atria_core.logger: For logging utilities.
    - atria_datasets.core.datasets.atria_dataset: For the base dataset class.
    - atria_datasets.core.datasets.config: For dataset configuration classes.
    - atria_datasets.core.datasets.downloads.download_manager: For managing dataset downloads.
    - atria_datasets.core.datasets.metadata: For dataset metadata management.
    - atria_datasets.core.datasets.splits: For dataset split management.
    - atria_core.types: For base data instance structures.

Author: Your Name (your.email@example.com)
Date: 2025-04-07
Version: 1.0.0
License: MIT
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic

from atria_core.logger import get_logger
from atria_core.types import DocumentInstance, ImageInstance

from atria_datasets.core.dataset.atria_dataset import AtriaDataset, AtriaDatasetConfig
from atria_datasets.core.storage.utilities import FileStorageType
from atria_datasets.core.typing.common import T_BaseDataInstance

if TYPE_CHECKING:
    from collections.abc import Generator
    from typing import TYPE_CHECKING, Any, Generic

    import datasets
    from atria_core.logger import get_logger
    from atria_core.types import DatasetMetadata, DatasetSplitType


logger = get_logger(__name__)


class AtriaHuggingfaceDatasetConfig(AtriaDatasetConfig):
    hf_repo: str
    hf_config_name: str


class AtriaHuggingfaceDataset(AtriaDataset, Generic[T_BaseDataInstance]):
    """
    A dataset class for Hugging Face datasets.

    This class extends the `AtriaDataset` class to provide functionality specific
    to datasets hosted on Hugging Face, including metadata extraction, split management,
    and runtime transformations.

    Attributes:
        _data_dir (Path): The directory where dataset files are stored.
        _config (AtriaDatasetConfig): The configuration for the datasets.
        _runtime_transforms (DataTransformsDict): Runtime transformations for training and evaluation.
        _active_split (DatasetSplit): The currently active dataset split.
        _active_split_config (SplitConfig): The configuration for the active split.
        _downloaded_files (Dict[str, Path]): A dictionary of downloaded files.
        _prepared_split_iterator (Iterator): The prepared iterator for the active split.
        _subset_indices (Optional[torch.Tensor]): Indices for a random subset of the datasets.
        _prepared_metadata (DatasetMetadata): Metadata for the datasets.
        _download_dir (Path): The directory for downloaded files.
        _download_manager (DownloadManager): The download manager for the datasets.
    """

    __abstract__ = True
    __config_cls__ = AtriaHuggingfaceDatasetConfig

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        assert isinstance(self.config, AtriaHuggingfaceDatasetConfig), (
            f"Expected config for {self.__class__.__name__} to be of type "
            f"AtriaHuggingfaceDatasetConfig, got {type(self.config)} instead."
        )
        self._dataset_builder = None
        self._hf_split_generators = None

    def prepare_downloads(self, data_dir: str, access_token: str | None = None) -> None:
        """
        Prepares the dataset by downloading and extracting files if data URLs are provided.

        Args:
            data_dir (str): The directory where the dataset files are stored.

        Returns:
            None
        """

        from atria_core.types import DatasetSplitType

        from atria_datasets.core.constants import _DEFAULT_DOWNLOAD_PATH

        if not self._downloads_prepared:
            download_dir = Path(data_dir) / _DEFAULT_DOWNLOAD_PATH
            download_dir.mkdir(parents=True, exist_ok=True)
            download_manager = self._prepare_download_manager(
                data_dir, download_dir=str(download_dir)
            )
            self._hf_split_generators = self._dataset_builder._split_generators(
                download_manager
            )
            HF_SPLIT_TO_ATRIA_SPLIT = {
                "train": DatasetSplitType.train,
                "validation": DatasetSplitType.validation,
                "test": DatasetSplitType.test,
            }
            self._hf_split_generators = {
                HF_SPLIT_TO_ATRIA_SPLIT[split_generator.name]: split_generator
                for split_generator in self._hf_split_generators
            }
            self._downloads_prepared = True

    def _prepare_splits(self, access_token: str | None = None) -> None:
        """Prepare splits without caching (direct iteration)."""
        from atria_datasets.core.dataset.split_iterator import HFSplitIterator

        self._dataset_builder = self._prepare_dataset_builder(self._data_dir)
        self.prepare_downloads(data_dir=self._data_dir, access_token=access_token)
        for split in self._available_splits():
            self._split_iterators[split] = HFSplitIterator(
                split=split,
                data_model=self.data_model,
                input_transform=self._input_transform,
                base_iterator=self._split_iterator(split, self._data_dir),
                max_len=self.get_max_split_samples(split),
            )

    def _prepare_cached_splits(
        self,
        access_token: str | None = None,
        cached_storage_type: FileStorageType = FileStorageType.MSGPACK,
        overwrite_existing: bool = False,
    ) -> None:
        """Prepare cached splits using DeltaLake storage."""

        from atria_core.types import DatasetSplitType

        from atria_datasets.core.dataset.split_iterator import HFSplitIterator

        storage_manager = self._get_storage_manager(cached_storage_type)

        info_saved = False
        for split in list(DatasetSplitType):
            split_exists = storage_manager.split_exists(split=split)
            if split_exists and overwrite_existing:
                logger.warning(f"Overwriting existing cached split {split.value}")
                storage_manager.purge_split(split)
                split_exists = False

            if not split_exists:
                self._dataset_builder = self._prepare_dataset_builder(self._data_dir)
                self.prepare_downloads(
                    data_dir=str(self._data_dir), access_token=access_token
                )
                if split not in self._available_splits():
                    continue
                if not info_saved:
                    self.save_dataset_info(self._storage_dir)
                storage_manager.write_split(
                    split_iterator=HFSplitIterator(
                        split=split,
                        data_model=self.data_model,
                        input_transform=self._input_transform,
                        base_iterator=self._split_iterator(split, self._data_dir),
                        max_len=self.get_max_split_samples(split),
                    )
                )
            else:
                logger.info(
                    f"Loading cached split {split.value} from {storage_manager.split_dir(split)}"
                )

            # Load split from storage
            self._split_iterators[split] = storage_manager.read_split(
                split=split, data_model=self.data_model, allowed_keys=self._allowed_keys
            )

    def _available_splits(self) -> list[DatasetSplitType]:
        """
        Returns a list of available dataset splits.

        Returns:
            List[DatasetSplitType]: A list of available dataset splits.
        """
        return list(self._hf_split_generators.keys())

    def _prepare_dataset_builder(self, data_dir: str) -> datasets.DatasetBuilder:
        """
        Prepares the Hugging Face dataset builder.

        Returns:
            datasets.DatasetBuilder: The Hugging Face dataset builder.
        """
        if self._dataset_builder is None:
            from datasets import load_dataset_builder

            self._dataset_builder = load_dataset_builder(
                self.config.hf_repo, name=self.config.hf_config_name, cache_dir=data_dir
            )
        return self._dataset_builder

    def _metadata(self) -> DatasetMetadata:
        """
        Extracts metadata from the Hugging Face datasets.

        Returns:
            DatasetMetadata: The metadata for the datasets.
        """
        from atria_core.types import DatasetMetadata

        return DatasetMetadata.from_huggingface_info(self._dataset_builder.info)

    def _prepare_download_manager(
        self, data_dir: str, download_dir: str
    ) -> datasets.DownloadManager:
        """
        Prepares the download manager for the datasets.

        Returns:
            DownloadManager: The prepared download manager.
        """

        if "packaged_modules" in str(self._dataset_builder.__module__):
            import datasets

            # If it is a packaged module, use the Hugging Face download manager.
            return datasets.DownloadManager(
                dataset_name=self.config.hf_config_name,
                data_dir=data_dir,
                download_config=datasets.DownloadConfig(
                    cache_dir=download_dir,
                    force_download=False,
                    force_extract=False,
                    use_etag=False,
                    delete_extracted=False,
                ),
                record_checksums=False,
            )
        else:
            import datasets

            return datasets.DownloadManager(
                data_dir=data_dir,
                download_config=datasets.DownloadConfig(
                    cache_dir=download_dir,
                    force_download=False,
                    force_extract=False,
                    use_etag=False,
                    delete_extracted=False,
                ),
            )

    def _split_iterator(  # type: ignore
        self, split: DatasetSplitType, data_dir: str
    ) -> Generator[Any, None, None]:
        """
        Returns an iterator for a specific dataset split.

        Args:
            split (DatasetSplit): The dataset split.
            hf_split_generator (datasets.SplitGenerator): The Hugging Face split generator.

        Yields:
            BaseDataInstanceType: The dataset instances for the specified split.
        """

        return self._prepare_dataset_builder(data_dir)._as_streaming_dataset_single(
            self._hf_split_generators[split.value]
        )


class AtriaHuggingfaceImageDataset(AtriaHuggingfaceDataset[ImageInstance]):
    """
    AtriaImageDataset is a specialized dataset class for handling image datasets.
    It inherits from AtriaDataset and provides additional functionality specific to image data.
    """

    __abstract__ = True
    __data_model__ = ImageInstance


class AtriaHuggingfaceDocumentDataset(AtriaHuggingfaceDataset[DocumentInstance]):
    """
    AtriaDocumentDataset is a specialized dataset class for handling document datasets.
    It inherits from AtriaDataset and provides additional functionality specific to document data.
    """

    __abstract__ = True
    __data_model__ = DocumentInstance
