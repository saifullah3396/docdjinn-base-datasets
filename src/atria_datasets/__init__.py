"""
Data Module

This module provides data-related utilities and components for the Atria framework. It includes dataset management, data transformations, and storage utilities.

Submodules:
    - batch_samplers: Batch sampling strategies for datasets.
    - dataset_splitters: Utilities for splitting datasets into training, validation, and test sets.
    - datasets: Dataset definitions and metadata management.
    - pipelines: Data pipelines for preprocessing and augmentation.
    - storage: File storage management for datasets.
    - structures: Data structures for representing various data types.
    - transforms: Data transformation utilities.

Author: Your Name (your.email@example.com)
Date: 2025-04-07
Version: 1.0.0
License: MIT
"""

# ruff: noqa

from typing import TYPE_CHECKING

import lazy_loader as lazy

# Ensure registry is initialized immediately
import atria_datasets.registry  # noqa: F401

if TYPE_CHECKING:
    import atria_datasets.registry  # noqa: F401 # Import the registry to ensure it is initialized
    from atria_datasets.core.dataset.atria_dataset import (
        AtriaDataset,
        AtriaDocumentDataset,
        AtriaImageDataset,
        DatasetLoadingMode,
    )
    from atria_datasets.core.dataset.atria_hub_dataset import AtriaHubDataset
    from atria_datasets.core.dataset.atria_huggingface_dataset import (
        AtriaHuggingfaceDataset,
        AtriaHuggingfaceDocumentDataset,
        AtriaHuggingfaceImageDataset,
    )
    from atria_datasets.core.dataset.split_iterator import SplitIterator
    from atria_datasets.core.dataset_splitters.standard_splitter import StandardSplitter
    from atria_datasets.core.download_manager.download_file_info import DownloadFileInfo
    from atria_datasets.core.download_manager.download_manager import DownloadManager
    from atria_datasets.core.download_manager.file_downloader import (
        FileDownloader,
        FTPFileDownloader,
        GoogleDriveDownloader,
        HTTPDownloader,
    )
    from atria_datasets.core.storage.deltalake_reader import DeltalakeReader
    from atria_datasets.core.storage.deltalake_storage_manager import (
        DeltalakeStorageManager,
    )
    from atria_datasets.core.storage.msgpack_shard_writer import (
        MsgpackFileWriter,
        MsgpackShardWriter,
    )
    from atria_datasets.core.storage.utilities import FileStorageType
    from atria_datasets.registry import (
        BATCH_SAMPLER,
        DATA_PIPELINE,
        DATASET,  # noqa
    )


__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules={"registry"},
    submod_attrs={
        "core.dataset.atria_dataset": [
            "AtriaDataset",
            "AtriaDocumentDataset",
            "AtriaImageDataset",
            "DatasetLoadingMode",
        ],
        "core.dataset.atria_huggingface_dataset": [
            "AtriaHuggingfaceDataset",
            "AtriaHuggingfaceDocumentDataset",
            "AtriaHuggingfaceImageDataset",
        ],
        "core.dataset.atria_hub_dataset": ["AtriaHubDataset"],
        "core.dataset.split_iterator": ["SplitIterator"],
        "core.dataset_splitters.standard_splitter": ["StandardSplitter"],
        "core.download_manager.download_file_info": ["DownloadFileInfo"],
        "core.download_manager.download_manager": ["DownloadManager"],
        "core.download_manager.file_downloader": [
            "FileDownloader",
            "FTPFileDownloader",
            "GoogleDriveDownloader",
            "HTTPDownloader",
        ],
        "core.storage.deltalake_reader": ["DeltalakeReader"],
        "core.storage.deltalake_storage_manager": ["DeltalakeStorageManager"],
        "core.storage.msgpack_shard_writer": [
            "MsgpackFileWriter",
            "MsgpackShardWriter",
        ],
        "core.storage.sharded_dataset_storage_manager": [
            "ShardedDatasetStorageManager"
        ],
        "core.storage.utilities": ["FileStorageType"],
        "registry": ["BATCH_SAMPLER", "DATA_PIPELINE", "DATASET"],
        "image_classification.cifar10": ["Cifar10"],
        "image_classification.cifar10_huggingface": ["HuggingfaceCifar10"],
    },
)
