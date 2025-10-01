"""
Shard List Datasets Module

This module defines the `MsgpackShardListDataset` and `TarShardListDataset` classes, which
provide utilities for loading and iterating over datasets stored in shard files. These classes
support Msgpack-based and tar-based shard formats, respectively.

Classes:
    - MsgpackShardListDataset: A dataset class for reading Msgpack-based shard files.
    - TarShardListDataset: A dataset class for reading tar-based shard files.

Dependencies:
    - shutil: For file and directory operations.
    - pathlib.Path: For handling file paths.
    - typing: For type annotations.
    - numpy: For numerical operations.
    - wids: For handling tar-based shard datasets.
    - datadings.reader.MsgpackReader: For reading Msgpack-based shard files.
    - torch.utils.data.Dataset: For creating PyTorch-compatible datasets.
    - atria_core.logger: For logging utilities.
    - atria_datasets.core.datasets.metadata.DatasetShardInfo: For shard metadata.

Author: Your Name (your.email@example.com)
Date: 2025-04-07
Version: 1.0.0
License: MIT
"""

import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from atria_core.logger import get_logger
from atria_core.types import DatasetShardInfo
from datadings.reader import MsgpackReader as MsgpackFileReader
from wids import ShardListDataset

logger = get_logger(__name__)


class MsgpackShardListDataset(Sequence[Any]):
    """
    A dataset class for reading Msgpack-based shard files.

    This class provides functionality for loading and iterating over datasets stored
    in Msgpack-based shard files. It supports efficient indexing and cumulative size
    calculations for handling multiple shards.

    Attributes:
        _shard_files (list[str]): A list of Msgpack file path for each shard.
        _cumulative_sizes (list[int]): Cumulative sizes of the shards for efficient indexing.
        _total_size (int): The total number of samples across all shards.
    """

    def __init__(self, shard_files: list[str]) -> None:
        """
        Initializes the `MsgpackShardListDataset`.

        Args:
            shard_files (List[DatasetShardInfo]): A list of shard metadata containing file URLs.
        """
        self._shard_files = shard_files
        self._shard_file_readers = [MsgpackFileReader(f) for f in shard_files]
        self._total_size: int = 0

        cumulative_sizes: list[int] = []
        for data in self._shard_file_readers:
            self._total_size += len(data)
            cumulative_sizes.append(self._total_size)
            data._close()
        self._cumulative_sizes = np.array(cumulative_sizes)

    @property
    def shard_files(self) -> list[DatasetShardInfo]:
        """
        Returns the list of shard files.

        Returns:
            List[DatasetShardInfo]: The list of shard metadata.
        """
        return self._shard_files

    def __getitem__(self, index: int) -> dict[str, Any]:  # type: ignore[override]
        """
        Retrieves a sample from the dataset by index.

        Args:
            index (int): The index of the sample to retrieve.

        Returns:
            Dict[str, Any]: The sample at the specified index.
        """
        shard_index = np.searchsorted(self._cumulative_sizes, index, side="right")
        if shard_index == 0:
            inner_index = index
        else:
            inner_index = index - self._cumulative_sizes[shard_index - 1]
        sample = self._shard_file_readers[shard_index][inner_index]
        sample.pop("key", None)
        return sample

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: The total number of samples.
        """
        return self._total_size

    def close(self) -> None:
        """
        Closes all shard file readers to release resources.
        """
        for reader in self._shard_file_readers:
            reader._close()


class TarShardListDataset(ShardListDataset):
    """
    A dataset class for reading tar-based shard files.

    This class provides functionality for loading and iterating over datasets stored
    in tar-based shard files. It supports efficient handling of shard metadata and
    caching for improved performance.

    Attributes:
        cache_dir (Path): The directory used for caching shard data.
    """

    def __init__(self, shard_info_list: list[DatasetShardInfo]) -> None:
        """
        Initializes the `TarShardListDataset`.

        Args:
            shard_files (List[DatasetShardInfo]): A list of shard metadata containing file URLs.
        """
        if isinstance(shard_info_list[0], DatasetShardInfo):
            self.shard_info_list = shard_info_list
            super().__init__(
                [shard_file.model_dump() for shard_file in shard_info_list]
            )
        else:
            import wids

            shard_info_list = [
                {"url": file, "nsamples": wids.wids.compute_num_samples(file)}
                for file in shard_info_list
            ]
            shard_info_list = [
                shard for shard in shard_info_list if shard["nsamples"] > 0
            ]

            super().__init__(shard_info_list)

        # Always clean the cache directory on startup (default is /tmp/wids)
        if Path(self.cache_dir).exists():
            shutil.rmtree(Path(self.cache_dir))
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

        # we remove all transformations as we have our own
        self.transformations = []  # type: ignore
