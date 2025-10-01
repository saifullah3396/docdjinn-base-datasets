"""
Shard Writer Actor Module

This module defines the `FileStorageShardWriterActor` class, which provides functionality
for writing dataset samples to storage shards in a distributed manner using Ray. It supports
multiple storage formats, such as Msgpack and WebDataset, and allows for preprocessing of
data samples before writing.

Classes:
    - FileStorageShardWriterActor: A Ray actor for managing shard writing operations.

Dependencies:
    - ray: For distributed processing.
    - webdataset: For handling WebDataset format.
    - atria_core.logger: For logging utilities.
    - atria_datasets.core.datasets.metadata.DatasetShardInfo: For shard metadata.
    - atria_datasets.core.storage.msgpack_shard_writer.MsgpackShardWriter: For Msgpack-based shard writing.
    - atria_datasets.core.storage.utilities.FileStorageType: For storage type enumeration.
    - atria_core.types.BaseDataInstance: For dataset sample structure.

Author: Your Name (your.email@example.com)
Date: 2025-04-07
Version: 1.0.0
License: MIT
"""

from collections.abc import Callable
from typing import TYPE_CHECKING

import ray
from atria_core.logger import get_logger

if TYPE_CHECKING:
    import pandas as pd
    from atria_core.types import BaseDataInstance

    from atria_datasets.core.storage.utilities import FileStorageType

logger = get_logger(__name__)


@ray.remote
class ShardWriterActor:
    """
    A Ray actor for managing shard writing operations.

    This class provides functionality for writing dataset samples to storage shards
    in a distributed manner. It supports multiple storage formats, such as Msgpack
    and WebDataset, and allows for preprocessing of data samples before writing.

    Attributes:
        _shard_df (pd.DataFrame): A DataFrame containing the dataset samples to be written.
        _data_model (type[BaseDataInstance]): The data model class for converting rows to samples.
        _preprocess_transform (Optional[Callable]): A callable for preprocessing data samples.
        _storage_type (FileStorageType): The type of storage format (e.g., Msgpack, WebDataset).
        _storage_file_pattern (str): The file naming pattern for shards.
        _max_shard_size (int): The maximum size of each shard in terms of sample count.
        _writer (Optional[Union[wds.ShardWriter, MsgpackShardWriter]]): The shard writer instance.
        _write_info (List[DatasetShardInfo]): Metadata about the written shards.
    """

    def __init__(
        self,
        shard_df: "pd.DataFrame",
        data_model: "type[BaseDataInstance]",
        storage_type: "FileStorageType",
        storage_file_pattern: str,
        max_shard_size: int,
        preprocess_transform: Callable | None = None,
    ):
        """
        Initializes the `FileStorageShardWriterActor`.

        Args:
            shard_df (pd.DataFrame): A DataFrame containing the dataset samples to be written.
            data_model (Type[BaseDataInstanceType]): The data model class for converting rows to samples.
            storage_type (FileStorageType): The type of storage format (e.g., Msgpack, WebDataset).
            storage_file_pattern (str): The file naming pattern for shards.
            max_shard_size (int): The maximum size of each shard in terms of sample count.
            preprocess_transform (Optional[Callable]): A callable for preprocessing data samples.
        """

        import webdataset as wds

        from atria_datasets.core.storage.msgpack_shard_writer import MsgpackShardWriter

        self._shard_df = shard_df
        self._data_model = data_model
        self._storage_type = storage_type
        self._storage_file_pattern = storage_file_pattern
        self._max_shard_size = max_shard_size
        self._preprocess_transform = preprocess_transform
        self._writer: wds.ShardWriter | MsgpackShardWriter | None = None

    def load(self):
        """
        Initializes the shard writer based on the storage type.

        Raises:
            ValueError: If the storage type is unsupported.
        """

        import webdataset as wds

        from atria_datasets.core.storage.msgpack_shard_writer import MsgpackShardWriter
        from atria_datasets.core.storage.utilities import FileStorageType

        if self._storage_type == FileStorageType.MSGPACK:
            self._writer = MsgpackShardWriter(
                self._storage_file_pattern, maxcount=self._max_shard_size
            )
        elif self._storage_type == FileStorageType.WEBDATASET:
            self._writer = wds.ShardWriter(
                self._storage_file_pattern, maxcount=self._max_shard_size
            )
        else:
            raise ValueError(
                f"Unsupported storage type: {self._storage_type}. Supported types are: {FileStorageType.MSGPACK}, {FileStorageType.WEBDATASET}"
            )

    def write_shard(self):
        import webdataset as wds
        from atria_core.types import DatasetShardInfo

        assert self._writer is not None, (
            "ShardWriterActor is not loaded. Call `load()` before writing samples."
        )

        write_info = []
        for idx, sample in self._shard_df.iterrows():
            sample = self._data_model.from_row(sample)
            sample = sample.load()
            sample = (
                self._preprocess_transform(sample)
                if self._preprocess_transform
                else sample
            )
            if isinstance(self._writer, wds.ShardWriter):
                # WebDataset provides the following encoders
                output_sample = {"__key__": str(sample.key)}
                for key, value in sample.model_dump().items():
                    if value is None:
                        continue
                    if isinstance(value, str):
                        output_sample[f"{key}.txt"] = value
                    else:
                        output_sample[f"{key}.mp"] = value
                self._writer.write(output_sample)
            else:
                output_sample = {}
                for key, value in sample.model_dump().items():
                    if value is None:
                        continue
                    output_sample[key] = value
                self._writer.write({**sample.model_dump(), "key": sample.key})

            if (
                self._writer.count >= self._max_shard_size
                or idx == len(self._shard_df) - 1
            ):
                write_info.append(
                    DatasetShardInfo(
                        url=self._writer.fname,
                        shard=self._writer.shard,
                        nsamples=self._writer.count,
                        filesize=self._writer.size,
                    )
                )

        return write_info

    def close(self):
        """
        Finalizes the shard writer and releases resources.

        Returns:
            List[DatasetShardInfo]: Metadata about the written shards.
        """
        if self._writer is not None:
            self._writer.close()
            self._writer = None
