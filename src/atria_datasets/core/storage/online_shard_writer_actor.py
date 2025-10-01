"""
Shard Writer Module

Provides a single-process OnlineShardWriter class and a Ray actor wrapper for distributed
writing of dataset samples to storage shards.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING

import ray
from atria_core.logger import get_logger
from atria_core.types import DatasetShardInfo

if TYPE_CHECKING:
    from atria_core.types import BaseDataInstance

    from atria_datasets.core.storage.utilities import FileStorageType


logger = get_logger(__name__)


class OnlineShardWriter:
    """
    Single-process shard writer for writing dataset samples to storage shards.

    Supports multiple storage formats (Msgpack, WebDataset) and optional preprocessing.
    """

    def __init__(
        self,
        data_model: type["BaseDataInstance"],
        storage_type: "FileStorageType",
        storage_file_pattern: str,
        max_shard_size: int,
        preprocess_transform: Callable | None = None,
    ):
        self._data_model = data_model
        self._storage_type = storage_type
        self._storage_file_pattern = storage_file_pattern
        self._max_shard_size = max_shard_size
        self._preprocess_transform = preprocess_transform
        self._writer = None
        self._write_info: list[DatasetShardInfo] = []

    def load(self):
        """Initializes the underlying shard writer based on the storage type."""
        import webdataset as wds
        from atria_core.types import DatasetShardInfo

        from atria_datasets.core.storage.msgpack_shard_writer import MsgpackShardWriter
        from atria_datasets.core.storage.utilities import FileStorageType

        if self._storage_type == FileStorageType.MSGPACK:
            self._writer = MsgpackShardWriter(
                self._storage_file_pattern,
                maxcount=self._max_shard_size,
                overwrite=True,
            )
        elif self._storage_type == FileStorageType.WEBDATASET:
            self._writer = wds.ShardWriter(
                self._storage_file_pattern, maxcount=self._max_shard_size
            )
        else:
            raise ValueError(
                f"Unsupported storage type: {self._storage_type}. Supported types are: "
                f"{FileStorageType.MSGPACK}, {FileStorageType.WEBDATASET}"
            )
        self._write_info.append(DatasetShardInfo(url=self._writer.fname))

    def write(self, index: int, sample: "BaseDataInstance"):
        """Writes a single dataset sample to the shard."""
        import webdataset as wds

        if self._writer is None:
            raise RuntimeError("ShardWriter is not loaded. Call `load()` first.")

        # Preprocess sample if needed
        if self._preprocess_transform:
            sample = self._preprocess_transform(index, sample)

        # Detect new shard creation and update info
        if getattr(self._writer, "shard", None) != getattr(
            self._write_info[-1], "shard", None
        ):
            self._write_info.append(
                DatasetShardInfo(
                    url=self._writer.fname,
                    shard=getattr(self._writer, "shard", None),
                    total=getattr(self._writer, "total", None),
                    nsamples=getattr(self._writer, "count", None),
                    filesize=getattr(self._writer, "size", None),
                )
            )

        # Write sample
        if isinstance(self._writer, wds.ShardWriter):
            output_sample = {"__key__": str(sample.key)}
            for key, value in sample.model_dump().items():
                if value is None:
                    continue
                output_sample[
                    f"{key}.txt" if isinstance(value, str) else f"{key}.mp"
                ] = value
            self._writer.write(output_sample)
        else:
            self._writer.write({**sample.model_dump(), "key": sample.key})

        # Update current shard info
        shard_info = self._write_info[-1]
        shard_info.url = self._writer.fname
        shard_info.shard = getattr(self._writer, "shard", None)
        shard_info.nsamples = getattr(self._writer, "count", None)
        shard_info.filesize = getattr(self._writer, "size", None)

    def close(self) -> list["DatasetShardInfo"]:
        """Finalizes the writer and returns shard metadata."""
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        return self._write_info


@ray.remote
class OnlineShardWriterActor:
    """Ray actor wrapper around the single-process ShardWriter."""

    def __init__(self, *args, **kwargs):
        self._writer = OnlineShardWriter(*args, **kwargs)

    def load(self):
        return self._writer.load()

    def write(self, *args, **kwargs):
        return self._writer.write(*args, **kwargs)

    def close(self):
        return self._writer.close()
