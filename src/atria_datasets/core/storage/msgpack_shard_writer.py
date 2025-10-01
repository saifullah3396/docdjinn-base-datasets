"""
Msgpack Shard Writer Module

This module defines the `MsgpackFileWriter` and `MsgpackShardWriter` classes, which provide
utilities for writing datasets to Msgpack-based shard files. These classes support writing
data in chunks, managing shard sizes, and handling unique keys for dataset samples.

Classes:
    - MsgpackFileWriter: A writer for file-based datasets that ensures unique keys for samples.
    - MsgpackShardWriter: A writer for managing dataset shards with configurable size and count limits.

Dependencies:
    - typing: For type annotations.
    - datadings.writer.Writer: For writing Msgpack-based datasets.
    - atria_core.logger: For logging utilities.

Author: Your Name (your.email@example.com)
Date: 2025-04-07
Version: 1.0.0
License: MIT
"""

from collections.abc import Callable
from typing import Any

from atria_core.logger import get_logger
from datadings.writer import Writer

logger = get_logger(__name__)


class MsgpackFileWriter(Writer):
    """
    Writer for file-based datasets.

    This class extends the `Writer` class to provide functionality for writing
    Msgpack-based datasets. It ensures that each sample has a unique "key" value.

    Methods:
        - write: Writes a sample to the dataset.
    """

    def _write_data(self, key, packed):
        """
        Writes packed data to the file.

        Args:
            key (str): The unique key for the sample.
            packed (bytes): The packed data to write.

        Raises:
            ValueError: If a duplicate key is encountered.
        """
        if key in self._keys_set:
            raise ValueError(f"Duplicate key {key!r} not allowed.")
        self._keys.append(key)
        self._keys_set.add(key)
        self._hash.update(packed)
        self._outfile.write(packed)
        self._offsets.append(self._outfile.tell())
        self.written += 1

    def write(self, sample: dict[str, Any]) -> int:
        """
        Writes a sample to the dataset.

        Args:
            sample (Dict[str, Any]): The sample to write, must contain a unique "key" value.

        Returns:
            int: The number of samples written.

        Raises:
            AssertionError: If the sample does not contain a "key" value.
        """
        assert "key" in sample, "Sample must contain a unique 'key' value."
        self._write(sample["key"], sample)
        return self.written


class MsgpackShardWriter:
    """
    A writer for managing dataset shards with configurable size and count limits.

    This class provides functionality for writing datasets to Msgpack-based shard files.
    It supports automatic shard creation based on size and count limits, and allows for
    post-processing of completed shards.

    Attributes:
        pattern (str): The file naming pattern for shards.
        maxcount (int): The maximum number of samples per shard. Defaults to 100,000.
        maxsize (float): The maximum size of each shard in bytes. Defaults to 3 GB.
        post (Optional[Callable]): A callable to execute after a shard is completed.
        start_shard (int): The starting shard index. Defaults to 0.
        verbose (int): The verbosity level for logging. Defaults to 1.
        opener (Optional[Callable]): A callable for opening files.
    """

    def __init__(
        self,
        pattern: str,
        maxcount: int = 100000,
        maxsize: float = 3e9,
        post: Callable | None = None,
        start_shard: int = 0,
        verbose: int = 0,
        opener: Callable | None = None,
        **kw,
    ):
        """
        Initializes the `MsgpackShardWriter`.

        Args:
            pattern (str): The file naming pattern for shards.
            maxcount (int): The maximum number of samples per shard. Defaults to 100,000.
            maxsize (float): The maximum size of each shard in bytes. Defaults to 3 GB.
            post (Optional[Callable]): A callable to execute after a shard is completed.
            start_shard (int): The starting shard index. Defaults to 0.
            verbose (int): The verbosity level for logging. Defaults to 1.
            opener (Optional[Callable]): A callable for opening files.
            **kw: Additional keyword arguments for the writer.
        """
        self.verbose = verbose
        self.kw = kw
        self.maxcount = maxcount
        self.maxsize = maxsize
        self.post = post

        self.writer = None
        self.shard = start_shard
        self.pattern = pattern
        self.total = 0
        self.count = 0
        self.size = 0
        self.fname = None
        self.opener = opener
        self.next_stream()

    def next_stream(self) -> None:
        """
        Closes the current shard and opens a new one.

        This method finalizes the current shard, increments the shard index, and
        initializes a new shard for writing.
        """
        self.finish()
        self.fname = self.pattern % self.shard  # type: ignore
        if self.verbose:
            logger.info(
                f"# Writing {self.fname}, {self.count} samples, {self.size / 1e9:.1f} GB, {self.total} total samples."
            )
        self.shard += 1
        if self.opener:
            self.writer = MsgpackFileWriter(self.opener(self.fname), **self.kw)  # type: ignore
        else:
            self.writer = MsgpackFileWriter(self.fname, **self.kw)  # type: ignore
        self.count = 0
        self.size = 0

    def write(self, obj: dict[str, Any]) -> None:
        """
        Writes a sample to the current shard.

        Args:
            obj (Dict[str, Any]): The sample to write.

        Notes:
            - If the current shard exceeds the size or count limits, a new shard is created.
        """
        if (
            self.writer is None
            or self.count >= self.maxcount
            or self.size >= self.maxsize
        ):
            self.next_stream()
        size = self.writer.write(obj)  # type: ignore
        self.count += 1
        self.total += 1
        self.size += size

    def finish(self) -> None:
        """
        Finalizes the current shard.

        This method closes the current shard and executes the post-processing callable
        if one is provided.
        """
        if self.writer is not None:
            self.writer.close()
            assert self.fname is not None
            if callable(self.post):
                self.post(self.fname)
            self.writer = None

    def close(self) -> None:
        """
        Closes the writer and releases resources.

        This method finalizes the current shard and deletes internal attributes.
        """
        self.finish()
        del self.writer
        del self.shard
        del self.count
        del self.size

    def __enter__(self):
        """
        Enters the context manager.

        Returns:
            MsgpackShardWriter: The current instance.
        """
        return self

    def __exit__(self, *args, **kw):
        """
        Exits the context manager.

        This method finalizes the writer and releases resources.
        """
        self.close()
