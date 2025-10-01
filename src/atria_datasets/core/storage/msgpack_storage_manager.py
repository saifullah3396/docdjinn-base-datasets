"""
Msgpack Storage Manager Module

This module defines the `FileStorageManager` class, which provides functionality for managing
the storage of datasets in various formats, such as Msgpack and WebDataset. It includes methods
for writing datasets to storage, reading datasets from storage, shuffling datasets, and managing
dataset shards.

Author: Your Name (your.email@example.com)
Date: 2025-04-07
Version: 1.0.0
License: MIT
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import tqdm
from atria_core.logger.logger import get_logger

from atria_datasets.core.storage.online_shard_writer_actor import OnlineShardWriter
from atria_datasets.core.storage.utilities import FileStorageType

if TYPE_CHECKING:
    from atria_core.types import BaseDataInstance, DatasetSplitType

    from atria_datasets.core.dataset.atria_dataset import SplitIterator


logger = get_logger(__name__)


def shard_writer_worker(
    worker_id,
    sample_queue,
    result_queue,
    storage_dir,
    config_name,
    split_name,
    data_model,
    max_shard_size,
    preprocess_transform,
):
    import queue

    try:
        shard_file_pattern = str(
            Path(storage_dir)
            / config_name
            / "shards"
            / split_name
            / f"{worker_id:06d}-%06d.msgpack"
        )
        writer = OnlineShardWriter(
            data_model=data_model,
            storage_type=FileStorageType.MSGPACK,
            storage_file_pattern=shard_file_pattern,
            max_shard_size=max_shard_size,
            preprocess_transform=preprocess_transform,
        )
        writer.load()

        while True:
            try:
                item = sample_queue.get(timeout=5)  # 5 second timeout
                if item is None:  # sentinel to stop
                    break
                idx, sample = item
                writer.write(idx, sample)
            except queue.Empty:
                # Timeout occurred, continue waiting
                continue
            except Exception as e:
                logger.error(f"Worker {worker_id} error processing sample: {e}")
                # Continue processing other samples
                continue

        result_queue.put(writer.close())
    except Exception as e:
        logger.error(f"Worker {worker_id} failed with error: {e}")
        result_queue.put([])  # Put empty result to avoid blocking


def write_split_parallel(
    split_iterator, storage_dir, config_name, num_processes=4, max_shard_size=100_000
):
    import multiprocessing as mp

    split_name = split_iterator.split.value
    data_iterator = iter(split_iterator)

    # create one queue per worker for samples
    sample_queues = [mp.Queue(maxsize=10) for _ in range(num_processes)]
    result_queue = mp.Queue()

    # start workers
    workers = []
    try:
        for i in range(num_processes):
            worker = mp.Process(
                target=shard_writer_worker,
                args=(
                    i,
                    sample_queues[i],
                    result_queue,
                    storage_dir,
                    config_name,
                    split_name,
                    split_iterator.data_model,
                    max_shard_size,
                    split_iterator._tf,
                ),
            )
            worker.start()
            workers.append(worker)

        # feed samples round-robin
        worker_cycle = itertools.cycle(range(num_processes))
        try:
            for idx, sample in tqdm.tqdm(
                data_iterator, desc=f"Writing split {split_name}"
            ):
                worker_id = next(worker_cycle)
                sample_queues[worker_id].put((idx, sample))
        except Exception as e:
            logger.error(f"Error feeding samples to workers: {e}")
            raise

        # send sentinel to stop workers
        for q in sample_queues:
            q.put(None)

        # collect results
        write_info = []
        for _ in range(num_processes):
            try:
                result = result_queue.get(timeout=30)
                write_info.extend(result)
            except Exception as e:
                logger.error(f"Error collecting results from worker: {e}")
                raise

        # wait for all workers to finish
        for w in workers:
            w.join(timeout=10)  # 10 second timeout for join
            if w.is_alive():
                logger.warning(
                    f"Worker {w.pid} did not terminate gracefully, forcing termination"
                )
                w.terminate()
                w.join()

    except Exception as e:
        # cleanup: terminate all workers if something goes wrong
        logger.error(f"Error in parallel writing: {e}")
        for w in workers:
            if w.is_alive():
                w.terminate()
                w.join()
        raise
    finally:
        # ensure all queues are closed
        for q in sample_queues:
            q.close()
            q.join_thread()
        result_queue.close()
        result_queue.join_thread()

    # renumber shards
    write_info = [x for x in write_info if x.nsamples > 0]
    for i, shard in enumerate(write_info):
        shard.shard = i + 1

    return write_info


def write_split_single(
    split_iterator, storage_dir, config_name, max_shard_size=100_000
):
    """
    Writes a dataset split to storage shards using a single process.

    Args:
        split_iterator: The split iterator containing the data to write
        storage_dir: The storage directory path
        config_name: The configuration name for the dataset
        max_shard_size: Maximum number of samples per shard

    Returns:
        list: List of shard information objects
    """
    split_name = split_iterator.split.value
    data_iterator = iter(split_iterator)

    shard_file_pattern = str(
        Path(storage_dir) / config_name / "shards" / split_name / "000000-%06d.msgpack"
    )

    writer = OnlineShardWriter(
        chunked_iterator=data_iterator,
        data_model=split_iterator.data_model,
        storage_type=FileStorageType.MSGPACK,
        storage_file_pattern=shard_file_pattern,
        max_shard_size=max_shard_size,
        preprocess_transform=split_iterator._tf,
    )
    writer.load()

    for idx, sample in tqdm.tqdm(data_iterator, desc=f"Writing split {split_name}"):
        writer.write(idx, sample)

    write_info = writer.close()
    write_info = [x for x in write_info if x.nsamples > 0]
    for i, shard in enumerate(write_info):
        shard.shard = i + 1

    return write_info


class MsgpackStorageManager:
    """
    Manager for storing datasets in MessagePack format with support for both
    single-process and multi-process writing.

    Args:
        storage_dir: Directory where the dataset shards will be stored
        config_name: Name/identifier for the dataset configuration
        num_processes: Number of processes to use for parallel writing (1 = single-process)
        max_memory: Maximum memory to use (currently unused but kept for compatibility)
        max_shard_size: Maximum number of samples per shard file
    """

    def __init__(
        self,
        storage_dir: str,
        config_name: str,
        num_processes: int = 8,
        max_memory: int = 1000_000_000,
        max_shard_size: int = 100_000,
    ):
        self._storage_dir = Path(storage_dir)
        self._config_name = config_name
        self._num_processes = num_processes
        self._max_memory = max_memory
        self._max_shard_size = max_shard_size

        if not self._storage_dir.exists():
            self._storage_dir.mkdir(parents=True, exist_ok=True)
        assert self._storage_dir.is_dir(), (
            f"Storage directory {self._storage_dir} must be a directory."
        )
        (self._storage_dir / config_name).mkdir(parents=True, exist_ok=True)

    def split_dir(self, split: DatasetSplitType) -> Path:
        return Path(self._storage_dir) / f"{self._config_name}/shards/{split.value}"

    def dataset_exists(self) -> bool:
        return (Path(self._storage_dir) / f"{self._config_name}/shards/").exists()

    def split_exists(self, split: DatasetSplitType) -> bool:
        return self.split_dir(split).exists()

    def split_files(self, split: DatasetSplitType) -> bool:
        return list(self.split_dir(split).glob("*.msgpack"))

    def purge_split(self, split: DatasetSplitType) -> None:
        import shutil

        split_dir = self.split_dir(split)
        if split_dir.exists():
            logger.info(
                f"Purging dataset split {split.value} from storage {split_dir}."
            )
            shutil.rmtree(split_dir)

    def write_split(self, split_iterator: SplitIterator) -> None:
        try:
            self._write(split_iterator=split_iterator)
        except Exception as e:
            self.purge_split(split_iterator.split)
            raise RuntimeError(
                f"Error while writing dataset split {split_iterator.split.value} to storage"
            ) from e
        except KeyboardInterrupt as e:
            self.purge_split(split_iterator.split)
            raise KeyboardInterrupt(
                "KeyboardInterrupt detected. Stopping dataset preparation..."
            ) from e

    def map_to_nullable_dtypes(
        self, row_serialization_types: dict[str, type]
    ) -> dict[str, str]:
        type_map = {str: "string", int: "Int64", float: "Float64", bool: "boolean"}
        return {
            col: type_map.get(tp, "object")
            for col, tp in row_serialization_types.items()
        }

    def read_split(
        self,
        split: DatasetSplitType,
        data_model: type[BaseDataInstance],
        output_transform: Callable | None = None,
        allowed_keys: set[str] | None = None,
    ) -> SplitIterator:
        from atria_datasets.core.dataset.atria_dataset import SplitIterator
        from atria_datasets.core.storage.shard_list_datasets import (
            MsgpackShardListDataset,
        )

        if not self.split_exists(split):
            raise RuntimeError(
                f"Dataset split {split.value} not prepared. Please call `write_split()` first."
            )

        if allowed_keys is not None:
            allowed_keys.add("sample_id")
            allowed_keys.add("index")

        def input_transform(sample):
            filtered_sample = {}
            for key in list(sample.keys()):
                if allowed_keys is not None and key not in allowed_keys:
                    continue
                filtered_sample[key] = sample[key]
            return data_model(**filtered_sample)

        return SplitIterator(
            split=split,
            base_iterator=MsgpackShardListDataset(self.split_files(split)),
            input_transform=input_transform,
            output_transform=output_transform,
            data_model=data_model,
        )

    def get_splits(self) -> list[DatasetSplitType]:
        """
        Get a list of available dataset splits in the storage directory.

        Returns:
            list[DatasetSplitType]: A list of available dataset splits.
        """
        from atria_core.types import DatasetSplitType

        return [split for split in DatasetSplitType if self.split_exists(split)]

    def _write(self, split_iterator: SplitIterator) -> Path:
        """
        Writes a dataset split to storage shards.

        Supports both single-process and multi-process modes using multiprocessing.
        """
        split_dir = self.split_dir(split=split_iterator.split)
        logger.info(
            f"Preprocessing dataset split {split_iterator.split.value} to cached msgpack storage {split_dir}"
        )
        split_iterator.disable_tf()

        if self._num_processes > 1:
            # --- Multi-process mode using multiprocessing ---
            write_info = write_split_parallel(
                split_iterator=split_iterator,
                storage_dir=str(self._storage_dir),
                config_name=self._config_name,
                num_processes=self._num_processes,
                max_shard_size=self._max_shard_size,
            )
        else:
            # --- Single-process mode ---
            write_info = write_split_single(
                split_iterator=split_iterator,
                storage_dir=str(self._storage_dir),
                config_name=self._config_name,
                max_shard_size=self._max_shard_size,
            )

        # Log the writing results
        total_samples = sum(shard.nsamples for shard in write_info)
        logger.info(
            f"Successfully wrote {total_samples} samples to {len(write_info)} shards "
            f"for split {split_iterator.split.value}"
        )

        split_iterator.enable_tf()
        return split_dir
