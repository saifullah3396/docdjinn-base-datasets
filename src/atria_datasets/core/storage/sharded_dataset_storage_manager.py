"""
File Storage Manager Module

This module defines the `FileStorageManager` class, which provides functionality for managing
the storage of datasets in various formats, such as Msgpack and WebDataset. It includes methods
for writing datasets to storage, reading datasets from storage, shuffling datasets, and managing
dataset shards.

Classes:
    - FileStorageManager: A class for managing dataset storage in file-based formats.

Dependencies:
    - glob: For file pattern matching.
    - itertools: For iterator utilities.
    - os: For file system operations.
    - shutil: For file and directory operations.
    - pathlib.Path: For handling file paths.
    - typing: For type annotations.
    - ray: For distributed processing.
    - tqdm: For progress tracking.
    - webdataset: For handling WebDataset format.
    - atria_core.logger: For logging utilities.
    - atria_registry: For registering storage managers.
    - atria_datasets.core.datasets: For dataset-related classes and metadata.
    - atria_datasets.core.storage.dataset_storage_manager: For the base storage manager class.
    - atria_datasets.core.storage.shard_writer_actor: For managing shard writers.
    - atria_datasets.core.storage.utilities: For utility functions and constants.

Author: Your Name (your.email@example.com)
Date: 2025-04-07
Version: 1.0.0
License: MIT
"""

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from atria_core.logger import get_logger
from atria_core.types import (
    BaseDataInstance,
    DatasetShardInfo,
    DatasetSplitType,
    SplitInfo,
)

from atria_datasets.core.dataset.atria_dataset import SplitIterator
from atria_datasets.core.storage.deltalake_reader import DeltalakeReader
from atria_datasets.core.storage.fixed_shard_writer_actor import ShardWriterActor
from atria_datasets.core.storage.utilities import (
    _RAY_RUNTIME_ENV,
    FileStorageType,
    FileUrlProvider,
)

if TYPE_CHECKING:
    import pandas as pd
    import ray

logger = get_logger(__name__)


class ShardedDatasetStorageManager:
    """
    A class for managing dataset storage in sharded formats.

    This class provides methods for writing datasets to storage, reading datasets from storage,
    shuffling datasets, and managing dataset shards. It supports multiple storage formats, such
    as Msgpack and WebDataset.

    Attributes:
        _storage_type (FileStorageType): The type of storage format (e.g., Msgpack, WebDataset).
        _num_processes (int): The number of processes for parallel processing. Defaults to 1.
        _max_shard_size (int): The maximum size of each shard. Defaults to 100,000.
        _force_reprepare (bool): Whether to overwrite existing storage. Defaults to False.
        _max_memory_per_actor (int): The maximum memory allocated per actor in bytes. Defaults to 500 MB.
        _max_concurrent_tasks_limit (int): The maximum number of concurrent tasks. Defaults to 100.
        _streaming_mode (bool): Whether to enable streaming mode for WebDataset. Defaults to False.
    """

    def __init__(
        self,
        storage_dir: str,
        storage_type: FileStorageType,
        num_processes: int = 8,
        max_shard_size: int = 100000,
        max_memory_per_actor: int = 500 * 1024 * 1024,
        max_concurrent_tasks_limit: int = 100,
        streaming_mode: bool = False,
    ):
        """
        Initializes the `FileStorageManager`.

        Args:
            storage_type (FileStorageType): The type of storage format. Defaults to Msgpack.
            num_processes (int): The number of processes for parallel processing. Defaults to 1.
            max_shard_size (int): The maximum size of each shard. Defaults to 100,000.
            max_memory_per_actor (int): The maximum memory allocated per actor in bytes. Defaults to 500 MB.
            max_concurrent_tasks_limit (int): The maximum number of concurrent tasks. Defaults to 100.
            streaming_mode (bool): Whether to enable streaming mode for WebDataset. Defaults to False.
        """
        self._storage_type = storage_type
        self._num_processes = num_processes
        self._max_shard_size = max_shard_size
        self._max_memory_per_actor = max_memory_per_actor
        self._max_concurrent_tasks_limit = max_concurrent_tasks_limit
        self._streaming_mode = streaming_mode

        self._storage_dir = Path(storage_dir)
        if not self._storage_dir.exists():
            self._storage_dir.mkdir(parents=True, exist_ok=True)
        assert self._storage_dir.is_dir(), (
            f"Storage directory {self._storage_dir} must be a directory."
        )

    def split_dir(self, split: "DatasetSplitType") -> Path:
        """
        Returns the directory path for the specified dataset split.

        Args:
            split (DatasetSplitType): The dataset split type (e.g., train, validation, test).
        Returns:
            str: The directory path for the specified dataset split.
        """

        return Path(self._storage_dir) / f"shards/{split.value}"

    def get_file_url_provider(self, split: DatasetSplitType) -> FileUrlProvider:
        file_url_provider = FileUrlProvider(
            base_dir=str(self.split_dir(split)), storage_type=self._storage_type
        )
        return file_url_provider

    def split_exists(self, split: DatasetSplitType) -> bool:
        try:
            split_info_path = self.get_file_url_provider(split=split).split_info_path
            if not split_info_path.exists():
                return False

            split_info = SplitInfo.from_file(
                self.get_file_url_provider(split=split).split_info_path
            )
            return split_info.num_examples > 0
        except FileNotFoundError:
            return False
        except Exception as e:
            logger.exception(
                f"Error reading split info for {split.value}: {e}. "
                "Split may be corrupted or not prepared correctly. Repreparing..."
            )
            return False

    def purge_split(self, split: DatasetSplitType) -> None:
        import shutil

        split_dir = self.split_dir(split)
        if split_dir.exists():
            shutil.rmtree(split_dir)

    def _prepare_iterator(self, split_iterator: SplitIterator):
        import tqdm

        return tqdm.tqdm(
            enumerate(iter(split_iterator)),
            f"Writing split to {self._storage_type.value} storage",
        )

    def _extract_write_info(
        self, shard_writer_actors: list["ray.actor.ActorHandle"]
    ) -> list[DatasetShardInfo]:
        """
        Extracts and aggregates write information from a list of shard writer actors.
        This method retrieves the write information from each shard writer actor by
        invoking their `close` method remotely using Ray. It then filters and returns
        only the entries with a positive count.
        Args:
            shard_writer_actors (List[ray.actor.ActorHandle]): A list of Ray actor handles
                representing the shard writer actors.
        Returns:
            List[DatasetShardInfo]: A list of `DatasetShardInfo` objects containing
                aggregated write information with a positive count.
        """
        import ray

        write_info = []
        for actor in shard_writer_actors:
            write_info += ray.get(actor.close.remote())
        write_info = [x for x in write_info if x.nsamples > 0]
        # reset the total number of shards
        for i, shard_info in enumerate(write_info):
            shard_info.shard = i + 1
        return write_info

    def _init_ray(self):
        import ray

        if not ray.is_initialized():
            ray.init(
                num_cpus=self._num_processes,
                local_mode=(self._num_processes == 1),
                runtime_env=_RAY_RUNTIME_ENV,
            )

    def _prepare_shards_and_setup_actors(
        self,
        df: "pd.DataFrame",
        data_model: "type[BaseDataInstance]",
        split: DatasetSplitType,
        file_url_provider: FileUrlProvider,
        preprocess_transform: Callable | None = None,
    ) -> list["ShardWriterActor"]:
        import ray

        if split == DatasetSplitType.train:
            df = df.sample(frac=1).reset_index(drop=True)
            logger.info(
                f"Shuffled dataframe for training split, total samples: {len(df)}"
            )

        actors = []
        total_len = len(df)
        for i in range(self._num_processes):
            start_idx = i * self._max_shard_size
            end_idx = min((i + 1) * self._max_shard_size, total_len)
            shard_df = df.iloc[start_idx:end_idx]
            actor = ShardWriterActor.options(memory=self._max_memory_per_actor).remote(  # type: ignore
                shard_df=shard_df,
                data_model=data_model,
                storage_type=self._storage_type,
                storage_file_pattern=file_url_provider.get_output_file_pattern(
                    process=i
                ),
                max_shard_size=self._max_shard_size,
                preprocess_transform=preprocess_transform,
            )
            actors.append(actor)

        ray.get([actor.load.remote() for actor in actors])
        return actors

    def write_split(
        self,
        split_iterator: SplitIterator,
        preprocess_transform: Callable | None = None,
    ):
        import ray

        try:
            assert isinstance(split_iterator._base_iterator, DeltalakeReader), (
                "Base iterator must be a DeltalakeReader."
            )

            df = split_iterator._base_iterator.df

            self._init_ray()

            split_dir = self.split_dir(split=split_iterator.split)
            logger.info(
                f"Preparing split {split_iterator.split.value} at {split_dir} with {len(split_iterator)} samples."
            )

            file_url_provider = self.get_file_url_provider(split=split_iterator.split)
            actors = self._prepare_shards_and_setup_actors(
                df=df,
                data_model=split_iterator.data_model,
                split=split_iterator.split,
                file_url_provider=file_url_provider,
                preprocess_transform=preprocess_transform,
            )

            write_info = []
            for actor in actors:
                result = ray.get(actor.write_shard.remote())  # type: ignore
                if result is None:
                    continue
                if isinstance(result, list):
                    write_info.extend(result)
                else:
                    write_info.append(result)

            write_info = [x for x in write_info if x.nsamples > 0]
            for i, shard_info in enumerate(write_info):
                shard_info.shard = i + 1
            SplitInfo.from_shard_info_list(write_info).to_file(
                file_url_provider.split_info_path
            )

            ray.get([actor.close.remote() for actor in actors])  # type: ignore

        except Exception as e:
            logger.error(f"Error in write_split: {e}")
            raise
        finally:
            if ray.is_initialized():
                ray.shutdown()

    def read_split(
        self,
        split: DatasetSplitType,
        data_model: type[BaseDataInstance],
        output_transform: Callable | None = None,
        allowed_keys: set[str] | None = None,
    ) -> SplitIterator:
        import webdataset as wds

        from atria_datasets.core.storage.shard_list_datasets import (
            MsgpackShardListDataset,
            TarShardListDataset,
        )
        from atria_datasets.core.storage.utilities import FileStorageType

        if allowed_keys is not None:
            allowed_keys.add("sample_id")
            allowed_keys.add("index")

        split_info = SplitInfo.from_file(
            self.get_file_url_provider(split=split).split_info_path
        )
        if self._storage_type == FileStorageType.MSGPACK:
            assert self._streaming_mode is False, (
                f"Streaming mode is not supported for {FileStorageType.MSGPACK.value} storage type."
            )

            def input_transform(sample):
                filtered_sample = {}
                for key in list(sample.keys()):
                    if allowed_keys is not None and key not in allowed_keys:
                        continue
                    filtered_sample[key] = sample[key]
                return data_model(**filtered_sample)

            return SplitIterator(
                split=split,
                base_iterator=MsgpackShardListDataset(split_info.shardlist),
                input_transform=input_transform,
                output_transform=output_transform,
                data_model=data_model,
            )
        elif self._storage_type == FileStorageType.WEBDATASET:

            def input_transform(sample):
                from atria_datasets.core.storage.utilities import default_decoder

                return data_model(**default_decoder(sample, allowed_keys=allowed_keys))

            if not self._streaming_mode:
                dataset = TarShardListDataset(split_info.shardlist)
            else:
                dataset = wds.WebDataset(
                    [shard.url for shard in split_info.shardlist],
                    resampled=True,
                    shardshuffle=True,
                    cache_dir=self.get_file_url_provider(split=split)._base_dir,
                    nodesplitter=wds.split_by_node,
                )

            return SplitIterator(
                split=split,
                base_iterator=dataset,
                input_transform=input_transform,
                output_transform=output_transform,
                data_model=data_model,
            )
        else:
            raise ValueError(
                f"Unsupported storage type: {self._storage_type.value}. "
                "Supported types are: Msgpack, WebDataset."
            )
