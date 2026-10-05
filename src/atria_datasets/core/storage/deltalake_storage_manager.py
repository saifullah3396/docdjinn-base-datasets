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
    - atria_datasets.core.datasets: For dataset-related classes and metadata.
    - atria_datasets.core.storage.dataset_storage_manager: For the base storage manager class.
    - atria_datasets.core.storage.shard_writer_actor: For managing shard writers.
    - atria_datasets.core.storage.utilities: For utility functions and constants.

Author: Your Name (your.email@example.com)
Date: 2025-04-07
Version: 1.0.0
License: MIT
"""

import multiprocessing as mp
import pickle
import queue
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import deltalake
import pandas as pd
import pyarrow as pa
import tqdm
from atria_core.logger.logger import get_logger

from atria_datasets.core.dataset.atria_dataset import DatasetLoadingMode
from atria_datasets.core.storage.deltalake_reader import DeltalakeReader

if TYPE_CHECKING:
    from atria_core.types import BaseDataInstance, DatasetSplitType

    from atria_datasets.core.dataset.atria_dataset import SplitIterator


logger = get_logger(__name__)


class SerializerWorker:
    def __init__(
        self,
        write_dir: str,
        split: str,
        tf: Callable,
        process_id: int = 0,
        max_shard_size: int = 100000,
        write_binary_tars: bool = True,
    ):
        """
        Initialize the SerializerWorker with a transformation function.
        """
        import webdataset as wds

        self._process_id = process_id
        self._tf = tf
        self._wds_writer: wds.ShardWriter | None = None
        self._current_shard = 0
        self._offset = 0
        self._current_shard_path = None
        self._write_dir = write_dir
        self._split = split
        self._max_shard_size = max_shard_size
        self._write_binary_tars = write_binary_tars

    def __call__(self, inputs: Any) -> Any:
        """
        Call the transformation function with the provided arguments.
        """

        import webdataset as wds
        from atria_core.types import BaseDataInstance

        if self._write_binary_tars and self._wds_writer is None:
            file_path = (
                Path(self._write_dir)
                / "shards"
                / self._split
                / f"{self._process_id:06d}-%06d.tar"
            )
            file_path.parent.mkdir(parents=True, exist_ok=True)
            self._wds_writer = wds.ShardWriter(
                str(file_path), maxcount=self._max_shard_size, verbose=0
            )
            self._current_shard = self._wds_writer.shard
            self._current_shard_path = (
                Path("shards") / self._split / Path(self._wds_writer.fname).name
            )
        if self._write_binary_tars and self._current_shard != self._wds_writer.shard:
            self._current_shard = self._wds_writer.shard
            self._current_shard_path = (
                Path("shards") / self._split / Path(self._wds_writer.fname).name
            )
            self._offset = 0

        sample: BaseDataInstance = self._tf(*inputs)
        if self._wds_writer is not None:
            sample_row = sample.to_row()
            for key in sample_row:
                if "file_path" in key:
                    file_path_key = key
                    content_key = key.replace("file_path", "content")
                    assert content_key in sample_row, (
                        f"Column '{content_key}' not found in sample row. Expected 'content' column for the corresponding "
                        f"file path '{file_path_key}'."
                    )
                    content = sample_row[content_key]
                    if content is None and sample_row[file_path_key] is None:
                        # nothing to write, skip
                        continue
                    assert content is not None, (
                        f"Content for key '{content_key}' is None. Expected a valid binary loaded content after load()."
                    )
                    tar_content = {"__key__": str(sample.key), content_key: content}
                    self._wds_writer.write(tar_content)
                    member = self._wds_writer.tarstream.tarstream.members[-1]
                    sample_row[file_path_key] = (
                        f"{self._current_shard_path}?offset={self._offset + 1536}&length={member.size}"
                    )
                    sample_row[content_key] = None
                    self._offset = self._wds_writer.tarstream.tarstream.offset
            return sample_row
        else:
            return sample.to_row()

    def close(self) -> None:
        """
        Close the WebDataset writer if it is initialized.
        """
        if self._wds_writer is not None:
            self._wds_writer.close()
            self._wds_writer = None


class ShardedDeltalakeStorageWriter(ABC):
    def __init__(self, manager, split_iterator, split_dir, iterator, total):
        self.manager = manager
        self.split_iterator = split_iterator
        self.split_dir = split_dir
        self.iterator = iterator
        self.total = total

        self.batch = []
        self.first_batch = True
        self.write_batch_size = None

    def _write_batch(self):
        logger.info(f"Writing batch of size {len(self.batch)} to storage.")
        mode = "overwrite" if self.first_batch else "append"
        deltalake.write_deltalake(
            self.split_dir,
            pa.Table.from_pylist(
                self.batch, schema=self.split_iterator.data_model.pa_schema()
            ),
            mode=mode,
        )
        self.first_batch = False
        self.batch.clear()

    def _handle_batch_write(self, result):
        self.batch.append(result)

        if self.write_batch_size is None:
            self.write_batch_size = self.manager._max_memory // len(
                pickle.dumps(result)
            )
            logger.info(
                f"Setting write batch size to {self.write_batch_size} based on max memory {self.manager._max_memory // 1000_000} MB"
            )

        if len(self.batch) >= self.write_batch_size:
            self._write_batch()

    @abstractmethod
    def write(self):
        pass


class SerialStorageWriter(ShardedDeltalakeStorageWriter):
    def write(self):
        worker = SerializerWorker(
            write_dir=self.manager._storage_dir / self.manager._config_name,
            split=self.split_iterator.split.value,
            tf=self.split_iterator._tf,
            write_binary_tars=True,
        )

        for result in tqdm.tqdm(
            map(worker, self.iterator),
            desc="Writing to Deltalake",
            total=self.total,
            unit="rows",
        ):
            self._handle_batch_write(result)

        if self.batch:
            self._write_batch()

        worker.close()


class ProducerWorker(mp.Process):
    def __init__(self, input_queue, output_queue, serializer_kwargs, **kwargs):
        super().__init__(**kwargs)
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.serializer_kwargs = serializer_kwargs

    def run(self):
        serializer = SerializerWorker(
            process_id=self._identity[0], **self.serializer_kwargs
        )
        while True:
            item = self.input_queue.get()
            if item == "STOP":
                break
            try:
                result = serializer(item)
                while True:
                    try:
                        self.output_queue.put(result, timeout=1)
                        break
                    except queue.Full:
                        continue
            except Exception as e:
                self.output_queue.put(e)
        serializer.close()
        self.output_queue.put("STOP")


class ParallelStorageWriter(ShardedDeltalakeStorageWriter):
    def write(self):
        input_queue = mp.Queue(maxsize=100)
        output_queue = mp.Queue(maxsize=100)
        serializer_kwargs = {
            "write_dir": self.manager._storage_dir / self.manager._config_name,
            "split": self.split_iterator.split.value,
            "tf": self.split_iterator._tf,
            "write_binary_tars": True,
        }

        num_producers = self.manager._num_processes - 1
        producers = [
            ProducerWorker(input_queue, output_queue, serializer_kwargs)
            for _ in range(num_producers)
        ]

        try:
            for p in producers:
                p.start()

            for item in tqdm.tqdm(
                self.iterator,
                desc="Writing to Deltalake",
                total=self.total,
                unit="rows",
            ):
                while not output_queue.empty():
                    result = output_queue.get()
                    if result == "STOP":
                        continue
                    if isinstance(result, Exception):
                        logger.error(f"Error in producer: {result}")
                        raise result
                    self._handle_batch_write(result)

                input_queue.put(item)

            for _ in producers:
                input_queue.put("STOP")

            stops_received = 0
            while stops_received < num_producers:
                result = output_queue.get()
                if isinstance(result, Exception):
                    logger.error(f"Error in producer: {result}")
                    raise result
                if result == "STOP":
                    stops_received += 1
                else:
                    self._handle_batch_write(result)

            if self.batch:
                self._write_batch()
        except Exception as e:
            logger.error(f"Error during parallel writing: {e}")
            for p in producers:
                p.terminate()
            raise e
        finally:
            for p in producers:
                p.join(timeout=5)
                if p.is_alive():
                    logger.warning(f"Producer {p.pid} did not shut down cleanly.")


class DeltalakeStorageManager:
    """
    A class for managing dataset storage in deltalake format
    """

    def __init__(
        self,
        storage_dir: str,
        config_name: str,
        num_processes: int = 8,
        max_memory: int = 1000_000_000,
        preprocess_transform: Callable | None = None,
    ):
        self._storage_dir = Path(storage_dir)
        self._config_name = config_name
        self._num_processes = num_processes
        self._max_memory = max_memory
        self._preprocess_transform = preprocess_transform

        if not self._storage_dir.exists():
            self._storage_dir.mkdir(parents=True, exist_ok=True)
        assert self._storage_dir.is_dir(), (
            f"Storage directory {self._storage_dir} must be a directory."
        )
        (self._storage_dir / config_name).mkdir(parents=True, exist_ok=True)

    def split_dir(self, split: "DatasetSplitType") -> Path:
        return Path(self._storage_dir) / f"{self._config_name}/delta/{split.value}"

    def dataset_exists(self) -> bool:
        return (Path(self._storage_dir) / f"{self._config_name}/delta/").exists()

    def split_exists(self, split: "DatasetSplitType") -> bool:
        return self.split_dir(split).exists()

    def purge_split(self, split: "DatasetSplitType") -> None:
        import shutil

        split_dir = self.split_dir(split)
        if split_dir.exists():
            logger.info(
                f"Purging dataset split {split.value} from storage {split_dir}."
            )
            shutil.rmtree(split_dir)

    def write_split(self, split_iterator: "SplitIterator") -> None:
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
        split: "DatasetSplitType",
        data_model: "BaseDataInstance",
        output_transform: Callable | None = None,
        allowed_keys: set[str] | None = None,
        streaming_mode: bool = False,
    ) -> "SplitIterator":
        from atria_datasets.core.dataset.atria_dataset import SplitIterator

        if not self.split_exists(split):
            raise RuntimeError(
                f"Dataset split {split.value} not prepared. Please call `write_split()` first."
            )

        if allowed_keys is not None:
            allowed_keys.add("index")
            allowed_keys.add("sample_id")

        return SplitIterator(
            split=split,
            base_iterator=DeltalakeReader.from_mode(
                table_path=str(self.split_dir(split=split)),
                storage_dir=self._storage_dir,
                config_name=self._config_name,
                data_model=data_model,
                mode=DatasetLoadingMode.local_streaming
                if streaming_mode
                else DatasetLoadingMode.in_memory,
                allowed_keys=allowed_keys,
            ),
            output_transform=output_transform,
            data_model=data_model,
        )

    def get_splits(self) -> list["DatasetSplitType"]:
        """
        Get a list of available dataset splits in the storage directory.

        Returns:
            list[DatasetSplitType]: A list of available dataset splits.
        """
        from atria_core.types import DatasetSplitType

        return [split for split in DatasetSplitType if self.split_exists(split)]

    def prepare_split_files(self, data_dir: str) -> list[tuple[str, str]]:
        import deltalake
        from atria_core.types import DatasetSplitType

        # Get all delta files in the storage directory
        delta_files = list(
            (self._storage_dir / self._config_name / "delta").glob("**/*.*")
        )

        # Create a list of tuples (source, target) for delta files
        files_src_tgt = {
            (str(f), str(f.relative_to(self._storage_dir)))
            for f in delta_files
            if f.is_file()
        }

        def map_file_path(file_path):
            if pd.isna(file_path):
                return None  # skip NaN
            parsed = urlparse(file_path)
            path = parsed.path

            if path.startswith("shards/"):
                tgt = str(Path(self._config_name) / path)
                files_src_tgt.add(
                    (str(Path(self._storage_dir) / self._config_name / path), tgt)
                )
            else:
                tgt = str(Path(self._config_name) / path)
                files_src_tgt.add((str(Path(data_dir) / path), tgt))
            return tgt

        for split in list(DatasetSplitType):
            if not self.split_exists(split):
                continue

            dt = deltalake.DeltaTable(self.split_dir(split=split))
            all_columns = [f.name for f in dt.schema().fields]

            # Find all file_path columns (case-insensitive)
            file_path_cols = [col for col in all_columns if "file_path" in col.lower()]
            content_cols = [
                file_path_col.replace("file_path", "content")
                for file_path_col in file_path_cols
            ]
            dataframe = dt.to_pandas(columns=content_cols + file_path_cols)
            for file_path_column, content_column in zip(
                file_path_cols, content_cols, strict=True
            ):
                if not dataframe[content_column].dropna().empty:
                    continue
                file_path_col_data = dataframe[file_path_column].dropna()
                if file_path_col_data.empty:
                    continue
                file_path_col_data.apply(map_file_path)
        return files_src_tgt

    def _write(self, split_iterator: "SplitIterator") -> Path:
        import itertools

        split_dir = self.split_dir(split=split_iterator.split)
        logger.info(
            f"Preprocessing dataset split {split_iterator.split.value} to cached deltalake storage {split_dir}"
        )
        split_iterator.disable_tf()

        try:
            total = len(split_iterator)
        except RuntimeError:
            total = None

        iterator = iter(split_iterator)
        if total is not None:
            iterator = itertools.islice(iterator, total)

        if self._num_processes <= 1:
            writer = SerialStorageWriter(
                self, split_iterator, split_dir, iterator, total
            )
        else:
            writer = ParallelStorageWriter(
                self, split_iterator, split_dir, iterator, total
            )

        writer.write()
        split_iterator.enable_tf()
        return split_dir
