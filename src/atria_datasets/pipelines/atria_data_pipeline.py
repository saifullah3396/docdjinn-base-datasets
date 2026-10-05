"""
Default Data Pipeline Module

This module defines the `DefaultDataPipeline` class, which provides a configurable
data pipeline for training, validation, and testing in machine learning workflows.
It supports dataset splitting, data loading, and batch sampling.

Classes:
    - DefaultDataPipeline: A configurable data pipeline for managing datasets and dataloaders.

Dependencies:
    - ignite.distributed: For distributed training utilities.
    - torch.utils.data: For dataset and dataloader utilities.
    - atria.data: For dataset and batch sampler management.
    - rich.pretty: For pretty-printing configurations.
    - wids: For chunked sampling in web-based datasets.

Author: Your Name (your.email@example.com)
Date: 2025-04-07
Version: 1.0.0
License: MIT
"""

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

import rich
from atria_core.logger.logger import get_logger
from atria_core.transforms.base import DataTransformsDict
from atria_core.types import DatasetSplitType
from atria_core.utilities.repr import RepresentationMixin
from rich.pretty import pretty_repr

from atria_datasets.core.batch_samplers.batch_samplers_dict import BatchSamplersDict
from atria_datasets.core.dataset.atria_dataset import AtriaDataset, DatasetLoadingMode
from atria_datasets.core.dataset_splitters.standard_splitter import StandardSplitter
from atria_datasets.core.storage.utilities import FileStorageType
from atria_datasets.pipelines.utilities import (
    auto_dataloader,
    default_collate,
    mmdet_pseudo_collate,
)

if TYPE_CHECKING:
    from torch.utils.data import DataLoader, Dataset  # type: ignore


logger = get_logger(__name__)


@dataclass
class DataloaderConfig:
    train_batch_size: int = 64
    eval_batch_size: int = 64
    num_workers: int = 4
    pin_memory: bool = True
    drop_last: bool = False

    @property
    def train_config(self) -> dict:
        config = {
            "batch_size": self.train_batch_size,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "drop_last": self.drop_last,
        }
        return ", ".join(f"{key}={value}" for key, value in config.items())

    @property
    def eval_config(self) -> dict:
        config = {
            "batch_size": self.eval_batch_size,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "drop_last": self.drop_last,
        }
        return ", ".join(f"{key}={value}" for key, value in config.items())


class AtriaDataPipeline(RepresentationMixin):
    """
    A configurable data pipeline for managing datasets and dataloaders.

    This class provides functionality for dataset setup, splitting, and creating
    dataloaders for training, validation, and testing phases. It supports distributed
    training and custom batch samplers.

    Attributes:
        _runtime_transforms (Optional[DataTransformsDict]): Runtime transformations to apply to the dataset.
        _storage_enabled (bool): Whether to enable dataset storage.
        _train_dataloader (partial): Builder for the training dataloader.
        _evaluation_dataloader (partial): Builder for the evaluation dataloader.
        _use_validation_set_for_test (bool): Whether to use the validation set for testing.
        _use_train_set_for_test (bool): Whether to use the training set for testing.
        _tar_sampling_chunk_size (int): Chunk size for tar-based datasets.
    """

    def __init__(
        self,
        dataset: AtriaDataset,
        data_dir: str | None = None,
        dataloader_config: DataloaderConfig = DataloaderConfig(
            train_batch_size=64,
            eval_batch_size=64,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
        ),
        batch_samplers: BatchSamplersDict = BatchSamplersDict(),
        preprocess_transforms: DataTransformsDict | None = DataTransformsDict(),
        use_validation_set_for_test: bool = False,
        use_train_set_for_test: bool = False,
        tar_sampling_chunk_size: int = 1000,
        # storage args
        enable_sharded_storage: bool = False,
        shard_storage_type: FileStorageType = FileStorageType.WEBDATASET,
        num_processes: int = 8,
        max_shard_size: int = 100000,
        max_memory_per_actor: int = 500 * 1024 * 1024,
        max_concurrent_tasks_limit: int = 100,
        streaming_mode: bool = False,
        # dataset split args
        dataset_splitting_enabled: bool = False,
        split_ratio: float = 0.9,
        # dataset init args
        access_token: str | None = None,
        overwrite_existing_cached: bool = False,
        overwrite_existing_shards: bool = False,
        # collate_fn
        collate_fn: str = "default_collate",
        # dataset_load_mode
        dataset_load_mode: DatasetLoadingMode = DatasetLoadingMode.in_memory,
    ):
        """
        Initializes the DefaultDataPipeline.

        Args:
            data_dir (Union[str, Path]): The directory containing the dataset.
            preprocess_transforms (Optional[DataTransformsDict]): Preprocessing transformations to apply to the dataset.
            runtime_transforms (Optional[Union[DataTransformsDict, partial]]): Runtime transformations to apply to the dataset.
            allowed_keys (Optional[list[str]]): List of allowed keys for the dataset.
            storage_enabled (bool): Whether to enable dataset storage.
            train_dataloader (partial): Builder for the training dataloader.
            evaluation_dataloader (partial): Builder for the evaluation dataloader.
            use_validation_set_for_test (bool): Whether to use the validation set for testing.
            use_train_set_for_test (bool): Whether to use the training set for testing.
            tar_sampling_chunk_size (int): Chunk size for tar-based datasets.
            sharded_storage_type (FileStorageType): The type of storage to use for the dataset.
            storage_file_key (str): The key for accessing files in the storage.
            preprocessing_batch_size (int): The batch size for preprocessing.
            num_processes (int): The number of processes to use for preprocessing.
            max_shard_size (int): The maximum size of each shard in the storage.
            max_memory_per_actor (int): The maximum memory per actor in bytes.
            max_concurrent_tasks_limit (int): The maximum number of concurrent tasks.
            streaming_mode (bool): Whether to enable streaming mode for the dataset.
            dataset_splitting_enabled (bool): Whether to enable dataset splitting.
            split_ratio (float): The ratio of the training split. Defaults to 0.9.
        """
        self._dataset = dataset
        self._data_dir = data_dir
        self._batch_samplers = batch_samplers
        self._preprocess_transforms = preprocess_transforms
        self._enable_sharded_storage = enable_sharded_storage
        self._dataloader_config = dataloader_config
        self._use_validation_set_for_test = use_validation_set_for_test
        self._use_train_set_for_test = use_train_set_for_test
        self._tar_sampling_chunk_size = tar_sampling_chunk_size
        self._dataset_load_mode = dataset_load_mode

        self._sharded_storage_kwargs = {}
        if self._enable_sharded_storage:
            self._sharded_storage_kwargs = {
                "shard_storage_type": shard_storage_type,
                "num_processes": num_processes,
                "max_shard_size": max_shard_size,
                "max_memory_per_actor": max_memory_per_actor,
                "max_concurrent_tasks_limit": max_concurrent_tasks_limit,
                "streaming_mode": streaming_mode,
            }

        self._dataset_splitter = None
        if dataset_splitting_enabled:
            self._dataset_splitter = StandardSplitter(
                split_ratio=split_ratio, shuffle=True
            )

        assert not (
            self._use_validation_set_for_test and self._use_train_set_for_test
        ), (
            "Only one of use_validation_set_for_test or use_train_set_for_test can be set to True."
        )

        self._access_token = access_token
        self._overwrite_existing_cached = overwrite_existing_cached
        self._overwrite_existing_shards = overwrite_existing_shards

        assert collate_fn in ["default_collate", "mmdet_pseudo_collate"], (
            f"collate_fn must be one of ['collate_fn', 'mmdet_pseudo_collate'], "
            f"but got {collate_fn}"
        )
        if collate_fn == "default_collate":
            self._collate_fn = default_collate
        elif collate_fn == "mmdet_pseudo_collate":
            self._collate_fn = mmdet_pseudo_collate

        self._train_dataloader = partial(
            auto_dataloader,
            batch_size=self._dataloader_config.train_batch_size,
            num_workers=self._dataloader_config.num_workers,
            pin_memory=self._dataloader_config.pin_memory,
            drop_last=self._dataloader_config.drop_last,
        )
        self._evaluation_dataloader = partial(
            auto_dataloader,
            batch_size=self._dataloader_config.eval_batch_size,
            num_workers=self._dataloader_config.num_workers,
            pin_memory=self._dataloader_config.pin_memory,
            drop_last=self._dataloader_config.drop_last,
        )

    @property
    def dataset_metadata(self):
        """
        Retrieves metadata from the dataset.

        Returns:
            Metadata: The metadata of the dataset.

        Raises:
            ValueError: If no dataset is available to extract metadata from.
        """
        return self._dataset.metadata

    def build(
        self,
        split: DatasetSplitType | None = None,
        runtime_transforms: DataTransformsDict = DataTransformsDict(),
        allowed_keys: set[str] | None = None,
    ) -> "AtriaDataPipeline":
        """
        Sets up the datasets for training, validation, and testing.

        Args:
            split (Optional[DatasetSplit]): The dataset split to set up. If None, sets up all splits.
        """
        import ignite.distributed as idist  # type: ignore

        if idist.get_rank() > 0:
            idist.barrier()

        if (
            split == DatasetSplitType.train
            or split is None
            or self._use_validation_set_for_test
            or self._use_train_set_for_test
        ):
            self._dataset.build(
                split=DatasetSplitType.train,
                data_dir=self._data_dir,
                runtime_transforms=runtime_transforms.composed_train,
                preprocess_transforms=self._preprocess_transforms.composed_train
                if self._preprocess_transforms
                else None,
                access_token=self._access_token,
                overwrite_existing_cached=self._overwrite_existing_cached,
                overwrite_existing_shards=self._overwrite_existing_shards,
                allowed_keys=allowed_keys,
                dataset_load_mode=self._dataset_load_mode,
                **self._sharded_storage_kwargs,  # type: ignore
            )
            try:
                self._dataset.build(
                    split=DatasetSplitType.validation,
                    data_dir=self._data_dir,
                    runtime_transforms=runtime_transforms.composed_evaluation,
                    preprocess_transforms=self._preprocess_transforms.composed_evaluation
                    if self._preprocess_transforms
                    else None,
                    access_token=self._access_token,
                    overwrite_existing_cached=self._overwrite_existing_cached,
                    overwrite_existing_shards=self._overwrite_existing_shards,
                    allowed_keys=allowed_keys,
                    dataset_load_mode=self._dataset_load_mode,
                    **self._sharded_storage_kwargs,  # type: ignore
                )
            except ValueError as e:
                if self._dataset_splitter is not None:
                    logger.info(
                        f"Using train/validation sampler [{self._dataset_splitter}] for splitting the "
                        f"dataset with following arguments: {pretty_repr(self._dataset_splitter)}"
                    )
                    assert self._dataset.train is not None, (
                        "Train dataset must be available to split into train and validation sets."
                    )
                    self._dataset.train, self._dataset.validation = (
                        self._dataset_splitter(self._dataset.train)
                    )
                else:
                    logger.warning(
                        f"Unable to load validation split from the dataset: {e}"
                    )

        if split == DatasetSplitType.test or split is None:
            self._dataset.build(
                split=DatasetSplitType.test,
                data_dir=self._data_dir,
                runtime_transforms=runtime_transforms.composed_evaluation,
                preprocess_transforms=self._preprocess_transforms.composed_evaluation
                if self._preprocess_transforms
                else None,
                access_token=self._access_token,
                overwrite_existing_cached=self._overwrite_existing_cached,
                overwrite_existing_shards=self._overwrite_existing_shards,
                allowed_keys=allowed_keys,
                dataset_load_mode=self._dataset_load_mode,
                **self._sharded_storage_kwargs,  # type: ignore
            )
            if self._dataset.test is None:
                if self._dataset.validation is not None:
                    logger.warning(
                        "No test dataset found in the datamodule. Using validation dataset for test runs."
                    )
                    self._dataset.test = self._dataset.validation  # type: ignore
                else:
                    logger.warning(
                        "No test dataset found in the datamodule. Using train dataset for test runs."
                    )
                    self._dataset.test = self._dataset.train  # type: ignore

        if self._use_validation_set_for_test:
            logger.info(
                "Using validation dataset for test runs. "
                "This will override the test dataset."
            )
            self._dataset.test = self._dataset.validation  # type: ignore

        if idist.get_rank() == 0:
            idist.barrier()

        if self._dataset.train is not None:
            logger.info(f"Loaded training dataset:\n{self._dataset.train}")
        if self._dataset.validation is not None:
            logger.info(f"Loaded validation dataset:\n{self._dataset.validation}")
        if self._dataset.test is not None:
            logger.info(f"Loaded test dataset:\n{self._dataset.test}")
        if self.dataset_metadata.dataset_labels is not None:
            logger.info(
                f"Labels found in the dataset:\n{pretty_repr(self.dataset_metadata.dataset_labels)}"
            )
        else:
            logger.warning("No labels found in the dataset.")
        return self

    def train_dataloader(self, shuffle: bool = True, **kwargs) -> "DataLoader":
        """
        Builds and returns the training dataloader.

        Args:
            shuffle (bool): Whether to shuffle the dataset. Defaults to True.
            **kwargs: Additional arguments for the dataloader.

        Returns:
            DataLoader: The training dataloader.
        """
        return self._build_train_dataloader(shuffle=shuffle, **kwargs)

    def validation_dataloader(self, **kwargs) -> "DataLoader | None":
        """
        Builds and returns the validation dataloader.

        Args:
            **kwargs: Additional arguments for the dataloader.

        Returns:
            Optional[DataLoader]: The validation dataloader, or None if no validation dataset is available.
        """
        if self._dataset.validation is None:
            return None
        return self._build_evaluation_dataloader(self._dataset.validation, **kwargs)

    def test_dataloader(self, **kwargs) -> "DataLoader | None":
        """
        Builds and returns the test dataloader.

        Args:
            **kwargs: Additional arguments for the dataloader.

        Returns:
            Optional[DataLoader]: The test dataloader, or None if no test dataset is available.
        """
        if self._dataset.test is None:
            return None
        return self._build_evaluation_dataloader(self._dataset.test, **kwargs)

    def _build_train_dataloader(self, shuffle: bool = True, **kwargs) -> "DataLoader":
        """
        Builds the training DataLoader with the specified configurations.
        Args:
            shuffle (bool, optional): Whether to shuffle the dataset. Defaults to True.
            **kwargs: Additional keyword arguments to customize the DataLoader.
        Returns:
            DataLoader: A PyTorch DataLoader instance configured for training.
        Notes:
            - If `shuffle` is True and the dataset is an instance of `TarShardListDataset`,
                a `ChunkedSampler` is used for sampling. Otherwise, a `RandomSampler` is used.
            - If `shuffle` is False, a `SequentialSampler` is used.
            - The `batch_size` is scaled by the distributed world size if applicable.
            - If a `batch_sampler` is provided in `self.batch_samplers.train`, it must be a
                `partial` initializer that takes a `torch.utils.data.Sampler` as input and
                returns a `torch.utils.data.BatchSampler`.
            - For distributed training (`idist.get_world_size() > 1`), `drop_last` is
                overridden to True to ensure consistent batch sizes across processes.
            - The DataLoader is built using `self.train_dataloader` with the
                specified dataset, sampler, and other configurations.
        """
        import ignite.distributed as idist
        from torch.utils.data import RandomSampler, SequentialSampler
        from wids import ChunkedSampler

        from atria_datasets.core.storage.shard_list_datasets import TarShardListDataset

        if shuffle:
            if isinstance(self._dataset.train, TarShardListDataset):
                # tar shard list dataset is a webdataset based dataset which requires chunked sampler
                sampler = ChunkedSampler(
                    self._dataset.train,
                    shuffle=True,
                    shufflefirst=True,
                    chunksize=self._tar_sampling_chunk_size,
                )
            else:
                sampler = RandomSampler(self._dataset.train)
        else:
            sampler = SequentialSampler(self._dataset.train)

        # we override kwargs here
        kwargs["sampler"] = sampler
        if "batch_size" in kwargs:
            kwargs["batch_size"] = kwargs["batch_size"] * idist.get_world_size()
        elif "batch_size" in self._train_dataloader.keywords:
            kwargs["batch_size"] = (
                self._train_dataloader.keywords["batch_size"] * idist.get_world_size()
            )
        if self._batch_samplers.train is not None:
            assert isinstance(self._batch_samplers.train, partial), (
                "batch_sampler must be a partial initializer which takes the torch.utils.data.Sampler "
                "as input and returns torch.utils.data.BatchSampler."
            )
            kwargs["batch_sampler"] = self._batch_samplers.train(sampler)
        if idist.get_world_size() > 1:
            kwargs["drop_last"] = True
            logger.info("Overriding drop last to True for distributed training.")
        return self._train_dataloader(
            dataset=self._dataset.train, collate_fn=self._collate_fn, **kwargs
        )

    def _build_evaluation_dataloader(
        self, dataset: "Dataset", **kwargs
    ) -> "DataLoader":
        """
        Builds the evaluation DataLoader for the given dataset.
        This method sets up a DataLoader for evaluation purposes, configuring
        the sampler, batch size, and other parameters. It ensures compatibility
        with distributed training setups and handles cases where the dataset
        size is not divisible by the number of processes.
        Args:
            dataset (Dataset): The dataset to be used for evaluation.
            **kwargs: Additional keyword arguments to configure the DataLoader.
        Returns:
            DataLoader: A DataLoader instance configured for evaluation.
        Raises:
            AssertionError: If the batch sampler is not a partial initializer
                            that takes a torch.utils.data.Sampler as input and
                            returns a torch.utils.data.BatchSampler.
        Notes:
            - If the dataset size is not divisible by the number of processes
                in a distributed setup, duplicate entries are added to ensure
                equal number of samples per process.
            - The batch size is adjusted based on the world size in distributed
                training.
        """
        import ignite.distributed as idist  # type: ignore
        from torch.utils.data import SequentialSampler  # type: ignore

        sampler = SequentialSampler(dataset)

        # we override kwargs here
        kwargs["sampler"] = sampler

        # get the overrides
        if "batch_size" in kwargs:
            kwargs["batch_size"] = kwargs["batch_size"] * idist.get_world_size()
        elif "batch_size" in self._evaluation_dataloader.keywords:
            kwargs["batch_size"] = (
                self._evaluation_dataloader.keywords["batch_size"]
                * idist.get_world_size()
            )
        if idist.get_world_size() > 1:
            if len(dataset) % idist.get_world_size() != 0:
                logger.warning(
                    "Enabling distributed evaluation with an eval dataset not divisible by process number. "
                    "This will slightly alter validation results as extra duplicate entries are added to achieve "
                    "equal num of samples per-process."
                )
        # properly setup the kwargs
        if self._batch_samplers.evaluation is not None:
            assert isinstance(self._batch_samplers.evaluation, partial), (
                "batch_sampler must be a partial initializer which takes the torch.utils.data.Sampler "
                "as input and returns torch.utils.data.BatchSampler."
            )
            kwargs["batch_sampler"] = self._batch_samplers.evaluation(sampler)
        return self._evaluation_dataloader(
            dataset,
            collate_fn=self._collate_fn,
            shuffle=False,
            drop_last=False,
            **kwargs,
        )

    def __rich_repr__(self) -> rich.pretty.RichReprResult:
        """
        Generates a rich representation of the object.

        Yields:
            RichReprResult: A generator of key-value pairs or values for the object's attributes.
        """
        for name, field_repr in self.__dict__.items():
            if name is None:
                yield field_repr
            else:
                yield name, field_repr
        yield "metadata", self.dataset_metadata
