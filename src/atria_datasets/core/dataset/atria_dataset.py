"""
Atria Dataset Module

This module defines the `AtriaDataset` class and its specialized subclasses, which serve as
base classes for datasets used in the Atria application. It provides comprehensive functionality
for managing dataset splits, configurations, metadata, runtime transformations, and storage.

Classes:
    - AtriaDataset: Generic base class for datasets in the Atria application
    - AtriaImageDataset: Specialized dataset class for image data
    - AtriaDocumentDataset: Specialized dataset class for document data

Key Features:
    - Multi-split dataset management (train/validation/test)
    - Flexible storage backends (DeltaLake, sharded files)
    - Runtime and preprocessing transformations
    - Download management for remote datasets
    - Hub integration for dataset sharing
    - Configurable caching and optimization

Author: Your Name (your.email@example.com)
Date: 2025-04-07
Version: 1.0.0
License: MIT
"""

import enum
import hashlib
import json
from abc import abstractmethod
from collections.abc import Callable, Generator, Sequence
from functools import cached_property
from pathlib import Path
from typing import Any, Generic, Self

from atria_core.logger import get_logger
from atria_core.types import (
    BaseDataInstance,
    DatasetMetadata,
    DatasetSplitType,
    DocumentInstance,
    ImageInstance,
)
from atria_core.utilities.repr import RepresentationMixin
from pydantic import BaseModel, ConfigDict

from atria_datasets.core.constants import _DEFAULT_DOWNLOAD_PATH
from atria_datasets.core.dataset.split_iterator import SplitIterator
from atria_datasets.core.storage.utilities import FileStorageType
from atria_datasets.core.typing.common import T_BaseDataInstance

logger = get_logger(__name__)


class DefaultOutputTransformer:
    def __init__(self, data_dir: str, max_cache_image_size: int = 1024):
        self._data_dir = data_dir
        self._max_cache_image_size = max_cache_image_size

    def __call__(self, sample: Any | T_BaseDataInstance) -> T_BaseDataInstance:
        assert isinstance(sample, BaseDataInstance), (
            f"Expected sample to be a BaseDataInstance, got {type(sample)}"
        )
        if hasattr(sample, "image") and sample.image is not None:
            sample.image.resize_with_aspect_ratio(max_size=self._max_cache_image_size)

        return sample.to_relative_file_paths(data_dir=self._data_dir)


class DatasetLoadingMode(str, enum.Enum):
    """
    Enum to represent the streaming mode of the dataset.

    Attributes:
        LOCAL: Dataset is downloaded and stored locally.
        STREAMING: Dataset is streamed directly from the Atria Hub.
    """

    in_memory = "in_memory"
    local_streaming = "local_streaming"
    online_streaming = "online_streaming"


class AtriaDatasetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset_name: str | None = None
    config_name: str = "default"
    max_train_samples: int | None = None
    max_validation_samples: int | None = None
    max_test_samples: int | None = None


class DatasetConfigMixin:
    __config_cls__: type[AtriaDatasetConfig]

    def __init__(self, **kwargs):
        config_cls = getattr(self.__class__, "__config_cls__", None)
        assert issubclass(config_cls, AtriaDatasetConfig), (
            f"{self.__class__.__name__} must define a __config_cls__ attribute "
            "that is a subclass of AtriaDatasetConfig."
        )
        self._config = config_cls(**kwargs)
        if self._config.dataset_name is None:
            self._config.dataset_name = self.__class__.__name__.lower()
        super().__init__()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # Validate presence of Config at class definition time
        if not hasattr(cls, "__config_cls__"):
            raise TypeError(
                f"{cls.__name__} must define a nested `__config_cls__` class."
            )

        if not issubclass(cls.__config_cls__, AtriaDatasetConfig):
            raise TypeError(
                f"{cls.__name__}.Config must subclass pydantic.AtriaDatasetConfig. Got {cls.__config_cls__} instead."
            )

    def prepare_build_config(self):
        from hydra_zen import builds
        from omegaconf import OmegaConf

        if self.__config_cls__ is None:
            raise TypeError(
                f"{self.__class__.__name__} must define a __config_cls__ attribute."
            )
        init_fields = {
            k: getattr(self._config, k) for k in self._config.__class__.model_fields
        }
        return OmegaConf.to_container(
            OmegaConf.create(
                builds(self.__class__, populate_full_signature=True, **init_fields)
            )
        )

    @cached_property
    def config(self) -> AtriaDatasetConfig:
        return self._config

    @cached_property
    def build_config(self) -> AtriaDatasetConfig:
        return self.prepare_build_config()

    @cached_property
    def config_hash(self) -> str:
        """
        Hash of the dataset configuration for versioning.

        Returns:
            8-character hash string based on configuration content
        """

        return hashlib.sha256(
            json.dumps(self.build_config, sort_keys=True).encode()
        ).hexdigest()[:8]


class AtriaDataset(
    Generic[T_BaseDataInstance], RepresentationMixin, DatasetConfigMixin
):
    """
    Generic base class for datasets in the Atria application.

    This class provides a comprehensive framework for managing datasets with support for:
    - Multiple data splits (train/validation/test)
    - Flexible storage backends (DeltaLake, sharded files)
    - Download management for remote datasets
    - Runtime and preprocessing transformations
    - Dataset versioning and configuration management
    - Hub integration for dataset sharing

    Type Parameters:
        T_BaseDataInstance: The type of data instances this dataset contains
            (must inherit from BaseDataInstance)

    Attributes:
        __data_model__: The data model class used for type validation
        __default_config_path__: Default path for dataset configuration files
        __default_metadata_path__: Default path for dataset metadata files
        __repr_fields__: Fields included in string representation

    Example:
        ```python
        # Create a custom dataset
        class MyDataset(AtriaDataset[DocumentInstance]):
            def _split_configs(self, data_dir: str) -> list[SplitConfig]:
                return [SplitConfig(split=DatasetSplitType.train, gen_kwargs={})]

            def _split_iterator(self, split: DatasetSplitType, **kwargs):
                # Return iterator for the split
                pass


        # Load and use dataset
        dataset = MyDataset(dataset_name="my_dataset")
        dataset.build_split(DatasetSplitType.train)
        for sample in dataset.train:
            print(sample)
        ```
    """

    __abstract__ = True
    __default_config_name__ = "default"
    __requires_access_token__ = False
    __extract_downloads__ = True
    __data_model__: type[T_BaseDataInstance] = None
    __default_config_path__ = "conf/dataset/{config_name}.yaml"
    __default_metadata_path__ = "metadata.yaml"
    __repr_fields__ = ["data_model", "data_dir", "train", "validation", "test"]
    __config_cls__ = AtriaDatasetConfig

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # config is now available at self._config
        if self.config.config_name == self.__default_config_name__:
            self.config.config_name = f"{self.config.config_name}-{self.config_hash}"

        self._config_path = self.__default_config_path__.format(
            config_name=self.config.config_name
        )

        self._downloaded_files: dict[str, Path] = {}
        self._split_iterators: dict[DatasetSplitType, SplitIterator] = {}
        self._downloads_prepared: bool = False
        self._sharded_splits_prepared: bool = False

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        if "__abstract__" in cls.__dict__ and cls.__dict__["__abstract__"]:
            return

        data_model = cls.__data_model__
        if data_model is None:
            raise TypeError(
                f"Class '{cls.__name__}' must define a __data_model__ attribute "
                "to specify the type of data instances."
            )
        if not issubclass(data_model, BaseDataInstance):
            raise TypeError(
                f"Class '{cls.__name__}.__data_model__' must be a type, "
                f"got {type(data_model).__name__}: {data_model}"
            )
        assert isinstance(cls.__requires_access_token__, bool), (
            f"Class '{cls.__name__}' must define __requires_access_token__ as a boolean."
        )
        assert isinstance(cls.__extract_downloads__, bool), (
            f"Class '{cls.__name__}' must define __extract_downloads__ as a boolean."
        )

    # ==================== Public Properties ====================

    @property
    def metadata(self) -> DatasetMetadata:
        """Dataset metadata containing description, version, and other information."""
        return self._metadata()

    @property
    def data_model(self) -> type[T_BaseDataInstance]:
        """The data model class used for type validation and instantiation."""
        return self.__data_model__

    # ==================== Split Properties ====================

    @property
    def train(self) -> SplitIterator[T_BaseDataInstance] | None:
        """Training split iterator. Returns None if training split is not available."""
        return self._split_iterators.get(DatasetSplitType.train, None)

    @train.setter
    def train(self, value: SplitIterator[T_BaseDataInstance]) -> None:
        """Set the training split iterator."""
        self._split_iterators[DatasetSplitType.train] = value

    @property
    def validation(self) -> SplitIterator[T_BaseDataInstance] | None:
        """Validation split iterator. Returns None if validation split is not available."""
        return self._split_iterators.get(DatasetSplitType.validation, None)

    @validation.setter
    def validation(self, value: SplitIterator[T_BaseDataInstance]) -> None:
        """Set the validation split iterator."""
        self._split_iterators[DatasetSplitType.validation] = value

    @property
    def test(self) -> SplitIterator[T_BaseDataInstance] | None:
        """Test split iterator. Returns None if test split is not available."""
        return self._split_iterators.get(DatasetSplitType.test, None)

    @test.setter
    def test(self, value: SplitIterator[T_BaseDataInstance]) -> None:
        """Set the test split iterator."""
        self._split_iterators[DatasetSplitType.test] = value

    # ==================== Class Methods ====================

    @classmethod
    def load_from_registry(
        cls,
        name: str,
        data_dir: str | None = None,
        provider: str | None = None,
        preprocess_transform: Callable | None = None,
        access_token: str | None = None,
        dataset_load_mode: DatasetLoadingMode = DatasetLoadingMode.local_streaming,
        overwrite_existing_cached: bool = False,
        allowed_keys: set[str] | None = None,
        num_processes: int = 8,
        cached_storage_type: FileStorageType = FileStorageType.MSGPACK,
        enable_cached_splits: bool = True,
        cache_artifacts: bool = True,
        build_kwargs: dict[str, Any] | None = None,
    ) -> Self:  # noqa: F821
        """
        Load a dataset from the Atria registry.

        Args:
            name: Dataset name, optionally with config (e.g., "dataset/config")
            data_dir: Custom data directory path
            provider: Registry provider name
            preprocess_transform: Transform function applied during preprocessing
            access_token: Authentication token for private datasets
            overwrite_existing_cached: Whether to overwrite cached data
            allowed_keys: Set of allowed keys to filter data
            build_kwargs: Additional arguments for dataset construction

        Returns:
            Loaded and configured dataset instance

        Raises:
            AssertionError: If loaded dataset's data model doesn't match expected type
            ImportError: If registry dependencies are not available
        """
        from atria_datasets import DATASET

        logger.info(f"Loading dataset {name} from registry.")
        build_kwargs = build_kwargs or {}
        dataset: AtriaDataset[T_BaseDataInstance] = DATASET.load_from_registry(
            module_name=f"{name}",
            provider=provider,
            return_config=False,
            **build_kwargs,
        )
        dataset.build(
            data_dir=data_dir,
            preprocess_transform=preprocess_transform,
            access_token=access_token,
            overwrite_existing_cached=overwrite_existing_cached,
            dataset_load_mode=dataset_load_mode,
            allowed_keys=allowed_keys,
            num_processes=num_processes,
            cached_storage_type=cached_storage_type,
            enable_cached_splits=enable_cached_splits,
            cache_artifacts=cache_artifacts,
        )
        return dataset

    def build(
        self,
        data_dir: str,
        split: DatasetSplitType | None = None,
        runtime_transforms: Callable | None = None,
        preprocess_transform: Callable | None = None,
        access_token: str | None = None,
        dataset_load_mode: DatasetLoadingMode = DatasetLoadingMode.local_streaming,
        overwrite_existing_cached: bool = False,
        allowed_keys: set[str] | None = None,
        num_processes: int = 8,
        cached_storage_type: FileStorageType = FileStorageType.MSGPACK,
        enable_cached_splits: bool = True,
        cache_artifacts: bool = True,
    ) -> None:
        """
        Build and prepare a dataset split for use.

        This method handles the complete pipeline of dataset preparation including:
        - Setting up data directories
        - Preparing cached or uncached splits
        - Setting up sharded storage if requested
        - Applying runtime transformations

        Args:
            data_dir: Custom data directory (overrides default)
            config_name: Configuration name for the dataset
            runtime_transforms: Transform function applied at runtime
            preprocess_transform: Transform function applied during preprocessing
            access_token: Authentication token for private datasets
            overwrite_existing_cached: Whether to overwrite existing cached data
            allowed_keys: Filter to include only specified keys
            enable_cached_splits: Whether to use cached storage (DeltaLake)
        """
        from atria_core.constants import _DEFAULT_ATRIA_DATASETS_CACHE_DIR

        if data_dir is None:
            data_dir = _DEFAULT_ATRIA_DATASETS_CACHE_DIR / self.config.dataset_name
            logger.warning(
                f"No data_dir provided. Using default cache directory:\n{data_dir}"
            )
        self._data_dir = self._validate_data_dir(data_dir)
        self._split = split
        self._num_processes = num_processes
        self._allowed_keys = allowed_keys
        self._dataset_load_mode = dataset_load_mode
        self._storage_dir = Path(data_dir) / "storage"
        self._overwrite_existing_cached = overwrite_existing_cached
        logger.info(f"Setting dataset storage directory: {self._storage_dir}")

        # Prepare splits based on caching preference
        if enable_cached_splits:
            self._prepare_cached_splits(
                access_token=access_token,
                cached_storage_type=cached_storage_type,
                preprocess_transform=preprocess_transform,
                cache_artifacts=cache_artifacts,
            )
        else:
            # first prepare uncached splits
            self._prepare_splits(access_token=access_token)

        # Apply runtime transformations
        if runtime_transforms is not None:
            for split, split_iterator in self._split_iterators.items():
                if self._split is not None and split != self._split:
                    continue
                split_iterator.output_transform = runtime_transforms

    def upload_to_hub(
        self,
        name: str | None = None,
        branch: str = "main",
        is_public: bool = False,
        overwrite_existing: bool = False,
    ) -> None:
        """
        Upload the dataset to Atria Hub for sharing and collaboration.

        Args:
            name: Dataset name on the hub (defaults to current dataset name)
            branch: Branch name (defaults to config-hash format)
            is_public: Whether to make the dataset publicly accessible

        Raises:
            ImportError: If atria_hub package is not installed
            Exception: If upload fails for any reason
        """
        try:
            from atria_hub.hub import AtriaHub
            from atriax_client.models.data_instance_type import DataInstanceType

            if name is None:
                name = self.config.dataset_name.replace("_", "-")

            def data_model_to_instance_type(
                data_model: type[T_BaseDataInstance],
            ) -> DataInstanceType:
                """Convert data model class to hub instance type."""
                if data_model == DocumentInstance:
                    return DataInstanceType.DOCUMENT_INSTANCE
                elif data_model == ImageInstance:
                    return DataInstanceType.IMAGE_INSTANCE
                else:
                    raise ValueError(f"Unsupported data model: {data_model}")

            hub = AtriaHub().initialize()
            dataset = hub.datasets.get_or_create(
                username=hub.auth.username,
                name=name,
                default_branch=branch,
                description=self.metadata.description,
                data_instance_type=data_model_to_instance_type(self.data_model),
                is_public=is_public,
            )

            logger.info(
                f"Uploading dataset to hub with name {hub.auth.username}/{name} "
                f"on branch {branch} and config_name {self.config.config_name}."
            )
            hub.datasets.upload_files(
                dataset=dataset,
                branch=branch,
                config_dir=self.config.config_name,
                dataset_files=self.prepare_dataset_files_from_dir(),
                overwrite_existing=overwrite_existing,
            )
            logger.info(
                f"Dataset {name} uploaded successfully to branch {branch}. "
                f"You can load it with name '{hub.auth.username}/{name}' and config_name '{self.config.config_name}'."
            )
        except ImportError as e:
            if e.path.startswith("atria_hub"):
                raise ImportError(
                    "The 'atria_hub' package is required to load datasets from the hub. "
                    "Please install it using 'uv add https://github.com/saifullah3396/atria_hub'."
                )
            raise e
        except Exception as e:
            logger.error(f"Failed to upload dataset to hub: {e}")
            raise

    def get_max_split_samples(self, split: DatasetSplitType) -> int | None:
        """
        Get the maximum number of samples allowed for a specific split.

        Args:
            split: The dataset split to check

        Returns:
            Maximum number of samples, or None if no limit is set
        """
        limits = {
            DatasetSplitType.train: self.config.max_train_samples,
            DatasetSplitType.validation: self.config.max_validation_samples,
            DatasetSplitType.test: self.config.max_test_samples,
        }
        return limits.get(split)

    def prepare_downloads(self, data_dir: str, access_token: str | None = None) -> None:
        """
        Download and prepare remote dataset files.

        Args:
            data_dir: Directory to download files to
            access_token: Authentication token for private resources

        Note:
            This method is idempotent - subsequent calls will not re-download files.
        """
        if self.__requires_access_token__ and access_token is None:
            logger.warning(
                "access_token must be passed to download this dataset. "
                f"See `{self.metadata.homepage}` for instructions to get the access token"
            )
            return
        if not self._downloads_prepared:
            if self._custom_download.__func__ is not AtriaDataset._custom_download:
                self._downloaded_files = self._custom_download(data_dir, access_token)
            else:
                from atria_datasets.core.download_manager.download_manager import (
                    DownloadManager,
                )

                download_dir = Path(data_dir) / _DEFAULT_DOWNLOAD_PATH
                download_dir.mkdir(parents=True, exist_ok=True)

                download_manager = DownloadManager(
                    data_dir=Path(data_dir), download_dir=download_dir
                )

                download_urls = self._download_urls()
                if len(download_urls) > 0:
                    self._downloaded_files = download_manager.download_and_extract(
                        download_urls, extract=self.__extract_downloads__
                    )

            self._downloads_prepared = True

    def save_dataset_info(self, storage_dir: str) -> None:
        """
        Save dataset configuration and metadata to files.

        Creates YAML files containing:
        - Dataset configuration (config.yaml)
        - Dataset metadata (metadata.yaml)
        """
        import yaml

        def write_yaml_file(file_path: Path, data: dict) -> None:
            """Write data to YAML file, creating directories as needed."""
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, "w") as f:
                yaml.dump(data, f, sort_keys=False)

        # Save configuration
        config_file_path = Path(storage_dir) / self._config_path
        logger.info("Saving dataset configuration to %s", config_file_path)
        write_yaml_file(config_file_path, self.build_config)

        metadata_file_path = Path(storage_dir) / self.__default_metadata_path__
        logger.info("Saving dataset metadata to %s", metadata_file_path)
        write_yaml_file(metadata_file_path, self.metadata.model_dump())

    def prepare_dataset_files_from_dir(self) -> list[tuple[str, str]]:
        """
        Get list of dataset files for upload or transfer operations.

        Args:
            storage_dir: Storage directory to scan (defaults to current storage directory)

        Returns:
            List of (local_path, relative_path) tuples for all dataset files
        """

        from atria_datasets.core.storage.deltalake_storage_manager import (
            DeltalakeStorageManager,
        )

        deltalake_storage_manager = DeltalakeStorageManager(
            storage_dir=self._storage_dir, config_name=self.config.config_name
        )

        # Collect split files
        dataset_files = [
            (
                str(Path(self._storage_dir) / self.__default_metadata_path__),
                self.__default_metadata_path__,
            ),
            (str(Path(self._storage_dir) / self._config_path), self._config_path),
        ]

        # get all split files from deltalake storage manager
        dataset_files.extend(
            deltalake_storage_manager.prepare_split_files(data_dir=self._data_dir)
        )

        return dataset_files

    # ==================== Private Methods ====================

    def _validate_data_dir(self, data_dir: str | Path) -> Path:
        """
        Validate and create data directory if needed.

        Args:
            data_dir: Directory path to validate

        Returns:
            Validated Path object

        Raises:
            AssertionError: If path exists but is not a directory
        """
        data_dir = Path(data_dir)

        if data_dir.exists():
            assert data_dir.is_dir(), (
                f"Data directory `{data_dir.absolute()}` exists but is not a directory."
            )
        else:
            logger.warning(
                f"Data directory `{data_dir.absolute()}` does not exist. Creating it."
            )
            data_dir.mkdir(parents=True, exist_ok=True)

        return str(data_dir)

    def _get_storage_manager(self, cached_storage_type: FileStorageType):
        if cached_storage_type == FileStorageType.DELTALAKE:
            from atria_datasets.core.storage.deltalake_storage_manager import (
                DeltalakeStorageManager,
            )

            return DeltalakeStorageManager(
                storage_dir=self._storage_dir,
                config_name=self.config.config_name,
                num_processes=self._num_processes,
            )
        elif cached_storage_type == FileStorageType.MSGPACK:
            from atria_datasets.core.storage.msgpack_storage_manager import (
                MsgpackStorageManager,
            )

            return MsgpackStorageManager(
                storage_dir=self._storage_dir,
                config_name=self.config.config_name,
                num_processes=self._num_processes,
            )
        else:
            raise ValueError(f"Unsupported storage type: {cached_storage_type}")

    def _prepare_cached_splits(
        self,
        access_token: str | None = None,
        cached_storage_type: FileStorageType = FileStorageType.DELTALAKE,
        preprocess_transform: Callable | None = None,
        cache_artifacts: bool = True,
        max_cache_image_size: int = 1024,
    ) -> None:
        """Prepare cached splits using DeltaLake storage."""
        storage_manager = self._get_storage_manager(cached_storage_type)

        assert self._dataset_load_mode in [
            DatasetLoadingMode.in_memory,
            DatasetLoadingMode.local_streaming,
        ], (
            f"Dataset loading mode {self._dataset_load_mode} is not supported for cached splits. "
            "Use 'in_memory' or 'local_streaming' modes."
            f"For online streaming, use the 'online_streaming' mode with AtriaHubDataset."
        )

        info_saved = False
        for split in self._available_splits():
            if self._split is not None and split != self._split:
                continue
            split_exists = storage_manager.split_exists(split=split)
            if split_exists and self._overwrite_existing_cached:
                logger.warning(f"Overwriting existing cached split {split.value}")
                storage_manager.purge_split(split)
                split_exists = False

            if not split_exists:
                self.prepare_downloads(
                    data_dir=str(self._data_dir), access_token=access_token
                )
                logger.info(f"Caching split [{split.value}] to {self._storage_dir}")
                storage_manager.write_split(
                    split_iterator=SplitIterator(
                        split=split,
                        data_model=self.data_model,
                        input_transform=self._input_transform,
                        output_transform=DefaultOutputTransformer(
                            self._data_dir, max_cache_image_size=max_cache_image_size
                        )
                        if preprocess_transform is None
                        else preprocess_transform,
                        base_iterator=self._split_iterator(split, self._data_dir),
                        max_len=self.get_max_split_samples(split),
                        load_from_disk=cache_artifacts,
                    )
                )
                if not info_saved:
                    self.save_dataset_info(self._storage_dir)
                    info_saved = True
            else:
                logger.info(
                    f"Loading cached split {split.value} from {storage_manager.split_dir(split)}"
                )

        for split in self._available_splits():
            if self._split is not None and split != self._split:
                continue
            self._split_iterators[split] = storage_manager.read_split(
                split=split, data_model=self.data_model, allowed_keys=self._allowed_keys
            )

    def _prepare_splits(self, access_token: str | None = None) -> None:
        """Prepare splits without caching (direct iteration)."""
        for split in self._available_splits():
            if self._split is not None and split != self._split:
                continue
            self.prepare_downloads(data_dir=self._data_dir, access_token=access_token)
            self._split_iterators[split] = SplitIterator(
                split=split,
                data_model=self.data_model,
                input_transform=self._input_transform,
                base_iterator=self._split_iterator(split, self._data_dir),
                max_len=self.get_max_split_samples(split),
            )

    def _input_transform(self, sample: Any | T_BaseDataInstance) -> T_BaseDataInstance:
        """
        Transform raw sample data into the dataset's data model.

        Args:
            sample: Raw sample data (dict, data model instance, or other format)

        Returns:
            Transformed sample as data model instance

        Raises:
            TypeError: If sample cannot be converted to the data model
        """
        if isinstance(sample, self.data_model):
            return sample
        elif isinstance(sample, dict):
            return self.data_model(**sample)
        else:
            raise TypeError(
                f"Cannot convert sample of type {type(sample)} to data model {self.data_model}"
            )

    def _download_urls(self) -> list[str]:
        """
        Get the list of URLs for downloading dataset files.

        This method should be overridden by subclasses to provide specific URLs
        for the dataset being implemented.

        Returns:
            List of URLs as strings
        """
        return []

    def _custom_download(
        self, data_dir: str, access_token: str | None = None
    ) -> dict[str, Path]:
        """This method can be overridden by subclasses to implement custom download logic.

        Args:
            data_dir: Directory to save downloaded files
            access_token: Authentication token for private resources

        Returns:
            Dictionary mapping download keys to downloaded file paths
        """
        raise NotImplementedError(
            "Subclasses must implement the `_custom_download` method to handle "
            "specific download logic."
        )

    # ==================== Abstract Methods ====================

    @abstractmethod
    def _metadata(self) -> DatasetMetadata:
        """
        Create and return dataset metadata.

        Subclasses should override this method to provide specific metadata
        including description, version, license, and other relevant information.

        Returns:
            DatasetMetadata object with dataset information
        """
        raise NotImplementedError("Subclasses must implement the `_metadata` method.")

    @abstractmethod
    def _available_splits(self) -> list[DatasetSplitType]:
        """
        List available dataset splits.

        Subclasses should override this method to return the splits that are
        available for the dataset (e.g., train, validation, test).

        Returns:
            List of DatasetSplitType values representing available splits
        """
        raise NotImplementedError(
            "Subclasses must implement the `_available_splits` method."
        )

    @abstractmethod
    def _split_iterator(
        self, split: DatasetSplitType, data_dir: str
    ) -> Sequence | Generator:
        """
        Create an iterator for a specific dataset split.

        Args:
            split: The dataset split to create iterator for
            **kwargs: Additional arguments from split configuration

        Returns:
            Iterator or generator yielding data samples for the split

        Note:
            Subclasses must implement this method to define how to iterate
            over the data for each split. The iterator should yield raw data
            that will be transformed by _input_transform.
        """
        raise NotImplementedError(
            "Subclasses must implement the `_split_iterator` method to provide "
            "an iterator for the specified dataset split."
        )


class AtriaImageDataset(AtriaDataset[ImageInstance]):
    """
    Specialized dataset class for handling image datasets.

    This class inherits from AtriaDataset and is specifically typed for ImageInstance
    data models, providing type safety and specialized functionality for image data.

    The class automatically handles:
    - Image-specific data validation
    - Proper type hints for image data
    - Integration with image processing pipelines

    Example:
        ```python
        class CustomImageDataset(AtriaImageDataset):
            def _split_configs(self, data_dir: str) -> list[SplitConfig]:
                return [
                    SplitConfig(
                        split=DatasetSplitType.train,
                        gen_kwargs={"image_dir": f"{data_dir}/train"},
                    )
                ]

            def _split_iterator(self, split: DatasetSplitType, **kwargs):
                # Yield image data samples
                pass
        ```
    """

    __abstract__: bool = True
    __data_model__ = ImageInstance


class AtriaDocumentDataset(AtriaDataset[DocumentInstance]):
    """
    Specialized dataset class for handling document datasets.

    This class inherits from AtriaDataset and is specifically typed for DocumentInstance
    data models, providing type safety and specialized functionality for document data.

    The class automatically handles:
    - Document-specific data validation
    - Proper type hints for document data
    - Integration with text processing pipelines

    Example:
        ```python
        class CustomDocumentDataset(AtriaDocumentDataset):
            def _split_configs(self, data_dir: str) -> list[SplitConfig]:
                return [
                    SplitConfig(
                        split=DatasetSplitType.train,
                        gen_kwargs={"text_dir": f"{data_dir}/train"},
                    )
                ]

            def _split_iterator(self, split: DatasetSplitType, **kwargs):
                # Yield document data samples
                pass
        ```
    """

    __abstract__: bool = True
    __data_model__ = DocumentInstance
