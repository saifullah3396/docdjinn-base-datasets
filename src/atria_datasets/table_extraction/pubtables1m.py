import os
from collections.abc import Generator
from pathlib import Path

from atria_core.logger.logger import get_logger
from atria_core.types import (
    AnnotatedObjectList,
    DatasetLabels,
    DatasetMetadata,
    DatasetSplitType,
    DocumentInstance,
    Image,
    LayoutAnalysisAnnotation,
)

from atria_datasets import DATASET, AtriaDocumentDataset
from atria_datasets.core.dataset.atria_dataset import AtriaDatasetConfig

from .utilities import read_pascal_voc

logger = get_logger(__name__)

_CITATION = """\
@software{smock2021tabletransformer,
    author = {Smock, Brandon and Pesala, Rohith},
    month = {06},
    title = {{Table Transformer}},
    url = {https://github.com/microsoft/table-transformer},
    version = {1.0.0},
    year = {2021}
}
"""

_DESCRIPTION = """\
PubTables-1M: Towards comprehensive table extraction from unstructured documents.
"""

_HOMEPAGE = "https://github.com/microsoft/table-transformer/"

_LICENSE = "https://github.com/microsoft/table-transformer/blob/main/LICENSE"

_DETECTION_URLS = [
    "https://huggingface.co/datasets/bsmock/pubtables-1m/resolve/main/PubTables-1M-Detection_Annotations_Test.tar.gz",
    "https://huggingface.co/datasets/bsmock/pubtables-1m/resolve/main/PubTables-1M-Detection_Annotations_Train.tar.gz",
    "https://huggingface.co/datasets/bsmock/pubtables-1m/resolve/main/PubTables-1M-Detection_Annotations_Val.tar.gz",
    "https://huggingface.co/datasets/bsmock/pubtables-1m/resolve/main/PubTables-1M-Detection_Filelists.tar.gz",
    "https://huggingface.co/datasets/bsmock/pubtables-1m/resolve/main/PubTables-1M-Detection_Images_Test.tar.gz",
    "https://huggingface.co/datasets/bsmock/pubtables-1m/resolve/main/PubTables-1M-Detection_Images_Train_Part1.tar.gz",
    "https://huggingface.co/datasets/bsmock/pubtables-1m/resolve/main/PubTables-1M-Detection_Images_Train_Part2.tar.gz",
    "https://huggingface.co/datasets/bsmock/pubtables-1m/resolve/main/PubTables-1M-Detection_Images_Val.tar.gz",
    "https://huggingface.co/datasets/bsmock/pubtables-1m/resolve/main/PubTables-1M-Detection_Page_Words.tar.gz",
]

_DETECTION_LABELS = ["table", "table rotated"]

_STRUCTURE_URLS = [
    "https://huggingface.co/datasets/bsmock/pubtables-1m/resolve/main/PubTables-1M-Structure_Annotations_Test.tar.gz",
    "https://huggingface.co/datasets/bsmock/pubtables-1m/resolve/main/PubTables-1M-Structure_Annotations_Train.tar.gz",
    "https://huggingface.co/datasets/bsmock/pubtables-1m/resolve/main/PubTables-1M-Structure_Annotations_Val.tar.gz",
    "https://huggingface.co/datasets/bsmock/pubtables-1m/resolve/main/PubTables-1M-Structure_Filelists.tar.gz",
    "https://huggingface.co/datasets/bsmock/pubtables-1m/resolve/main/PubTables-1M-Structure_Images_Test.tar.gz",
    "https://huggingface.co/datasets/bsmock/pubtables-1m/resolve/main/PubTables-1M-Structure_Images_Train.tar.gz",
    "https://huggingface.co/datasets/bsmock/pubtables-1m/resolve/main/PubTables-1M-Structure_Images_Val.tar.gz",
    "https://huggingface.co/datasets/bsmock/pubtables-1m/resolve/main/PubTables-1M-Structure_Table_Words.tar.gz",
]

_STRUCTURE_LABELS = [
    "table",
    "table column",
    "table row",
    "table column header",
    "table projected row header",
    "table spanning cell",
]


def folder_iterator(folder):
    for subdir, dirs, files in os.walk(folder):
        for file in files:
            yield os.path.join(subdir, file)


class SplitIterator:
    def __init__(self, task: str, data_dir: str, split: DatasetSplitType):
        self.split = split
        self.data_dir = Path(data_dir)
        self.task = task

        # Get file lists
        if split == DatasetSplitType.test:
            split_name = "test"
        elif split == DatasetSplitType.validation:
            split_name = "val"
        elif split == DatasetSplitType.train:
            split_name = "train"

        self.xml_filelist = (
            self.data_dir
            / f"PubTables-1M-{self.task.upper()}_Filelists"
            / f"{split_name}_filelist.txt"
        )
        self.images_filelist = (
            self.data_dir
            / f"PubTables-1M-{self.task.upper()}_Filelists"
            / "images_filelist.txt"
        )

        # Collect paths
        self.anns_paths = {}
        self.images_paths = {}

        # Get annotation paths
        for split_dir in ["Test", "Train", "Val"]:
            ann_dir = (
                self.data_dir
                / f"PubTables-1M-{self.task.upper()}_Annotations_{split_dir}"
            )
            if ann_dir.exists():
                for ann_file in folder_iterator(ann_dir):
                    rel_path = Path(ann_file).relative_to(ann_dir)
                    self.anns_paths[str(rel_path)] = Path(ann_file)

        # Get image paths
        for split_dir in ["Test", "Train", "Val"]:
            img_dir = (
                self.data_dir / f"PubTables-1M-{self.task.upper()}_Images_{split_dir}"
            )
            if img_dir.exists():
                for img_file in folder_iterator(img_dir):
                    rel_path = Path(img_file).relative_to(img_dir)
                    self.images_paths[str(rel_path)] = Path(img_file)

    def __iter__(self) -> Generator[DocumentInstance, None, None]:
        # Read XML file list
        with open(self.xml_filelist) as file:
            lines = file.readlines()
            lines = [l.split("/")[-1] for l in lines]
        xml_file_names = {
            f.strip().replace(".xml", "") for f in lines if f.strip().endswith(".xml")
        }

        # Read images file list
        with open(self.images_filelist) as file:
            lines = file.readlines()
        image_file_paths = {
            f.strip().replace(".jpg", "") for f in lines if f.strip().endswith(".jpg")
        }

        file_paths = sorted(xml_file_names.intersection(image_file_paths))
        logger.info(f"Generating {len(file_paths)} samples...")

        for sample_file_path in file_paths:
            # Find annotation file
            ann_file = None
            for ann_path in self.anns_paths:
                if sample_file_path + ".xml" in ann_path:
                    ann_file = self.anns_paths[ann_path]
                    break

            # Find image file
            img_file = None
            for img_path in self.images_paths:
                if sample_file_path + ".jpg" in img_path:
                    img_file = self.images_paths[img_path]
                    break

            if ann_file and img_file and ann_file.exists() and img_file.exists():
                labels = (
                    _STRUCTURE_LABELS if self.task == "structure" else _DETECTION_LABELS
                )
                annotated_objects = read_pascal_voc(ann_file, labels=labels)

                yield DocumentInstance(
                    sample_id=Path(img_file).name,
                    image=Image(file_path=img_file),
                    annotations=[
                        LayoutAnalysisAnnotation(
                            annotated_objects=AnnotatedObjectList.from_list(
                                annotated_objects
                            )
                        )
                    ],
                )

    def __len__(self) -> int:
        with open(self.xml_filelist) as file:
            lines = file.readlines()
        return len([l for l in lines if l.strip().endswith(".xml")])


class PubTables1MConfig(AtriaDatasetConfig):
    task: str = "structure"  # "structure" or "detection"


@DATASET.register(
    "pubtables1m",
    configs=[
        PubTables1MConfig(
            config_name="detection_1k",
            task="detection",
            max_train_samples=1000,
            max_validation_samples=1000,
        ),
        PubTables1MConfig(
            config_name="structure_1k",
            task="structure",
            max_train_samples=1000,
            max_validation_samples=1000,
        ),
    ],
)
class PubTables1M(AtriaDocumentDataset):
    __config_cls__ = PubTables1MConfig

    def _download_urls(self) -> list[str]:
        return _STRUCTURE_URLS if self.config.task == "structure" else _DETECTION_URLS

    def _metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            description=_DESCRIPTION,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            license=_LICENSE,
            dataset_labels=DatasetLabels(
                layout=_STRUCTURE_LABELS
                if self.config.task == "structure"
                else _DETECTION_LABELS
            ),
        )

    def _available_splits(self) -> list[DatasetSplitType]:
        return [
            DatasetSplitType.train,
            DatasetSplitType.validation,
            DatasetSplitType.test,
        ]

    def _split_iterator(
        self, split: DatasetSplitType, data_dir: str
    ) -> Generator[DocumentInstance, None, None]:
        return SplitIterator(task=self.config.task, split=split, data_dir=data_dir)
