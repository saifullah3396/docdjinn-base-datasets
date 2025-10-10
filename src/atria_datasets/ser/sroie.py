import json
import os
from collections.abc import Generator
from pathlib import Path

from atria_core.types import (
    BoundingBoxList,
    DatasetLabels,
    DatasetMetadata,
    DatasetSplitType,
    DocumentContent,
    DocumentInstance,
    EntityLabelingAnnotation,
    Image,
    Label,
    LabelList,
)

from atria_datasets import DATASET
from atria_datasets.core.dataset.atria_dataset import AtriaDocumentDataset

from .utilities import _normalize_bbox

# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """"""

# You can copy an official description
_DESCRIPTION = """SROIE Receipts Dataset"""

_HOMEPAGE = "https://rrc.cvc.uab.es/?ch=13"

_LICENSE = "Apache-2.0 license"

_CLASSES = [
    "O",
    "B-COMPANY",
    "I-COMPANY",
    "B-DATE",
    "I-DATE",
    "B-ADDRESS",
    "I-ADDRESS",
    "B-TOTAL",
    "I-TOTAL",
]

_DATA_URLS = {
    "sroie": (
        "https://drive.google.com/file/d/1ZyxAw1d-9UvhgNLGRvsJK4gBCMf0VpGD/view?usp=sharing",
        ".zip",
    )
}


class SplitIterator:
    def __init__(self, split: DatasetSplitType, data_dir: str):
        self.split = split
        self.data_dir = Path(data_dir)

        if split == DatasetSplitType.train:
            self.split_dir = self.data_dir / "sroie/sroie/train"
        elif split == DatasetSplitType.test:
            self.split_dir = self.data_dir / "sroie/sroie/test"

        self.ann_dir = self.split_dir / "tagged"
        self.image_dir = self.split_dir / "images"

    def _load_content_and_annotations(
        self, annotation_path: Path, image_size: tuple[int, int]
    ) -> tuple[DocumentContent, EntityLabelingAnnotation]:
        with open(annotation_path, encoding="utf8") as f:
            sample = json.load(f)

        words = []
        word_bboxes = []
        word_labels = []

        for word, box, label in zip(
            sample["words"], sample["bbox"], sample["labels"], strict=True
        ):
            word = word.strip()
            if len(word) == 0:
                continue
            words.append(word)
            word_bboxes.append(_normalize_bbox(box, image_size))
            word_labels.append(label)

        return (
            DocumentContent(
                words=words, word_bboxes=BoundingBoxList(value=word_bboxes, normalized=True)
            ),
            EntityLabelingAnnotation(
                word_labels=LabelList.from_list(
                    [
                        Label(value=_CLASSES.index(label), name=label)
                        for label in word_labels
                    ]
                )
            ),
        )

    def __iter__(self) -> Generator[DocumentInstance, None, None]:
        for filename in sorted(os.listdir(self.image_dir)):
            image = Image(file_path=self.image_dir / Path(filename).name)
            content, annotation = self._load_content_and_annotations(
                annotation_path=self.ann_dir / Path(filename).with_suffix(".json"),
                image_size=image.size,
            )
            yield DocumentInstance(
                sample_id=Path(filename).name,
                image=image,
                content=content,
                annotations=[annotation],
            )

    def __len__(self) -> int:
        return len(os.listdir(self.image_dir))


@DATASET.register("sroie")
class SROIE(AtriaDocumentDataset):
    def _download_urls(self) -> dict[str, tuple[str, str]]:
        return _DATA_URLS

    def _metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            citation=_CITATION,
            homepage=_HOMEPAGE,
            description=_DESCRIPTION,
            license=_LICENSE,
            dataset_labels=DatasetLabels(ser=_CLASSES),
        )

    def _available_splits(self) -> list[DatasetSplitType]:
        return [DatasetSplitType.train, DatasetSplitType.test]

    def _split_iterator(
        self, split: DatasetSplitType, data_dir: str
    ) -> Generator[DocumentInstance, None, None]:
        return SplitIterator(split=split, data_dir=data_dir)
