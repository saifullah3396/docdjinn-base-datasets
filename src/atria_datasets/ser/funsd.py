import json
import os
from collections.abc import Generator
from pathlib import Path

from atria_core.logger.logger import get_logger
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

from atria_datasets.core.dataset.atria_dataset import (
    AtriaDatasetConfig,
    AtriaDocumentDataset,
)

from .utilities import (
    _get_line_bboxes,
    _normalize_bbox,
    _sorted_indices_in_reading_order,
)

logger = get_logger(__name__)

# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """"""

# You can copy an official description
_DESCRIPTION = """FUNSD Dataset"""

_HOMEPAGE = "https://guillaumejaume.github.io/FUNSD/"
_DATA_URLS = ["https://guillaumejaume.github.io/FUNSD/dataset.zip"]

_LICENSE = "Apache-2.0 license"

_CLASSES = [
    "O",
    "B-HEADER",
    "I-HEADER",
    "B-QUESTION",
    "I-QUESTION",
    "B-ANSWER",
    "I-ANSWER",
]


class FUNSDConfig(AtriaDatasetConfig):
    apply_reading_order_correction: bool = True


class SplitIterator:
    def __init__(self, split: DatasetSplitType, data_dir: str, config: FUNSDConfig):
        self.split = split
        self.data_dir = Path(data_dir)

        if split == DatasetSplitType.train:
            self.split_dir = self.data_dir / "dataset/dataset/training_data"
        elif split == DatasetSplitType.test:
            self.split_dir = self.data_dir / "dataset/dataset/testing_data"

        self.ann_dir = self.split_dir / "annotations"
        self.image_dir = self.split_dir / "images"
        self.config = config

    def _load_content_and_annotations(
        self, annotation_path: Path, image_size: tuple[int, int]
    ) -> tuple[DocumentContent, EntityLabelingAnnotation]:
        words = []
        word_bboxes = []
        word_segment_level_bboxes = []
        word_labels = []

        with open(annotation_path, encoding="utf8") as f:
            annotation = json.load(f)

        for item in annotation["form"]:
            cur_line_bboxes = []
            words_item, label_item = item["words"], item["label"]
            words_item = [w for w in words_item if w["text"].strip() != ""]
            if len(words_item) == 0:
                continue

            if label_item == "other":
                for w in words_item:
                    words.append(w["text"])
                    word_labels.append("O")
                    cur_line_bboxes.append(_normalize_bbox(w["box"], image_size))
            else:
                words.append(words_item[0]["text"])
                word_labels.append("B-" + label_item.upper())
                cur_line_bboxes.append(
                    _normalize_bbox(words_item[0]["box"], image_size)
                )
                for w in words_item[1:]:
                    words.append(w["text"])
                    word_labels.append("I-" + label_item.upper())
                    cur_line_bboxes.append(_normalize_bbox(w["box"], image_size))

            # add per word box
            word_bboxes.extend(cur_line_bboxes)

            # add segment level box
            cur_line_bboxes = _get_line_bboxes(cur_line_bboxes)
            word_segment_level_bboxes.extend(cur_line_bboxes)

        # sort the word reading order
        if self.config.apply_reading_order_correction:
            sorted_indces = _sorted_indices_in_reading_order(word_bboxes)
            words = [words[i] for i in sorted_indces]
            word_bboxes = [word_bboxes[i] for i in sorted_indces]
            word_labels = [word_labels[i] for i in sorted_indces]
            word_segment_level_bboxes = [
                word_segment_level_bboxes[i] for i in sorted_indces
            ]

        return (
            DocumentContent(
                words=words,
                word_bboxes=BoundingBoxList(value=word_bboxes, normalized=True),
                word_segment_level_bboxes=BoundingBoxList(
                    value=word_segment_level_bboxes, normalized=True
                ),
            ),
            EntityLabelingAnnotation(
                word_labels=LabelList.from_list(
                    [
                        Label(value=_CLASSES.index(word_label), name=word_label)
                        for word_label in word_labels
                    ]
                )
            ),
        )

    def __iter__(self) -> Generator[DocumentInstance, None, None]:
        for filename in sorted(os.listdir(self.image_dir)):
            image = Image(file_path=self.image_dir / Path(filename).name)
            content, annotation = self._load_content_and_annotations(
                annotation_path=self.ann_dir / Path(filename).with_suffix(".json"),
                image_size=(image.source_width, image.source_height),
            )
            yield DocumentInstance(
                sample_id=Path(filename).name,
                image=image,
                content=content,
                annotations=[annotation],
            )

    def __len__(self) -> int:
        return len(os.listdir(self.image_dir))


class FUNSD(AtriaDocumentDataset):
    __config_cls__ = FUNSDConfig

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

    def _available_splits(self):
        return [DatasetSplitType.train, DatasetSplitType.test]

    def _split_iterator(
        self, split: DatasetSplitType, data_dir: str
    ) -> Generator[DocumentInstance, None, None]:
        return SplitIterator(split=split, data_dir=data_dir, config=self.config)
