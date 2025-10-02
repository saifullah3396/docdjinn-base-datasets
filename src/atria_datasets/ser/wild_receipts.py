import json
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
from atria_datasets.core.dataset.atria_dataset import (
    AtriaDatasetConfig,
    AtriaDocumentDataset,
)

from .utilities import _normalize_bbox, _sorted_indices_in_reading_order

# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """"""

# You can copy an official description
_DESCRIPTION = """WildReceipts Dataset"""

_HOMEPAGE = ""

_LICENSE = "Apache-2.0 license"

_DATA_URLS = "https://download.openmmlab.com/mmocr/data/wildreceipt.tar"

_CLASSES = [
    "B-Store_name_value",
    "B-Store_name_key",
    "B-Store_addr_value",
    "B-Store_addr_key",
    "B-Tel_value",
    "B-Tel_key",
    "B-Date_value",
    "B-Date_key",
    "B-Time_value",
    "B-Time_key",
    "B-Prod_item_value",
    "B-Prod_item_key",
    "B-Prod_quantity_value",
    "B-Prod_quantity_key",
    "B-Prod_price_value",
    "B-Prod_price_key",
    "B-Subtotal_value",
    "B-Subtotal_key",
    "B-Tax_value",
    "B-Tax_key",
    "B-Tips_value",
    "B-Tips_key",
    "B-Total_value",
    "B-Total_key",
    "O",
]


class WildReceiptsConfig(AtriaDatasetConfig):
    apply_reading_order_correction: bool = True


class SplitIterator:
    def __init__(
        self, split: DatasetSplitType, data_dir: str, config: WildReceiptsConfig
    ):
        self.split = split
        self.data_dir = Path(data_dir)
        self.config = config

        if split == DatasetSplitType.train:
            self.split_file_path = (
                self.data_dir / "wildreceipt/wildreceipt" / "train.txt"
            )
        elif split == DatasetSplitType.test:
            self.split_file_path = (
                self.data_dir / "wildreceipt/wildreceipt" / "test.txt"
            )

    def _read_label_map(self, filepath: Path) -> dict[int, str]:
        label_map = {}
        with open(filepath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                idx, label = line.split(maxsplit=1)
                label_map[int(idx)] = label.strip()
        return label_map

    def _load_content_and_annotations(
        self, sample_info: dict, image_size: tuple[int, int]
    ) -> tuple[DocumentContent, EntityLabelingAnnotation]:
        words = []
        labels = []
        bboxes = []
        id2labels = self._read_label_map(self.split_file_path.parent / "class_list.txt")

        for i in sample_info["annotations"]:
            label = id2labels[i["label"]]
            if (
                label == "Ignore" or i["text"] == ""
            ):  # label 0 is attached to ignore so we skip it
                continue
            if label in ["Others"]:
                label = "O"
            else:
                label = "B-" + label
            labels.append(label)
            words.append(i["text"])
            bboxes.append(
                _normalize_bbox(
                    [i["box"][6], i["box"][7], i["box"][2], i["box"][3]], image_size
                )
            )

        # sort the word reading order
        if self.config.apply_reading_order_correction:
            sorted_indices = _sorted_indices_in_reading_order(bboxes)
            words = [words[i] for i in sorted_indices]
            word_labels = [labels[i] for i in sorted_indices]
            word_bboxes = [bboxes[i] for i in sorted_indices]
        else:
            word_labels = labels
            word_bboxes = bboxes

        return (
            DocumentContent(
                words=words, word_bboxes=BoundingBoxList(value=word_bboxes)
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
        item_list = []
        with open(self.split_file_path) as f:
            for line in f:
                item_list.append(line.rstrip("\n\r"))

        for filename in item_list:
            sample_info = json.loads(filename)
            image = Image(
                file_path=self.split_file_path.parent / sample_info["file_name"]
            )
            content, annotations = self._load_content_and_annotations(
                sample_info=sample_info, image_size=image.size
            )
            yield DocumentInstance(
                sample_id=sample_info["file_name"],
                image=image,
                content=content,
                annotations=[annotations],
            )

    def __len__(self) -> int:
        with open(self.split_file_path) as f:
            return sum(1 for _ in f)


@DATASET.register("wild_receipts")
class WildReceipts(AtriaDocumentDataset):
    __config_cls__ = WildReceiptsConfig

    def _download_urls(self) -> list[str]:
        return _DATA_URLS

    def _metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            citation=_CITATION,
            description=_DESCRIPTION,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            dataset_labels=DatasetLabels(ser=_CLASSES),
        )

    def _available_splits(self) -> list[DatasetSplitType]:
        return [DatasetSplitType.train, DatasetSplitType.test]

    def _split_iterator(
        self, split: DatasetSplitType, data_dir: str
    ) -> Generator[DocumentInstance, None, None]:
        return SplitIterator(split=split, data_dir=data_dir, config=self.config)
