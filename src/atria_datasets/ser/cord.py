import io
import json
from collections.abc import Generator
from typing import Any

import pandas as pd
from atria_core.logger.logger import get_logger
from atria_core.types import (
    BoundingBoxList,
    DatasetLabels,
    DatasetMetadata,
    DatasetSplitType,
    DocumentContent,
    DocumentInstance,
    EntityLabelingAnnotation,
    Label,
    LabelList,
)

from atria_datasets.core.dataset.atria_dataset import AtriaDocumentDataset

from .utilities import _get_line_bboxes, _normalize_bbox

logger = get_logger(__name__)

_CITATION = """"""
_DESCRIPTION = """CORD Dataset"""
_HOMEPAGE = "https://github.com/clovaai/cord"
_LICENSE = "Apache-2.0 license"

_CLASSES = [
    "O",
    "B-MENU.NM",
    "B-MENU.NUM",
    "B-MENU.UNITPRICE",
    "B-MENU.CNT",
    "B-MENU.DISCOUNTPRICE",
    "B-MENU.PRICE",
    "B-MENU.ITEMSUBTOTAL",
    "B-MENU.VATYN",
    "B-MENU.ETC",
    "B-MENU.SUB.NM",
    "B-MENU.SUB.UNITPRICE",
    "B-MENU.SUB.CNT",
    "B-MENU.SUB.PRICE",
    "B-MENU.SUB.ETC",
    "B-VOID_MENU.NM",
    "B-VOID_MENU.PRICE",
    "B-SUB_TOTAL.SUBTOTAL_PRICE",
    "B-SUB_TOTAL.DISCOUNT_PRICE",
    "B-SUB_TOTAL.SERVICE_PRICE",
    "B-SUB_TOTAL.OTHERSVC_PRICE",
    "B-SUB_TOTAL.TAX_PRICE",
    "B-SUB_TOTAL.ETC",
    "B-TOTAL.TOTAL_PRICE",
    "B-TOTAL.TOTAL_ETC",
    "B-TOTAL.CASHPRICE",
    "B-TOTAL.CHANGEPRICE",
    "B-TOTAL.CREDITCARDPRICE",
    "B-TOTAL.EMONEYPRICE",
    "B-TOTAL.MENUTYPE_CNT",
    "B-TOTAL.MENUQTY_CNT",
    "I-MENU.NM",
    "I-MENU.NUM",
    "I-MENU.UNITPRICE",
    "I-MENU.CNT",
    "I-MENU.DISCOUNTPRICE",
    "I-MENU.PRICE",
    "I-MENU.ITEMSUBTOTAL",
    "I-MENU.VATYN",
    "I-MENU.ETC",
    "I-MENU.SUB.NM",
    "I-MENU.SUB.UNITPRICE",
    "I-MENU.SUB.CNT",
    "I-MENU.SUB.PRICE",
    "I-MENU.SUB.ETC",
    "I-VOID_MENU.NM",
    "I-VOID_MENU.PRICE",
    "I-SUB_TOTAL.SUBTOTAL_PRICE",
    "I-SUB_TOTAL.DISCOUNT_PRICE",
    "I-SUB_TOTAL.SERVICE_PRICE",
    "I-SUB_TOTAL.OTHERSVC_PRICE",
    "I-SUB_TOTAL.TAX_PRICE",
    "I-SUB_TOTAL.ETC",
    "I-TOTAL.TOTAL_PRICE",
    "I-TOTAL.TOTAL_ETC",
    "I-TOTAL.CASHPRICE",
    "I-TOTAL.CHANGEPRICE",
    "I-TOTAL.CREDITCARDPRICE",
    "I-TOTAL.EMONEYPRICE",
    "I-TOTAL.MENUTYPE_CNT",
    "I-TOTAL.MENUQTY_CNT",
]

_BASE_HF_REPO = "https://huggingface.co/datasets/naver-clova-ix/cord-v2"
_DATA_URLS = [
    f"{_BASE_HF_REPO}/resolve/main/data/train-00000-of-00004-b4aaeceff1d90ecb.parquet",
    f"{_BASE_HF_REPO}/resolve/main/data/train-00001-of-00004-7dbbe248962764c5.parquet",
    f"{_BASE_HF_REPO}/resolve/main/data/train-00002-of-00004-688fe1305a55e5cc.parquet",
    f"{_BASE_HF_REPO}/resolve/main/data/train-00003-of-00004-2d0cd200555ed7fd.parquet",
    f"{_BASE_HF_REPO}/resolve/main/data/validation-00000-of-00001-cc3c5779fe22e8ca.parquet",
    f"{_BASE_HF_REPO}/resolve/main/data/test-00000-of-00001-9c204eb3f4e11791.parquet",
]


class SplitIterator:
    def __init__(self, split: DatasetSplitType, data_files: pd.DataFrame):
        self.split = split
        self.data_files = data_files
        self.data = pd.concat(
            [pd.read_parquet(f) for f in data_files], ignore_index=True
        )

    def _quad_to_box(self, quad: dict[str, int]) -> tuple[int, int, int, int]:
        box = (max(0, quad["x1"]), max(0, quad["y1"]), quad["x3"], quad["y3"])
        if box[3] < box[1]:
            bbox = list(box)
            tmp = bbox[3]
            bbox[3] = bbox[1]
            bbox[1] = tmp
            box = tuple(bbox)
        if box[2] < box[0]:
            bbox = list(box)
            tmp = bbox[2]
            bbox[2] = bbox[0]
            bbox[0] = tmp
            box = tuple(bbox)
        return box

    def _load_content_and_annotations(
        self, row: pd.Series, image_size: tuple[int, int]
    ) -> tuple[str, DocumentContent, EntityLabelingAnnotation]:
        words = []
        word_bboxes = []
        word_segment_level_bboxes = []
        word_labels = []
        annotation = json.loads(row["ground_truth"])
        for item in annotation["valid_line"]:
            cur_line_bboxes = []
            line_words, word_label = item["words"], item["category"]
            line_words = [w for w in line_words if w["text"].strip() != ""]
            if len(line_words) == 0:
                continue
            if word_label == "other":
                for w in line_words:
                    words.append(w["text"])
                    word_labels.append("O")
                    cur_line_bboxes.append(
                        _normalize_bbox(self._quad_to_box(w["quad"]), image_size)
                    )
            else:
                words.append(line_words[0]["text"])
                word_label = word_label.upper().replace("MENU.SUB_", "MENU.SUB.")
                word_labels.append("B-" + word_label)
                cur_line_bboxes.append(
                    _normalize_bbox(
                        self._quad_to_box(line_words[0]["quad"]), image_size
                    )
                )
                for w in line_words[1:]:
                    words.append(w["text"])
                    word_label = word_label.upper().replace("MENU.SUB_", "MENU.SUB.")
                    word_labels.append("I-" + word_label)
                    cur_line_bboxes.append(
                        _normalize_bbox(self._quad_to_box(w["quad"]), image_size)
                    )

            word_bboxes.extend(cur_line_bboxes)
            cur_line_bboxes = _get_line_bboxes(cur_line_bboxes)
            word_segment_level_bboxes.extend(cur_line_bboxes)
        return (
            annotation["meta"]["image_id"],
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

    def __iter__(self) -> Generator[tuple[str, dict[str, Any]], None, None]:
        from atria_core.types import Image
        from PIL import Image as PILImageModule

        for _, row in self.data.iterrows():
            image = Image(
                content=PILImageModule.open(io.BytesIO(row["image"]["bytes"]))
            )
            image_id, content, annotation = self._load_content_and_annotations(
                row, image.size
            )
            yield DocumentInstance(
                sample_id=str(image_id),
                image=image,
                content=content,
                annotations=[annotation],
            )


class CORD(AtriaDocumentDataset):
    def _download_urls(self) -> list[str]:
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
        return [
            DatasetSplitType.train,
            DatasetSplitType.validation,
            DatasetSplitType.test,
        ]

    def _split_iterator(
        self, split: DatasetSplitType, data_files: pd.DataFrame
    ) -> Generator[tuple[str, dict[str, Any]], None, None]:
        return SplitIterator(
            split,
            [
                path
                for key, path in self._downloaded_files.items()
                if split.value in key
            ],
        )
