# Copyright 2022 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DockBank dataset"""

import json
from collections.abc import Generator, Iterable
from pathlib import Path

from atria_core.logger.logger import get_logger
from atria_core.types import (
    BoundingBoxList,
    DatasetLabels,
    DatasetMetadata,
    DatasetSplitType,
    DocumentInstance,
    Image,
    Label,
    LabelList,
)
from atria_core.types.generic.annotations import EntityLabelingAnnotation
from atria_core.types.generic.document_content import DocumentContent

from atria_datasets.core.dataset.atria_dataset import (
    AtriaDatasetConfig,
    AtriaDocumentDataset,
)

logger = get_logger(__name__)

# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """\
@misc{li2020docbank,
    title={DocBank: A Benchmark Dataset for Document Layout Analysis},
    author={Minghao Li and Yiheng Xu and Lei Cui and Shaohan Huang and Furu Wei and Zhoujun Li and Ming Zhou},
    year={2020},
    eprint={2006.01038},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
"""

# You can copy an official description
_DESCRIPTION = """\
DocBank is a new large-scale dataset that is constructed using a weak supervision approach.
It enables models to integrate both the textual and layout information for downstream tasks.
The current DocBank dataset totally includes 500K document pages, where 400K for training, 50K for validation and 50K for testing.
"""

_HOMEPAGE = "https://doc-analysis.github.io/docbank-page/index.html"

_LICENSE = "Apache-2.0 license"

_DATA_URLS = [
    "https://huggingface.co/datasets/liminghao1630/DocBank/resolve/main/DocBank_500K_ori_img.zip.001",
    "https://huggingface.co/datasets/liminghao1630/DocBank/resolve/main/DocBank_500K_ori_img.zip.002",
    "https://huggingface.co/datasets/liminghao1630/DocBank/resolve/main/DocBank_500K_ori_img.zip.003",
    "https://huggingface.co/datasets/liminghao1630/DocBank/resolve/main/DocBank_500K_ori_img.zip.004",
    "https://huggingface.co/datasets/liminghao1630/DocBank/resolve/main/DocBank_500K_ori_img.zip.005",
    "https://huggingface.co/datasets/liminghao1630/DocBank/resolve/main/DocBank_500K_ori_img.zip.006",
    "https://huggingface.co/datasets/liminghao1630/DocBank/resolve/main/DocBank_500K_ori_img.zip.007",
    "https://huggingface.co/datasets/liminghao1630/DocBank/resolve/main/DocBank_500K_ori_img.zip.008",
    "https://huggingface.co/datasets/liminghao1630/DocBank/resolve/main/DocBank_500K_ori_img.zip.009",
    "https://huggingface.co/datasets/liminghao1630/DocBank/resolve/main/DocBank_500K_ori_img.zip.010",
    "https://huggingface.co/datasets/liminghao1630/DocBank/resolve/main/DocBank_500K_txt.zip",
    "https://huggingface.co/datasets/liminghao1630/DocBank/resolve/main/MSCOCO_Format_Annotation.zip",
]

_CLASSES = [
    "abstract",
    "author",
    "caption",
    "equation",
    "figure",
    "footer",
    "list",
    "paragraph",
    "reference",
    "section",
    "table",
    "title",
    "date",
]


class DocBankConfig(AtriaDatasetConfig):
    max_words_per_sample: int = 4000


class SplitIterator:
    def __init__(self, split: DatasetSplitType, data_dir: str, config: DocBankConfig):
        self.split = split
        self.data_dir = Path(data_dir)
        self.config = config

        # Set up paths based on split
        if split == DatasetSplitType.train:
            self.split_filepath = (
                self.data_dir / "MSCOCO_Format_Annotation/500K_train.json"
            )
        elif split == DatasetSplitType.test:
            self.split_filepath = (
                self.data_dir / "MSCOCO_Format_Annotation/500K_test.json"
            )
        elif split == DatasetSplitType.validation:
            self.split_filepath = (
                self.data_dir / "MSCOCO_Format_Annotation/500K_valid.json"
            )

        self.image_base_dir = (
            self.data_dir / "DocBank_500K_ori_img/DocBank_500K_ori_img/"
        )
        self.annotation_base_dir = self.data_dir / "DocBank_500K_txt/DocBank_500K_txt/"

        with open(self.split_filepath) as f:
            logger.info(f"Loading split data from {self.split_filepath}")
            self.split_data = json.load(f)

    def _load_content_and_annotations(
        self, text_file: Path, image_file_path: Path
    ) -> tuple[DocumentContent, EntityLabelingAnnotation]:
        words = []
        word_bboxes = []
        word_labels = []

        with open(text_file, encoding="utf8") as fp:
            for line in fp.readlines():
                tts = line.split("\t")
                if not len(tts) == 10:
                    logger.warning(f"Incomplete line in file {text_file}")
                    continue

                word = tts[0]
                bbox = list(map(int, tts[1:5]))
                structure = tts[9]

                if len(word) == 0:
                    continue
                x1, y1, x2, y2 = bbox
                area = (x2 - x1) * (y2 - y1)
                if (
                    word == "##LTLine##" or word == "##LTFigure##" and area < 10
                ):  # remove ltline and ltfigures with very small noisy features
                    continue

                words.append(word)
                word_bboxes.append(bbox)  # boxes are already normalized 0 to 1000
                word_labels.append(structure.strip())

        return (
            DocumentContent(
                words=words,
                word_bboxes=BoundingBoxList(value=word_bboxes, normalized=False),
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
        for idx in range(len(self.split_data["images"])):
            image_file_path = self.split_data["images"][idx]["file_name"]
            if not (self.image_base_dir / image_file_path).exists():
                logger.warning(
                    f"Image file {image_file_path} does not exist in {self.image_base_dir}"
                )
                continue
            text_file = self.annotation_base_dir / (
                image_file_path.replace("_ori.jpg", "") + ".txt"
            )

            content, annotation = self._load_content_and_annotations(
                text_file, self.image_base_dir / image_file_path
            )

            if (
                len(content.words) > 0
                and len(content.words) < self.config.max_words_per_sample
            ):
                yield DocumentInstance(
                    sample_id=Path(image_file_path).name,
                    image=Image(file_path=self.image_base_dir / image_file_path),
                    content=content,
                    annotations=[annotation],
                )

    def __len__(self) -> int:
        return len(self.split_data["images"])


class DocBankLER(AtriaDocumentDataset):
    __config_cls__ = DocBankConfig

    def _download_urls(self) -> list[str]:
        return _DATA_URLS

    def _metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            description=_DESCRIPTION,
            citation=_CITATION,
            license=_LICENSE,
            homepage=_HOMEPAGE,
            dataset_labels=DatasetLabels(ser=_CLASSES),
        )

    def _available_splits(self) -> list[DatasetSplitType]:
        return [
            DatasetSplitType.train,
            DatasetSplitType.validation,
            DatasetSplitType.test,
        ]

    def _split_iterator(
        self, split: DatasetSplitType, data_dir: str
    ) -> Iterable[DocumentInstance]:
        return SplitIterator(split=split, data_dir=data_dir, config=self.config)
