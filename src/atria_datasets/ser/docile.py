import base64
import io
from pathlib import Path

import PIL
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
from docile.dataset import KILE_FIELDTYPES, LIR_FIELDTYPES, Dataset

from atria_datasets import DATASET
from atria_datasets.core.dataset.atria_dataset import (
    AtriaDatasetConfig,
    AtriaDocumentDataset,
)

from .docile_utils.preprocessor import (
    generate_unique_entities,
    load_docile_dataset,
    prepare_docile_dataset,
)

# pyarrow_hotfix.uninstall()
# pyarrow.PyExtensionType.set_auto_load(True)

logger = get_logger(__name__)


# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """"""

# You can copy an official description
_DESCRIPTION = """Docile Dataset"""

_HOMEPAGE = "https://github.com/rossumai/docile"

_LICENSE = "Apache-2.0 license"


_ALL_LABELS = generate_unique_entities()
_KILE_LABELS = (
    ["O"] + [f"B-{x}" for x in KILE_FIELDTYPES] + [f"I-{x}" for x in KILE_FIELDTYPES]
)
_LIR_LABELS = (
    ["O"] + [f"B-{x}" for x in LIR_FIELDTYPES] + [f"I-{x}" for x in LIR_FIELDTYPES]
)

_DATA_URLS = [
    "https://docile-dataset-rossum.s3.eu-west-1.amazonaws.com/{access_token}/annotated-trainval.zip",
    "https://docile-dataset-rossum.s3.eu-west-1.amazonaws.com/{access_token}/test.zip",
]

_SYNTHETIC_DATA_URLS = [
    "https://docile-dataset-rossum.s3.eu-west-1.amazonaws.com/{access_token}/synthetic.zip"
]


def filter_empty_words(row):
    filtered_indices = [idx for idx, x in enumerate(row["words"]) if x != ""]
    for k in ["words", "word_bboxes", "word_bboxes_segment_level", "word_labels"]:
        if k not in row.keys():
            continue

        if k == "words":
            row[k] = [row[k][idx].strip() for idx in filtered_indices]
        else:
            row[k] = [row[k][idx] for idx in filtered_indices]
    return row


class DocileConfig(AtriaDatasetConfig):
    synthetic: bool = False
    overlap_threshold: float = 0.5
    image_shape: tuple = (1024, 1024)
    type: str = "kile"  # "kile" or "lir"


class SplitIterator:
    def __init__(self, dataset, label_names, config: DocileConfig):
        self.dataset = dataset
        self.label_names = label_names
        self.config = config

    def _remap_labels_to_task_labels(self, labels):
        import numpy as np

        all_labels = np.array(_ALL_LABELS)
        if self.config.type == "kile":
            label_map = _KILE_LABELS
        elif self.config.type == "lir":
            label_map = _LIR_LABELS

        labels_to_idx = dict(zip(label_map, range(len(label_map)), strict=True))
        remapped_labels = []
        for label in labels:
            # each label is a boolean map to multiple unique entities in _ALL_LABELS
            # here we only take those labels that are present in the label_map (KILE OR LIR or other)
            sample_label = [x for x in all_labels[label] if x in label_map]
            if len(sample_label) > 0:  # now we take the label index from the label_map
                remapped_labels.append(labels_to_idx[sample_label[0]])
            else:
                remapped_labels.append(labels_to_idx[label_map[0]])
        return remapped_labels

    def __iter__(self):
        for row in self.dataset:
            row["ner_tags"] = self._remap_labels_to_task_labels(row["ner_tags"])
            row["tokens"] = list(row["tokens"])
            yield DocumentInstance(
                sample_id=str(row["id"]),
                image=Image(
                    content=PIL.Image.open(io.BytesIO(base64.b64decode(row["img"])))
                ),
                content=DocumentContent(
                    words=row["tokens"],
                    word_bboxes=BoundingBoxList(value=row["bboxes"]),
                ),
                annotations=[
                    EntityLabelingAnnotation(
                        word_labels=LabelList.from_list(
                            [
                                Label(value=label, name=self.label_names[label])
                                for label in row["ner_tags"]
                            ]
                        )
                    )
                ],
            )


@DATASET.register(
    "docile",
    configs=[
        DocileConfig(synthetic=False, type="kile"),
        DocileConfig(synthetic=False, type="lir"),
        DocileConfig(synthetic=True, type="kile"),
        DocileConfig(synthetic=True, type="lir"),
    ],
)
class Docile(AtriaDocumentDataset):
    __config_cls__ = DocileConfig
    __requires_access_token__ = True

    def _download_urls(self) -> list[str]:
        if self.config.synthetic:
            return [
                url.format(access_token=self._access_token)
                for url in _SYNTHETIC_DATA_URLS
            ]
        return [url.format(access_token=self._access_token) for url in _DATA_URLS]

    def _available_splits(self) -> list[DatasetSplitType]:
        return [DatasetSplitType.train, DatasetSplitType.validation]

    def _metadata(self) -> DatasetMetadata:
        if self.config.type == "kile":
            labels = _KILE_LABELS
        elif self.config.type == "lir":
            labels = _KILE_LABELS
        return DatasetMetadata(
            citation=_CITATION,
            homepage=_HOMEPAGE,
            description=_DESCRIPTION,
            license=_LICENSE,
            dataset_labels=DatasetLabels(ser=labels),
        )

    def label_names(self):
        if self.config.type == "kile":
            return _KILE_LABELS
        elif self.config.type == "lir":
            return _LIR_LABELS
        else:
            return _ALL_LABELS

    def _prepare_dataset(self, split: DatasetSplitType, data_dir: str):
        split_dir = "annotated-trainval"
        if not hasattr(self, "_dataset"):
            data_dir = Path(data_dir)
            split_name = "val" if split == DatasetSplitType.validation else "train"
            docile_dataset = Dataset(
                split_name, data_dir / split_dir, load_annotations=False, load_ocr=False
            )
            preprocessed_name = f"{docile_dataset.split_name}_multilabel_preprocessed_withImgs_{self.config.image_shape[0]}x{self.config.image_shape[1]}.json"
            if not (data_dir / split_dir / split_dir / preprocessed_name).exists():
                prepare_docile_dataset(
                    docile_dataset,
                    self.config.overlap_threshold,
                    data_dir / split_dir,
                    image_shape=self.config.image_shape,
                )
            label_names = self.label_names()
            dataset = load_docile_dataset(
                docile_dataset,
                data_dir / split_dir,
                image_shape=self.config.image_shape,
            )
        return dataset, label_names

    def _split_iterator(self, split: DatasetSplitType, data_dir: str):
        dataset, label_names = self._prepare_dataset(split, data_dir)
        return SplitIterator(
            dataset=dataset, label_names=label_names, config=self.config
        )
