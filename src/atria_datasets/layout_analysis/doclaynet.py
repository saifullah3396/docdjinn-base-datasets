from typing import Any

from atria_core.types import (
    AnnotatedObject,
    BoundingBox,
    BoundingBoxMode,
    ClassificationAnnotation,
    DatasetLabels,
    DatasetMetadata,
    DocumentInstance,
    Image,
    Label,
    LayoutAnalysisAnnotation,
)

from atria_datasets import DATASET
from atria_datasets.core.dataset.atria_huggingface_dataset import (
    AtriaHuggingfaceDatasetConfig,
    AtriaHuggingfaceDocumentDataset,
)

_DOC_CLASSES = [
    "financial_reports",
    "scientific_articles",
    "laws_and_regulations",
    "government_tenders",
    "manuals",
    "patents",
]

_LAYOUT_CLASSES = [
    "Caption ",
    "Footnote",
    "Formula",
    "List-item",
    "Page-footer",
    "Page-header",
    "Picture",
    "Section-header",
    "Table",
    "Text",
    "Title ",
]


@DATASET.register(
    "doclaynet",
    configs=[
        AtriaHuggingfaceDatasetConfig(
            hf_repo="ds4sd/DocLayNet", hf_config_name="2022.08"
        )
    ],
)
class DocLayNet(AtriaHuggingfaceDocumentDataset):
    def _metadata(self) -> DatasetMetadata:
        metadata = super()._metadata()
        metadata.dataset_labels = DatasetLabels(
            classification=_DOC_CLASSES, layout=_LAYOUT_CLASSES
        )
        return metadata

    def _input_transform(self, sample: dict[str, Any]) -> DocumentInstance:
        annotated_objects = []
        image = Image(content=sample["image"])
        for ann in sample["objects"]:
            if ann.get("ignore", False):
                continue

            bbox = BoundingBox(value=ann["bbox"], mode=BoundingBoxMode.XYWH)
            if not bbox.is_valid:
                continue

            if ann["area"] <= 0 or bbox.width < 1 or bbox.height < 1:
                continue

            category_idx = ann["category_id"] - 1
            if category_idx < 0 or category_idx > len(_LAYOUT_CLASSES):
                continue

            annotated_objects.append(
                AnnotatedObject(
                    label=Label(value=category_idx, name=_LAYOUT_CLASSES[category_idx]),
                    bbox=bbox.switch_mode(),
                    segmentation=ann["segmentation"],
                    iscrowd=ann["iscrowd"],
                )
            )
        return DocumentInstance(
            sample_id=sample["image_id"],
            image=image,
            annotations=[
                ClassificationAnnotation(
                    label=Label(
                        value=_DOC_CLASSES.index(sample["doc_category"]),
                        name=sample["doc_category"],
                    )
                ),
                LayoutAnalysisAnnotation(annotated_objects=annotated_objects),
            ],
        )
