from typing import Any

from atria_core.types import (
    AnnotatedObject,
    AnnotatedObjectList,
    BoundingBox,
    BoundingBoxMode,
    DatasetLabels,
    DatasetMetadata,
    DocumentInstance,
    Image,
    Label,
    LayoutAnalysisAnnotation,
)

from atria_datasets import AtriaHuggingfaceDocumentDataset

_CLASSES = ["text", "title", "list", "table", "figure"]


class PubLayNet(AtriaHuggingfaceDocumentDataset):
    def _metadata(self) -> DatasetMetadata:
        metadata = super()._metadata()
        metadata.dataset_labels = DatasetLabels(layout=_CLASSES)
        return metadata

    def _input_transform(self, sample: dict[str, Any]) -> DocumentInstance:
        annotated_objects = []
        image = Image(content=sample["image"])
        for ann in sample["annotations"]:
            if ann.get("ignore", False):
                continue

            bbox = BoundingBox(value=ann["bbox"], mode=BoundingBoxMode.XYWH)
            if not bbox.is_valid:
                continue

            if ann["area"] <= 0 or bbox.width < 1 or bbox.height < 1:
                continue

            category_idx = ann["category_id"] - 1
            if category_idx < 0 or category_idx > len(_CLASSES):
                continue

            annotated_objects.append(
                AnnotatedObject(
                    label=Label(value=category_idx, name=_CLASSES[category_idx]),
                    bbox=BoundingBox(value=ann["bbox"], mode=BoundingBoxMode.XYWH)
                    .switch_mode()
                    .normalize(width=image.width, height=image.height),
                    segmentation=ann["segmentation"],
                    iscrowd=bool(ann["iscrowd"]),
                )
            )

        return DocumentInstance(
            sample_id=str(sample["id"]),
            image=image,
            annotations=[
                LayoutAnalysisAnnotation(
                    annotated_objects=AnnotatedObjectList.from_list(annotated_objects)
                )
            ],
        )
