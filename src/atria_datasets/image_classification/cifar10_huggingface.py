from typing import Any

from atria_core.types import ClassificationAnnotation, Image, ImageInstance, Label

from atria_datasets import AtriaHuggingfaceImageDataset


class HuggingfaceCifar10(AtriaHuggingfaceImageDataset):
    def _input_transform(self, sample: dict[str, Any]) -> ImageInstance:
        return ImageInstance(
            image=Image(content=sample["img"]),
            annotations=[
                ClassificationAnnotation(
                    label=Label(
                        value=sample["label"],
                        name=self.metadata.dataset_labels.classification[
                            sample["label"]
                        ],
                    )
                )
            ],
        )
