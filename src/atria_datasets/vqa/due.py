import uuid
from collections.abc import Iterable
from pathlib import Path

import PIL
from atria_core.logger import get_logger
from atria_core.types import (
    PDF,
    BoundingBoxList,
    DatasetLabels,
    DatasetMetadata,
    DatasetSplitType,
    DocumentInstance,
)
from atria_core.types.generic.annotations import ExtractiveQAAnnotation
from atria_core.types.generic.document_content import DocumentContent
from atria_core.types.generic.image import Image
from atria_core.types.generic.question_answer_pair import ExtractiveQAPair
from pdf2image import convert_from_path
from PIL.Image import Image as PILImage

from atria_datasets import DATASET, AtriaDocumentDataset
from atria_datasets.core.dataset.atria_dataset import AtriaDatasetConfig

logger = get_logger(__name__)

_CITATION = """\
@article{Kumar2014StructuralSF,
    title={Structural similarity for document image classification and retrieval},
    author={Jayant Kumar and Peng Ye and David S. Doermann},
    journal={Pattern Recognit. Lett.},
    year={2014},
    volume={43},
    pages={119-126}
}
"""

_DESCRIPTION = """\
The Tobacco3482 dataset consists of 3842 grayscale images in 10 classes. In this version, the dataset is plit into 2782 training images, and 700 test images.
"""

_HOMEPAGE = "https://www.kaggle.com/datasets/patrickaudriaz/tobacco3482jpg"
_LICENSE = "https://www.industrydocuments.ucsf.edu/help/copyright/"
_DATA_URLS = {
    "datasets": "https://applica-public.s3.eu-west-1.amazonaws.com/due/datasets/{config_name}.tar.gz",
    "pdfs": "https://applica-public.s3.eu-west-1.amazonaws.com/due/pdfs/{config_name}.tar.gz",
}


PIL.Image.MAX_IMAGE_PIXELS = 933120000


class DueBenchmarkExtractiveQAConfig(AtriaDatasetConfig):
    """BuilderConfig for DueBenchmark. This configuration is taken from generate_memmaps from the DueBenchmark library."""

    # corpus args
    segment_levels: tuple = ("tokens", "pages")
    ocr_engine: str = "microsoft_cv"
    answers_extraction_method: str = "v1"  # v1, v2, v1_v2, v2_v1, v3
    image_dpi: int = 72


def find_answers_in_words(words, answers, extraction_method="v1"):
    from atria_datasets.vqa.utilities import (
        extract_start_end_index_v1,
        extract_start_end_index_v2,
        extract_start_end_index_v3,
    )

    if extraction_method == "v1":
        return extract_start_end_index_v1(answers, words)
    elif extraction_method == "v2":
        return extract_start_end_index_v2(answers, words)
    elif extraction_method == "v1_v2":
        processed_answers, all_not_found = extract_start_end_index_v1(answers, words)
        if all_not_found:
            processed_answers, _ = extract_start_end_index_v2(answers, words)
        return processed_answers, all_not_found
    elif extraction_method == "v2_v1":
        processed_answers, all_not_found = extract_start_end_index_v2(answers, words)
        if all_not_found:
            processed_answers, _ = extract_start_end_index_v1(answers, words)
        return processed_answers, all_not_found
    elif extraction_method == "v3":
        processed_answers, all_not_found = extract_start_end_index_v3(answers, words)
        return processed_answers, all_not_found
    else:
        raise ValueError(f"Extraction method {extraction_method} not supported")


class SplitIterator:
    def __init__(
        self,
        split: DatasetSplitType,
        data_dir: str,
        config: DueBenchmarkExtractiveQAConfig,
    ):
        from benchmarker.data.reader.benchmark_dataset import BenchmarkDataset

        self._config = config

        # make path for preprocessed data
        self._data_dir = data_dir
        split = "dev" if split.value == "validation" else split.value
        extracted_name = config.config_name[2:]
        if config.config_name[2:] == "InfographicsVQA":
            extracted_name = "infographics_vqa"
        if config.config_name[2:] == "DocVQA":
            extracted_name = "docvqa"
        if config.config_name[2:] == "PWC":
            extracted_name = "AxCell"
        if config.config_name[2:] == "KleisterCharity":
            extracted_name = "kleister-charity"
        self._benchmark_dataset = BenchmarkDataset(
            directory=Path(data_dir)
            / "datasets"
            / config.config_name[2:]  # base name then extracted name
            / "aws_neurips_time"
            / extracted_name,
            split=split,
            ocr=config.ocr_engine,
            segment_levels=config.segment_levels,
        )

    def _extract_answers(self, answers: list[str], words: list):
        processed_answers, _ = find_answers_in_words(
            [word.lower() for word in words],
            [answer.lower() for answer in answers],
            self._config.answers_extraction_method,
        )
        answer_start_indices = [
            int(ans["answer_start_index"]) for ans in processed_answers
        ]
        answer_end_indices = [int(ans["answer_end_index"]) for ans in processed_answers]
        gold_answers = [ans["gold_answer"] for ans in processed_answers]
        return answer_start_indices, answer_end_indices, gold_answers

    def _generate_samples(self):
        for document in self._benchmark_dataset:
            sample = {
                "sample_id": document.identifier,
                "document_2d": document.document_2d,
            }
            keys = document.annotations.keys()
            question = list(keys)[0]
            values = document.annotations[question]
            if not values:
                values = ["None"]
            answer_start_indices, answer_end_indices, gold_answers = (
                self._extract_answers(values, document.document_2d.tokens)
            )
            page_ranges = document.document_2d.seg_data["pages"]["ranges"]
            page_bboxes = document.document_2d.seg_data["pages"]["org_bboxes"]
            page_idx = 0
            if len(page_ranges) > 1:
                answer_idx = answer_start_indices[0]
                if answer_idx > -1:
                    for p_idx, p_range in enumerate(page_ranges):
                        if answer_idx >= p_range[0] and answer_idx <= p_range[1]:
                            page_idx = p_idx
                            break
            page_range = page_ranges[page_idx]
            page_bbox = page_bboxes[page_idx]

            sample["page_idx"] = page_idx
            sample["page_size"] = (page_bbox[2], page_bbox[3])
            sample["tokens_in_page"] = document.document_2d.tokens[
                page_range[0] : page_range[1] + 1
            ]
            sample["token_bboxes_in_page"] = document.document_2d.seg_data["tokens"][
                "org_bboxes"
            ][page_range[0] : page_range[1] + 1].tolist()
            # normalize page bboxes
            page_width, page_height = page_bbox[2], page_bbox[3]
            sample["token_bboxes_in_page"] = [
                [
                    bbox[0] / page_width,
                    bbox[1] / page_height,
                    bbox[2] / page_width,
                    bbox[3] / page_height,
                ]
                for bbox in sample["token_bboxes_in_page"]
            ]
            sample["annotations"] = {
                "question": question,
                "answers": gold_answers,
                "answer_start_indices": [
                    ans - page_range[0] if ans != -1 else -1
                    for ans in answer_start_indices
                ],
                "answer_end_indices": [
                    ans - page_range[0] if ans != -1 else -1
                    for ans in answer_end_indices
                ],
            }
            yield sample

    def __iter__(self):
        last_sample_id = None
        last_sample_page_idx = None

        def group_sample(grouped: list[any]):
            grouped = {
                "page_idx": grouped[0]["page_idx"],
                "page_size": grouped[0]["page_size"],
                "tokens_in_page": grouped[0]["tokens_in_page"],
                "token_bboxes_in_page": grouped[0]["token_bboxes_in_page"],
                "sample_id": grouped[0]["sample_id"],
                "document_2d": grouped[0]["document_2d"],
                "annotations": [g["annotations"] for g in grouped],
            }

            # we remap all due benchmark keys to what we require in our datasets
            pdf_file_path = (
                Path(self._data_dir)
                / "pdfs"
                / self._config.config_name[2:]  # base name then extracted name
                / self._config.config_name[2:]
                / (grouped["sample_id"])
            )
            if not str(pdf_file_path).endswith(".pdf"):
                pdf_file_path = Path(str(pdf_file_path) + ".pdf")

            # if the file does not exist we ignore it
            if not pdf_file_path.exists():
                logger.warning(f"File {pdf_file_path} not found. Skipping it")
                return None
            return grouped

        grouped = []
        for sample in self._generate_samples():
            current_sample_id = sample["sample_id"]
            current_sample_page_idx = sample["page_idx"]

            if last_sample_id is not None and last_sample_page_idx is not None:
                if (
                    last_sample_id != current_sample_id
                    or last_sample_page_idx != current_sample_page_idx
                ):
                    group_sampled = group_sample(grouped)
                    if group_sampled is not None:
                        yield group_sampled
                    grouped = []

            grouped.append(sample)
            last_sample_id = sample["sample_id"]
            last_sample_page_idx = sample["page_idx"]

        if len(grouped) > 0:
            group_sampled = group_sample(grouped)
            if group_sampled is not None:
                yield group_sampled

    def __len__(self) -> int:
        return len(self._benchmark_dataset)


@DATASET.register(
    "due_benchmark",
    configs=[
        DueBenchmarkExtractiveQAConfig(config_name="ExDocVQA"),
        DueBenchmarkExtractiveQAConfig(config_name="ExPWC", ocr_engine="tesseract"),
        DueBenchmarkExtractiveQAConfig(config_name="ExDeepForm"),
        DueBenchmarkExtractiveQAConfig(config_name="ExTabFact", ocr_engine="tesseract"),
        DueBenchmarkExtractiveQAConfig(config_name="ExWikiTableQuestions"),
        DueBenchmarkExtractiveQAConfig(config_name="ExInfographicsVQA"),
        DueBenchmarkExtractiveQAConfig(config_name="ExKleisterCharity"),
    ],
)
class DueBenchmarkExtractiveQA(AtriaDocumentDataset):
    __config_cls__ = DueBenchmarkExtractiveQAConfig

    def _download_urls(self) -> list[str]:
        return {
            f"{key}/{self.config.config_name[2:]}": url.format(
                config_name=self.config.config_name[2:]
            )
            for key, url in _DATA_URLS.items()
        }

    def _metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            citation=_CITATION,
            description=_DESCRIPTION,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            dataset_labels=DatasetLabels(),
        )

    def _available_splits(self):
        return [
            DatasetSplitType.train,
            DatasetSplitType.test,
            DatasetSplitType.validation,
        ]

    def _split_iterator(
        self, split: DatasetSplitType, data_dir: str
    ) -> Iterable[tuple[Path, Path, int]]:
        return SplitIterator(split=split, data_dir=Path(data_dir), config=self.config)

    # Load the specific page from PDF as image
    def _load_pdf_page_as_image(
        self,
        pdf_path: Path,
        page_number: int,
        page_size: tuple[int, int],
        dpi: int = 72,
    ) -> PILImage:
        """Load a specific page from PDF as PIL Image."""
        try:
            # Convert specific page to image (page_number is 0-indexed for pdf2image)
            pages = convert_from_path(
                pdf_path,
                first_page=page_number + 1,
                last_page=page_number + 1,
                dpi=dpi,
                size=page_size,
            )
            if pages:
                return pages[0]
            else:
                raise ValueError(f"Could not load page {page_number} from {pdf_path}")
        except Exception as e:
            logger.error(f"Error loading page {page_number} from {pdf_path}: {e}")
            raise

    def _input_transform(self, sample: tuple[Path, Path, int]) -> DocumentInstance:
        # we remap all due benchmark keys to what we require in our datasets
        pdf_file_path = (
            Path(self._data_dir)
            / "pdfs"
            / self.config.config_name[2:]  # base name then extracted name
            / self.config.config_name[2:]
            / (sample["sample_id"])
        )

        # if the file does not exist we ignore it
        if not str(pdf_file_path).endswith(".pdf"):
            pdf_file_path = Path(str(pdf_file_path) + ".pdf")

        # Load the PDF page as image
        page_image = self._load_pdf_page_as_image(
            pdf_file_path,
            sample["page_idx"],
            sample["page_size"],
            self.config.image_dpi,
        )
        # print("sample", sample["sample_id"])
        # print("page idx", sample["page_idx"])
        # print("image size", page_image)
        # print("sample", sample["tokens_in_page"])
        # print('answer_start_indices',sample["annotations"][0]['answer_start_indices'])
        # print('answer_end_indices',sample["annotations"][0]['answer_end_indices'])
        # print("question", sample["annotations"][0]["question"])
        # print("answers", sample["annotations"][0]["answers"])
        # if sample["annotations"][0]["answer_start_indices"][0] > -1:
        #     answer_texts = [
        #         sample["tokens_in_page"][i]
        #         for i in range(
        #             sample["annotations"][0]["answer_start_indices"][0],
        #             sample["annotations"][0]["answer_end_indices"][0] + 1,
        #         )
        #     ]
        #     print("test answer", " ".join(answer_texts))

        # resize image to max height and width
        # print('sample["annotations"]',sample["annotations"])
        doc = DocumentInstance(
            sample_id=sample["sample_id"] + f"-{uuid.uuid4().hex[:4]}",
            page_id=sample["page_idx"],
            pdf=PDF(file_path=str(pdf_file_path)),
            image=Image(content=page_image),
            content=DocumentContent(
                words=sample["tokens_in_page"],
                word_bboxes=BoundingBoxList(
                    value=sample["token_bboxes_in_page"], normalized=True
                ),
            ),
            annotations=[
                ExtractiveQAAnnotation(
                    qa_pairs=[
                        ExtractiveQAPair(
                            id=i,
                            question_text=question["question"],
                            answer_start=[
                                int(x) for x in question["answer_start_indices"]
                            ],
                            answer_end=[int(x) for x in question["answer_end_indices"]],
                            answer_text=question["answers"],
                        )
                        for i, question in enumerate(sample["annotations"])
                    ]
                )
            ],
        )

        # # draw bounding boxes on image for each word
        # import matplotlib.patches as patches
        # import matplotlib.pyplot as plt

        # # Create figure and axis for drawing bounding boxes
        # fig, ax = plt.subplots(1, figsize=(12, 8))
        # ax.imshow(page_image)

        # # Draw bounding boxes for each word
        # for i, bbox in enumerate(doc.content.word_bboxes.value):
        #     # bbox format: [x_min, y_min, x_max, y_max] (normalized)
        #     x_min = bbox[0] * page_image.width
        #     y_min = bbox[1] * page_image.height
        #     width = (bbox[2] - bbox[0]) * page_image.width
        #     height = (bbox[3] - bbox[1]) * page_image.height

        #     # Create rectangle patch
        #     rect = patches.Rectangle(
        #         (x_min, y_min),
        #         width,
        #         height,
        #         linewidth=1,
        #         edgecolor="red",
        #         facecolor="none",
        #         alpha=0.7,
        #     )
        #     ax.add_patch(rect)

        # ax.set_xlim(0, page_image.width)
        # ax.set_ylim(page_image.height, 0)  # Flip y-axis for image coordinates
        # ax.axis("off")

        # # Create output directory if it doesn't exist
        # output_dir = Path("debug_images")
        # output_dir.mkdir(exist_ok=True)

        # # Save the image with bounding boxes
        # output_path = (
        #     output_dir / f"{sample['sample_id']}_page_{sample['page_idx']}_bbox.png"
        # )
        # plt.tight_layout()
        # plt.savefig(output_path, dpi=150, bbox_inches="tight")
        # plt.close()

        # logger.info(f"Saved image with bounding boxes to {output_path}")

        return doc
