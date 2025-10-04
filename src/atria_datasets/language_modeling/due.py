from collections.abc import Iterable
from pathlib import Path

from atria_core.logger import get_logger
from atria_core.types import (
    PDF,
    BoundingBoxList,
    DatasetLabels,
    DatasetMetadata,
    DatasetSplitType,
    DocumentInstance,
)
from atria_core.types.generic.annotations import GenerativeQAAnnotation
from atria_core.types.generic.document_content import DocumentContent
from atria_core.types.generic.question_answer_pair import GenerativeQAItem

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
_DUE_DATASETS = [
    "DocVQA",
    "PWC",
    "DeepForm",
    "TabFact",
    "WikiTableQuestions",
    "InfographicsVQA",
    "KleisterCharity",
]


class DueBenchmarkConfig(AtriaDatasetConfig):
    """BuilderConfig for DueBenchmark. This configuration is taken from generate_memmaps from the DueBenchmark library."""

    # tokenizer args
    model_path: str = "google-t5/t5-large"
    model_type: str = "t5"
    use_fast_tokenizer: bool = True
    max_encoder_length: int = 1024

    # corpus args
    unescape_prefix: bool = False
    unescape_values: bool = True
    use_prefix: bool = True
    prefix_separator: str = ":"
    values_separator: str = "|"
    single_property: bool = True
    use_none_answers: bool = False
    use_fast_tokenizer: bool = True
    limit: int = -1
    case_augmentation: bool = False
    segment_levels: tuple = ("tokens", "pages")
    long_page_strategy: str = "FIRST_PART"
    ocr_engine: str = "microsoft_cv"
    lowercase_expected: bool = False
    lowercase_input: bool = False
    train_strategy: str = "all_items"
    dev_strategy: str = "concat"
    test_strategy: str = "concat"
    augment_tokens_from_file: str = ""
    img_matrix_order: int = 0
    processes: int = 1
    imap_chunksize: int = 100
    skip_text_tokens: bool = False


class SplitIterator:
    def __init__(
        self, split: DatasetSplitType, data_dir: str, config: DueBenchmarkConfig
    ):
        from benchmarker.data.reader import Corpus, qa_strategies
        from benchmarker.data.reader.benchmark_dataset import BenchmarkDataset

        self._config = config

        # make path for preprocessed data
        self._data_dir = data_dir
        corpus = Corpus(
            unescape_prefix=config.unescape_prefix,
            unescape_values=config.unescape_values,
            use_prefix=config.use_prefix,
            prefix_separator=config.prefix_separator,
            values_separator=config.values_separator,
            single_property=config.single_property,
            use_none_answers=config.use_none_answers,
            case_augmentation=config.case_augmentation,
            lowercase_expected=config.lowercase_expected,
            lowercase_input=config.lowercase_input,
            train_strategy=getattr(qa_strategies, config.train_strategy),
            dev_strategy=getattr(qa_strategies, config.dev_strategy),
            test_strategy=getattr(qa_strategies, config.test_strategy),
            augment_tokens_from_file=config.augment_tokens_from_file,
        )

        split = "dev" if split.value == "validation" else split.value
        extracted_name = config.config_name
        if config.config_name == "InfographicsVQA":
            extracted_name = "infographics_vqa"
        if config.config_name == "DocVQA":
            extracted_name = "docvqa"
        if config.config_name == "PWC":
            extracted_name = "AxCell"
        if config.config_name == "KleisterCharity":
            extracted_name = "kleister-charity"
        benchmark_dataset = BenchmarkDataset(
            directory=Path(data_dir)
            / "datasets"
            / config.config_name  # base name then extracted name
            / "aws_neurips_time"
            / extracted_name,
            split=split,
            ocr=config.ocr_engine,
            segment_levels=config.segment_levels,
        )

        setattr(corpus, "_" + split, benchmark_dataset)
        self._dataset = getattr(corpus, split)
        if self._dataset is None:
            raise ValueError(f"Split {split} not found in corpus")

        # self._data_converter = T5DownstreamDataConverter(
        #     load_tokenizer(
        #         Path(self._config.model_path),
        #         model_type=self._config.model_type,
        #         convert_to_fast_tokenizer=self._config.use_fast_tokenizer,
        #     ),
        #     segment_levels=self._config.segment_levels,
        #     max_seq_length=self._config.max_encoder_length,
        #     long_page_strategy=LongPageStrategy(self._config.long_page_strategy),
        #     img_matrix_order=self._config.img_matrix_order,
        #     processes=self._config.processes,
        #     imap_chunksize=self._config.imap_chunksize,
        #     skip_text_tokens=self._config.skip_text_tokens,
        # )

        # self._dataset = self._data_converter.generate_features(self._dataset)

    def __iter__(self):
        last_sample_id = None
        grouped = []
        for i, sample in enumerate(self._dataset):
            if self._config.limit > 0 and i >= self._config.limit:
                break

            current_sample_id = sample.identifier

            if last_sample_id is not None and last_sample_id != current_sample_id:
                grouped = {
                    "sample_id": grouped[0].identifier,
                    "document_2d": grouped[0].document_2d,
                    "annotations": [
                        {
                            "input_prefix": g.input_prefix,
                            "output_prefix": g.output_prefix,
                            "output": g.output,
                        }
                        for g in grouped
                    ],
                }

                # we remap all due benchmark keys to what we require in our datasets
                pdf_file_path = (
                    Path(self._data_dir)
                    / "pdfs"
                    / self._config.config_name  # base name then extracted name
                    / self._config.config_name
                    / (grouped["sample_id"])
                )
                if not str(pdf_file_path).endswith(".pdf"):
                    pdf_file_path = Path(str(pdf_file_path) + ".pdf")

                # if the file does not exist we ignore it
                if pdf_file_path.exists():
                    yield grouped
                else:
                    logger.warning(f"File {pdf_file_path} not found. Skipping it")
                grouped = []

            grouped.append(sample)
            last_sample_id = sample.identifier

    def __len__(self) -> int:
        return len(self._dataset)


@DATASET.register(
    "due_benchmark",
    configs=[
        DueBenchmarkConfig(config_name="DocVQA", train_strategy="all_items"),
        DueBenchmarkConfig(
            config_name="PWC", train_strategy="concat", ocr_engine="tesseract"
        ),
        DueBenchmarkConfig(config_name="DeepForm", train_strategy="all_items"),
        DueBenchmarkConfig(
            config_name="TabFact", train_strategy="all_items", ocr_engine="tesseract"
        ),
        DueBenchmarkConfig(config_name="WikiTableQuestions", train_strategy="concat"),
        DueBenchmarkConfig(config_name="InfographicsVQA", train_strategy="all_items"),
        DueBenchmarkConfig(config_name="KleisterCharity", train_strategy="all_items"),
    ],
)
class DueBenchmark(AtriaDocumentDataset):
    __config_cls__ = DueBenchmarkConfig

    def _download_urls(self) -> list[str]:
        return {
            f"{key}/{self.config.config_name}": url.format(
                config_name=self.config.config_name
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

    def _input_transform(self, sample: tuple[Path, Path, int]) -> DocumentInstance:
        import uuid

        # we remap all due benchmark keys to what we require in our datasets
        pdf_file_path = (
            Path(self._data_dir)
            / "pdfs"
            / self.config.config_name  # base name then extracted name
            / self.config.config_name
            / (sample["sample_id"])
        )

        # if the file does not exist we ignore it
        if not str(pdf_file_path).endswith(".pdf"):
            pdf_file_path = Path(str(pdf_file_path) + ".pdf")

        # resize image to max height and width
        doc = DocumentInstance(
            sample_id=sample["sample_id"] + f"-{uuid.uuid4().hex[:4]}",
            pdf=PDF(file_path=str(pdf_file_path)),
            content=DocumentContent(
                words=sample["document_2d"].tokens,
                word_bboxes=BoundingBoxList(
                    value=sample["document_2d"]
                    .seg_data["tokens"]["org_bboxes"]
                    .tolist(),
                    normalized=False,
                ),
            ),
            annotations=[
                GenerativeQAAnnotation(
                    qa_pairs=[
                        GenerativeQAItem(
                            input_prefix=ann["input_prefix"],
                            output_prefix=ann["output_prefix"],
                            output=ann["output"],
                        )
                        for ann in sample["annotations"]
                    ]
                )
            ],
        )
        return doc
