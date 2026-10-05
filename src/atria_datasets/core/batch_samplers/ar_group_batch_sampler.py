"""
Aspect Ratio Group Batch Sampler Module

This module defines the `AspectRatioGroupBatchSampler` class, which is a batch
sampler that groups input sample images based on their aspect ratios. It extends
the `GroupBatchSampler` and uses a grouping strategy to improve training efficiency
for models sensitive to input dimensions.

Classes:
    - AspectRatioGroupBatchSampler: A batch sampler that groups samples based on
        their aspect ratios.

Dependencies:
    - torch: For dataset sampling utilities.
    - atria_datasets.core.batch_samplers.group_batch_sampler: Base class for group batch samplers.
    - atria_datasets.core.batch_samplers.utilities: Utility functions for creating aspect ratio groups.

Author: Your Name (your.email@example.com)
Date: 2025-04-07
Version: 1.0.0
License: MIT
"""

from typing import TYPE_CHECKING

from atria_datasets.core.batch_samplers.group_batch_sampler import GroupBatchSampler
from atria_datasets.core.batch_samplers.utilities import _create_aspect_ratio_groups

if TYPE_CHECKING:
    from torch.utils.data.sampler import RandomSampler, SequentialSampler


class AspectRatioGroupBatchSampler(GroupBatchSampler):
    """
    A batch sampler that groups input sample images based on their aspect ratio.

    This sampler extends the `GroupBatchSampler` and uses a grouping strategy
    based on aspect ratios of the input data. It ensures that samples with
    similar aspect ratios are grouped together, which can improve training
    efficiency for models sensitive to input dimensions.

    Attributes:
        group_factor (List[int]): A list of integers used to define the aspect
            ratio groups. The grouping is determined by these factors.
        group_ids (List[int]): A list of group IDs corresponding to the aspect
            ratio groups.
    """

    def __init__(
        self,
        sampler: SequentialSampler | RandomSampler,
        batch_size: int,
        drop_last: bool,
        group_factor: list[int],
    ) -> None:
        """
        Initializes the AspectRatioGroupBatchSampler.

        Args:
            sampler (Sampler): The base sampler that provides the indices of
                the dataset.
            batch_size (int): The number of samples per batch.
            drop_last (bool): Whether to drop the last incomplete batch if the
                dataset size is not divisible by the batch size.
            group_factor (List[int]): A list of integers used to define the
                aspect ratio groups.

        Attributes:
            group_factor (List[int]): Stores the aspect ratio grouping factors.
            group_ids (List[int]): Stores the group IDs for the dataset samples.
        """
        super().__init__(sampler, batch_size, drop_last)
        self.group_factor = group_factor
        self.group_ids = _create_aspect_ratio_groups(
            self.sampler.data_source,  # type: ignore[union-attr]
            k=self.group_factor,
        )
