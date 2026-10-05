"""
Group Batch Sampler Module

This module defines the `GroupBatchSampler` class, which is a batch sampler
that groups input samples based on predefined group IDs. It ensures that
mini-batches contain elements from the same group and maintains an ordering
close to the original sampler.

Classes:
    - GroupBatchSampler: A batch sampler that groups samples based on group IDs.

Dependencies:
    - collections.defaultdict: For managing grouped buffers.
    - torch.utils.data.sampler.BatchSampler: Base class for batch samplers.
    - torch.utils.data.sampler.Sampler: Base class for samplers.
    - atria_datasets.core.batch_samplers.utilities: Utility functions for sampling.

Author: Your Name (your.email@example.com)
Date: 2025-04-07
Version: 1.0.0
License: MIT
"""

from collections import defaultdict

from torch.utils.data.sampler import BatchSampler, Sampler


class GroupBatchSampler(BatchSampler):
    """
    A batch sampler that groups input samples based on predefined group IDs.

    This sampler wraps another sampler to yield mini-batches of indices, ensuring
    that each batch contains elements from the same group. It also tries to maintain
    an ordering close to the original sampler.

    Attributes:
        sampler (Sampler): The base sampler that provides dataset indices.
        batch_size (int): The size of each mini-batch.
        drop_last (bool): Whether to drop the last incomplete batch if the dataset
            size is not divisible by the batch size.
        group_ids (list[int]): A list of group IDs corresponding to the dataset samples.
    """

    def __init__(self, sampler: Sampler, batch_size: int, drop_last: bool) -> None:
        """
        Initializes the GroupBatchSampler.

        Args:
            sampler (Sampler): The base sampler that provides dataset indices.
            batch_size (int): The size of each mini-batch.
            drop_last (bool): Whether to drop the last incomplete batch if the dataset
                size is not divisible by the batch size.
        """
        super().__init__(sampler, batch_size, drop_last)

    def __iter__(self):
        """
        Iterates over the dataset and yields mini-batches of indices.

        This method groups samples based on their group IDs and ensures that
        each mini-batch contains elements from the same group. If there are
        remaining elements that do not satisfy the group criteria, they are
        included in the final batches to ensure deterministic sampling.

        Yields:
            list[int]: A mini-batch of indices from the same group.

        Raises:
            AssertionError: If the buffer size exceeds the batch size.
        """
        from atria_datasets.core.batch_samplers.utilities import _repeat_to_at_least

        buffer_per_group = defaultdict(list)
        samples_per_group = defaultdict(list)

        num_batches = 0
        for idx in self.sampler:
            group_id = self.group_ids[idx]
            buffer_per_group[group_id].append(idx)
            samples_per_group[group_id].append(idx)
            if len(buffer_per_group[group_id]) == self.batch_size:
                yield buffer_per_group[group_id]
                num_batches += 1
                del buffer_per_group[group_id]
            assert len(buffer_per_group[group_id]) < self.batch_size

        # Handle remaining elements to ensure deterministic sampling
        expected_num_batches = len(self)
        num_remaining = expected_num_batches - num_batches
        if num_remaining > 0:
            for group_id, _ in sorted(
                buffer_per_group.items(), key=lambda x: len(x[1]), reverse=True
            ):
                remaining = self.batch_size - len(buffer_per_group[group_id])
                samples_from_group_id = _repeat_to_at_least(
                    samples_per_group[group_id], remaining
                )
                buffer_per_group[group_id].extend(samples_from_group_id[:remaining])
                assert len(buffer_per_group[group_id]) == self.batch_size
                yield buffer_per_group[group_id]
                num_remaining -= 1
                if num_remaining == 0:
                    break
        assert num_remaining == 0

    def __len__(self):
        """
        Returns the number of mini-batches that can be generated.

        Returns:
            int: The number of mini-batches.

        Note:
            The number of mini-batches is calculated as the floor division of
            the dataset size by the batch size.
        """
        return len(self.sampler) // self.batch_size
