# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Modified from EDM repo
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Facilities for reporting and collecting training statistics across
multiple processes and devices. The interface is designed to minimize
synchronization overhead as well as the amount of boilerplate in user
code."""


# ----------------------------------------------------------------------------

_rank = 0  # Rank of the current process.
_sync_device = (
    None  # Device to use for multiprocess communication. None = single-process.
)
_sync_called = False  # Has _sync() been called yet?
# ----------------------------------------------------------------------------


def init_multiprocessing(rank, sync_device):
    r"""Initializes `torch_utils.training_stats` for collecting statistics
    across multiple processes.

    This function must be called after
    `torch.distributed.init_process_group()` and before `Collector.update()`.
    The call is not necessary if multi-process collection is not needed.

    Args:
        rank:           Rank of the current process.
        sync_device:    PyTorch device to use for inter-process
                        communication, or None to disable multi-process
                        collection. Typically `torch.device('cuda', rank)`.
    """
    global _rank, _sync_device
    assert not _sync_called
    _rank = rank
    _sync_device = sync_device
