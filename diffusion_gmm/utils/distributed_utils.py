# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/


import os

import torch
import torch.distributed as dist

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


def init_distributed():
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "29500"
    if "RANK" not in os.environ:
        os.environ["RANK"] = "0"
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = "0"
    if "WORLD_SIZE" not in os.environ:
        os.environ["WORLD_SIZE"] = "1"

    backend = "gloo" if os.name == "nt" else "nccl"
    dist.init_process_group(backend=backend, init_method="env://")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))

    sync_device = torch.device("cuda") if get_world_size() > 1 else None
    init_multiprocessing(rank=get_rank(), sync_device=sync_device)


def get_rank():
    return dist.get_rank() if dist.is_initialized() else 0


def get_world_size():
    return dist.get_world_size() if dist.is_initialized() else 1


def should_stop():
    return False


def update_progress(cur, total):
    _ = cur, total


def print0(*args, **kwargs):
    if get_rank() == 0:
        print(*args, **kwargs)
