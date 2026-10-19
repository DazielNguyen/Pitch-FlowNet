"""Samplers and reductions for exact epoch metrics under DDP."""
from typing import List, Tuple

import torch
import torch.distributed as dist
from torch.utils.data import Sampler


class ShardedSequentialSampler(Sampler):
    """Each index exactly once across all ranks, in a fixed order, without padding.

    Rank r receives indices r, r + W, r + 2W, ... Ranks may receive different counts; nothing is
    duplicated, so per-sample sums can be added across ranks without double counting.
    """

    def __init__(self, num_samples: int, num_replicas: int, rank: int):
        if not 0 <= rank < num_replicas:
            raise ValueError("rank must be in [0, num_replicas)")
        self.num_samples, self.num_replicas, self.rank = num_samples, num_replicas, rank

    def positions(self) -> List[int]:
        return list(range(self.rank, self.num_samples, self.num_replicas))

    def __iter__(self):
        return iter(self.positions())

    def __len__(self):
        return len(self.positions())


def _initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def reduce_sum_count(total: torch.Tensor, count: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sum (total, count) over all ranks in float64. Without a process group, returns the inputs."""
    t = torch.stack([total.detach().double().reshape(()), count.detach().double().reshape(())])
    if _initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t[0], t[1]


def gather_per_rank(values: torch.Tensor) -> List[torch.Tensor]:
    """all_gather a small 1-D float64 tensor; without a process group, a one-element list."""
    v = values.detach().double()
    if not _initialized():
        return [v]
    out = [torch.zeros_like(v) for _ in range(dist.get_world_size())]
    dist.all_gather(out, v)
    return out
