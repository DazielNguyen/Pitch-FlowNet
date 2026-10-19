"""Sharding without padding and exact sum/count reduction across processes (gloo, CPU)."""
import os
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import _common  # noqa: F401
from piper_train.camera_ready.distributed import ShardedSequentialSampler, gather_per_rank, reduce_sum_count


def _worker(rank, world, init_file, values, out_file):
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world)
    mine = [values[i] for i in ShardedSequentialSampler(len(values), world, rank)]
    total, count = reduce_sum_count(torch.tensor(sum(mine), dtype=torch.float64), torch.tensor(float(len(mine))))
    parts = gather_per_rank(torch.tensor([sum(mine), float(len(mine))], dtype=torch.float64))
    if rank == 0:
        torch.save({"total": total, "count": count, "parts": parts}, out_file)
    dist.barrier()
    dist.destroy_process_group()


class ShardingTest(unittest.TestCase):
    def test_each_index_once(self):
        for n in range(0, 13):
            for world in range(1, 5):
                seen = [i for r in range(world) for i in ShardedSequentialSampler(n, world, r)]
                self.assertEqual(sorted(seen), list(range(n)), (n, world))

    def test_rejects_bad_rank(self):
        with self.assertRaises(ValueError):
            ShardedSequentialSampler(5, 2, 2)


class ReductionTest(unittest.TestCase):
    def _run(self, values, world):
        with tempfile.TemporaryDirectory() as d:
            init_file, out_file = os.path.join(d, "init"), os.path.join(d, "out.pt")
            mp.spawn(_worker, args=(world, init_file, values, out_file), nprocs=world, join=True)
            return torch.load(out_file)

    def test_global_mean_equals_single_process_for_uneven_shards(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 100.0]   # 7 values over 2 ranks: 4 + 3, not symmetric
        res = self._run(values, 2)
        self.assertEqual(int(res["count"]), 7)
        self.assertAlmostEqual(float(res["total"] / res["count"]), sum(values) / len(values), places=12)
        self.assertEqual([int(p[1]) for p in res["parts"]], [4, 3])
        # The unweighted mean of per-rank means would be wrong here; the reduction must not use it.
        mean_of_means = sum(float(p[0] / p[1]) for p in res["parts"]) / 2
        self.assertNotAlmostEqual(mean_of_means, sum(values) / len(values), places=6)

    def test_without_process_group(self):
        total, count = reduce_sum_count(torch.tensor(3.0), torch.tensor(2.0))
        self.assertEqual((float(total), float(count)), (3.0, 2.0))


if __name__ == "__main__":
    unittest.main()
