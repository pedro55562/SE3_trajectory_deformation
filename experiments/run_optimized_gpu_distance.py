"""Simple runner for the selected optimized GPU house-distance path."""

from __future__ import annotations

import argparse
import statistics

import torch

from benchmark_optimized_gpu_distance import Variant
from gpu_distance_bottlenecks import _build_caches, _cuda_time


GAMMA = 2.0
EPSILON = 1e-3


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--repeats", type=int, default=100)
    parser.add_argument("--queries", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=10)
    return parser.parse_args()


def make_optimized_variant(houses, queries, object_count):
    return Variant(
        "optimized_compiled_f32_cached",
        torch.float32,
        True,   # compiled Holder
        None,   # float32 output
        True,   # cached static SE(3)
        False,  # compiled pose was slower
        houses,
        queries,
        object_count,
    )


def summarize_result(results, pose_gradient, object_count):
    distances = torch.empty(
        object_count,
        device="cuda",
        dtype=torch.float32,
    )

    for result in results:
        distances.index_copy_(
            0,
            result.house.device_indices,
            result.distances.reshape(-1),
        )

    min_distance, min_index = torch.min(distances, dim=0)
    min_index = int(min_index.item())

    query_gradient = pose_gradient[0][0].detach().cpu()
    house_gradient = pose_gradient[1][min_index].detach().cpu()

    print("\nResult from last query:")
    print(f"  minimum distance : {min_distance.item():.8f}")
    print(f"  obstacle index   : {min_index}")
    print(f"  query SE(3) grad : {query_gradient.tolist()}")
    print(f"  house SE(3) grad : {house_gradient.tolist()}")


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    device = torch.device("cuda")

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}; CUDA: {torch.version.cuda}")
    print("Building house/cache...")

    objects, _, _, houses, queries = _build_caches(
        max(args.queries, args.warmup),
        device,
    )

    object_count = len(objects)

    variant = make_optimized_variant(
        houses,
        queries,
        object_count,
    )

    print(f"House obstacles: {object_count}")
    print("Warming up...")

    for i in range(args.warmup):
        variant.full(i % args.queries, GAMMA, EPSILON)

    torch.cuda.synchronize()

    print(f"Running {args.repeats} optimized full queries...")

    times = []
    output = None

    for i in range(args.repeats):
        elapsed_ms, output = _cuda_time(
            lambda i=i: variant.full(
                i % args.queries,
                GAMMA,
                EPSILON,
            )
        )

        times.append(elapsed_ms)

    mean_ms = statistics.fmean(times)
    median_ms = statistics.median(times)
    std_ms = statistics.pstdev(times)

    print("\nPerformance:")
    print(f"  mean   : {mean_ms:.3f} ms")
    print(f"  median : {median_ms:.3f} ms")
    print(f"  std    : {std_ms:.3f} ms")
    print(f"  min    : {min(times):.3f} ms")
    print(f"  max    : {max(times):.3f} ms")
    print(f"  rate   : {1000.0 / mean_ms:.1f} Hz")

    if output is not None:
        summarize_result(
            output[0],
            output[1],
            object_count,
        )


if __name__ == "__main__":
    main()