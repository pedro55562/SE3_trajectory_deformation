"""Benchmark the former and corrected cached CUDA SE(3) gradient paths."""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import defaultdict
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
UAIBOT_DIR = PROJECT_DIR / "UAIbotPy"
EXPERIMENTS_DIR = PROJECT_DIR / "experiments"
for path in (PROJECT_DIR, UAIBOT_DIR, EXPERIMENTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import torch

import benchmark_gpu_house_distance as base
from benchmark_optimized_gpu_distance import Variant
from gpu_distance_bottlenecks import _build_caches, _cuda_time


def _buggy_group_pose_gradient_cached(query, result, static, generators):
    """Former cached contraction, retained only as a benchmark baseline."""
    house = static.geometry
    vertices_a = query.vertices[0]
    edges_a = query.edges[0]
    normals_a = query.normals[0]
    vertex_count_a = vertices_a.shape[0]
    vertex_count_b = house.vertices.shape[1]
    edge_count_a = edges_a.shape[0]
    edge_count_b = house.edges.shape[1]
    face_count_a = normals_a.shape[0]
    face_count_b = house.normals.shape[1]

    weights_a, weights_b = base._split_direction_weights(
        result.grad_pnv,
        edge_count_a,
        edge_count_b,
        face_count_a,
        face_count_b,
        vertex_count_a,
        vertex_count_b,
    )
    directions_a = torch.cat((edges_a, normals_a), dim=0)
    directions_a_h = torch.cat(
        (directions_a, torch.zeros_like(directions_a[:, :1])), dim=-1
    )
    vertices_a_h = torch.cat(
        (vertices_a, torch.ones_like(vertices_a[:, :1])), dim=-1
    )

    generators_t = generators.transpose(-1, -2)
    prod_a_s_a = torch.einsum(
        "di,gij,vj->dvg", directions_a_h, generators, vertices_a_h
    )
    prod_a_st_b = torch.einsum(
        "di,gji,mvj->mdvg", directions_a_h, generators_t, static.vertices_h
    )
    prod_b_s_a = torch.einsum(
        "mdi,gij,vj->mdvg", static.directions_h, generators, vertices_a_h
    )
    prod_b_st_a = torch.einsum(
        "mdi,gji,vj->mdvg", static.directions_h, generators_t, vertices_a_h
    )

    a_self = torch.einsum("mdab,dag->mg", weights_a, prod_a_s_a)
    a_cross = torch.einsum("mdab,mdbg->mg", weights_a, prod_a_st_b)
    b_from_a = torch.einsum("mdab,mdag->mg", weights_b, prod_b_s_a)
    b_from_a_t = torch.einsum("mdab,mdag->mg", weights_b, prod_b_st_a)
    b_self = torch.einsum("mdab,mdbg->mg", weights_b, static.prod_b_s_b)

    pair_grad_a = 2.0 * a_self - a_cross + b_from_a
    pair_grad_b = -a_cross + b_from_a_t - 2.0 * b_self
    return pair_grad_a, pair_grad_b


def _buggy_pose(variant, query, results):
    grad_a = torch.zeros(
        1, 6, dtype=query.vertices.dtype, device=query.vertices.device
    )
    grad_b = torch.zeros(
        variant.object_count,
        6,
        dtype=query.vertices.dtype,
        device=query.vertices.device,
    )
    for result, static in zip(results, variant.static_pose):
        pair_a, pair_b = _buggy_group_pose_gradient_cached(
            query, result, static, variant.generators
        )
        grad_a[0] += pair_a.sum(dim=0)
        grad_b.index_copy_(0, static.geometry.device_indices, pair_b)
    return grad_a, grad_b


def _full_before(variant, index, gamma, epsilon):
    query_compute, query_pose = variant.queries(index)
    results = variant.pnv(query_compute, gamma, epsilon)
    return results, _buggy_pose(variant, query_pose, results)


def _measure(operation):
    elapsed, _ = _cuda_time(operation)
    return elapsed


def _summary(values):
    return statistics.fmean(values), statistics.pstdev(values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--gamma", type=float, default=base.GAMMA)
    parser.add_argument("--epsilon", type=float, default=base.EPSILON)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    if min(args.queries, args.warmup, args.repeats) <= 0:
        raise ValueError("query, warm-up, and repeat counts must be positive")

    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"PyTorch: {torch.__version__}; CUDA runtime: {torch.version.cuda}")
    print("Building the existing house geometry and static GPU caches...")
    objects, _, _, houses, queries = _build_caches(
        max(args.queries, args.warmup), device
    )
    variant = Variant(
        "compiled_f32_cached",
        torch.float32,
        True,
        None,
        True,
        False,
        houses,
        queries,
        len(objects),
    )

    query_compute, query_pose = variant.queries(0)
    check_results = variant.pnv(query_compute, args.gamma, args.epsilon)
    fixed_gradients = variant.pose(query_pose, check_results)
    reference_gradients = base._legacy_python_pose_gradients(
        query_pose, check_results, len(objects)
    )
    torch.testing.assert_close(
        fixed_gradients[0], reference_gradients[0], rtol=2e-4, atol=2e-4
    )
    torch.testing.assert_close(
        fixed_gradients[1], reference_gradients[1], rtol=2e-4, atol=2e-4
    )
    torch.cuda.synchronize(device)
    print(
        "Corrected cached/vectorized gradient matches the corrected pairwise path."
    )

    def distance(index):
        query_compute, _ = variant.queries(index)
        return variant.distance(query_compute, args.gamma, args.epsilon)

    def pnv(index):
        query_compute, _ = variant.queries(index)
        return variant.pnv(query_compute, args.gamma, args.epsilon)

    print(
        f"Warming up {args.warmup} iterations of every CUDA path "
        "(excluded from statistics)..."
    )
    for iteration in range(args.warmup):
        index = iteration % args.queries
        distance(index)
        pnv(index)
        _full_before(variant, index, args.gamma, args.epsilon)
        variant.full(index, args.gamma, args.epsilon)
    torch.cuda.synchronize(device)

    times = {
        "before": defaultdict(list),
        "after": defaultdict(list),
    }
    for iteration in range(args.repeats):
        index = iteration % args.queries
        operations = {
            "distance": {
                "before": lambda i=index: distance(i),
                "after": lambda i=index: distance(i),
            },
            "pnv": {
                "before": lambda i=index: pnv(i),
                "after": lambda i=index: pnv(i),
            },
            "full": {
                "before": lambda i=index: _full_before(
                    variant, i, args.gamma, args.epsilon
                ),
                "after": lambda i=index: variant.full(
                    i, args.gamma, args.epsilon
                ),
            },
        }
        order = (
            ("before", "after") if iteration % 2 == 0 else ("after", "before")
        )
        for stage in ("distance", "pnv", "full"):
            for version in order:
                times[version][stage].append(_measure(operations[stage][version]))
    torch.cuda.synchronize(device)

    labels = {
        "distance": "distance only",
        "pnv": "distance + grad_pnv",
        "full": "distance + full SE(3)",
    }
    print("\nCUDA-event timing (ms)")
    print(
        "stage                      before mean/std       "
        "after mean/std        difference"
    )
    for stage in ("distance", "pnv", "full"):
        before_mean, before_std = _summary(times["before"][stage])
        after_mean, after_std = _summary(times["after"][stage])
        difference = after_mean - before_mean
        percent = 100.0 * difference / before_mean
        speedup = before_mean / after_mean
        print(
            f"{labels[stage]:25s} "
            f"{before_mean:8.3f} / {before_std:7.3f}  "
            f"{after_mean:8.3f} / {after_std:7.3f}  "
            f"{difference:+8.3f} ms ({percent:+6.2f}%), {speedup:.3f}x"
        )

    full_before = statistics.fmean(times["before"]["full"])
    full_after = statistics.fmean(times["after"]["full"])
    print(
        "\nPerformance summary: the distance and grad_pnv implementations are "
        "unchanged; only the cached/vectorized SE(3) contraction differs."
    )
    print(
        f"Full-gradient mean changed from {full_before:.3f} ms to "
        f"{full_after:.3f} ms ({full_before / full_after:.3f}x)."
    )


if __name__ == "__main__":
    main()
