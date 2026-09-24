"""Benchmark and validate opt-in implementation-level GPU optimizations."""

from __future__ import annotations

import argparse
import gc
import statistics
import sys
from dataclasses import dataclass
from typing import Callable
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile


PROJECT_DIR = Path(__file__).resolve().parents[2]
UAIBOT_DIR = PROJECT_DIR / "UAIbotPy"
VALIDATION_DIR = Path(__file__).resolve().parent
for path in (PROJECT_DIR, UAIBOT_DIR, VALIDATION_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import _gpu_house_distance as base
from uaibot.gpu.distance import (
    holder_distance_optimized,
    holder_distance_with_grad_optimized,
    se3_generators_cached,
)



@dataclass
class ErrorAccumulator:
    count: int = 0
    finite_count: int = 0
    actual_nonfinite: int = 0
    reference_nonfinite: int = 0
    nonfinite_mismatch: int = 0
    absolute_sum: float = 0.0
    relative_sum: float = 0.0
    absolute_max: float = 0.0
    relative_max: float = 0.0

    def update(self, actual: torch.Tensor, reference: torch.Tensor) -> None:
        actual64 = actual.detach().to(dtype=torch.float64)
        reference64 = reference.detach().to(dtype=torch.float64)
        actual_finite = torch.isfinite(actual64)
        reference_finite = torch.isfinite(reference64)
        finite = actual_finite & reference_finite
        self.count += actual64.numel()
        self.actual_nonfinite += (~actual_finite).sum().item()
        self.reference_nonfinite += (~reference_finite).sum().item()
        self.nonfinite_mismatch += (actual_finite != reference_finite).sum().item()
        self.finite_count += finite.sum().item()
        if not finite.any():
            return
        absolute = (actual64[finite] - reference64[finite]).abs()
        # A documented floor prevents division by zero while still exposing
        # errors on reference values close to zero.
        relative = absolute / reference64[finite].abs().clamp_min(1e-12)
        self.absolute_sum += absolute.sum().item()
        self.relative_sum += relative.sum().item()
        self.absolute_max = max(self.absolute_max, absolute.max().item())
        self.relative_max = max(self.relative_max, relative.max().item())

    def summary(self) -> dict[str, float]:
        return {
            "mean_abs": self.absolute_sum / self.finite_count,
            "max_abs": self.absolute_max,
            "mean_rel": self.relative_sum / self.finite_count,
            "max_rel": self.relative_max,
            "actual_nonfinite": self.actual_nonfinite,
            "reference_nonfinite": self.reference_nonfinite,
            "nonfinite_mismatch": self.nonfinite_mismatch,
        }

def _cuda_time(operation: Callable[[], object]) -> tuple[float, object]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = operation()
    end.record()
    end.synchronize()
    return start.elapsed_time(end), result

def _build_caches(query_count: int, device: torch.device):
    house_objects = base.create_house(
        base.build_plan_data()
    )
    house_polyhedra = [
        base._as_gpu_compatible_polyhedron(obj)
        for obj in house_objects
    ]
    query_obstacle = base.ub.Box(
        htm=np.identity(4),
        name="precision_experiment_query",
        width=base.QUERY_OBSTACLE_SIZE[0],
        depth=base.QUERY_OBSTACLE_SIZE[1],
        height=base.QUERY_OBSTACLE_SIZE[2],
        color="red",
    )
    query_poses = base._make_query_poses(query_count, seed=0)
    house_components = [
        base.extract_VEF(obj) for obj in house_polyhedra
    ]
    query_components = [
        base.extract_VEF(query_obstacle, pose) for pose in query_poses
    ]
    house_cpu = base._group_and_stack(house_components)
    query_cpu = base._group_and_stack(query_components)[0]
    house_cached = [
        base._batch_to_device(batch, device) for batch in house_cpu
    ]
    query_cached = base._batch_to_device(query_cpu, device)
    torch.cuda.synchronize(device)
    return house_polyhedra, house_cpu, query_cpu, house_cached, query_cached

def _dtype_batch(batch, dtype: torch.dtype):
    if dtype == torch.float32:
        vertices, edges, normals = batch.vertices, batch.edges, batch.normals
    elif dtype == torch.float64:
        vertices, edges, normals = (
            batch.vertices_f64,
            batch.edges_f64,
            batch.normals_f64,
        )
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return base.GeometryBatch(
        indices=batch.indices,
        vertices=vertices,
        edges=edges,
        normals=normals,
        device_indices=batch.device_indices,
    )


def _query_at(batch, index: int):
    return base.GeometryBatch(
        indices=[index],
        vertices=batch.vertices[index : index + 1],
        edges=batch.edges[index : index + 1],
        normals=batch.normals[index : index + 1],
    )


class Variant:
    def __init__(
        self,
        name,
        compute_dtype,
        compiled,
        output_dtype,
        cached_pose,
        compiled_pose,
        houses_source,
        queries_source,
        object_count,
    ):
        self.name = name
        self.compute_dtype = compute_dtype
        self.compiled = compiled
        self.output_dtype = output_dtype
        self.cached_pose = cached_pose
        self.compiled_pose = compiled_pose
        self.object_count = object_count
        self.houses_compute = [
            _dtype_batch(batch, compute_dtype) for batch in houses_source
        ]
        self.queries_compute = _dtype_batch(queries_source, compute_dtype)
        self.houses_pose = [
            _dtype_batch(
                batch,
                output_dtype if output_dtype is not None else compute_dtype,
            )
            for batch in houses_source
        ]
        self.queries_pose = _dtype_batch(
            queries_source,
            output_dtype if output_dtype is not None else compute_dtype,
        )
        self.generators = se3_generators_cached(
            self.queries_pose.vertices.device, self.queries_pose.vertices.dtype
        )
        self.static_pose = (
            base._build_static_pose_cache(self.houses_pose, self.generators)
            if cached_pose
            else None
        )

    def queries(self, index):
        return _query_at(self.queries_compute, index), _query_at(
            self.queries_pose, index
        )

    def distance(self, query_compute, gamma, eps):
        return [
            holder_distance_optimized(
                query_compute.vertices,
                query_compute.edges,
                query_compute.normals,
                house.vertices,
                house.edges,
                house.normals,
                gamma,
                eps,
                compile_aggregation=self.compiled,
            )
            for house in self.houses_compute
        ]

    def pnv(self, query_compute, gamma, eps):
        results = []
        for compute_house, pose_house in zip(
            self.houses_compute, self.houses_pose
        ):
            distance, grad = holder_distance_with_grad_optimized(
                query_compute.vertices,
                query_compute.edges,
                query_compute.normals,
                compute_house.vertices,
                compute_house.edges,
                compute_house.normals,
                gamma,
                eps,
                dtype=self.compute_dtype,
                output_dtype=self.output_dtype,
                compile_aggregation=self.compiled,
            )
            results.append(base.PNVResult(pose_house, distance, grad))
        return results

    def pose(self, query_pose, results):
        if self.cached_pose:
            return base._vectorized_pose_gradients_cached(
                query_pose,
                results,
                self.static_pose,
                self.object_count,
                self.generators,
                self.compiled_pose,
            )
        return base._vectorized_pose_gradients(
            query_pose, results, self.object_count
        )

    def full(self, index, gamma, eps):
        query_compute, query_pose = self.queries(index)
        results = self.pnv(query_compute, gamma, eps)
        pose = self.pose(query_pose, results)
        return results, pose


class ReferenceVariant:
    name = "reference_production"

    def __init__(self, houses, queries, object_count):
        self.houses = houses
        self.queries_source = queries
        self.object_count = object_count

    def queries(self, index):
        query = base._query_at(self.queries_source, index)
        return query, query

    def distance(self, query_compute, gamma, eps):
        return base._distance_only(query_compute, self.houses, gamma, eps)

    def pnv(self, query_compute, gamma, eps):
        return base._distance_and_pnv_gradient(
            query_compute, self.houses, gamma, eps
        )

    def pose(self, query_pose, results):
        return base._vectorized_pose_gradients(
            query_pose, results, self.object_count
        )

    def full(self, index, gamma, eps):
        query_compute, query_pose = self.queries(index)
        results = self.pnv(query_compute, gamma, eps)
        pose = self.pose(query_pose, results)
        return results, pose


def benchmark_variant(variant, query_count, warmup, repeats, gamma, eps):
    for index in range(warmup):
        query_compute, query_pose = variant.queries(index % query_count)
        variant.distance(query_compute, gamma, eps)
        results = variant.pnv(query_compute, gamma, eps)
        variant.pose(query_pose, results)
    torch.cuda.synchronize()

    times = defaultdict(list)
    for index in range(repeats):
        query_compute, query_pose = variant.queries(index % query_count)
        elapsed, _ = _cuda_time(
            lambda: variant.distance(query_compute, gamma, eps)
        )
        times["distance"].append(elapsed)
        elapsed, results = _cuda_time(
            lambda: variant.pnv(query_compute, gamma, eps)
        )
        times["pnv"].append(elapsed)
        elapsed, _ = _cuda_time(lambda: variant.pose(query_pose, results))
        times["pose"].append(elapsed)
        elapsed, _ = _cuda_time(
            lambda: variant.full(index % query_count, gamma, eps)
        )
        times["full"].append(elapsed)
    return times


def _time_summary(values):
    return (
        statistics.fmean(values),
        statistics.median(values),
        statistics.pstdev(values),
    )


def print_benchmarks(all_times):
    print("\nCUDA-event timings (ms):")
    print("  variant                      stage       mean    median       std")
    for variant_name, times in all_times.items():
        for stage in ("distance", "pnv", "pose", "full"):
            mean, median, std = _time_summary(times[stage])
            print(
                f"  {variant_name:28s} {stage:9s} "
                f"{mean:9.3f} {median:9.3f} {std:9.3f}"
            )


def _profile_variant(variant, gamma, eps):
    variant.full(0, gamma, eps)
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    starting_memory = torch.cuda.memory_allocated()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        outputs = variant.full(0, gamma, eps)
        torch.cuda.synchronize()

    keys = prof.key_averages()
    pow_events = [event for event in keys if event.key == "aten::pow"]
    pow_us = sum(
        float(getattr(event, "self_device_time_total", 0.0) or 0.0)
        for event in pow_events
    )
    cuda_events = [
        event
        for event in prof.events()
        if str(getattr(event, "device_type", "")).endswith("CUDA")
    ]
    kernel_cuda_us = sum(
        float(
            getattr(event, "self_device_time_total", 0.0)
            or getattr(event, "device_time_total", 0.0)
            or 0.0
        )
        for event in cuda_events
    )
    peak_delta = torch.cuda.max_memory_allocated() - starting_memory
    peak_absolute = torch.cuda.max_memory_allocated()
    del outputs
    return {
        "self_cuda_ms": kernel_cuda_us / 1000.0,
        "kernel_count": len(cuda_events),
        "pow_calls": sum(event.count for event in pow_events),
        "pow_cuda_ms": pow_us / 1000.0,
        "peak_delta_mib": peak_delta / 2**20,
        "resident_mib": starting_memory / 2**20,
        "peak_absolute_mib": peak_absolute / 2**20,
    }


def compare_outputs(actual, reference, accumulators):
    actual_results, actual_pose = actual
    reference_results, reference_pose = reference
    for actual_result, reference_result in zip(actual_results, reference_results):
        accumulators["distance"].update(
            actual_result.distances, reference_result.distances
        )
        accumulators["grad_pnv"].update(
            actual_result.grad_pnv, reference_result.grad_pnv
        )
    accumulators["se3_query"].update(actual_pose[0], reference_pose[0])
    accumulators["se3_house"].update(actual_pose[1], reference_pose[1])


def validate_variants(reference, variants, query_count, gamma, eps):
    accumulators = {
        variant.name: {
            "distance": ErrorAccumulator(),
            "grad_pnv": ErrorAccumulator(),
            "se3_query": ErrorAccumulator(),
            "se3_house": ErrorAccumulator(),
        }
        for variant in variants
    }
    near_counts = {"1e-2": 0, "1e-3": 0, "1e-4": 0}
    for index in range(query_count):
        reference_output = reference.full(index, gamma, eps)
        minimum_absolute = min(
            result.distances.abs().min().item()
            for result in reference_output[0]
        )
        for label, threshold in (("1e-2", 1e-2), ("1e-3", 1e-3), ("1e-4", 1e-4)):
            near_counts[label] += minimum_absolute <= threshold
        for variant in variants:
            compare_outputs(
                variant.full(index, gamma, eps),
                reference_output,
                accumulators[variant.name],
            )
    torch.cuda.synchronize()
    print(f"\nValidation over {query_count} cached random configurations")
    print(f"  near-contact counts by min |distance|: {near_counts}")
    print("  variant/output                  mean_abs      max_abs     mean_rel      max_rel")
    for variant in variants:
        for output_name, accumulator in accumulators[variant.name].items():
            summary = accumulator.summary()
            print(
                f"  {variant.name + '/' + output_name:31s} "
                f"{summary['mean_abs']:12.4e} {summary['max_abs']:12.4e} "
                f"{summary['mean_rel']:12.4e} {summary['max_rel']:12.4e} "
                f"nonfinite(actual/ref/mismatch)="
                f"{summary['actual_nonfinite']}/"
                f"{summary['reference_nonfinite']}/"
                f"{summary['nonfinite_mismatch']}"
            )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--gamma", type=float, default=base.GAMMA)
    parser.add_argument("--epsilon", type=float, default=base.EPSILON)
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--skip-profiler", action="store_true")
    parser.add_argument(
        "--selection",
        choices=("all", "reference", "final"),
        default="all",
        help="Limit timing/profiling to a reference-only or final-only process.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if min(args.queries, args.warmup, args.repeats) <= 0:
        raise ValueError("query and repeat counts must be positive")

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}; CUDA: {torch.version.cuda}")
    objects, _, _, houses, queries = _build_caches(
        max(args.queries, args.warmup), torch.device("cuda")
    )
    object_count = len(objects)
    reference = ReferenceVariant(houses, queries, object_count)
    variants = [
        Variant(
            "optimized_compiled_f32_cached",
            torch.float32,
            True,
            None,
            True,
            False,
            houses,
            queries,
            object_count,
        ),
    ]

    selected = [reference, *variants]
    if args.selection == "reference":
        selected = [reference]
    elif args.selection == "final":
        selected = [variants[0]]

    all_times = {}
    for variant in selected:
        print(f"Benchmarking {variant.name}...", flush=True)
        all_times[variant.name] = benchmark_variant(
            variant,
            args.queries,
            args.warmup,
            args.repeats,
            args.gamma,
            args.epsilon,
        )
    print_benchmarks(all_times)

    if not args.skip_profiler:
        print("\nProfiler summary for full cached queries:")
        print(
            "  variant                      CUDA ms   kernels  pow calls    pow ms  "
            "resident MiB  peak MiB  query delta MiB"
        )
        for variant in selected:
            result = _profile_variant(variant, args.gamma, args.epsilon)
            print(
                f"  {variant.name:28s} {result['self_cuda_ms']:9.3f} "
                f"{result['kernel_count']:9d} {result['pow_calls']:10d} "
                f"{result['pow_cuda_ms']:9.3f} {result['resident_mib']:12.3f} "
                f"{result['peak_absolute_mib']:9.3f} {result['peak_delta_mib']:15.3f}"
            )

    if not args.skip_validation:
        validation_variants = variants if args.selection == "all" else []
        if args.selection == "final":
            validation_variants = [variants[0]]
        if validation_variants:
            validate_variants(
                reference,
                validation_variants,
                args.queries,
                args.gamma,
                args.epsilon,
            )


if __name__ == "__main__":
    main()
