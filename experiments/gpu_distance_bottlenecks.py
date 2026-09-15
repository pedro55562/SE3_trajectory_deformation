"""Isolated GPU experiments for the cached house distance benchmark.

This file intentionally does not alter or monkey-patch the production UAIbot
implementation.  It reproduces the current formulas locally where a selectable
dtype or finer CUDA-event boundaries are required.
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile


PROJECT_DIR = Path(__file__).resolve().parents[1]
UAIBOT_DIR = PROJECT_DIR / "UAIbotPy"
for path in (PROJECT_DIR, UAIBOT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import benchmark_gpu_house_distance as benchmark_module
from uaibot.gpu.distance import _aggregate_holder_distance, holder_distance
from uaibot.gpu.functional import holder_max, holder_min, shaping_function
from uaibot.gpu.pairwise import (
    pairwise_concat_objects,
    pairwise_difference,
    pairwise_direction_vertex_dot,
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


def _fmt_ms(values: list[float]) -> str:
    return (
        f"{statistics.fmean(values):9.3f} ms mean  "
        f"{statistics.median(values):9.3f} ms median  "
        f"{statistics.pstdev(values):8.3f} ms std"
    )


def _cuda_time(operation: Callable[[], object]) -> tuple[float, object]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = operation()
    end.record()
    end.synchronize()
    return start.elapsed_time(end), result


def _native_distance_with_grad(
    vertex_a: torch.Tensor,
    edges_a: torch.Tensor,
    normals_a: torch.Tensor,
    vertex_b: torch.Tensor,
    edges_b: torch.Tensor,
    normals_b: torch.Tensor,
    gamma: float,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Current production formula, but retain the input dtype in the outputs."""
    with torch.no_grad():
        edges_ab = pairwise_concat_objects(edges_a, edges_b)
        edges_ab = torch.cat((edges_ab, -edges_ab), dim=2)
        face_normals_ab = pairwise_concat_objects(normals_a, normals_b)
        vertices_ab = pairwise_difference(vertex_a, vertex_b)
        directions_ab = torch.cat((edges_ab, face_normals_ab), dim=2)
        pnv = pairwise_direction_vertex_dot(directions_ab, vertices_ab)

    pnv_leaf = pnv.detach().requires_grad_(True)
    x1 = holder_min(pnv_leaf, gamma, dim=3, eps=eps)
    x1_shaped = shaping_function(x1, gamma, eps=eps)
    x2 = holder_max(x1_shaped, gamma, dim=2, eps=eps)
    distance = shaping_function(x2, gamma, eps=eps)
    distance.sum().backward()
    return distance.detach(), pnv_leaf.grad.detach()


def _as_dtype_batch(batch, dtype: torch.dtype):
    """Create a GeometryBatch referencing the requested cached representation."""
    return benchmark_module.GeometryBatch(
        indices=batch.indices,
        vertices=batch.vertices.to(dtype=dtype),
        edges=batch.edges.to(dtype=dtype),
        normals=batch.normals.to(dtype=dtype),
        device_indices=batch.device_indices,
        vertices_f64=batch.vertices_f64,
        edges_f64=batch.edges_f64,
        normals_f64=batch.normals_f64,
    )


def _query_view(batch, index: int):
    return benchmark_module.GeometryBatch(
        indices=[index],
        vertices=batch.vertices[index : index + 1],
        edges=batch.edges[index : index + 1],
        normals=batch.normals[index : index + 1],
        device_indices=None,
        vertices_f64=None,
        edges_f64=None,
        normals_f64=None,
    )


def _distance_dtype(query, houses, gamma: float, eps: float):
    return [
        holder_distance(
            query.vertices,
            query.edges,
            query.normals,
            house.vertices,
            house.edges,
            house.normals,
            gamma,
            eps,
        )
        for house in houses
    ]


def _pnv_dtype(query, houses, gamma: float, eps: float):
    results = []
    for house in houses:
        distance, grad_pnv = _native_distance_with_grad(
            query.vertices,
            query.edges,
            query.normals,
            house.vertices,
            house.edges,
            house.normals,
            gamma,
            eps,
        )
        results.append(benchmark_module.PNVResult(house, distance, grad_pnv))
    return results


def _pose_dtype(query, pnv_results, object_count: int):
    return benchmark_module._vectorized_pose_gradients(
        query, pnv_results, object_count
    )


def _build_caches(query_count: int, device: torch.device):
    house_objects = benchmark_module.create_house(
        benchmark_module.build_plan_data()
    )
    house_polyhedra = [
        benchmark_module._as_gpu_compatible_polyhedron(obj)
        for obj in house_objects
    ]
    query_obstacle = benchmark_module.ub.Box(
        htm=np.identity(4),
        name="precision_experiment_query",
        width=benchmark_module.QUERY_OBSTACLE_SIZE[0],
        depth=benchmark_module.QUERY_OBSTACLE_SIZE[1],
        height=benchmark_module.QUERY_OBSTACLE_SIZE[2],
        color="red",
    )
    query_poses = benchmark_module._make_query_poses(query_count, seed=0)
    house_components = [
        benchmark_module.extract_VEF(obj) for obj in house_polyhedra
    ]
    query_components = [
        benchmark_module.extract_VEF(query_obstacle, pose) for pose in query_poses
    ]
    house_cpu = benchmark_module._group_and_stack(house_components)
    query_cpu = benchmark_module._group_and_stack(query_components)[0]
    house_cached = [
        benchmark_module._batch_to_device(batch, device) for batch in house_cpu
    ]
    query_cached = benchmark_module._batch_to_device(query_cpu, device)
    torch.cuda.synchronize(device)
    return house_polyhedra, house_cpu, query_cpu, house_cached, query_cached


def precision_experiment(
    houses_cached,
    query_cached,
    object_count: int,
    query_count: int,
    warmup: int,
    repeats: int,
    gamma: float,
    eps: float,
) -> None:
    print("\n=== 1. Precision experiment ===")
    caches = {}
    for dtype in (torch.float64, torch.float32):
        caches[dtype] = (
            [_as_dtype_batch(batch, dtype) for batch in houses_cached],
            _as_dtype_batch(query_cached, dtype),
        )

    timings: dict[torch.dtype, dict[str, list[float]]] = {}
    for dtype in (torch.float64, torch.float32):
        houses, queries = caches[dtype]
        stage_times = defaultdict(list)

        for index in range(warmup):
            query = _query_view(queries, index % query_count)
            _distance_dtype(query, houses, gamma, eps)
            pnv_results = _pnv_dtype(query, houses, gamma, eps)
            _pose_dtype(query, pnv_results, object_count)
        torch.cuda.synchronize()

        for index in range(repeats):
            query = _query_view(queries, index % query_count)
            elapsed, _ = _cuda_time(
                lambda: _distance_dtype(query, houses, gamma, eps)
            )
            stage_times["distance_only"].append(elapsed)

            elapsed, pnv_results = _cuda_time(
                lambda: _pnv_dtype(query, houses, gamma, eps)
            )
            stage_times["distance_pnv"].append(elapsed)

            elapsed, _ = _cuda_time(
                lambda: _pose_dtype(query, pnv_results, object_count)
            )
            stage_times["se3_chain"].append(elapsed)
            stage_times["full_query"].append(
                stage_times["distance_pnv"][-1]
                + stage_times["se3_chain"][-1]
            )

        timings[dtype] = stage_times
        print(f"\n{dtype}:")
        for name in ("distance_only", "distance_pnv", "se3_chain", "full_query"):
            print(f"  {name:16s} {_fmt_ms(stage_times[name])}")

    print("\nfloat64 / float32 mean runtime ratio:")
    for name in ("distance_only", "distance_pnv", "se3_chain", "full_query"):
        ratio = statistics.fmean(timings[torch.float64][name]) / statistics.fmean(
            timings[torch.float32][name]
        )
        print(f"  {name:16s} {ratio:9.3f}x")

    # Preserve the exact production mixture as a baseline: the Holder path is
    # float64, holder_distance_with_grad casts its outputs to float32, and the
    # SE(3) chain rule consumes the original float32 geometry.
    production_times = defaultdict(list)
    for index in range(warmup):
        query = benchmark_module._query_at(query_cached, index % query_count)
        benchmark_module._distance_only(query, houses_cached, gamma, eps)
        results = benchmark_module._distance_and_pnv_gradient(
            query, houses_cached, gamma, eps
        )
        benchmark_module._vectorized_pose_gradients(query, results, object_count)
    torch.cuda.synchronize()
    for index in range(repeats):
        query = benchmark_module._query_at(query_cached, index % query_count)
        elapsed, _ = _cuda_time(
            lambda: benchmark_module._distance_only(
                query, houses_cached, gamma, eps
            )
        )
        production_times["distance_only"].append(elapsed)
        elapsed, results = _cuda_time(
            lambda: benchmark_module._distance_and_pnv_gradient(
                query, houses_cached, gamma, eps
            )
        )
        production_times["distance_pnv"].append(elapsed)
        elapsed, _ = _cuda_time(
            lambda: benchmark_module._vectorized_pose_gradients(
                query, results, object_count
            )
        )
        production_times["se3_chain"].append(elapsed)
        production_times["full_query"].append(
            production_times["distance_pnv"][-1]
            + production_times["se3_chain"][-1]
        )
    print("\nExact production mixed-dtype baseline (f64 Holder -> f32 outputs/SE3):")
    for name in ("distance_only", "distance_pnv", "se3_chain", "full_query"):
        print(f"  {name:16s} {_fmt_ms(production_times[name])}")

    errors = {
        "distance": ErrorAccumulator(),
        "grad_pnv": ErrorAccumulator(),
        "se3_query": ErrorAccumulator(),
        "se3_house": ErrorAccumulator(),
    }
    houses64, queries64 = caches[torch.float64]
    houses32, queries32 = caches[torch.float32]
    for index in range(query_count):
        query64 = _query_view(queries64, index)
        query32 = _query_view(queries32, index)
        results64 = _pnv_dtype(query64, houses64, gamma, eps)
        results32 = _pnv_dtype(query32, houses32, gamma, eps)
        for result32, result64 in zip(results32, results64):
            errors["distance"].update(result32.distances, result64.distances)
            errors["grad_pnv"].update(result32.grad_pnv, result64.grad_pnv)
        pose64 = _pose_dtype(query64, results64, object_count)
        pose32 = _pose_dtype(query32, results32, object_count)
        errors["se3_query"].update(pose32[0], pose64[0])
        errors["se3_house"].update(pose32[1], pose64[1])

    torch.cuda.synchronize()
    print("\nfloat32 error relative to float64 (relative denominator floor=1e-12):")
    print("  output          mean_abs       max_abs      mean_rel       max_rel")
    for name, accumulator in errors.items():
        values = accumulator.summary()
        print(
            f"  {name:13s} {values['mean_abs']:12.5e} {values['max_abs']:12.5e} "
            f"{values['mean_rel']:12.5e} {values['max_rel']:12.5e}"
        )


def _record_segment(events, name: str, operation: Callable[[], object]):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = operation()
    end.record()
    events.append((name, start, end))
    return result


def _profile_grad_batch_once(query, house, gamma: float, eps: float):
    events = []

    def construct_directions():
        edges_ab = pairwise_concat_objects(query.edges, house.edges)
        edges_ab = torch.cat((edges_ab, -edges_ab), dim=2)
        faces_ab = pairwise_concat_objects(query.normals, house.normals)
        return torch.cat((edges_ab, faces_ab), dim=2)

    with torch.no_grad():
        directions = _record_segment(
            events, "edge_normal_construction", construct_directions
        )
        vertex_differences = _record_segment(
            events,
            "vertex_differences",
            lambda: pairwise_difference(query.vertices, house.vertices),
        )
        pnv = _record_segment(
            events,
            "direction_vertex_dot",
            lambda: pairwise_direction_vertex_dot(directions, vertex_differences),
        )

    pnv_leaf = pnv.detach().requires_grad_(True)
    x1 = _record_segment(
        events,
        "holder_reduction_1",
        lambda: holder_min(pnv_leaf, gamma, dim=3, eps=eps),
    )
    x1_shaped = _record_segment(
        events,
        "shaping_1",
        lambda: shaping_function(x1, gamma, eps=eps),
    )
    x2 = _record_segment(
        events,
        "holder_reduction_2",
        lambda: holder_max(x1_shaped, gamma, dim=2, eps=eps),
    )
    distance = _record_segment(
        events,
        "shaping_2",
        lambda: shaping_function(x2, gamma, eps=eps),
    )

    def backward():
        distance.sum().backward()
        return pnv_leaf.grad

    grad = _record_segment(events, "backward_autograd", backward)
    cast_output = _record_segment(
        events,
        "output_cast",
        lambda: (distance.detach().float(), grad.float()),
    )
    events[-1][2].synchronize()
    elapsed = {name: start.elapsed_time(end) for name, start, end in events}
    return elapsed, pnv.shape, cast_output


def per_stage_experiment(
    houses_cached,
    query_cached,
    warmup: int,
    repeats: int,
    gamma: float,
    eps: float,
) -> None:
    print("\n=== 2. Per-stage CUDA-event profile (float64 production precision) ===")
    houses = [_as_dtype_batch(batch, torch.float64) for batch in houses_cached]
    queries = _as_dtype_batch(query_cached, torch.float64)
    query = _query_view(queries, 0)

    names = (
        "edge_normal_construction",
        "vertex_differences",
        "direction_vertex_dot",
        "holder_reduction_1",
        "shaping_1",
        "holder_reduction_2",
        "shaping_2",
        "backward_autograd",
        "output_cast",
    )
    for batch_number, house in enumerate(houses, start=1):
        for _ in range(warmup):
            _profile_grad_batch_once(query, house, gamma, eps)
        samples = defaultdict(list)
        pnv_shape = None
        for _ in range(repeats):
            elapsed, pnv_shape, _ = _profile_grad_batch_once(
                query, house, gamma, eps
            )
            for name in names:
                samples[name].append(elapsed[name])

        print(
            f"\nBatch {batch_number}: M={house.vertices.shape[0]}, "
            f"V={house.vertices.shape[1]}, E={house.edges.shape[1]}, "
            f"F={house.normals.shape[1]}, pnv={tuple(pnv_shape)}"
        )
        total = 0.0
        for name in names:
            mean = statistics.fmean(samples[name])
            total += mean
            print(f"  {name:26s} {_fmt_ms(samples[name])}")
        print(f"  {'sum_of_segments':26s} {total:9.3f} ms")


def _canonical_direction(vector: np.ndarray, decimals: int = 6) -> tuple[float, ...]:
    vector = np.asarray(vector, dtype=np.float64)
    nonzero = np.flatnonzero(np.abs(vector) > 10 ** (-decimals))
    if nonzero.size and vector[nonzero[0]] < 0:
        vector = -vector
    return tuple(np.round(vector, decimals=decimals))


def _signed_direction(vector: np.ndarray, decimals: int = 6) -> tuple[float, ...]:
    return tuple(np.round(np.asarray(vector, dtype=np.float64), decimals=decimals))


def _weighted_holder_min(values, weights, gamma: float, dim: int):
    p = gamma + 1.0
    shape = [1] * values.ndim
    shape[dim] = weights.numel()
    weights = weights.reshape(shape)
    minimum = values.min(dim=dim, keepdim=True).values
    positive_values = torch.where(values > 0, values, torch.ones_like(values))
    positive = (
        (weights * positive_values.pow(-p))
        .sum(dim=dim, keepdim=True)
        .pow(-1.0 / p)
    )
    negative_values = torch.clamp(-values, min=0.0)
    negative = -(
        (weights * negative_values.pow(p))
        .sum(dim=dim, keepdim=True)
        .pow(1.0 / p)
    )
    return torch.where(
        minimum > 0,
        positive,
        torch.where(minimum < 0, negative, torch.zeros_like(minimum)),
    ).squeeze(dim)


def _weighted_aggregate(values, weights, gamma: float, eps: float):
    first = holder_min(values, gamma, dim=3, eps=eps)
    shaped = shaping_function(first, gamma, eps=eps)
    second = -_weighted_holder_min(-shaped, weights, gamma, dim=2)
    return shaping_function(second, gamma, eps=eps)


def _direction_groups(directions: torch.Tensor):
    groups = defaultdict(list)
    for index, direction in enumerate(directions.detach().cpu().numpy()):
        groups[_signed_direction(direction)].append(index)
    return list(groups.values())


def _redundancy_equivalence_check(
    query_cpu,
    house_cpu,
    gamma: float,
    eps: float,
) -> tuple[int, int, float, float, float, float]:
    vertex_a = query_cpu.vertices[0:1].double()
    edges_a = query_cpu.edges[0:1].double()
    normals_a = query_cpu.normals[0:1].double()
    vertex_b = house_cpu.vertices[0:1].double()
    edges_b = house_cpu.edges[0:1].double()
    normals_b = house_cpu.normals[0:1].double()

    edges_ab = pairwise_concat_objects(edges_a, edges_b)
    edges_ab = torch.cat((edges_ab, -edges_ab), dim=2)
    faces_ab = pairwise_concat_objects(normals_a, normals_b)
    directions = torch.cat((edges_ab, faces_ab), dim=2)
    vertices = pairwise_difference(vertex_a, vertex_b)
    pnv = pairwise_direction_vertex_dot(directions, vertices)

    full_leaf = pnv.detach().requires_grad_(True)
    full_distance = benchmark_module.holder_distance(
        vertex_a,
        edges_a,
        normals_a,
        vertex_b,
        edges_b,
        normals_b,
        gamma,
        eps,
    )
    full_distance_from_leaf = _aggregate_holder_distance(full_leaf, gamma, eps)
    full_distance_from_leaf.sum().backward()

    groups = _direction_groups(directions[0, 0])
    representatives = torch.tensor([group[0] for group in groups], dtype=torch.long)
    weights = torch.tensor([len(group) for group in groups], dtype=torch.float64)
    unique_leaf = pnv[:, :, representatives].detach().requires_grad_(True)
    weighted_distance = _weighted_aggregate(unique_leaf, weights, gamma, eps)
    weighted_distance.sum().backward()

    grouped_full_gradient = torch.stack(
        [full_leaf.grad[:, :, group].sum(dim=2) for group in groups], dim=2
    )
    distance_error = (weighted_distance - full_distance).abs().max().item()
    leaf_distance_error = (
        full_distance_from_leaf - full_distance
    ).abs().max().item()
    gradient_error = (
        unique_leaf.grad - grouped_full_gradient
    ).abs().max().item()
    gradient_relative = (
        (unique_leaf.grad - grouped_full_gradient).abs()
        / grouped_full_gradient.abs().clamp_min(1e-12)
    ).max().item()
    return (
        directions.shape[2],
        len(groups),
        distance_error,
        leaf_distance_error,
        gradient_error,
        gradient_relative,
    )


def direction_redundancy_experiment(
    house_cpu,
    query_cpu,
    gamma: float,
    eps: float,
) -> None:
    print("\n=== 3. Direction redundancy experiment ===")
    query_edges = query_cpu.edges[0].numpy()
    query_unique = len({_canonical_direction(edge) for edge in query_edges})
    print(
        f"Query box: stored edges={len(query_edges)}, "
        f"unique unoriented edge directions={query_unique}"
    )
    print(
        "  batch   M   stored_E   unique_unoriented_E(min/mean/max)   "
        "unique_signed_all(min/mean/max)"
    )
    current_total = 0
    edge_dedup_total = 0
    fully_grouped_total = 0
    for batch_number, batch in enumerate(house_cpu, start=1):
        counts = [
            len({_canonical_direction(edge) for edge in edges.numpy()})
            for edges in batch.edges
        ]
        m = len(batch.indices)
        vertex_b = batch.vertices.shape[1]
        edge_b = batch.edges.shape[1]
        face_b = batch.normals.shape[1]
        current_d = 2 * len(query_edges) + 2 * edge_b + query_cpu.normals.shape[1] + face_b
        current_elements = m * current_d * query_cpu.vertices.shape[1] * vertex_b
        # This estimate retains face normals as separate entries and replaces
        # repeated physical edge directions by one +/- pair plus a weight.
        compressed_ds = [
            2 * query_unique
            + 2 * count
            + query_cpu.normals.shape[1]
            + face_b
            for count in counts
        ]
        edge_dedup_elements = sum(compressed_ds) * query_cpu.vertices.shape[1] * vertex_b

        signed_counts = []
        for local_index in range(m):
            ea = query_cpu.edges[0]
            eb = batch.edges[local_index]
            fa = query_cpu.normals[0]
            fb = batch.normals[local_index]
            directions = torch.cat((ea, eb, -ea, -eb, fa, fb), dim=0)
            signed_counts.append(len({_signed_direction(x.numpy()) for x in directions}))
        fully_grouped_elements = sum(signed_counts) * query_cpu.vertices.shape[1] * vertex_b

        current_total += current_elements
        edge_dedup_total += edge_dedup_elements
        fully_grouped_total += fully_grouped_elements
        print(
            f"  {batch_number:5d} {m:4d} {edge_b:10d} "
            f"{min(counts):3d}/{statistics.fmean(counts):5.1f}/{max(counts):3d} "
            f"{min(signed_counts):3d}/{statistics.fmean(signed_counts):5.1f}/"
            f"{max(signed_counts):3d}"
        )
        print(
            f"        pnv elements: current={current_elements}, "
            f"unique_edges={edge_dedup_elements}, "
            f"all_equal_signed={fully_grouped_elements}"
        )

        check = _redundancy_equivalence_check(query_cpu, batch, gamma, eps)
        print(
            f"        representative signed directions {check[0]} -> {check[1]}; "
            f"weighted distance max_abs={check[2]:.3e}; "
            f"projection reconstruction max_abs={check[3]:.3e}; "
            f"aggregated grad max_abs={check[4]:.3e}, max_rel={check[5]:.3e}"
        )

    print("\nProjected-size estimates:")
    print(f"  current:                    {current_total:12d} elements")
    print(
        f"  unique edges + weights:     {edge_dedup_total:12d} elements "
        f"({100.0 * edge_dedup_total / current_total:5.1f}% of current)"
    )
    print(
        f"  all equal signed directions:{fully_grouped_total:12d} elements "
        f"({100.0 * fully_grouped_total / current_total:5.1f}% of current)"
    )
    print(
        "Weighted equivalence: after the vertex reduction, m identical direction "
        "entries contribute m*f(x) inside the relevant Holder power sum. The "
        "weighted formula applies that factor explicitly. Its derivative with "
        "respect to a shared projected value equals the sum of the m original "
        "derivatives; the representative checks above compare those quantities."
    )


def pytorch_profiler_experiment(
    houses_cached,
    query_cached,
    object_count: int,
    gamma: float,
    eps: float,
) -> None:
    print("\n=== 4. PyTorch profiler: one production full cached query ===")
    query = benchmark_module._query_at(query_cached, 0)
    # Warm exactly the production path that will be profiled.
    pnv_results = benchmark_module._distance_and_pnv_gradient(
        query, houses_cached, gamma, eps
    )
    benchmark_module._vectorized_pose_gradients(
        query, pnv_results, object_count
    )
    del pnv_results
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        pnv_results = benchmark_module._distance_and_pnv_gradient(
            query, houses_cached, gamma, eps
        )
        benchmark_module._vectorized_pose_gradients(
            query, pnv_results, object_count
        )
        torch.cuda.synchronize()

    print(
        prof.key_averages(group_by_input_shape=False).table(
            sort_by="self_cuda_time_total", row_limit=35
        )
    )
    keys = prof.key_averages(group_by_input_shape=False)
    categories = {
        "pow": {"aten::pow"},
        "reductions": {"aten::sum", "aten::min", "aten::amax", "aten::amin"},
        "matmul_bmm": {"aten::matmul", "aten::bmm", "aten::mm"},
        "einsum": {"aten::einsum"},
        "casts_copies": {"aten::to", "aten::_to_copy", "aten::copy_"},
    }
    print("Selected operator totals (inclusive CUDA time can overlap child ops):")
    for category, operator_names in categories.items():
        matched = [event for event in keys if event.key in operator_names]
        cuda_us = sum(
            float(getattr(event, "self_device_time_total", 0.0) or 0.0)
            for event in matched
        )
        calls = sum(event.count for event in matched)
        memory = sum(
            int(
                getattr(
                    event,
                    "self_device_memory_usage",
                    getattr(event, "self_cuda_memory_usage", 0),
                )
                or 0
            )
            for event in matched
        )
        print(
            f"  {category:14s} calls={calls:5d} "
            f"self_cuda={cuda_us / 1000.0:9.3f} ms "
            f"net_self_device_mem={memory / 2**20:9.3f} MiB"
        )

    cuda_events = [
        event
        for event in prof.events()
        if str(getattr(event, "device_type", "")).endswith("CUDA")
    ]
    kernel_totals = defaultdict(lambda: [0, 0.0])
    for event in cuda_events:
        duration = float(getattr(event, "self_device_time_total", 0.0) or 0.0)
        if duration == 0.0:
            duration = float(getattr(event, "device_time_total", 0.0) or 0.0)
        kernel_totals[event.name][0] += 1
        kernel_totals[event.name][1] += duration
    largest = sorted(kernel_totals.items(), key=lambda item: item[1][1], reverse=True)
    print(f"CUDA kernel events: {len(cuda_events)}")
    print("Largest CUDA kernel families:")
    for name, (count, duration_us) in largest[:15]:
        print(f"  {duration_us / 1000.0:9.3f} ms  {count:4d}x  {name}")
    print(
        f"Peak allocated during profiled query: "
        f"{torch.cuda.max_memory_allocated() / 2**20:.3f} MiB"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--stage-repeats", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=benchmark_module.GAMMA)
    parser.add_argument("--epsilon", type=float, default=benchmark_module.EPSILON)
    parser.add_argument("--skip-profiler", action="store_true")
    parser.add_argument(
        "--experiment",
        choices=("all", "precision", "stages", "directions", "profiler"),
        default="all",
        help="Run one experiment in isolation or the complete suite.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for these experiments")
    if min(args.queries, args.repeats, args.stage_repeats) <= 0:
        raise ValueError("query and repeat counts must be positive")

    device = torch.device("cuda")
    print(f"PyTorch {torch.__version__}, CUDA runtime {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print("Building the shared cached geometry once...")
    house_objects, house_cpu, query_cpu, houses_cached, query_cached = _build_caches(
        max(args.queries, args.warmup), device
    )
    print(
        f"Cached {len(house_objects)} house objects in "
        f"{len(houses_cached)} homogeneous batches"
    )

    if args.experiment in ("all", "precision"):
        precision_experiment(
            houses_cached,
            query_cached,
            len(house_objects),
            args.queries,
            args.warmup,
            args.repeats,
            args.gamma,
            args.epsilon,
        )
    if args.experiment in ("all", "stages"):
        per_stage_experiment(
            houses_cached,
            query_cached,
            args.warmup,
            args.stage_repeats,
            args.gamma,
            args.epsilon,
        )
    if args.experiment in ("all", "directions"):
        direction_redundancy_experiment(
            house_cpu, query_cpu, args.gamma, args.epsilon
        )
    if args.experiment in ("all", "profiler") and not args.skip_profiler:
        pytorch_profiler_experiment(
            houses_cached,
            query_cached,
            len(house_objects),
            args.gamma,
            args.epsilon,
        )


if __name__ == "__main__":
    main()
