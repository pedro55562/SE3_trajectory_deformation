"""Benchmark GPU distance/gradient queries against the L-shaped house.

The script profiles one-time geometry setup separately, then benchmarks three
cached-query modes. House construction, primitive conversion, random sampling,
pose construction, and CUDA warm-up remain outside the benchmark statistics.
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np


# Use the checked-out UAIbotPy submodule, including its feat/gpu-holder code.
PROJECT_DIR = Path(__file__).resolve().parent
UAIBOT_DIR = PROJECT_DIR / "UAIbotPy"
if str(UAIBOT_DIR) not in sys.path:
    sys.path.insert(0, str(UAIBOT_DIR))

import torch
import uaibot as ub
from uaibot.gpu.distance import (
    holder_distance,
    holder_distance_with_grad,
    pnv_grad_SE3,
    se3_generators,
    se3_generators_cached,
)
from uaibot.gpu.geometry import extract_VEF

# l_house_uaibot exposes the existing house construction in two stages.  The
# alias makes the construction entry point explicit without copying any of it.
from l_house_uaibot import build_plan_data
from l_house_uaibot import build_uaibot_objects as create_house


N = 1000
RANDOM_SEED = 0
GAMMA = 2.0
EPSILON = 1e-3

# Centers of the added obstacle are sampled uniformly from this box (meters).
SAMPLE_BOX_MIN = np.array([0.0, 0.0, 0.25], dtype=float)
SAMPLE_BOX_MAX = np.array([16.0, 20.0, 2.50], dtype=float)

QUERY_OBSTACLE_SIZE = (0.30, 0.30, 0.30)
CYLINDER_SIDES = 12


def _sphere_as_polytope(ball: ub.Ball) -> ub.ConvexPolytope:
    """Return a circumscribed dodecahedral approximation of ``ball``."""
    golden_ratio = (1.0 + math.sqrt(5.0)) / 2.0
    normals = []
    for first_sign in (-1.0, 1.0):
        for second_sign in (-1.0, 1.0):
            normals.extend(
                (
                    (0.0, first_sign, second_sign * golden_ratio),
                    (first_sign, second_sign * golden_ratio, 0.0),
                    (second_sign * golden_ratio, 0.0, first_sign),
                )
            )

    A = np.asarray(normals, dtype=float)
    A /= np.linalg.norm(A, axis=1, keepdims=True)
    b = np.full(A.shape[0], ball.radius, dtype=float)
    return ub.ConvexPolytope(
        htm=ball.htm,
        name=f"{ball.name}_polytope",
        A=A,
        b=b,
        color=ball.color,
    )


def _cylinder_as_polytope(
    cylinder: ub.Cylinder, sides: int = CYLINDER_SIDES
) -> ub.ConvexPolytope:
    """Return a regular-prism approximation of ``cylinder``."""
    angles = np.linspace(0.0, 2.0 * math.pi, sides, endpoint=False)
    radial_normals = np.column_stack(
        (np.cos(angles), np.sin(angles), np.zeros(sides))
    )
    A = np.vstack(
        (
            radial_normals,
            np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]),
        )
    )
    b = np.concatenate(
        (
            np.full(sides, cylinder.radius),
            np.full(2, cylinder.height / 2.0),
        )
    )
    return ub.ConvexPolytope(
        htm=cylinder.htm,
        name=f"{cylinder.name}_polytope",
        A=A,
        b=b,
        color=cylinder.color,
    )


def _as_gpu_compatible_polyhedron(obj):
    """Convert a UAIbot primitive to a GPU-distance-compatible object."""
    if isinstance(obj, (ub.Box, ub.ConvexPolytope)):
        return obj
    if isinstance(obj, ub.Ball):
        return _sphere_as_polytope(obj)
    if isinstance(obj, ub.Cylinder):
        return _cylinder_as_polytope(obj)
    raise TypeError(
        f"House object {getattr(obj, 'name', '<unnamed>')!r} has unsupported "
        f"type {type(obj).__name__}; expected Box, ConvexPolytope, Ball, or Cylinder."
    )


def _make_query_poses(iterations: int, seed: int) -> list[np.matrix]:
    """Sample query positions and construct their poses outside timed code."""
    rng = np.random.default_rng(seed)
    positions = rng.uniform(SAMPLE_BOX_MIN, SAMPLE_BOX_MAX, size=(iterations, 3))
    return [ub.Utils.trn(position) for position in positions]


@dataclass
class GeometryBatch:
    """A topology-homogeneous V/E/F batch and its original object indices."""

    indices: list[int]
    vertices: torch.Tensor
    edges: torch.Tensor
    normals: torch.Tensor
    device_indices: torch.Tensor | None = None
    vertices_f64: torch.Tensor | None = None
    edges_f64: torch.Tensor | None = None
    normals_f64: torch.Tensor | None = None


@dataclass
class PNVResult:
    """Distance and pnv-gradient output for one homogeneous obstacle batch."""

    house: GeometryBatch
    distances: torch.Tensor
    grad_pnv: torch.Tensor


@dataclass
class StaticPoseBatch:
    """Query-independent tensors used by the optimized SE(3) chain rule."""

    geometry: GeometryBatch
    directions_h: torch.Tensor
    vertices_h: torch.Tensor
    prod_b_s_b: torch.Tensor


def _build_static_pose_cache(
    house_batches: list[GeometryBatch], generators: torch.Tensor
) -> list[StaticPoseBatch]:
    caches = []
    for house in house_batches:
        directions = torch.cat((house.edges, house.normals), dim=1)
        directions_h = torch.cat(
            (directions, torch.zeros_like(directions[..., :1])), dim=-1
        )
        vertices_h = torch.cat(
            (house.vertices, torch.ones_like(house.vertices[..., :1])), dim=-1
        )
        prod_b_s_b = torch.einsum(
            "mdi,gij,mvj->mdvg", directions_h, generators, vertices_h
        )
        caches.append(
            StaticPoseBatch(house, directions_h, vertices_h, prod_b_s_b)
        )
    return caches


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed(device: torch.device, operation):
    """Time an operation, including all asynchronous CUDA work it launches."""
    _synchronize(device)
    start = perf_counter()
    result = operation()
    _synchronize(device)
    return perf_counter() - start, result


def _group_and_stack(components) -> list[GeometryBatch]:
    grouped_indices = defaultdict(list)
    for index, (vertices, edges, normals) in enumerate(components):
        signature = (vertices.shape[0], edges.shape[0], normals.shape[0])
        grouped_indices[signature].append(index)

    return [
        GeometryBatch(
            indices=indices,
            vertices=torch.stack([components[i][0] for i in indices]),
            edges=torch.stack([components[i][1] for i in indices]),
            normals=torch.stack([components[i][2] for i in indices]),
        )
        for indices in grouped_indices.values()
    ]


def _batch_to_device(batch: GeometryBatch, device: torch.device) -> GeometryBatch:
    vertices = batch.vertices.to(device)
    edges = batch.edges.to(device)
    normals = batch.normals.to(device)
    return GeometryBatch(
        indices=batch.indices,
        vertices=vertices,
        edges=edges,
        normals=normals,
        device_indices=torch.tensor(batch.indices, dtype=torch.long, device=device),
        # holder_distance_with_grad converts all inputs to float64 internally.
        # Cache that representation too, avoiding six device-side conversions
        # of the static tensors on every query.
        vertices_f64=vertices.to(dtype=torch.float64),
        edges_f64=edges.to(dtype=torch.float64),
        normals_f64=normals.to(dtype=torch.float64),
    )


def _query_at(query_batch: GeometryBatch, index: int) -> GeometryBatch:
    """Return a zero-copy, one-object view into the preloaded query tensors."""
    return GeometryBatch(
        indices=[index],
        vertices=query_batch.vertices[index : index + 1],
        edges=query_batch.edges[index : index + 1],
        normals=query_batch.normals[index : index + 1],
        vertices_f64=query_batch.vertices_f64[index : index + 1],
        edges_f64=query_batch.edges_f64[index : index + 1],
        normals_f64=query_batch.normals_f64[index : index + 1],
    )


def _distance_only(
    query: GeometryBatch,
    house_batches: list[GeometryBatch],
    gamma: float,
    epsilon: float,
) -> list[torch.Tensor]:
    return [
        holder_distance(
            query.vertices_f64,
            query.edges_f64,
            query.normals_f64,
            house.vertices_f64,
            house.edges_f64,
            house.normals_f64,
            gamma,
            epsilon,
        )
        for house in house_batches
    ]


def _distance_and_pnv_gradient(
    query: GeometryBatch,
    house_batches: list[GeometryBatch],
    gamma: float,
    epsilon: float,
) -> list[PNVResult]:
    results = []
    for house in house_batches:
        distances, grad_pnv = holder_distance_with_grad(
            query.vertices_f64,
            query.edges_f64,
            query.normals_f64,
            house.vertices_f64,
            house.edges_f64,
            house.normals_f64,
            gamma,
            epsilon,
        )
        results.append(PNVResult(house, distances, grad_pnv))
    return results


def _split_direction_weights(
    grad_pnv: torch.Tensor,
    edge_count_a: int,
    edge_count_b: int,
    face_count_a: int,
    face_count_b: int,
    vertex_count_a: int,
    vertex_count_b: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Combine +/- edge weights and append face-normal weights."""
    grad = grad_pnv[0].reshape(
        grad_pnv.shape[1], -1, vertex_count_a, vertex_count_b
    )
    edge_a_pos = 0
    edge_b_pos = edge_count_a
    edge_a_neg = edge_count_a + edge_count_b
    edge_b_neg = 2 * edge_count_a + edge_count_b
    face_a = 2 * edge_count_a + 2 * edge_count_b
    face_b = face_a + face_count_a

    weights_a = torch.cat(
        (
            grad[:, edge_a_pos:edge_b_pos]
            - grad[:, edge_a_neg:edge_b_neg],
            grad[:, face_a:face_b],
        ),
        dim=1,
    )
    weights_b = torch.cat(
        (
            grad[:, edge_b_pos:edge_a_neg]
            - grad[:, edge_b_neg:face_a],
            grad[:, face_b : face_b + face_count_b],
        ),
        dim=1,
    )
    return weights_a, weights_b


def _vectorized_group_pose_gradient(
    query: GeometryBatch,
    result: PNVResult,
    generators: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the existing SE(3) chain rule across a full obstacle batch."""
    house = result.house
    vertices_a = query.vertices[0]
    edges_a = query.edges[0]
    normals_a = query.normals[0]
    vertices_b = house.vertices
    edges_b = house.edges
    normals_b = house.normals

    vertex_count_a = vertices_a.shape[0]
    vertex_count_b = vertices_b.shape[1]
    edge_count_a = edges_a.shape[0]
    edge_count_b = edges_b.shape[1]
    face_count_a = normals_a.shape[0]
    face_count_b = normals_b.shape[1]

    weights_a, weights_b = _split_direction_weights(
        result.grad_pnv,
        edge_count_a,
        edge_count_b,
        face_count_a,
        face_count_b,
        vertex_count_a,
        vertex_count_b,
    )
    directions_a = torch.cat((edges_a, normals_a), dim=0)
    directions_b = torch.cat((edges_b, normals_b), dim=1)

    directions_a_h = torch.cat(
        (directions_a, torch.zeros_like(directions_a[:, :1])), dim=-1
    )
    directions_b_h = torch.cat(
        (directions_b, torch.zeros_like(directions_b[..., :1])), dim=-1
    )
    vertices_a_h = torch.cat(
        (vertices_a, torch.ones_like(vertices_a[:, :1])), dim=-1
    )
    vertices_b_h = torch.cat(
        (vertices_b, torch.ones_like(vertices_b[..., :1])), dim=-1
    )

    generators_t = generators.transpose(-1, -2)
    prod_a_s_a = torch.einsum(
        "di,gij,vj->dvg", directions_a_h, generators, vertices_a_h
    )
    prod_a_st_b = torch.einsum(
        "di,gji,mvj->mdvg", directions_a_h, generators_t, vertices_b_h
    )
    prod_b_s_a = torch.einsum(
        "mdi,gij,vj->mdvg", directions_b_h, generators, vertices_a_h
    )
    prod_b_st_a = torch.einsum(
        "mdi,gji,vj->mdvg", directions_b_h, generators_t, vertices_a_h
    )
    prod_b_s_b = torch.einsum(
        "mdi,gij,mvj->mdvg", directions_b_h, generators, vertices_b_h
    )

    a_self = torch.einsum("mdab,dag->mg", weights_a, prod_a_s_a)
    a_cross = torch.einsum("mdab,mdbg->mg", weights_a, prod_a_st_b)
    b_from_a = torch.einsum("mdab,mdag->mg", weights_b, prod_b_s_a)
    b_from_a_t = torch.einsum("mdab,mdag->mg", weights_b, prod_b_st_a)
    b_self = torch.einsum("mdab,mdbg->mg", weights_b, prod_b_s_b)

    pair_grad_a = 2.0 * a_self - a_cross + b_from_a
    pair_grad_b = -a_cross + b_from_a_t - 2.0 * b_self
    return pair_grad_a, pair_grad_b


def _vectorized_group_pose_gradient_cached(
    query: GeometryBatch,
    result: PNVResult,
    static: StaticPoseBatch,
    generators: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Equivalent SE(3) contraction reusing query-independent house tensors."""
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

    weights_a, weights_b = _split_direction_weights(
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


_compiled_pose_group = None


def _get_compiled_pose_group():
    """Lazily compile the fixed-shape cached SE(3) group contraction."""
    global _compiled_pose_group
    if _compiled_pose_group is None:
        _compiled_pose_group = torch.compile(
            _vectorized_group_pose_gradient_cached,
            fullgraph=True,
            mode="reduce-overhead",
        )
    return _compiled_pose_group


def _vectorized_pose_gradients(
    query: GeometryBatch,
    pnv_results: list[PNVResult],
    house_object_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = query.vertices.device
    generators = se3_generators().to(device=device, dtype=query.vertices.dtype)
    grad_a = torch.zeros(1, 6, dtype=query.vertices.dtype, device=device)
    grad_b = torch.zeros(
        house_object_count, 6, dtype=query.vertices.dtype, device=device
    )

    for result in pnv_results:
        pair_grad_a, pair_grad_b = _vectorized_group_pose_gradient(
            query, result, generators
        )
        grad_a[0] += pair_grad_a.sum(dim=0)
        grad_b.index_copy_(0, result.house.device_indices, pair_grad_b)
    return grad_a, grad_b


def _vectorized_pose_gradients_cached(
    query: GeometryBatch,
    pnv_results: list[PNVResult],
    static_batches: list[StaticPoseBatch],
    house_object_count: int,
    generators: torch.Tensor | None = None,
    compile_groups: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Optimized pose-gradient path with an explicit static-house cache."""
    device = query.vertices.device
    if generators is None:
        generators = se3_generators_cached(device, query.vertices.dtype)
    grad_a = torch.zeros(1, 6, dtype=query.vertices.dtype, device=device)
    grad_b = torch.zeros(
        house_object_count, 6, dtype=query.vertices.dtype, device=device
    )

    group_operation = (
        _get_compiled_pose_group()
        if compile_groups
        else _vectorized_group_pose_gradient_cached
    )
    for result, static in zip(pnv_results, static_batches):
        pair_grad_a, pair_grad_b = group_operation(
            query, result, static, generators
        )
        grad_a[0] += pair_grad_a.sum(dim=0)
        grad_b.index_copy_(0, static.geometry.device_indices, pair_grad_b)
    return grad_a, grad_b


def _legacy_python_pose_gradients(
    query: GeometryBatch,
    pnv_results: list[PNVResult],
    house_object_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The original per-pair SE(3) loop, retained only for profiling/checking."""
    device = query.vertices.device
    grad_a = torch.zeros(1, 6, dtype=torch.float32, device=device)
    grad_b = torch.zeros(house_object_count, 6, dtype=torch.float32, device=device)

    vertices_a = query.vertices[0]
    edges_a = query.edges[0]
    normals_a = query.normals[0]
    dirs_a = torch.cat((edges_a, normals_a), dim=0)
    vertex_count_a = vertices_a.shape[0]
    edge_count_a = edges_a.shape[0]
    face_count_a = normals_a.shape[0]

    for result in pnv_results:
        house = result.house
        edge_count_b = house.edges.shape[1]
        face_count_b = house.normals.shape[1]
        vertex_count_b = house.vertices.shape[1]
        direction_count = result.grad_pnv.shape[2]

        for local_index, global_index in enumerate(house.indices):
            vertices_b = house.vertices[local_index]
            edges_b = house.edges[local_index]
            normals_b = house.normals[local_index]
            dirs_b = torch.cat((edges_b, normals_b), dim=0)
            pnv_grads = pnv_grad_SE3(
                normals_A=dirs_a,
                vertices_A=vertices_a,
                normals_B=dirs_b,
                vertices_B=vertices_b,
            )

            pair_grad_a = torch.zeros(
                direction_count,
                vertex_count_a,
                vertex_count_b,
                6,
                device=device,
            )
            pair_grad_b = torch.zeros_like(pair_grad_a)
            a_edge_neg = edge_count_a + edge_count_b
            b_edge_neg = 2 * edge_count_a + edge_count_b
            a_face = 2 * edge_count_a + 2 * edge_count_b
            b_face = a_face + face_count_a

            pair_grad_a[:edge_count_a] = pnv_grads["nA_grad_HA"][:edge_count_a]
            pair_grad_b[:edge_count_a] = pnv_grads["nA_grad_HB"][:edge_count_a]
            pair_grad_a[edge_count_a:a_edge_neg] = pnv_grads["nB_grad_HA"][
                :edge_count_b
            ]
            pair_grad_b[edge_count_a:a_edge_neg] = pnv_grads["nB_grad_HB"][
                :edge_count_b
            ]
            pair_grad_a[a_edge_neg:b_edge_neg] = -pnv_grads["nA_grad_HA"][
                :edge_count_a
            ]
            pair_grad_b[a_edge_neg:b_edge_neg] = -pnv_grads["nA_grad_HB"][
                :edge_count_a
            ]
            pair_grad_a[b_edge_neg:a_face] = -pnv_grads["nB_grad_HA"][
                :edge_count_b
            ]
            pair_grad_b[b_edge_neg:a_face] = -pnv_grads["nB_grad_HB"][
                :edge_count_b
            ]
            pair_grad_a[a_face:b_face] = pnv_grads["nA_grad_HA"][edge_count_a:]
            pair_grad_b[a_face:b_face] = pnv_grads["nA_grad_HB"][edge_count_a:]
            pair_grad_a[b_face:] = pnv_grads["nB_grad_HA"][edge_count_b:]
            pair_grad_b[b_face:] = pnv_grads["nB_grad_HB"][edge_count_b:]

            grad_pnv = result.grad_pnv[0, local_index].reshape(
                direction_count, vertex_count_a, vertex_count_b
            )
            grad_a[0] += torch.einsum("dab,dabk->k", grad_pnv, pair_grad_a)
            grad_b[global_index] += torch.einsum(
                "dab,dabk->k", grad_pnv, pair_grad_b
            )
    return grad_a, grad_b


def benchmark(
    iterations: int = N,
    seed: int = RANDOM_SEED,
    gamma: float = GAMMA,
    epsilon: float = EPSILON,
    device: str = "cuda",
) -> tuple[int, float, float]:
    """Profile setup and benchmark all three distance/gradient modes."""
    if iterations <= 0:
        raise ValueError("iterations must be greater than zero")

    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable to PyTorch. Check `nvidia-smi` and the NVIDIA "
            "driver; use `--device cpu` only for a non-GPU smoke test."
        )

    # Scene construction and primitive conversion remain outside the requested
    # V/E/F profile and all per-query measurements.
    print(f"Using device: {torch_device}", flush=True)
    print("Creating house (not timed)...", flush=True)
    house_obstacles = create_house(build_plan_data())
    print(
        f"Converting {len(house_obstacles)} house obstacles (not timed)...",
        flush=True,
    )
    house_polyhedra = [_as_gpu_compatible_polyhedron(obj) for obj in house_obstacles]

    query_obstacle = ub.Box(
        htm=np.identity(4),
        name="benchmark_query_obstacle",
        width=QUERY_OBSTACLE_SIZE[0],
        depth=QUERY_OBSTACLE_SIZE[1],
        height=QUERY_OBSTACLE_SIZE[2],
        color="red",
    )
    query_poses = _make_query_poses(iterations, seed)

    # Extract once. No call to extract_VEF occurs below this point.
    extract_start = perf_counter()
    house_components = [extract_VEF(obj) for obj in house_polyhedra]
    house_extract_time = perf_counter() - extract_start
    query_extract_start = perf_counter()
    query_components = [extract_VEF(query_obstacle, pose) for pose in query_poses]
    query_extract_time = perf_counter() - query_extract_start
    extract_time = house_extract_time + query_extract_time

    stack_start = perf_counter()
    house_cpu_batches = _group_and_stack(house_components)
    query_cpu_batches = _group_and_stack(query_components)
    if len(query_cpu_batches) != 1:
        raise RuntimeError("All query boxes should have one V/E/F signature")
    query_cpu_batch = query_cpu_batches[0]
    stack_time = perf_counter() - stack_start

    transfer_time, transferred = _timed(
        torch_device,
        lambda: (
            [_batch_to_device(batch, torch_device) for batch in house_cpu_batches],
            _batch_to_device(query_cpu_batch, torch_device),
        ),
    )
    house_batches, query_batch = transferred

    print(
        "Static house V/E/F cache: "
        f"{len(house_batches)} homogeneous batch(es), extracted once, "
        "transferred once in float32/float64; "
        "per-iteration extraction/transfers: 0",
        flush=True,
    )
    for batch in house_batches:
        print(
            "  batch: "
            f"objects={len(batch.indices)}, V={batch.vertices.shape[1]}, "
            f"E={batch.edges.shape[1]}, F={batch.normals.shape[1]}",
            flush=True,
        )

    # Warm up every measured path, including the new batched SE(3) chain rule.
    print("Warming up distance/gradient kernels (not timed)...", flush=True)
    warm_query = _query_at(query_batch, 0)
    _distance_only(warm_query, house_batches, gamma, epsilon)
    warm_pnv = _distance_and_pnv_gradient(
        warm_query, house_batches, gamma, epsilon
    )
    _vectorized_pose_gradients(warm_query, warm_pnv, len(house_polyhedra))
    _synchronize(torch_device)

    # Decompose one cached query. The legacy loop is measured once, solely to
    # identify its cost and validate the vectorized replacement numerically.
    forward_time, _ = _timed(
        torch_device,
        lambda: _distance_only(warm_query, house_batches, gamma, epsilon),
    )
    pnv_time, profile_pnv = _timed(
        torch_device,
        lambda: _distance_and_pnv_gradient(
            warm_query, house_batches, gamma, epsilon
        ),
    )
    legacy_pose_time, legacy_gradients = _timed(
        torch_device,
        lambda: _legacy_python_pose_gradients(
            warm_query, profile_pnv, len(house_polyhedra)
        ),
    )
    vector_pose_time, vector_gradients = _timed(
        torch_device,
        lambda: _vectorized_pose_gradients(
            warm_query, profile_pnv, len(house_polyhedra)
        ),
    )
    torch.testing.assert_close(
        vector_gradients[0], legacy_gradients[0], rtol=2e-4, atol=2e-4
    )
    torch.testing.assert_close(
        vector_gradients[1], legacy_gradients[1], rtol=2e-4, atol=2e-4
    )

    total_profile_time = pnv_time + vector_pose_time
    autograd_overhead = pnv_time - forward_time
    first_call_with_setup = extract_time + stack_time + transfer_time + total_profile_time
    print("\nOne-call timing decomposition:")
    print(
        f"Original SE(3) path: {len(house_polyhedra)} Python pair iterations; "
        f"vectorized path: 0 pair iterations across {len(house_batches)} batches"
    )
    print(f"1. extract_VEF / geometry preparation: {extract_time:.6f} s (one-time)")
    print(f"   static house extract_VEF:             {house_extract_time:.6f} s")
    print(f"   all pre-sampled query extract_VEF:    {query_extract_time:.6f} s")
    print(f"2. grouping and torch.stack:            {stack_time:.6f} s (one-time)")
    transfer_name = (
        "CPU -> GPU transfer" if torch_device.type == "cuda" else "CPU tensor placement"
    )
    print(f"3. {transfer_name + ':':<36}{transfer_time:.6f} s (one-time)")
    print(f"4. Holder distance forward only:        {forward_time:.6f} s")
    print(f"5. holder_distance_with_grad total:      {pnv_time:.6f} s")
    print(f"   autograd/pnv overhead vs. forward:    {autograd_overhead:.6f} s")
    print(f"6a. SE(3) pose gradient, Python loop:    {legacy_pose_time:.6f} s")
    print(f"6b. SE(3) pose gradient, vectorized:     {vector_pose_time:.6f} s")
    print(f"7. total cached full-gradient query:     {total_profile_time:.6f} s")
    print(f"   first query including V/E/F setup:    {first_call_with_setup:.6f} s")
    print(
        "   vectorized/legacy SE(3) speedup:     "
        f"{legacy_pose_time / vector_pose_time:.2f}x"
    )

    print(f"\nRunning {iterations} timed iteration(s) in all modes...", flush=True)
    distance_times = []
    pnv_times = []
    pose_times = []
    for index in range(iterations):
        query = _query_at(query_batch, index)
        elapsed, distances = _timed(
            torch_device,
            lambda: _distance_only(query, house_batches, gamma, epsilon),
        )
        distance_times.append(elapsed)

        elapsed, pnv_results = _timed(
            torch_device,
            lambda: _distance_and_pnv_gradient(
                query, house_batches, gamma, epsilon
            ),
        )
        pnv_times.append(elapsed)

        elapsed, pose_gradients = _timed(
            torch_device,
            lambda: _vectorized_pose_gradients(
                query, pnv_results, len(house_polyhedra)
            ),
        )
        pose_times.append(elapsed)
        _ = distances, pose_gradients

    full_times = [
        pnv_elapsed + pose_elapsed
        for pnv_elapsed, pose_elapsed in zip(pnv_times, pose_times)
    ]
    evaluation_count = iterations * len(house_polyhedra)
    print(f"\nDistance/gradient evaluations: {evaluation_count}")
    print(
        "Distance only:              "
        f"{statistics.fmean(distance_times):.6f} s avg, "
        f"{statistics.pstdev(distance_times):.6f} s std"
    )
    print(
        "Distance + pnv gradient:    "
        f"{statistics.fmean(pnv_times):.6f} s avg, "
        f"{statistics.pstdev(pnv_times):.6f} s std"
    )
    print(
        "SE(3) gradient (vectorized): "
        f"{statistics.fmean(pose_times):.6f} s avg, "
        f"{statistics.pstdev(pose_times):.6f} s std"
    )
    average = statistics.fmean(full_times)
    standard_deviation = statistics.pstdev(full_times)
    print(
        "Distance + full SE(3):      "
        f"{average:.6f} s avg, {standard_deviation:.6f} s std"
    )
    return evaluation_count, average, standard_deviation


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-n", "--iterations", type=int, default=N)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--gamma", type=float, default=GAMMA)
    parser.add_argument("--epsilon", type=float, default=EPSILON)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    benchmark(
        iterations=args.iterations,
        seed=args.seed,
        gamma=args.gamma,
        epsilon=args.epsilon,
        device=args.device,
    )


if __name__ == "__main__":
    main()
