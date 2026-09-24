"""House geometry batching and corrected SE(3) pose-gradient helpers."""

from __future__ import annotations

import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# Use the current local UAIbotPy GPU implementation.
PROJECT_DIR = Path(__file__).resolve().parents[2]
UAIBOT_DIR = PROJECT_DIR / "UAIbotPy"
if str(UAIBOT_DIR) not in sys.path:
    sys.path.insert(0, str(UAIBOT_DIR))

import torch
import uaibot as ub
from uaibot.gpu.distance import (
    holder_distance,
    holder_distance_with_grad,
    se3_generators,
    se3_generators_cached,
)
from uaibot.gpu.geometry import extract_VEF

# l_house_uaibot exposes the existing house construction in two stages.  The
# alias makes the construction entry point explicit without copying any of it.
from house.l_house_uaibot import build_plan_data
from house.l_house_uaibot import build_uaibot_objects as create_house


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

    prod_a_s_b = torch.einsum(
        "di,gij,mvj->mdvg", directions_a_h, generators, vertices_b_h
    )
    prod_b_s_a = torch.einsum(
        "mdi,gij,vj->mdvg", directions_b_h, generators, vertices_a_h
    )

    a_from_b = torch.einsum("mdab,mdbg->mg", weights_a, prod_a_s_b)
    b_from_a = torch.einsum("mdab,mdag->mg", weights_b, prod_b_s_a)

    pair_grad_a = a_from_b + b_from_a
    pair_grad_b = -pair_grad_a
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

    prod_a_s_b = torch.einsum(
        "di,gij,mvj->mdvg", directions_a_h, generators, static.vertices_h
    )
    prod_b_s_a = torch.einsum(
        "mdi,gij,vj->mdvg", static.directions_h, generators, vertices_a_h
    )

    a_from_b = torch.einsum("mdab,mdbg->mg", weights_a, prod_a_s_b)
    b_from_a = torch.einsum("mdab,mdag->mg", weights_b, prod_b_s_a)

    pair_grad_a = a_from_b + b_from_a
    pair_grad_b = -pair_grad_a
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
