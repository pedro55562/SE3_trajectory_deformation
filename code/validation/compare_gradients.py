"""Compare PyTorch, C++, and finite-difference SE(3) pose gradients."""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[2]
UAIBOT_DIR = PROJECT_DIR / "UAIbotPy"
for path in (PROJECT_DIR, UAIBOT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import torch
import uaibot as ub
import uaibot_cpp_bind as ub_cpp
from uaibot.gpu.distance import (
    holder_distance,
    holder_distance_objects_with_grad,
)
from uaibot.gpu.geometry import extract_VEF


GAMMA = 2.0
EPSILON = 1e-3
EPS_EDGE = 1e-6
COMPONENTS = ("vx", "vy", "vz", "wx", "wy", "wz")
SIZE_A = (0.80, 0.60, 0.50)
SIZE_B = (0.70, 0.50, 0.90)


@dataclass(frozen=True)
class PoseCase:
    name: str
    htm_a: np.ndarray
    htm_b: np.ndarray


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def _rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    axis_hat = _skew(axis)
    return (
        np.eye(3)
        + math.sin(angle) * axis_hat
        + (1.0 - math.cos(angle)) * (axis_hat @ axis_hat)
    )


def _htm(
    translation: tuple[float, float, float],
    axis: tuple[float, float, float] = (1.0, 0.0, 0.0),
    angle: float = 0.0,
) -> np.ndarray:
    result = np.eye(4)
    result[:3, :3] = _rotation(np.asarray(axis), angle)
    result[:3, 3] = translation
    return result


def _left_increment(component: int, amount: float) -> np.ndarray:
    result = np.eye(4)
    if component < 3:
        result[component, 3] = amount
    else:
        axis = np.zeros(3)
        axis[component - 3] = 1.0
        result[:3, :3] = _rotation(axis, amount)
    return result


def _make_box(htm: np.ndarray, size: tuple[float, float, float], name: str):
    return ub.Box(
        htm=htm,
        name=name,
        width=size[0],
        depth=size[1],
        height=size[2],
        color="red",
    )


def _torch_distance(
    box_a, box_b, htm_a: np.ndarray, htm_b: np.ndarray, device: torch.device
) -> float:
    vertices_a, edges_a, normals_a = extract_VEF(box_a, htm_a)
    vertices_b, edges_b, normals_b = extract_VEF(box_b, htm_b)
    with torch.no_grad():
        distance = holder_distance(
            vertices_a[None].to(device=device, dtype=torch.float64),
            edges_a[None].to(device=device, dtype=torch.float64),
            normals_a[None].to(device=device, dtype=torch.float64),
            vertices_b[None].to(device=device, dtype=torch.float64),
            edges_b[None].to(device=device, dtype=torch.float64),
            normals_b[None].to(device=device, dtype=torch.float64),
            GAMMA,
            EPSILON,
        )
    return float(distance.item())


def _torch_analytical(box_a, box_b, device: torch.device):
    distance, gradients = holder_distance_objects_with_grad(
        [box_a],
        [box_b],
        gamma=GAMMA,
        eps=EPSILON,
        device=device,
    )
    return (
        float(distance[0, 0].item()),
        gradients["grad_HA"][0].detach().cpu().double().numpy(),
        gradients["grad_HB"][0].detach().cpu().double().numpy(),
    )


def _pose_gradient(
    vertices: np.ndarray,
    vertex_gradient: np.ndarray,
    directions: np.ndarray,
    direction_gradient: np.ndarray,
) -> np.ndarray:
    result = np.zeros(6)
    result[:3] = vertex_gradient.sum(axis=0)
    result[3:] = np.cross(vertices, vertex_gradient).sum(axis=0)
    result[3:] += np.cross(directions, direction_gradient).sum(axis=0)
    return result


def _cpp_analytical(box_a, box_b):
    result = box_a.signed_distance(
        box_b,
        gamma=GAMMA,
        is_conservative=False,
        skip_gradient=False,
        epsilon=EPSILON,
        eps_edge=EPS_EDGE,
        mode="c++",
    )
    distance = float(result[0])
    grad_vertices_a = np.asarray(result[2], dtype=float)
    grad_vertices_b = np.asarray(result[3], dtype=float)
    grad_directions = np.asarray(result[4], dtype=float)

    cpp_a = box_a.cpp_obj
    cpp_b = box_b.cpp_obj
    vertices_a = np.asarray(ub_cpp.get_box_vertices(cpp_a), dtype=float)
    vertices_b = np.asarray(ub_cpp.get_box_vertices(cpp_b), dtype=float)
    face_a, face_b, edge_directions = ub_cpp.get_candidate_normals(
        cpp_a, cpp_b, False, EPS_EDGE
    )
    face_a = np.asarray(face_a, dtype=float)
    face_b = np.asarray(face_b, dtype=float)
    edge_directions = np.asarray(edge_directions, dtype=float)

    edge_count_a = extract_VEF(box_a)[1].shape[0]
    edge_count_b = extract_VEF(box_b)[1].shape[0]
    edge_a_rows = np.r_[
        0:edge_count_a,
        edge_count_a + edge_count_b : 2 * edge_count_a + edge_count_b,
    ]
    edge_b_rows = np.r_[
        edge_count_a : edge_count_a + edge_count_b,
        2 * edge_count_a + edge_count_b : 2 * (edge_count_a + edge_count_b),
    ]

    face_a_rows = np.arange(face_a.shape[0])
    edge_start = face_a.shape[0]
    edge_a_rows = edge_start + edge_a_rows
    edge_b_rows = edge_start + edge_b_rows
    face_b_rows = edge_start + edge_directions.shape[0] + np.arange(face_b.shape[0])
    directions = np.vstack((face_a, edge_directions, face_b))

    rows_a = np.concatenate((face_a_rows, edge_a_rows))
    rows_b = np.concatenate((edge_b_rows, face_b_rows))
    grad_a = _pose_gradient(
        vertices_a,
        grad_vertices_a,
        directions[rows_a],
        grad_directions[rows_a],
    )
    grad_b = _pose_gradient(
        vertices_b,
        grad_vertices_b,
        directions[rows_b],
        grad_directions[rows_b],
    )
    return distance, grad_a, grad_b


def _finite_difference(
    box_a,
    box_b,
    htm_a: np.ndarray,
    htm_b: np.ndarray,
    device: torch.device,
    step: float,
):
    gradients = []
    for target in ("A", "B"):
        gradient = np.empty(6)
        for component in range(6):
            plus = _left_increment(component, step)
            minus = _left_increment(component, -step)
            if target == "A":
                distance_plus = _torch_distance(
                    box_a, box_b, plus @ htm_a, htm_b, device
                )
                distance_minus = _torch_distance(
                    box_a, box_b, minus @ htm_a, htm_b, device
                )
            else:
                distance_plus = _torch_distance(
                    box_a, box_b, htm_a, plus @ htm_b, device
                )
                distance_minus = _torch_distance(
                    box_a, box_b, htm_a, minus @ htm_b, device
                )
            gradient[component] = (distance_plus - distance_minus) / (2.0 * step)
        gradients.append(gradient)
    return tuple(gradients)


def _cases(random_count: int, seed: int) -> list[PoseCase]:
    cases = [
        PoseCase("separated", _htm((0.1, -0.2, 0.1)), _htm((2.0, 0.4, -0.1))),
        PoseCase("close", _htm((0.0, 0.0, 0.0)), _htm((0.76, 0.03, 0.02))),
        PoseCase(
            "intersecting", _htm((0.0, 0.0, 0.0)), _htm((0.25, 0.10, 0.05))
        ),
        PoseCase(
            "rotated",
            _htm((0.2, -0.3, 0.15), (1.0, 2.0, -1.0), 0.55),
            _htm((1.0, 0.25, -0.2), (-2.0, 1.0, 1.0), -0.80),
        ),
    ]
    rng = np.random.default_rng(seed)
    for index in range(random_count):
        axis_a = rng.normal(size=3)
        axis_b = rng.normal(size=3)
        cases.append(
            PoseCase(
                f"random_{index:02d}",
                _htm(
                    tuple(rng.uniform(-0.5, 0.5, 3)),
                    tuple(axis_a),
                    rng.uniform(-1.2, 1.2),
                ),
                _htm(
                    tuple(rng.uniform(-0.8, 1.4, 3)),
                    tuple(axis_b),
                    rng.uniform(-1.2, 1.2),
                ),
            )
        )
    return cases


def _errors(actual: np.ndarray, reference: np.ndarray):
    absolute = np.abs(actual - reference)
    relative = absolute / np.maximum(np.abs(reference), 1e-8)
    return absolute, relative


def _format(vector: np.ndarray) -> str:
    return np.array2string(
        vector,
        precision=7,
        suppress_small=False,
        floatmode="fixed",
        max_line_width=140,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--step", type=float, default=1e-3)
    parser.add_argument("--random-count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--atol", type=float, default=4e-3)
    args = parser.parse_args()

    if os.environ.get("CPP_SO_FOUND") != "1":
        raise RuntimeError("The existing uaibot_cpp_bind extension is required")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.step <= 0 or args.random_count < 1:
        raise ValueError("--step must be positive and --random-count must be at least 1")

    device = torch.device(args.device)
    error_sets = {"torch/fd": [], "cpp/fd": [], "torch/cpp": []}
    relative_error_sets = {"torch/fd": [], "cpp/fd": [], "torch/cpp": []}
    distance_errors = []
    distance_relative_errors = []

    print(f"component order: {list(COMPONENTS)}")
    print("perturbation: H' = exp(xi^) H (left)")
    print(
        f"device={device}, gamma={GAMMA}, epsilon={EPSILON}, "
        f"finite-difference step={args.step:g}"
    )

    for case in _cases(args.random_count, args.seed):
        box_a = _make_box(case.htm_a, SIZE_A, f"{case.name}_a")
        box_b = _make_box(case.htm_b, SIZE_B, f"{case.name}_b")
        torch_distance, torch_a, torch_b = _torch_analytical(box_a, box_b, device)
        cpp_distance, cpp_a, cpp_b = _cpp_analytical(box_a, box_b)
        finite_a, finite_b = _finite_difference(
            box_a, box_b, case.htm_a, case.htm_b, device, args.step
        )
        distance_errors.append(abs(torch_distance - cpp_distance))
        distance_relative_errors.append(
            abs(torch_distance - cpp_distance) / max(abs(cpp_distance), 1e-8)
        )

        print(f"\n=== {case.name} ===")
        print(
            f"distance torch={torch_distance:+.9f} cpp={cpp_distance:+.9f} "
            f"abs_error={abs(torch_distance - cpp_distance):.3e}"
        )
        for pose, torch_grad, cpp_grad, finite_grad in (
            ("A", torch_a, cpp_a, finite_a),
            ("B", torch_b, cpp_b, finite_b),
        ):
            torch_abs, torch_rel = _errors(torch_grad, finite_grad)
            cpp_abs, cpp_rel = _errors(cpp_grad, finite_grad)
            cross_abs, cross_rel = _errors(torch_grad, cpp_grad)
            error_sets["torch/fd"].extend(torch_abs)
            error_sets["cpp/fd"].extend(cpp_abs)
            error_sets["torch/cpp"].extend(cross_abs)
            relative_error_sets["torch/fd"].extend(torch_rel)
            relative_error_sets["cpp/fd"].extend(cpp_rel)
            relative_error_sets["torch/cpp"].extend(cross_rel)
            print(f"pose {pose}:")
            print(f"  torch analytical : {_format(torch_grad)}")
            print(f"  C++ analytical   : {_format(cpp_grad)}")
            print(f"  central finite-d : {_format(finite_grad)}")
            print(f"  abs torch/fd     : {_format(torch_abs)}")
            print(f"  rel torch/fd     : {_format(torch_rel)}")
            print(f"  abs C++/fd       : {_format(cpp_abs)}")
            print(f"  rel C++/fd       : {_format(cpp_rel)}")
            print(f"  abs torch/C++    : {_format(cross_abs)}")
            print(f"  rel torch/C++    : {_format(cross_rel)}")

    print("\n=== final error summary (all cases, both poses, all components) ===")
    print(
        f"distance torch/C++ absolute max={max(distance_errors):.3e} "
        f"mean={np.mean(distance_errors):.3e}; "
        f"relative max={max(distance_relative_errors):.3e} "
        f"mean={np.mean(distance_relative_errors):.3e}"
    )
    for name, values in error_sets.items():
        values_array = np.asarray(values)
        relative_array = np.asarray(relative_error_sets[name])
        print(
            f"{name:10s} absolute max={values_array.max():.3e} "
            f"mean={values_array.mean():.3e}; "
            f"relative max={relative_array.max():.3e} "
            f"mean={relative_array.mean():.3e}"
        )

    worst = max(max(values) for values in error_sets.values())
    if worst > args.atol:
        raise SystemExit(
            f"FAILED: maximum absolute gradient error {worst:.3e} "
            f"exceeds tolerance {args.atol:.3e}"
        )
    print(f"PASS: maximum absolute gradient error {worst:.3e} <= {args.atol:.3e}")


if __name__ == "__main__":
    main()
