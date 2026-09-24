"""Short value, projected-gradient, and SE(3) finite-difference checks."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import expm

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "UAIbotPy"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import _gpu_house_distance as base
import uaibot as ub
from uaibot.gpu.distance import (
    holder_distance_optimized,
    holder_distance_with_grad,
    holder_distance_with_grad_optimized,
    se3_generators_cached,
)
from uaibot.gpu.geometry import extract_VEF

GAMMA = 2.0
EPSILON = 1e-3
STEP = 1e-3


def _batch(obj, htm):
    vertices, edges, normals = extract_VEF(obj, htm)
    return base.GeometryBatch(
        indices=[0],
        vertices=vertices[None],
        edges=edges[None],
        normals=normals[None],
        device_indices=torch.tensor([0]),
    )


def _args(query, house):
    return (
        query.vertices, query.edges, query.normals,
        house.vertices, house.edges, house.normals,
        GAMMA, EPSILON,
    )


def _left_increment(component, amount):
    twist_hat = np.zeros((4, 4))
    if component < 3:
        twist_hat[component, 3] = amount
    else:
        omega = np.zeros(3)
        omega[component - 3] = amount
        x, y, z = omega
        twist_hat[:3, :3] = ((0, -z, y), (z, 0, -x), (-y, x, 0))
    return expm(twist_hat)


def _finite_difference(box_a, box_b, htm_a, htm_b):
    def distance(pose_a, pose_b):
        query = _batch(box_a, pose_a)
        house = _batch(box_b, pose_b)
        return holder_distance_optimized(*_args(query, house)).item()

    gradients = []
    for target in ("a", "b"):
        values = []
        for component in range(6):
            plus = _left_increment(component, STEP)
            minus = _left_increment(component, -STEP)
            if target == "a":
                high = distance(plus @ htm_a, htm_b)
                low = distance(minus @ htm_a, htm_b)
            else:
                high = distance(htm_a, plus @ htm_b)
                low = distance(htm_a, minus @ htm_b)
            values.append((high - low) / (2 * STEP))
        gradients.append(torch.tensor(values, dtype=torch.float32))
    return gradients


def main():
    htm_a = np.eye(4)
    htm_b = np.eye(4)
    htm_b[:3, :3] = np.asarray(ub.Utils.rotz(0.2))[:3, :3]
    htm_b[:3, 3] = (1.1, 0.2, 0.1)
    box_a = ub.Box(htm=htm_a, name="validation_a", width=0.7, depth=0.5, height=0.4)
    box_b = ub.Box(htm=htm_b, name="validation_b", width=0.6, depth=0.4, height=0.5)
    query = _batch(box_a, htm_a)
    house = _batch(box_b, htm_b)
    reference_distance, reference_pnv = holder_distance_with_grad(*_args(query, house))
    optimized_distance, optimized_pnv = holder_distance_with_grad_optimized(
        *_args(query, house), dtype=torch.float32
    )
    torch.testing.assert_close(optimized_distance, reference_distance, rtol=1e-4, atol=2e-5)
    torch.testing.assert_close(optimized_pnv, reference_pnv, rtol=1e-3, atol=2e-5)

    generators = se3_generators_cached(torch.device("cpu"), torch.float32)
    reference = base._vectorized_group_pose_gradient(
        query, base.PNVResult(house, reference_distance, reference_pnv), generators
    )
    static = base._build_static_pose_cache([house], generators)[0]
    optimized = base._vectorized_group_pose_gradient_cached(
        query, base.PNVResult(house, optimized_distance, optimized_pnv), static, generators
    )
    for actual, expected in zip(optimized, reference):
        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=2e-3)
    finite_difference = _finite_difference(box_a, box_b, htm_a, htm_b)
    for actual, expected in zip(optimized, finite_difference):
        torch.testing.assert_close(actual[0], expected, rtol=1e-2, atol=5e-3)
    print("PASS: optimized distance, projected gradient, and cached SE(3) gradient")


if __name__ == "__main__":
    main()
