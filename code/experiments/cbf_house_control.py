"""First-order SE(3) CBF control through the repository's existing house."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import expm


PROJECT_DIR = Path(__file__).resolve().parents[2]
UAIBOT_DIR = PROJECT_DIR / "UAIbotPy"
VALIDATION_DIR = PROJECT_DIR / "code" / "validation"
for path in (PROJECT_DIR, UAIBOT_DIR, VALIDATION_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import _gpu_house_distance as gpu_house
import uaibot as ub
from house.l_house_uaibot import build_plan_data, build_uaibot_objects
from uaibot.gpu.distance import (
    holder_distance_with_grad_optimized,
    se3_generators_cached,
)
from uaibot.gpu.geometry import extract_VEF


GAMMA = 2.0
EPSILON = 1e-3
SAFE_DISTANCE = 0.05
NEARBY_DISTANCE = .4
ETA = 1.0
CONTROLLER_GAIN = 2
DAMPING = 1e-3
DT = 0.01
MAX_TIME = 15.0
GOAL_TOLERANCE = 2e-3
VELOCITY_LIMITS = np.array([.90, .90, .5, 1.80, 1.80, 1.80])
BODY_SIZE = (0.40, 0.30, 0.20)


# Both poses lie in the living-room portion of the existing L-shaped house.
htm_start = ub.Utils.trn([8.75, 1.00, 1.50])
htm_goal = ub.Utils.trn([-.2, 7.00, -0.20]) * htm_start #* ub.Utils.roty(np.pi/7) * ub.Utils.rotx(np.pi/2)


def _task_function(htm: np.ndarray, htm_target: np.ndarray):
    """Pose task and spatial-twist Jacobian from Robot._task_function."""
    htm = np.asarray(htm, dtype=float)
    htm_target = np.asarray(htm_target, dtype=float)
    position = htm[:3, 3]
    x_axis, y_axis, z_axis = htm[:3, 0], htm[:3, 1], htm[:3, 2]
    target_position = htm_target[:3, 3]
    target_x = htm_target[:3, 0]
    target_y = htm_target[:3, 1]
    target_z = htm_target[:3, 2]

    task = np.zeros(6)
    task[:3] = position - target_position
    task[3] = max(1.0 - target_x @ x_axis, 0.0)
    task[4] = max(1.0 - target_y @ y_axis, 0.0)
    task[5] = max(1.0 - target_z @ z_axis, 0.0)

    jacobian = np.zeros((6, 6))
    jacobian[:3, :3] = np.eye(3)
    jacobian[:3, 3:] = -np.asarray(ub.Utils.S(position))
    jacobian[3, 3:] = target_x @ np.asarray(ub.Utils.S(x_axis))
    jacobian[4, 3:] = target_y @ np.asarray(ub.Utils.S(y_axis))
    jacobian[5, 3:] = target_z @ np.asarray(ub.Utils.S(z_axis))
    return task, jacobian


def _nominal_velocity(htm: np.ndarray) -> tuple[np.ndarray, float]:
    task, jacobian = _task_function(htm, htm_goal)
    velocity = np.asarray(
        ub.Utils.dp_inv_solve(
            jacobian, -CONTROLLER_GAIN * np.tanh(task), DAMPING
        )
    ).reshape(6)
    return velocity, float(np.linalg.norm(task))


def _twist_hat(twist: np.ndarray) -> np.ndarray:
    hat = np.zeros((4, 4))
    hat[:3, :3] = np.asarray(ub.Utils.S(twist[3:]))
    hat[:3, 3] = twist[:3]
    return hat


def _integrate_left(htm: np.ndarray, twist: np.ndarray) -> np.ndarray:
    return expm(DT * _twist_hat(twist)) @ htm


def _build_gpu_cache(house_objects, device: torch.device):
    collision_objects = [
        gpu_house._as_gpu_compatible_polyhedron(obj) for obj in house_objects
    ]
    components = [extract_VEF(obj) for obj in collision_objects]
    house_batches = [
        gpu_house._batch_to_device(batch, device)
        for batch in gpu_house._group_and_stack(components)
    ]
    generators = se3_generators_cached(device, torch.float32)
    static_batches = gpu_house._build_static_pose_cache(
        house_batches, generators
    )
    return collision_objects, house_batches, static_batches, generators


def _local_body_geometry(body, device: torch.device):
    vertices, edges, normals = extract_VEF(body, np.eye(4))
    return (
        vertices.to(device=device, dtype=torch.float32),
        edges.to(device=device, dtype=torch.float32),
        normals.to(device=device, dtype=torch.float32),
    )


def _body_batch(local_geometry, htm: np.ndarray, device: torch.device):
    rotation = torch.as_tensor(
        htm[:3, :3], dtype=torch.float32, device=device
    )
    translation = torch.as_tensor(
        htm[:3, 3], dtype=torch.float32, device=device
    )
    local_vertices, local_edges, local_normals = local_geometry
    vertices = local_vertices @ rotation.T + translation
    edges = local_edges @ rotation.T
    normals = local_normals @ rotation.T
    return gpu_house.GeometryBatch(
        indices=[0],
        vertices=vertices[None],
        edges=edges[None],
        normals=normals[None],
    )


def _distances_and_gradients(
    query,
    house_batches,
    static_batches,
    generators,
    obstacle_count: int,
):
    device = query.vertices.device
    distances = torch.empty(
        obstacle_count, dtype=query.vertices.dtype, device=device
    )
    gradients = torch.empty(
        obstacle_count, 6, dtype=query.vertices.dtype, device=device
    )

    for house, static in zip(house_batches, static_batches):
        batch_distances, grad_pnv = holder_distance_with_grad_optimized(
            query.vertices,
            query.edges,
            query.normals,
            house.vertices,
            house.edges,
            house.normals,
            GAMMA,
            EPSILON,
            dtype=torch.float32,
        )
        result = gpu_house.PNVResult(house, batch_distances, grad_pnv)
        pair_gradient, _ = gpu_house._vectorized_group_pose_gradient_cached(
            query, result, static, generators
        )
        distances.index_copy_(
            0, house.device_indices, batch_distances.reshape(-1)
        )
        gradients.index_copy_(0, house.device_indices, pair_gradient)

    return distances, gradients


def _cbf_control(
    nominal: np.ndarray,
    distances: torch.Tensor,
    gradients: torch.Tensor,
) -> np.ndarray:
    nearby = distances <= NEARBY_DISTANCE
    nearby_distances = distances[nearby].detach().cpu().double().numpy()
    nearby_gradients = gradients[nearby].detach().cpu().double().numpy()

    cbf_rhs = -ETA * (nearby_distances - SAFE_DISTANCE)
    constraint_matrix = np.vstack(
        (nearby_gradients, np.eye(6), -np.eye(6))
    )
    constraint_rhs = np.concatenate(
        (cbf_rhs, -VELOCITY_LIMITS, -VELOCITY_LIMITS)
    )
    solution = ub.Utils.solve_qp(
        H=np.eye(6),
        f=-nominal,
        A=constraint_matrix,
        b=constraint_rhs,
    )
    return np.asarray(solution, dtype=float).reshape(6)


def run(save_animation: bool = True, max_steps: int | None = None):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the GPU Holder CBF simulation")

    device = torch.device("cuda")
    house_objects = build_uaibot_objects(build_plan_data())
    controlled_body = ub.Box(
        htm=htm_start,
        name="cbf_controlled_body",
        width=BODY_SIZE[0],
        depth=BODY_SIZE[1],
        height=BODY_SIZE[2],
        color="#FF8C00",
    )
    goal_frame = ub.Frame(
        htm=htm_goal,
        name="cbf_goal_frame",
        size=1.0,
    )
    start_frame = ub.Frame(
        htm=htm_start,
        name="cbf_start_frame",
        size=1.0,
    )
    body_frame = ub.Frame(
        htm=htm_start,
        name="cbf_body_frame",
        size=0.30,
    )
    simulation = ub.Simulation(
        house_objects
        + [controlled_body, start_frame, goal_frame, body_frame]
    )

    collision_objects, house_batches, static_batches, generators = (
        _build_gpu_cache(house_objects, device)
    )
    local_geometry = _local_body_geometry(controlled_body, device)
    obstacle_count = len(collision_objects)

    htm = np.asarray(htm_start, dtype=float).copy()
    controlled_body.set_ani_frame(htm)
    body_frame.set_ani_frame(htm)
    minimum_distance = np.inf
    goal_reached = False
    final_error = np.inf
    step_limit = (
        int(MAX_TIME / DT) if max_steps is None else min(max_steps, int(MAX_TIME / DT))
    )

    for step in range(step_limit):
        query = _body_batch(local_geometry, htm, device)
        distances, gradients = _distances_and_gradients(
            query,
            house_batches,
            static_batches,
            generators,
            obstacle_count,
        )
        minimum_distance = min(minimum_distance, float(distances.min().item()))
        nominal, final_error = _nominal_velocity(htm)
        if final_error <= GOAL_TOLERANCE:
            goal_reached = True
            break
        # control = nominal     
        control = _cbf_control(nominal, distances, gradients)
        htm = _integrate_left(htm, control)
        controlled_body.add_ani_frame((step + 1) * DT, htm)
        body_frame.add_ani_frame((step + 1) * DT, htm)

    _, final_error = _nominal_velocity(htm)
    goal_reached = goal_reached or final_error <= GOAL_TOLERANCE

    if save_animation:
        html_dir = PROJECT_DIR / "code" / "html"
        html_dir.mkdir(parents=True, exist_ok=True)
        simulation.save(str(html_dir), "cbf_house_control")

    print(f"Goal reached: {goal_reached}")
    print(f"Final task error: {final_error:.6f}")
    print(f"Minimum distance observed: {minimum_distance:.6f} m")
    return goal_reached, final_error, minimum_distance


if __name__ == "__main__":
    run()
