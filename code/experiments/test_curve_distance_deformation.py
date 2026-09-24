"""Pointwise SE(3) curve deformation from a cylinder-box distance gradient."""

import sys
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import expm
from scipy.ndimage import gaussian_filter1d


PROJECT_DIR = Path(__file__).resolve().parents[2]
for path in (PROJECT_DIR, PROJECT_DIR / "UAIbotPy", PROJECT_DIR / "code" / "validation"):
    sys.path.insert(0, str(path))

import _gpu_house_distance as gpu_distance
import uaibot as ub
from uaibot.gpu.distance import holder_distance_with_grad_optimized, se3_generators_cached
from uaibot.gpu.geometry import extract_VEF
from uaibot.simobjects.curve import Curve, CurveSE3


# Geometry: an offset box overlaps the robot along one side of the line.
NUM_POINTS = 800
CURVE_LENGTH = 5.0
BOX_SIZE = 1.6
BOX_CENTER_X = 0.40
BOX_CENTER_Y = 0.40
ROBOT_RADIUS = 0.30
ROBOT_HEIGHT = 0.50
SCENE_HEIGHT = 0.0  # Initial height of the robot centers.

# Distance-gradient integration.
GPU_BATCH_SIZE = 128
MAX_ITERATIONS = 160
DT = 0.0125
GAIN = 2.0
DELTA = 0.30
TAPER_WIDTH = 0.20  # Fade each update near the influence boundary.
GAMMA = 2.0
EPSILON = 1e-3

# Neighbor coupling keeps the sampled curve coherent.
SMOOTHING_WIDTH = 0.20  # Meters along the curve.
SMOOTHING_BLEND = 0.85  # Fraction of neighbor-averaged 6D twist.
SHAPE_WIDTH = 0.08     # Meters along the curve.
SHAPE_GAIN = 4.0       # Position coupling rate.


def make_initial_poses():
    """A straight, constant-orientation path whose robot overlaps the offset box."""
    poses = np.repeat(np.eye(4)[None], NUM_POINTS, axis=0)
    poses[:, :3, :3] = np.asarray(ub.Utils.roty(np.deg2rad(15.0)))[:3, :3]
    poses[:, 0, 3] = np.linspace(-CURVE_LENGTH / 2, CURVE_LENGTH / 2, NUM_POINTS)
    poses[:, 2, 3] = SCENE_HEIGHT
    return poses


def make_geometry(box, robot, device):
    """Cache fixed box geometry and a local prism approximation of the cylinder."""
    robot_polytope = gpu_distance._as_gpu_compatible_polyhedron(robot)
    local_robot = tuple(
        part.to(device) for part in extract_VEF(robot_polytope, np.eye(4))
    )
    box_vertices, box_edges, box_normals = extract_VEF(box)
    fixed_box = gpu_distance.GeometryBatch(
        indices=[0],
        vertices=box_vertices[None].to(device),
        edges=box_edges[None].to(device),
        normals=box_normals[None].to(device),
    )
    generators = se3_generators_cached(device, torch.float32)
    return local_robot, fixed_box, generators


def evaluate_batch(poses, local_robot, fixed_box, generators, device):
    """Return D and its left SE(3) gradient for a group of robot poses."""
    rotations = torch.as_tensor(poses[:, :3, :3], dtype=torch.float32, device=device)
    positions = torch.as_tensor(poses[:, :3, 3], dtype=torch.float32, device=device)
    vertices, edges, normals = local_robot
    rotations_t = rotations.transpose(1, 2)
    robots = gpu_distance.GeometryBatch(
        indices=list(range(len(poses))),
        vertices=vertices[None] @ rotations_t + positions[:, None],
        edges=edges[None] @ rotations_t,
        normals=normals[None] @ rotations_t,
    )

    distances, pnv_gradient = holder_distance_with_grad_optimized(
        fixed_box.vertices, fixed_box.edges, fixed_box.normals,
        robots.vertices, robots.edges, robots.normals,
        GAMMA, EPSILON, dtype=torch.float32,
    )
    result = gpu_distance.PNVResult(robots, distances, pnv_gradient)
    _, robot_gradient = gpu_distance._vectorized_group_pose_gradient(
        fixed_box, result, generators
    )
    return distances[0].detach().cpu().numpy(), robot_gradient.detach().cpu().numpy()


def evaluate_curve(poses, local_robot, fixed_box, generators, device):
    """Evaluate every pose in bounded GPU batches, preserving curve order."""
    distances = np.empty(len(poses))
    gradients = np.empty((len(poses), 6))
    for start in range(0, len(poses), GPU_BATCH_SIZE):
        stop = min(start + GPU_BATCH_SIZE, len(poses))
        distances[start:stop], gradients[start:stop] = evaluate_batch(
            poses[start:stop], local_robot, fixed_box, generators, device
        )
    if not np.isfinite(distances).all() or not np.isfinite(gradients).all():
        raise RuntimeError("Non-finite distance or SE(3) gradient")
    return distances, gradients


def integrate_pose(pose, twist):
    """Apply exp(dt * S(xi)) on the left; linear components come first."""
    # xi = [v_x, v_y, v_z, omega_x, omega_y, omega_z].
    twist_hat = np.zeros((4, 4))
    twist_hat[:3, :3] = np.asarray(ub.Utils.S(twist[3:]))
    twist_hat[:3, 3] = twist[:3]
    return expm(DT * twist_hat) @ pose


def deformation_twists(curve, gradients, distances, active, point_spacing):
    """Couple nearby updates while moving each center only sideways in y."""
    twists = GAIN * gradients
    twists[~active] = 0.0
    neighbor_twists = gaussian_filter1d(
        twists, sigma=SMOOTHING_WIDTH / point_spacing, axis=0, mode="nearest"
    )
    twists = (1.0 - SMOOTHING_BLEND) * twists + SMOOTHING_BLEND * neighbor_twists

    nearby_positions = gaussian_filter1d(
        curve.positions, sigma=SHAPE_WIDTH / point_spacing, axis=0, mode="nearest"
    )
    twists[:, :3] += SHAPE_GAIN * (nearby_positions - curve.positions)

    # For a left spatial twist, p_dot = v + omega x p. Cancel the
    # rotation-induced x/z motion so each curve station moves only in y.
    rotational_motion = np.cross(twists[:, 3:], curve.positions)
    point_velocity = twists[:, :3] + rotational_motion
    point_velocity[:, 0] = 0.0
    point_velocity[:, 2] = 0.0
    twists[:, :3] = point_velocity - rotational_motion
    weights = np.sqrt(np.clip((DELTA - distances) / TAPER_WIDTH, 0.0, 1.0))
    weights[[0, -1]] = 0.0
    twists *= weights[:, None]  # Exactly zero once D >= delta.
    return twists


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this GPU gradient experiment")
    device = torch.device("cuda")

    box = ub.Box(
        htm=ub.Utils.trn([BOX_CENTER_X, BOX_CENTER_Y, SCENE_HEIGHT]),
        name="fixed_box",
        width=BOX_SIZE, depth=BOX_SIZE, height=BOX_SIZE,
        color="#777777", opacity=0.35,
    )
    robot_model = ub.Cylinder(
        name="robot_geometry", radius=ROBOT_RADIUS, height=ROBOT_HEIGHT
    )
    local_robot, fixed_box, generators = make_geometry(box, robot_model, device)

    initial_poses = make_initial_poses()
    curve = CurveSE3(
        name="moving_curve", points=initial_poses, color="orange", size=0.02,
        frame_size=0.18,
        frame_indices=np.linspace(0, NUM_POINTS - 1, 9, dtype=int).tolist(),
    )
    reference = Curve(
        name="initial_curve", points=initial_poses[:, :3, 3],
        color="blue", size=0.02,
    )
    shown_robots = []
    for index in (NUM_POINTS // 4, NUM_POINTS // 2, 3 * NUM_POINTS // 4):
        shown_robots.append((index, ub.Cylinder(
            htm=initial_poses[index], name=f"robot_at_{index}",
            radius=ROBOT_RADIUS, height=ROBOT_HEIGHT,
            color="#FF8C00", opacity=0.3,
        )))

    distances, gradients = evaluate_curve(
        curve.htm, local_robot, fixed_box, generators, device
    )
    initial_minimum = float(distances.min())
    if initial_minimum >= 0:
        raise RuntimeError("Initial robot trajectory must collide with the box")
    if np.any(distances[[0, -1]] < DELTA):
        raise RuntimeError("Fixed endpoints must start outside the influence distance")

    point_spacing = CURVE_LENGTH / (NUM_POINTS - 1)

    iterations = 0
    for step in range(1, MAX_ITERATIONS + 1):
        active = distances < DELTA
        active[[0, -1]] = False
        if not active.any():
            break

        twists = deformation_twists(curve, gradients, distances, active, point_spacing)

        next_poses = curve.htm.copy()
        for index in np.flatnonzero(active):
            next_poses[index] = integrate_pose(next_poses[index], twists[index])
        time = step * DT
        curve.add_ani_frame(time, next_poses)
        for index, robot in shown_robots:
            robot.add_ani_frame(time, next_poses[index])

        # Every new step uses the distances and gradients at the updated curve.
        distances, gradients = evaluate_curve(
            curve.htm, local_robot, fixed_box, generators, device
        )
        iterations = step

    np.testing.assert_array_equal(curve[0], initial_poses[0])
    np.testing.assert_array_equal(curve[-1], initial_poses[-1])

    html_dir = PROJECT_DIR / "code" / "html"
    html_dir.mkdir(parents=True, exist_ok=True)
    ub.Simulation(
        [box, reference, curve] + [robot for _, robot in shown_robots],
        show_grid=False, show_world_frame=False,
    ).save(
        str(html_dir), "test_curve_distance_deformation"
    )

    unresolved = int(np.count_nonzero(distances[1:-1] < DELTA))
    neighbor_gap = np.linalg.norm(np.diff(curve.positions, axis=0), axis=1).max()
    vertical_deviation = np.abs(curve.positions[:, 2] - SCENE_HEIGHT).max()
    longitudinal_deviation = np.abs(curve.positions[:, 0] - initial_poses[:, 0, 3]).max()
    relative_rotations = curve.orientations @ np.swapaxes(
        initial_poses[:, :3, :3], 1, 2
    )
    rotation_traces = np.trace(relative_rotations, axis1=1, axis2=2)
    rotation_angles = np.arccos(np.clip((rotation_traces - 1.0) / 2.0, -1.0, 1.0))
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Points: {NUM_POINTS}; iterations: {iterations}; delta={DELTA}, gain={GAIN}, dt={DT}")
    print(f"Minimum distance: {initial_minimum:.6f} -> {float(distances.min()):.6f} m")
    print(f"Interior points still below delta: {unresolved}/{NUM_POINTS - 2}")
    print(f"Maximum neighbor gap: {neighbor_gap:.4f} m")
    print(f"Maximum vertical center deviation: {vertical_deviation:.6f} m")
    print(f"Maximum longitudinal center deviation: {longitudinal_deviation:.6f} m")
    print(f"Maximum orientation change: {np.rad2deg(rotation_angles.max()):.1f} degrees")
    print(f"HTML: {html_dir / 'test_curve_distance_deformation.html'}")


if __name__ == "__main__":
    main()
