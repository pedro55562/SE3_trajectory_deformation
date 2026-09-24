"""Animate position and orientation of a dense SE(3) curve."""

import sys
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR / "UAIbotPy"))

from uaibot import Simulation
from uaibot.simobjects.curve import Curve, CurveSE3


# Edit these parameters to change the example.
NUM_POINTS = 800                 # Number of poses along the curve.
CURVE_LENGTH = 4.0               # Initial length along x, in meters.
NUM_VISIBLE_FRAMES = 13         # Orientation axes shown along the curve.
POINT_SIZE = 0.025
FRAME_SIZE = 0.15
DT = 0.01                       # Seconds between animation keyframes.
DEFORMATION_STEP = 1.0 / 120.0   # Fraction of the deformation applied per step.
Y_AMPLITUDE = 1.5               # Maximum translation in y, in meters.
YAW_AMPLITUDE_DEG = 60.0        # Maximum rotation about z, in degrees.
PROFILE_SIGMA = 0.25            # Width of the Gaussian over s in [0, 1].


def main():
    if (NUM_POINTS < 3 or not 1 <= NUM_VISIBLE_FRAMES <= NUM_POINTS
            or DT <= 0 or not 0 < DEFORMATION_STEP <= 1 or PROFILE_SIGMA <= 0):
        raise ValueError("Check NUM_POINTS, NUM_VISIBLE_FRAMES, DT, DEFORMATION_STEP and PROFILE_SIGMA")

    poses = np.repeat(np.eye(4)[None, :, :], NUM_POINTS, axis=0)
    poses[:, 0, 3] = np.linspace(0.0, CURVE_LENGTH, NUM_POINTS)
    curve = CurveSE3(
        name="deformed_curve", points=poses, color="orange",
        size=POINT_SIZE, frame_size=FRAME_SIZE,
        frame_indices=np.linspace(0, NUM_POINTS - 1, NUM_VISIBLE_FRAMES, dtype=int).tolist(),
    )
    before = Curve(name="initial_curve", points=curve.positions.copy(),
                   color="blue", size=POINT_SIZE)

    initial_poses = curve.htm.copy()
    s = np.linspace(0.0, 1.0, len(curve))
    profile = np.exp(-((s - 0.5) / PROFILE_SIGMA) ** 2)
    profile[[0, -1]] = 0.0  # Keep both endpoint poses fixed.
    displacement = Y_AMPLITUDE * profile
    yaw = np.deg2rad(YAW_AMPLITUDE_DEG) * profile

    steps = int(np.ceil(1.0 / DEFORMATION_STEP))
    for step in range(1, steps + 1):
        fraction = 1.0 if step == steps else step * DEFORMATION_STEP
        progress = 0.5 - 0.5 * np.cos(np.pi * fraction)
        updated_poses = initial_poses.copy()
        updated_poses[:, 1, 3] += progress * displacement
        angles = progress * yaw
        updated_poses[:, 0, 0] = np.cos(angles)
        updated_poses[:, 0, 1] = -np.sin(angles)
        updated_poses[:, 1, 0] = np.sin(angles)
        updated_poses[:, 1, 1] = np.cos(angles)
        curve.add_ani_frame(step * DT, updated_poses)

    middle = len(curve) // 2
    np.testing.assert_array_equal(curve.get_points_at_time(0), initial_poses)
    np.testing.assert_array_equal(curve[0], initial_poses[0])
    np.testing.assert_array_equal(curve[-1], initial_poses[-1])
    np.testing.assert_allclose(curve.positions[middle, 1], displacement[middle])
    np.testing.assert_allclose(curve.orientations[middle, 0, 0], np.cos(yaw[middle]))

    html_dir = PROJECT_DIR / "code" / "html"
    html_dir.mkdir(parents=True, exist_ok=True)
    Simulation([before, curve]).save(str(html_dir), "test_curve_deformation")
    print(f"Saved {len(curve)} points and {steps + 1} keyframes "
          f"({steps * DT:.2f} s) to {html_dir / 'test_curve_deformation.html'}")


if __name__ == "__main__":
    main()
