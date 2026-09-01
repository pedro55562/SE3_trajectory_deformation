"""
tree.py
=======

Arvore grande feita com primitivas geometricas do UAIbot.

Convencao:
    x -> largura
    y -> profundidade
    z -> altura

A origem da arvore fica no chao, no centro do tronco.
"""

import numpy as np
import uaibot as ub


# =============================================================================
# Helpers internos
# =============================================================================


def _base_htm(htm):
    """Retorna identidade quando htm=None."""
    return np.eye(4) if htm is None else htm


def _segment_htm(p0, p1):
    """Cria a HTM de um cilindro ligando p0 a p1."""
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)

    direction = p1 - p0
    length = np.linalg.norm(direction)

    if length < 1e-9:
        raise ValueError("Cylinder segment must have nonzero length.")

    z_axis = direction / length

    helper = np.array([0.0, 0.0, 1.0])

    if abs(np.dot(z_axis, helper)) > 0.95:
        helper = np.array([1.0, 0.0, 0.0])

    x_axis = np.cross(helper, z_axis)
    x_axis /= np.linalg.norm(x_axis)

    y_axis = np.cross(z_axis, x_axis)

    T = np.eye(4)
    T[0:3, 0] = x_axis
    T[0:3, 1] = y_axis
    T[0:3, 2] = z_axis
    T[0:3, 3] = 0.5 * (p0 + p1)

    return T, length


def _cylinder_between(
    name,
    htm,
    p0,
    p1,
    radius,
    color,
    opacity=1.0,
):
    T_local, length = _segment_htm(p0, p1)

    return ub.Cylinder(
        name=name,
        htm=_base_htm(htm) @ T_local,
        radius=radius,
        height=length,
        color=color,
        opacity=opacity,
    )


def _ball(
    name,
    htm,
    xyz,
    radius,
    color,
    opacity=1.0,
):
    return ub.Ball(
        name=name,
        htm=_base_htm(htm) @ ub.Utils.trn(xyz),
        radius=radius,
        color=color,
        opacity=opacity,
    )


# =============================================================================
# Arvore
# =============================================================================


def create_large_tree(
    htm=None,
    name="large_tree",
    height=8.0,
    crown_radius=3.0,
    trunk_radius=0.45,
    branches_per_level=5,
    include_foliage=True,
    seed=7,
    trunk_color="#6B4423",
    branch_color="#76502B",
    leaf_colors=(
        "#2E7D32",
        "#388E3C",
        "#43A047",
        "#2F6F35",
    ),
):
    """
    Cria uma arvore grande usando cilindros e esferas.

    A arvore possui:
        - tronco segmentado;
        - galhos principais;
        - galhos secundarios;
        - copa formada por esferas.

    A geometria e deterministica para um mesmo seed.
    """

    if height <= 0:
        raise ValueError("'height' must be positive.")

    if crown_radius <= 0:
        raise ValueError("'crown_radius' must be positive.")

    if trunk_radius <= 0:
        raise ValueError("'trunk_radius' must be positive.")

    rng = np.random.default_rng(seed)

    objs = []


    # =========================================================================
    # Tronco
    # =========================================================================

    trunk_points = np.array([
        [0.00,  0.00, 0.00 * height],
        [0.03, -0.01, 0.19 * height],
        [-0.04, 0.04, 0.37 * height],
        [0.05,  0.01, 0.54 * height],
        [-0.03, -0.04, 0.69 * height],
        [0.00,  0.00, 0.82 * height],
    ])

    trunk_radii = trunk_radius * np.array([
        1.00,
        0.91,
        0.80,
        0.67,
        0.52,
    ])

    for i in range(len(trunk_points) - 1):

        objs.append(
            _cylinder_between(
                f"{name}_trunk_{i + 1}",
                htm,
                trunk_points[i],
                trunk_points[i + 1],
                trunk_radii[i],
                trunk_color,
            )
        )


    # =========================================================================
    # Galhos
    # =========================================================================

    level_z = np.array([
        0.38,
        0.48,
        0.58,
        0.67,
        0.75,
    ]) * height

    level_scale = [
        1.00,
        1.00,
        0.94,
        0.84,
        0.72,
    ]

    foliage_centers = []

    branch_id = 0
    twig_id = 0

    for level, (z0, scale) in enumerate(
        zip(level_z, level_scale)
    ):

        angle_offset = level * 0.61

        for j in range(branches_per_level):

            branch_id += 1

            theta = (
                2 * np.pi * j / branches_per_level
                + angle_offset
                + rng.uniform(-0.20, 0.20)
            )

            horizontal = (
                crown_radius
                * scale
                * rng.uniform(0.74, 1.00)
            )

            rise = height * rng.uniform(
                0.055,
                0.11,
            )

            p0 = np.array([
                rng.uniform(-0.05, 0.05),
                rng.uniform(-0.05, 0.05),
                z0,
            ])

            p1 = p0 + np.array([
                horizontal * np.cos(theta),
                horizontal * np.sin(theta),
                rise,
            ])

            branch_radius = (
                trunk_radius
                * (0.36 - 0.035 * level)
            )

            objs.append(
                _cylinder_between(
                    f"{name}_branch_{branch_id}",
                    htm,
                    p0,
                    p1,
                    branch_radius,
                    branch_color,
                )
            )

            foliage_centers.append(p1)


            # =================================================================
            # Galhos secundarios
            # =================================================================

            for side in (-1.0, 1.0):

                twig_id += 1

                start_ratio = rng.uniform(
                    0.52,
                    0.68,
                )

                q0 = p0 + start_ratio * (p1 - p0)

                twig_angle = (
                    theta
                    + side * rng.uniform(0.38, 0.70)
                )

                twig_length = (
                    horizontal
                    * rng.uniform(0.34, 0.50)
                )

                twig_rise = (
                    height
                    * rng.uniform(0.025, 0.070)
                )

                q1 = q0 + np.array([
                    twig_length * np.cos(twig_angle),
                    twig_length * np.sin(twig_angle),
                    twig_rise,
                ])

                objs.append(
                    _cylinder_between(
                        f"{name}_twig_{twig_id}",
                        htm,
                        q0,
                        q1,
                        branch_radius * 0.52,
                        branch_color,
                    )
                )

                foliage_centers.append(q1)


    # =========================================================================
    # Galhos superiores
    # =========================================================================

    top_origin = np.array([
        0.0,
        0.0,
        0.70 * height,
    ])

    for i in range(7):

        theta = 2 * np.pi * i / 7 + 0.25

        radial = (
            crown_radius
            * rng.uniform(0.30, 0.55)
        )

        p1 = np.array([
            radial * np.cos(theta),
            radial * np.sin(theta),
            height * rng.uniform(0.88, 0.98),
        ])

        objs.append(
            _cylinder_between(
                f"{name}_top_branch_{i + 1}",
                htm,
                top_origin,
                p1,
                trunk_radius * 0.18,
                branch_color,
            )
        )

        foliage_centers.append(p1)


    # =========================================================================
    # Copa
    # =========================================================================

    if include_foliage:

        leaf_id = 0

        for center in foliage_centers:

            leaf_id += 1

            radius = (
                rng.uniform(0.46, 0.68)
                * crown_radius / 3.0
            )

            offset = np.array([
                rng.uniform(-0.16, 0.16),
                rng.uniform(-0.16, 0.16),
                rng.uniform(-0.05, 0.22),
            ]) * crown_radius

            objs.append(
                _ball(
                    f"{name}_leaf_cluster_{leaf_id}",
                    htm,
                    center + offset,
                    radius,
                    leaf_colors[
                        leaf_id % len(leaf_colors)
                    ],
                )
            )


        # Preenchimento interno da copa

        inner_count = 18

        for i in range(inner_count):

            theta = (
                2 * np.pi * i / inner_count
                + rng.uniform(-0.15, 0.15)
            )

            radial = (
                crown_radius
                * rng.uniform(0.18, 0.66)
            )

            z = (
                height
                * rng.uniform(0.62, 0.91)
            )

            center = np.array([
                radial * np.cos(theta),
                radial * np.sin(theta),
                z,
            ])

            objs.append(
                _ball(
                    f"{name}_inner_leaf_{i + 1}",
                    htm,
                    center,
                    rng.uniform(0.52, 0.76)
                    * crown_radius / 3.0,
                    leaf_colors[
                        i % len(leaf_colors)
                    ],
                )
            )

    return objs