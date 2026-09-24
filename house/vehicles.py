"""
vehicles.py
===========

Veiculos simples e detalhados construidos apenas com primitivas do UAIbot.

Convencao adotada:
    x -> largura (esquerda/direita)
    y -> comprimento do veiculo (frente/traseira)
    z -> altura

A frente dos veiculos aponta para -y. A origem local fica no centro da
projecao do veiculo sobre o chao (z = 0).

Todas as funcoes:
    - recebem uma HTM global opcional;
    - permitem customizar dimensoes e cores principais;
    - retornam LIST[simobject];
    - calculam as posicoes a partir das dimensoes das proprias pecas.

Exemplo:
    import uaibot as ub
    from vehicles import create_work_pickup

    truck = create_work_pickup(
        htm=ub.Utils.trn([0.0, 0.0, 0.0]) @ ub.Utils.rotz(0.25),
        include_cargo=True,
    )

    sim = ub.Simulation.create_sim_grid(truck)
    sim.run()
"""

import numpy as np
import uaibot as ub


# =============================================================================
# Helpers internos
# =============================================================================


def _base_htm(htm):
    """Retorna identidade quando htm=None."""
    return np.eye(4) if htm is None else htm


def _pose(htm, xyz, rotation=None):
    """Aplica uma pose local sobre a pose global do veiculo."""
    T = _base_htm(htm) @ ub.Utils.trn(xyz)
    if rotation is not None:
        T = T @ rotation
    return T


def _box(name, htm, xyz, width, depth, height, color, opacity=1.0, rotation=None):
    return ub.Box(
        name=name,
        htm=_pose(htm, xyz, rotation),
        width=width,
        depth=depth,
        height=height,
        color=color,
        opacity=opacity,
    )


def _cyl_x(name, htm, xyz, radius, length, color, opacity=1.0):
    """Cilindro com eixo ao longo de x, util para rodas/eixos."""
    return ub.Cylinder(
        name=name,
        htm=_pose(htm, xyz, ub.Utils.roty(np.pi / 2)),
        radius=radius,
        height=length,
        color=color,
        opacity=opacity,
    )


def _cyl_z(name, htm, xyz, radius, height, color, opacity=1.0):
    return ub.Cylinder(
        name=name,
        htm=_pose(htm, xyz),
        radius=radius,
        height=height,
        color=color,
        opacity=opacity,
    )


def _validate_positive(**values):
    for key, value in values.items():
        if value <= 0:
            raise ValueError(f"'{key}' must be positive.")


def _local_offset_xyz(center_xyz, local_offset=None, rotation=None):
    """Aplica um deslocamento local considerando a rotacao da peca."""
    xyz = np.array(center_xyz, dtype=float)
    if local_offset is None:
        return xyz.tolist()

    offset = np.array(local_offset, dtype=float)
    if rotation is None:
        return (xyz + offset).tolist()

    rot = np.array(rotation, dtype=float)[:3, :3]
    return (xyz + rot @ offset).tolist()


def _add_glass_frame(
    objs,
    htm,
    name,
    center_xyz,
    glass_width,
    glass_height,
    depth,
    color,
    rotation=None,
    side_trim_w=None,
    top_trim_h=None,
    overlap=0.012,
):
    """Adiciona uma moldura simples em volta de um vidro retangular."""
    side_trim_w = max(0.028, glass_width * 0.055) if side_trim_w is None else side_trim_w
    top_trim_h = max(0.030, glass_height * 0.080) if top_trim_h is None else top_trim_h

    top_bottom_w = glass_width + 2 * (side_trim_w * 0.72)
    side_bar_h = glass_height + 2 * (top_trim_h * 0.42)

    frame_parts = (
        ("top",    [0.0, 0.0,  glass_height / 2 + top_trim_h / 2 - overlap], top_bottom_w, depth, top_trim_h),
        ("bottom", [0.0, 0.0, -glass_height / 2 - top_trim_h / 2 + overlap], top_bottom_w, depth, top_trim_h),
        ("left",   [-glass_width / 2 - side_trim_w / 2 + overlap, 0.0, 0.0], side_trim_w, depth, side_bar_h),
        ("right",  [ glass_width / 2 + side_trim_w / 2 - overlap, 0.0, 0.0], side_trim_w, depth, side_bar_h),
    )

    for suffix, local_offset, width, part_depth, height in frame_parts:
        objs.append(_box(
            f"{name}_{suffix}", htm,
            _local_offset_xyz(center_xyz, local_offset, rotation),
            width, part_depth, height, color,
            rotation=rotation,
        ))


def _add_wheel(
    objs,
    htm,
    name,
    x,
    y,
    z,
    radius,
    tire_width,
    tire_color="#151515",
    rim_color="#B8BDC3",
    hub_color="#5B6168",
):
    """Adiciona pneu, aro e cubo com o mesmo centro geometrico."""
    rim_radius = radius * 0.56
    hub_radius = radius * 0.20

    objs.append(_cyl_x(
        f"{name}_tire", htm, [x, y, z],
        radius, tire_width, tire_color,
    ))
    objs.append(_cyl_x(
        f"{name}_rim", htm, [x, y, z],
        rim_radius, tire_width * 1.025, rim_color,
    ))
    objs.append(_cyl_x(
        f"{name}_hub", htm, [x, y, z],
        hub_radius, tire_width * 1.045, hub_color,
    ))


def _add_pallet(
    objs,
    htm,
    name,
    center_x,
    center_y,
    z_bottom,
    width,
    depth,
    height,
    wood_color="#A97847",
):
    """Pallet simples com runners e ripas, totalmente contido em width x depth."""
    runner_h = height * 0.42
    slat_h = height - runner_h
    runner_w = min(width * 0.12, 0.10)
    slat_gap = min(depth * 0.025, 0.025)
    n_slats = 6
    slat_depth = (depth - (n_slats - 1) * slat_gap) / n_slats

    runner_x = width / 2 - runner_w / 2
    for idx, px in enumerate((-runner_x, 0.0, runner_x), 1):
        objs.append(_box(
            f"{name}_runner_{idx}", htm,
            [center_x + px, center_y, z_bottom + runner_h / 2],
            runner_w, depth, runner_h, wood_color,
        ))

    first_y = -depth / 2 + slat_depth / 2
    for idx in range(n_slats):
        py = first_y + idx * (slat_depth + slat_gap)
        objs.append(_box(
            f"{name}_slat_{idx+1}", htm,
            [center_x, center_y + py, z_bottom + runner_h + slat_h / 2],
            width, slat_depth, slat_h, wood_color,
        ))


def _add_crate(
    objs,
    htm,
    name,
    center_x,
    center_y,
    z_bottom,
    width,
    depth,
    height,
    color="#B88752",
    band_color="#5E4936",
):
    """Caixa grande com duas cintas decorativas externas."""
    objs.append(_box(
        f"{name}_body", htm,
        [center_x, center_y, z_bottom + height / 2],
        width, depth, height, color,
    ))

    band_w = min(width * 0.055, 0.045)
    band_depth = depth + 0.006
    band_height = height + 0.006
    band_x = width * 0.27
    for idx, px in enumerate((-band_x, band_x), 1):
        objs.append(_box(
            f"{name}_band_{idx}", htm,
            [center_x + px, center_y, z_bottom + height / 2],
            band_w, band_depth, band_height, band_color,
        ))


def _add_barrel(
    objs,
    htm,
    name,
    center_x,
    center_y,
    z_bottom,
    radius,
    height,
    color="#315F78",
    ring_color="#24282C",
):
    """Tonel em pe com dois aneis externos."""
    objs.append(_cyl_z(
        f"{name}_body", htm,
        [center_x, center_y, z_bottom + height / 2],
        radius, height, color,
    ))

    ring_h = min(0.035, height * 0.055)
    ring_radius = radius * 1.035
    ring_z = height * 0.18
    for idx, z_rel in enumerate((ring_z, height - ring_z), 1):
        objs.append(_cyl_z(
            f"{name}_ring_{idx}", htm,
            [center_x, center_y, z_bottom + z_rel],
            ring_radius, ring_h, ring_color,
        ))


# =============================================================================
# Caminhonete cabine simples / pickup de trabalho
# =============================================================================


def create_work_pickup(
    htm=None,
    name="work_pickup",
    length=5.25,
    width=1.98,
    wheel_radius=0.40,
    wheel_width=0.26,
    body_color="#2F5D7C",
    accent_color="#20252A",
    glass_color="#8CB7C9",
    rim_color="#BFC4C9",
    include_cargo=True,
    cargo_scale=1.0,
):
    """
    Caminhonete cabine simples inspirada em pickups grandes de trabalho.

    A geometria e dividida em segmentos longitudinais (parachoque, capo,
    cabine, folga e cacamba). A carga usa somente as dimensoes INTERNAS da
    cacamba, evitando atravessar paredes ou tampa traseira.
    """
    _validate_positive(
        length=length,
        width=width,
        wheel_radius=wheel_radius,
        wheel_width=wheel_width,
        cargo_scale=cargo_scale,
    )
    if length < 4.2:
        raise ValueError("'length' is too small for the pickup proportions.")
    if width < 1.55:
        raise ValueError("'width' is too small for the pickup proportions.")
    if not 0.55 <= cargo_scale <= 1.15:
        raise ValueError("'cargo_scale' must be between 0.55 and 1.15.")

    objs = []

    # -------------------------------------------------------------------------
    # Dimensoes principais calculadas
    # -------------------------------------------------------------------------
    bumper_depth = 0.025 * length
    hood_length = 0.25 * length
    cab_length = 0.27 * length
    cab_bed_gap = 0.018 * length
    bed_length = length - 2 * bumper_depth - hood_length - cab_length - cab_bed_gap

    body_width = 0.90 * width
    cab_width = 0.91 * width
    bed_outer_width = 0.94 * width

    chassis_height = 0.16
    wheel_z = wheel_radius
    body_bottom = wheel_radius + 0.055
    lower_body_height = 0.33
    lower_body_top = body_bottom + lower_body_height

    hood_height = 0.30
    cabin_lower_height = 0.34
    glass_height = 0.47
    roof_thickness = 0.075

    # Longitudinal: frente = -y.
    y_front = -length / 2
    hood_start = y_front + bumper_depth
    hood_end = hood_start + hood_length
    cab_start = hood_end
    cab_end = cab_start + cab_length
    bed_start = cab_end + cab_bed_gap
    bed_end = length / 2 - bumper_depth

    hood_y = (hood_start + hood_end) / 2
    cab_y = (cab_start + cab_end) / 2
    bed_y = (bed_start + bed_end) / 2

    front_axle_y = hood_start + 0.50 * hood_length
    rear_axle_y = bed_start + 0.66 * bed_length
    wheel_x = body_width / 2 + wheel_width * 0.43

    # -------------------------------------------------------------------------
    # Rodas e chassi
    # -------------------------------------------------------------------------
    for side, sx in (("left", -1.0), ("right", 1.0)):
        _add_wheel(
            objs, htm, f"{name}_front_{side}_wheel",
            sx * wheel_x, front_axle_y, wheel_z,
            wheel_radius, wheel_width,
            rim_color=rim_color,
        )
        _add_wheel(
            objs, htm, f"{name}_rear_{side}_wheel",
            sx * wheel_x, rear_axle_y, wheel_z,
            wheel_radius, wheel_width,
            rim_color=rim_color,
        )

    chassis_width = body_width * 0.62
    chassis_length = length - 2 * bumper_depth
    chassis_z = body_bottom - chassis_height / 2 - 0.02
    objs.append(_box(
        f"{name}_chassis", htm,
        [0.0, 0.0, chassis_z],
        chassis_width, chassis_length, chassis_height, "#303438",
    ))

    # Eixos: ficam entre as rodas e abaixo da carroceria.
    axle_length = 2 * wheel_x
    axle_radius = 0.045
    objs.append(_cyl_x(
        f"{name}_front_axle", htm,
        [0.0, front_axle_y, wheel_z],
        axle_radius, axle_length, "#3A3D40",
    ))
    objs.append(_cyl_x(
        f"{name}_rear_axle", htm,
        [0.0, rear_axle_y, wheel_z],
        axle_radius, axle_length, "#3A3D40",
    ))

    # -------------------------------------------------------------------------
    # Parte inferior da carroceria
    # -------------------------------------------------------------------------
    body_segment_length = bed_end - hood_start
    body_segment_y = (hood_start + bed_end) / 2
    objs.append(_box(
        f"{name}_lower_body", htm,
        [0.0, body_segment_y, body_bottom + lower_body_height / 2],
        body_width, body_segment_length, lower_body_height, body_color,
    ))

    # Parachoques nao entram na carroceria: encostam nas faces frontal/traseira.
    bumper_h = 0.17
    bumper_w = width * 0.95
    bumper_z = body_bottom + bumper_h / 2
    objs.append(_box(
        f"{name}_front_bumper", htm,
        [0.0, y_front + bumper_depth / 2, bumper_z],
        bumper_w, bumper_depth, bumper_h, accent_color,
    ))
    objs.append(_box(
        f"{name}_rear_bumper", htm,
        [0.0, length / 2 - bumper_depth / 2, bumper_z],
        bumper_w, bumper_depth, bumper_h, accent_color,
    ))

    # Capo / frente alta.
    objs.append(_box(
        f"{name}_hood", htm,
        [0.0, hood_y, lower_body_top + hood_height / 2],
        body_width * 0.96, hood_length, hood_height, body_color,
    ))

    # Grade frontal, farois e pequena entrada de ar no topo do capo.
    front_face_y = hood_start - 0.012
    grille_w = body_width * 0.52
    grille_h = hood_height * 0.56
    grille_z = lower_body_top + hood_height * 0.48
    objs.append(_box(
        f"{name}_grille", htm,
        [0.0, front_face_y, grille_z],
        grille_w, 0.025, grille_h, "#252A2E",
    ))

    light_w = body_width * 0.17
    light_h = hood_height * 0.34
    light_x = body_width / 2 - light_w / 2 - 0.055
    for side, sx in (("left", -1.0), ("right", 1.0)):
        objs.append(_box(
            f"{name}_headlight_{side}", htm,
            [sx * light_x, front_face_y - 0.003, grille_z + 0.02],
            light_w, 0.022, light_h, "#E7F2DA", opacity=0.93,
        ))

    air_intake_w = body_width * 0.38
    objs.append(_box(
        f"{name}_hood_intake", htm,
        [0.0, hood_y - hood_length * 0.08, lower_body_top + hood_height + 0.015],
        air_intake_w, hood_length * 0.16, 0.025, accent_color,
    ))

    # -------------------------------------------------------------------------
    # Cabine simples
    # -------------------------------------------------------------------------
    cabin_lower_z = lower_body_top + cabin_lower_height / 2
    objs.append(_box(
        f"{name}_cab_lower", htm,
        [0.0, cab_y, cabin_lower_z],
        cab_width, cab_length, cabin_lower_height, body_color,
    ))

    window_bottom = lower_body_top + cabin_lower_height
    roof_z = window_bottom + glass_height + roof_thickness / 2
    roof_width = cab_width * 0.975
    roof_length = cab_length * 0.93
    roof_y = cab_y + cab_length * 0.03
    objs.append(_box(
        f"{name}_roof", htm,
        [0.0, roof_y, roof_z],
        roof_width, roof_length, roof_thickness, body_color,
    ))

    # Pilares externos delimitam os vidros laterais, evitando uma grande caixa
    # transparente sobreposta a cabine inteira.
    pillar_w = max(0.065, cab_length * 0.055)
    pillar_depth = pillar_w
    pillar_z = window_bottom + glass_height / 2
    side_shell_x = cab_width / 2
    front_pillar_y = cab_start + pillar_depth / 2
    rear_pillar_y = cab_end - pillar_depth / 2

    for side, sx in (("left", -1.0), ("right", 1.0)):
        # Pilares A e B.
        for p_name, py in (("a", front_pillar_y), ("b", rear_pillar_y)):
            objs.append(_box(
                f"{name}_{side}_{p_name}_pillar", htm,
                [sx * (side_shell_x - 0.035), py, pillar_z],
                0.07, pillar_depth, glass_height, body_color,
            ))

        # Janela lateral exatamente entre os pilares.
        side_window_depth = cab_length - 2 * pillar_depth - 0.025
        side_window_y = (front_pillar_y + rear_pillar_y) / 2
        side_window_h = glass_height * 0.84
        side_window_w = 0.026
        # Leve embutimento no painel lateral: parte do vidro entra alguns milimetros
        # na casca da cabine, reduzindo o vao visual com a estrutura.
        side_window_center = [sx * (side_shell_x - 0.003), side_window_y, pillar_z]
        objs.append(_box(
            f"{name}_{side}_window", htm,
            side_window_center,
            side_window_w, side_window_depth, side_window_h,
            glass_color, opacity=0.62,
        ))

        # Moldura em volta da janela lateral, para ligar melhor vidro, teto e pilares.
        side_frame_w = 0.060
        side_frame_h = 0.052
        side_frame_depth = 0.065
        frame_x = sx * (side_shell_x - 0.015)
        side_frame_z_top = pillar_z + side_window_h / 2 + side_frame_h / 2 - 0.018
        side_frame_z_bottom = pillar_z - side_window_h / 2 - side_frame_h / 2 + 0.014
        side_frame_vertical_h = side_window_h + 0.060
        side_frame_front_y = side_window_y - side_window_depth / 2 - side_frame_depth / 2 + 0.016
        side_frame_rear_y = side_window_y + side_window_depth / 2 + side_frame_depth / 2 - 0.016
        objs.append(_box(
            f"{name}_{side}_window_frame_top", htm,
            [frame_x, side_window_y, side_frame_z_top],
            side_frame_w, side_window_depth + 0.055, side_frame_h, body_color,
        ))
        objs.append(_box(
            f"{name}_{side}_window_frame_bottom", htm,
            [frame_x, side_window_y, side_frame_z_bottom],
            side_frame_w, side_window_depth + 0.040, side_frame_h, body_color,
        ))
        objs.append(_box(
            f"{name}_{side}_window_frame_front", htm,
            [frame_x, side_frame_front_y, pillar_z],
            side_frame_w, side_frame_depth, side_frame_vertical_h, body_color,
        ))
        objs.append(_box(
            f"{name}_{side}_window_frame_rear", htm,
            [frame_x, side_frame_rear_y, pillar_z],
            side_frame_w, side_frame_depth, side_frame_vertical_h, body_color,
        ))

        # Espelho fora da largura da cabine e alinhado com a parte frontal da janela.
        mirror_w = 0.15
        mirror_depth = 0.10
        mirror_h = 0.09
        mirror_x = sx * (side_shell_x + mirror_w / 2 + 0.045)
        mirror_y = cab_start + cab_length * 0.28
        mirror_z = window_bottom + glass_height * 0.48
        objs.append(_box(
            f"{name}_{side}_mirror", htm,
            [mirror_x, mirror_y, mirror_z],
            mirror_w, mirror_depth, mirror_h, accent_color,
        ))
        stalk_len = abs(mirror_x) - side_shell_x - mirror_w / 2
        if stalk_len > 0.01:
            objs.append(_box(
                f"{name}_{side}_mirror_stalk", htm,
                [sx * (side_shell_x + stalk_len / 2), mirror_y, mirror_z],
                stalk_len, 0.025, 0.025, accent_color,
            ))

        # Macaneta na porta, logo abaixo da janela.
        handle_w = 0.16
        objs.append(_box(
            f"{name}_{side}_door_handle", htm,
            [sx * (side_shell_x + 0.012), cab_y + cab_length * 0.17,
             window_bottom - cabin_lower_height * 0.30],
            0.024, handle_w, 0.028, accent_color,
        ))

        # Estribo abaixo da cabine.
        step_w = 0.13
        step_x = sx * (cab_width / 2 + step_w / 2 + 0.025)
        step_z = body_bottom - 0.025
        objs.append(_box(
            f"{name}_{side}_step", htm,
            [step_x, cab_y, step_z],
            step_w, cab_length * 0.78, 0.065, accent_color,
        ))

    # Para-brisa e vidro traseiro. Angulos pequenos sao aplicados em torno de x
    # e os centros ficam dentro do vao definido pelos pilares e teto.
    windshield_w = cab_width * 0.78
    windshield_h = glass_height * 0.82
    windshield_angle = np.deg2rad(-12.0)
    rear_angle = np.deg2rad(7.0)
    glass_center_z = window_bottom + glass_height * 0.50

    pickup_windshield_center = [0.0, cab_start - 0.010, glass_center_z]
    pickup_rear_window_center = [0.0, cab_end + 0.010, glass_center_z]
    pickup_rear_window_w = windshield_w * 0.82
    pickup_rear_window_h = windshield_h * 0.76

    objs.append(_box(
        f"{name}_windshield", htm,
        pickup_windshield_center,
        windshield_w, 0.022, windshield_h, glass_color,
        opacity=0.60, rotation=ub.Utils.rotx(windshield_angle),
    ))
    _add_glass_frame(
        objs, htm, f"{name}_windshield_frame",
        pickup_windshield_center,
        windshield_w, windshield_h, 0.026, body_color,
        rotation=ub.Utils.rotx(windshield_angle),
        side_trim_w=0.050,
        top_trim_h=0.040,
    )
    objs.append(_box(
        f"{name}_rear_window", htm,
        pickup_rear_window_center,
        pickup_rear_window_w, 0.022, pickup_rear_window_h, glass_color,
        opacity=0.60, rotation=ub.Utils.rotx(rear_angle),
    ))
    _add_glass_frame(
        objs, htm, f"{name}_rear_window_frame",
        pickup_rear_window_center,
        pickup_rear_window_w, pickup_rear_window_h, 0.026, body_color,
        rotation=ub.Utils.rotx(rear_angle),
        side_trim_w=0.046,
        top_trim_h=0.038,
    )

    # -------------------------------------------------------------------------
    # Cacamba
    # -------------------------------------------------------------------------
    bed_floor_thickness = 0.09
    bed_floor_bottom = lower_body_top
    bed_floor_top = bed_floor_bottom + bed_floor_thickness
    bed_wall_h = 0.52
    bed_wall_t = 0.075
    tailgate_t = 0.085
    front_wall_t = 0.075

    objs.append(_box(
        f"{name}_bed_floor", htm,
        [0.0, bed_y, bed_floor_bottom + bed_floor_thickness / 2],
        bed_outer_width, bed_length, bed_floor_thickness, body_color,
    ))

    bed_side_x = bed_outer_width / 2 - bed_wall_t / 2
    bed_wall_z = bed_floor_top + bed_wall_h / 2
    for side, sx in (("left", -1.0), ("right", 1.0)):
        objs.append(_box(
            f"{name}_bed_{side}_wall", htm,
            [sx * bed_side_x, bed_y, bed_wall_z],
            bed_wall_t, bed_length, bed_wall_h, body_color,
        ))

    front_wall_y = bed_start + front_wall_t / 2
    tailgate_y = bed_end - tailgate_t / 2
    objs.append(_box(
        f"{name}_bed_front_wall", htm,
        [0.0, front_wall_y, bed_wall_z],
        bed_outer_width - 2 * bed_wall_t, front_wall_t, bed_wall_h, body_color,
    ))
    objs.append(_box(
        f"{name}_tailgate", htm,
        [0.0, tailgate_y, bed_wall_z],
        bed_outer_width - 2 * bed_wall_t, tailgate_t, bed_wall_h, body_color,
    ))

    # Taillights ficam na face externa da tampa traseira.
    tail_light_w = 0.18
    tail_light_h = 0.28
    tail_light_x = bed_outer_width / 2 - tail_light_w / 2 - 0.035
    tail_light_z = bed_floor_top + bed_wall_h * 0.53
    rear_face_y = bed_end + 0.010
    for side, sx in (("left", -1.0), ("right", 1.0)):
        objs.append(_box(
            f"{name}_taillight_{side}", htm,
            [sx * tail_light_x, rear_face_y, tail_light_z],
            tail_light_w, 0.022, tail_light_h, "#B22A2A", opacity=0.94,
        ))

    # -------------------------------------------------------------------------
    # Carga: calculada exclusivamente pelo vao interno da cacamba
    # -------------------------------------------------------------------------
    if include_cargo:
        clearance = 0.045
        inner_x_min = -bed_outer_width / 2 + bed_wall_t + clearance
        inner_x_max = bed_outer_width / 2 - bed_wall_t - clearance
        inner_y_min = bed_start + front_wall_t + clearance
        inner_y_max = bed_end - tailgate_t - clearance
        inner_width = inner_x_max - inner_x_min
        inner_length = inner_y_max - inner_y_min

        pallet_gap = max(0.055, inner_length * 0.035)
        pallet_depth = (inner_length - pallet_gap) / 2
        pallet_width = inner_width * 0.96
        pallet_h = 0.105
        cargo_floor_z = bed_floor_top

        pallet1_y = inner_y_min + pallet_depth / 2
        pallet2_y = inner_y_max - pallet_depth / 2

        _add_pallet(
            objs, htm, f"{name}_cargo_pallet_front",
            0.0, pallet1_y, cargo_floor_z,
            pallet_width, pallet_depth, pallet_h,
        )
        _add_pallet(
            objs, htm, f"{name}_cargo_pallet_rear",
            0.0, pallet2_y, cargo_floor_z,
            pallet_width, pallet_depth, pallet_h,
            wood_color="#94653C",
        )

        pallet_top_z = cargo_floor_z + pallet_h

        # Pallet dianteiro: duas caixas grandes lado a lado e uma caixa superior.
        box_gap = max(0.045, inner_width * 0.03)
        bottom_box_w = (pallet_width - box_gap) / 2
        bottom_box_d = pallet_depth * 0.82
        bottom_box_h = min(0.58 * cargo_scale, 0.72)
        bottom_box_x = bottom_box_w / 2 + box_gap / 2

        _add_crate(
            objs, htm, f"{name}_cargo_box_front_left",
            -bottom_box_x, pallet1_y, pallet_top_z,
            bottom_box_w, bottom_box_d, bottom_box_h,
            color="#B88A55",
        )
        _add_crate(
            objs, htm, f"{name}_cargo_box_front_right",
            bottom_box_x, pallet1_y, pallet_top_z,
            bottom_box_w, bottom_box_d, bottom_box_h,
            color="#A97847",
        )

        top_box_w = pallet_width * 0.60
        top_box_d = pallet_depth * 0.63
        top_box_h = min(0.42 * cargo_scale, 0.52)
        _add_crate(
            objs, htm, f"{name}_cargo_box_front_top",
            0.0, pallet1_y, pallet_top_z + bottom_box_h,
            top_box_w, top_box_d, top_box_h,
            color="#C49A68",
        )

        # Pallet traseiro: caixa larga + tonel lado a lado.
        barrel_radius = min(0.245 * cargo_scale, inner_width * 0.16)
        cargo_gap = max(0.055, inner_width * 0.035)
        rear_crate_w = pallet_width - 2 * barrel_radius - cargo_gap
        rear_crate_w = max(rear_crate_w, pallet_width * 0.50)
        rear_crate_d = pallet_depth * 0.80
        rear_crate_h = min(0.68 * cargo_scale, 0.80)

        total_pair_w = rear_crate_w + cargo_gap + 2 * barrel_radius
        left_edge = -total_pair_w / 2
        crate_x = left_edge + rear_crate_w / 2
        barrel_x = left_edge + rear_crate_w + cargo_gap + barrel_radius

        _add_crate(
            objs, htm, f"{name}_cargo_rear_crate",
            crate_x, pallet2_y, pallet_top_z,
            rear_crate_w, rear_crate_d, rear_crate_h,
            color="#9E7348",
        )

        barrel_h = min(0.72 * cargo_scale, 0.82)
        _add_barrel(
            objs, htm, f"{name}_cargo_barrel",
            barrel_x, pallet2_y, pallet_top_z,
            barrel_radius, barrel_h,
            color="#2F6F83",
        )

    return objs


# =============================================================================
# Crossover compacto / carro de uso diario
# =============================================================================


def create_compact_crossover(
    htm=None,
    name="compact_crossover",
    length=4.28,
    width=1.82,
    wheel_radius=0.34,
    wheel_width=0.22,
    body_color="#8F3D36",
    accent_color="#1E2327",
    glass_color="#8DB4C4",
    rim_color="#C0C5C9",
    include_roof_rails=True,
):
    """
    Crossover compacto inspirado em carros urbanos modernos.

    O perfil e formado por carroceria baixa, capo, cabine longa, teto, pilares,
    vidros, rodas, farois, lanternas, retrovisores e detalhes externos.
    """
    _validate_positive(
        length=length,
        width=width,
        wheel_radius=wheel_radius,
        wheel_width=wheel_width,
    )
    if length < 3.5:
        raise ValueError("'length' is too small for the crossover proportions.")
    if width < 1.45:
        raise ValueError("'width' is too small for the crossover proportions.")

    objs = []

    bumper_depth = 0.028 * length
    hood_length = 0.225 * length
    cabin_length = 0.535 * length
    rear_length = length - 2 * bumper_depth - hood_length - cabin_length

    body_width = 0.92 * width
    cabin_width = 0.88 * width

    wheel_z = wheel_radius
    body_bottom = wheel_radius + 0.035
    lower_body_h = 0.34
    lower_body_top = body_bottom + lower_body_h
    hood_h = 0.24

    y_front = -length / 2
    hood_start = y_front + bumper_depth
    hood_end = hood_start + hood_length
    cabin_start = hood_end
    cabin_end = cabin_start + cabin_length
    rear_start = cabin_end
    rear_end = length / 2 - bumper_depth

    front_axle_y = -0.285 * length
    rear_axle_y = 0.285 * length
    wheel_x = body_width / 2 + wheel_width * 0.39

    # -------------------------------------------------------------------------
    # Rodas / chassi
    # -------------------------------------------------------------------------
    for side, sx in (("left", -1.0), ("right", 1.0)):
        _add_wheel(
            objs, htm, f"{name}_front_{side}_wheel",
            sx * wheel_x, front_axle_y, wheel_z,
            wheel_radius, wheel_width,
            rim_color=rim_color,
        )
        _add_wheel(
            objs, htm, f"{name}_rear_{side}_wheel",
            sx * wheel_x, rear_axle_y, wheel_z,
            wheel_radius, wheel_width,
            rim_color=rim_color,
        )

    chassis_h = 0.12
    chassis_z = body_bottom - chassis_h / 2 - 0.018
    objs.append(_box(
        f"{name}_chassis", htm,
        [0.0, 0.0, chassis_z],
        body_width * 0.66, length - 2 * bumper_depth, chassis_h, "#2D3135",
    ))

    # -------------------------------------------------------------------------
    # Carroceria inferior e parachoques
    # -------------------------------------------------------------------------
    body_length = rear_end - hood_start
    body_y = (hood_start + rear_end) / 2
    objs.append(_box(
        f"{name}_lower_body", htm,
        [0.0, body_y, body_bottom + lower_body_h / 2],
        body_width, body_length, lower_body_h, body_color,
    ))

    bumper_h = 0.14
    bumper_z = body_bottom + bumper_h / 2
    objs.append(_box(
        f"{name}_front_bumper", htm,
        [0.0, y_front + bumper_depth / 2, bumper_z],
        width * 0.94, bumper_depth, bumper_h, accent_color,
    ))
    objs.append(_box(
        f"{name}_rear_bumper", htm,
        [0.0, length / 2 - bumper_depth / 2, bumper_z],
        width * 0.94, bumper_depth, bumper_h, accent_color,
    ))

    # Capo e parte traseira superior baixa.
    hood_y = (hood_start + hood_end) / 2
    objs.append(_box(
        f"{name}_hood", htm,
        [0.0, hood_y, lower_body_top + hood_h / 2],
        body_width * 0.94, hood_length, hood_h, body_color,
    ))

    rear_upper_h = hood_h * 0.92
    rear_y = (rear_start + rear_end) / 2
    objs.append(_box(
        f"{name}_rear_upper", htm,
        [0.0, rear_y, lower_body_top + rear_upper_h / 2],
        body_width * 0.93, rear_length, rear_upper_h, body_color,
    ))

    # Grade e farois frontais.
    front_face_y = hood_start - 0.010
    grille_w = body_width * 0.46
    grille_h = hood_h * 0.46
    grille_z = lower_body_top + hood_h * 0.42
    objs.append(_box(
        f"{name}_grille", htm,
        [0.0, front_face_y, grille_z],
        grille_w, 0.020, grille_h, "#25292C",
    ))

    headlight_w = body_width * 0.19
    headlight_h = 0.10
    headlight_x = body_width / 2 - headlight_w / 2 - 0.035
    for side, sx in (("left", -1.0), ("right", 1.0)):
        objs.append(_box(
            f"{name}_headlight_{side}", htm,
            [sx * headlight_x, front_face_y - 0.003, lower_body_top + hood_h * 0.62],
            headlight_w, 0.020, headlight_h, "#EAF0DD", opacity=0.95,
        ))

    # -------------------------------------------------------------------------
    # Cabine / greenhouse
    # -------------------------------------------------------------------------
    cabin_lower_h = 0.25
    cabin_lower_z = lower_body_top + cabin_lower_h / 2
    cabin_y = (cabin_start + cabin_end) / 2
    objs.append(_box(
        f"{name}_cabin_lower", htm,
        [0.0, cabin_y, cabin_lower_z],
        cabin_width, cabin_length, cabin_lower_h, body_color,
    ))

    window_bottom = lower_body_top + cabin_lower_h
    glass_h = 0.47
    roof_t = 0.065
    roof_z = window_bottom + glass_h + roof_t / 2
    roof_length = cabin_length * 0.93
    roof_y = cabin_y + cabin_length * 0.030
    objs.append(_box(
        f"{name}_roof", htm,
        [0.0, roof_y, roof_z],
        cabin_width * 0.955, roof_length, roof_t, body_color,
    ))

    # Divisao das janelas laterais em dianteira e traseira com pilar B real.
    edge_pillar = max(0.060, cabin_length * 0.034)
    b_pillar = max(0.075, cabin_length * 0.040)
    usable_side_len = cabin_length - 2 * edge_pillar - b_pillar
    front_window_len = usable_side_len * 0.48
    rear_window_len = usable_side_len - front_window_len

    front_window_y = cabin_start + edge_pillar + front_window_len / 2
    b_pillar_y = cabin_start + edge_pillar + front_window_len + b_pillar / 2
    rear_window_y = b_pillar_y + b_pillar / 2 + rear_window_len / 2
    glass_z = window_bottom + glass_h * 0.50
    side_x = cabin_width / 2

    for side, sx in (("left", -1.0), ("right", 1.0)):
        # Pilares A, B e C.
        a_y = cabin_start + edge_pillar / 2
        c_y = cabin_end - edge_pillar / 2
        for p_name, py, p_depth in (
            ("a", a_y, edge_pillar),
            ("b", b_pillar_y, b_pillar),
            ("c", c_y, edge_pillar),
        ):
            objs.append(_box(
                f"{name}_{side}_{p_name}_pillar", htm,
                [sx * (side_x - 0.030), py, glass_z],
                0.06, p_depth, glass_h, body_color,
            ))

        front_window_h = glass_h * 0.84
        rear_window_h = glass_h * 0.82
        side_window_w = 0.026
        # O centro vai alguns milimetros para dentro da lateral para reduzir o vao.
        front_window_center_x = sx * (side_x - 0.004)
        rear_window_center_x = sx * (side_x - 0.004)
        objs.append(_box(
            f"{name}_{side}_front_window", htm,
            [front_window_center_x, front_window_y, glass_z],
            side_window_w, front_window_len, front_window_h, glass_color, opacity=0.60,
        ))
        objs.append(_box(
            f"{name}_{side}_rear_window", htm,
            [rear_window_center_x, rear_window_y, glass_z],
            side_window_w, rear_window_len, rear_window_h, glass_color, opacity=0.60,
        ))

        # Moldura lateral superior, um pouco maior, agora entrando levemente sob o teto.
        upper_frame_h = 0.074
        upper_frame_w = 0.118
        upper_frame_len = cabin_length - 2 * edge_pillar + 0.050
        upper_frame_z = window_bottom + glass_h - upper_frame_h / 2 - 0.012
        objs.append(_box(
            f"{name}_{side}_upper_door_frame", htm,
            [sx * (side_x - 0.024), cabin_y, upper_frame_z],
            upper_frame_w, upper_frame_len, upper_frame_h, body_color,
        ))

        # Barra inferior curta para ajudar a "abraçar" os vidros pela lateral.
        lower_frame_h = 0.042
        lower_frame_w = 0.094
        lower_frame_z = window_bottom - lower_frame_h / 2 + 0.012
        objs.append(_box(
            f"{name}_{side}_lower_window_frame", htm,
            [sx * (side_x - 0.012), cabin_y, lower_frame_z],
            lower_frame_w, cabin_length - 2 * edge_pillar - 0.030, lower_frame_h, body_color,
        ))

        # Pequenos detalhes verticais nas portas ajudam a marcar a estrutura
        # da porta e fazem a transicao entre carroceria, vidro e moldura.
        hinge_w = 0.035
        hinge_d = 0.075
        hinge_h = 0.085
        hinge_x = sx * (side_x + 0.012)
        hinge_z = window_bottom - hinge_h * 0.42
        for idx, py in enumerate((
            cabin_start + edge_pillar * 1.25,
            b_pillar_y - b_pillar * 0.58,
        ), 1):
            objs.append(_box(
                f"{name}_{side}_door_hinge_{idx}", htm,
                [hinge_x, py, hinge_z],
                hinge_w, hinge_d, hinge_h, accent_color,
            ))

        # Retrovisores mais proximos da posicao original, na transicao capo/cabine.
        mirror_w = 0.13
        mirror_depth = 0.09
        mirror_h = 0.075
        mirror_x = sx * (side_x + mirror_w / 2 + 0.035)
        mirror_y = cabin_start + cabin_length * 0.14
        mirror_z = window_bottom + glass_h * 0.38
        objs.append(_box(
            f"{name}_{side}_mirror", htm,
            [mirror_x, mirror_y, mirror_z],
            mirror_w, mirror_depth, mirror_h, accent_color,
        ))
        stalk_len = 0.040
        objs.append(_box(
            f"{name}_{side}_mirror_stalk", htm,
            [sx * (side_x + stalk_len / 2 + 0.006), mirror_y, mirror_z - 0.005],
            stalk_len, 0.028, 0.028, accent_color,
        ))

        # Duas macanetas, posicionadas sob o centro de cada janela.
        handle_z = window_bottom - cabin_lower_h * 0.28
        for door_name, py in (("front", front_window_y), ("rear", rear_window_y)):
            objs.append(_box(
                f"{name}_{side}_{door_name}_handle", htm,
                [sx * (side_x + 0.012), py, handle_z],
                0.022, 0.13, 0.024, accent_color,
            ))

        # Saia lateral baixa.
        skirt_w = 0.07
        objs.append(_box(
            f"{name}_{side}_skirt", htm,
            [sx * (body_width / 2 + skirt_w / 2), body_y, body_bottom + 0.055],
            skirt_w, body_length * 0.62, 0.095, accent_color,
        ))

    # Para-brisa e vidro traseiro levemente inclinados.
    windshield_w = cabin_width * 0.78
    windshield_h = glass_h * 0.84
    crossover_windshield_center = [0.0, cabin_start - 0.010, glass_z]
    crossover_rear_glass_center = [0.0, cabin_end + 0.010, glass_z]
    crossover_rear_glass_w = windshield_w * 0.88
    crossover_rear_glass_h = windshield_h * 0.78
    windshield_rotation = ub.Utils.rotx(np.deg2rad(-18.0))
    rear_glass_rotation = ub.Utils.rotx(np.deg2rad(16.0))

    objs.append(_box(
        f"{name}_windshield", htm,
        crossover_windshield_center,
        windshield_w, 0.020, windshield_h, glass_color,
        opacity=0.60, rotation=windshield_rotation,
    ))
    _add_glass_frame(
        objs, htm, f"{name}_windshield_frame",
        crossover_windshield_center,
        windshield_w, windshield_h, 0.024, body_color,
        rotation=windshield_rotation,
        side_trim_w=0.046,
        top_trim_h=0.036,
    )
    objs.append(_box(
        f"{name}_rear_glass", htm,
        crossover_rear_glass_center,
        crossover_rear_glass_w, 0.020, crossover_rear_glass_h, glass_color,
        opacity=0.60, rotation=rear_glass_rotation,
    ))
    _add_glass_frame(
        objs, htm, f"{name}_rear_glass_frame",
        crossover_rear_glass_center,
        crossover_rear_glass_w, crossover_rear_glass_h, 0.024, body_color,
        rotation=rear_glass_rotation,
        side_trim_w=0.042,
        top_trim_h=0.034,
    )

    # Lanternas traseiras.
    tail_w = body_width * 0.16
    tail_x = body_width / 2 - tail_w / 2 - 0.035
    tail_z = lower_body_top + rear_upper_h * 0.54
    rear_face_y = rear_end + 0.010
    for side, sx in (("left", -1.0), ("right", 1.0)):
        objs.append(_box(
            f"{name}_taillight_{side}", htm,
            [sx * tail_x, rear_face_y, tail_z],
            tail_w, 0.020, 0.16, "#B5282C", opacity=0.95,
        ))

    # Aerofolio curto, discreto.
    spoiler_w = cabin_width * 0.86
    spoiler_d = 0.12
    spoiler_z = roof_z + roof_t / 2 + 0.035
    objs.append(_box(
        f"{name}_spoiler", htm,
        [0.0, cabin_end - spoiler_d / 2, spoiler_z],
        spoiler_w, spoiler_d, 0.045, accent_color,
    ))

    # Rack de teto opcional: barras no sentido longitudinal e duas travessas.
    if include_roof_rails:
        rail_w = 0.035
        rail_h = 0.04
        rail_len = roof_length * 0.80
        rail_x = cabin_width * 0.34
        rail_z = roof_z + roof_t / 2 + rail_h / 2 + 0.012

        for side, sx in (("left", -1.0), ("right", 1.0)):
            objs.append(_box(
                f"{name}_roof_rail_{side}", htm,
                [sx * rail_x, roof_y, rail_z],
                rail_w, rail_len, rail_h, accent_color,
            ))

        crossbar_w = 2 * rail_x + rail_w
        crossbar_d = 0.035
        for idx, frac in enumerate((-0.25, 0.25), 1):
            objs.append(_box(
                f"{name}_roof_crossbar_{idx}", htm,
                [0.0, roof_y + frac * rail_len, rail_z + 0.012],
                crossbar_w, crossbar_d, 0.028, accent_color,
            ))

    return objs


# =============================================================================
# Catalogo
# =============================================================================


VEHICLE_CATALOG = {
    "work_pickup": create_work_pickup,
    "compact_crossover": create_compact_crossover,
}


def available_vehicles():
    """Retorna os nomes dos veiculos disponiveis."""
    return sorted(VEHICLE_CATALOG.keys())


def create_vehicle(vehicle_type, **kwargs):
    """Cria um veiculo pelo nome do catalogo."""
    if vehicle_type not in VEHICLE_CATALOG:
        raise ValueError(
            f"Unknown vehicle '{vehicle_type}'. "
            f"Available vehicles: {', '.join(available_vehicles())}"
        )
    return VEHICLE_CATALOG[vehicle_type](**kwargs)


__all__ = [
    "create_work_pickup",
    "create_compact_crossover",
    "VEHICLE_CATALOG",
    "available_vehicles",
    "create_vehicle",
]
