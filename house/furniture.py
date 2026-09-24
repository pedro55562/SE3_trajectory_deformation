"""
furniture.py
============

Biblioteca simples de mobiliario/cenario construida apenas com primitivas do UAIbot.

Convencao adotada para TODOS os objetos:
    x -> largura
    y -> profundidade
    z -> altura

A origem local fica, sempre que fizer sentido, no centro da projecao do objeto
sobre o chao (z = 0).

Cada funcao:
    - recebe uma HTM global opcional;
    - permite customizar as dimensoes principais;
    - retorna uma list com objetos do UAIbot;
    - nao usa Group, pois os objetos sao pensados como estaticos.

Exemplo:
    desk = create_office_desk(
        htm=ub.Utils.trn([1.0, 2.0, 0.0]) @ ub.Utils.rotz(np.pi/4),
        width=1.4,
        depth=0.7,
    )

    sim = ub.Simulation.create_sim_grid(desk)
    sim.run()
"""

import numpy as np
import uaibot as ub


# =============================================================================
# Helpers internos
# =============================================================================

def _base_htm(htm):
    """Retorna identidade se htm=None."""
    return np.eye(4) if htm is None else htm


def _pose(htm, xyz, rotation=None):
    """Aplica uma pose local sobre a pose global do objeto composto."""
    T = _base_htm(htm) @ ub.Utils.trn(xyz)
    if rotation is not None:
        T = T @ rotation
    return T


def _box(name, htm, xyz, width, depth, height, color, opacity=1.0):
    return ub.Box(
        name=name,
        htm=_pose(htm, xyz),
        width=width,
        depth=depth,
        height=height,
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


def _cyl_x(name, htm, xyz, radius, length, color, opacity=1.0):
    return ub.Cylinder(
        name=name,
        htm=_pose(htm, xyz, ub.Utils.roty(np.pi / 2)),
        radius=radius,
        height=length,
        color=color,
        opacity=opacity,
    )


def _cyl_y(name, htm, xyz, radius, length, color, opacity=1.0):
    return ub.Cylinder(
        name=name,
        htm=_pose(htm, xyz, ub.Utils.rotx(np.pi / 2)),
        radius=radius,
        height=length,
        color=color,
        opacity=opacity,
    )


def _ball(name, htm, xyz, radius, color, opacity=1.0):
    return ub.Ball(
        name=name,
        htm=_pose(htm, xyz),
        radius=radius,
        color=color,
        opacity=opacity,
    )


# =============================================================================
# Mesas
# =============================================================================

def create_office_desk(
    htm=None,
    name="office_desk",
    width=1.40,
    depth=0.70,
    height=0.75,
    top_thickness=0.04,
    leg_radius=0.018,
    bar_radius=0.015,
    leg_inset_x=0.08,
    leg_inset_y=0.08,
    side_bar_height=0.10,
    back_bar_height=0.35,
    top_color="#8B5A2B",
    frame_color="#303030",
):
    """Mesa de escritorio com tampo, 4 hastes, 2 pes laterais e barra traseira."""
    objs = []

    x = width / 2 - leg_inset_x
    y = depth / 2 - leg_inset_y
    leg_h = height - top_thickness

    objs.append(_box(
        f"{name}_top", htm,
        [0, 0, height - top_thickness / 2],
        width, depth, top_thickness, top_color
    ))

    positions = [
        (-x, -y), (-x, y),
        (x, -y), (x, y),
    ]

    for i, (px, py) in enumerate(positions, 1):
        objs.append(_cyl_z(
            f"{name}_leg_{i}", htm,
            [px, py, leg_h / 2],
            leg_radius, leg_h, frame_color
        ))

    objs.append(_cyl_y(
        f"{name}_side_bar_left", htm,
        [-x, 0, side_bar_height],
        bar_radius, 2 * y, frame_color
    ))

    objs.append(_cyl_y(
        f"{name}_side_bar_right", htm,
        [x, 0, side_bar_height],
        bar_radius, 2 * y, frame_color
    ))

    objs.append(_cyl_x(
        f"{name}_back_bar", htm,
        [0, y, back_bar_height],
        bar_radius, 2 * x, frame_color
    ))

    return objs


def create_dining_table(
    htm=None,
    name="dining_table",
    width=1.80,
    depth=0.90,
    height=0.76,
    top_thickness=0.055,
    leg_width=0.075,
    leg_depth=0.075,
    leg_inset_x=0.10,
    leg_inset_y=0.10,
    top_color="#7A4B2A",
    leg_color="#4A2E1A",
):
    """Mesa de jantar retangular classica com quatro pes quadrados."""
    objs = []

    objs.append(_box(
        f"{name}_top", htm,
        [0, 0, height - top_thickness / 2],
        width, depth, top_thickness, top_color
    ))

    leg_h = height - top_thickness
    x = width / 2 - leg_inset_x - leg_width / 2
    y = depth / 2 - leg_inset_y - leg_depth / 2

    for i, (px, py) in enumerate([
        (-x, -y), (-x, y), (x, -y), (x, y)
    ], 1):
        objs.append(_box(
            f"{name}_leg_{i}", htm,
            [px, py, leg_h / 2],
            leg_width, leg_depth, leg_h, leg_color
        ))

    return objs


def create_coffee_table(
    htm=None,
    name="coffee_table",
    width=1.10,
    depth=0.60,
    height=0.42,
    top_thickness=0.045,
    leg_width=0.055,
    inset=0.07,
    top_color="#8B5A2B",
    leg_color="#3A2A20",
):
    """Mesa baixa para sala."""
    return create_dining_table(
        htm=htm,
        name=name,
        width=width,
        depth=depth,
        height=height,
        top_thickness=top_thickness,
        leg_width=leg_width,
        leg_depth=leg_width,
        leg_inset_x=inset,
        leg_inset_y=inset,
        top_color=top_color,
        leg_color=leg_color,
    )


def create_side_table(
    htm=None,
    name="side_table",
    width=0.48,
    depth=0.48,
    height=0.52,
    top_thickness=0.035,
    leg_radius=0.018,
    inset=0.055,
    top_color="#8A6040",
    leg_color="#303030",
):
    """Mesa lateral pequena com quatro pes cilindricos."""
    objs = []

    objs.append(_box(
        f"{name}_top", htm,
        [0, 0, height - top_thickness / 2],
        width, depth, top_thickness, top_color
    ))

    leg_h = height - top_thickness
    x = width / 2 - inset
    y = depth / 2 - inset

    for i, (px, py) in enumerate([
        (-x, -y), (-x, y), (x, -y), (x, y)
    ], 1):
        objs.append(_cyl_z(
            f"{name}_leg_{i}", htm,
            [px, py, leg_h / 2],
            leg_radius, leg_h, leg_color
        ))

    return objs


def create_round_table(
    htm=None,
    name="round_table",
    radius=0.55,
    height=0.75,
    top_thickness=0.045,
    pedestal_radius=0.065,
    base_radius=0.32,
    base_thickness=0.035,
    top_color="#8B5A2B",
    frame_color="#303030",
):
    """Mesa redonda com pedestal central."""
    objs = []

    objs.append(_cyl_z(
        f"{name}_top", htm,
        [0, 0, height - top_thickness / 2],
        radius, top_thickness, top_color
    ))

    objs.append(_cyl_z(
        f"{name}_pedestal", htm,
        [0, 0, (height - top_thickness) / 2],
        pedestal_radius, height - top_thickness, frame_color
    ))

    objs.append(_cyl_z(
        f"{name}_base", htm,
        [0, 0, base_thickness / 2],
        base_radius, base_thickness, frame_color
    ))

    return objs


# =============================================================================
# Assentos
# =============================================================================

def create_chair(
    htm=None,
    name="chair",
    width=0.46,
    depth=0.48,
    seat_height=0.46,
    total_height=0.90,
    seat_thickness=0.045,
    leg_width=0.04,
    back_thickness=0.04,
    back_margin=0.035,
    frame_color="#543A28",
    seat_color="#7A5438",
):
    """Cadeira simples de jantar."""
    objs = []

    objs.append(_box(
        f"{name}_seat", htm,
        [0, 0, seat_height],
        width, depth, seat_thickness, seat_color
    ))

    leg_h = seat_height - seat_thickness / 2
    x = width / 2 - leg_width / 2 - 0.025
    y = depth / 2 - leg_width / 2 - 0.025

    for i, (px, py) in enumerate([
        (-x, -y), (-x, y), (x, -y), (x, y)
    ], 1):
        objs.append(_box(
            f"{name}_leg_{i}", htm,
            [px, py, leg_h / 2],
            leg_width, leg_width, leg_h, frame_color
        ))

    back_h = total_height - seat_height
    objs.append(_box(
        f"{name}_back", htm,
        [0, depth / 2 - back_thickness / 2 - back_margin,
         seat_height + back_h / 2],
        width, back_thickness, back_h, seat_color
    ))

    return objs


def create_office_chair(
    htm=None,
    name="office_chair",
    seat_width=0.50,
    seat_depth=0.48,
    seat_height=0.50,
    seat_thickness=0.08,
    back_height=0.58,
    back_thickness=0.07,
    pedestal_radius=0.035,
    base_radius=0.30,
    caster_radius=0.025,
    seat_color="#252525",
    frame_color="#404040",
):
    """Cadeira de escritorio simplificada, com pedestal e base radial."""
    objs = []

    objs.append(_box(
        f"{name}_seat", htm,
        [0, 0, seat_height],
        seat_width, seat_depth, seat_thickness, seat_color
    ))

    objs.append(_box(
        f"{name}_back", htm,
        [0, seat_depth / 2 - back_thickness / 2,
         seat_height + back_height / 2],
        seat_width * 0.90, back_thickness, back_height, seat_color
    ))

    stem_h = seat_height - seat_thickness / 2 - caster_radius
    objs.append(_cyl_z(
        f"{name}_stem", htm,
        [0, 0, caster_radius + stem_h / 2],
        pedestal_radius, stem_h, frame_color
    ))

    # Base radial: cada braco fica orientado radialmente
    # e a rodinha fica levemente "encaixada" na ponta do braco.
    arm_length = base_radius * 0.95
    arm_radius = pedestal_radius * 0.55
    arm_z = caster_radius * 1.7
    caster_offset = arm_length - caster_radius * 0.35

    for i, ang in enumerate(np.linspace(0, 2 * np.pi, 5, endpoint=False), 1):
        cx = 0.5 * arm_length * np.cos(ang)
        cy = 0.5 * arm_length * np.sin(ang)

        arm_htm = (
            _base_htm(htm)
            @ ub.Utils.trn([cx, cy, arm_z])
            @ ub.Utils.rotz(ang)
            @ ub.Utils.roty(np.pi / 2)
        )

        objs.append(ub.Cylinder(
            name=f"{name}_base_arm_{i}",
            htm=arm_htm,
            radius=arm_radius,
            height=arm_length,
            color=frame_color
        ))

        objs.append(_ball(
            f"{name}_caster_{i}", htm,
            [caster_offset * np.cos(ang),
             caster_offset * np.sin(ang),
             caster_radius],
            caster_radius, frame_color
        ))

    return objs


def create_stool(
    htm=None,
    name="stool",
    seat_radius=0.20,
    height=0.48,
    seat_thickness=0.05,
    leg_radius=0.017,
    leg_spread=0.13,
    seat_color="#7A4B2A",
    frame_color="#303030",
):
    """Banqueta redonda de quatro pes."""
    objs = []

    objs.append(_cyl_z(
        f"{name}_seat", htm,
        [0, 0, height - seat_thickness / 2],
        seat_radius, seat_thickness, seat_color
    ))

    leg_h = height - seat_thickness

    for i, (x, y) in enumerate([
        (-leg_spread, -leg_spread),
        (-leg_spread, leg_spread),
        (leg_spread, -leg_spread),
        (leg_spread, leg_spread),
    ], 1):
        objs.append(_cyl_z(
            f"{name}_leg_{i}", htm,
            [x, y, leg_h / 2],
            leg_radius, leg_h, frame_color
        ))

    return objs


def create_bench(
    htm=None,
    name="bench",
    width=1.30,
    depth=0.42,
    height=0.45,
    seat_thickness=0.06,
    leg_width=0.07,
    leg_inset=0.16,
    seat_color="#835A36",
    leg_color="#3C3028",
):
    """Banco simples comprido."""
    objs = []

    objs.append(_box(
        f"{name}_seat", htm,
        [0, 0, height - seat_thickness / 2],
        width, depth, seat_thickness, seat_color
    ))

    leg_h = height - seat_thickness
    x = width / 2 - leg_inset

    for i, px in enumerate([-x, x], 1):
        objs.append(_box(
            f"{name}_leg_{i}", htm,
            [px, 0, leg_h / 2],
            leg_width, depth * 0.80, leg_h, leg_color
        ))

    return objs


# =============================================================================
# Quarto
# =============================================================================

def create_bed(
    htm=None,
    name="bed",
    width=1.38,
    length=1.88,
    frame_height=0.28,
    frame_thickness=0.08,
    mattress_height=0.20,
    mattress_inset=0.035,
    headboard_height=1.05,
    headboard_thickness=0.08,
    pillow_width=0.50,
    pillow_depth=0.28,
    pillow_height=0.10,
    frame_color="#60452F",
    mattress_color="#E8E2D8",
    pillow_color="#F3EFE8",
):
    """
    Cama com estrutura, colchao, cabeceira e dois travesseiros.

    +y e o lado da cabeceira.
    """
    objs = []

    # Base/estrutura
    objs.append(_box(
        f"{name}_frame", htm,
        [0, 0, frame_height / 2],
        width, length, frame_height, frame_color
    ))

    mattress_z = frame_height + mattress_height / 2
    objs.append(_box(
        f"{name}_mattress", htm,
        [0, 0, mattress_z],
        width - 2 * mattress_inset,
        length - 2 * mattress_inset,
        mattress_height,
        mattress_color
    ))

    objs.append(_box(
        f"{name}_headboard", htm,
        [0, length / 2 - headboard_thickness / 2,
         headboard_height / 2],
        width, headboard_thickness, headboard_height, frame_color
    ))

    pillow_y = length / 2 - 0.25
    pillow_z = frame_height + mattress_height + pillow_height / 2

    for i, px in enumerate([-width * 0.22, width * 0.22], 1):
        objs.append(_box(
            f"{name}_pillow_{i}", htm,
            [px, pillow_y, pillow_z],
            pillow_width, pillow_depth, pillow_height, pillow_color
        ))

    return objs


def create_single_bed(htm=None, name="single_bed", **kwargs):
    """Atalho para cama de solteiro."""
    params = dict(width=0.96, length=1.88)
    params.update(kwargs)
    return create_bed(htm=htm, name=name, **params)


def create_queen_bed(htm=None, name="queen_bed", **kwargs):
    """Atalho para cama queen."""
    params = dict(width=1.58, length=1.98)
    params.update(kwargs)
    return create_bed(htm=htm, name=name, **params)


def create_nightstand(
    htm=None,
    name="nightstand",
    width=0.48,
    depth=0.40,
    height=0.55,
    body_height=0.42,
    leg_height=0.13,
    leg_width=0.035,
    drawer_gap=0.015,
    body_color="#7A5235",
    drawer_color="#8A6040",
    handle_color="#303030",
):
    """Criado-mudo com duas gavetas e quatro pes."""
    objs = []

    body_z = leg_height + body_height / 2
    objs.append(_box(
        f"{name}_body", htm,
        [0, 0, body_z],
        width, depth, body_height, body_color
    ))

    drawer_h = (body_height - 3 * drawer_gap) / 2

    for i in range(2):
        z = leg_height + drawer_gap + drawer_h / 2 + i * (drawer_h + drawer_gap)
        objs.append(_box(
            f"{name}_drawer_{i+1}", htm,
            [0, -depth / 2 - 0.005, z],
            width * 0.90, 0.018, drawer_h * 0.88, drawer_color
        ))
        objs.append(_cyl_y(
            f"{name}_handle_{i+1}", htm,
            [0, -depth / 2 - 0.03, z],
            0.008, 0.05, handle_color
        ))

    x = width / 2 - leg_width
    y = depth / 2 - leg_width

    for i, (px, py) in enumerate([
        (-x, -y), (-x, y), (x, -y), (x, y)
    ], 1):
        objs.append(_box(
            f"{name}_leg_{i}", htm,
            [px, py, leg_height / 2],
            leg_width, leg_width, leg_height, body_color
        ))

    return objs


def create_dresser(
    htm=None,
    name="dresser",
    width=1.05,
    depth=0.48,
    height=0.82,
    rows=3,
    body_thickness=0.035,
    drawer_gap=0.02,
    body_color="#745036",
    drawer_color="#8A6040",
    handle_color="#303030",
):
    """Comoda com numero configuravel de gavetas."""
    objs = []

    objs.append(_box(
        f"{name}_body", htm,
        [0, 0, height / 2],
        width, depth, height, body_color
    ))

    usable_h = height - 2 * body_thickness
    drawer_h = (usable_h - (rows + 1) * drawer_gap) / rows

    for i in range(rows):
        z = body_thickness + drawer_gap + drawer_h / 2 + i * (drawer_h + drawer_gap)

        objs.append(_box(
            f"{name}_drawer_{i+1}", htm,
            [0, -depth / 2 - 0.006, z],
            width - 2 * body_thickness,
            0.022,
            drawer_h,
            drawer_color
        ))

        objs.append(_cyl_y(
            f"{name}_handle_{i+1}", htm,
            [0, -depth / 2 - 0.035, z],
            0.008, 0.07, handle_color
        ))

    return objs


# =============================================================================
# Sala
# =============================================================================

def create_sofa(
    htm=None,
    name="sofa",
    width=2.00,
    depth=0.88,
    seat_height=0.43,
    total_height=0.88,
    base_height=0.18,
    seat_depth=0.60,
    cushion_height=0.16,
    back_thickness=0.16,
    arm_width=0.16,
    arm_height=0.62,
    leg_height=0.07,
    leg_width=0.055,
    seat_back_overlap=0.015,
    seat_side_overlap=0.01,
    body_color="#495867",
    cushion_color="#596979",
    leg_color="#302820",
):
    """
    Sofa de dois/tres lugares formado por caixas.

    Convencao:
        - a base fica apoiada sobre os pes;
        - o assento repousa diretamente sobre a base, sem vao;
        - o encosto toca/penetra levemente o assento;
        - os bracos envolvem lateralmente o assento.

    `seat_height` representa a altura aproximada da SUPERFICIE do assento.
    A geometria e ajustada para evitar espacos visuais entre base e almofada.
    """
    objs = []

    # --------------------------------------------------
    # Pes
    # --------------------------------------------------
    x_leg = width / 2 - arm_width
    y_leg = depth / 2 - leg_width * 1.5

    for i, (px, py) in enumerate([
        (-x_leg, -y_leg),
        (-x_leg,  y_leg),
        ( x_leg, -y_leg),
        ( x_leg,  y_leg),
    ], 1):
        objs.append(_box(
            f"{name}_leg_{i}", htm,
            [px, py, leg_height / 2],
            leg_width, leg_width, leg_height, leg_color
        ))

    # --------------------------------------------------
    # Base estrutural
    # --------------------------------------------------
    base_bottom = leg_height
    base_top = base_bottom + base_height

    objs.append(_box(
        f"{name}_base", htm,
        [0, 0, base_bottom + base_height / 2],
        width, depth, base_height, body_color
    ))

    # --------------------------------------------------
    # Assento
    # --------------------------------------------------
    # O assento precisa ENCOSTAR na base.
    #
    # Se o seat_height pedido permitir, usamos a altura de superficie
    # fornecida. Caso contrario, garantimos no minimo contato com a base.
    desired_seat_bottom = seat_height - cushion_height
    seat_bottom = max(base_top, desired_seat_bottom)

    # Para evitar um vao de poucos milimetros por arredondamento,
    # fazemos uma pequena penetracao geometrica na base.
    seat_bottom -= 0.005
    seat_z = seat_bottom + cushion_height / 2

    seat_width = width - 2 * arm_width + 2 * seat_side_overlap

    # O fundo do assento entra levemente sob o encosto.
    back_front_y = depth / 2 - back_thickness
    seat_center_y = back_front_y - seat_depth / 2 + seat_back_overlap

    objs.append(_box(
        f"{name}_seat", htm,
        [0, seat_center_y, seat_z],
        seat_width,
        seat_depth,
        cushion_height,
        cushion_color
    ))

    seat_top = seat_bottom + cushion_height

    # --------------------------------------------------
    # Encosto
    # --------------------------------------------------
    # Parte inferior do encosto entra levemente no assento.
    back_bottom = seat_top - 0.035
    back_h = total_height - back_bottom

    if back_h <= 0:
        raise ValueError("'total_height' must be greater than the seat height.")

    objs.append(_box(
        f"{name}_back", htm,
        [
            0,
            depth / 2 - back_thickness / 2,
            back_bottom + back_h / 2,
        ],
        seat_width,
        back_thickness,
        back_h,
        cushion_color
    ))

    # --------------------------------------------------
    # Bracos
    # --------------------------------------------------
    # Comecam na base e sobem ate arm_height.
    # Um pequeno overlap lateral elimina frestas junto ao assento.
    arm_bottom = base_bottom
    actual_arm_height = max(arm_height - arm_bottom, 0.01)

    for side, x in [
        ("left",  -width / 2 + arm_width / 2),
        ("right",  width / 2 - arm_width / 2),
    ]:
        objs.append(_box(
            f"{name}_arm_{side}", htm,
            [x, 0, arm_bottom + actual_arm_height / 2],
            arm_width + 0.01,
            depth,
            actual_arm_height,
            body_color
        ))

    return objs


def create_armchair(
    htm=None,
    name="armchair",
    width=0.90,
    depth=0.85,
    **kwargs,
):
    """Poltrona baseada na geometria do sofa."""
    return create_sofa(
        htm=htm,
        name=name,
        width=width,
        depth=depth,
        **kwargs,
    )


def create_tv_stand(
    htm=None,
    name="tv_stand",
    width=1.60,
    depth=0.42,
    height=0.48,
    body_height=0.32,
    leg_height=0.12,
    leg_width=0.04,
    center_opening_width=0.55,
    body_color="#5A4638",
    shelf_color="#695242",
):
    """Rack para TV com nicho central."""
    objs = []

    z0 = leg_height

    objs.append(_box(
        f"{name}_top", htm,
        [0, 0, height - 0.025],
        width, depth, 0.05, shelf_color
    ))

    objs.append(_box(
        f"{name}_bottom", htm,
        [0, 0, z0 + 0.025],
        width, depth, 0.05, body_color
    ))

    side_w = (width - center_opening_width) / 2

    for side, x in [
        ("left", -(center_opening_width / 2 + side_w / 2)),
        ("right", +(center_opening_width / 2 + side_w / 2)),
    ]:
        objs.append(_box(
            f"{name}_cabinet_{side}", htm,
            [x, 0, z0 + body_height / 2],
            side_w, depth, body_height, body_color
        ))

    x = width / 2 - 0.08
    y = depth / 2 - 0.06

    for i, (px, py) in enumerate([
        (-x, -y), (-x, y), (x, -y), (x, y)
    ], 1):
        objs.append(_box(
            f"{name}_leg_{i}", htm,
            [px, py, leg_height / 2],
            leg_width, leg_width, leg_height, body_color
        ))

    return objs


def create_tv(
    htm=None,
    name="tv",
    width=1.10,
    height=0.64,
    thickness=0.045,
    bottom_height=0.60,
    stand_height=0.12,
    stand_width=0.34,
    screen_color="#111111",
    frame_color="#202020",
    stand_color="#303030",
):
    """Televisor em pe com pedestal simples."""
    objs = []

    screen_z = bottom_height + height / 2

    objs.append(_box(
        f"{name}_screen", htm,
        [0, 0, screen_z],
        width, thickness, height, screen_color
    ))

    # Moldura inferior simples
    objs.append(_box(
        f"{name}_frame_bottom", htm,
        [0, -thickness / 2 - 0.008, bottom_height + 0.015],
        width, 0.016, 0.03, frame_color
    ))

    objs.append(_box(
        f"{name}_stand", htm,
        [0, 0, bottom_height - stand_height / 2],
        0.055, 0.055, stand_height, stand_color
    ))

    objs.append(_box(
        f"{name}_stand_base", htm,
        [0, 0, bottom_height - stand_height],
        stand_width, 0.22, 0.025, stand_color
    ))

    return objs


def create_bookshelf(
    htm=None,
    name="bookshelf",
    width=0.90,
    depth=0.32,
    height=1.85,
    shelf_count=5,
    board_thickness=0.035,
    back_thickness=0.018,
    frame_color="#704D34",
    back_color="#5D402C",
):
    """Estante aberta com prateleiras."""
    objs = []

    # Laterais
    x = width / 2 - board_thickness / 2
    for side, px in [("left", -x), ("right", x)]:
        objs.append(_box(
            f"{name}_side_{side}", htm,
            [px, 0, height / 2],
            board_thickness, depth, height, frame_color
        ))

    # Fundo
    objs.append(_box(
        f"{name}_back", htm,
        [0, depth / 2 - back_thickness / 2, height / 2],
        width, back_thickness, height, back_color
    ))

    # Base, topo e prateleiras
    levels = np.linspace(
        board_thickness / 2,
        height - board_thickness / 2,
        shelf_count + 1
    )

    for i, z in enumerate(levels):
        objs.append(_box(
            f"{name}_shelf_{i}", htm,
            [0, 0, z],
            width, depth, board_thickness, frame_color
        ))

    return objs


def create_wall_shelf(
    htm=None,
    name="wall_shelf",
    width=0.90,
    depth=0.24,
    thickness=0.035,
    height_from_floor=1.30,
    bracket_width=0.025,
    bracket_height=0.16,
    shelf_color="#7A5235",
    bracket_color="#303030",
):
    """Prateleira de parede; origem permanece no chao."""
    objs = []

    objs.append(_box(
        f"{name}_board", htm,
        [0, 0, height_from_floor],
        width, depth, thickness, shelf_color
    ))

    x = width * 0.33

    for side, px in [("left", -x), ("right", x)]:
        objs.append(_box(
            f"{name}_bracket_{side}", htm,
            [px, depth / 2 - bracket_width / 2,
             height_from_floor - bracket_height / 2],
            bracket_width, bracket_width, bracket_height, bracket_color
        ))

    return objs


# =============================================================================
# Armarios
# =============================================================================

def create_wardrobe(
    htm=None,
    name="wardrobe",
    width=1.45,
    depth=0.58,
    height=2.00,
    door_gap=0.015,
    door_thickness=0.025,
    body_color="#6C4A32",
    door_color="#7C573C",
    handle_color="#303030",
):
    """Guarda-roupa de duas portas."""
    objs = []

    objs.append(_box(
        f"{name}_body", htm,
        [0, 0, height / 2],
        width, depth, height, body_color
    ))

    door_w = (width - 3 * door_gap) / 2
    y_front = -depth / 2 - door_thickness / 2

    for i, sign in enumerate([-1, 1], 1):
        x = sign * (door_w / 2 + door_gap / 2)
        objs.append(_box(
            f"{name}_door_{i}", htm,
            [x, y_front, height / 2],
            door_w, door_thickness, height - 2 * door_gap, door_color
        ))

        handle_x = sign * (door_gap / 2 + 0.035)
        objs.append(_cyl_z(
            f"{name}_handle_{i}", htm,
            [handle_x, y_front - 0.025, height * 0.53],
            0.009, 0.18, handle_color
        ))

    return objs


def create_cabinet(
    htm=None,
    name="cabinet",
    width=0.80,
    depth=0.42,
    height=1.15,
    door_thickness=0.022,
    body_color="#6B4C35",
    door_color="#7C5A40",
    handle_color="#303030",
):
    """Armario baixo/medio de duas portas."""
    return create_wardrobe(
        htm=htm,
        name=name,
        width=width,
        depth=depth,
        height=height,
        door_thickness=door_thickness,
        body_color=body_color,
        door_color=door_color,
        handle_color=handle_color,
    )


# =============================================================================
# Cozinha e lavanderia
# =============================================================================

def create_kitchen_counter(
    htm=None,
    name="kitchen_counter",
    width=1.80,
    depth=0.62,
    height=0.90,
    countertop_thickness=0.045,
    toe_kick_height=0.10,
    door_gap=0.018,
    body_color="#D0CBC1",
    countertop_color="#4F4F4F",
    door_color="#DDD8CF",
    handle_color="#303030",
):
    """Bancada de cozinha com tampo e quatro portas."""
    objs = []

    body_h = height - countertop_thickness - toe_kick_height

    objs.append(_box(
        f"{name}_body", htm,
        [0, 0, toe_kick_height + body_h / 2],
        width, depth, body_h, body_color
    ))

    objs.append(_box(
        f"{name}_countertop", htm,
        [0, 0, height - countertop_thickness / 2],
        width + 0.04, depth + 0.04, countertop_thickness, countertop_color
    ))

    door_w = (width - 5 * door_gap) / 4
    y = -depth / 2 - 0.012

    for i in range(4):
        x = -width / 2 + door_gap + door_w / 2 + i * (door_w + door_gap)
        z = toe_kick_height + body_h / 2

        objs.append(_box(
            f"{name}_door_{i+1}", htm,
            [x, y, z],
            door_w, 0.022, body_h - 2 * door_gap, door_color
        ))

        objs.append(_cyl_z(
            f"{name}_handle_{i+1}", htm,
            [x + door_w * 0.30, y - 0.025, z],
            0.007, 0.12, handle_color
        ))

    return objs


def create_kitchen_island(
    htm=None,
    name="kitchen_island",
    width=1.55,
    depth=0.88,
    height=0.92,
    countertop_thickness=0.055,
    overhang=0.10,
    body_color="#D8D3C9",
    countertop_color="#505050",
):
    """Ilha de cozinha simples."""
    objs = []

    body_h = height - countertop_thickness

    objs.append(_box(
        f"{name}_body", htm,
        [0, 0, body_h / 2],
        width - 2 * overhang, depth - 2 * overhang, body_h, body_color
    ))

    objs.append(_box(
        f"{name}_countertop", htm,
        [0, 0, height - countertop_thickness / 2],
        width, depth, countertop_thickness, countertop_color
    ))

    return objs


def create_refrigerator(
    htm=None,
    name="refrigerator",
    width=0.72,
    depth=0.72,
    height=1.82,
    door_thickness=0.035,
    freezer_fraction=0.32,
    gap=0.015,
    body_color="#BFC3C5",
    door_color="#D5D8DA",
    handle_color="#444444",
):
    """Geladeira duplex simplificada."""
    objs = []

    objs.append(_box(
        f"{name}_body", htm,
        [0, 0, height / 2],
        width, depth, height, body_color
    ))

    freezer_h = height * freezer_fraction
    fridge_h = height - freezer_h - gap
    y = -depth / 2 - door_thickness / 2

    objs.append(_box(
        f"{name}_fridge_door", htm,
        [0, y, fridge_h / 2],
        width - 2 * gap, door_thickness, fridge_h - gap, door_color
    ))

    objs.append(_box(
        f"{name}_freezer_door", htm,
        [0, y, fridge_h + gap + freezer_h / 2],
        width - 2 * gap, door_thickness, freezer_h - gap, door_color
    ))

    for suffix, z in [
        ("fridge", fridge_h * 0.67),
        ("freezer", fridge_h + gap + freezer_h * 0.45),
    ]:
        objs.append(_cyl_z(
            f"{name}_{suffix}_handle", htm,
            [width * 0.32, y - 0.03, z],
            0.009, min(0.35, height * 0.18), handle_color
        ))

    return objs


def create_stove(
    htm=None,
    name="stove",
    width=0.62,
    depth=0.62,
    height=0.90,
    top_thickness=0.045,
    oven_door_height=0.48,
    burner_radius=0.085,
    body_color="#BFC3C5",
    top_color="#292929",
    oven_color="#191919",
    burner_color="#080808",
):
    """Fogao de quatro bocas com forno."""
    objs = []

    objs.append(_box(
        f"{name}_body", htm,
        [0, 0, (height - top_thickness) / 2],
        width, depth, height - top_thickness, body_color
    ))

    objs.append(_box(
        f"{name}_top", htm,
        [0, 0, height - top_thickness / 2],
        width, depth, top_thickness, top_color
    ))

    y_front = -depth / 2 - 0.012
    objs.append(_box(
        f"{name}_oven_door", htm,
        [0, y_front, oven_door_height / 2 + 0.11],
        width * 0.84, 0.025, oven_door_height, oven_color
    ))

    bx = width * 0.25
    by = depth * 0.24

    for i, (x, y) in enumerate([
        (-bx, -by), (-bx, by), (bx, -by), (bx, by)
    ], 1):
        objs.append(_cyl_z(
            f"{name}_burner_{i}", htm,
            [x, y, height + 0.006],
            burner_radius, 0.012, burner_color
        ))

    return objs


def create_washing_machine(
    htm=None,
    name="washing_machine",
    width=0.62,
    depth=0.64,
    height=0.86,
    door_radius=0.22,
    body_color="#E4E5E6",
    panel_color="#C5C7C8",
    door_color="#30363B",
):
    """Maquina de lavar frontal simplificada."""
    objs = []

    objs.append(_box(
        f"{name}_body", htm,
        [0, 0, height / 2],
        width, depth, height, body_color
    ))

    y_front = -depth / 2 - 0.025

    objs.append(_cyl_y(
        f"{name}_door", htm,
        [0, y_front, height * 0.45],
        door_radius, 0.05, door_color
    ))

    objs.append(_box(
        f"{name}_panel", htm,
        [0, -depth / 2 - 0.015, height * 0.82],
        width * 0.86, 0.025, height * 0.16, panel_color
    ))

    return objs


# =============================================================================
# Escritorio
# =============================================================================

def create_filing_cabinet(
    htm=None,
    name="filing_cabinet",
    width=0.48,
    depth=0.56,
    height=0.72,
    drawers=3,
    gap=0.018,
    body_color="#70757A",
    drawer_color="#858A8E",
    handle_color="#303030",
):
    """Gaveteiro de escritorio."""
    objs = []

    objs.append(_box(
        f"{name}_body", htm,
        [0, 0, height / 2],
        width, depth, height, body_color
    ))

    drawer_h = (height - (drawers + 1) * gap) / drawers
    y = -depth / 2 - 0.014

    for i in range(drawers):
        z = gap + drawer_h / 2 + i * (drawer_h + gap)

        objs.append(_box(
            f"{name}_drawer_{i+1}", htm,
            [0, y, z],
            width - 2 * gap, 0.026, drawer_h, drawer_color
        ))

        objs.append(_cyl_y(
            f"{name}_handle_{i+1}", htm,
            [0, y - 0.025, z],
            0.007, 0.07, handle_color
        ))

    return objs


def create_monitor(
    htm=None,
    name="monitor",
    width=0.54,
    height=0.32,
    thickness=0.035,
    screen_bottom=0.18,
    stand_height=0.16,
    base_width=0.22,
    screen_color="#111111",
    frame_color="#262626",
):
    """Monitor de computador de mesa."""
    objs = []

    objs.append(_box(
        f"{name}_screen", htm,
        [0, 0, screen_bottom + height / 2],
        width, thickness, height, screen_color
    ))

    objs.append(_box(
        f"{name}_stand", htm,
        [0, 0, stand_height / 2],
        0.045, 0.045, stand_height, frame_color
    ))

    objs.append(_box(
        f"{name}_base", htm,
        [0, 0, 0.012],
        base_width, 0.15, 0.024, frame_color
    ))

    return objs


# =============================================================================
# Decoracao / pequenos objetos
# =============================================================================

def create_floor_lamp(
    htm=None,
    name="floor_lamp",
    height=1.62,
    base_radius=0.16,
    base_thickness=0.025,
    pole_radius=0.015,
    shade_radius=0.18,
    shade_height=0.24,
    base_color="#303030",
    shade_color="#E8D7B0",
):
    """Abajur/luminaria de piso simplificada."""
    objs = []

    objs.append(_cyl_z(
        f"{name}_base", htm,
        [0, 0, base_thickness / 2],
        base_radius, base_thickness, base_color
    ))

    pole_h = height - shade_height
    objs.append(_cyl_z(
        f"{name}_pole", htm,
        [0, 0, base_thickness + pole_h / 2],
        pole_radius, pole_h, base_color
    ))

    objs.append(_cyl_z(
        f"{name}_shade", htm,
        [0, 0, height - shade_height / 2],
        shade_radius, shade_height, shade_color
    ))

    return objs


def create_potted_plant(
    htm=None,
    name="potted_plant",
    pot_radius=0.16,
    pot_height=0.24,
    plant_height=0.70,
    stem_radius=0.018,
    leaf_radius=0.10,
    leaf_count=7,
    pot_color="#9A5D35",
    stem_color="#356B3B",
    leaf_color="#3F7E44",
):
    """Vaso com haste e folhas esfericas, util para criar geometria menos regular."""
    objs = []

    objs.append(_cyl_z(
        f"{name}_pot", htm,
        [0, 0, pot_height / 2],
        pot_radius, pot_height, pot_color
    ))

    stem_h = plant_height - pot_height
    objs.append(_cyl_z(
        f"{name}_stem", htm,
        [0, 0, pot_height + stem_h / 2],
        stem_radius, stem_h, stem_color
    ))

    for i in range(leaf_count):
        a = 2 * np.pi * i / leaf_count
        radial = 0.07 + 0.025 * (i % 3)
        z = pot_height + stem_h * (0.35 + 0.55 * i / max(leaf_count - 1, 1))
        x = radial * np.cos(a)
        y = radial * np.sin(a)

        objs.append(_ball(
            f"{name}_leaf_{i+1}", htm,
            [x, y, z],
            leaf_radius * (0.85 + 0.08 * (i % 2)),
            leaf_color
        ))

    return objs


def create_rug(
    htm=None,
    name="rug",
    width=1.80,
    depth=1.20,
    thickness=0.012,
    color="#8A6D5A",
):
    """Tapete retangular fino."""
    return [
        _box(
            f"{name}_body", htm,
            [0, 0, thickness / 2],
            width, depth, thickness, color
        )
    ]


def create_trash_bin(
    htm=None,
    name="trash_bin",
    radius=0.16,
    height=0.34,
    color="#555555",
):
    """Lixeira cilindrica simples."""
    return [
        _cyl_z(
            f"{name}_body", htm,
            [0, 0, height / 2],
            radius, height, color
        )
    ]


# =============================================================================
# Catalogo
# =============================================================================

FURNITURE_CATALOG = {
    # Mesas
    "office_desk": create_office_desk,
    "dining_table": create_dining_table,
    "coffee_table": create_coffee_table,
    "side_table": create_side_table,
    "round_table": create_round_table,

    # Assentos
    "chair": create_chair,
    "office_chair": create_office_chair,
    "stool": create_stool,
    "bench": create_bench,

    # Quarto
    "bed": create_bed,
    "single_bed": create_single_bed,
    "queen_bed": create_queen_bed,
    "nightstand": create_nightstand,
    "dresser": create_dresser,

    # Sala
    "sofa": create_sofa,
    "armchair": create_armchair,
    "tv_stand": create_tv_stand,
    "tv": create_tv,
    "bookshelf": create_bookshelf,
    "wall_shelf": create_wall_shelf,

    # Armarios
    "wardrobe": create_wardrobe,
    "cabinet": create_cabinet,

    # Cozinha/lavanderia
    "kitchen_counter": create_kitchen_counter,
    "kitchen_island": create_kitchen_island,
    "refrigerator": create_refrigerator,
    "stove": create_stove,
    "washing_machine": create_washing_machine,

    # Escritorio
    "filing_cabinet": create_filing_cabinet,
    "monitor": create_monitor,

    # Decoracao
    "floor_lamp": create_floor_lamp,
    "potted_plant": create_potted_plant,
    "rug": create_rug,
    "trash_bin": create_trash_bin,
}


def available_items():
    """Retorna os nomes dos itens disponiveis no catalogo."""
    return sorted(FURNITURE_CATALOG.keys())


def create_item(item_type, **kwargs):
    """
    Cria um item pelo nome do catalogo.

    Exemplo:
        sofa = create_item(
            "sofa",
            htm=ub.Utils.trn([1, 0, 0]),
            width=2.2,
        )
    """
    if item_type not in FURNITURE_CATALOG:
        raise ValueError(
            f"Unknown item '{item_type}'. "
            f"Available items: {', '.join(available_items())}"
        )

    return FURNITURE_CATALOG[item_type](**kwargs)


__all__ = [
    "create_office_desk",
    "create_dining_table",
    "create_coffee_table",
    "create_side_table",
    "create_round_table",
    "create_chair",
    "create_office_chair",
    "create_stool",
    "create_bench",
    "create_bed",
    "create_single_bed",
    "create_queen_bed",
    "create_nightstand",
    "create_dresser",
    "create_sofa",
    "create_armchair",
    "create_tv_stand",
    "create_tv",
    "create_bookshelf",
    "create_wall_shelf",
    "create_wardrobe",
    "create_cabinet",
    "create_kitchen_counter",
    "create_kitchen_island",
    "create_refrigerator",
    "create_stove",
    "create_washing_machine",
    "create_filing_cabinet",
    "create_monitor",
    "create_floor_lamp",
    "create_potted_plant",
    "create_rug",
    "create_trash_bin",
    "FURNITURE_CATALOG",
    "available_items",
    "create_item",
]
