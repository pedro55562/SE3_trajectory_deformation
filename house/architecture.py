
"""
architecture.py
===============

Elementos arquitetonicos modulares para cenarios UAIbot.

Convencoes:
    x -> largura/comprimento principal
    y -> profundidade/espessura
    z -> altura

Paredes:
    comprimento em x, espessura em y, altura em z.

Portas:
    angle = 0       -> fechada
    angle = pi/4    -> aberta 45 graus
    angle = pi/2    -> aberta 90 graus

Todas as funcoes retornam list de objetos UAIbot.
"""

import numpy as np
import uaibot as ub


# =============================================================================
# Helpers
# =============================================================================

def _base_htm(htm):
    return np.eye(4) if htm is None else htm


def _pose(htm, xyz, rotation=None):
    T = _base_htm(htm) @ ub.Utils.trn(xyz)
    if rotation is not None:
        T = T @ rotation
    return T


def _box(name, htm, xyz, width, depth, height, color,
         opacity=1.0, rotation=None):
    return ub.Box(
        name=name,
        htm=_pose(htm, xyz, rotation),
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


def _validate_positive(**values):
    for key, value in values.items():
        if value <= 0:
            raise ValueError(f"'{key}' must be positive.")


def _validate_opening(length, wall_height, opening_width,
                      opening_height, bottom, offset):
    _validate_positive(
        length=length,
        wall_height=wall_height,
        opening_width=opening_width,
        opening_height=opening_height,
    )

    if bottom < 0:
        raise ValueError("'bottom' cannot be negative.")

    left = offset - opening_width / 2
    right = offset + opening_width / 2

    if left <= -length / 2 + 0.04 or right >= length / 2 - 0.04:
        raise ValueError("Opening does not fit horizontally inside wall.")

    if bottom + opening_height >= wall_height:
        raise ValueError("Opening does not fit vertically inside wall.")



def _create_hinged_leaf(
    htm,
    name,
    hinge_x,
    width,
    height,
    thickness,
    direction,
    angle,
    color,
    edge_color="#5D3D28",
    knob_color="#C9A44C",
    knob_radius=0.026,
    knob_height=0.95,
    include_knob=True,
    include_hinges=True,
):
    """
    Cria uma folha de porta a partir do eixo REAL da dobradica.

    direction = +1 -> a folha sai da dobradica em direcao a +x quando fechada.
    direction = -1 -> a folha sai da dobradica em direcao a -x quando fechada.

    A transformacao e feita como:
        T_folha = T_base * Tr(hinge_x, 0, 0) * Rz(angle)

    Assim, todas as pecas da folha sao definidas no referencial da dobradica e
    giram rigidamente em torno do mesmo eixo, sem recalcular centros em um
    referencial intermediario.
    """
    _validate_positive(
        width=width,
        height=height,
        thickness=thickness,
        knob_radius=knob_radius,
    )

    if direction not in (-1, 1, -1.0, 1.0):
        raise ValueError("'direction' must be +1 or -1.")

    direction = float(direction)
    hinge_htm = _pose(
        htm,
        [hinge_x, 0.0, 0.0],
        ub.Utils.rotz(float(angle)),
    )

    objs = []

    # A caixa ocupa x=[0,width] para direction=+1 e x=[-width,0]
    # para direction=-1. Portanto x=0 e exatamente o eixo da dobradica.
    objs.append(_box(
        f"{name}_leaf",
        hinge_htm,
        [direction * width / 2, 0.0, height / 2],
        width,
        thickness,
        height,
        color,
    ))

    edge_w = min(0.032, width * 0.04)
    objs.append(_box(
        f"{name}_free_edge",
        hinge_htm,
        [direction * (width - edge_w / 2), 0.0, height / 2],
        edge_w,
        thickness + 0.004,
        height,
        edge_color,
    ))

    if include_knob:
        knob_dist = max(width - 0.085, width * 0.72)
        objs.append(_cyl_z(
            f"{name}_knob",
            hinge_htm,
            [
                direction * knob_dist,
                -thickness / 2 - knob_radius * 0.50,
                knob_height,
            ],
            knob_radius,
            0.022,
            knob_color,
        ))

    if include_hinges:
        hinge_radius = min(0.010, thickness * 0.28)
        hinge_h = 0.075
        for i, z in enumerate((0.30, height * 0.50, height - 0.30), 1):
            objs.append(_cyl_z(
                f"{name}_hinge_{i}",
                hinge_htm,
                [0.0, 0.0, z],
                hinge_radius,
                hinge_h,
                edge_color,
            ))

    return objs


# =============================================================================
# Piso / lajes / pilares
# =============================================================================

def create_floor(
    htm=None,
    name="floor",
    width=4.0,
    depth=4.0,
    thickness=0.10,
    color="#B8B0A2",
):
    """Piso com a face superior em z=0."""
    _validate_positive(width=width, depth=depth, thickness=thickness)

    return [_box(
        f"{name}_body",
        htm,
        [0, 0, -thickness / 2],
        width, depth, thickness, color
    )]


def create_slab(
    htm=None,
    name="slab",
    width=4.0,
    depth=4.0,
    thickness=0.16,
    bottom_z=0.0,
    color="#A8A8A8",
):
    """Laje com face inferior em bottom_z."""
    _validate_positive(width=width, depth=depth, thickness=thickness)

    return [_box(
        f"{name}_body",
        htm,
        [0, 0, bottom_z + thickness / 2],
        width, depth, thickness, color
    )]


def create_column(
    htm=None,
    name="column",
    width=0.25,
    depth=0.25,
    height=2.70,
    color="#D5D0C8",
):
    _validate_positive(width=width, depth=depth, height=height)

    return [_box(
        f"{name}_body",
        htm,
        [0, 0, height / 2],
        width, depth, height, color
    )]


# =============================================================================
# Paredes
# =============================================================================

def create_wall(
    htm=None,
    name="wall",
    length=4.0,
    height=2.70,
    thickness=0.12,
    color="#E3DED3",
):
    _validate_positive(length=length, height=height, thickness=thickness)

    return [_box(
        f"{name}_body",
        htm,
        [0, 0, height / 2],
        length, thickness, height, color
    )]


def create_wall_with_door(
    htm=None,
    name="wall_with_door",
    length=4.0,
    height=2.70,
    thickness=0.12,
    door_width=0.90,
    door_height=2.10,
    door_offset=0.0,
    include_door=False,
    door_angle=0.0,
    hinge_side="left",
    double_door=False,
    double_door_opening_mode="same_side",
    double_door_gap=0.008,
    door_side_clearance=0.006,
    door_top_clearance=0.010,
    wall_color="#E3DED3",
    door_color="#7A5438",
):
    """
    Parede com abertura de porta.

    ``door_width`` e a largura livre da abertura na parede.

    Para ``double_door=False``, ``door_angle`` controla a folha unica.

    Para ``double_door=True``, a abertura e dividida em duas folhas com
    dobradicas nos batentes externos. Um ``door_angle`` positivo abre as duas
    folhas para +y; um valor negativo abre ambas para -y. As folhas giram com
    sinais opostos porque seus eixos de dobradica sao espelhados.
    """
    _validate_opening(
        length, height, door_width, door_height, 0.0, door_offset
    )

    if double_door_gap < 0:
        raise ValueError("'double_door_gap' cannot be negative.")
    if door_side_clearance < 0 or door_top_clearance < 0:
        raise ValueError("Door clearances cannot be negative.")

    usable_door_width = door_width - 2 * door_side_clearance
    usable_door_height = door_height - door_top_clearance

    if usable_door_width <= 0 or usable_door_height <= 0:
        raise ValueError("Door clearances are too large for the opening.")
    if double_door and double_door_gap >= usable_door_width:
        raise ValueError("'double_door_gap' is too large for the door width.")

    objs = []

    wall_left = -length / 2
    wall_right = length / 2
    open_left = door_offset - door_width / 2
    open_right = door_offset + door_width / 2

    left_len = open_left - wall_left
    right_len = wall_right - open_right
    header_h = height - door_height

    objs.append(_box(
        f"{name}_left",
        htm,
        [wall_left + left_len / 2, 0, height / 2],
        left_len, thickness, height, wall_color
    ))

    objs.append(_box(
        f"{name}_right",
        htm,
        [open_right + right_len / 2, 0, height / 2],
        right_len, thickness, height, wall_color
    ))

    objs.append(_box(
        f"{name}_header",
        htm,
        [door_offset, 0, door_height + header_h / 2],
        door_width, thickness, header_h, wall_color
    ))

    if include_door:
        door_htm = _pose(htm, [door_offset, 0, 0])
        door_thickness = min(0.045, thickness * 0.55)

        if double_door:
            if double_door_opening_mode not in ("same_side", "opposite_sides"):
                raise ValueError(
                    "'double_door_opening_mode' must be 'same_side' or "
                    "'opposite_sides'."
                )

            a = float(door_angle)
            right_angle = -a if double_door_opening_mode == "same_side" else a

            objs += create_double_door(
                htm=door_htm,
                name=f"{name}_door",
                total_width=usable_door_width,
                height=usable_door_height,
                thickness=door_thickness,
                left_angle=a,
                right_angle=right_angle,
                gap=double_door_gap,
                color=door_color,
            )
        else:
            objs += create_door(
                htm=door_htm,
                name=f"{name}_door",
                width=usable_door_width,
                height=usable_door_height,
                thickness=door_thickness,
                angle=door_angle,
                hinge_side=hinge_side,
                color=door_color,
            )

    return objs


def create_wall_with_double_door(
    htm=None,
    name="wall_with_double_door",
    length=4.5,
    height=2.70,
    thickness=0.12,
    door_width=1.60,
    door_height=2.10,
    door_offset=0.0,
    include_door=True,
    door_angle=0.0,
    opening_mode="same_side",
    center_gap=0.008,
    side_clearance=0.004,
    top_clearance=0.008,
    wall_color="#E3DED3",
    door_color="#7A5438",
):
    """
    Parede com porta dupla calculada diretamente no referencial da parede.

    door_width e a largura TOTAL do vao da porta, nao a largura de cada folha.

    Geometria quando fechada:

        open_left                                      open_right
           |                                               |
           |--clear--[ folha E ]--gap--[ folha D ]--clear--|
                    ^                              ^
                 hinge E                        hinge D

    opening_mode:
        "same_side"
            As duas folhas abrem para o mesmo lado da parede, como uma porta
            francesa convencional. Para door_angle > 0, ambas avancam para +y.

        "opposite_sides"
            Cada folha abre para um lado diferente da parede. Para
            door_angle > 0, a esquerda vai para +y e a direita para -y.

    O eixo de cada dobradica e calculado a partir dos LIMITES REAIS DO VAO.
    Nenhuma folha e posicionada a partir do centro de outra porta.
    """
    _validate_opening(
        length, height, door_width, door_height, 0.0, door_offset
    )

    if opening_mode not in ("same_side", "opposite_sides"):
        raise ValueError(
            "'opening_mode' must be 'same_side' or 'opposite_sides'."
        )

    if center_gap < 0 or side_clearance < 0 or top_clearance < 0:
        raise ValueError("Door clearances cannot be negative.")

    open_left = door_offset - door_width / 2
    open_right = door_offset + door_width / 2

    # Pivos ficam junto aos batentes, descontando apenas a pequena folga real.
    left_hinge_x = open_left + side_clearance
    right_hinge_x = open_right - side_clearance
    hinge_span = right_hinge_x - left_hinge_x

    if center_gap >= hinge_span:
        raise ValueError("'center_gap' is too large for the door opening.")

    leaf_width = (hinge_span - center_gap) / 2
    leaf_height = door_height - top_clearance

    if leaf_width <= 0 or leaf_height <= 0:
        raise ValueError("Door clearances are too large for the opening.")

    # Checagem geometrica explicita da posicao fechada.
    left_free_x = left_hinge_x + leaf_width
    right_free_x = right_hinge_x - leaf_width
    actual_center_gap = right_free_x - left_free_x

    if not np.isclose(actual_center_gap, center_gap, atol=1e-10):
        raise RuntimeError("Internal double-door geometry calculation failed.")

    objs = []

    # -------------------------------------------------------------------------
    # Parede: abertura TOTAL door_width.
    # -------------------------------------------------------------------------
    wall_left = -length / 2
    wall_right = length / 2

    left_len = open_left - wall_left
    right_len = wall_right - open_right
    header_h = height - door_height

    objs.append(_box(
        f"{name}_left",
        htm,
        [wall_left + left_len / 2, 0.0, height / 2],
        left_len,
        thickness,
        height,
        wall_color,
    ))

    objs.append(_box(
        f"{name}_right",
        htm,
        [open_right + right_len / 2, 0.0, height / 2],
        right_len,
        thickness,
        height,
        wall_color,
    ))

    objs.append(_box(
        f"{name}_header",
        htm,
        [door_offset, 0.0, door_height + header_h / 2],
        door_width,
        thickness,
        header_h,
        wall_color,
    ))

    if not include_door:
        return objs

    # A folha fica ligeiramente mais fina que a parede.
    leaf_thickness = min(0.045, thickness * 0.55)
    a = float(door_angle)

    if opening_mode == "same_side":
        # Como a folha direita cresce em -x, ela precisa do angulo oposto para
        # que as DUAS extremidades livres avancem para o mesmo sinal de y.
        left_angle = a
        right_angle = -a
    else:
        # Mesmo angulo global -> por crescerem em direcoes x opostas, as folhas
        # avancam para lados opostos da parede (+y e -y).
        left_angle = a
        right_angle = a

    objs += _create_hinged_leaf(
        htm=htm,
        name=f"{name}_door_left",
        hinge_x=left_hinge_x,
        width=leaf_width,
        height=leaf_height,
        thickness=leaf_thickness,
        direction=+1,
        angle=left_angle,
        color=door_color,
    )

    objs += _create_hinged_leaf(
        htm=htm,
        name=f"{name}_door_right",
        hinge_x=right_hinge_x,
        width=leaf_width,
        height=leaf_height,
        thickness=leaf_thickness,
        direction=-1,
        angle=right_angle,
        color=door_color,
    )

    return objs


def create_wall_with_window(
    htm=None,
    name="wall_with_window",
    length=4.0,
    height=2.70,
    thickness=0.12,
    window_width=1.20,
    window_height=1.10,
    sill_height=0.90,
    window_offset=0.0,
    include_window=True,
    wall_color="#E3DED3",
):
    _validate_opening(
        length, height,
        window_width, window_height,
        sill_height, window_offset
    )

    objs = []

    wall_left = -length / 2
    wall_right = length / 2
    open_left = window_offset - window_width / 2
    open_right = window_offset + window_width / 2

    left_len = open_left - wall_left
    right_len = wall_right - open_right
    top_start = sill_height + window_height
    top_h = height - top_start

    objs.append(_box(
        f"{name}_left",
        htm,
        [wall_left + left_len / 2, 0, height / 2],
        left_len, thickness, height, wall_color
    ))

    objs.append(_box(
        f"{name}_right",
        htm,
        [open_right + right_len / 2, 0, height / 2],
        right_len, thickness, height, wall_color
    ))

    objs.append(_box(
        f"{name}_bottom",
        htm,
        [window_offset, 0, sill_height / 2],
        window_width, thickness, sill_height, wall_color
    ))

    objs.append(_box(
        f"{name}_header",
        htm,
        [window_offset, 0, top_start + top_h / 2],
        window_width, thickness, top_h, wall_color
    ))

    if include_window:
        objs += create_window(
            htm=_pose(htm, [window_offset, 0, sill_height - 0.01]),
            name=f"{name}_window",
            width=window_width + 0.02,
            height=window_height + 0.02,
            frame_depth=thickness + 0.03,
        )

    return objs


def create_wall_with_door_and_window(
    htm=None,
    name="wall_door_window",
    length=5.50,
    height=2.70,
    thickness=0.12,
    door_width=0.90,
    door_height=2.10,
    door_offset=-1.45,
    window_width=1.30,
    window_height=1.05,
    sill_height=0.95,
    window_offset=1.20,
    include_door=True,
    door_angle=0.0,
    hinge_side="left",
    double_door=False,
    double_door_gap=0.010,
    door_side_clearance=0.006,
    door_top_clearance=0.010,
    include_window=True,
    wall_color="#E3DED3",
    door_color="#7A5438",
):
    """Parede com uma abertura de porta e uma abertura de janela."""
    _validate_opening(
        length, height,
        door_width, door_height,
        0.0, door_offset
    )
    _validate_opening(
        length, height,
        window_width, window_height,
        sill_height, window_offset
    )

    if double_door_gap < 0:
        raise ValueError("'double_door_gap' cannot be negative.")
    if door_side_clearance < 0 or door_top_clearance < 0:
        raise ValueError("Door clearances cannot be negative.")

    usable_door_width = door_width - 2 * door_side_clearance
    usable_door_height = door_height - door_top_clearance
    if usable_door_width <= 0 or usable_door_height <= 0:
        raise ValueError("Door clearances are too large for the opening.")
    if double_door and double_door_gap >= usable_door_width:
        raise ValueError("'double_door_gap' is too large for the door width.")

    door_l = door_offset - door_width / 2
    door_r = door_offset + door_width / 2
    win_l = window_offset - window_width / 2
    win_r = window_offset + window_width / 2

    if not (door_r < win_l or win_r < door_l):
        raise ValueError("Door and window openings overlap.")

    objs = []

    wall_left = -length / 2
    wall_right = length / 2

    openings = sorted([
        ("door", door_l, door_r),
        ("window", win_l, win_r),
    ], key=lambda item: item[1])

    cursor = wall_left

    for i, (_, left, right) in enumerate(openings):
        seg_len = left - cursor
        if seg_len > 0:
            objs.append(_box(
                f"{name}_full_{i}",
                htm,
                [cursor + seg_len / 2, 0, height / 2],
                seg_len, thickness, height, wall_color
            ))
        cursor = right

    end_len = wall_right - cursor
    if end_len > 0:
        objs.append(_box(
            f"{name}_full_end",
            htm,
            [cursor + end_len / 2, 0, height / 2],
            end_len, thickness, height, wall_color
        ))

    door_header_h = height - door_height
    objs.append(_box(
        f"{name}_door_header",
        htm,
        [door_offset, 0, door_height + door_header_h / 2],
        door_width, thickness, door_header_h, wall_color
    ))

    objs.append(_box(
        f"{name}_window_sill_wall",
        htm,
        [window_offset, 0, sill_height / 2],
        window_width, thickness, sill_height, wall_color
    ))

    win_top = sill_height + window_height
    win_header_h = height - win_top

    objs.append(_box(
        f"{name}_window_header_wall",
        htm,
        [window_offset, 0, win_top + win_header_h / 2],
        window_width, thickness, win_header_h, wall_color
    ))

    if include_door:
        door_htm = _pose(htm, [door_offset, 0, 0])
        door_thickness = min(0.045, thickness * 0.55)

        if double_door:
            a = float(door_angle)
            objs += create_double_door(
                htm=door_htm,
                name=f"{name}_door",
                total_width=usable_door_width,
                height=usable_door_height,
                thickness=door_thickness,
                left_angle=a,
                right_angle=-a,
                gap=double_door_gap,
                color=door_color,
            )
        else:
            objs += create_door(
                htm=door_htm,
                name=f"{name}_door",
                width=usable_door_width,
                height=usable_door_height,
                thickness=door_thickness,
                angle=door_angle,
                hinge_side=hinge_side,
                color=door_color,
            )

    if include_window:
        objs += create_window(
            htm=_pose(htm, [window_offset, 0, sill_height - 0.01]),
            name=f"{name}_window",
            width=window_width + 0.02,
            height=window_height + 0.02,
            frame_depth=thickness + 0.03,
        )

    return objs


# =============================================================================
# Portas
# =============================================================================

def create_door(
    htm=None,
    name="door",
    width=0.88,
    height=2.06,
    thickness=0.04,
    angle=0.0,
    hinge_side="left",
    color="#7A5438",
    edge_color="#5D3D28",
    knob_color="#C9A44C",
    knob_radius=0.026,
    knob_height=0.95,
    include_knob=True,
    include_hinges=True,
):
    """
    Porta cujo giro acontece em torno da dobradica real.

    A origem local fica no centro da abertura, no piso.
    """
    _validate_positive(
        width=width,
        height=height,
        thickness=thickness,
        knob_radius=knob_radius,
    )

    if hinge_side not in ("left", "right"):
        raise ValueError("hinge_side must be 'left' or 'right'.")

    objs = []

    hinge_sign = -1.0 if hinge_side == "left" else 1.0
    hinge_x = hinge_sign * width / 2

    # vetor da dobradica ate o centro da folha
    v = np.array([-hinge_sign * width / 2, 0.0])

    c = np.cos(angle)
    s = np.sin(angle)
    R = np.array([[c, -s], [s, c]])

    center_xy = np.array([hinge_x, 0.0]) + R @ v
    Rz = ub.Utils.rotz(angle)

    objs.append(_box(
        f"{name}_leaf",
        htm,
        [center_xy[0], center_xy[1], height / 2],
        width, thickness, height, color,
        rotation=Rz,
    ))

    # borda oposta / acabamento
    edge_w = min(0.032, width * 0.04)
    free_edge_x = -hinge_sign * (width / 2 - edge_w / 2)
    free_vec = np.array([free_edge_x - hinge_x, 0.0])
    free_xy = np.array([hinge_x, 0.0]) + R @ free_vec

    objs.append(_box(
        f"{name}_free_edge",
        htm,
        [free_xy[0], free_xy[1], height / 2],
        edge_w, thickness + 0.004, height, edge_color,
        rotation=Rz,
    ))

    if include_knob:
        knob_x_local = -hinge_sign * (width / 2 - 0.085)
        knob_vec = np.array([
            knob_x_local - hinge_x,
            -thickness / 2 - knob_radius * 0.50,
        ])
        knob_xy = np.array([hinge_x, 0.0]) + R @ knob_vec

        objs.append(_cyl_z(
            f"{name}_knob",
            htm,
            [knob_xy[0], knob_xy[1], knob_height],
            knob_radius,
            0.022,
            knob_color,
        ))

    if include_hinges:
        hinge_radius = min(0.010, thickness * 0.28)
        hinge_h = 0.075

        for i, z in enumerate((0.30, height * 0.50, height - 0.30), 1):
            objs.append(_cyl_z(
                f"{name}_hinge_{i}",
                htm,
                [hinge_x, 0.0, z],
                hinge_radius,
                hinge_h,
                edge_color,
            ))

    return objs


def create_double_door(
    htm=None,
    name="double_door",
    total_width=1.60,
    height=2.10,
    thickness=0.042,
    left_angle=0.0,
    right_angle=0.0,
    gap=0.010,
    color="#765136",
    edge_color="#5D3D28",
    knob_color="#C9A44C",
    knob_radius=0.026,
    knob_height=0.95,
    include_knobs=True,
    include_hinges=True,
):
    """
    Porta dupla isolada com dois pivôs externos.

    A origem local e o centro do conjunto. ``total_width`` e a distancia entre
    os dois pivôs externos. Quando fechada:

        hinge E ---- folha E ---- gap ---- folha D ---- hinge D
       -W/2                                           +W/2

    ``left_angle`` e ``right_angle`` sao angulos globais de cada folha.
    Para uma abertura convencional para o mesmo lado, use por exemplo
    left_angle=+pi/3 e right_angle=-pi/3.
    """
    _validate_positive(
        total_width=total_width,
        height=height,
        thickness=thickness,
        knob_radius=knob_radius,
    )

    if gap < 0 or gap >= total_width:
        raise ValueError("'gap' must satisfy 0 <= gap < total_width.")

    leaf_width = (total_width - gap) / 2
    left_hinge_x = -total_width / 2
    right_hinge_x = total_width / 2

    objs = []

    objs += _create_hinged_leaf(
        htm=htm,
        name=f"{name}_left",
        hinge_x=left_hinge_x,
        width=leaf_width,
        height=height,
        thickness=thickness,
        direction=+1,
        angle=left_angle,
        color=color,
        edge_color=edge_color,
        knob_color=knob_color,
        knob_radius=knob_radius,
        knob_height=knob_height,
        include_knob=include_knobs,
        include_hinges=include_hinges,
    )

    objs += _create_hinged_leaf(
        htm=htm,
        name=f"{name}_right",
        hinge_x=right_hinge_x,
        width=leaf_width,
        height=height,
        thickness=thickness,
        direction=-1,
        angle=right_angle,
        color=color,
        edge_color=edge_color,
        knob_color=knob_color,
        knob_radius=knob_radius,
        knob_height=knob_height,
        include_knob=include_knobs,
        include_hinges=include_hinges,
    )

    return objs


# =============================================================================
# Janelas
# =============================================================================

def create_window(
    htm=None,
    name="window",
    width=1.20,
    height=1.05,
    frame_width=0.050,
    frame_depth=0.070,
    glass_thickness=0.012,
    frame_color="#F0EEE9",
    glass_color="#9CC8D8",
    glass_opacity=0.38,
    divider=True,
):
    # frame_depth pode ser maior que a espessura da parede para
    # criar uma pequena interseção geométrica e evitar frestas visuais.
    _validate_positive(
        width=width,
        height=height,
        frame_width=frame_width,
        frame_depth=frame_depth,
        glass_thickness=glass_thickness,
    )

    inner_w = width - 2 * frame_width
    inner_h = height - 2 * frame_width

    if inner_w <= 0 or inner_h <= 0:
        raise ValueError("Window frame too thick.")

    objs = []

    fw = frame_width

    objs += [
        _box(
            f"{name}_left", htm,
            [-width / 2 + fw / 2, 0, height / 2],
            fw, frame_depth, height, frame_color
        ),
        _box(
            f"{name}_right", htm,
            [width / 2 - fw / 2, 0, height / 2],
            fw, frame_depth, height, frame_color
        ),
        _box(
            f"{name}_bottom", htm,
            [0, 0, fw / 2],
            width, frame_depth, fw, frame_color
        ),
        _box(
            f"{name}_top", htm,
            [0, 0, height - fw / 2],
            width, frame_depth, fw, frame_color
        ),
        _box(
            f"{name}_glass", htm,
            [0, 0, height / 2],
            inner_w, glass_thickness, inner_h,
            glass_color, opacity=glass_opacity
        ),
    ]

    if divider:
        d = fw * 0.65
        objs += [
            _box(
                f"{name}_divider_v", htm,
                [0, 0, height / 2],
                d, frame_depth * 0.80, inner_h, frame_color
            ),
            _box(
                f"{name}_divider_h", htm,
                [0, 0, height / 2],
                inner_w, frame_depth * 0.80, d, frame_color
            ),
        ]

    return objs


def create_bay_window(
    htm=None,
    name="bay_window",
    width=1.80,
    height=1.20,
    projection=0.32,
    frame_depth=0.06,
    frame_color="#EEEAE2",
    glass_color="#9CC8D8",
):
    """Bay window simplificada em tres paineis."""
    center_w = width * 0.58
    side_w = (width - center_w) / 2
    side_angle = np.arctan2(projection, side_w)

    objs = []

    objs += create_window(
        htm=_pose(htm, [0, -projection, 0]),
        name=f"{name}_center",
        width=center_w,
        height=height,
        frame_depth=frame_depth,
        frame_color=frame_color,
        glass_color=glass_color,
    )

    x = center_w / 2 + side_w / 2
    y = -projection / 2

    objs += create_window(
        htm=_pose(htm, [-x, y, 0], ub.Utils.rotz(side_angle)),
        name=f"{name}_left",
        width=side_w,
        height=height,
        frame_depth=frame_depth,
        frame_color=frame_color,
        glass_color=glass_color,
        divider=False,
    )

    objs += create_window(
        htm=_pose(htm, [x, y, 0], ub.Utils.rotz(-side_angle)),
        name=f"{name}_right",
        width=side_w,
        height=height,
        frame_depth=frame_depth,
        frame_color=frame_color,
        glass_color=glass_color,
        divider=False,
    )

    return objs


# =============================================================================
# Escadas / corrimao
# =============================================================================

def create_stairs(
    htm=None,
    name="stairs",
    width=1.0,
    total_run=3.0,
    total_rise=2.70,
    n_steps=15,
    direction="positive_y",
    color="#9C856C",
):
    """
    Escada reta.

    direction:
        positive_y
        negative_y
    """
    _validate_positive(
        width=width,
        total_run=total_run,
        total_rise=total_rise,
    )

    if n_steps < 2:
        raise ValueError("'n_steps' must be at least 2.")

    if direction not in ("positive_y", "negative_y"):
        raise ValueError("Invalid direction.")

    sign = 1.0 if direction == "positive_y" else -1.0

    run = total_run / n_steps
    rise = total_rise / n_steps

    objs = []

    for i in range(n_steps):
        h = (i + 1) * rise
        y = sign * (-total_run / 2 + run / 2 + i * run)

        objs.append(_box(
            f"{name}_step_{i+1}",
            htm,
            [0, y, h / 2],
            width, run, h, color
        ))

    return objs


def create_railing(
    htm=None,
    name="railing",
    length=3.0,
    height=0.95,
    post_spacing=0.65,
    post_width=0.035,
    rail_depth=0.050,
    rail_height=0.045,
    color="#4A4A4A",
):
    _validate_positive(
        length=length,
        height=height,
        post_spacing=post_spacing,
    )

    n_posts = max(2, int(np.ceil(length / post_spacing)) + 1)
    xs = np.linspace(-length / 2, length / 2, n_posts)

    objs = []

    for i, x in enumerate(xs, 1):
        objs.append(_box(
            f"{name}_post_{i}",
            htm,
            [x, 0, height / 2],
            post_width, post_width, height, color
        ))

    objs.append(_box(
        f"{name}_rail",
        htm,
        [0, 0, height],
        length + post_width,
        rail_depth,
        rail_height,
        color
    ))

    return objs


# =============================================================================
# Garagem
# =============================================================================

def create_garage_door(
    htm=None,
    name="garage_door",
    width=4.80,
    height=2.30,
    thickness=0.045,
    panel_count=5,
    open_fraction=0.0,
    panel_color="#D7D4CC",
    trim_color="#5A5A5A",
):
    """
    Porta seccionada simples.

    open_fraction:
        0 -> fechada
        1 -> recolhida para cima
    """
    _validate_positive(width=width, height=height, thickness=thickness)

    if panel_count < 2:
        raise ValueError("'panel_count' must be at least 2.")

    if not 0 <= open_fraction <= 1:
        raise ValueError("'open_fraction' must be between 0 and 1.")

    objs = []

    ph = height / panel_count
    lift = open_fraction * height

    for i in range(panel_count):
        z = ph / 2 + i * ph + lift
        objs.append(_box(
            f"{name}_panel_{i+1}",
            htm,
            [0, 0, z],
            width, thickness, ph * 0.94, panel_color
        ))

    trim_w = 0.055

    objs += [
        _box(
            f"{name}_track_left", htm,
            [-width / 2 - trim_w / 2, 0.01, height / 2],
            trim_w, thickness * 0.8, height, trim_color
        ),
        _box(
            f"{name}_track_right", htm,
            [width / 2 + trim_w / 2, 0.01, height / 2],
            trim_w, thickness * 0.8, height, trim_color
        ),
    ]

    return objs


# =============================================================================
# Telhados
# =============================================================================

def create_gable_roof(
    htm=None,
    name="gable_roof",
    width=6.0,
    depth=8.0,
    base_height=2.70,
    pitch_angle=np.deg2rad(30),
    thickness=0.10,
    overhang=0.25,
    color="#6E3F33",
):
    """
    Telhado de duas aguas.
    Cumeeira paralela ao eixo y.
    """
    _validate_positive(width=width, depth=depth, thickness=thickness)

    half_span = width / 2 + overhang
    roof_depth = depth + 2 * overhang

    slope_len = half_span / np.cos(pitch_angle)
    rise = half_span * np.tan(pitch_angle)

    x_center = half_span / 2
    z_center = base_height + rise / 2

    objs = [
        _box(
            f"{name}_left",
            htm,
            [-x_center, 0, z_center],
            slope_len, roof_depth, thickness, color,
            rotation=ub.Utils.roty(-pitch_angle),
            opacity=0.1,
        ),
        _box(
            f"{name}_right",
            htm,
            [x_center, 0, z_center],
            slope_len, roof_depth, thickness, color,
            rotation=ub.Utils.roty(pitch_angle),
            opacity=0.1,
        ),
    ]

    ridge_radius = thickness * 0.42

    objs.append(
        ub.Cylinder(
            name=f"{name}_ridge",
            htm=_pose(
                htm,
                [0, 0, base_height + rise],
                ub.Utils.rotx(np.pi / 2),
            ),
            radius=ridge_radius,
            height=roof_depth,
            color="#593229",
            opacity=0.1,
        )
    )

    return objs


def create_flat_roof(
    htm=None,
    name="flat_roof",
    width=6.0,
    depth=8.0,
    base_height=2.70,
    thickness=0.16,
    overhang=0.15,
    color="#8A8A84",
):
    return [_box(
        f"{name}_slab",
        htm,
        [0, 0, base_height + thickness / 2],
        width + 2 * overhang,
        depth + 2 * overhang,
        thickness,
        color,
    )]


# =============================================================================
# Entrada / varanda
# =============================================================================

def create_porch_steps(
    htm=None,
    name="porch_steps",
    width=1.50,
    total_depth=0.90,
    total_height=0.45,
    n_steps=3,
    color="#A99C8D",
):
    _validate_positive(
        width=width,
        total_depth=total_depth,
        total_height=total_height,
    )

    if n_steps < 1:
        raise ValueError("'n_steps' must be at least 1.")

    d = total_depth / n_steps
    h = total_height / n_steps

    objs = []

    for i in range(n_steps):
        step_h = (i + 1) * h
        step_depth = total_depth - i * d

        objs.append(_box(
            f"{name}_step_{i+1}",
            htm,
            [0, -(total_depth - step_depth) / 2, step_h / 2],
            width, step_depth, step_h, color
        ))

    return objs


# =============================================================================
# Catalogo
# =============================================================================

ARCHITECTURE_CATALOG = {
    "floor": create_floor,
    "slab": create_slab,
    "column": create_column,

    "wall": create_wall,
    "wall_with_door": create_wall_with_door,
    "wall_with_double_door": create_wall_with_double_door,
    "wall_with_window": create_wall_with_window,
    "wall_with_door_and_window": create_wall_with_door_and_window,

    "door": create_door,
    "double_door": create_double_door,

    "window": create_window,
    "bay_window": create_bay_window,

    "stairs": create_stairs,
    "railing": create_railing,

    "garage_door": create_garage_door,

    "gable_roof": create_gable_roof,
    "flat_roof": create_flat_roof,

    "porch_steps": create_porch_steps,
}


def available_architecture():
    return sorted(ARCHITECTURE_CATALOG.keys())


def create_architecture(architecture_type, **kwargs):
    if architecture_type not in ARCHITECTURE_CATALOG:
        raise ValueError(
            f"Unknown architecture type '{architecture_type}'. "
            f"Available: {', '.join(available_architecture())}"
        )

    return ARCHITECTURE_CATALOG[architecture_type](**kwargs)


__all__ = [
    "create_floor",
    "create_slab",
    "create_column",
    "create_wall",
    "create_wall_with_door",
    "create_wall_with_double_door",
    "create_wall_with_window",
    "create_wall_with_door_and_window",
    "create_door",
    "create_double_door",
    "create_window",
    "create_bay_window",
    "create_stairs",
    "create_railing",
    "create_garage_door",
    "create_gable_roof",
    "create_flat_roof",
    "create_porch_steps",
    "ARCHITECTURE_CATALOG",
    "available_architecture",
    "create_architecture",
]
