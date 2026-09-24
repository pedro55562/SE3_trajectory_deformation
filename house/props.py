"""
props.py
========

Objetos pequenos e conjuntos utilitarios para compor cenarios em UAIbot.

Convencao:
    x -> largura
    y -> profundidade
    z -> altura

Origem local:
    - Em objetos apoiados sobre superficies, a origem fica no centro da
      projecao sobre o chao/superficie (z = 0 local do objeto).
    - Todos os objetos retornam LIST[simobject], para manter o estilo simples.

Exemplo:
    import uaibot as ub
    from props import create_computer_setup

    objs = create_computer_setup(
        htm=ub.Utils.trn([0.0, 0.0, 0.75]),
        dual_monitor=True,
    )

    sim = ub.Simulation.create_sim_grid(objs)
    sim.run()
"""

import numpy as np
import uaibot as ub


# =============================================================================
# Helpers internos
# =============================================================================

def _base_htm(htm):
    return np.eye(4) if htm is None else htm


def _pose(htm, xyz, rotation=None):
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
# Livros e papelaria
# =============================================================================

def create_book(
    htm=None,
    name="book",
    width=0.16,
    depth=0.024,
    height=0.23,
    cover_color="#355C7D",
    page_color="#F4F0E6",
    cover_thickness=0.003,
):
    """
    Livro simples.

    Convencao local:
        - width  -> largura (x)
        - depth  -> espessura do livro (y)
        - height -> altura do livro (z)

    O livro fica "em pe", bom para colocar em estantes.
    """
    objs = []

    if cover_thickness < 0:
        raise ValueError("'cover_thickness' must be non-negative.")

    page_w = max(width - 2 * cover_thickness, 0.005)
    page_d = max(depth - 2 * cover_thickness, 0.005)
    page_h = max(height - 2 * cover_thickness, 0.005)

    # Miolo
    objs.append(_box(
        f"{name}_pages", htm,
        [0, 0, height / 2],
        page_w, page_d, page_h, page_color
    ))

    # Capa principal
    objs.append(_box(
        f"{name}_cover", htm,
        [0, 0, height / 2],
        width, depth, height, cover_color, opacity=0.98
    ))

    return objs


def create_book_flat(
    htm=None,
    name="book_flat",
    width=0.23,
    depth=0.16,
    thickness=0.024,
    cover_color="#6C5B7B",
    page_color="#F4F0E6",
):
    """Livro deitado."""
    objs = []

    objs.append(_box(
        f"{name}_pages", htm,
        [0, 0, thickness / 2],
        width * 0.92, depth * 0.92, thickness * 0.85, page_color
    ))

    objs.append(_box(
        f"{name}_cover", htm,
        [0, 0, thickness / 2],
        width, depth, thickness, cover_color
    ))

    return objs


def create_book_row(
    htm=None,
    name="book_row",
    n_books=10,
    book_width=0.16,
    book_depth=0.024,
    book_height=0.23,
    spacing=0.004,
    lean_angle=0.0,
    colors=None,
):
    """
    Fileira de livros em pe ao longo de x.

    A base dos livros comeca em z = 0.
    """
    objs = []

    if colors is None:
        colors = [
            "#355C7D", "#6C5B7B", "#C06C84", "#F67280",
            "#99B898", "#8C5E58", "#4D7EA8", "#7A4E2D",
        ]

    total = n_books * book_width + max(n_books - 1, 0) * spacing
    x0 = -total / 2 + book_width / 2

    for i in range(n_books):
        x = x0 + i * (book_width + spacing)
        color = colors[i % len(colors)]

        local_htm = _pose(
            htm,
            [x, 0, 0],
            ub.Utils.rotz(lean_angle if i % 2 == 0 else -lean_angle * 0.5)
        )

        objs += create_book(
            htm=local_htm,
            name=f"{name}_{i+1}",
            width=book_width,
            depth=book_depth,
            height=book_height,
            cover_color=color,
        )

    return objs


def create_book_stack(
    htm=None,
    name="book_stack",
    n_books=5,
    width=0.23,
    depth=0.16,
    thickness=0.024,
    spacing=0.003,
    rotation_step=np.pi / 40,
    colors=None,
):
    """Pilha de livros deitados."""
    objs = []

    if colors is None:
        colors = [
            "#355C7D", "#6C5B7B", "#C06C84", "#99B898",
            "#8C5E58", "#4D7EA8", "#7A4E2D",
        ]

    z = 0.0
    for i in range(n_books):
        color = colors[i % len(colors)]
        local_htm = _pose(
            htm,
            [0, 0, z],
            ub.Utils.rotz((i - n_books / 2) * rotation_step)
        )

        objs += create_book_flat(
            htm=local_htm,
            name=f"{name}_{i+1}",
            width=width,
            depth=depth,
            thickness=thickness,
            cover_color=color,
        )
        z += thickness + spacing

    return objs


def create_notebook(
    htm=None,
    name="notebook",
    width=0.21,
    depth=0.15,
    thickness=0.012,
    cover_color="#4B6584",
    page_color="#F6F4EE",
    spiral_color="#404040",
):
    """Caderno simples com espiral."""
    objs = []

    objs.append(_box(
        f"{name}_body", htm,
        [0, 0, thickness / 2],
        width, depth, thickness, cover_color
    ))

    objs.append(_box(
        f"{name}_pages", htm,
        [0, 0, thickness * 0.52],
        width * 0.94, depth * 0.94, thickness * 0.72, page_color
    ))

    objs.append(_cyl_x(
        f"{name}_spiral", htm,
        [0, depth / 2 - 0.01, thickness / 2],
        0.004, width * 0.90, spiral_color
    ))

    return objs


def create_pen(
    htm=None,
    name="pen",
    length=0.145,
    radius=0.005,
    color="#2D4059",
    orientation="x",
):
    """Caneta simples."""
    if orientation == "x":
        return [_cyl_x(f"{name}_body", htm, [0, 0, radius], radius, length, color)]
    elif orientation == "y":
        return [_cyl_y(f"{name}_body", htm, [0, 0, radius], radius, length, color)]
    elif orientation == "z":
        return [_cyl_z(f"{name}_body", htm, [0, 0, length / 2], radius, length, color)]
    else:
        raise ValueError("orientation must be 'x', 'y' or 'z'.")


def create_pen_holder(
    htm=None,
    name="pen_holder",
    cup_radius=0.035,
    cup_height=0.11,
    pen_count=4,
    cup_color="#7C7C7C",
    pen_colors=None,
):
    """Porta-canetas com canetas simples."""
    objs = []

    objs.append(_cyl_z(
        f"{name}_cup", htm,
        [0, 0, cup_height / 2],
        cup_radius, cup_height, cup_color
    ))

    if pen_colors is None:
        pen_colors = ["#2D4059", "#EA5455", "#00ADB5", "#F07B3F"]

    for i in range(pen_count):
        ang = 2 * np.pi * i / max(pen_count, 1)
        r = cup_radius * 0.45
        x = r * np.cos(ang)
        y = r * np.sin(ang)
        color = pen_colors[i % len(pen_colors)]

        objs += create_pen(
            htm=_pose(htm, [x, y, 0]),
            name=f"{name}_pen_{i+1}",
            length=0.12,
            radius=0.0045,
            color=color,
            orientation="z",
        )

    return objs


# =============================================================================
# Computador e escritorio
# =============================================================================

def create_keyboard(
    htm=None,
    name="keyboard",
    width=0.44,
    depth=0.15,
    height=0.018,
    keyplate_color="#2B2B2B",
    body_color="#3A3A3A",
):
    """Teclado simplificado."""
    objs = []

    objs.append(_box(
        f"{name}_body", htm,
        [0, 0, height / 2],
        width, depth, height, body_color
    ))

    objs.append(_box(
        f"{name}_keyplate", htm,
        [0, 0, height * 0.70],
        width * 0.95, depth * 0.90, height * 0.35, keyplate_color
    ))

    return objs


def create_mouse(
    htm=None,
    name="mouse",
    width=0.065,
    depth=0.11,
    height=0.035,
    color="#2B2B2B",
):
    """Mouse simples."""
    return [
        _box(
            f"{name}_body", htm,
            [0, 0, height / 2],
            width, depth, height, color
        )
    ]


def create_mousepad(
    htm=None,
    name="mousepad",
    width=0.24,
    depth=0.20,
    thickness=0.005,
    color="#202020",
):
    """Mousepad fino."""
    return [
        _box(
            f"{name}_body", htm,
            [0, 0, thickness / 2],
            width, depth, thickness, color
        )
    ]


def create_desktop_tower(
    htm=None,
    name="desktop_tower",
    width=0.22,
    depth=0.46,
    height=0.46,
    body_color="#262626",
    panel_color="#111111",
    accent_color="#00ADB5",
):
    """Gabinete de computador."""
    objs = []

    objs.append(_box(
        f"{name}_body", htm,
        [0, 0, height / 2],
        width, depth, height, body_color
    ))

    objs.append(_box(
        f"{name}_front_panel", htm,
        [0, -depth / 2 - 0.006, height * 0.52],
        width * 0.92, 0.012, height * 0.88, panel_color
    ))

    objs.append(_ball(
        f"{name}_power_button", htm,
        [0, -depth / 2 - 0.012, height * 0.82],
        0.010, accent_color
    ))

    objs.append(_box(
        f"{name}_light", htm,
        [0, -depth / 2 - 0.010, height * 0.25],
        width * 0.50, 0.006, 0.02, accent_color
    ))

    return objs


def create_small_speaker(
    htm=None,
    name="speaker",
    width=0.085,
    depth=0.10,
    height=0.15,
    body_color="#2C2C2C",
    cone_color="#151515",
):
    """Caixa de som pequena."""
    objs = []

    objs.append(_box(
        f"{name}_body", htm,
        [0, 0, height / 2],
        width, depth, height, body_color
    ))

    objs.append(_cyl_y(
        f"{name}_woofer", htm,
        [0, -depth / 2 - 0.006, height * 0.42],
        min(width, height) * 0.22, 0.012, cone_color
    ))

    objs.append(_cyl_y(
        f"{name}_tweeter", htm,
        [0, -depth / 2 - 0.006, height * 0.74],
        min(width, height) * 0.10, 0.012, cone_color
    ))

    return objs


def create_speaker_pair(
    htm=None,
    name="speaker_pair",
    spacing=0.36,
    **speaker_kwargs,
):
    """Par de caixas de som."""
    objs = []
    objs += create_small_speaker(
        htm=_pose(htm, [-spacing / 2, 0, 0]),
        name=f"{name}_left",
        **speaker_kwargs,
    )
    objs += create_small_speaker(
        htm=_pose(htm, [spacing / 2, 0, 0]),
        name=f"{name}_right",
        **speaker_kwargs,
    )
    return objs


def create_webcam(
    htm=None,
    name="webcam",
    width=0.08,
    depth=0.05,
    height=0.04,
    color="#1C1C1C",
    lens_color="#111111",
):
    """Webcam simples."""
    objs = []

    objs.append(_box(
        f"{name}_body", htm,
        [0, 0, height / 2],
        width, depth, height, color
    ))

    objs.append(_ball(
        f"{name}_lens", htm,
        [0, -depth / 2 - 0.01, height / 2],
        min(width, height) * 0.14, lens_color
    ))

    return objs


def create_monitor_prop(
    htm=None,
    name="monitor_prop",
    width=0.58,
    height=0.34,
    thickness=0.04,
    screen_bottom=0.18,
    stand_height=0.16,
    base_width=0.24,
    screen_color="#111111",
    frame_color="#262626",
    accent_color="#333333",
):
    """Monitor para usar nos props/sets.

    A tela comeca em ``screen_bottom`` e o suporte e prolongado
    automaticamente ate entrar alguns milimetros no frame. Isso evita
    o vao visual entre a haste e a tela mesmo quando ``stand_height``
    for ligeiramente menor que ``screen_bottom``.
    """
    objs = []

    base_height = 0.024
    stand_width = 0.045
    stand_depth = 0.045

    # Pequena penetracao no frame para que nao apareca uma fresta por
    # arredondamento/renderizacao. ``stand_height`` continua funcionando
    # como altura minima desejada para a haste.
    stand_overlap = min(0.012, max(0.006, height * 0.025))
    stand_top = max(stand_height, screen_bottom + stand_overlap)
    stand_bottom = base_height * 0.55
    effective_stand_height = stand_top - stand_bottom

    # Frame traseiro primeiro; a tela fica levemente a frente dele.
    objs.append(_box(
        f"{name}_frame_back", htm,
        [0, thickness * 0.18, screen_bottom + height / 2],
        width, thickness * 0.55, height, frame_color
    ))

    objs.append(_box(
        f"{name}_screen", htm,
        [0, -thickness * 0.12, screen_bottom + height / 2],
        width * 0.94, thickness * 0.38, height * 0.90, screen_color
    ))

    objs.append(_box(
        f"{name}_stand", htm,
        [0, thickness * 0.10, stand_bottom + effective_stand_height / 2],
        stand_width, stand_depth, effective_stand_height, accent_color
    ))

    objs.append(_box(
        f"{name}_base", htm,
        [0, 0, base_height / 2],
        base_width, 0.18, base_height, accent_color
    ))

    return objs


def create_computer_setup(
    htm=None,
    name="computer_setup",
    dual_monitor=False,
    include_speakers=True,
    include_webcam=False,
    include_mousepad=True,
    include_tower=True,
    monitor_width=0.52,
    keyboard_width=0.44,
    desk_depth_hint=0.70,
    tower_side="right",
):
    """
    Setup de computador para ser colocado sobre uma mesa.

    A origem desse setup deve ser interpretada como o centro da area ocupada
    sobre o tampo da mesa. Os offsets sao calculados a partir das dimensoes
    dos objetos, evitando sobreposicao quando o setup muda de tamanho.
    """
    objs = []

    if tower_side not in ("left", "right"):
        raise ValueError("tower_side must be 'left' or 'right'.")

    monitor_y = 0.12
    monitor_height = 0.34
    monitor_screen_bottom = 0.18
    monitor_gap = 0.045

    tower_width = 0.22
    tower_depth = 0.46
    tower_gap = 0.055

    speaker_width = 0.085
    speaker_height = 0.15
    speaker_gap = 0.035

    # -------------------------------------------------------------------------
    # Monitor(es)
    # -------------------------------------------------------------------------
    if dual_monitor:
        # Distancia entre centros = uma largura inteira + uma pequena folga.
        # Assim os dois monitores nunca se atravessam.
        mon_spacing = monitor_width + monitor_gap
        monitor_centers = (-mon_spacing / 2, mon_spacing / 2)

        for side, x in zip(("left", "right"), monitor_centers):
            objs += create_monitor_prop(
                htm=_pose(htm, [x, monitor_y, 0]),
                name=f"{name}_monitor_{side}",
                width=monitor_width,
                height=monitor_height,
                screen_bottom=monitor_screen_bottom,
            )

        monitor_half_span = mon_spacing / 2 + monitor_width / 2
    else:
        monitor_centers = (0.0,)
        objs += create_monitor_prop(
            htm=_pose(htm, [0, monitor_y, 0]),
            name=f"{name}_monitor",
            width=monitor_width,
            height=monitor_height,
            screen_bottom=monitor_screen_bottom,
        )
        monitor_half_span = monitor_width / 2

    monitor_top = monitor_screen_bottom + monitor_height
    webcam_target = _pose(htm, [0, monitor_y, monitor_top + 0.005])

    # -------------------------------------------------------------------------
    # Teclado
    # -------------------------------------------------------------------------
    objs += create_keyboard(
        htm=_pose(htm, [0, -0.12, 0]),
        name=f"{name}_keyboard",
        width=keyboard_width,
    )

    # Mouse + mousepad
    mousepad_x = keyboard_width / 2 + 0.14
    if include_mousepad:
        objs += create_mousepad(
            htm=_pose(htm, [mousepad_x, -0.11, 0]),
            name=f"{name}_mousepad",
        )
        objs += create_mouse(
            htm=_pose(htm, [mousepad_x, -0.11, 0.005]),
            name=f"{name}_mouse",
        )
    else:
        objs += create_mouse(
            htm=_pose(htm, [mousepad_x, -0.11, 0]),
            name=f"{name}_mouse",
        )

    # -------------------------------------------------------------------------
    # Caixas de som
    # -------------------------------------------------------------------------
    if include_speakers:
        # As caixas ficam fora do envelope horizontal dos monitores, em vez de
        # serem posicionadas por um multiplicador arbitrario da largura.
        speaker_x = monitor_half_span + speaker_gap + speaker_width / 2

        objs += create_small_speaker(
            htm=_pose(htm, [-speaker_x, monitor_y, 0]),
            name=f"{name}_speakers_left",
            width=speaker_width,
            height=speaker_height,
        )
        objs += create_small_speaker(
            htm=_pose(htm, [speaker_x, monitor_y, 0]),
            name=f"{name}_speakers_right",
            width=speaker_width,
            height=speaker_height,
        )

    # Webcam
    if include_webcam:
        objs += create_webcam(
            htm=webcam_target,
            name=f"{name}_webcam",
        )

    # -------------------------------------------------------------------------
    # Gabinete
    # -------------------------------------------------------------------------
    if include_tower:
        # Coloca o gabinete completamente para fora do envelope dos monitores.
        # Se houver caixas no mesmo lado, reserva tambem o espaco delas.
        occupied_half_span = monitor_half_span
        if include_speakers:
            occupied_half_span += speaker_gap + speaker_width

        tower_x_abs = occupied_half_span + tower_gap + tower_width / 2
        tower_x = tower_x_abs if tower_side == "right" else -tower_x_abs

        # Mantem o gabinete na regiao traseira da mesa, mas sem depender de um
        # numero magico que muda de comportamento com o tamanho do tampo.
        max_rear_y = max(0.0, desk_depth_hint / 2 - tower_depth / 2 - 0.025)
        tower_y = min(monitor_y, max_rear_y)

        objs += create_desktop_tower(
            htm=_pose(htm, [tower_x, tower_y, 0]),
            name=f"{name}_tower",
            width=tower_width,
            depth=tower_depth,
        )

    return objs


# =============================================================================
# Objetos cotidianos
# =============================================================================

def create_mug(
    htm=None,
    name="mug",
    radius=0.04,
    height=0.095,
    color="#EDEDED",
    handle_color="#EDEDED",
):
    """Caneca simples."""
    objs = []

    objs.append(_cyl_z(
        f"{name}_body", htm,
        [0, 0, height / 2],
        radius, height, color
    ))

    objs.append(_cyl_y(
        f"{name}_handle", htm,
        [radius * 1.05, 0, height * 0.55],
        radius * 0.18, radius * 0.70, handle_color
    ))

    return objs


def create_box_package(
    htm=None,
    name="box_package",
    width=0.34,
    depth=0.26,
    height=0.24,
    color="#B08B5B",
    tape_color="#E8D9B5",
):
    """Caixa de papelao."""
    objs = []

    objs.append(_box(
        f"{name}_body", htm,
        [0, 0, height / 2],
        width, depth, height, color
    ))

    objs.append(_box(
        f"{name}_tape_x", htm,
        [0, 0, height + 0.001],
        width * 0.14, depth, 0.004, tape_color
    ))

    objs.append(_box(
        f"{name}_tape_y", htm,
        [0, 0, height + 0.002],
        width, depth * 0.14, 0.004, tape_color
    ))

    return objs


def create_desk_lamp(
    htm=None,
    name="desk_lamp",
    base_radius=0.08,
    base_thickness=0.02,
    stem_radius=0.012,
    stem_height=0.26,
    head_radius=0.05,
    head_length=0.12,
    base_color="#303030",
    shade_color="#D8C8A8",
):
    """Luminaria de mesa simplificada."""
    objs = []

    objs.append(_cyl_z(
        f"{name}_base", htm,
        [0, 0, base_thickness / 2],
        base_radius, base_thickness, base_color
    ))

    objs.append(_cyl_z(
        f"{name}_stem", htm,
        [0, 0, base_thickness + stem_height / 2],
        stem_radius, stem_height, base_color
    ))

    objs.append(_cyl_x(
        f"{name}_head", htm,
        [head_length / 2, 0, base_thickness + stem_height],
        head_radius, head_length, shade_color
    ))

    return objs


def create_router(
    htm=None,
    name="router",
    width=0.18,
    depth=0.11,
    height=0.028,
    antenna_height=0.12,
    color="#F4F4F4",
    antenna_color="#333333",
):
    """Roteador Wi-Fi simples."""
    objs = []

    objs.append(_box(
        f"{name}_body", htm,
        [0, 0, height / 2],
        width, depth, height, color
    ))

    ax = width * 0.30
    ay = depth * 0.20

    for i, (x, y) in enumerate([
        (-ax, ay), (ax, ay)
    ], 1):
        objs.append(_cyl_z(
            f"{name}_antenna_{i}", htm,
            [x, y, height + antenna_height / 2],
            0.004, antenna_height, antenna_color
        ))

    return objs


# =============================================================================
# Composicoes uteis
# =============================================================================

def create_study_accessories(
    htm=None,
    name="study_accessories",
    include_notebook=True,
    include_mug=True,
    include_pen_holder=True,
    include_lamp=True,
):
    """Conjunto de pequenos acessorios de estudo/escritorio."""
    objs = []

    if include_notebook:
        objs += create_notebook(
            htm=_pose(htm, [-0.16, 0.02, 0]),
            name=f"{name}_notebook",
        )

    if include_mug:
        objs += create_mug(
            htm=_pose(htm, [0.18, 0.04, 0]),
            name=f"{name}_mug",
        )

    if include_pen_holder:
        objs += create_pen_holder(
            htm=_pose(htm, [0.24, -0.12, 0]),
            name=f"{name}_pen_holder",
        )

    if include_lamp:
        objs += create_desk_lamp(
            htm=_pose(htm, [-0.28, 0.08, 0]),
            name=f"{name}_lamp",
        )

    return objs


def create_tabletop_clutter(
    htm=None,
    name="tabletop_clutter",
    include_books=True,
    include_box=False,
    include_router=False,
):
    """Conjunto generico para preencher uma mesa, rack ou bancada."""
    objs = []

    if include_books:
        objs += create_book_stack(
            htm=_pose(htm, [-0.18, 0.02, 0]),
            name=f"{name}_book_stack",
            n_books=4,
        )
        objs += create_notebook(
            htm=_pose(htm, [0.08, -0.10, 0]),
            name=f"{name}_notebook",
        )

    objs += create_mug(
        htm=_pose(htm, [0.20, 0.10, 0]),
        name=f"{name}_mug",
    )

    if include_box:
        objs += create_box_package(
            htm=_pose(htm, [0.35, 0.00, 0]),
            name=f"{name}_box",
            width=0.20, depth=0.16, height=0.14,
        )

    if include_router:
        objs += create_router(
            htm=_pose(htm, [0.00, 0.14, 0]),
            name=f"{name}_router",
        )

    return objs


# =============================================================================
# Catalogo
# =============================================================================

PROP_CATALOG = {
    "book": create_book,
    "book_flat": create_book_flat,
    "book_row": create_book_row,
    "book_stack": create_book_stack,
    "notebook": create_notebook,
    "pen": create_pen,
    "pen_holder": create_pen_holder,

    "keyboard": create_keyboard,
    "mouse": create_mouse,
    "mousepad": create_mousepad,
    "desktop_tower": create_desktop_tower,
    "speaker": create_small_speaker,
    "speaker_pair": create_speaker_pair,
    "webcam": create_webcam,
    "monitor_prop": create_monitor_prop,
    "computer_setup": create_computer_setup,

    "mug": create_mug,
    "box_package": create_box_package,
    "desk_lamp": create_desk_lamp,
    "router": create_router,

    "study_accessories": create_study_accessories,
    "tabletop_clutter": create_tabletop_clutter,
}


def available_props():
    """Retorna os nomes das funcoes do catalogo."""
    return sorted(PROP_CATALOG.keys())


def create_prop(prop_type, **kwargs):
    """Cria um prop pelo nome do catalogo."""
    if prop_type not in PROP_CATALOG:
        raise ValueError(
            f"Unknown prop '{prop_type}'. "
            f"Available props: {', '.join(available_props())}"
        )
    return PROP_CATALOG[prop_type](**kwargs)


__all__ = [
    "create_book",
    "create_book_flat",
    "create_book_row",
    "create_book_stack",
    "create_notebook",
    "create_pen",
    "create_pen_holder",
    "create_keyboard",
    "create_mouse",
    "create_mousepad",
    "create_desktop_tower",
    "create_small_speaker",
    "create_speaker_pair",
    "create_webcam",
    "create_monitor_prop",
    "create_computer_setup",
    "create_mug",
    "create_box_package",
    "create_desk_lamp",
    "create_router",
    "create_study_accessories",
    "create_tabletop_clutter",
    "PROP_CATALOG",
    "available_props",
    "create_prop",
]
