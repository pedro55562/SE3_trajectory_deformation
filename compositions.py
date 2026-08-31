"""
compositions.py
===============

Composicoes prontas de mobiliario + props para montar cenarios mais ricos em
UAIbot sem precisar posicionar cada objeto manualmente.

A convencao segue furniture.py e props.py:
    x -> largura
    y -> profundidade
    z -> altura

A origem de cada composicao fica no chao. Objetos pequenos sao posicionados
sobre tampos e prateleiras usando as dimensoes reais do movel correspondente.

Exemplo:
    import uaibot as ub
    from compositions import create_pc_desk

    objs = create_pc_desk(
        htm=ub.Utils.trn([1.0, 0.0, 0.0]) @ ub.Utils.rotz(0.3),
        include_chair=True,
    )

    sim = ub.Simulation.create_sim_grid(objs)
    sim.run()
"""

import numpy as np
import uaibot as ub

from furniture import (
    create_armchair,
    create_bookshelf,
    create_floor_lamp,
    create_nightstand,
    create_office_chair,
    create_office_desk,
    create_potted_plant,
    create_rug,
    create_side_table,
    create_single_bed,
    create_trash_bin,
    create_tv,
    create_tv_stand,
    create_wall_shelf,
)
from props import (
    create_book_row,
    create_book_stack,
    create_box_package,
    create_computer_setup,
    create_desk_lamp,
    create_desktop_tower,
    create_mug,
    create_notebook,
    create_pen_holder,
    create_router,
    create_small_speaker,
)


# =============================================================================
# Helpers internos
# =============================================================================


def _base_htm(htm):
    """Retorna identidade quando htm=None."""
    return np.eye(4) if htm is None else htm


def _pose(htm, xyz, rotation=None):
    """Aplica uma pose local sobre a pose global da composicao."""
    T = _base_htm(htm) @ ub.Utils.trn(xyz)
    if rotation is not None:
        T = T @ rotation
    return T


def _side_sign(side):
    if side == "right":
        return 1.0
    if side == "left":
        return -1.0
    raise ValueError("side must be 'left' or 'right'.")


# =============================================================================
# Estantes e prateleiras preenchidas
# =============================================================================


def create_filled_bookshelf(
    htm=None,
    name="filled_bookshelf",
    width=0.90,
    depth=0.32,
    height=1.85,
    shelf_count=5,
    board_thickness=0.035,
    back_thickness=0.018,
    fill_ratio=0.82,
    include_book_stacks=True,
    include_boxes=True,
    include_notebooks=True,
    frame_color="#704D34",
    back_color="#5D402C",
):
    """
    Estante ja preenchida com livros e pequenos objetos.

    A quantidade e largura dos livros sao calculadas a partir da largura util
    de cada prateleira. O preenchimento e deterministico, mas varia de uma
    prateleira para outra para nao ficar visualmente repetitivo.
    """
    if shelf_count < 1:
        raise ValueError("'shelf_count' must be at least 1.")
    if not 0.25 <= fill_ratio <= 1.0:
        raise ValueError("'fill_ratio' must be between 0.25 and 1.0.")

    objs = []

    objs += create_bookshelf(
        htm=htm,
        name=f"{name}_frame",
        width=width,
        depth=depth,
        height=height,
        shelf_count=shelf_count,
        board_thickness=board_thickness,
        back_thickness=back_thickness,
        frame_color=frame_color,
        back_color=back_color,
    )

    # O create_bookshelf cria shelf_count+1 tabuas igualmente espacadas.
    levels = np.linspace(
        board_thickness / 2,
        height - board_thickness / 2,
        shelf_count + 1,
    )

    usable_width = width - 2 * board_thickness - 0.05
    bay_pitch = levels[1] - levels[0]
    usable_bay_height = bay_pitch - board_thickness

    # Livros ligeiramente mais baixos que o vao para nunca tocar a prateleira
    # superior. A largura e pequena para a fileira parecer uma estante real.
    book_height = min(0.25, usable_bay_height * 0.78)
    book_depth = min(0.038, depth * 0.16)
    book_width = np.clip(usable_width / 9.0, 0.07, 0.105)
    book_spacing = 0.005

    # Frente da estante e -y; os livros ficam proximos da borda frontal.
    books_y = -depth / 2 + book_depth / 2 + 0.018

    for shelf_idx in range(shelf_count):
        shelf_surface_z = levels[shelf_idx] + board_thickness / 2

        # Padroes alternados deixam a estante menos artificial.
        pattern = shelf_idx % 4

        if pattern == 0 and include_boxes:
            # Livros na esquerda + caixa pequena na direita.
            box_w = min(0.18, usable_width * 0.24)
            reserved = box_w + 0.055
            row_available = max(usable_width - reserved, book_width)
            n_books = max(
                2,
                int((row_available * fill_ratio + book_spacing) /
                    (book_width + book_spacing)),
            )
            row_span = n_books * book_width + (n_books - 1) * book_spacing
            row_x = -usable_width / 2 + row_span / 2

            objs += create_book_row(
                htm=_pose(htm, [row_x, books_y, shelf_surface_z]),
                name=f"{name}_shelf_{shelf_idx+1}_books",
                n_books=n_books,
                book_width=book_width,
                book_depth=book_depth,
                book_height=book_height,
                spacing=book_spacing,
                lean_angle=np.pi / 90,
            )

            box_x = usable_width / 2 - box_w / 2
            objs += create_box_package(
                htm=_pose(htm, [box_x, books_y + 0.015, shelf_surface_z]),
                name=f"{name}_shelf_{shelf_idx+1}_box",
                width=box_w,
                depth=min(0.15, depth * 0.58),
                height=min(0.13, usable_bay_height * 0.42),
            )

        elif pattern == 1 and include_book_stacks:
            # Fileira na esquerda + pequena pilha deitada na direita.
            stack_w = min(0.20, usable_width * 0.28)
            reserved = stack_w + 0.045
            row_available = max(usable_width - reserved, book_width)
            n_books = max(
                2,
                int((row_available * min(fill_ratio + 0.08, 1.0) + book_spacing) /
                    (book_width + book_spacing)),
            )
            row_span = n_books * book_width + (n_books - 1) * book_spacing
            row_x = -usable_width / 2 + row_span / 2

            objs += create_book_row(
                htm=_pose(htm, [row_x, books_y, shelf_surface_z]),
                name=f"{name}_shelf_{shelf_idx+1}_books",
                n_books=n_books,
                book_width=book_width,
                book_depth=book_depth,
                book_height=book_height * 0.96,
                spacing=book_spacing,
                lean_angle=np.pi / 75,
            )

            stack_x = usable_width / 2 - stack_w / 2
            objs += create_book_stack(
                htm=_pose(htm, [stack_x, books_y + 0.018, shelf_surface_z]),
                name=f"{name}_shelf_{shelf_idx+1}_stack",
                n_books=4,
                width=stack_w,
                depth=min(0.14, depth * 0.50),
                thickness=0.022,
                spacing=0.002,
                rotation_step=np.pi / 65,
            )

        else:
            # Fileira quase cheia, deslocada levemente para variar o visual.
            target_width = usable_width * min(
                0.97,
                fill_ratio + 0.05 * (shelf_idx % 3),
            )
            n_books = max(
                3,
                int((target_width + book_spacing) /
                    (book_width + book_spacing)),
            )
            row_span = n_books * book_width + (n_books - 1) * book_spacing
            add_notebook = (
                include_notebooks
                and pattern == 3
                and row_span < usable_width - 0.16
            )

            if add_notebook:
                # Reserva explicitamente o lado direito para o caderno.
                x_shift = -usable_width / 2 + row_span / 2
            else:
                x_shift = 0.0
                if row_span < usable_width - 0.04:
                    x_shift = (-1 if shelf_idx % 2 == 0 else 1) * 0.025

            objs += create_book_row(
                htm=_pose(htm, [x_shift, books_y, shelf_surface_z]),
                name=f"{name}_shelf_{shelf_idx+1}_books",
                n_books=n_books,
                book_width=book_width,
                book_depth=book_depth,
                book_height=book_height * (0.92 + 0.03 * (shelf_idx % 3)),
                spacing=book_spacing,
                lean_angle=np.pi / 100,
            )

            # Em algumas prateleiras, um caderno deitado ocupa o pequeno espaco
            # livre sem atravessar a fileira.
            if add_notebook:
                notebook_x = usable_width / 2 - 0.075
                objs += create_notebook(
                    htm=_pose(
                        htm,
                        [notebook_x, books_y + 0.015, shelf_surface_z],
                        ub.Utils.rotz(np.pi / 18),
                    ),
                    name=f"{name}_shelf_{shelf_idx+1}_notebook",
                    width=0.14,
                    depth=0.105,
                    thickness=0.012,
                )

    return objs



def create_decorated_wall_shelf(
    htm=None,
    name="decorated_wall_shelf",
    width=1.00,
    depth=0.24,
    thickness=0.035,
    height_from_floor=1.30,
    include_books=True,
    include_mug=True,
    include_box=True,
):
    """Prateleira de parede preenchida com livros e pequenos objetos."""
    objs = []

    objs += create_wall_shelf(
        htm=htm,
        name=f"{name}_shelf",
        width=width,
        depth=depth,
        thickness=thickness,
        height_from_floor=height_from_floor,
    )

    surface_z = height_from_floor + thickness / 2
    usable_width = width - 0.08

    if include_books:
        n_books = max(3, int(usable_width * 0.48 / 0.085))
        bw = 0.078
        spacing = 0.004
        row_span = n_books * bw + (n_books - 1) * spacing
        x = -usable_width / 2 + row_span / 2
        objs += create_book_row(
            htm=_pose(htm, [x, -0.055, surface_z]),
            name=f"{name}_books",
            n_books=n_books,
            book_width=bw,
            book_depth=0.032,
            book_height=0.21,
            spacing=spacing,
            lean_angle=np.pi / 90,
        )

    if include_box:
        objs += create_box_package(
            htm=_pose(htm, [width * 0.22, -0.045, surface_z]),
            name=f"{name}_box",
            width=0.16,
            depth=0.12,
            height=0.11,
        )

    if include_mug:
        objs += create_mug(
            htm=_pose(htm, [width * 0.39, -0.05, surface_z]),
            name=f"{name}_mug",
            radius=0.035,
            height=0.085,
        )

    return objs


# =============================================================================
# Escritorio / estudo
# =============================================================================


def create_pc_desk(
    htm=None,
    name="pc_desk",
    width=1.40,
    depth=0.70,
    height=0.75,
    top_thickness=0.04,
    dual_monitor=False,
    tower_side="right",
    monitor_width=0.46,
    include_speakers=True,
    include_webcam=False,
    include_book_stack=True,
    include_mug=True,
    include_pen_holder=False,
    include_chair=True,
    tower_on_desk=None,
):
    """
    Mesa completa com computador, teclado, mouse e detalhes de escritorio.

    A pilha de livros fica no canto oposto ao gabinete. Os offsets respeitam
    largura/profundidade do tampo para manter todos os objetos sobre a mesa.
    """
    tower_sign = _side_sign(tower_side)
    objs = []

    # Em setups duplos, monitor + caixas + gabinete ocupam largura demais para
    # uma mesa comum. Por padrao o gabinete vai para baixo do tampo nesses
    # casos. Em monitor unico ele permanece sobre a mesa.
    if tower_on_desk is None:
        tower_on_desk = not dual_monitor

    objs += create_office_desk(
        htm=htm,
        name=f"{name}_desk",
        width=width,
        depth=depth,
        height=height,
        top_thickness=top_thickness,
    )

    # O topo da mesa termina exatamente em z=height.
    surface_z = height

    # Limita apenas quando necessario para manter o conjunto dentro do tampo.
    # As constantes abaixo sao as mesmas folgas/dimensoes usadas em
    # create_computer_setup.
    desk_half_width = width / 2
    side_margin = 0.025
    speaker_extra = 0.12 if include_speakers else 0.0

    if dual_monitor:
        # Para dois monitores, o semi-envelope e aproximadamente w + 0.0225.
        top_extra = 0.0225 + speaker_extra
        if tower_on_desk:
            top_extra += 0.055 + 0.22
        max_monitor_width = desk_half_width - side_margin - top_extra
    else:
        # Para um monitor, o semi-envelope e w/2.
        top_extra = speaker_extra
        if tower_on_desk:
            top_extra += 0.055 + 0.22
        max_monitor_width = 2 * (desk_half_width - side_margin - top_extra)

    effective_monitor_width = max(0.30, min(monitor_width, max_monitor_width))

    objs += create_computer_setup(
        htm=_pose(htm, [0.0, 0.0, surface_z]),
        name=f"{name}_computer",
        dual_monitor=dual_monitor,
        include_speakers=include_speakers,
        include_webcam=include_webcam,
        include_mousepad=True,
        include_tower=tower_on_desk,
        monitor_width=effective_monitor_width,
        keyboard_width=min(0.44, width * 0.34),
        desk_depth_hint=depth,
        tower_side=tower_side,
    )

    if not tower_on_desk:
        # Gabinete sob a mesa: afastado das pernas laterais e centralizado em y.
        tower_width = 0.22
        tower_depth = min(0.46, depth * 0.72)
        tower_x = tower_sign * (width / 2 - 0.25)
        objs += create_desktop_tower(
            htm=_pose(htm, [tower_x, 0.0, 0.0]),
            name=f"{name}_computer_tower",
            width=tower_width,
            depth=tower_depth,
            height=min(0.46, height - top_thickness - 0.08),
        )

    # Pilha no canto frontal oposto ao gabinete.
    accessory_x = -tower_sign * (width / 2 - 0.14)
    accessory_y = -depth / 2 + 0.11

    if include_book_stack:
        objs += create_book_stack(
            htm=_pose(
                htm,
                [accessory_x, accessory_y, surface_z],
                ub.Utils.rotz(-tower_sign * np.pi / 18),
            ),
            name=f"{name}_book_stack",
            n_books=5,
            width=min(0.22, width * 0.16),
            depth=min(0.15, depth * 0.24),
            thickness=0.021,
            spacing=0.002,
            rotation_step=np.pi / 55,
        )

    if include_mug:
        mug_x = accessory_x + tower_sign * 0.20
        mug_y = accessory_y + 0.015
        objs += create_mug(
            htm=_pose(htm, [mug_x, mug_y, surface_z]),
            name=f"{name}_mug",
            radius=0.035,
            height=0.085,
        )

    if include_pen_holder:
        pen_x = -tower_sign * (width / 2 - 0.07)
        pen_y = depth / 2 - 0.08
        objs += create_pen_holder(
            htm=_pose(htm, [pen_x, pen_y, surface_z]),
            name=f"{name}_pen_holder",
            cup_radius=0.03,
            cup_height=0.10,
        )

    if include_chair:
        # A frente da mesa e -y. Giramos a cadeira para que o encosto fique
        # para fora e o assento fique voltado para a mesa.
        chair_y = -depth / 2 - 0.42
        objs += create_office_chair(
            htm=_pose(htm, [0.0, chair_y, 0.0], ub.Utils.rotz(np.pi)),
            name=f"{name}_chair",
            seat_width=0.48,
            seat_depth=0.46,
        )

    return objs



def create_study_desk(
    htm=None,
    name="study_desk",
    width=1.25,
    depth=0.62,
    height=0.75,
    top_thickness=0.04,
    include_lamp=True,
    include_chair=True,
):
    """Mesa de estudo sem PC, com livros, caderno, caneca e papelaria."""
    objs = []

    objs += create_office_desk(
        htm=htm,
        name=f"{name}_desk",
        width=width,
        depth=depth,
        height=height,
        top_thickness=top_thickness,
    )

    z = height

    objs += create_book_stack(
        htm=_pose(htm, [-width * 0.34, depth * 0.16, z], ub.Utils.rotz(-0.12)),
        name=f"{name}_books",
        n_books=6,
        width=0.22,
        depth=0.15,
        thickness=0.022,
    )

    objs += create_notebook(
        htm=_pose(htm, [0.02, -depth * 0.12, z], ub.Utils.rotz(np.pi / 16)),
        name=f"{name}_notebook",
        width=0.24,
        depth=0.17,
    )

    objs += create_pen_holder(
        htm=_pose(htm, [width * 0.34, depth * 0.20, z]),
        name=f"{name}_pens",
        cup_radius=0.032,
        cup_height=0.105,
    )

    objs += create_mug(
        htm=_pose(htm, [width * 0.30, -depth * 0.16, z]),
        name=f"{name}_mug",
        radius=0.037,
        height=0.09,
    )

    if include_lamp:
        objs += create_desk_lamp(
            htm=_pose(htm, [-width * 0.38, -depth * 0.18, z]),
            name=f"{name}_lamp",
            base_radius=0.065,
            stem_height=0.23,
            head_radius=0.045,
            head_length=0.10,
        )

    if include_chair:
        objs += create_office_chair(
            htm=_pose(
                htm,
                [0.0, -depth / 2 - 0.40, 0.0],
                ub.Utils.rotz(np.pi),
            ),
            name=f"{name}_chair",
            seat_width=0.47,
            seat_depth=0.45,
        )

    return objs


# =============================================================================
# Sala / entretenimento
# =============================================================================


def create_decorated_tv_stand(
    htm=None,
    name="decorated_tv_stand",
    width=1.60,
    depth=0.42,
    height=0.48,
    body_height=0.32,
    leg_height=0.12,
    center_opening_width=0.55,
    tv_width=1.04,
    tv_height=0.60,
    include_speakers=True,
    include_books=True,
    include_router=True,
    include_box=True,
):
    """
    Rack completo com TV em cima e objetos dentro do nicho central.

    A base da TV e recalculada para repousar exatamente sobre o tampo do rack,
    em vez de reutilizar o bottom_height padrao pensado para uma TV no chao.
    """
    objs = []

    objs += create_tv_stand(
        htm=htm,
        name=f"{name}_rack",
        width=width,
        depth=depth,
        height=height,
        body_height=body_height,
        leg_height=leg_height,
        center_opening_width=center_opening_width,
    )

    rack_top_z = height

    # create_tv posiciona a base do pedestal em bottom_height-stand_height.
    # Com a pequena correcao de meia espessura, a base encosta em z=0 local.
    tv_pedestal_height = 0.12
    tv_base_thickness = 0.025
    tv_bottom_height = tv_pedestal_height + tv_base_thickness / 2

    objs += create_tv(
        htm=_pose(htm, [0.0, 0.015, rack_top_z]),
        name=f"{name}_tv",
        width=min(tv_width, width * 0.72),
        height=tv_height,
        bottom_height=tv_bottom_height,
        stand_height=tv_pedestal_height,
        stand_width=min(0.34, width * 0.24),
    )

    if include_speakers:
        speaker_x = width / 2 - 0.10
        for side, sx in (("left", -speaker_x), ("right", speaker_x)):
            objs += create_small_speaker(
                htm=_pose(htm, [sx, -0.02, rack_top_z]),
                name=f"{name}_speaker_{side}",
                width=0.075,
                depth=0.085,
                height=0.13,
            )

    # O fundo do nicho e o topo da tabua inferior do rack.
    niche_floor_z = leg_height + 0.05
    niche_front_y = -depth / 2 + 0.095

    if include_books:
        objs += create_book_stack(
            htm=_pose(htm, [-center_opening_width * 0.24, niche_front_y, niche_floor_z]),
            name=f"{name}_niche_books",
            n_books=4,
            width=min(0.20, center_opening_width * 0.38),
            depth=min(0.13, depth * 0.42),
            thickness=0.020,
            spacing=0.002,
            rotation_step=np.pi / 75,
        )

    if include_router:
        objs += create_router(
            htm=_pose(htm, [center_opening_width * 0.24, niche_front_y, niche_floor_z]),
            name=f"{name}_router",
            width=min(0.16, center_opening_width * 0.30),
            depth=0.095,
            height=0.024,
            antenna_height=0.095,
        )

    if include_box:
        # Caixa menor e mais ao fundo, deslocada para nao cobrir livros/roteador.
        objs += create_box_package(
            htm=_pose(htm, [0.0, depth * 0.18, niche_floor_z]),
            name=f"{name}_niche_box",
            width=min(0.16, center_opening_width * 0.30),
            depth=0.11,
            height=0.09,
        )

    return objs


# =============================================================================
# Cantos completos de ambiente
# =============================================================================


def create_reading_corner(
    htm=None,
    name="reading_corner",
    include_rug=True,
    include_plant=True,
):
    """Poltrona + mesa lateral + luminaria + livros, caneca e decoracao."""
    objs = []

    if include_rug:
        objs += create_rug(
            htm=_pose(htm, [0.25, -0.05, 0.0]),
            name=f"{name}_rug",
            width=1.75,
            depth=1.35,
        )

    objs += create_armchair(
        htm=_pose(htm, [0.0, 0.0, 0.012]),
        name=f"{name}_armchair",
        width=0.92,
        depth=0.86,
    )

    table_x = 0.72
    table_height = 0.52
    objs += create_side_table(
        htm=_pose(htm, [table_x, -0.02, 0.012]),
        name=f"{name}_side_table",
        width=0.46,
        depth=0.46,
        height=table_height,
    )

    objs += create_book_stack(
        htm=_pose(htm, [table_x - 0.07, -0.02, table_height + 0.012]),
        name=f"{name}_books",
        n_books=4,
        width=0.19,
        depth=0.13,
        thickness=0.021,
        rotation_step=np.pi / 70,
    )

    objs += create_mug(
        htm=_pose(htm, [table_x + 0.13, -0.08, table_height + 0.012]),
        name=f"{name}_mug",
        radius=0.034,
        height=0.085,
    )

    objs += create_floor_lamp(
        htm=_pose(htm, [-0.67, 0.18, 0.012]),
        name=f"{name}_floor_lamp",
        height=1.55,
        base_radius=0.14,
        shade_radius=0.17,
    )

    if include_plant:
        objs += create_potted_plant(
            htm=_pose(htm, [0.78, 0.55, 0.012]),
            name=f"{name}_plant",
            pot_radius=0.13,
            pot_height=0.21,
            plant_height=0.62,
            leaf_radius=0.085,
            leaf_count=7,
        )

    return objs



def create_bedside_corner(
    htm=None,
    name="bedside_corner",
    bed_width=0.96,
    bed_length=1.88,
    nightstand_side="right",
    include_lamp=True,
    include_mug=True,
):
    """Cama de solteiro com criado-mudo, livros e pequenos objetos."""
    side_sign = _side_sign(nightstand_side)
    objs = []

    objs += create_single_bed(
        htm=htm,
        name=f"{name}_bed",
        width=bed_width,
        length=bed_length,
    )

    nightstand_width = 0.44
    nightstand_depth = 0.38
    nightstand_height = 0.53
    ns_x = side_sign * (bed_width / 2 + nightstand_width / 2 + 0.08)
    ns_y = bed_length / 2 - nightstand_depth / 2 - 0.02

    objs += create_nightstand(
        htm=_pose(htm, [ns_x, ns_y, 0.0]),
        name=f"{name}_nightstand",
        width=nightstand_width,
        depth=nightstand_depth,
        height=nightstand_height,
        body_height=0.40,
        leg_height=0.13,
    )

    top_z = nightstand_height

    objs += create_book_stack(
        htm=_pose(htm, [ns_x - side_sign * 0.09, ns_y - 0.06, top_z]),
        name=f"{name}_books",
        n_books=3,
        width=0.17,
        depth=0.12,
        thickness=0.019,
        rotation_step=np.pi / 75,
    )

    if include_mug:
        objs += create_mug(
            htm=_pose(htm, [ns_x + side_sign * 0.13, ns_y - 0.08, top_z]),
            name=f"{name}_mug",
            radius=0.031,
            height=0.08,
        )

    if include_lamp:
        objs += create_desk_lamp(
            htm=_pose(htm, [ns_x + side_sign * 0.05, ns_y + 0.11, top_z]),
            name=f"{name}_lamp",
            base_radius=0.055,
            base_thickness=0.016,
            stem_height=0.20,
            head_radius=0.038,
            head_length=0.085,
        )

    return objs



def create_home_office_corner(
    htm=None,
    name="home_office_corner",
    dual_monitor=False,
    include_rug=True,
):
    """
    Pequeno home office pronto: mesa com PC, cadeira, estante cheia, planta e
    lixeira. Serve como bloco maior para cenarios internos.
    """
    objs = []

    if include_rug:
        objs += create_rug(
            htm=_pose(htm, [0.0, -0.20, 0.0]),
            name=f"{name}_rug",
            width=3.10,
            depth=2.05,
            thickness=0.010,
            color="#75685D",
        )
        z_floor = 0.010
    else:
        z_floor = 0.0

    objs += create_pc_desk(
        htm=_pose(htm, [0.0, 0.28, z_floor]),
        name=f"{name}_workstation",
        width=1.45,
        depth=0.72,
        height=0.75,
        dual_monitor=dual_monitor,
        tower_side="right",
        monitor_width=0.45 if dual_monitor else 0.48,
        include_speakers=True,
        include_webcam=False,
        include_book_stack=True,
        include_mug=True,
        include_pen_holder=True,
        include_chair=True,
    )

    # Estante ao lado esquerdo da mesa, levemente recuada.
    objs += create_filled_bookshelf(
        htm=_pose(htm, [-1.28, 0.35, z_floor]),
        name=f"{name}_bookshelf",
        width=0.84,
        depth=0.30,
        height=1.78,
        shelf_count=5,
        fill_ratio=0.78,
        include_book_stacks=True,
        include_boxes=True,
    )

    objs += create_potted_plant(
        htm=_pose(htm, [1.12, 0.48, z_floor]),
        name=f"{name}_plant",
        pot_radius=0.14,
        pot_height=0.22,
        plant_height=0.72,
        leaf_radius=0.09,
        leaf_count=8,
    )

    objs += create_trash_bin(
        htm=_pose(htm, [0.95, -0.52, z_floor]),
        name=f"{name}_trash",
        radius=0.12,
        height=0.28,
    )

    return objs


# =============================================================================
# Catalogo
# =============================================================================


COMPOSITION_CATALOG = {
    "filled_bookshelf": create_filled_bookshelf,
    "decorated_wall_shelf": create_decorated_wall_shelf,
    "pc_desk": create_pc_desk,
    "study_desk": create_study_desk,
    "decorated_tv_stand": create_decorated_tv_stand,
    "reading_corner": create_reading_corner,
    "bedside_corner": create_bedside_corner,
    "home_office_corner": create_home_office_corner,
}


def available_compositions():
    """Retorna os nomes das composicoes prontas."""
    return sorted(COMPOSITION_CATALOG.keys())


def create_composition(composition_type, **kwargs):
    """Cria uma composicao pelo nome do catalogo."""
    if composition_type not in COMPOSITION_CATALOG:
        raise ValueError(
            f"Unknown composition '{composition_type}'. "
            f"Available compositions: {', '.join(available_compositions())}"
        )
    return COMPOSITION_CATALOG[composition_type](**kwargs)


__all__ = [
    "create_filled_bookshelf",
    "create_decorated_wall_shelf",
    "create_pc_desk",
    "create_study_desk",
    "create_decorated_tv_stand",
    "create_reading_corner",
    "create_bedside_corner",
    "create_home_office_corner",
    "COMPOSITION_CATALOG",
    "available_compositions",
    "create_composition",
]
