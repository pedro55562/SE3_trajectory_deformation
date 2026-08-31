"""
test_props_showcase.py
======================

Showroom para testar props.py.
Cria varios props e alguns conjuntos compostos em uma unica simulacao.

Uso:
    python test_props_showcase.py
"""

import numpy as np
import uaibot as ub

from props import (
    create_book,
    create_book_row,
    create_book_stack,
    create_notebook,
    create_pen_holder,
    create_keyboard,
    create_mouse,
    create_mousepad,
    create_desktop_tower,
    create_small_speaker,
    create_speaker_pair,
    create_monitor_prop,
    create_computer_setup,
    create_mug,
    create_box_package,
    create_desk_lamp,
    create_router,
    create_study_accessories,
    create_tabletop_clutter,
)


def pose(x, y, z=0.0, yaw=0.0):
    return ub.Utils.trn([x, y, z]) @ ub.Utils.rotz(yaw)


def main():
    objects = []

    # -------------------------------------------------------------------------
    # Linha 1 - livros
    # -------------------------------------------------------------------------
    objects += create_book(
        htm=pose(-4.5, 2.6, 0.0),
        name="book_single"
    )

    objects += create_book_row(
        htm=pose(-2.6, 2.6, 0.0),
        name="book_row_demo",
        n_books=12,
        lean_angle=np.pi / 70
    )

    objects += create_book_stack(
        htm=pose(0.0, 2.6, 0.0),
        name="book_stack_demo",
        n_books=6
    )

    objects += create_notebook(
        htm=pose(1.8, 2.6, 0.0, yaw=np.pi / 8),
        name="notebook_demo"
    )

    objects += create_pen_holder(
        htm=pose(3.4, 2.6, 0.0),
        name="pen_holder_demo"
    )

    # -------------------------------------------------------------------------
    # Linha 2 - periféricos
    # -------------------------------------------------------------------------
    objects += create_keyboard(
        htm=pose(-4.5, 0.6, 0.0),
        name="keyboard_demo"
    )

    objects += create_mousepad(
        htm=pose(-3.5, 0.6, 0.0),
        name="mousepad_demo"
    )
    objects += create_mouse(
        htm=pose(-3.5, 0.6, 0.006),
        name="mouse_demo"
    )

    objects += create_desktop_tower(
        htm=pose(-2.2, 0.6, 0.0),
        name="tower_demo"
    )

    objects += create_small_speaker(
        htm=pose(-0.9, 0.6, 0.0),
        name="speaker_demo"
    )

    objects += create_speaker_pair(
        htm=pose(0.7, 0.6, 0.0),
        name="speaker_pair_demo"
    )

    objects += create_monitor_prop(
        htm=pose(2.6, 0.6, 0.0),
        name="monitor_demo"
    )

    # -------------------------------------------------------------------------
    # Linha 3 - conjuntos compostos
    # -------------------------------------------------------------------------
    objects += create_computer_setup(
        htm=pose(-3.6, -2.0, 0.75),
        name="computer_setup_single",
        dual_monitor=False,
        include_speakers=True,
        include_webcam=True,
        include_mousepad=True,
        include_tower=True,
        tower_side="right",
    )

    objects += create_computer_setup(
        htm=pose(-0.5, -2.0, 0.75),
        name="computer_setup_dual",
        dual_monitor=True,
        include_speakers=True,
        include_webcam=False,
        include_mousepad=True,
        include_tower=True,
        tower_side="left",
    )

    objects += create_study_accessories(
        htm=pose(2.1, -2.0, 0.75),
        name="study_accessories_demo",
        include_notebook=True,
        include_mug=True,
        include_pen_holder=True,
        include_lamp=True,
    )

    objects += create_tabletop_clutter(
        htm=pose(4.5, -2.0, 0.75),
        name="tabletop_clutter_demo",
        include_books=True,
        include_box=True,
        include_router=True,
    )

    # -------------------------------------------------------------------------
    # Linha 4 - props avulsos cotidianos
    # -------------------------------------------------------------------------
    objects += create_mug(
        htm=pose(-4.0, -4.5, 0.0),
        name="mug_demo"
    )

    objects += create_box_package(
        htm=pose(-2.4, -4.5, 0.0),
        name="box_demo"
    )

    objects += create_desk_lamp(
        htm=pose(-0.6, -4.5, 0.0),
        name="lamp_demo"
    )

    objects += create_router(
        htm=pose(1.0, -4.5, 0.0),
        name="router_demo"
    )

    sim = ub.Simulation.create_sim_grid(objects)
    sim.run()


if __name__ == "__main__":
    main()
