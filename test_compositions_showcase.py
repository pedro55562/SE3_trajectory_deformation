"""
test_compositions_showcase.py
=============================

Showroom simples para validar visualmente todas as composicoes de
compositions.py.

Uso:
    python test_compositions_showcase.py
"""

import numpy as np
import uaibot as ub

from compositions import (
    create_bedside_corner,
    create_decorated_tv_stand,
    create_decorated_wall_shelf,
    create_filled_bookshelf,
    create_home_office_corner,
    create_pc_desk,
    create_reading_corner,
    create_study_desk,
)


def pose(x, y, z=0.0, yaw=0.0):
    return ub.Utils.trn([x, y, z]) @ ub.Utils.rotz(yaw)


def main():
    objects = []

    # Linha 1
    objects += create_filled_bookshelf(
        htm=pose(-5.2, 2.7),
        name="demo_filled_bookshelf",
    )

    objects += create_pc_desk(
        htm=pose(-2.2, 2.7),
        name="demo_pc_desk",
        dual_monitor=False,
        tower_side="right",
        include_chair=True,
    )

    objects += create_study_desk(
        htm=pose(1.2, 2.7),
        name="demo_study_desk",
    )

    objects += create_decorated_wall_shelf(
        htm=pose(4.2, 2.7),
        name="demo_wall_shelf",
    )

    # Linha 2
    objects += create_decorated_tv_stand(
        htm=pose(-4.3, -1.0),
        name="demo_tv_stand",
    )

    objects += create_reading_corner(
        htm=pose(-0.8, -1.0, yaw=np.pi / 12),
        name="demo_reading_corner",
    )

    objects += create_bedside_corner(
        htm=pose(2.6, -1.3),
        name="demo_bedside",
        nightstand_side="right",
    )

    # Linha 3: bloco maior
    objects += create_home_office_corner(
        htm=pose(-1.0, -5.5),
        name="demo_home_office",
        dual_monitor=False,
    )

    print(f"Total de primitivas: {len(objects)}")

    sim = ub.Simulation.create_sim_grid(objects)
    sim.run()


if __name__ == "__main__":
    main()
