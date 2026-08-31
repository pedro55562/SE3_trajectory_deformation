
import numpy as np
import uaibot as ub

from architecture import (
    create_wall,
    create_wall_with_door,
    create_wall_with_window,
    create_wall_with_door_and_window,
    create_double_door,
    create_bay_window,
    create_stairs,
    create_railing,
    create_garage_door,
    create_gable_roof,
    create_porch_steps,
)


def pose(x, y, z=0.0, yaw=0.0):
    return ub.Utils.trn([x, y, z]) @ ub.Utils.rotz(yaw)


def main():
    objs = []

    # Paredes / portas em diferentes estados
    objs += create_wall(
        htm=pose(-7, 4),
        name="wall_simple",
        length=3.2,
    )

    objs += create_wall_with_door(
        htm=pose(-3, 4),
        name="door_closed",
        length=3.4,
        include_door=True,
        door_angle=0.0,
    )

    objs += create_wall_with_door(
        htm=pose(1, 4),
        name="door_45",
        length=3.4,
        include_door=True,
        door_angle=np.pi / 4,
    )

    objs += create_wall_with_door(
        htm=pose(5, 4),
        name="door_90",
        length=3.4,
        include_door=True,
        door_angle=np.pi / 2,
    )

    # Janela
    objs += create_wall_with_window(
        htm=pose(-5.5, 0),
        name="wall_window",
        length=4.0,
        window_width=1.35,
        window_height=1.10,
        sill_height=0.88,
    )

    # Parede com porta e janela
    objs += create_wall_with_door_and_window(
        htm=pose(0, 0),
        name="combo",
        length=5.2,
        door_offset=-1.35,
        window_offset=1.15,
        door_angle=np.pi / 3,
    )

    # Porta dupla
    objs += create_double_door(
        htm=pose(5, 0),
        name="double_door",
        total_width=1.65,
        left_angle=np.pi / 3,
        right_angle=-np.pi / 3,
    )

    # Bay window
    objs += create_bay_window(
        htm=pose(-6, -4, 0.8),
        name="bay_window",
        width=1.8,
        height=1.15,
    )

    # Escada
    objs += create_stairs(
        htm=pose(-2, -4),
        name="stairs",
        width=1.05,
        total_run=3.0,
        total_rise=2.7,
        n_steps=15,
    )

    # Corrimao
    objs += create_railing(
        htm=pose(1.2, -4),
        name="railing",
        length=2.8,
    )

    # Garagem fechada / meio aberta
    objs += create_garage_door(
        htm=pose(-4.5, -8),
        name="garage_closed",
        width=4.2,
        open_fraction=0.0,
    )

    objs += create_garage_door(
        htm=pose(0.5, -8),
        name="garage_half",
        width=4.2,
        open_fraction=0.5,
    )

    # Degraus de entrada
    objs += create_porch_steps(
        htm=pose(4.2, -8),
        name="porch_steps",
    )

    # Telhado isolado para inspecao
    objs += create_gable_roof(
        htm=pose(8.0, -6.5),
        name="gable_roof",
        width=4.0,
        depth=5.0,
        base_height=1.2,
        pitch_angle=np.deg2rad(30),
    )
    
    objs += create_wall_with_door(
        htm=pose(5, 7),
        name="double_wall_door",
        length=4.0,
        door_width=1.65,
        include_door=True,
        double_door=True,
        door_angle=np.pi * 2/3,
    )
    sim = ub.Simulation.create_sim_grid(objs)
    sim.run()


if __name__ == "__main__":
    main()
