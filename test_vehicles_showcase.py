"""
test_vehicles_showcase.py
=========================

Showroom simples para validar visualmente os veiculos de vehicles.py.

Uso:
    python test_vehicles_showcase.py

Este script assume que vehicles.py esta no mesmo diretorio.
"""

import numpy as np
import uaibot as ub

from vehicles import create_compact_crossover, create_work_pickup


def pose(x, y, z=0.0, yaw=0.0):
    """HTM no chao, com translacao e rotacao em torno de z."""
    return ub.Utils.trn([x, y, z]) @ ub.Utils.rotz(yaw)


def main():
    objects = []

    # -------------------------------------------------------------------------
    # Linha 1: pickups de trabalho
    # -------------------------------------------------------------------------
    objects += create_work_pickup(
        htm=pose(-2.8, 2.2, yaw=np.deg2rad(-8.0)),
        name="demo_pickup_blue",
        body_color="#315F7D",
        include_cargo=True,
        cargo_scale=1.00,
    )

    objects += create_work_pickup(
        htm=pose(3.0, 2.0, yaw=np.deg2rad(10.0)),
        name="demo_pickup_white",
        body_color="#D6D8D7",
        accent_color="#25282B",
        include_cargo=True,
        cargo_scale=0.84,
    )

    # -------------------------------------------------------------------------
    # Linha 2: crossovers compactos
    # -------------------------------------------------------------------------
    objects += create_compact_crossover(
        htm=pose(-2.2, -3.2, yaw=np.deg2rad(14.0)),
        name="demo_crossover_red",
        body_color="#963F37",
        include_roof_rails=True,
    )

    objects += create_compact_crossover(
        htm=pose(2.5, -3.1, yaw=np.deg2rad(-12.0)),
        name="demo_crossover_gray",
        body_color="#6B7075",
        accent_color="#1D2226",
        include_roof_rails=False,
    )

    print(f"Total de primitivas: {len(objects)}")
    print("Veiculos: 2 pickups + 2 crossovers")

    sim = ub.Simulation.create_sim_grid(objects)
    sim.run()


if __name__ == "__main__":
    main()
