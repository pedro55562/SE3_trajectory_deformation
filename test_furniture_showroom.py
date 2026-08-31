"""
test_furniture_showroom.py
==========================

Script de simulacao que cria DOIS itens de cada tipo definido em furniture.py
e mostra tudo em uma unica simulacao.

Uso:
    python test_furniture_showroom.py

Observacao:
    Este script assume que furniture.py esta no mesmo diretorio.
"""

import numpy as np
import uaibot as ub

from furniture import available_items, create_item


# =============================================================================
# Helpers
# =============================================================================

def make_pose(x, y, yaw=0.0):
    """Cria uma HTM no chao, com translacao em x,y e rotacao em torno de z."""
    return ub.Utils.trn([x, y, 0.0]) @ ub.Utils.rotz(yaw)


def extend_objects(dst, src):
    """Adiciona objetos de uma lista na outra."""
    dst.extend(src)


def default_kwargs_for(item_type):
    """
    Parametros opcionais por item.
    Em geral os defaults do furniture.py ja bastam; aqui colocamos ajustes
    leves para alguns objetos ficarem mais interessantes visualmente.
    """
    custom = {
        "office_desk": dict(width=1.40, depth=0.70, height=0.75),
        "dining_table": dict(width=1.80, depth=0.90, height=0.76),
        "coffee_table": dict(width=1.10, depth=0.60),
        "side_table": dict(width=0.50, depth=0.50),
        "round_table": dict(radius=0.55),

        "chair": dict(width=0.46, depth=0.48),
        "office_chair": dict(seat_width=0.50, seat_depth=0.48),
        "stool": dict(seat_radius=0.20),
        "bench": dict(width=1.30, depth=0.42),

        "bed": dict(width=1.38, length=1.88),
        "single_bed": dict(),
        "queen_bed": dict(),
        "nightstand": dict(width=0.48),
        "dresser": dict(width=1.05, rows=3),

        "sofa": dict(width=2.00, depth=0.88),
        "armchair": dict(width=0.90, depth=0.85),
        "tv_stand": dict(width=1.60),
        "tv": dict(width=1.10, height=0.64),
        "bookshelf": dict(width=0.90, height=1.85),
        "wall_shelf": dict(width=0.90, height_from_floor=1.30),

        "wardrobe": dict(width=1.45, height=2.00),
        "cabinet": dict(width=0.80, height=1.15),

        "kitchen_counter": dict(width=1.80, depth=0.62),
        "kitchen_island": dict(width=1.55, depth=0.88),
        "refrigerator": dict(width=0.72, height=1.82),
        "stove": dict(width=0.62),
        "washing_machine": dict(width=0.62),

        "filing_cabinet": dict(width=0.48, drawers=3),
        "monitor": dict(width=0.54, height=0.32),

        "floor_lamp": dict(height=1.62),
        "potted_plant": dict(plant_height=0.70),
        "rug": dict(width=1.80, depth=1.20),
        "trash_bin": dict(radius=0.16),
    }
    return custom.get(item_type, {}).copy()


def variant_kwargs_for(item_type, variant_idx):
    """
    Retorna pequenas variacoes para a segunda copia de cada item.
    variant_idx = 0 -> primeira copia
    variant_idx = 1 -> segunda copia
    """
    if variant_idx == 0:
        return {}

    custom = {
        "office_desk": dict(width=1.20, depth=0.60),
        "dining_table": dict(width=1.60, depth=0.85),
        "coffee_table": dict(width=0.95, depth=0.55),
        "side_table": dict(width=0.42, depth=0.42),
        "round_table": dict(radius=0.45),

        "chair": dict(width=0.42, depth=0.44),
        "office_chair": dict(seat_width=0.46, seat_depth=0.44),
        "stool": dict(seat_radius=0.17, height=0.45),
        "bench": dict(width=1.10, depth=0.38),

        "bed": dict(width=1.58, length=1.98),
        "single_bed": dict(width=0.88, length=1.88),
        "queen_bed": dict(width=1.68, length=2.00),
        "nightstand": dict(width=0.42, depth=0.36),
        "dresser": dict(width=0.95, rows=4),

        "sofa": dict(width=2.30, depth=0.92),
        "armchair": dict(width=1.00, depth=0.88),
        "tv_stand": dict(width=1.40),
        "tv": dict(width=0.90, height=0.52),
        "bookshelf": dict(width=1.10, shelf_count=6),
        "wall_shelf": dict(width=1.10, height_from_floor=1.50),

        "wardrobe": dict(width=1.65, height=2.10),
        "cabinet": dict(width=0.90, height=1.30),

        "kitchen_counter": dict(width=1.40, depth=0.60),
        "kitchen_island": dict(width=1.35, depth=0.78),
        "refrigerator": dict(width=0.80, height=1.90),
        "stove": dict(width=0.70),
        "washing_machine": dict(width=0.60, depth=0.60),

        "filing_cabinet": dict(width=0.44, drawers=4),
        "monitor": dict(width=0.62, height=0.36),

        "floor_lamp": dict(height=1.78),
        "potted_plant": dict(plant_height=0.95, leaf_count=9),
        "rug": dict(width=1.40, depth=0.90),
        "trash_bin": dict(radius=0.13, height=0.28),
    }
    return custom.get(item_type, {}).copy()


# =============================================================================
# Montagem do showroom
# =============================================================================

def create_showroom():
    item_types = available_items()

    # Layout:
    # para cada item, duas copias lado a lado
    # e cada "par" ocupa uma celula da grade principal.
    pairs_per_row = 4
    row_spacing = 4.2
    col_spacing = 6.0

    # separacao interna entre as duas copias do mesmo item
    internal_dx = 1.55

    objects = []

    for idx, item_type in enumerate(item_types):
        row = idx // pairs_per_row
        col = idx % pairs_per_row

        cx = col * col_spacing
        cy = -row * row_spacing

        # primeira copia
        kwargs1 = default_kwargs_for(item_type)
        kwargs1.update(variant_kwargs_for(item_type, 0))
        kwargs1["name"] = f"{item_type}_A"
        kwargs1["htm"] = make_pose(cx - internal_dx / 2, cy, yaw=0.0)

        # segunda copia
        kwargs2 = default_kwargs_for(item_type)
        kwargs2.update(variant_kwargs_for(item_type, 1))
        kwargs2["name"] = f"{item_type}_B"
        kwargs2["htm"] = make_pose(cx + internal_dx / 2, cy, yaw=np.pi / 6)

        try:
            extend_objects(objects, create_item(item_type, **kwargs1))
            extend_objects(objects, create_item(item_type, **kwargs2))
        except Exception as exc:
            print(f"[WARN] Falha ao criar '{item_type}': {exc}")

    return objects


# =============================================================================
# Main
# =============================================================================

def main():
    objects = create_showroom()

    print(f"Total de primitivas criadas: {len(objects)}")
    print(f"Total de tipos de item: {len(available_items())}")
    print(f"Total de itens compostos esperados: {2 * len(available_items())}")

    sim = ub.Simulation()
    sim.add(objects)
    sim.run()


if __name__ == "__main__":
    main()
