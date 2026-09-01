"""
l_house_uaibot.py
=================

Casa terrea grande em L para UAIbot.

A planta e declarada como dados antes de gerar os objetos 3D. As paredes sao
derivadas de pontos inicial/final; portas sao somente vaos, sem folhas.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

import uaibot as ub


PROJECT_DIR = Path(__file__).resolve().parent

from architecture import (
    available_architecture,
    create_floor,
    create_gable_roof,
    create_porch_steps,
    create_wall,
    create_wall_with_door,
    create_wall_with_door_and_window,
    create_wall_with_window,
)
from compositions import available_compositions, create_composition
from furniture import available_items, create_item
from props import available_props
from vehicles import available_vehicles, create_vehicle


EPS = 1e-6
WALL_HEIGHT = 2.80
WALL_THICKNESS = 0.14
DOOR_HEIGHT = 2.15
WINDOW_HEIGHT = 1.10
WINDOW_SILL = 0.90
FLOOR_THICKNESS = 0.06
OUTDOOR_THICKNESS = 0.04


FLOOR_COLORS = {
    "porcelanato_claro": "#CFC6B8",
    "madeira_quente": "#A97745",
    "ceramica_hidraulica": "#9DB0A3",
    "ceramica_cozinha": "#B6B9AF",
    "cimento_pavimentado": "#8F9694",
    "grama": "#3E8F49",
    "agua_piscina": "#42B7D7",
    "deck_convivio": "#B48152",
    "borda_piscina": "#D8D1C3",
}


INTERIOR_WALL_MARGIN = WALL_THICKNESS / 2 + 0.04
OUTDOOR_EDGE_MARGIN = 0.08
DOOR_CLEARANCE_DEPTH = 0.55
DOOR_CLEARANCE_SIDE = 0.22
WINDOW_CLEARANCE_DEPTH = 0.42
WINDOW_TALL_OBJECT_LIMIT = WINDOW_SILL + 0.02
MIN_ROOM_FREE_RATIO = 0.38
MIN_OBJECT_GAP = 0.03


FURNITURE_REGISTRY = {}


FURNITURE_LAYOUT = [
    {
        "id": "mesa_escritorio",
        "room": "escritorio",
        "center": [3.30, 3.45, 0.0],
        "dimensions": [1.40, 1.45, 0.95],
        "local_bounds": {"x": [-0.70, 0.70], "y": [-1.10, 0.35], "z": [0.0, 0.95]},
        "yaw": -math.pi / 2,
        "function": "compositions.create_composition",
        "composition_type": "pc_desk",
        "params": {"width": 1.40, "depth": 0.70, "include_book_stack": True, "include_chair": True},
        "clearance": {"wall": INTERIOR_WALL_MARGIN, "opening": DOOR_CLEARANCE_DEPTH, "between_objects": MIN_OBJECT_GAP},
    },
    {
        "id": "estante_escritorio",
        "room": "escritorio",
        "center": [0.62, 4.73, 0.0],
        "dimensions": [0.90, 0.32, 1.85],
        "yaw": 0.0,
        "function": "compositions.create_composition",
        "composition_type": "filled_bookshelf",
        "params": {"width": 0.90, "depth": 0.32, "height": 1.85},
        "clearance": {"wall": INTERIOR_WALL_MARGIN, "opening": DOOR_CLEARANCE_DEPTH, "between_objects": MIN_OBJECT_GAP},
    },
    {
        "id": "estante_escritorio2",
        "room": "escritorio",
        "center": [3.50, 2.2, 0.0],
        "dimensions": [0.90, 0.32, 1.85],
        "yaw": -np.pi/2,
        "function": "compositions.create_composition",
        "composition_type": "filled_bookshelf",
        "params": {"width": 0.90, "depth": 0.32, "height": 1.85},
        "clearance": {"wall": INTERIOR_WALL_MARGIN, "opening": DOOR_CLEARANCE_DEPTH, "between_objects": MIN_OBJECT_GAP},
    },
    {
        "id": "arquivo_escritorio",
        "room": "escritorio",
        "center": [3.42, 0.35, 0.0],
        "dimensions": [0.48, 0.56, 0.72],
        "yaw": - np.pi/2,
        "function": "furniture.create_item",
        "item_type": "filing_cabinet",
        "params": {"width": 0.48, "depth": 0.56, "height": 0.72},
        "clearance": {"wall": INTERIOR_WALL_MARGIN, "opening": DOOR_CLEARANCE_DEPTH, "between_objects": MIN_OBJECT_GAP},
    },
    {
        "id": "arquivo_escritorio2",
        "room": "escritorio",
        "center": [3.42, 0.83, 0.0],
        "dimensions": [0.48, 0.56, 0.72],
        "yaw": - np.pi/2,
        "function": "furniture.create_item",
        "item_type": "filing_cabinet",
        "params": {"width": 0.48, "depth": 0.56, "height": 0.72},
        "clearance": {"wall": INTERIOR_WALL_MARGIN, "opening": DOOR_CLEARANCE_DEPTH, "between_objects": MIN_OBJECT_GAP},
    },
    {
        "id": "arquivo_escritorio3",
        "room": "escritorio",
        "center": [3.42, 1.31, 0.0],
        "dimensions": [0.48, 0.56, 0.72],
        "yaw": - np.pi/2,
        "function": "furniture.create_item",
        "item_type": "filing_cabinet",
        "params": {"width": 0.48, "depth": 0.56, "height": 0.72},
        "clearance": {"wall": INTERIOR_WALL_MARGIN, "opening": DOOR_CLEARANCE_DEPTH, "between_objects": MIN_OBJECT_GAP},
    },

    {
        "id": "sofa_sala",
        "room": "sala",
        "center": [12.60, 2.62, 0.0],
        "dimensions": [2.60, 0.88, 0.88],
        "yaw": 0.0,
        "function": "furniture.create_item",
        "item_type": "sofa",
        "params": {"width": 2.60, "depth": 0.88},
        "clearance": {"wall": INTERIOR_WALL_MARGIN, "opening": DOOR_CLEARANCE_DEPTH, "between_objects": MIN_OBJECT_GAP},
    },
    {
        "id": "rack_tv_decorado_sala",
        "room": "sala",
        "center": [12.60, .36, 0.0],
        "dimensions": [1.60, 0.42, 1.25],
        "yaw": 0.0,
        "function": "compositions.create_composition",
        "composition_type": "decorated_tv_stand",
        "params": {"width": 1.60, "depth": 0.42, "height": 0.48},
        "clearance": {"wall": INTERIOR_WALL_MARGIN, "opening": DOOR_CLEARANCE_DEPTH, "between_objects": MIN_OBJECT_GAP},
    },
    {
        "id": "mesa_centro_sala",
        "room": "sala",
        "center": [12.60, 1.48, 0.0],
        "dimensions": [1.10, 0.60, 0.42],
        "yaw": 0.0,
        "function": "furniture.create_item",
        "item_type": "coffee_table",
        "params": {"width": 1.10, "depth": 0.60},
        "clearance": {"wall": INTERIOR_WALL_MARGIN, "opening": DOOR_CLEARANCE_DEPTH, "between_objects": MIN_OBJECT_GAP},
    },
    {
        "id": "poltrona_sala",
        "room": "sala",
        "center": [14.95, 1.48, 0.0],
        "dimensions": [0.90, 0.85, 0.88],
        "yaw": -math.pi / 2,
        "function": "furniture.create_item",
        "item_type": "armchair",
        "params": {"width": 0.90, "depth": 0.85},
        "clearance": {"wall": INTERIOR_WALL_MARGIN, "opening": DOOR_CLEARANCE_DEPTH, "between_objects": MIN_OBJECT_GAP},
    },
    {
        "id": "mesa_jantar",
        "room": "sala_de_jantar",
        "center": [10.20, 4.62, 0.0],
        "dimensions": [1.20, 1.20, 0.75],
        "yaw": 0.0,
        "function": "furniture.create_item",
        "item_type": "round_table",
        "params": {"radius": 0.60, "height": 0.75},
        "clearance": {"wall": INTERIOR_WALL_MARGIN, "opening": DOOR_CLEARANCE_DEPTH, "between_objects": MIN_OBJECT_GAP},
    },
    {
        "id": "cadeira_jantar_oeste",
        "room": "sala_de_jantar",
        "center": [9.6, 4.62, 0.0],
        "dimensions": [0.46, 0.48, 0.90],
        "yaw": math.pi / 2,
        "function": "furniture.create_item",
        "item_type": "chair",
        "params": {"width": 0.46, "depth": 0.48},
        "clearance": {"wall": INTERIOR_WALL_MARGIN, "opening": DOOR_CLEARANCE_DEPTH, "between_objects": MIN_OBJECT_GAP},
    },
    {
        "id": "cadeira_jantar_leste",
        "room": "sala_de_jantar",
        "center": [10.75, 4.62, 0.0],
        "dimensions": [0.46, 0.48, 0.90],
        "yaw": -math.pi / 2,
        "function": "furniture.create_item",
        "item_type": "chair",
        "params": {"width": 0.46, "depth": 0.48},
        "clearance": {"wall": INTERIOR_WALL_MARGIN, "opening": DOOR_CLEARANCE_DEPTH, "between_objects": MIN_OBJECT_GAP},
    },
    {
        "id": "cadeira_jantar_norte",
        "room": "sala_de_jantar",
        "center": [10.20, 5.25, 0.0],
        "dimensions": [0.46, 0.48, 0.90],
        "yaw": 0,
        "function": "furniture.create_item",
        "item_type": "chair",
        "params": {"width": 0.46, "depth": 0.48},
        "clearance": {"wall": INTERIOR_WALL_MARGIN, "opening": DOOR_CLEARANCE_DEPTH, "between_objects": MIN_OBJECT_GAP},
    },
    {
        "id": "cadeira_jantar_sul",
        "room": "sala_de_jantar",
        "center": [10.20, 4, 0.0],
        "dimensions": [0.46, 0.48, 0.90],
        "yaw": np.pi,
        "function": "furniture.create_item",
        "item_type": "chair",
        "params": {"width": 0.46, "depth": 0.48},
        "clearance": {"wall": INTERIOR_WALL_MARGIN, "opening": DOOR_CLEARANCE_DEPTH, "between_objects": MIN_OBJECT_GAP},
    },
    {
        "id": "geladeira_cozinha",
        "room": "cozinha",
        "center": [15.42, 3.78, 0.0],
        "dimensions": [0.72, 0.72, 1.82],
        "yaw": np.pi,
        "function": "furniture.create_item",
        "item_type": "refrigerator",
        "params": {"width": 0.72, "depth": 0.72, "height": 1.82},
        "clearance": {"wall": INTERIOR_WALL_MARGIN, "opening": DOOR_CLEARANCE_DEPTH, "between_objects": MIN_OBJECT_GAP},
    },
    {
        "id": "fogao_cozinha",
        "room": "cozinha",
        "center": [14.35, 3.73, 0.0],
        "dimensions": [0.62, 0.62, 0.90],
        "yaw": np.pi,
        "function": "furniture.create_item",
        "item_type": "stove",
        "params": {"width": 0.62, "depth": 0.62, "height": 0.90},
        "clearance": {"wall": INTERIOR_WALL_MARGIN, "opening": DOOR_CLEARANCE_DEPTH, "between_objects": MIN_OBJECT_GAP},
    },
    {
        "id": "bancada_pia_cozinha",
        "room": "cozinha",
        "center": [14.10, 5.56, 0.0],
        "dimensions": [1.80, 0.62, 0.90],
        "yaw": 0.0,
        "function": "furniture.create_item",
        "item_type": "kitchen_counter",
        "params": {"width": 1.80, "depth": 0.62, "height": 0.90},
        "can_be_under_window": True,
        "clearance": {"wall": INTERIOR_WALL_MARGIN, "opening": DOOR_CLEARANCE_DEPTH, "between_objects": MIN_OBJECT_GAP},
    },
    {
        "id": "armario_baixo_cozinha",
        "room": "cozinha",
        "center": [15.66, 5.00, 0.0],
        "dimensions": [0.80, 0.42, 0.86],
        "yaw": -math.pi / 2,
        "function": "furniture.create_item",
        "item_type": "cabinet",
        "params": {"width": 0.80, "depth": 0.42, "height": 0.86},
        "clearance": {"wall": INTERIOR_WALL_MARGIN, "opening": DOOR_CLEARANCE_DEPTH, "between_objects": MIN_OBJECT_GAP},
    },
    {
        "id": "cama_quarto_1",
        "room": "quarto_1",
        "center": [2.20, 9.82, 0.0],
        "dimensions": [1.58, 1.98, 1.05],
        "yaw": 0.0,
        "function": "furniture.create_item",
        "item_type": "queen_bed",
        "params": {},
        "clearance": {"wall": INTERIOR_WALL_MARGIN, "opening": DOOR_CLEARANCE_DEPTH, "between_objects": MIN_OBJECT_GAP},
    },
    {
        "id": "criado_quarto_1_oeste",
        "room": "quarto_1",
        "center": [1.12, 10.6, 0.0],
        "dimensions": [0.48, 0.40, 0.55],
        "yaw": 0.0,
        "function": "furniture.create_item",
        "item_type": "nightstand",
        "params": {"width": 0.48, "depth": 0.40, "height": 0.55},
        "clearance": {"wall": INTERIOR_WALL_MARGIN, "opening": DOOR_CLEARANCE_DEPTH, "between_objects": MIN_OBJECT_GAP},
    },
    {
        "id": "criado_quarto_1_leste",
        "room": "quarto_1",
        "center": [3.28, 10.6, 0.0],
        "dimensions": [0.48, 0.40, 0.55],
        "yaw": 0.0,
        "function": "furniture.create_item",
        "item_type": "nightstand",
        "params": {"width": 0.48, "depth": 0.40, "height": 0.55},
        "clearance": {"wall": INTERIOR_WALL_MARGIN, "opening": DOOR_CLEARANCE_DEPTH, "between_objects": MIN_OBJECT_GAP},
    },
    {
        "id": "guarda_roupa_quarto_1",
        "room": "quarto_1",
        "center": [4.98, 10.12, 0.0],
        "dimensions": [1.45, 0.58, 2.00],
        "yaw": - math.pi / 2,
        "function": "furniture.create_item",
        "item_type": "wardrobe",
        "params": {"width": 1.45, "depth": 0.58, "height": 2.00},
        "clearance": {"wall": INTERIOR_WALL_MARGIN, "opening": DOOR_CLEARANCE_DEPTH, "between_objects": MIN_OBJECT_GAP},
    },
    {
        "id": "comoda_quarto_1",
        "room": "quarto_1",
        "center": [1.20, 6.42, 0.0],
        "dimensions": [1.05, 0.48, 0.82],
        "yaw": np.pi,
        "function": "furniture.create_item",
        "item_type": "dresser",
        "params": {"width": 1.05, "depth": 0.48, "height": 0.82},
        "clearance": {"wall": INTERIOR_WALL_MARGIN, "opening": DOOR_CLEARANCE_DEPTH, "between_objects": MIN_OBJECT_GAP},
    },
    {
        "id": "mesa_pc_quarto_1",
        "room": "quarto_1",
        "center": [4.35, 6.5, 0.0],
        "dimensions": [1.40, 1.45, 0.95],
        "local_bounds": {"x": [-0.70, 0.70], "y": [-1.10, 0.35], "z": [0.0, 0.95]},
        "yaw": np.pi,
        "function": "compositions.create_composition",
        "composition_type": "pc_desk",
        "params": {"width": 1.40, "depth": 0.70, "include_book_stack": True, "include_chair": True},
        "clearance": {"wall": INTERIOR_WALL_MARGIN, "opening": DOOR_CLEARANCE_DEPTH, "between_objects": MIN_OBJECT_GAP},
    },
    {
        "id": "cama_quarto_2",
        "room": "quarto_2",
        "center": [2.18, 12.18, 0.0],
        "dimensions": [1.58, 1.98, 1.05],
        "yaw": math.pi,
        "function": "furniture.create_item",
        "item_type": "queen_bed",
        "params": {},
        "clearance": {"wall": INTERIOR_WALL_MARGIN, "opening": DOOR_CLEARANCE_DEPTH, "between_objects": MIN_OBJECT_GAP},
    },
    {
        "id": "criado_quarto_2_oeste",
        "room": "quarto_2",
        "center": [1.10, 11.35, 0.0],
        "dimensions": [0.48, 0.40, 0.55],
        "yaw":  np.pi,
        "function": "furniture.create_item",
        "item_type": "nightstand",
        "params": {"width": 0.48, "depth": 0.40, "height": 0.55},
        "clearance": {"wall": INTERIOR_WALL_MARGIN, "opening": DOOR_CLEARANCE_DEPTH, "between_objects": MIN_OBJECT_GAP},
    },
    {
        "id": "criado_quarto_2_leste",
        "room": "quarto_2",
        "center": [3.26, 11.35, 0.0],
        "dimensions": [0.48, 0.40, 0.55],
        "yaw":  np.pi,
        "function": "furniture.create_item",
        "item_type": "nightstand",
        "params": {"width": 0.48, "depth": 0.40, "height": 0.55},
        "clearance": {"wall": INTERIOR_WALL_MARGIN, "opening": DOOR_CLEARANCE_DEPTH, "between_objects": MIN_OBJECT_GAP},
    },
    {
        "id": "guarda_roupa_quarto_2",
        "room": "quarto_2",
        "center": [5.00, 15.12, 0.0],
        "dimensions": [1.45, 0.58, 2.00],
        "yaw": - math.pi / 2,
        "function": "furniture.create_item",
        "item_type": "wardrobe",
        "params": {"width": 1.45, "depth": 0.58, "height": 2.00},
        "clearance": {"wall": INTERIOR_WALL_MARGIN, "opening": DOOR_CLEARANCE_DEPTH, "between_objects": MIN_OBJECT_GAP},
    },
    {
        "id": "comoda_quarto_2",
        "room": "quarto_2",
        "center": [4.20, 11.42, 0.0],
        "dimensions": [1.05, 0.48, 0.82],
        "yaw": np.pi,
        "function": "furniture.create_item",
        "item_type": "dresser",
        "params": {"width": 1.05, "depth": 0.48, "height": 0.82},
        "clearance": {"wall": INTERIOR_WALL_MARGIN, "opening": DOOR_CLEARANCE_DEPTH, "between_objects": MIN_OBJECT_GAP},
    },
    {
        "id": "mesa_pc_quarto_2",
        "room": "quarto_2",
        "center": [0.5, 15, 0.0],
        "dimensions": [1.40, 1.45, 0.95],
        "local_bounds": {"x": [-0.70, 0.70], "y": [-1.10, 0.35], "z": [0.0, 0.95]},
        "yaw": math.pi /2 ,
        "function": "compositions.create_composition",
        "composition_type": "pc_desk",
        "params": {"width": 1.40, "depth": 0.70, "include_book_stack": True, "include_chair": True},
        "clearance": {"wall": INTERIOR_WALL_MARGIN, "opening": DOOR_CLEARANCE_DEPTH, "between_objects": MIN_OBJECT_GAP},
    },
    {
        "id": "pia_banheiro_social",
        "room": "banheiro_social",
        "center": [4.20, 0.36, 0.0],
        "dimensions": [0.55, 0.42, 0.85],
        "yaw": np.pi,
        "function": "local.create_bathroom_vanity",
        "params": {"width": 0.55, "depth": 0.42, "height": 0.85},
        "can_be_under_window": True,
        "clearance": {"wall": INTERIOR_WALL_MARGIN, "opening": DOOR_CLEARANCE_DEPTH, "between_objects": MIN_OBJECT_GAP},
    },
    {
        "id": "vaso_banheiro_social",
        "room": "banheiro_social",
        "center": [4.18, 1.20, 0.0],
        "dimensions": [0.50, 0.70, 0.78],
        "yaw": np.pi/2,
        "function": "local.create_toilet",
        "params": {"width": 0.50, "depth": 0.70, "height": 0.78},
        "clearance": {"wall": INTERIOR_WALL_MARGIN, "opening": DOOR_CLEARANCE_DEPTH, "between_objects": MIN_OBJECT_GAP},
    },
    {
        "id": "box_banheiro_social",
        "room": "banheiro_social",
        "center": [4.38, 2.00, 0.0],
        "dimensions": [0.78, 0.76, 2.05],
        "yaw": 0.0,
        "function": "local.create_shower_box",
        "params": {"width": 0.78, "depth": 0.76, "height": 2.05},
        "clearance": {"wall": INTERIOR_WALL_MARGIN, "opening": DOOR_CLEARANCE_DEPTH, "between_objects": MIN_OBJECT_GAP},
    },
    {
        "id": "pia_banheiro_familia",
        "room": "banheiro_familia",
        "center": [4.20, 2.86, 0.0],
        "dimensions": [0.55, 0.42, 0.85],
        "yaw": np.pi,
        "function": "local.create_bathroom_vanity",
        "params": {"width": 0.55, "depth": 0.42, "height": 0.85},
        "clearance": {"wall": INTERIOR_WALL_MARGIN, "opening": DOOR_CLEARANCE_DEPTH, "between_objects": MIN_OBJECT_GAP},
    },
    {
        "id": "vaso_banheiro_familia",
        "room": "banheiro_familia",
        "center": [4.18, 3.70, 0.0],
        "dimensions": [0.50, 0.70, 0.78],
        "yaw": np.pi/2,
        "function": "local.create_toilet",
        "params": {"width": 0.50, "depth": 0.70, "height": 0.78},
        "clearance": {"wall": INTERIOR_WALL_MARGIN, "opening": DOOR_CLEARANCE_DEPTH, "between_objects": MIN_OBJECT_GAP},
    },
    {
        "id": "box_banheiro_familia",
        "room": "banheiro_familia",
        "center": [4.31, 4.50, 0.0],
        "dimensions": [0.72, 0.74, 2.05],
        "yaw": 0.0,
        "function": "local.create_shower_box",
        "params": {"width": 0.72, "depth": 0.74, "height": 2.05},
        "clearance": {"wall": INTERIOR_WALL_MARGIN, "opening": DOOR_CLEARANCE_DEPTH, "between_objects": MIN_OBJECT_GAP},
    },
    {
        "id": "mesa_area_reuniao",
        "room": "area_reuniao",
        "center": [13.50, 17.55, 0.0],
        "dimensions": [1.30, 1.30, 0.75],
        "yaw": 0.0,
        "function": "furniture.create_item",
        "item_type": "round_table",
        "params": {"radius": 0.65, "height": 0.75},
        "clearance": {"edge": OUTDOOR_EDGE_MARGIN, "between_objects": MIN_OBJECT_GAP},
    },
    {
        "id": "cadeira_reuniao_norte",
        "room": "area_reuniao",
        "center": [13.50, 18.65, 0.0],
        "dimensions": [0.46, 0.48, 0.90],
        "yaw": 0,
        "function": "furniture.create_item",
        "item_type": "chair",
        "params": {"width": 0.46, "depth": 0.48},
        "clearance": {"edge": OUTDOOR_EDGE_MARGIN, "between_objects": MIN_OBJECT_GAP},
    },
    {
        "id": "cadeira_reuniao_sul",
        "room": "area_reuniao",
        "center": [13.50, 16.45, 0.0],
        "dimensions": [0.46, 0.48, 0.90],
        "yaw": math.pi,
        "function": "furniture.create_item",
        "item_type": "chair",
        "params": {"width": 0.46, "depth": 0.48},
        "clearance": {"edge": OUTDOOR_EDGE_MARGIN, "between_objects": MIN_OBJECT_GAP},
    },
    {
        "id": "cadeira_reuniao_leste",
        "room": "area_reuniao",
        "center": [14.60, 17.55, 0.0],
        "dimensions": [0.46, 0.48, 0.90],
        "yaw": -math.pi / 2,
        "function": "furniture.create_item",
        "item_type": "chair",
        "params": {"width": 0.46, "depth": 0.48},
        "clearance": {"edge": OUTDOOR_EDGE_MARGIN, "between_objects": MIN_OBJECT_GAP},
    },
    {
        "id": "cadeira_reuniao_oeste",
        "room": "area_reuniao",
        "center": [12.40, 17.55, 0.0],
        "dimensions": [0.46, 0.48, 0.90],
        "yaw": math.pi / 2,
        "function": "furniture.create_item",
        "item_type": "chair",
        "params": {"width": 0.46, "depth": 0.48},
        "clearance": {"edge": OUTDOOR_EDGE_MARGIN, "between_objects": MIN_OBJECT_GAP},
    },
    {
        "id": "espreguicadeira_piscina_1",
        "room": "area_piscina",
        "center": [8.32, 11.20, 0.0],
        "dimensions": [1.65, 0.48, 0.45],
        "yaw": math.pi / 2,
        "function": "furniture.create_item",
        "item_type": "bench",
        "params": {"width": 1.65, "depth": 0.48, "height": 0.45},
        "clearance": {"edge": OUTDOOR_EDGE_MARGIN, "between_objects": MIN_OBJECT_GAP},
    },
    {
        "id": "espreguicadeira_piscina_2",
        "room": "area_piscina",
        "center": [8.32, 13.35, 0.0],
        "dimensions": [1.65, 0.48, 0.45],
        "yaw": math.pi / 2,
        "function": "furniture.create_item",
        "item_type": "bench",
        "params": {"width": 1.65, "depth": 0.48, "height": 0.45},
        "clearance": {"edge": OUTDOOR_EDGE_MARGIN, "between_objects": MIN_OBJECT_GAP},
    },
]


VEHICLE_LAYOUT = [
    {
        "id": "caminhonete_carregada",
        "area": "area_carro",
        "center": [10.90, -3.15, 0.0],
        "dimensions": [1.98, 5.25, 1.75],
        "yaw": 0.0,
        "function": "vehicles.create_vehicle",
        "vehicle_type": "work_pickup",
        "params": {"include_cargo": True, "body_color": "#2F5D7C"},
        "clearance": {"edge": 0.10, "opening": 0.75, "between_objects": 0.20},
    },
    {
        "id": "crossover_compacto",
        "area": "area_carro",
        "center": [13.65, -3.10, 0.0],
        "dimensions": [1.82, 4.28, 1.55],
        "yaw": 0.0,
        "function": "vehicles.create_vehicle",
        "vehicle_type": "compact_crossover",
        "params": {"body_color": "#8F3D36", "include_roof_rails": True},
        "clearance": {"edge": 0.10, "opening": 0.75, "between_objects": 0.20},
    },
]


TREE_LAYOUT = [
    {
        "id": "arvore_1",
        "center": [2.5, -4.0, 0.0],
        "dimensions": [6.0, 6.0, 8.0],
        "yaw": 0.0,
        "function": "local.create_large_tree",
        "params": {
            "height": 8.0,
            "crown_radius": 3.0,
            "trunk_radius": 0.45,
            "branches_per_level": 5,
            "seed": 7,
        },
    },
    {
        "id": "arvore_2",
        "center": [18.40, 11.00, 0.0],
        "dimensions": [6.0, 6.0, 8.0],
        "yaw": 0.0,
        "function": "local.create_large_tree",
        "params": {
            "height": 8.0,
            "crown_radius": 3.0,
            "trunk_radius": 0.45,
            "branches_per_level": 5,
            "seed": 7,
        },
    },
]


def pose(x: float, y: float, z: float = 0.0, yaw: float = 0.0):
    return ub.Utils.trn([x, y, z]) @ ub.Utils.rotz(yaw)


def r2(value: float) -> float:
    return round(float(value), 4)


def rect(name, x0, x1, y0, y1, z0=0.0, z1=0.0, kind=None):
    width = x1 - x0
    depth = y1 - y0
    if width <= EPS or depth <= EPS:
        raise ValueError(f"Retangulo invalido: {name}")

    data = {
        "id": name,
        "kind": kind,
        "limits": {
            "x": [r2(x0), r2(x1)],
            "y": [r2(y0), r2(y1)],
            "z": [r2(z0), r2(z1)],
        },
        "center": [r2((x0 + x1) / 2), r2((y0 + y1) / 2), r2((z0 + z1) / 2)],
        "width": r2(width),
        "depth": r2(depth),
        "area": r2(width * depth),
    }
    return data


def room(name, label, x0, x1, y0, y1, floor_type):
    data = rect(name, x0, x1, y0, y1, 0.0, WALL_HEIGHT, "room")
    data.update(
        {
            "label": label,
            "floor_type": floor_type,
            "walls": [],
            "access_openings": [],
            "windows": [],
        }
    )
    return data


def interval_overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def rect_overlap_area(a, b):
    ax0, ax1 = a["limits"]["x"]
    ay0, ay1 = a["limits"]["y"]
    bx0, bx1 = b["limits"]["x"]
    by0, by1 = b["limits"]["y"]
    return interval_overlap(ax0, ax1, bx0, bx1) * interval_overlap(ay0, ay1, by0, by1)


def point_close(p, q, tol=EPS):
    return abs(p[0] - q[0]) <= tol and abs(p[1] - q[1]) <= tol


def between(v, a, b, tol=EPS):
    return min(a, b) - tol <= v <= max(a, b) + tol


def wall_axis(start, end):
    if abs(start[1] - end[1]) <= EPS and abs(start[0] - end[0]) > EPS:
        return "x"
    if abs(start[0] - end[0]) <= EPS and abs(start[1] - end[1]) > EPS:
        return "y"
    raise ValueError(f"Parede deve ser ortogonal: {start} -> {end}")


def normalize_segment(start, end):
    start = tuple(float(v) for v in start)
    end = tuple(float(v) for v in end)
    axis = wall_axis(start, end)
    if axis == "x" and start[0] > end[0]:
        start, end = end, start
    if axis == "y" and start[1] > end[1]:
        start, end = end, start
    return start, end


def wall_spec(wall_id, start, end, rooms, doors=None, windows=None, wall_type="internal"):
    start, end = normalize_segment(start, end)
    axis = wall_axis(start, end)
    length = abs(end[0] - start[0]) if axis == "x" else abs(end[1] - start[1])
    center = [(start[0] + end[0]) / 2, (start[1] + end[1]) / 2, WALL_HEIGHT / 2]
    yaw = 0.0 if axis == "x" else math.pi / 2
    return {
        "id": wall_id,
        "type": wall_type,
        "start": [r2(start[0]), r2(start[1])],
        "end": [r2(end[0]), r2(end[1])],
        "center": [r2(v) for v in center],
        "length": r2(length),
        "height": WALL_HEIGHT,
        "thickness": WALL_THICKNESS,
        "axis": axis,
        "orientation": "horizontal_x" if axis == "x" else "vertical_y",
        "yaw_rad": r2(yaw),
        "rooms": list(rooms),
        "door_openings": doors or [],
        "windows": windows or [],
    }


def opening(opening_id, wall_id, center_along_wall, width, connects):
    return {
        "id": opening_id,
        "kind": "door_opening_without_leaf",
        "wall": wall_id,
        "center_along_wall": r2(center_along_wall),
        "width": r2(width),
        "height": r2(DOOR_HEIGHT),
        "bottom_z": 0.0,
        "connects": list(connects),
        "has_physical_door": False,
    }


def window(window_id, wall_id, center_along_wall, width, rooms):
    return {
        "id": window_id,
        "wall": wall_id,
        "center_along_wall": r2(center_along_wall),
        "width": r2(width),
        "height": r2(WINDOW_HEIGHT),
        "sill_height": r2(WINDOW_SILL),
        "rooms": list(rooms),
    }


def build_plan_data():
    rooms = [
        room("escritorio", "Escritorio", 0.0, 3.8, 0.0, 5.0, "madeira_quente"),
        room("banheiro_social", "Banheiro social", 3.8, 5.4, 0.0, 2.5, "ceramica_hidraulica"),
        room("banheiro_familia", "Banheiro familia", 3.8, 5.4, 2.5, 5.0, "ceramica_hidraulica"),
        room("circulacao_frontal", "Circulacao frontal", 5.4, 7.0, 0.0, 6.0, "porcelanato_claro"),
        room("circulacao_transversal", "Circulacao transversal", 0.0, 5.4, 5.0, 6.0, "porcelanato_claro"),
        room("quarto_1", "Quarto 1", 0.0, 5.4, 6.0, 11.0, "madeira_quente"),
        room("quarto_2", "Quarto 2", 0.0, 5.4, 11.0, 16.0, "madeira_quente"),
        room("corredor_privativo", "Corredor privativo", 5.4, 7.0, 6.0, 16.0, "porcelanato_claro"),
        room("sala", "Sala", 7.0, 16.0, 0.0, 3.3, "porcelanato_claro"),
        room("sala_de_jantar", "Sala de jantar", 7.0, 12.2, 3.3, 6.0, "porcelanato_claro"),
        room("cozinha", "Cozinha", 12.2, 16.0, 3.3, 6.0, "ceramica_cozinha"),
    ]

    doors = {
        "vao_entrada_principal": opening("vao_entrada_principal", "parede_frente_sala", 8.85, 1.20, ["exterior_frente", "sala"]),
        "vao_saida_jantar_quintal": opening("vao_saida_jantar_quintal", "parede_fundos_jantar", 8.40, 1.40, ["sala_de_jantar", "quintal"]),
        "vao_saida_corredor_quintal": opening("vao_saida_corredor_quintal", "parede_leste_corredor", 13.00, 1.10, ["corredor_privativo", "quintal"]),
        "vao_banheiro_social": opening("vao_banheiro_social", "parede_banheiro_social_hall", 1.25, 0.80, ["banheiro_social", "circulacao_frontal"]),
        "vao_banheiro_familia": opening("vao_banheiro_familia", "parede_banheiro_familia_hall", 3.75, 0.80, ["banheiro_familia", "circulacao_frontal"]),
        "vao_escritorio": opening("vao_escritorio", "parede_escritorio_circulacao", 2.15, 0.90, ["escritorio", "circulacao_transversal"]),
        "vao_quarto_1": opening("vao_quarto_1", "parede_quarto1_corredor", 8.50, 0.90, ["quarto_1", "corredor_privativo"]),
        "vao_quarto_2": opening("vao_quarto_2", "parede_quarto2_corredor", 13.50, 0.90, ["quarto_2", "corredor_privativo"]),
        "vao_hall_sala": opening("vao_hall_sala", "parede_hall_sala", 1.60, 1.20, ["circulacao_frontal", "sala"]),
        "vao_hall_jantar": opening("vao_hall_jantar", "parede_hall_jantar", 4.65, 1.20, ["circulacao_frontal", "sala_de_jantar"]),
        "vao_sala_jantar": opening("vao_sala_jantar", "parede_sala_jantar", 9.50, 2.20, ["sala", "sala_de_jantar"]),
        "vao_jantar_cozinha": opening("vao_jantar_cozinha", "parede_jantar_cozinha", 4.65, 1.50, ["sala_de_jantar", "cozinha"]),
    }

    windows = {
        "janela_frente_escritorio": window("janela_frente_escritorio", "parede_frente_escritorio", 1.90, 1.60, ["escritorio"]),
        "janela_lateral_escritorio": window("janela_lateral_escritorio", "parede_oeste_escritorio", 2.55, 1.30, ["escritorio"]),
        "janela_banheiro_social": window("janela_banheiro_social", "parede_frente_banheiro_social", 4.60, 0.70, ["banheiro_social"]),
        "janela_frente_sala": window("janela_frente_sala", "parede_frente_sala", 13.35, 2.00, ["sala"]),
        "janela_leste_sala": window("janela_leste_sala", "parede_leste_sala", 1.75, 1.50, ["sala"]),
        "janela_leste_cozinha": window("janela_leste_cozinha", "parede_leste_cozinha", 4.75, 1.20, ["cozinha"]),
        "janela_fundos_jantar": window("janela_fundos_jantar", "parede_fundos_jantar", 11.20, 1.20, ["sala_de_jantar"]),
        "janela_fundos_cozinha": window("janela_fundos_cozinha", "parede_fundos_cozinha", 14.10, 1.40, ["cozinha"]),
        "janela_corredor_quintal": window("janela_corredor_quintal", "parede_leste_corredor", 8.50, 1.20, ["corredor_privativo"]),
        "janela_oeste_quarto1": window("janela_oeste_quarto1", "parede_oeste_quarto1", 8.50, 1.50, ["quarto_1"]),
        "janela_oeste_quarto2": window("janela_oeste_quarto2", "parede_oeste_quarto2", 13.50, 1.50, ["quarto_2"]),
        "janela_norte_quarto2": window("janela_norte_quarto2", "parede_norte_quarto2", 2.70, 1.70, ["quarto_2"]),
    }

    walls = [
        wall_spec("parede_frente_escritorio", (0.0, 0.0), (3.8, 0.0), ["exterior_frente", "escritorio"], windows=["janela_frente_escritorio"], wall_type="external"),
        wall_spec("parede_frente_banheiro_social", (3.8, 0.0), (5.4, 0.0), ["exterior_frente", "banheiro_social"], windows=["janela_banheiro_social"], wall_type="external"),
        wall_spec("parede_frente_circulacao", (5.4, 0.0), (7.0, 0.0), ["exterior_frente", "circulacao_frontal"], wall_type="external"),
        wall_spec("parede_frente_sala", (7.0, 0.0), (16.0, 0.0), ["exterior_frente", "sala"], doors=["vao_entrada_principal"], windows=["janela_frente_sala"], wall_type="external"),
        wall_spec("parede_leste_sala", (16.0, 0.0), (16.0, 3.3), ["exterior_leste", "sala"], windows=["janela_leste_sala"], wall_type="external"),
        wall_spec("parede_leste_cozinha", (16.0, 3.3), (16.0, 6.0), ["exterior_leste", "cozinha"], windows=["janela_leste_cozinha"], wall_type="external"),
        wall_spec("parede_fundos_jantar", (7.0, 6.0), (12.2, 6.0), ["quintal", "sala_de_jantar"], doors=["vao_saida_jantar_quintal"], windows=["janela_fundos_jantar"], wall_type="external"),
        wall_spec("parede_fundos_cozinha", (12.2, 6.0), (16.0, 6.0), ["quintal", "cozinha"], windows=["janela_fundos_cozinha"], wall_type="external"),
        wall_spec("parede_leste_corredor", (7.0, 6.0), (7.0, 16.0), ["quintal", "corredor_privativo"], doors=["vao_saida_corredor_quintal"], windows=["janela_corredor_quintal"], wall_type="external"),
        wall_spec("parede_norte_quarto2", (0.0, 16.0), (7.0, 16.0), ["exterior_norte", "quarto_2", "corredor_privativo"], windows=["janela_norte_quarto2"], wall_type="external"),
        wall_spec("parede_oeste_quarto2", (0.0, 11.0), (0.0, 16.0), ["exterior_oeste", "quarto_2"], windows=["janela_oeste_quarto2"], wall_type="external"),
        wall_spec("parede_oeste_quarto1", (0.0, 6.0), (0.0, 11.0), ["exterior_oeste", "quarto_1"], windows=["janela_oeste_quarto1"], wall_type="external"),
        wall_spec("parede_oeste_circulacao", (0.0, 5.0), (0.0, 6.0), ["exterior_oeste", "circulacao_transversal"], wall_type="external"),
        wall_spec("parede_oeste_escritorio", (0.0, 0.0), (0.0, 5.0), ["exterior_oeste", "escritorio"], windows=["janela_lateral_escritorio"], wall_type="external"),
        wall_spec("parede_escritorio_banheiros", (3.8, 0.0), (3.8, 5.0), ["escritorio", "banheiros"], wall_type="internal"),
        wall_spec("parede_divisoria_banheiros", (3.8, 2.5), (5.4, 2.5), ["banheiro_social", "banheiro_familia"], wall_type="internal"),
        wall_spec("parede_banheiro_social_hall", (5.4, 0.0), (5.4, 2.5), ["banheiro_social", "circulacao_frontal"], doors=["vao_banheiro_social"], wall_type="internal"),
        wall_spec("parede_banheiro_familia_hall", (5.4, 2.5), (5.4, 5.0), ["banheiro_familia", "circulacao_frontal"], doors=["vao_banheiro_familia"], wall_type="internal"),
        wall_spec("parede_escritorio_circulacao", (0.0, 5.0), (3.8, 5.0), ["escritorio", "circulacao_transversal"], doors=["vao_escritorio"], wall_type="internal"),
        wall_spec("parede_banheiro_familia_circulacao", (3.8, 5.0), (5.4, 5.0), ["banheiro_familia", "circulacao_transversal"], wall_type="internal"),
        wall_spec("parede_quarto1_sul", (0.0, 6.0), (5.4, 6.0), ["quarto_1", "circulacao_transversal"], wall_type="internal"),
        wall_spec("parede_divisoria_quartos", (0.0, 11.0), (5.4, 11.0), ["quarto_1", "quarto_2"], wall_type="internal"),
        wall_spec("parede_quarto1_corredor", (5.4, 6.0), (5.4, 11.0), ["quarto_1", "corredor_privativo"], doors=["vao_quarto_1"], wall_type="internal"),
        wall_spec("parede_quarto2_corredor", (5.4, 11.0), (5.4, 16.0), ["quarto_2", "corredor_privativo"], doors=["vao_quarto_2"], wall_type="internal"),
        wall_spec("parede_hall_sala", (7.0, 0.0), (7.0, 3.3), ["circulacao_frontal", "sala"], doors=["vao_hall_sala"], wall_type="internal"),
        wall_spec("parede_hall_jantar", (7.0, 3.3), (7.0, 6.0), ["circulacao_frontal", "sala_de_jantar"], doors=["vao_hall_jantar"], wall_type="internal"),
        wall_spec("parede_sala_jantar", (7.0, 3.3), (12.2, 3.3), ["sala", "sala_de_jantar"], doors=["vao_sala_jantar"], wall_type="internal"),
        wall_spec("parede_sala_cozinha", (12.2, 3.3), (16.0, 3.3), ["sala", "cozinha"], wall_type="internal"),
        wall_spec("parede_jantar_cozinha", (12.2, 3.3), (12.2, 6.0), ["sala_de_jantar", "cozinha"], doors=["vao_jantar_cozinha"], wall_type="internal"),
    ]

    outdoor = {
        "area_carro": rect("area_carro", 6.8, 15.0, -6.0, -0.4, -OUTDOOR_THICKNESS, 0.0, "pavement"),
        "piscina": rect("piscina", 10.0, 16.2, 10.2, 14.7, 0.02, 0.05, "pool"),
        "borda_piscina": rect("borda_piscina", 9.55, 16.65, 9.75, 15.15, -OUTDOOR_THICKNESS, 0.0, "pool_coping"),
        "area_reuniao": rect("area_reuniao", 10.0, 17.0, 15.5, 19.5, -OUTDOOR_THICKNESS, 0.0, "meeting_patio"),
        "grama_frente_lateral": rect("grama_frente_lateral", -2.0, 6.4, -6.0, -0.4, -OUTDOOR_THICKNESS, 0.0, "grass"),
        "grama_lateral_oeste": rect("grama_lateral_oeste", -2.0, 0.0, -0.4, 22.0, -OUTDOOR_THICKNESS, 0.0, "grass"),
        "grama_fundos_ala": rect("grama_fundos_ala", 0.0, 7.0, 16.0, 22.0, -OUTDOOR_THICKNESS, 0.0, "grass"),
        "grama_centro_quintal": rect("grama_centro_quintal", 7.0, 20.0, 6.0, 9.75, -OUTDOOR_THICKNESS, 0.0, "grass"),
        "grama_lateral_piscina": rect("grama_lateral_piscina", 7.0, 9.55, 9.75, 20.0, -OUTDOOR_THICKNESS, 0.0, "grass"),
        "grama_leste_quintal": rect("grama_leste_quintal", 17.0, 20.0, 9.75, 22.0, -OUTDOOR_THICKNESS, 0.0, "grass"),
        "grama_entre_piscina_patio": rect("grama_entre_piscina_patio", 9.55, 17.0, 15.15, 15.5, -OUTDOOR_THICKNESS, 0.0, "grass"),
        "grama_fundos_patio": rect("grama_fundos_patio", 7.0, 20.0, 19.5, 22.0, -OUTDOOR_THICKNESS, 0.0, "grass"),
    }

    roof = [
        {
            "id": "telhado_ala_frontal",
            "type": "gable_roof",
            "aligned_wall_bounds": {"x": [0.0, 16.0], "y": [0.0, 6.0], "z": [WALL_HEIGHT, WALL_HEIGHT]},
            "center": [8.0, 3.0, WALL_HEIGHT],
            "width": 6.0,
            "depth": 16.0,
            "yaw_rad": r2(math.pi / 2),
            "overhang": 0.35,
            "pitch_deg": 24.0,
        },
        {
            "id": "telhado_ala_privativa",
            "type": "gable_roof",
            "aligned_wall_bounds": {"x": [0.0, 7.0], "y": [6.0, 16.0], "z": [WALL_HEIGHT, WALL_HEIGHT]},
            "center": [3.5, 11.0, WALL_HEIGHT],
            "width": 7.0,
            "depth": 10.0,
            "yaw_rad": 0.0,
            "overhang": 0.35,
            "pitch_deg": 24.0,
        },
    ]

    plan = {
        "metadata": {
            "name": "Casa terrea grande em L",
            "front_direction": "-y",
            "notes": [
                "A base do L fica na frente: x=0..16, y=0..6.",
                "A ala privativa sobe para os fundos: x=0..7, y=0..16.",
                "Nao ha garagem e todas as portas sao apenas vaos sem folhas.",
            ],
            "tolerance": EPS,
            "wall_height": WALL_HEIGHT,
            "wall_thickness": WALL_THICKNESS,
            "available_architecture": available_architecture(),
            "available_furniture": available_items(),
            "available_props": available_props(),
            "available_vehicles": available_vehicles(),
            "available_compositions": available_compositions(),
        },
        "footprint": {
            "shape": "L",
            "components": [
                rect("footprint_ala_frontal", 0.0, 16.0, 0.0, 6.0, 0.0, WALL_HEIGHT, "footprint_component"),
                rect("footprint_ala_privativa", 0.0, 7.0, 6.0, 16.0, 0.0, WALL_HEIGHT, "footprint_component"),
            ],
            "outer_polygon": [[0.0, 0.0], [16.0, 0.0], [16.0, 6.0], [7.0, 6.0], [7.0, 16.0], [0.0, 16.0]],
            "area": 166.0,
        },
        "rooms": rooms,
        "walls": walls,
        "door_openings": list(doors.values()),
        "windows": list(windows.values()),
        "floors": [],
        "outdoor": outdoor,
        "roof": roof,
        "validation": {},
    }

    for item in rooms:
        plan["floors"].append(
            {
                "id": f"piso_{item['id']}",
                "room": item["id"],
                "floor_type": item["floor_type"],
                "limits": item["limits"],
                "center": item["center"],
                "width": item["width"],
                "depth": item["depth"],
                "thickness": FLOOR_THICKNESS,
            }
        )

    attach_wall_references(plan)
    enrich_openings_and_windows(plan)
    plan["furniture"] = build_furniture_registry(plan)
    plan["vehicles"] = build_vehicle_registry(plan)
    plan["trees"] = build_tree_registry(plan)
    return plan


def attach_wall_references(plan):
    rooms_by_id = {item["id"]: item for item in plan["rooms"]}
    for wall in plan["walls"]:
        for room_id in wall["rooms"]:
            if room_id in rooms_by_id:
                rooms_by_id[room_id]["walls"].append(wall["id"])
        for opening_id in wall["door_openings"]:
            for room_id in next(o for o in plan["door_openings"] if o["id"] == opening_id)["connects"]:
                if room_id in rooms_by_id:
                    rooms_by_id[room_id]["access_openings"].append(opening_id)
        for window_id in wall["windows"]:
            for room_id in next(w for w in plan["windows"] if w["id"] == window_id)["rooms"]:
                if room_id in rooms_by_id:
                    rooms_by_id[room_id]["windows"].append(window_id)


def enrich_openings_and_windows(plan):
    wall_by_id = {wall["id"]: wall for wall in plan["walls"]}
    for item in plan["door_openings"]:
        wall = wall_by_id[item["wall"]]
        axis = wall["axis"]
        c = item["center_along_wall"]
        half = item["width"] / 2
        if axis == "x":
            world_center = [c, wall["start"][1], item["height"] / 2]
            bounds = {"x": [c - half, c + half], "y": [wall["start"][1] - WALL_THICKNESS / 2, wall["start"][1] + WALL_THICKNESS / 2], "z": [0.0, item["height"]]}
        else:
            world_center = [wall["start"][0], c, item["height"] / 2]
            bounds = {"x": [wall["start"][0] - WALL_THICKNESS / 2, wall["start"][0] + WALL_THICKNESS / 2], "y": [c - half, c + half], "z": [0.0, item["height"]]}
        item["position"] = {"world_center": [r2(v) for v in world_center], "local_offset_from_wall_center": r2(local_offset(wall, c))}
        item["limits"] = {k: [r2(v[0]), r2(v[1])] for k, v in bounds.items()}

    for item in plan["windows"]:
        wall = wall_by_id[item["wall"]]
        axis = wall["axis"]
        c = item["center_along_wall"]
        half = item["width"] / 2
        z0 = item["sill_height"]
        z1 = item["sill_height"] + item["height"]
        if axis == "x":
            world_center = [c, wall["start"][1], (z0 + z1) / 2]
            bounds = {"x": [c - half, c + half], "y": [wall["start"][1] - WALL_THICKNESS / 2, wall["start"][1] + WALL_THICKNESS / 2], "z": [z0, z1]}
        else:
            world_center = [wall["start"][0], c, (z0 + z1) / 2]
            bounds = {"x": [wall["start"][0] - WALL_THICKNESS / 2, wall["start"][0] + WALL_THICKNESS / 2], "y": [c - half, c + half], "z": [z0, z1]}
        item["position"] = {"world_center": [r2(v) for v in world_center], "local_offset_from_wall_center": r2(local_offset(wall, c))}
        item["limits"] = {k: [r2(v[0]), r2(v[1])] for k, v in bounds.items()}


def local_offset(wall, center_along_wall):
    if wall["axis"] == "x":
        return center_along_wall - wall["center"][0]
    return center_along_wall - wall["center"][1]


def rotated_bbox(center, dimensions, yaw):
    cx, cy, z0 = center
    width, depth, height = dimensions
    c = math.cos(yaw)
    s = math.sin(yaw)
    corners = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            lx = sx * width / 2
            ly = sy * depth / 2
            corners.append((cx + c * lx - s * ly, cy + s * lx + c * ly))
    xs = [item[0] for item in corners]
    ys = [item[1] for item in corners]
    return {
        "x": [r2(min(xs)), r2(max(xs))],
        "y": [r2(min(ys)), r2(max(ys))],
        "z": [r2(z0), r2(z0 + height)],
    }


def rotated_bounds_bbox(center, local_bounds, yaw):
    cx, cy, cz = center
    c = math.cos(yaw)
    s = math.sin(yaw)
    corners = []
    for lx in local_bounds["x"]:
        for ly in local_bounds["y"]:
            corners.append((cx + c * lx - s * ly, cy + s * lx + c * ly))
    xs = [item[0] for item in corners]
    ys = [item[1] for item in corners]
    return {
        "x": [r2(min(xs)), r2(max(xs))],
        "y": [r2(min(ys)), r2(max(ys))],
        "z": [r2(cz + local_bounds["z"][0]), r2(cz + local_bounds["z"][1])],
    }


def registry_rect(registry_id, bbox, kind="furniture_bbox"):
    return {
        "id": registry_id,
        "kind": kind,
        "limits": {
            "x": bbox["x"],
            "y": bbox["y"],
            "z": bbox["z"],
        },
        "center": [
            r2((bbox["x"][0] + bbox["x"][1]) / 2),
            r2((bbox["y"][0] + bbox["y"][1]) / 2),
            r2((bbox["z"][0] + bbox["z"][1]) / 2),
        ],
        "width": r2(bbox["x"][1] - bbox["x"][0]),
        "depth": r2(bbox["y"][1] - bbox["y"][0]),
        "area": r2((bbox["x"][1] - bbox["x"][0]) * (bbox["y"][1] - bbox["y"][0])),
    }


def build_furniture_registry(plan):
    registry = {}
    for item in FURNITURE_LAYOUT:
        if "local_bounds" in item:
            bbox = rotated_bounds_bbox(item["center"], item["local_bounds"], item["yaw"])
        else:
            bbox = rotated_bbox(item["center"], item["dimensions"], item["yaw"])
        entry = {
            "id": item["id"],
            "room": item["room"],
            "center": [r2(v) for v in item["center"]],
            "orientation": {
                "yaw_rad": r2(item["yaw"]),
                "yaw_deg": r2(math.degrees(item["yaw"])),
            },
            "dimensions": {
                "width": r2(item["dimensions"][0]),
                "depth": r2(item["dimensions"][1]),
                "height": r2(item["dimensions"][2]),
            },
            "bounding_box": bbox,
            "function": item["function"],
            "clearance": item["clearance"],
            "params": item["params"],
        }
        for key in ("item_type", "composition_type", "mounted_on", "allow_overlap_with", "can_be_under_window"):
            if key in item:
                entry[key] = item[key]
        if "local_bounds" in item:
            entry["local_bounds"] = item["local_bounds"]
        registry[item["id"]] = entry

    global FURNITURE_REGISTRY
    FURNITURE_REGISTRY = registry
    return registry


def build_vehicle_registry(plan):
    registry = {}
    for item in VEHICLE_LAYOUT:
        bbox = rotated_bbox(item["center"], item["dimensions"], item["yaw"])
        registry[item["id"]] = {
            "id": item["id"],
            "area": item["area"],
            "center": [r2(v) for v in item["center"]],
            "orientation": {
                "yaw_rad": r2(item["yaw"]),
                "yaw_deg": r2(math.degrees(item["yaw"])),
            },
            "dimensions": {
                "width": r2(item["dimensions"][0]),
                "depth": r2(item["dimensions"][1]),
                "height": r2(item["dimensions"][2]),
            },
            "bounding_box": bbox,
            "function": item["function"],
            "vehicle_type": item["vehicle_type"],
            "params": item["params"],
            "clearance": item["clearance"],
        }
    return registry


def build_tree_registry(plan):
    registry = {}
    for item in TREE_LAYOUT:
        bbox = rotated_bbox(item["center"], item["dimensions"], item["yaw"])
        registry[item["id"]] = {
            "id": item["id"],
            "center": [r2(v) for v in item["center"]],
            "orientation": {
                "yaw_rad": r2(item["yaw"]),
                "yaw_deg": r2(math.degrees(item["yaw"])),
            },
            "dimensions": {
                "width": r2(item["dimensions"][0]),
                "depth": r2(item["dimensions"][1]),
                "height": r2(item["dimensions"][2]),
            },
            "bounding_box": bbox,
            "function": item["function"],
            "params": item["params"],
        }
    return registry


def placement_area_for_room(plan, room_id):
    for item in plan["rooms"]:
        if item["id"] == room_id:
            return item
    if room_id == "area_reuniao":
        return plan["outdoor"]["area_reuniao"]
    if room_id == "area_piscina":
        return rect("area_piscina_mobiliario", 7.0, 20.0, 9.75, 20.0, 0.0, 2.20, "outdoor_furniture_zone")
    raise KeyError(f"Area de mobiliario desconhecida: {room_id}")


def rect_from_bbox(item):
    return registry_rect(item["id"], item["bounding_box"], "furniture_bbox")


def bbox_inside_area(bbox, area, margin):
    ax0, ax1 = area["limits"]["x"]
    ay0, ay1 = area["limits"]["y"]
    return (
        bbox["x"][0] >= ax0 + margin - EPS
        and bbox["x"][1] <= ax1 - margin + EPS
        and bbox["y"][0] >= ay0 + margin - EPS
        and bbox["y"][1] <= ay1 - margin + EPS
    )


def rects_overlap(a, b, tolerance=EPS):
    return rect_overlap_area(a, b) > tolerance


def can_overlap(a, b):
    allowed_a = set(a.get("allow_overlap_with", []))
    allowed_b = set(b.get("allow_overlap_with", []))
    return b["id"] in allowed_a or a["id"] in allowed_b


def room_side_of_wall(area, wall):
    ax0, ax1 = area["limits"]["x"]
    ay0, ay1 = area["limits"]["y"]
    if wall["axis"] == "x":
        wy = wall["start"][1]
        if ay0 >= wy - EPS:
            return "positive_y"
        if ay1 <= wy + EPS:
            return "negative_y"
    else:
        wx = wall["start"][0]
        if ax0 >= wx - EPS:
            return "positive_x"
        if ax1 <= wx + EPS:
            return "negative_x"
    return "overlap"


def door_clearance_zone(plan, opening, room_id):
    wall = next(item for item in plan["walls"] if item["id"] == opening["wall"])
    area = placement_area_for_room(plan, room_id)
    side = room_side_of_wall(area, wall)
    c = opening["center_along_wall"]
    half = opening["width"] / 2 + DOOR_CLEARANCE_SIDE

    if wall["axis"] == "x":
        x0 = c - half
        x1 = c + half
        wy = wall["start"][1]
        if side == "positive_y":
            y0, y1 = wy, wy + DOOR_CLEARANCE_DEPTH
        elif side == "negative_y":
            y0, y1 = wy - DOOR_CLEARANCE_DEPTH, wy
        else:
            y0, y1 = wy - DOOR_CLEARANCE_DEPTH / 2, wy + DOOR_CLEARANCE_DEPTH / 2
    else:
        y0 = c - half
        y1 = c + half
        wx = wall["start"][0]
        if side == "positive_x":
            x0, x1 = wx, wx + DOOR_CLEARANCE_DEPTH
        elif side == "negative_x":
            x0, x1 = wx - DOOR_CLEARANCE_DEPTH, wx
        else:
            x0, x1 = wx - DOOR_CLEARANCE_DEPTH / 2, wx + DOOR_CLEARANCE_DEPTH / 2

    ax0, ax1 = area["limits"]["x"]
    ay0, ay1 = area["limits"]["y"]
    x0, x1 = max(x0, ax0), min(x1, ax1)
    y0, y1 = max(y0, ay0), min(y1, ay1)
    if x1 - x0 <= EPS or y1 - y0 <= EPS:
        return None
    return rect(f"zona_livre_{opening['id']}_{room_id}", x0, x1, y0, y1, 0.0, DOOR_HEIGHT, "door_clearance")


def furniture_blocks_window(plan, furniture):
    if furniture.get("can_be_under_window"):
        return False
    if furniture["bounding_box"]["z"][1] <= WINDOW_TALL_OBJECT_LIMIT:
        return False

    room_id = furniture["room"]
    if room_id not in {item["id"] for item in plan["rooms"]}:
        return False
    bbox = furniture["bounding_box"]
    area = placement_area_for_room(plan, room_id)

    for win in plan["windows"]:
        if room_id not in win["rooms"]:
            continue
        wall = next(item for item in plan["walls"] if item["id"] == win["wall"])
        side = room_side_of_wall(area, wall)
        if wall["axis"] == "x":
            wx0, wx1 = win["limits"]["x"]
            wy = wall["start"][1]
            if side == "positive_y":
                zone = rect("window_zone", wx0, wx1, wy, wy + WINDOW_CLEARANCE_DEPTH, 0.0, WALL_HEIGHT)
            else:
                zone = rect("window_zone", wx0, wx1, wy - WINDOW_CLEARANCE_DEPTH, wy, 0.0, WALL_HEIGHT)
        else:
            wy0, wy1 = win["limits"]["y"]
            wx = wall["start"][0]
            if side == "positive_x":
                zone = rect("window_zone", wx, wx + WINDOW_CLEARANCE_DEPTH, wy0, wy1, 0.0, WALL_HEIGHT)
            else:
                zone = rect("window_zone", wx - WINDOW_CLEARANCE_DEPTH, wx, wy0, wy1, 0.0, WALL_HEIGHT)
        if rects_overlap(rect_from_bbox(furniture), zone):
            return True
    return False


def validate_furniture(plan, errors, checks):
    furniture = list(plan.get("furniture", {}).values())
    by_room = {}

    for item in furniture:
        area = placement_area_for_room(plan, item["room"])
        margin = item["clearance"].get("wall", item["clearance"].get("edge", OUTDOOR_EDGE_MARGIN))
        if not bbox_inside_area(item["bounding_box"], area, margin):
            errors.append(f"Movel fora da area correta ou dentro da parede: {item['id']} em {item['room']}")
        by_room.setdefault(item["room"], []).append(item)

        item_rect = rect_from_bbox(item)
        for opening in plan["door_openings"]:
            if item["room"] not in opening["connects"]:
                continue
            zone = door_clearance_zone(plan, opening, item["room"])
            if zone is not None and rects_overlap(item_rect, zone):
                errors.append(f"Movel bloqueia vao: {item['id']} x {opening['id']}")

        if furniture_blocks_window(plan, item):
            errors.append(f"Movel alto invade zona de janela: {item['id']}")

        if item["room"] == "area_piscina":
            for protected in (plan["outdoor"]["piscina"], plan["outdoor"]["area_carro"]):
                if rects_overlap(item_rect, protected):
                    errors.append(f"Movel externo invade area proibida: {item['id']} x {protected['id']}")
            for part in plan["footprint"]["components"]:
                if rects_overlap(item_rect, part):
                    errors.append(f"Movel externo entra na casa: {item['id']} x {part['id']}")

    for i, a in enumerate(furniture):
        rect_a = rect_from_bbox(a)
        for b in furniture[i + 1 :]:
            if can_overlap(a, b):
                continue
            if rects_overlap(rect_a, rect_from_bbox(b), tolerance=MIN_OBJECT_GAP * MIN_OBJECT_GAP):
                errors.append(f"Moveis sobrepostos: {a['id']} x {b['id']}")

    for room_id, items in by_room.items():
        area = placement_area_for_room(plan, room_id)
        furniture_area = sum(rect_from_bbox(item)["area"] for item in items if "mounted_on" not in item)
        max_ratio = 0.72 if room_id.startswith("banheiro") else 0.62
        if furniture_area / area["area"] > max_ratio:
            errors.append(f"Circulacao insuficiente por excesso de moveis em {room_id}")

    checks["furniture_inside_correct_area"] = not any("fora da area" in e for e in errors)
    checks["furniture_does_not_block_openings"] = not any("bloqueia vao" in e for e in errors)
    checks["furniture_respects_windows"] = not any("janela" in e for e in errors)
    checks["furniture_no_overlap"] = not any("Moveis sobrepostos" in e for e in errors)
    checks["furniture_reasonable_circulation"] = not any("Circulacao insuficiente" in e for e in errors)


def principal_entry_clearance_zone(plan):
    opening = next(item for item in plan["door_openings"] if item["id"] == "vao_entrada_principal")
    c = opening["center_along_wall"]
    half = opening["width"] / 2 + 0.35
    return rect(
        "zona_livre_externa_entrada_principal",
        c - half,
        c + half,
        -1.10,
        0.0,
        0.0,
        DOOR_HEIGHT,
        "external_entry_clearance",
    )


def validate_vehicles(plan, errors, checks):
    vehicles = list(plan.get("vehicles", {}).values())
    parking = plan["outdoor"]["area_carro"]
    entry_zone = principal_entry_clearance_zone(plan)

    for item in vehicles:
        item_rect = rect_from_bbox(item)
        if item["area"] != "area_carro":
            errors.append(f"Veiculo registrado em area inesperada: {item['id']} -> {item['area']}")
        if not bbox_inside_area(item["bounding_box"], parking, item["clearance"]["edge"]):
            errors.append(f"Veiculo fora da area de estacionamento: {item['id']}")
        if rects_overlap(item_rect, entry_zone):
            errors.append(f"Veiculo bloqueia entrada principal: {item['id']}")
        for part in plan["footprint"]["components"]:
            if rects_overlap(item_rect, part):
                errors.append(f"Veiculo invade a casa: {item['id']} x {part['id']}")

    for i, a in enumerate(vehicles):
        rect_a = rect_from_bbox(a)
        for b in vehicles[i + 1 :]:
            if rects_overlap(rect_a, rect_from_bbox(b), tolerance=0.04):
                errors.append(f"Veiculos sobrepostos: {a['id']} x {b['id']}")

    checks["vehicles_inside_parking_area"] = not any("Veiculo fora" in e for e in errors)
    checks["vehicles_do_not_block_entry"] = not any("bloqueia entrada" in e for e in errors)
    checks["vehicles_no_overlap"] = not any("Veiculos sobrepostos" in e for e in errors)
    checks["vehicles_do_not_invade_house"] = not any("Veiculo invade" in e for e in errors)


def rect_inside_footprint(room_rect, footprint_components):
    rx0, rx1 = room_rect["limits"]["x"]
    ry0, ry1 = room_rect["limits"]["y"]
    for comp in footprint_components:
        cx0, cx1 = comp["limits"]["x"]
        cy0, cy1 = comp["limits"]["y"]
        if rx0 >= cx0 - EPS and rx1 <= cx1 + EPS and ry0 >= cy0 - EPS and ry1 <= cy1 + EPS:
            return True
    return False


def validate_plan(plan):
    errors = []
    checks = {}
    rooms = plan["rooms"]
    walls = plan["walls"]
    wall_by_id = {wall["id"]: wall for wall in walls}
    openings = plan["door_openings"]
    windows = plan["windows"]

    for item in rooms:
        if item["width"] <= 0 or item["depth"] <= 0 or item["area"] <= 0:
            errors.append(f"Comodo invalido: {item['id']}")
        if not rect_inside_footprint(item, plan["footprint"]["components"]):
            errors.append(f"Comodo fora da planta em L: {item['id']}")
    checks["room_dimensions"] = len(errors) == 0

    for i, a in enumerate(rooms):
        for b in rooms[i + 1 :]:
            area = rect_overlap_area(a, b)
            if area > EPS:
                errors.append(f"Sobreposicao de comodos: {a['id']} x {b['id']} = {area}")
    checks["room_floor_no_overlap"] = not any("Sobreposicao de comodos" in e for e in errors)

    total_room_area = sum(item["area"] for item in rooms)
    if abs(total_room_area - plan["footprint"]["area"]) > EPS:
        errors.append(f"Area dos comodos {total_room_area} difere da area da casa {plan['footprint']['area']}")
    checks["rooms_cover_footprint_area"] = abs(total_room_area - plan["footprint"]["area"]) <= EPS

    validate_wall_connections(walls, errors, checks)
    validate_opening_like_items(openings, wall_by_id, "abertura", errors)
    validate_opening_like_items(windows, wall_by_id, "janela", errors)
    validate_wall_opening_overlap(openings, windows, wall_by_id, errors)
    validate_outdoor(plan, errors, checks)
    validate_roof(plan, errors, checks)
    validate_furniture(plan, errors, checks)
    validate_vehicles(plan, errors, checks)

    if any(item["has_physical_door"] for item in openings):
        errors.append("Ha porta fisica registrada, mas deveriam existir somente vaos.")

    checks["door_openings_without_leaf"] = not any(item["has_physical_door"] for item in openings)
    plan["validation"] = {"ok": len(errors) == 0, "checks": checks, "errors": errors}
    if errors:
        raise ValueError("Falhas de validacao:\n- " + "\n- ".join(errors))


def validate_wall_connections(walls, errors, checks):
    external = [wall for wall in walls if wall["type"] == "external"]
    polygon = [(0.0, 0.0), (16.0, 0.0), (16.0, 6.0), (7.0, 6.0), (7.0, 16.0), (0.0, 16.0), (0.0, 0.0)]
    ext_points = []
    for wall in external:
        ext_points.append((wall["start"][0], wall["start"][1]))
        ext_points.append((wall["end"][0], wall["end"][1]))
    for point in polygon[:-1]:
        if not any(point_close(point, p) for p in ext_points):
            errors.append(f"Canto externo sem parede conectada: {point}")
    checks["external_corners_connected"] = not any("Canto externo" in e for e in errors)

    for wall in walls:
        for key in ("start", "end"):
            p = (wall[key][0], wall[key][1])
            if not endpoint_supported(p, wall["id"], walls):
                errors.append(f"Extremidade de parede sem conexao: {wall['id']}:{key} em {p}")
    checks["wall_endpoints_supported"] = not any("Extremidade de parede" in e for e in errors)

    for i, a in enumerate(walls):
        for b in walls[i + 1 :]:
            if walls_cross_improperly(a, b):
                errors.append(f"Paredes atravessando uma a outra: {a['id']} x {b['id']}")
    checks["no_improper_wall_crossings"] = not any("atravessando" in e for e in errors)


def endpoint_supported(point, own_wall_id, walls):
    for wall in walls:
        if wall["id"] == own_wall_id:
            continue
        start = (wall["start"][0], wall["start"][1])
        end = (wall["end"][0], wall["end"][1])
        if point_close(point, start) or point_close(point, end):
            return True
        if point_on_segment(point, start, end):
            return True
    return False


def point_on_segment(point, start, end):
    axis = wall_axis(start, end)
    if axis == "x":
        return abs(point[1] - start[1]) <= EPS and between(point[0], start[0], end[0])
    return abs(point[0] - start[0]) <= EPS and between(point[1], start[1], end[1])


def walls_cross_improperly(a, b):
    if a["axis"] == b["axis"]:
        return False
    horizontal = a if a["axis"] == "x" else b
    vertical = b if a["axis"] == "x" else a
    x = vertical["start"][0]
    y = horizontal["start"][1]
    hs = (horizontal["start"][0], horizontal["start"][1])
    he = (horizontal["end"][0], horizontal["end"][1])
    vs = (vertical["start"][0], vertical["start"][1])
    ve = (vertical["end"][0], vertical["end"][1])
    if not (between(x, hs[0], he[0]) and between(y, vs[1], ve[1])):
        return False
    at_h_end = point_close((x, y), hs) or point_close((x, y), he)
    at_v_end = point_close((x, y), vs) or point_close((x, y), ve)
    return not (at_h_end or at_v_end)


def validate_opening_like_items(items, wall_by_id, label, errors):
    for item in items:
        wall = wall_by_id.get(item["wall"])
        if wall is None:
            errors.append(f"{label} referencia parede inexistente: {item['id']}")
            continue
        c = item["center_along_wall"]
        half = item["width"] / 2
        if wall["axis"] == "x":
            a0, a1 = wall["start"][0], wall["end"][0]
        else:
            a0, a1 = wall["start"][1], wall["end"][1]
        if c - half < a0 + 0.04 - EPS or c + half > a1 - 0.04 + EPS:
            errors.append(f"{label} fora dos limites da parede: {item['id']} em {item['wall']}")
        z0 = item.get("bottom_z", item.get("sill_height", 0.0))
        z1 = z0 + item["height"]
        if z0 < -EPS or z1 > wall["height"] + EPS:
            errors.append(f"{label} fora da altura da parede: {item['id']} em {item['wall']}")


def validate_wall_opening_overlap(openings, windows, wall_by_id, errors):
    by_wall = {}
    for item in openings + windows:
        by_wall.setdefault(item["wall"], []).append(item)
    for wall_id, items in by_wall.items():
        wall = wall_by_id[wall_id]
        spans = []
        for item in items:
            c = item["center_along_wall"]
            spans.append((c - item["width"] / 2, c + item["width"] / 2, item["id"]))
        for i, a in enumerate(spans):
            for b in spans[i + 1 :]:
                if interval_overlap(a[0], a[1], b[0], b[1]) > EPS:
                    errors.append(f"Aberturas sobrepostas na parede {wall['id']}: {a[2]} x {b[2]}")


def validate_outdoor(plan, errors, checks):
    outdoor = plan["outdoor"]
    house_parts = plan["footprint"]["components"]
    protected = [
        outdoor["area_carro"],
        outdoor["borda_piscina"],
        outdoor["area_reuniao"],
    ]
    grass = [item for item in outdoor.values() if item["kind"] == "grass"]

    for grass_rect in grass:
        for part in house_parts:
            if rect_overlap_area(grass_rect, part) > EPS:
                errors.append(f"Grama entra na casa: {grass_rect['id']} x {part['id']}")
        for item in protected:
            if rect_overlap_area(grass_rect, item) > EPS:
                errors.append(f"Grama sobre {item['id']}: {grass_rect['id']}")

    if rect_overlap_area(outdoor["area_carro"], outdoor["borda_piscina"]) > EPS:
        errors.append("Area do carro sobrepoe piscina.")
    if rect_overlap_area(outdoor["area_carro"], outdoor["area_reuniao"]) > EPS:
        errors.append("Area do carro sobrepoe area de reuniao.")

    checks["grass_outside_house_pool_car_patio"] = not any("Grama" in e for e in errors)
    checks["outdoor_areas_no_bad_overlap"] = not any("Area do carro" in e for e in errors)


def validate_roof(plan, errors, checks):
    expected = [
        {"x": [0.0, 16.0], "y": [0.0, 6.0]},
        {"x": [0.0, 7.0], "y": [6.0, 16.0]},
    ]
    for roof, exp in zip(plan["roof"], expected):
        bounds = roof["aligned_wall_bounds"]
        if abs(bounds["x"][0] - exp["x"][0]) > EPS or abs(bounds["x"][1] - exp["x"][1]) > EPS:
            errors.append(f"Telhado desalinhado em x: {roof['id']}")
        if abs(bounds["y"][0] - exp["y"][0]) > EPS or abs(bounds["y"][1] - exp["y"][1]) > EPS:
            errors.append(f"Telhado desalinhado em y: {roof['id']}")
        if abs(roof["center"][2] - WALL_HEIGHT) > EPS:
            errors.append(f"Telhado nao nasce no topo das paredes: {roof['id']}")
    checks["roof_aligned_to_walls"] = not any("Telhado" in e for e in errors)


def local_box(name, htm, xyz, width, depth, height, color, opacity=1.0, rotation=None):
    local_htm = htm @ ub.Utils.trn(xyz)
    if rotation is not None:
        local_htm = local_htm @ rotation
    return ub.Box(
        name=name,
        htm=local_htm,
        width=width,
        depth=depth,
        height=height,
        color=color,
        opacity=opacity,
    )


def local_cylinder_z(name, htm, xyz, radius, height, color, opacity=1.0):
    return ub.Cylinder(
        name=name,
        htm=htm @ ub.Utils.trn(xyz),
        radius=radius,
        height=height,
        color=color,
        opacity=opacity,
    )


def local_cylinder_y(name, htm, xyz, radius, length, color, opacity=1.0):
    return ub.Cylinder(
        name=name,
        htm=htm @ ub.Utils.trn(xyz) @ ub.Utils.rotx(math.pi / 2),
        radius=radius,
        height=length,
        color=color,
        opacity=opacity,
    )


def tree_base_htm(htm):
    return np.eye(4) if htm is None else htm


def tree_segment_htm(p0, p1):
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)

    direction = p1 - p0
    length = np.linalg.norm(direction)
    if length < 1e-9:
        raise ValueError("Cylinder segment must have nonzero length.")

    z_axis = direction / length
    helper = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(z_axis, helper)) > 0.95:
        helper = np.array([1.0, 0.0, 0.0])

    x_axis = np.cross(helper, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    transform = np.eye(4)
    transform[0:3, 0] = x_axis
    transform[0:3, 1] = y_axis
    transform[0:3, 2] = z_axis
    transform[0:3, 3] = 0.5 * (p0 + p1)
    return transform, length


def tree_cylinder_between(name, htm, p0, p1, radius, color, opacity=1.0):
    local_htm, length = tree_segment_htm(p0, p1)
    return ub.Cylinder(
        name=name,
        htm=tree_base_htm(htm) @ local_htm,
        radius=radius,
        height=length,
        color=color,
        opacity=opacity,
    )


def tree_ball(name, htm, xyz, radius, color, opacity=1.0):
    return ub.Ball(
        name=name,
        htm=tree_base_htm(htm) @ ub.Utils.trn(xyz),
        radius=radius,
        color=color,
        opacity=opacity,
    )


def create_large_tree(
    htm=None,
    name="large_tree",
    height=8.0,
    crown_radius=3.0,
    trunk_radius=0.45,
    branches_per_level=5,
    include_foliage=True,
    seed=7,
    trunk_color="#6B4423",
    branch_color="#76502B",
    leaf_colors=(
        "#2E7D32",
        "#388E3C",
        "#43A047",
        "#2F6F35",
    ),
):
    if height <= 0:
        raise ValueError("'height' must be positive.")
    if crown_radius <= 0:
        raise ValueError("'crown_radius' must be positive.")
    if trunk_radius <= 0:
        raise ValueError("'trunk_radius' must be positive.")

    rng = np.random.default_rng(seed)
    objs = []

    trunk_points = np.array([
        [0.00, 0.00, 0.00 * height],
        [0.03, -0.01, 0.19 * height],
        [-0.04, 0.04, 0.37 * height],
        [0.05, 0.01, 0.54 * height],
        [-0.03, -0.04, 0.69 * height],
        [0.00, 0.00, 0.82 * height],
    ])
    trunk_radii = trunk_radius * np.array([1.00, 0.91, 0.80, 0.67, 0.52])

    for i in range(len(trunk_points) - 1):
        objs.append(
            tree_cylinder_between(
                f"{name}_trunk_{i + 1}",
                htm,
                trunk_points[i],
                trunk_points[i + 1],
                trunk_radii[i],
                trunk_color,
            )
        )

    level_z = np.array([0.38, 0.48, 0.58, 0.67, 0.75]) * height
    level_scale = [1.00, 1.00, 0.94, 0.84, 0.72]
    foliage_centers = []
    branch_id = 0
    twig_id = 0

    for level, (z0, scale) in enumerate(zip(level_z, level_scale)):
        angle_offset = level * 0.61
        for j in range(branches_per_level):
            branch_id += 1
            theta = (
                2 * np.pi * j / branches_per_level
                + angle_offset
                + rng.uniform(-0.20, 0.20)
            )
            horizontal = crown_radius * scale * rng.uniform(0.74, 1.00)
            rise = height * rng.uniform(0.055, 0.11)
            p0 = np.array([
                rng.uniform(-0.05, 0.05),
                rng.uniform(-0.05, 0.05),
                z0,
            ])
            p1 = p0 + np.array([
                horizontal * np.cos(theta),
                horizontal * np.sin(theta),
                rise,
            ])
            branch_radius = trunk_radius * (0.36 - 0.035 * level)

            objs.append(
                tree_cylinder_between(
                    f"{name}_branch_{branch_id}",
                    htm,
                    p0,
                    p1,
                    branch_radius,
                    branch_color,
                )
            )
            foliage_centers.append(p1)

            for side in (-1.0, 1.0):
                twig_id += 1
                start_ratio = rng.uniform(0.52, 0.68)
                q0 = p0 + start_ratio * (p1 - p0)
                twig_angle = theta + side * rng.uniform(0.38, 0.70)
                twig_length = horizontal * rng.uniform(0.34, 0.50)
                twig_rise = height * rng.uniform(0.025, 0.070)
                q1 = q0 + np.array([
                    twig_length * np.cos(twig_angle),
                    twig_length * np.sin(twig_angle),
                    twig_rise,
                ])

                objs.append(
                    tree_cylinder_between(
                        f"{name}_twig_{twig_id}",
                        htm,
                        q0,
                        q1,
                        branch_radius * 0.52,
                        branch_color,
                    )
                )
                foliage_centers.append(q1)

    top_origin = np.array([0.0, 0.0, 0.70 * height])
    for i in range(7):
        theta = 2 * np.pi * i / 7 + 0.25
        radial = crown_radius * rng.uniform(0.30, 0.55)
        p1 = np.array([
            radial * np.cos(theta),
            radial * np.sin(theta),
            height * rng.uniform(0.88, 0.98),
        ])

        objs.append(
            tree_cylinder_between(
                f"{name}_top_branch_{i + 1}",
                htm,
                top_origin,
                p1,
                trunk_radius * 0.18,
                branch_color,
            )
        )
        foliage_centers.append(p1)

    if include_foliage:
        leaf_id = 0
        for center in foliage_centers:
            leaf_id += 1
            radius = rng.uniform(0.46, 0.68) * crown_radius / 3.0
            offset = np.array([
                rng.uniform(-0.16, 0.16),
                rng.uniform(-0.16, 0.16),
                rng.uniform(-0.05, 0.22),
            ]) * crown_radius
            objs.append(
                tree_ball(
                    f"{name}_leaf_cluster_{leaf_id}",
                    htm,
                    center + offset,
                    radius,
                    leaf_colors[leaf_id % len(leaf_colors)],
                )
            )

        inner_count = 18
        for i in range(inner_count):
            theta = 2 * np.pi * i / inner_count + rng.uniform(-0.15, 0.15)
            radial = crown_radius * rng.uniform(0.18, 0.66)
            center = np.array([
                radial * np.cos(theta),
                radial * np.sin(theta),
                height * rng.uniform(0.62, 0.91),
            ])
            objs.append(
                tree_ball(
                    f"{name}_inner_leaf_{i + 1}",
                    htm,
                    center,
                    rng.uniform(0.52, 0.76) * crown_radius / 3.0,
                    leaf_colors[i % len(leaf_colors)],
                )
            )

    return objs


def create_bathroom_vanity(htm=None, name="bathroom_vanity", width=0.55, depth=0.42, height=0.85):
    base = np.eye(4) if htm is None else htm
    objs = [
        local_box(f"{name}_cabinet", base, [0, 0, height * 0.42], width, depth, height * 0.84, "#ECE8DF"),
        local_box(f"{name}_top", base, [0, 0, height - 0.025], width + 0.04, depth + 0.04, 0.05, "#D6D2C8"),
        local_box(f"{name}_basin", base, [0, -depth * 0.05, height + 0.025], width * 0.62, depth * 0.48, 0.05, "#F7F7F5"),
        local_cylinder_z(f"{name}_drain", base, [0, -depth * 0.05, height + 0.055], 0.018, 0.006, "#777777"),
        local_cylinder_z(f"{name}_faucet", base, [0, depth * 0.18, height + 0.10], 0.012, 0.12, "#AEB4B8"),
    ]
    return objs


def create_toilet(htm=None, name="toilet", width=0.50, depth=0.70, height=0.78):
    base = np.eye(4) if htm is None else htm
    objs = [
        local_box(f"{name}_tank", base, [0, depth * 0.30, height * 0.62], width * 0.82, depth * 0.18, height * 0.42, "#F5F4F0"),
        local_box(f"{name}_base", base, [0, -depth * 0.10, height * 0.16], width * 0.52, depth * 0.48, height * 0.32, "#F5F4F0"),
        local_cylinder_z(f"{name}_bowl", base, [0, -depth * 0.12, height * 0.38], width * 0.34, height * 0.26, "#F5F4F0"),
        local_cylinder_z(f"{name}_seat", base, [0, -depth * 0.12, height * 0.53], width * 0.36, 0.035, "#E6E4DE"),
    ]
    return objs


def create_shower_box(htm=None, name="shower_box", width=0.78, depth=0.76, height=2.05):
    base = np.eye(4) if htm is None else htm
    frame = "#B6BABD"
    glass = "#A8D3DF"
    objs = [
        local_box(f"{name}_base", base, [0, 0, 0.035], width, depth, 0.07, "#EFEDE6"),
        local_box(f"{name}_back_glass", base, [0, depth / 2 - 0.015, height / 2], width, 0.03, height, glass, opacity=0.32),
        local_box(f"{name}_left_glass", base, [-width / 2 + 0.015, 0, height / 2], 0.03, depth, height, glass, opacity=0.32),
        local_box(f"{name}_front_bar", base, [0, -depth / 2 + 0.015, height * 0.52], width, 0.03, 0.045, frame),
        local_box(f"{name}_top_bar", base, [0, -depth / 2 + 0.015, height - 0.03], width, 0.035, 0.06, frame),
        local_cylinder_y(f"{name}_shower_head", base, [width * 0.22, depth * 0.36, height * 0.82], 0.045, 0.05, "#858B8E"),
    ]
    return objs


def render_furniture_item(item):
    htm = pose(item["center"][0], item["center"][1], item["center"][2], item["orientation"]["yaw_rad"])
    params = item.get("params", {}).copy()
    function_name = item["function"]
    if function_name == "furniture.create_item":
        return create_item(item["item_type"], htm=htm, name=item["id"], **params)
    if function_name == "compositions.create_composition":
        return create_composition(item["composition_type"], htm=htm, name=item["id"], **params)
    if function_name == "local.create_bathroom_vanity":
        return create_bathroom_vanity(htm=htm, name=item["id"], **params)
    if function_name == "local.create_toilet":
        return create_toilet(htm=htm, name=item["id"], **params)
    if function_name == "local.create_shower_box":
        return create_shower_box(htm=htm, name=item["id"], **params)
    raise ValueError(f"Funcao de mobiliario desconhecida: {function_name}")


def render_furniture(plan):
    objects = []
    for item in plan.get("furniture", {}).values():
        objects += render_furniture_item(item)
    return objects


def render_vehicle_item(item):
    htm = pose(item["center"][0], item["center"][1], item["center"][2], item["orientation"]["yaw_rad"])
    params = item.get("params", {}).copy()
    return create_vehicle(item["vehicle_type"], htm=htm, name=item["id"], **params)


def render_vehicles(plan):
    objects = []
    for item in plan.get("vehicles", {}).values():
        objects += render_vehicle_item(item)
    return objects


def render_tree_item(item):
    htm = pose(item["center"][0], item["center"][1], item["center"][2], item["orientation"]["yaw_rad"])
    params = item.get("params", {}).copy()
    function_name = item["function"]
    if function_name == "local.create_large_tree":
        return create_large_tree(htm=htm, name=item["id"], **params)
    raise ValueError(f"Funcao de arvore desconhecida: {function_name}")


def render_trees(plan):
    objects = []
    for item in plan.get("trees", {}).values():
        objects += render_tree_item(item)
    return objects


def render_wall(wall, opening_by_id, window_by_id):
    htm = pose(wall["center"][0], wall["center"][1], 0.0, wall["yaw_rad"])
    doors = [opening_by_id[item] for item in wall["door_openings"]]
    wins = [window_by_id[item] for item in wall["windows"]]
    kwargs = {
        "htm": htm,
        "name": wall["id"],
        "length": wall["length"],
        "height": wall["height"],
        "thickness": wall["thickness"],
        "wall_color": "#E6E0D6",
    }

    if doors and wins:
        door = doors[0]
        win = wins[0]
        return create_wall_with_door_and_window(
            **kwargs,
            door_width=door["width"],
            door_height=door["height"],
            door_offset=door["position"]["local_offset_from_wall_center"],
            include_door=False,
            window_width=win["width"],
            window_height=win["height"],
            sill_height=win["sill_height"],
            window_offset=win["position"]["local_offset_from_wall_center"],
            include_window=True,
        )
    if doors:
        door = doors[0]
        return create_wall_with_door(
            **kwargs,
            door_width=door["width"],
            door_height=door["height"],
            door_offset=door["position"]["local_offset_from_wall_center"],
            include_door=False,
        )
    if wins:
        win = wins[0]
        return create_wall_with_window(
            **kwargs,
            window_width=win["width"],
            window_height=win["height"],
            sill_height=win["sill_height"],
            window_offset=win["position"]["local_offset_from_wall_center"],
            include_window=True,
        )

    return create_wall(
        htm=htm,
        name=wall["id"],
        length=wall["length"],
        height=wall["height"],
        thickness=wall["thickness"],
        color="#E6E0D6",
    )


def add_floor(objects, data, name, floor_type, thickness, top_z=0.0):
    cx, cy, _ = data["center"]
    objects += create_floor(
        htm=pose(cx, cy, top_z),
        name=name,
        width=data["width"],
        depth=data["depth"],
        thickness=thickness,
        color=FLOOR_COLORS[floor_type],
    )


def build_uaibot_objects(plan):
    objects = []
    for floor in plan["floors"]:
        add_floor(objects, floor, floor["id"], floor["floor_type"], FLOOR_THICKNESS)

    outdoor = plan["outdoor"]
    add_floor(objects, outdoor["area_carro"], "pavimento_area_carro", "cimento_pavimentado", OUTDOOR_THICKNESS)
    add_floor(objects, outdoor["borda_piscina"], "borda_piscina", "borda_piscina", OUTDOOR_THICKNESS)
    add_floor(
        objects,
        outdoor["piscina"],
        "agua_piscina",
        "agua_piscina",
        0.03,
        top_z=outdoor["piscina"]["limits"]["z"][1],
    )
    add_floor(objects, outdoor["area_reuniao"], "deck_area_reuniao", "deck_convivio", OUTDOOR_THICKNESS)
    for item in outdoor.values():
        if item["kind"] == "grass":
            add_floor(objects, item, item["id"], "grama", OUTDOOR_THICKNESS)

    opening_by_id = {item["id"]: item for item in plan["door_openings"]}
    window_by_id = {item["id"]: item for item in plan["windows"]}
    for wall in plan["walls"]:
        objects += render_wall(wall, opening_by_id, window_by_id)

    for roof in plan["roof"]:
        objects += create_gable_roof(
            htm=pose(roof["center"][0], roof["center"][1], 0.0, roof["yaw_rad"]),
            name=roof["id"],
            width=roof["width"],
            depth=roof["depth"],
            base_height=WALL_HEIGHT,
            pitch_angle=math.radians(roof["pitch_deg"]),
            overhang=roof["overhang"],
            color="#7A4036",
        )

    objects += create_porch_steps(
        htm=pose(8.85, -0.55),
        name="degrau_entrada_principal",
        width=1.70,
        total_depth=0.70,
        total_height=0.18,
        n_steps=2,
        color="#AFA69A",
    )

    objects += render_furniture(plan)
    objects += render_vehicles(plan)
    objects += render_trees(plan)

    return objects


def save_registry(plan):
    path = PROJECT_DIR / "house_registry.json"
    with path.open("w", encoding="utf-8") as file:
        json.dump(plan, file, indent=2, ensure_ascii=False)
    return path


def save_simulation(objects):
    sim = ub.Simulation(
        objects
    )
    sim.save(str(PROJECT_DIR), "l_house_uaibot")
    return PROJECT_DIR / "l_house_uaibot.html"


def assert_geometry_matches_source_registry(plan):
    source_path = PROJECT_DIR / "house_registry.json"
    if not source_path.exists():
        return

    with source_path.open("r", encoding="utf-8") as file:
        source = json.load(file)

    geometry_keys = [
        "footprint",
        "rooms",
        "walls",
        "door_openings",
        "windows",
        "floors",
        "roof",
    ]
    mismatches = [key for key in geometry_keys if source.get(key) != plan.get(key)]
    if mismatches:
        raise ValueError(
            "O house_registry.json existente nao bate com a geometria "
            f"reconstruida pelo script: {', '.join(mismatches)}"
        )


def print_summary(plan, registry_path, html_path, object_count):
    print("\n=== CASA TERREA GRANDE EM L - RESUMO DA PLANTA ===")
    print(f"Frente: direcao {plan['metadata']['front_direction']}")
    print(f"Area interna total: {plan['footprint']['area']:.2f} m2")
    print(f"Cantos externos: {plan['footprint']['outer_polygon']}")
    print(f"Objetos UAIbot gerados: {object_count}")
    print(f"Registro salvo em: {registry_path}")
    print(f"HTML salvo em: {html_path}")

    print("\n-- Comodos --")
    for item in plan["rooms"]:
        print(
            f"{item['id']}: x={item['limits']['x']} y={item['limits']['y']} "
            f"centro={item['center']} {item['width']}x{item['depth']} m "
            f"area={item['area']} m2 piso={item['floor_type']} "
            f"paredes={len(item['walls'])} vaos={item['access_openings']} janelas={item['windows']}"
        )

    print("\n-- Paredes --")
    for wall in plan["walls"]:
        print(
            f"{wall['id']}: {wall['start']} -> {wall['end']} "
            f"centro={wall['center']} comp={wall['length']} orient={wall['orientation']} "
            f"vaos={wall['door_openings']} janelas={wall['windows']}"
        )

    print("\n-- Vaos de portas, sem folhas --")
    for item in plan["door_openings"]:
        print(
            f"{item['id']}: conecta={item['connects']} parede={item['wall']} "
            f"centro={item['position']['world_center']} largura={item['width']} "
            f"altura={item['height']} limites={item['limits']}"
        )

    print("\n-- Janelas --")
    for item in plan["windows"]:
        print(
            f"{item['id']}: parede={item['wall']} comodos={item['rooms']} "
            f"centro={item['position']['world_center']} largura={item['width']} limites={item['limits']}"
        )

    print("\n-- Areas externas --")
    for item in plan["outdoor"].values():
        print(f"{item['id']}: tipo={item['kind']} x={item['limits']['x']} y={item['limits']['y']} area={item['area']} m2")

    print("\n-- Telhado --")
    for item in plan["roof"]:
        print(
            f"{item['id']}: bounds={item['aligned_wall_bounds']} centro={item['center']} "
            f"width={item['width']} depth={item['depth']} yaw={item['yaw_rad']}"
        )

    print("\n-- Moveis por comodo/area --")
    furniture_by_room = {}
    for item in plan.get("furniture", {}).values():
        furniture_by_room.setdefault(item["room"], []).append(item)

    for room_id in sorted(furniture_by_room):
        print(f"{room_id}:")
        for item in furniture_by_room[room_id]:
            dims = item["dimensions"]
            catalog_type = item.get("item_type", item.get("composition_type", "local"))
            print(
                f"  {item['id']}: funcao={item['function']} "
                f"tipo={catalog_type} centro={item['center']} "
                f"yaw={item['orientation']['yaw_rad']} "
                f"dim={dims['width']}x{dims['depth']}x{dims['height']} "
                f"bbox={item['bounding_box']}"
            )

    print("\n-- Veiculos --")
    for item in plan.get("vehicles", {}).values():
        dims = item["dimensions"]
        print(
            f"{item['id']}: area={item['area']} tipo={item['vehicle_type']} "
            f"centro={item['center']} yaw={item['orientation']['yaw_rad']} "
            f"dim={dims['width']}x{dims['depth']}x{dims['height']} "
            f"bbox={item['bounding_box']} params={item['params']}"
        )

    print("\n-- Arvores --")
    for item in plan.get("trees", {}).values():
        dims = item["dimensions"]
        print(
            f"{item['id']}: funcao={item['function']} centro={item['center']} "
            f"yaw={item['orientation']['yaw_rad']} "
            f"dim={dims['width']}x{dims['depth']}x{dims['height']} "
            f"bbox={item['bounding_box']} params={item['params']}"
        )

    print("\n-- Validacao --")
    print(json.dumps(plan["validation"], indent=2, ensure_ascii=False))


def main():
    plan = build_plan_data()
    assert_geometry_matches_source_registry(plan)
    #validate_plan(plan)
    registry_path = save_registry(plan)
    objects = build_uaibot_objects(plan)
    print(f"Gerados {len(objects)} objetos UAIbot.")
    html_path = save_simulation(objects)
    # print_summary(plan, registry_path, html_path, len(objects))


if __name__ == "__main__":
    main()
