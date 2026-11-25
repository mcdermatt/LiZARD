#!/usr/bin/env python3
"""
generate_training_data.py
=========================
Generates synthetic multi-view LiDAR training scans for LiZARD and writes them
into a data directory (default: ./data/).

Produces:
  - <out_dir>/scans/scan_000.npz ... scan_<N-1>.npz : Individual pose scans
  - <out_dir>/training_scans.npz                    : Consolidated training tensors
  - <out_dir>/scene_config.json                     : Scene geometry & sensor intrinsics
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.spatial.transform import Rotation as R

from synthetic_data_generation_utils import (
    Box,
    Cylinder,
    LidarProjectionModel,
    Scene,
    Sphere,
    compute_look_at_rotation,
    generate_point_cloud,
)


def add_cabin(
    scene: Scene,
    x_front: float = 4.6,
    y_center: float = 0.0,
    wall_density: float = 100.0,
    glass_density: float = 8.5,
) -> None:
    """Adds a rustic wooden cabin facing -X with an open door, partially transparent windows,
    a pitched gable roof, chimney, and simple interior furniture.

    With glass_density=8.5 across a 0.06m pane, normal-incidence LiDAR rays return
    off the window ~40% of the time (1 - exp(-8.5 * 0.06)) and transmit ~60% of the
    time into the cabin interior, providing genuine multimodal CDF supervision.
    """
    t = 0.20
    depth = 4.0
    x_back = x_front + depth
    x_mid = x_front + 0.5 * depth
    wall_h = 2.80

    # --- Front wall (x = x_front, facing clearing) with centered open doorway + 2 windows ---
    # Doorway cutout: y in [-0.55, +0.55], z in [0.0, 2.15]
    # Front windows:  y in [+/-0.95, +/-1.85], z in [1.05, 1.95]
    scene.add_box(
        center=[x_front, y_center, 2.475],
        size=[t, 4.60, 0.65],
        density=wall_density,
        name="cabin_front_header",
    )
    for side_tag, sgn in (("L", 1.0), ("R", -1.0)):
        scene.add_box(
            center=[x_front, y_center + sgn * 0.75, 1.075],
            size=[t, 0.40, 2.15],
            density=wall_density,
            name=f"cabin_front_pillar_{side_tag}_inner",
        )
        scene.add_box(
            center=[x_front, y_center + sgn * 2.075, 1.075],
            size=[t, 0.45, 2.15],
            density=wall_density,
            name=f"cabin_front_pillar_{side_tag}_outer",
        )
        scene.add_box(
            center=[x_front, y_center + sgn * 1.40, 0.525],
            size=[t, 0.90, 1.05],
            density=wall_density,
            name=f"cabin_front_sill_{side_tag}",
        )
        scene.add_box(
            center=[x_front, y_center + sgn * 1.40, 2.05],
            size=[t, 0.90, 0.20],
            density=wall_density,
            name=f"cabin_front_lintel_{side_tag}",
        )
        scene.add_box(
            center=[x_front, y_center + sgn * 1.40, 1.50],
            size=[0.06, 0.90, 0.90],
            density=glass_density,
            name=f"cabin_window_front_{side_tag}",
        )

    # Open door hinged at (x_front - 0.05, y_center + 0.55), swung 95 deg outward
    door_w, door_h, door_t = 1.05, 2.10, 0.07
    door_yaw_deg = -95.0
    door_rad = np.deg2rad(door_yaw_deg)
    # Local +Y maps to (-sin(rad), cos(rad)); pin +0.5*door_w end to hinge
    hinge_x, hinge_y = x_front - 0.05, y_center + 0.55
    door_cx = hinge_x - 0.5 * door_w * (-np.sin(door_rad))
    door_cy = hinge_y - 0.5 * door_w * np.cos(door_rad)
    scene.add_box(
        center=[door_cx, door_cy, 0.5 * door_h],
        size=[door_t, door_w, door_h],
        rotation=[0.0, 0.0, door_yaw_deg],
        density=wall_density,
        name="cabin_door_open",
    )

    # --- Back wall (solid) ---
    scene.add_box(
        center=[x_back, y_center, 0.5 * wall_h],
        size=[t, 4.60, wall_h],
        density=wall_density,
        name="cabin_wall_back",
    )

    # --- North (+Y) & South (-Y) side walls, each with a partially transparent window ---
    for side_tag, sgn in (("north", 1.0), ("south", -1.0)):
        y_wall = y_center + sgn * 2.20
        scene.add_box(
            center=[x_front + 0.75, y_wall, 0.5 * wall_h],
            size=[1.30, t, wall_h],
            density=wall_density,
            name=f"cabin_{side_tag}_front_pillar",
        )
        scene.add_box(
            center=[x_back - 0.75, y_wall, 0.5 * wall_h],
            size=[1.30, t, wall_h],
            density=wall_density,
            name=f"cabin_{side_tag}_back_pillar",
        )
        scene.add_box(
            center=[x_mid, y_wall, 0.525],
            size=[1.20, t, 1.05],
            density=wall_density,
            name=f"cabin_{side_tag}_sill",
        )
        scene.add_box(
            center=[x_mid, y_wall, 2.375],
            size=[1.20, t, 0.85],
            density=wall_density,
            name=f"cabin_{side_tag}_header",
        )
        scene.add_box(
            center=[x_mid, y_wall, 1.50],
            size=[1.20, 0.06, 0.90],
            density=glass_density,
            name=f"cabin_window_{side_tag}",
        )

    # --- Gable attic infill panels tucked under the pitched roof ---
    for end_tag, x_end in (("front", x_front), ("back", x_back)):
        scene.add_box(
            center=[x_end, y_center, 3.01],
            size=[t, 2.40, 0.42],
            density=wall_density,
            name=f"cabin_gable_{end_tag}_lower",
        )
        scene.add_box(
            center=[x_end, y_center, 3.385],
            size=[t, 1.10, 0.33],
            density=wall_density,
            name=f"cabin_gable_{end_tag}_upper",
        )

    # --- Pitched gable roof + chimney ---
    scene.add_box(
        center=[x_mid, y_center + 1.375, 3.225],
        size=[5.10, 3.05, 0.14],
        rotation=[-24.5, 0.0, 0.0],
        density=wall_density,
        name="cabin_roof_north",
    )
    scene.add_box(
        center=[x_mid, y_center - 1.375, 3.225],
        size=[5.10, 3.05, 0.14],
        rotation=[24.5, 0.0, 0.0],
        density=wall_density,
        name="cabin_roof_south",
    )
    scene.add_box(
        center=[x_mid + 1.0, y_center + 1.15, 3.90],
        size=[0.55, 0.55, 1.40],
        density=wall_density,
        name="cabin_chimney",
    )

    # --- Interior props (visible through open door and semi-transparent windows) ---
    scene.add_box(
        center=[x_mid + 0.4, y_center, 0.78],
        size=[1.10, 1.50, 0.08],
        density=wall_density,
        name="cabin_table_top",
    )
    scene.add_cylinder(
        center=[x_mid + 0.4, y_center, 0.37],
        radius=0.14,
        height=0.74,
        density=wall_density,
        name="cabin_table_pedestal",
    )
    scene.add_box(
        center=[x_back - 0.8, y_center - 1.40, 0.45],
        size=[0.80, 0.80, 0.90],
        density=wall_density,
        name="cabin_crate",
    )


def add_tree(
    scene: Scene,
    base_xy: Tuple[float, float],
    trunk_height: float = 4.2,
    trunk_radius: float = 0.28,
    crown_radius: float = 1.45,
    num_branches: int = 3,
    foliage_density: float = 0.24,
    az_offset_deg: float = 0.0,
    lean_xy: Tuple[float, float] = (0.0, 0.0),
    name_prefix: str = "tree",
) -> None:
    """Adds a multi-part tree with cylindrical trunk, angled branches, and porous foliage spheres."""
    bx, by = float(base_xy[0]), float(base_xy[1])
    base = np.array([bx, by, 0.0], dtype=np.float64)

    trunk_axis = np.array([lean_xy[0], lean_xy[1], 1.0], dtype=np.float64)
    trunk_axis /= np.linalg.norm(trunk_axis)
    trunk_center = base + 0.5 * trunk_height * trunk_axis
    trunk_top = base + trunk_height * trunk_axis

    # Solid woody trunk
    scene.add_cylinder(
        center=trunk_center.tolist(),
        radius=trunk_radius,
        height=trunk_height,
        axis=trunk_axis.tolist(),
        density=80.0,
        name=f"{name_prefix}_trunk",
    )

    # Angled cylindrical branches + side foliage clusters
    if num_branches > 0:
        attach_fracs = np.linspace(0.42, 0.76, num_branches)
        for b_idx in range(num_branches):
            az_deg = az_offset_deg + (360.0 * b_idx / num_branches) + (18.0 if b_idx % 2 else -12.0)
            elev_deg = 24.0 + 14.0 * ((b_idx % 3) / 2.0)
            az = np.deg2rad(az_deg)
            el = np.deg2rad(elev_deg)
            b_dir = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])

            p_attach = base + (attach_fracs[b_idx] * trunk_height) * trunk_axis
            b_len = crown_radius * (0.95 - 0.12 * (b_idx / max(1, num_branches - 1)))
            b_rad = max(0.06, trunk_radius * 0.36)
            b_center = p_attach + 0.5 * b_len * b_dir
            b_tip = p_attach + b_len * b_dir

            scene.add_cylinder(
                center=b_center.tolist(),
                radius=b_rad,
                height=b_len,
                axis=b_dir.tolist(),
                density=60.0,
                name=f"{name_prefix}_branch_{b_idx + 1}",
            )
            # Porous foliage puff at the branch tip
            side_r = crown_radius * (0.56 + 0.06 * (b_idx % 2))
            foliage_center = b_tip + np.array([0.0, 0.0, 0.15])
            scene.add_sphere(
                center=foliage_center.tolist(),
                radius=side_r,
                density=foliage_density,
                name=f"{name_prefix}_foliage_side_{b_idx + 1}",
            )

    # Overlapping porous crown spheres at the top of the trunk
    crown_main_c = trunk_top + np.array([0.0, 0.0, 0.30 * crown_radius])
    scene.add_sphere(
        center=crown_main_c.tolist(),
        radius=crown_radius,
        density=foliage_density,
        name=f"{name_prefix}_crown_main",
    )
    lobe_az = np.deg2rad(az_offset_deg + 45.0)
    crown_lobe_c = trunk_top + np.array([
        0.42 * crown_radius * np.cos(lobe_az),
        0.42 * crown_radius * np.sin(lobe_az),
        0.52 * crown_radius,
    ])
    scene.add_sphere(
        center=crown_lobe_c.tolist(),
        radius=0.76 * crown_radius,
        density=foliage_density * 0.85,
        name=f"{name_prefix}_crown_upper",
    )


def add_chain_link_fence(
    scene: Scene,
    waypoints_xy: List[Tuple[float, float]],
    height: float = 2.30,
    mesh_thickness: float = 0.04,
    mesh_density: float = 14.0,
    post_spacing: float = 2.1,
    post_radius: float = 0.045,
    rail_radius: float = 0.025,
    name_prefix: str = "chain_link_fence",
) -> None:
    """Adds a multi-segment chain-link fence with solid metal posts/rails and
    partially transparent wire-mesh panels.

    With mesh_density=14.0 across mesh_thickness=0.04m, normal-incidence LiDAR rays
    return off the wire mesh ~43% of the time (1 - exp(-14.0 * 0.04)) and transmit
    ~57% of the time to hit trees, shrubs, and terrain behind the fence.
    """
    if len(waypoints_xy) < 2:
        return

    post_coords: List[Tuple[float, float]] = []
    for seg_i in range(len(waypoints_xy) - 1):
        p0 = np.array(waypoints_xy[seg_i], dtype=np.float64)
        p1 = np.array(waypoints_xy[seg_i + 1], dtype=np.float64)
        delta = p1 - p0
        seg_len = float(np.linalg.norm(delta))
        if seg_len < 1e-4:
            continue
        seg_dir = delta / seg_len
        yaw_deg = float(np.rad2deg(np.arctan2(seg_dir[1], seg_dir[0])))
        mid_xy = 0.5 * (p0 + p1)

        # Partially transparent chain-link wire mesh panel (local X = along fence, Y = thickness)
        scene.add_box(
            center=[float(mid_xy[0]), float(mid_xy[1]), 0.5 * height],
            size=[seg_len, mesh_thickness, height - 0.06],
            rotation=[0.0, 0.0, yaw_deg],
            density=mesh_density,
            name=f"{name_prefix}_mesh_seg{seg_i + 1}",
        )

        # Solid horizontal top rail & lower tension rail
        rail_axis = [float(seg_dir[0]), float(seg_dir[1]), 0.0]
        scene.add_cylinder(
            center=[float(mid_xy[0]), float(mid_xy[1]), height - 0.03],
            radius=rail_radius,
            height=seg_len,
            axis=rail_axis,
            density=100.0,
            name=f"{name_prefix}_top_rail_seg{seg_i + 1}",
        )
        scene.add_cylinder(
            center=[float(mid_xy[0]), float(mid_xy[1]), 0.50 * height],
            radius=rail_radius * 0.8,
            height=seg_len,
            axis=rail_axis,
            density=100.0,
            name=f"{name_prefix}_mid_rail_seg{seg_i + 1}",
        )

        # Solid vertical fence posts spaced along segment
        n_spans = max(1, int(round(seg_len / post_spacing)))
        for k in range(n_spans + 1):
            if seg_i > 0 and k == 0:
                continue  # avoid duplicating shared corner post
            xy = p0 + (k / n_spans) * delta
            post_coords.append((float(xy[0]), float(xy[1])))

    for p_idx, (px, py) in enumerate(post_coords):
        scene.add_cylinder(
            center=[px, py, 0.5 * (height + 0.05)],
            radius=post_radius,
            height=height + 0.05,
            axis=[0.0, 0.0, 1.0],
            density=100.0,
            name=f"{name_prefix}_post_{p_idx + 1}",
        )


def build_forest_scene() -> Scene:
    """Builds a ~22m x 22m open-sky forest clearing with a rustic cabin, chain-link fence,
    dense ground shrubs, and multi-part trees.

    Features:
      - Open-sky forest floor with fallen log, boulders, and abundant porous ground shrubs.
      - Rustic cabin on the +X side with an open swung door, 4 partially transparent
        windows (glass density=8.5), pitched gable roof, chimney, and interior props.
      - Big 2-segment semi-transparent chain-link fence (~43% normal-incidence return)
        with solid steel posts/rails occluding the West & North-West trees and shrubs.
      - 12 trees surrounding the clearing with solid cylindrical trunks, angled branches,
        and low-density volumetric foliage canopies at both the top and side branches.
    """
    scene = Scene()

    # 1. Forest floor & natural ground clutter
    scene.add_box(center=[0.0, 0.0, -0.15], size=[24.0, 24.0, 0.30], density=100.0, name="ground")
    scene.add_cylinder(
        center=[-4.2, -3.6, 0.22],
        radius=0.22,
        height=3.4,
        axis=[0.82, 0.57, 0.0],
        density=80.0,
        name="fallen_log",
    )
    scene.add_sphere(center=[-2.4, 4.8, 0.28], radius=0.60, density=100.0, name="boulder_1")
    scene.add_sphere(center=[3.6, -4.6, 0.22], radius=0.45, density=100.0, name="boulder_2")

    # Abundant porous ground shrubs & understory bushes (multi-lobe & standalone)
    shrub_specs = [
        # (name, (x, y, z), radius, density)
        ("understory_bush_1",      ( 3.4,  3.3, 0.55), 0.85, 0.35),
        ("understory_bush_2",      (-4.8,  1.9, 0.60), 0.95, 0.30),
        ("ground_shrub_NW_1",      (-4.2,  3.0, 0.42), 0.72, 0.55),
        ("ground_shrub_NW_2",      (-3.6,  3.2, 0.35), 0.55, 0.45),
        ("ground_shrub_W_1",       (-4.5, -0.5, 0.48), 0.82, 0.40),
        ("ground_shrub_W_2",       (-6.4,  1.4, 0.58), 0.95, 0.35),
        ("ground_shrub_SW_1",      (-2.8, -3.7, 0.45), 0.78, 0.50),
        ("ground_shrub_SW_2",      (-3.4, -4.1, 0.38), 0.62, 0.45),
        ("ground_shrub_S_1",       (-1.5, -3.9, 0.40), 0.70, 0.60),
        ("ground_shrub_S_2",       ( 0.8, -4.2, 0.50), 0.85, 0.38),
        ("ground_shrub_SE_1",      ( 2.6, -4.1, 0.44), 0.76, 0.52),
        ("ground_shrub_SE_2",      ( 3.2, -3.6, 0.34), 0.55, 0.48),
        ("ground_shrub_N_1",       (-0.4,  4.8, 0.52), 0.88, 0.42),
        ("ground_shrub_N_2",       ( 2.4,  4.9, 0.48), 0.80, 0.50),
        ("ground_shrub_NE_1",      ( 4.4,  3.8, 0.46), 0.75, 0.45),
        ("ground_shrub_clearing_1",(-1.2,  3.3, 0.34), 0.58, 0.65),
        ("ground_shrub_clearing_2",( 1.7, -3.3, 0.36), 0.62, 0.55),
        ("ground_shrub_cabin_N",   ( 4.1,  2.7, 0.42), 0.68, 0.58),
        ("ground_shrub_cabin_S",   ( 4.1, -2.8, 0.45), 0.72, 0.52),
    ]
    for s_name, s_xyz, s_r, s_d in shrub_specs:
        scene.add_sphere(center=list(s_xyz), radius=s_r, density=s_d, name=s_name)

    # 2. Big 2-segment semi-transparent chain-link fence occluding West & North-West trees/shrubs
    #    Clear of the r=2.5m sensor orbit while sitting directly in front of tree_sapling_SW,
    #    tree_W1, tree_NW1, tree_N1, and deep forest trees behind them.
    add_chain_link_fence(
        scene,
        waypoints_xy=[(-3.2, -2.9), (-3.2, 3.6), (2.4, 4.4)],
        height=2.30,
        mesh_thickness=0.04,
        mesh_density=14.0,
        post_spacing=2.1,
        name_prefix="chain_link_fence",
    )

    # 3. Cabin on the +X side of the clearing (facing the sensor clearing)
    add_cabin(scene, x_front=4.6, y_center=0.0, wall_density=100.0, glass_density=8.5)

    # 4. Trees ringing the clearing (clear of the r=2.5m orbit and cabin front sightline)
    tree_specs = [
        # (name, (x, y), trunk_h, trunk_r, crown_r, n_branches, foliage_dens, az_off, lean)
        ("tree_sapling_SW", (-3.6, -1.6), 3.0, 0.14, 0.95, 2, 0.18,  20.0, ( 0.03, -0.02)),
        ("tree_N1",         ( 1.2,  5.4), 4.2, 0.28, 1.45, 3, 0.24,  15.0, (-0.02,  0.03)),
        ("tree_NW1",        (-3.8,  4.2), 3.8, 0.25, 1.35, 3, 0.22,  75.0, ( 0.02,  0.02)),
        ("tree_W1",         (-5.4,  0.4), 4.6, 0.34, 1.65, 4, 0.20,  10.0, (-0.03,  0.01)),
        ("tree_SW1",        (-5.0, -4.8), 4.5, 0.28, 1.50, 3, 0.28, 130.0, ( 0.02, -0.03)),
        ("tree_S1",         (-0.6, -5.0), 3.7, 0.24, 1.30, 3, 0.18, 210.0, (-0.02, -0.02)),
        ("tree_SE1",        ( 4.0, -5.6), 4.3, 0.27, 1.45, 3, 0.24, 290.0, ( 0.01, -0.02)),
        ("tree_NE1",        ( 5.6,  5.4), 4.8, 0.30, 1.55, 3, 0.22, 340.0, ( 0.02,  0.02)),
        ("tree_W_deep",     (-8.5,  3.6), 5.2, 0.32, 1.70, 3, 0.25,  45.0, (-0.02,  0.02)),
        ("tree_SW_deep",    (-8.2, -2.8), 5.0, 0.30, 1.60, 3, 0.30, 165.0, ( 0.01, -0.02)),
        ("tree_N_deep",     (-1.8,  8.6), 5.0, 0.30, 1.65, 3, 0.24, 250.0, (-0.01,  0.03)),
        ("tree_S_deep",     ( 1.8, -8.5), 4.9, 0.29, 1.60, 3, 0.26,  95.0, ( 0.02, -0.02)),
    ]
    for name, xy, th, tr, cr, nb, fd, az0, lean in tree_specs:
        add_tree(
            scene,
            base_xy=xy,
            trunk_height=th,
            trunk_radius=tr,
            crown_radius=cr,
            num_branches=nb,
            foliage_density=fd,
            az_offset_deg=az0,
            lean_xy=lean,
            name_prefix=name,
        )

    return scene


def build_default_room_scene() -> Scene:
    """Builds the legacy 10m x 10m x 3m enclosed room with 4 pillars and 6 porous spheres."""
    scene = Scene()
    wall_density = 100.0  # Solid boundaries (near-surface returns)

    # Floor & 4 Walls (10m x 10m x 3m interior)
    scene.add_box(center=[0.0, 0.0, -0.1], size=[10.4, 10.4, 0.2], density=wall_density, name="floor")
    scene.add_box(center=[0.0, 5.1, 1.5],  size=[10.4, 0.2, 3.0],  density=wall_density, name="wall_north")
    scene.add_box(center=[0.0, -5.1, 1.5], size=[10.4, 0.2, 3.0],  density=wall_density, name="wall_south")
    scene.add_box(center=[5.1, 0.0, 1.5],  size=[0.2, 10.4, 3.0],  density=wall_density, name="wall_east")
    scene.add_box(center=[-5.1, 0.0, 1.5], size=[0.2, 10.4, 3.0],  density=wall_density, name="wall_west")

    # 4 Cylinders (Pillars) with density = 1.0
    cyl_density = 1.0
    scene.add_cylinder(center=[2.5, 2.5, 1.5],   radius=0.4, height=3.0, density=cyl_density, name="pillar_NE")
    scene.add_cylinder(center=[-2.5, 2.5, 1.5],  radius=0.4, height=3.0, density=cyl_density, name="pillar_NW")
    scene.add_cylinder(center=[2.5, -2.5, 1.5],  radius=0.4, height=3.0, density=cyl_density, name="pillar_SE")
    scene.add_cylinder(center=[-2.5, -2.5, 1.5], radius=0.4, height=3.0, density=cyl_density, name="pillar_SW")

    # 6 Low-Density Spheres (Porous foliage / clutter with volumetric penetration)
    scene.add_sphere(center=[2.0, 0.0, 1.0],   radius=0.8,  density=0.15, name="sphere_1")
    scene.add_sphere(center=[-2.0, 0.5, 1.2],  radius=0.9,  density=0.10, name="sphere_2")
    scene.add_sphere(center=[0.0, 3.0, 1.5],   radius=1.1,  density=0.08, name="sphere_3")
    scene.add_sphere(center=[0.0, -2.8, 1.0],  radius=0.7,  density=0.20, name="sphere_4")
    scene.add_sphere(center=[3.2, -1.8, 1.3],  radius=1.0,  density=0.12, name="sphere_5")
    scene.add_sphere(center=[-3.0, -2.0, 1.4], radius=0.85, density=0.18, name="sphere_6")

    return scene


def generate_orbit_poses(
    num_poses: int = 48,
    radius: float = 1.5,
    height: float = 1.4,
    target: Tuple[float, float, float] = (0.0, 0.0, 1.2),
    height_variation: float = 0.45,
    max_pitch_deg: float = 6.0,
    max_roll_deg: float = 6.0,
    seed: Optional[int] = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates sensor poses in a circular ring facing the target point, with
    optional Z-height variation and slight local pitch/roll perturbations so
    vertical LiDAR scan rings interleave cleanly across viewpoints.
    """
    rng = np.random.default_rng(seed)
    angles = np.linspace(0.0, 2.0 * np.pi, num_poses, endpoint=False)
    positions: List[np.ndarray] = []
    rotations: List[np.ndarray] = []

    for theta in angles:
        # Smooth multi-cycle vertical wave + slight jitter so every azimuth sees varied heights
        if height_variation > 0.0:
            dz = height_variation * (0.65 * np.sin(3.0 * theta) + 0.35 * rng.uniform(-1.0, 1.0))
        else:
            dz = 0.0
        z = max(0.35, float(height + dz))

        pos = np.array([radius * np.cos(theta), radius * np.sin(theta), z], dtype=np.float64)
        rot = compute_look_at_rotation(eye=pos, target=list(target))

        # Apply slight local sensor-frame roll (about +X) & pitch (about +Y) perturbations
        if max_roll_deg > 0.0 or max_pitch_deg > 0.0:
            roll_deg = rng.uniform(-max_roll_deg, max_roll_deg) if max_roll_deg > 0.0 else 0.0
            pitch_deg = rng.uniform(-max_pitch_deg, max_pitch_deg) if max_pitch_deg > 0.0 else 0.0
            rot = rot @ R.from_euler("xy", [roll_deg, pitch_deg], degrees=True).as_matrix()

        positions.append(pos)
        rotations.append(rot)

    return np.array(positions), np.array(rotations)


def serialize_scene(scene: Scene) -> List[Dict[str, Any]]:
    """Serializes Scene primitives into JSON-compatible dicts."""
    items: List[Dict[str, Any]] = []
    for obj in scene.objects:
        if isinstance(obj, Box):
            items.append({
                "type": "box",
                "name": obj.name,
                "center": obj.center.tolist(),
                "size": obj.size.tolist(),
                "density": obj.density,
                "rotation": obj.rotation.tolist(),
            })
        elif isinstance(obj, Cylinder):
            items.append({
                "type": "cylinder",
                "name": obj.name,
                "center": obj.center.tolist(),
                "radius": obj.radius,
                "height": obj.height,
                "axis": obj.axis.tolist(),
                "has_caps": obj.has_caps,
                "density": obj.density,
            })
        elif isinstance(obj, Sphere):
            items.append({
                "type": "sphere",
                "name": obj.name,
                "center": obj.center.tolist(),
                "radius": obj.radius,
                "density": obj.density,
            })
    return items


def load_scene_from_config(config_path: Path) -> Tuple[Scene, LidarProjectionModel, Dict[str, Any]]:
    """Reconstructs Scene and LidarProjectionModel from scene_config.json."""
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))

    scene = Scene()
    for item in cfg.get("scene_objects", []):
        kind = item["type"]
        if kind == "box":
            scene.add_box(
                center=item["center"],
                size=item["size"],
                density=item["density"],
                rotation=item.get("rotation"),
                name=item.get("name"),
            )
        elif kind == "cylinder":
            scene.add_cylinder(
                center=item["center"],
                radius=item["radius"],
                height=item["height"],
                density=item["density"],
                axis=item.get("axis", [0.0, 0.0, 1.0]),
                has_caps=item.get("has_caps", True),
                name=item.get("name"),
            )
        elif kind == "sphere":
            scene.add_sphere(
                center=item["center"],
                radius=item["radius"],
                density=item["density"],
                name=item.get("name"),
            )

    s_cfg = cfg["sensor"]
    sensor = LidarProjectionModel(
        elevation_min=s_cfg["elevation_min"],
        elevation_max=s_cfg["elevation_max"],
        num_detectors=s_cfg["num_detectors"],
        num_azimuth=s_cfg["num_azimuth"],
        azimuth_min=s_cfg.get("azimuth_min", -180.0),
        azimuth_max=s_cfg.get("azimuth_max", 180.0),
        min_range=s_cfg.get("min_range", 0.05),
        max_range=s_cfg.get("max_range", 30.0),
    )
    return scene, sensor, cfg


def build_scene(scene_type: str = "forest") -> Scene:
    """Factory for supported synthetic scenes ('forest' or 'room')."""
    if scene_type == "forest":
        return build_forest_scene()
    if scene_type == "room":
        return build_default_room_scene()
    raise ValueError(f"Unknown scene_type '{scene_type}'. Expected 'forest' or 'room'.")


def generate_dataset(
    out_dir: Union[str, Path] = "data",
    scene_type: str = "forest",
    num_poses: int = 48,
    orbit_radius: Optional[float] = None,
    orbit_height: float = 1.4,
    height_variation: float = 0.45,
    max_pitch_deg: float = 6.0,
    max_roll_deg: float = 6.0,
    num_detectors: int = 32,
    num_azimuth: int = 180,
    elev_min: float = -25.0,
    elev_max: float = 15.0,
    max_range: float = 30.0,
    seed: int = 42,
    save: bool = True,
) -> Dict[str, Any]:
    """Generates multi-view synthetic LiDAR scans, optionally saving them to disk."""
    out_path = Path(out_dir)
    scans_dir = out_path / "scans"
    if save:
        scans_dir.mkdir(parents=True, exist_ok=True)

    if orbit_radius is None:
        orbit_radius = 2.5 if scene_type == "forest" else 0.5

    scene = build_scene(scene_type)
    sensor = LidarProjectionModel(
        elevation_min=elev_min,
        elevation_max=elev_max,
        num_detectors=num_detectors,
        num_azimuth=num_azimuth,
        max_range=max_range,
    )
    positions, rotations = generate_orbit_poses(
        num_poses=num_poses,
        radius=orbit_radius,
        height=orbit_height,
        height_variation=height_variation,
        max_pitch_deg=max_pitch_deg,
        max_roll_deg=max_roll_deg,
        seed=seed,
    )

    dest_msg = f" -> {out_path.resolve()}" if save else " (in-memory)"
    print(
        f"Generating {num_poses} scans of '{scene_type}' scene ({len(scene)} primitives, "
        f"{sensor.total_rays:,} rays/scan, orbit_r={orbit_radius:.2f}m, "
        f"z={orbit_height:.2f}+/-{height_variation:.2f}m, "
        f"pitch/roll=+/-{max_pitch_deg:.1f}/{max_roll_deg:.1f}deg){dest_msg}"
    )

    all_rays_o = []
    all_rays_d = []
    all_hit_ranges = []
    all_hit_masks = []
    all_pose_indices = []

    for i in range(num_poses):
        pc = generate_point_cloud(
            sensor=sensor,
            scene=scene,
            sensor_position=positions[i],
            sensor_rotation=rotations[i],
            method="sample",
            return_placement="volume",
            return_misses=True,
            random_seed=seed + i,
        )

        rays_o = pc.rays_o.astype(np.float32)
        rays_d = pc.ray_directions.astype(np.float32)
        hit_ranges = pc.ranges.astype(np.float32)
        hit_masks = (pc.probabilities > 0.0).astype(np.float32)

        if save:
            np.savez_compressed(
                scans_dir / f"scan_{i:03d}.npz",
                rays_o=rays_o,
                rays_d=rays_d,
                hit_ranges=hit_ranges,
                hit_masks=hit_masks,
                sensor_position=positions[i].astype(np.float32),
                sensor_rotation=rotations[i].astype(np.float32),
            )

        all_rays_o.append(rays_o)
        all_rays_d.append(rays_d)
        all_hit_ranges.append(hit_ranges)
        all_hit_masks.append(hit_masks)
        all_pose_indices.append(np.full(len(rays_o), i, dtype=np.int32))

    rays_o_cat = np.concatenate(all_rays_o, axis=0)
    rays_d_cat = np.concatenate(all_rays_d, axis=0)
    hit_ranges_cat = np.concatenate(all_hit_ranges, axis=0)
    hit_masks_cat = np.concatenate(all_hit_masks, axis=0)
    pose_idx_cat = np.concatenate(all_pose_indices, axis=0)
    positions_f32 = positions.astype(np.float32)
    rotations_f32 = rotations.astype(np.float32)

    valid_hits = hit_masks_cat > 0.0
    max_hit = float(hit_ranges_cat[valid_hits].max()) if np.any(valid_hits) else 0.0

    config = {
        "scene_type": scene_type,
        "num_poses": num_poses,
        "orbit_radius": orbit_radius,
        "orbit_height": orbit_height,
        "height_variation": height_variation,
        "max_pitch_deg": max_pitch_deg,
        "max_roll_deg": max_roll_deg,
        "total_rays": int(len(rays_o_cat)),
        "hit_rate": float(valid_hits.mean()),
        "max_hit_range": max_hit,
        "seed": seed,
        "sensor": {
            "elevation_min": sensor.elevation_min_deg,
            "elevation_max": sensor.elevation_max_deg,
            "num_detectors": sensor.num_detectors,
            "num_azimuth": sensor.num_azimuth,
            "azimuth_min": sensor.azimuth_min_deg,
            "azimuth_max": sensor.azimuth_max_deg,
            "min_range": sensor.min_range,
            "max_range": sensor.max_range,
        },
        "scene_objects": serialize_scene(scene),
    }

    if save:
        consolidated_path = out_path / "training_scans.npz"
        np.savez(
            consolidated_path,
            rays_o=rays_o_cat,
            rays_d=rays_d_cat,
            hit_ranges=hit_ranges_cat,
            hit_masks=hit_masks_cat,
            pose_indices=pose_idx_cat,
            sensor_positions=positions_f32,
            sensor_rotations=rotations_f32,
        )
        config_path = out_path / "scene_config.json"
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

        print(
            f"Done! Saved {len(rays_o_cat):,} rays ({valid_hits.mean() * 100:.1f}% hits, "
            f"max hit range {max_hit:.2f}m)\n"
            f"  - Per-pose scans: {scans_dir} ({num_poses} files)\n"
            f"  - Consolidated:   {consolidated_path}\n"
            f"  - Scene config:   {config_path}"
        )
    else:
        print(
            f"Done! Generated {len(rays_o_cat):,} rays ({valid_hits.mean() * 100:.1f}% hits, "
            f"max hit range {max_hit:.2f}m)"
        )

    return {
        "scene": scene,
        "sensor": sensor,
        "rays_o": rays_o_cat,
        "rays_d": rays_d_cat,
        "hit_ranges": hit_ranges_cat,
        "hit_masks": hit_masks_cat,
        "pose_indices": pose_idx_cat,
        "positions": positions_f32,
        "rotations": rotations_f32,
        "config": config,
    }


def load_training_dataset(data_dir: Union[str, Path] = "data") -> Dict[str, Any]:
    """Loads consolidated scans, poses, Scene, and LidarProjectionModel from data_dir."""
    data_path = Path(data_dir)
    consolidated_path = data_path / "training_scans.npz"
    config_path = data_path / "scene_config.json"

    if not consolidated_path.exists():
        raise FileNotFoundError(
            f"Missing '{consolidated_path}'. Run generate_training_data.py first."
        )

    with np.load(consolidated_path) as data:
        rays_o = data["rays_o"].astype(np.float32)
        rays_d = data["rays_d"].astype(np.float32)
        hit_ranges = data["hit_ranges"].astype(np.float32)
        hit_masks = data["hit_masks"].astype(np.float32)
        pose_indices = data["pose_indices"].astype(np.int32) if "pose_indices" in data else None
        positions = data["sensor_positions"].astype(np.float32) if "sensor_positions" in data else None
        rotations = data["sensor_rotations"].astype(np.float32) if "sensor_rotations" in data else None

    if config_path.exists():
        scene, sensor, config = load_scene_from_config(config_path)
    else:
        scene = build_forest_scene()
        sensor = LidarProjectionModel(
            elevation_min=-25.0,
            elevation_max=15.0,
            num_detectors=32,
            num_azimuth=180,
            max_range=30.0,
        )
        config = {}

    return {
        "scene": scene,
        "sensor": sensor,
        "rays_o": rays_o,
        "rays_d": rays_d,
        "hit_ranges": hit_ranges,
        "hit_masks": hit_masks,
        "pose_indices": pose_indices,
        "positions": positions,
        "rotations": rotations,
        "config": config,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic multi-view LiDAR training scans into ./data/"
    )
    parser.add_argument("--out-dir", type=str, default="data", help="Output directory for training scans (default: data)")
    parser.add_argument(
        "--scene",
        type=str,
        choices=["forest", "room"],
        default="forest",
        help="Synthetic scene preset: 'forest' (cabin + multi-branch trees) or 'room' (legacy 10x10m box) (default: forest)",
    )
    parser.add_argument("--num-poses", type=int, default=48, help="Number of orbiting sensor poses (default: 48)")
    parser.add_argument(
        "--orbit-radius",
        type=float,
        default=None,
        help="Sensor orbit radius in meters (default: 2.5 for forest, 0.5 for room)",
    )
    parser.add_argument("--orbit-height", type=float, default=1.4, help="Base sensor height Z in meters (default: 1.4)")
    parser.add_argument("--height-variation", type=float, default=0.45, help="Max +/- Z height variation around orbit height in meters (default: 0.45)")
    parser.add_argument("--max-pitch", type=float, default=6.0, help="Max +/- random sensor pitch perturbation in degrees (default: 6.0)")
    parser.add_argument("--max-roll", type=float, default=6.0, help="Max +/- random sensor roll perturbation in degrees (default: 6.0)")
    parser.add_argument("--num-detectors", type=int, default=32, help="Vertical LiDAR channels (default: 32)")
    parser.add_argument("--num-azimuth", type=int, default=180, help="Azimuthal rays per rotation (default: 180)")
    parser.add_argument("--elev-min", type=float, default=-25.0, help="Minimum elevation angle in degrees (default: -25.0)")
    parser.add_argument("--elev-max", type=float, default=15.0, help="Maximum elevation angle in degrees (default: 15.0)")
    parser.add_argument("--max-range", type=float, default=30.0, help="Sensor max range in meters (default: 30.0)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for stochastic volumetric sampling (default: 42)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_dataset(
        out_dir=args.out_dir,
        scene_type=args.scene,
        num_poses=args.num_poses,
        orbit_radius=args.orbit_radius,
        orbit_height=args.orbit_height,
        height_variation=args.height_variation,
        max_pitch_deg=args.max_pitch,
        max_roll_deg=args.max_roll,
        num_detectors=args.num_detectors,
        num_azimuth=args.num_azimuth,
        elev_min=args.elev_min,
        elev_max=args.elev_max,
        max_range=args.max_range,
        seed=args.seed,
        save=True,
    )


if __name__ == "__main__":
    main()
