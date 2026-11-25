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
from typing import Any, Dict, List, Tuple, Union

import numpy as np

from synthetic_data_generation_utils import (
    Box,
    Cylinder,
    LidarProjectionModel,
    Scene,
    Sphere,
    compute_look_at_rotation,
    generate_point_cloud,
)


def build_default_room_scene() -> Scene:
    """Builds the 10m x 10m x 3m enclosed room with 4 pillars and 6 porous spheres."""
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
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates sensor poses in a circular ring facing the target point."""
    angles = np.linspace(0.0, 2.0 * np.pi, num_poses, endpoint=False)
    positions: List[np.ndarray] = []
    rotations: List[np.ndarray] = []

    for theta in angles:
        pos = np.array([radius * np.cos(theta), radius * np.sin(theta), height], dtype=np.float64)
        rot = compute_look_at_rotation(eye=pos, target=list(target))
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


def generate_dataset(
    out_dir: Union[str, Path] = "data",
    num_poses: int = 48,
    orbit_radius: float = 0.5,
    orbit_height: float = 1.4,
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

    scene = build_default_room_scene()
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
    )

    dest_msg = f" -> {out_path.resolve()}" if save else " (in-memory)"
    print(f"Generating {num_poses} scans ({sensor.total_rays:,} rays/scan){dest_msg}")

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
        "num_poses": num_poses,
        "orbit_radius": orbit_radius,
        "orbit_height": orbit_height,
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
        scene = build_default_room_scene()
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
    parser.add_argument("--num-poses", type=int, default=48, help="Number of orbiting sensor poses (default: 48)")
    parser.add_argument("--orbit-radius", type=float, default=0.5, help="Sensor orbit radius in meters (default: 0.5)")
    parser.add_argument("--orbit-height", type=float, default=1.4, help="Sensor height Z in meters (default: 1.4)")
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
        num_poses=args.num_poses,
        orbit_radius=args.orbit_radius,
        orbit_height=args.orbit_height,
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
