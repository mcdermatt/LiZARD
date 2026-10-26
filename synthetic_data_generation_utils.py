"""
synthetic_data_generation_utils.py

Utilities for synthetic LiDAR data generation from prospective 3D scenes.
Designed for probabilistic LiDAR modeling, CDF estimation, and point cloud synthesis.
"""

from typing import List, Tuple, Optional, Union, Dict, Any
import numpy as np
from scipy.spatial.transform import Rotation as R


# ==============================================================================
# 1. LiDAR Projection Model (Sensor Intrinsics)
# ==============================================================================

class LidarProjectionModel:
    """
    Defines LiDAR sensor projection intrinsics.
    
    Parameters
    ----------
    elevation_min : float
        Minimum elevation angle (default in degrees, e.g. -15.0).
    elevation_max : float
        Maximum elevation angle (default in degrees, e.g. 15.0).
    num_detectors : int
        Number of vertical detectors / elevation channels (vertical resolution).
    num_azimuth : int
        Number of points per rotation (azimuthal resolution).
    azimuth_min : float, optional
        Minimum azimuth angle (default -180.0 degrees).
    azimuth_max : float, optional
        Maximum azimuth angle (default 180.0 degrees).
    degrees : bool, optional
        If True (default), elevation and azimuth angles are provided in degrees.
        If False, they are in radians.
    min_range : float, optional
        Minimum valid range (default 0.05 meters).
    max_range : float, optional
        Maximum valid range (default 100.0 meters).
    """

    def __init__(
        self,
        elevation_min: float = -15.0,
        elevation_max: float = 15.0,
        num_detectors: int = 16,
        num_azimuth: int = 360,
        azimuth_min: float = -180.0,
        azimuth_max: float = 180.0,
        degrees: bool = True,
        min_range: float = 0.05,
        max_range: float = 100.0,
    ):
        self.degrees = degrees
        self.num_detectors = int(num_detectors)
        self.num_azimuth = int(num_azimuth)
        self.min_range = float(min_range)
        self.max_range = float(max_range)

        if degrees:
            self.elevation_min_rad = np.deg2rad(elevation_min)
            self.elevation_max_rad = np.deg2rad(elevation_max)
            self.azimuth_min_rad = np.deg2rad(azimuth_min)
            self.azimuth_max_rad = np.deg2rad(azimuth_max)
            self.elevation_min_deg = float(elevation_min)
            self.elevation_max_deg = float(elevation_max)
            self.azimuth_min_deg = float(azimuth_min)
            self.azimuth_max_deg = float(azimuth_max)
        else:
            self.elevation_min_rad = float(elevation_min)
            self.elevation_max_rad = float(elevation_max)
            self.azimuth_min_rad = float(azimuth_min)
            self.azimuth_max_rad = float(azimuth_max)
            self.elevation_min_deg = float(np.rad2deg(elevation_min))
            self.elevation_max_deg = float(np.rad2deg(elevation_max))
            self.azimuth_min_deg = float(np.rad2deg(azimuth_min))
            self.azimuth_max_deg = float(np.rad2deg(azimuth_max))

        # Generate elevation angles (vertical scan channels)
        if self.num_detectors > 1:
            self.elevations = np.linspace(
                self.elevation_min_rad, self.elevation_max_rad, self.num_detectors
            )
        else:
            self.elevations = np.array([(self.elevation_min_rad + self.elevation_max_rad) / 2.0])

        # Generate azimuth angles (horizontal rotation steps)
        # Typically endpoint is excluded for a full 360 degree rotation
        full_circle = np.isclose(abs(self.azimuth_max_rad - self.azimuth_min_rad), 2 * np.pi)
        self.azimuths = np.linspace(
            self.azimuth_min_rad,
            self.azimuth_max_rad,
            self.num_azimuth,
            endpoint=not full_circle,
        )

        # Precompute canonical unit ray directions in sensor local frame
        # Standard sensor frame: X = forward, Y = left, Z = up
        # phi = elevation, theta = azimuth
        elev_grid, azim_grid = np.meshgrid(self.elevations, self.azimuths, indexing="ij")
        # elev_grid shape: (num_detectors, num_azimuth)
        cos_elev = np.cos(elev_grid)
        dx = cos_elev * np.cos(azim_grid)
        dy = cos_elev * np.sin(azim_grid)
        dz = np.sin(elev_grid)

        # Shape: (num_detectors, num_azimuth, 3)
        self._local_ray_directions = np.stack([dx, dy, dz], axis=-1)

    @property
    def total_rays(self) -> int:
        """Total number of rays emitted in a full scan."""
        return self.num_detectors * self.num_azimuth

    def get_ray_directions(
        self,
        rotation: Optional[Union[np.ndarray, R, List[float]]] = None,
        flatten: bool = True,
    ) -> np.ndarray:
        """
        Get unit ray direction vectors in world coordinates.
        
        Parameters
        ----------
        rotation : optional
            Sensor orientation with respect to world frame. Can be:
            - 3x3 rotation matrix
            - scipy.spatial.transform.Rotation object
            - 3 Euler angles [roll, pitch, yaw] in degrees
            - None (identity)
        flatten : bool
            If True, returns shape (total_rays, 3).
            If False, returns shape (num_detectors, num_azimuth, 3).
        """
        dirs = self._local_ray_directions
        if rotation is not None:
            rot_mat = _parse_rotation(rotation)
            # dirs shape: (num_det, num_az, 3)
            # Apply rotation: d_world = rot_mat @ d_local
            dirs = dirs @ rot_mat.T

        if flatten:
            return dirs.reshape(-1, 3)
        return dirs

    def get_rays(
        self,
        sensor_position: Union[np.ndarray, List[float]],
        sensor_rotation: Optional[Union[np.ndarray, R, List[float]]] = None,
        flatten: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate ray origins and unit directions in world coordinates.
        
        Parameters
        ----------
        sensor_position : array-like of shape (3,)
            Sensor xyz coordinates in world frame.
        sensor_rotation : optional
            Sensor orientation in world frame.
        flatten : bool
            If True, returns arrays of shape (total_rays, 3).
        
        Returns
        -------
        rays_o : np.ndarray
            Ray origins in world frame.
        rays_d : np.ndarray
            Ray unit directions in world frame.
        """
        pos = np.asarray(sensor_position, dtype=np.float64).reshape(1, 1, 3)
        dirs = self.get_ray_directions(rotation=sensor_rotation, flatten=False)

        # Broadcast origins to match shape of directions
        origins = np.broadcast_to(pos, dirs.shape).copy()

        if flatten:
            return origins.reshape(-1, 3), dirs.reshape(-1, 3)
        return origins, dirs

    def __repr__(self) -> str:
        return (
            f"LidarProjectionModel(elevation=[{self.elevation_min_deg:.1f}°, {self.elevation_max_deg:.1f}°], "
            f"detectors={self.num_detectors}, azimuth_points={self.num_azimuth}, "
            f"azimuth=[{self.azimuth_min_deg:.1f}°, {self.azimuth_max_deg:.1f}°])"
        )


# Backward compatibility aliases
LidarIntrinsics = LidarProjectionModel
LidarSensor = LidarProjectionModel


# ==============================================================================
# 2. Scene Object Primitives (Box, Sphere, Cylinder)
# ==============================================================================

class SceneObject:
    """
    Abstract base class for all geometric objects in the prospective scene.
    
    Attributes
    ----------
    density : float
        Return likelihood per meter (extinction / attenuation rate lambda >= 0).
    name : str
        Descriptive name of the object.
    """

    def __init__(self, density: float = 1.0, name: Optional[str] = None):
        if density < 0.0:
            raise ValueError("Density must be non-negative.")
        self.density = float(density)
        self.name = name or self.__class__.__name__

    def intersect_rays(
        self, rays_o: np.ndarray, rays_d: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Vectorized ray-primitive intersection.
        
        Parameters
        ----------
        rays_o : np.ndarray of shape (N, 3)
        rays_d : np.ndarray of shape (N, 3)
        
        Returns
        -------
        t_enter : np.ndarray of shape (N,)
            Entry distance along ray (inf if no intersection).
        t_exit : np.ndarray of shape (N,)
            Exit distance along ray (inf if no intersection).
        n_enter : np.ndarray of shape (N, 3)
            Surface normal at entry point in world frame.
        n_exit : np.ndarray of shape (N, 3)
            Surface normal at exit point in world frame.
        """
        raise NotImplementedError

    def to_vedo(self, color="gray", alpha=0.5):
        """Convert object into a vedo 3D actor for interactive visualization."""
        raise NotImplementedError


class Sphere(SceneObject):
    """
    A 3D sphere primitive.
    
    Parameters
    ----------
    center : array-like of shape (3,)
        Center [x, y, z] of the sphere.
    radius : float
        Radius of the sphere.
    density : float
        Return likelihood per meter (>= 0).
    name : str, optional
    """

    def __init__(
        self,
        center: Union[np.ndarray, List[float]],
        radius: float,
        density: float = 1.0,
        name: Optional[str] = None,
    ):
        super().__init__(density=density, name=name)
        self.center = np.asarray(center, dtype=np.float64)
        if radius <= 0.0:
            raise ValueError("Radius must be positive.")
        self.radius = float(radius)

    def intersect_rays(
        self, rays_o: np.ndarray, rays_d: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        N = rays_o.shape[0]
        v = rays_o - self.center  # (N, 3)
        a = np.sum(rays_d**2, axis=-1)  # (N,)
        b = 2.0 * np.sum(v * rays_d, axis=-1)  # (N,)
        c = np.sum(v**2, axis=-1) - self.radius**2  # (N,)

        disc = b**2 - 4.0 * a * c  # (N,)
        valid = disc >= 0.0

        t_enter = np.full(N, np.inf, dtype=np.float64)
        t_exit = np.full(N, np.inf, dtype=np.float64)
        n_enter = np.zeros((N, 3), dtype=np.float64)
        n_exit = np.zeros((N, 3), dtype=np.float64)

        if np.any(valid):
            sqrt_disc = np.sqrt(np.maximum(0.0, disc[valid]))
            a_v = a[valid]
            b_v = b[valid]
            t1 = (-b_v - sqrt_disc) / (2.0 * a_v)
            t2 = (-b_v + sqrt_disc) / (2.0 * a_v)

            # Ensure t1 <= t2
            t_min = np.minimum(t1, t2)
            t_max = np.maximum(t1, t2)

            # Valid if ray hits and exit is in front of ray origin
            in_front = t_max > 1e-6
            val_idx = np.where(valid)[0][in_front]

            if len(val_idx) > 0:
                t_enter[val_idx] = t_min[in_front]
                t_exit[val_idx] = t_max[in_front]

                # Compute normals: n = (p - center) / radius
                p_enter = rays_o[val_idx] + t_enter[val_idx, None] * rays_d[val_idx]
                p_exit = rays_o[val_idx] + t_exit[val_idx, None] * rays_d[val_idx]

                n_enter[val_idx] = (p_enter - self.center) / self.radius
                n_exit[val_idx] = (p_exit - self.center) / self.radius

        return t_enter, t_exit, n_enter, n_exit

    def to_vedo(self, color="red", alpha=0.4):
        try:
            import vedo
            actor = vedo.Sphere(pos=self.center, r=self.radius, c=color, alpha=alpha)
            actor.name = self.name
            return actor
        except ImportError:
            return None

    def __repr__(self) -> str:
        return f"Sphere(center={self.center.tolist()}, radius={self.radius}, density={self.density})"


class Box(SceneObject):
    """
    A 3D Box primitive (can be axis-aligned or oriented).
    
    Parameters
    ----------
    center : array-like of shape (3,)
        Center [x, y, z] of the box.
    size : array-like of shape (3,)
        Full extents [dx, dy, dz] along box principal axes.
    density : float
        Return likelihood per meter (>= 0).
    rotation : optional
        Orientation of the box relative to world axes. Can be:
        - 3x3 rotation matrix
        - scipy.spatial.transform.Rotation object
        - 3 Euler angles [roll, pitch, yaw] in degrees
        - None (axis-aligned)
    name : str, optional
    """

    def __init__(
        self,
        center: Union[np.ndarray, List[float]],
        size: Union[np.ndarray, List[float]],
        density: float = 1.0,
        rotation: Optional[Union[np.ndarray, R, List[float]]] = None,
        name: Optional[str] = None,
    ):
        super().__init__(density=density, name=name)
        self.center = np.asarray(center, dtype=np.float64)
        self.size = np.asarray(size, dtype=np.float64)
        if np.any(self.size <= 0.0):
            raise ValueError("All box dimensions must be positive.")
        self.rotation = _parse_rotation(rotation)

    def intersect_rays(
        self, rays_o: np.ndarray, rays_d: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        N = rays_o.shape[0]
        # Transform rays to box local frame
        # In local frame: center is at origin, box is [-half_size, +half_size]
        h = self.size / 2.0
        o_local = (rays_o - self.center) @ self.rotation
        d_local = rays_d @ self.rotation

        # Kay-Kajiya slab method
        with np.errstate(divide="ignore", invalid="ignore"):
            inv_d = np.where(np.abs(d_local) > 1e-12, 1.0 / d_local, np.inf * np.sign(d_local + 1e-15))
            t_a = (-h - o_local) * inv_d
            t_b = (h - o_local) * inv_d

        t_min_axes = np.minimum(t_a, t_b)  # (N, 3)
        t_max_axes = np.maximum(t_a, t_b)  # (N, 3)

        t_enter_raw = np.max(t_min_axes, axis=-1)  # (N,)
        t_exit_raw = np.min(t_max_axes, axis=-1)  # (N,)

        valid = (t_enter_raw <= t_exit_raw) & (t_exit_raw > 1e-6)

        t_enter = np.full(N, np.inf, dtype=np.float64)
        t_exit = np.full(N, np.inf, dtype=np.float64)
        n_enter = np.zeros((N, 3), dtype=np.float64)
        n_exit = np.zeros((N, 3), dtype=np.float64)

        if np.any(valid):
            val_idx = np.where(valid)[0]
            t_enter[val_idx] = t_enter_raw[val_idx]
            t_exit[val_idx] = t_exit_raw[val_idx]

            # Face normal determination
            # Which axis was hit for entry and exit?
            axis_enter = np.argmax(t_min_axes[val_idx], axis=-1)
            axis_exit = np.argmin(t_max_axes[val_idx], axis=-1)

            n_local_enter = np.zeros((len(val_idx), 3))
            n_local_exit = np.zeros((len(val_idx), 3))

            for i, ax in enumerate(axis_enter):
                # Normal points outward
                n_local_enter[i, ax] = -np.sign(d_local[val_idx[i], ax])
            for i, ax in enumerate(axis_exit):
                n_local_exit[i, ax] = np.sign(d_local[val_idx[i], ax])

            # Convert local normals to world frame: n_world = R @ n_local
            n_enter[val_idx] = n_local_enter @ self.rotation.T
            n_exit[val_idx] = n_local_exit @ self.rotation.T

        return t_enter, t_exit, n_enter, n_exit

    def to_vedo(self, color="blue", alpha=0.4):
        try:
            import vedo
            actor = vedo.Box(pos=self.center, length=self.size[0], width=self.size[1], height=self.size[2], c=color, alpha=alpha)
            # Apply rotation if present
            # vedo Box allows orientation or rotation
            rot_r = R.from_matrix(self.rotation)
            euler = rot_r.as_euler('xyz', degrees=True)
            actor.rotate_x(euler[0]).rotate_y(euler[1]).rotate_z(euler[2])
            actor.name = self.name
            return actor
        except ImportError:
            return None

    def __repr__(self) -> str:
        return f"Box(center={self.center.tolist()}, size={self.size.tolist()}, density={self.density})"


class Cylinder(SceneObject):
    """
    A finite 3D Cylinder primitive with top and bottom circular caps.
    
    Parameters
    ----------
    center : array-like of shape (3,)
        Center [x, y, z] of the cylinder.
    radius : float
        Radius of circular cross section.
    height : float
        Total height / length along the cylinder axis.
    axis : array-like of shape (3,), optional
        Orientation vector of cylinder central axis (default [0, 0, 1] / Z-axis).
    density : float
        Return likelihood per meter (>= 0).
    has_caps : bool, optional
        Whether the cylinder has closed top and bottom end caps (default True).
    name : str, optional
    """

    def __init__(
        self,
        center: Union[np.ndarray, List[float]],
        radius: float,
        height: float,
        density: float = 1.0,
        axis: Union[np.ndarray, List[float]] = (0.0, 0.0, 1.0),
        has_caps: bool = True,
        name: Optional[str] = None,
    ):
        super().__init__(density=density, name=name)
        self.center = np.asarray(center, dtype=np.float64)
        if radius <= 0.0 or height <= 0.0:
            raise ValueError("Radius and height must be positive.")
        self.radius = float(radius)
        self.height = float(height)
        self.has_caps = bool(has_caps)

        # Build orthonormal basis where local Z is aligned with cylinder axis
        axis_vec = np.asarray(axis, dtype=np.float64)
        axis_norm = np.linalg.norm(axis_vec)
        if axis_norm < 1e-9:
            raise ValueError("Axis vector must not be zero.")
        self.axis = axis_vec / axis_norm

        self.rotation = _build_orthonormal_basis(self.axis)

    def intersect_rays(
        self, rays_o: np.ndarray, rays_d: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        N = rays_o.shape[0]
        # Transform rays into local frame where axis is along Z and center is origin
        o_local = (rays_o - self.center) @ self.rotation
        d_local = rays_d @ self.rotation

        half_h = self.height / 2.0
        r_sq = self.radius**2

        t_enter = np.full(N, np.inf, dtype=np.float64)
        t_exit = np.full(N, np.inf, dtype=np.float64)
        n_enter = np.zeros((N, 3), dtype=np.float64)
        n_exit = np.zeros((N, 3), dtype=np.float64)

        # 1. Lateral cylinder intersection: x^2 + y^2 = r^2
        A = d_local[:, 0]**2 + d_local[:, 1]**2
        B = 2.0 * (o_local[:, 0] * d_local[:, 0] + o_local[:, 1] * d_local[:, 1])
        C = o_local[:, 0]**2 + o_local[:, 1]**2 - r_sq

        disc = B**2 - 4.0 * A * C
        has_lateral = (A > 1e-12) & (disc >= 0.0)

        # Array of candidate hit times and local normals for each ray (up to 4 candidates)
        all_hits: List[List[Tuple[float, np.ndarray]]] = [[] for _ in range(N)]

        if np.any(has_lateral):
            lat_indices = np.where(has_lateral)[0]
            sqrt_disc = np.sqrt(np.maximum(0.0, disc[lat_indices]))
            A_sub = A[lat_indices]
            B_sub = B[lat_indices]

            t1 = (-B_sub - sqrt_disc) / (2.0 * A_sub)
            t2 = (-B_sub + sqrt_disc) / (2.0 * A_sub)

            z1 = o_local[lat_indices, 2] + t1 * d_local[lat_indices, 2]
            z2 = o_local[lat_indices, 2] + t2 * d_local[lat_indices, 2]

            for idx, i_ray in enumerate(lat_indices):
                if -half_h <= z1[idx] <= half_h:
                    p1 = o_local[i_ray] + t1[idx] * d_local[i_ray]
                    norm1 = np.array([p1[0] / self.radius, p1[1] / self.radius, 0.0])
                    all_hits[i_ray].append((t1[idx], norm1))
                if -half_h <= z2[idx] <= half_h:
                    p2 = o_local[i_ray] + t2[idx] * d_local[i_ray]
                    norm2 = np.array([p2[0] / self.radius, p2[1] / self.radius, 0.0])
                    all_hits[i_ray].append((t2[idx], norm2))

        # 2. Top and bottom end caps: z = +half_h and z = -half_h
        if self.has_caps:
            non_zero_dz = np.abs(d_local[:, 2]) > 1e-12
            if np.any(non_zero_dz):
                cap_indices = np.where(non_zero_dz)[0]
                t_top = (half_h - o_local[cap_indices, 2]) / d_local[cap_indices, 2]
                t_bot = (-half_h - o_local[cap_indices, 2]) / d_local[cap_indices, 2]

                x_top = o_local[cap_indices, 0] + t_top * d_local[cap_indices, 0]
                y_top = o_local[cap_indices, 1] + t_top * d_local[cap_indices, 1]
                top_inside = (x_top**2 + y_top**2) <= r_sq

                x_bot = o_local[cap_indices, 0] + t_bot * d_local[cap_indices, 0]
                y_bot = o_local[cap_indices, 1] + t_bot * d_local[cap_indices, 1]
                bot_inside = (x_bot**2 + y_bot**2) <= r_sq

                norm_top = np.array([0.0, 0.0, 1.0])
                norm_bot = np.array([0.0, 0.0, -1.0])

                for idx, i_ray in enumerate(cap_indices):
                    if top_inside[idx]:
                        all_hits[i_ray].append((t_top[idx], norm_top))
                    if bot_inside[idx]:
                        all_hits[i_ray].append((t_bot[idx], norm_bot))

        # Sort and select entry and exit for each ray
        for i in range(N):
            hits = all_hits[i]
            if len(hits) >= 2:
                # Sort by distance
                hits.sort(key=lambda item: item[0])
                t_in, n_in = hits[0]
                t_out, n_out = hits[-1]
                if t_out > 1e-6:
                    t_enter[i] = t_in
                    t_exit[i] = t_out
                    # Transform local normals to world frame
                    n_enter[i] = n_in @ self.rotation.T
                    n_exit[i] = n_out @ self.rotation.T
            elif len(hits) == 1:
                # Tangent or numerical border
                t_hit, n_hit = hits[0]
                if t_hit > 1e-6:
                    t_enter[i] = t_hit
                    t_exit[i] = t_hit
                    n_enter[i] = n_hit @ self.rotation.T
                    n_exit[i] = n_hit @ self.rotation.T

        return t_enter, t_exit, n_enter, n_exit

    def to_vedo(self, color="green", alpha=0.4):
        try:
            import vedo
            # Cylinder from base to top
            p1 = self.center - (self.height / 2.0) * self.axis
            p2 = self.center + (self.height / 2.0) * self.axis
            actor = vedo.Cylinder(pos=(p1 + p2) / 2.0, r=self.radius, height=self.height, axis=self.axis, c=color, alpha=alpha)
            actor.name = self.name
            return actor
        except ImportError:
            return None

    def __repr__(self) -> str:
        return (
            f"Cylinder(center={self.center.tolist()}, radius={self.radius}, "
            f"height={self.height}, axis={self.axis.tolist()}, density={self.density})"
        )


# ==============================================================================
# 3. Prospective Scene
# ==============================================================================

class Scene:
    """
    Prospective 3D Scene containing objects with specified geometries and densities.
    Starts empty and can be populated with boxes, spheres, and cylinders.
    """

    def __init__(self):
        self.objects: List[SceneObject] = []

    def add_sphere(
        self,
        center: Union[np.ndarray, List[float]],
        radius: float,
        density: float = 1.0,
        name: Optional[str] = None,
    ) -> Sphere:
        """Add a sphere primitive to the scene."""
        sphere = Sphere(center=center, radius=radius, density=density, name=name)
        self.objects.append(sphere)
        return sphere

    def add_box(
        self,
        center: Union[np.ndarray, List[float]],
        size: Union[np.ndarray, List[float]],
        density: float = 1.0,
        rotation: Optional[Union[np.ndarray, R, List[float]]] = None,
        name: Optional[str] = None,
    ) -> Box:
        """Add a box primitive to the scene."""
        box = Box(center=center, size=size, density=density, rotation=rotation, name=name)
        self.objects.append(box)
        return box

    def add_cylinder(
        self,
        center: Union[np.ndarray, List[float]],
        radius: float,
        height: float,
        density: float = 1.0,
        axis: Union[np.ndarray, List[float]] = (0.0, 0.0, 1.0),
        has_caps: bool = True,
        name: Optional[str] = None,
    ) -> Cylinder:
        """Add a cylinder primitive to the scene."""
        cyl = Cylinder(
            center=center,
            radius=radius,
            height=height,
            density=density,
            axis=axis,
            has_caps=has_caps,
            name=name,
        )
        self.objects.append(cyl)
        return cyl

    def add_object(self, obj: SceneObject) -> SceneObject:
        """Add an instantiated SceneObject to the scene."""
        if not isinstance(obj, SceneObject):
            raise TypeError("Object must be an instance of SceneObject.")
        self.objects.append(obj)
        return obj

    def clear(self):
        """Remove all objects from the scene."""
        self.objects.clear()

    def __len__(self) -> int:
        return len(self.objects)

    def __getitem__(self, idx: int) -> SceneObject:
        return self.objects[idx]

    def to_vedo(self, palette: Optional[List[str]] = None, alpha: float = 0.4) -> List[Any]:
        """Convert all objects in scene into vedo actors for 3D visualization."""
        if palette is None:
            palette = ["red", "blue", "green", "orange", "purple", "cyan", "yellow"]
        actors = []
        for i, obj in enumerate(self.objects):
            color = palette[i % len(palette)]
            actor = obj.to_vedo(color=color, alpha=alpha)
            if actor is not None:
                actors.append(actor)
        return actors

    def __repr__(self) -> str:
        return f"Scene(objects={len(self.objects)})"


# ==============================================================================
# 4. PointCloud Data Container
# ==============================================================================

class PointCloud:
    """
    Point cloud generated by LiDAR raycasting on a prospective scene.
    
    Attributes
    ----------
    points : np.ndarray of shape (N, 3)
        3D Cartesian coordinates of returns.
    probabilities : np.ndarray of shape (N,)
        Calculated return probabilities for each point.
    ranges : np.ndarray of shape (N,)
        Distances from the sensor origin.
    normals : np.ndarray of shape (N, 3)
        Surface normals at contact points.
    ray_directions : np.ndarray of shape (N, 3)
        Unit ray directions from which returns originated.
    rays_o : np.ndarray of shape (N, 3)
        Sensor origin for each return.
    object_indices : np.ndarray of shape (N,)
        Index of the scene object contacted.
    ray_indices : np.ndarray of shape (N,)
        Original linear ray index in the LiDAR projection model.
    metadata : dict
        Additional metadata (sensor position, resolution, etc.).
    """

    def __init__(
        self,
        points: np.ndarray,
        probabilities: np.ndarray,
        ranges: np.ndarray,
        normals: Optional[np.ndarray] = None,
        ray_directions: Optional[np.ndarray] = None,
        rays_o: Optional[np.ndarray] = None,
        object_indices: Optional[np.ndarray] = None,
        ray_indices: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.points = np.asarray(points, dtype=np.float64)
        self.probabilities = np.asarray(probabilities, dtype=np.float64)
        self.ranges = np.asarray(ranges, dtype=np.float64)
        N = len(self.points)

        self.normals = (
            np.asarray(normals, dtype=np.float64)
            if normals is not None
            else np.zeros((N, 3), dtype=np.float64)
        )
        self.ray_directions = (
            np.asarray(ray_directions, dtype=np.float64)
            if ray_directions is not None
            else np.zeros((N, 3), dtype=np.float64)
        )
        self.rays_o = (
            np.asarray(rays_o, dtype=np.float64)
            if rays_o is not None
            else np.zeros((N, 3), dtype=np.float64)
        )
        self.object_indices = (
            np.asarray(object_indices, dtype=np.int32)
            if object_indices is not None
            else -np.ones(N, dtype=np.int32)
        )
        self.ray_indices = (
            np.asarray(ray_indices, dtype=np.int32)
            if ray_indices is not None
            else -np.ones(N, dtype=np.int32)
        )
        self.metadata = metadata or {}

    def __len__(self) -> int:
        return len(self.points)

    def to_xyz(self) -> np.ndarray:
        """Return (N, 3) numpy array of XYZ coordinates."""
        return self.points

    def to_vedo(
        self,
        color_by: str = "probability",
        cmap: str = "viridis",
        point_size: float = 4.0,
    ):
        """
        Convert to vedo.Points for interactive visualization.
        
        Parameters
        ----------
        color_by : str
            Scalar to color points by ('probability', 'range', 'z', or constant color name).
        cmap : str
            Colormap name (e.g. 'viridis', 'plasma', 'jet', 'coolwarm').
        point_size : float
            Rendered size of points.
        """
        try:
            import vedo
            if len(self.points) == 0:
                return vedo.Points(np.empty((0, 3)))

            actor = vedo.Points(self.points, r=point_size)
            if color_by == "probability":
                actor.cmap(cmap, self.probabilities)
                actor.add_scalarbar(title="Return Probability")
            elif color_by == "range":
                actor.cmap(cmap, self.ranges)
                actor.add_scalarbar(title="Range (m)")
            elif color_by == "z":
                actor.cmap(cmap, self.points[:, 2])
                actor.add_scalarbar(title="Z (m)")
            else:
                actor.c(color_by)
            return actor
        except ImportError:
            raise ImportError("vedo is required for to_vedo(). Please install vedo.")

    def save_ply(self, filepath: str):
        """Save point cloud as a standard ASCII PLY file."""
        with open(filepath, "w") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(self.points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property float intensity\n")
            f.write("property float range\n")
            f.write("end_header\n")
            for i in range(len(self.points)):
                p = self.points[i]
                prob = self.probabilities[i]
                r = self.ranges[i]
                f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {prob:.4f} {r:.4f}\n")

    def __repr__(self) -> str:
        return f"PointCloud(N={len(self.points)}, mean_prob={np.mean(self.probabilities) if len(self) else 0.0:.3f})"


# ==============================================================================
# 5. Ray Tracing and Return Probability Computation
# ==============================================================================

def trace_scene_rays(
    rays_o: np.ndarray,
    rays_d: np.ndarray,
    scene: Scene,
    max_range: float = 100.0,
    min_range: float = 0.05,
) -> List[Dict[str, Any]]:
    """
    Traces a batch of rays through the scene, finding all contacted surfaces,
    and calculates the return probability on each surface working outwards from the sensor.
    
    Mathematical Formulation:
    -------------------------
    For each ray r(t) = o + t * d:
    Contacted surfaces are sorted by distance: t_1 < t_2 < ... < t_K.
    For each contact k:
        chord length = path length through the medium
        p_k = 1 - exp(-density * chord)   (interaction probability if ray reaches surface k)
    
    Working outwards from sensor:
        T_1 = 1.0 (initial transmittance)
        For k = 1, ..., K:
            P(return at surface k) = T_k * p_k
            CDF_k = 1 - T_{k+1} = CDF_{k-1} + P(return at surface k)
            T_{k+1} = T_k * (1 - p_k)
        P(no return / sky) = T_{K+1}
    
    Returns
    -------
    list of dicts (one per ray) containing:
        'hits': list of dicts for each contacted surface:
            - 't': distance along ray
            - 'point': 3D coordinates (3,)
            - 'normal': surface normal (3,)
            - 'object_idx': index of object in scene
            - 'object_name': name of object
            - 'density': density of object
            - 'chord': length traversed through object
            - 'local_prob': p_k (probability if reached)
            - 'return_prob': P_k (observed marginal return probability)
            - 'cdf': cumulative return probability up to this surface
        'remaining_transmittance': probability that ray escapes without return
    """
    N = rays_o.shape[0]
    num_objects = len(scene)

    if num_objects == 0:
        return [{"hits": [], "remaining_transmittance": 1.0} for _ in range(N)]

    # Intersect all rays with all objects
    # t_enters: (num_objects, N), t_exits: (num_objects, N)
    t_enters = np.zeros((num_objects, N))
    t_exits = np.zeros((num_objects, N))
    n_enters = np.zeros((num_objects, N, 3))
    n_exits = np.zeros((num_objects, N, 3))

    for obj_i, obj in enumerate(scene.objects):
        t_in, t_out, n_in, n_out = obj.intersect_rays(rays_o, rays_d)
        t_enters[obj_i] = t_in
        t_exits[obj_i] = t_out
        n_enters[obj_i] = n_in
        n_exits[obj_i] = n_out

    results = []

    for ray_i in range(N):
        ray_origin = rays_o[ray_i]
        ray_dir = rays_d[ray_i]

        # Collect all valid contacted objects for this ray
        candidate_hits = []
        for obj_i in range(num_objects):
            t_in = t_enters[obj_i, ray_i]
            t_out = t_exits[obj_i, ray_i]

            # Ray hits object if within sensor range
            if t_out > min_range and t_in < max_range and t_in < t_out:
                # Clamp entry to min_range if sensor is inside the object
                actual_in = max(t_in, min_range)
                actual_out = min(t_out, max_range)
                chord = max(0.0, actual_out - actual_in)

                if chord > 0.0:
                    candidate_hits.append({
                        "t": actual_in,
                        "t_in": actual_in,
                        "t_out": actual_out,
                        "chord": chord,
                        "point": ray_origin + actual_in * ray_dir,
                        "point_in": ray_origin + actual_in * ray_dir,
                        "point_out": ray_origin + actual_out * ray_dir,
                        "normal": n_enters[obj_i, ray_i],
                        "normal_in": n_enters[obj_i, ray_i],
                        "normal_out": n_exits[obj_i, ray_i],
                        "object_idx": obj_i,
                        "object_name": scene.objects[obj_i].name,
                        "density": scene.objects[obj_i].density,
                    })

        # Sort contacts in order of distance from sensor: working outwards
        candidate_hits.sort(key=lambda hit: hit["t_in"])

        # Calculate return probability on each surface
        transmittance = 1.0
        cumulative_prob = 0.0
        processed_hits = []

        for hit in candidate_hits:
            density = hit["density"]
            chord = hit["chord"]

            # Interaction probability on this surface given ray reached it
            # P_local = 1 - exp(-density * chord)
            p_local = 1.0 - np.exp(-density * chord)

            # Marginal return probability observed by sensor
            p_return = transmittance * p_local
            cumulative_prob += p_return

            # Update transmittance past this surface
            transmittance *= (1.0 - p_local)

            processed_hits.append({
                "t": hit["t_in"],
                "t_in": hit["t_in"],
                "t_out": hit["t_out"],
                "chord": chord,
                "point": hit["point"],
                "point_in": hit["point_in"],
                "point_out": hit["point_out"],
                "normal": hit["normal"],
                "normal_in": hit["normal_in"],
                "normal_out": hit["normal_out"],
                "object_idx": hit["object_idx"],
                "object_name": hit["object_name"],
                "density": density,
                "local_prob": float(p_local),
                "return_prob": float(p_return),
                "cdf": float(cumulative_prob),
            })

        results.append({
            "hits": processed_hits,
            "remaining_transmittance": float(transmittance),
        })

    return results


# ==============================================================================
# 6. Point Cloud Generation Method
# ==============================================================================

def generate_point_cloud(
    sensor: LidarProjectionModel,
    scene: Scene,
    sensor_position: Union[np.ndarray, List[float]],
    sensor_rotation: Optional[Union[np.ndarray, R, List[float]]] = None,
    method: str = "sample",
    return_placement: str = "volume",
    min_probability: float = 0.001,
    random_seed: Optional[int] = None,
    return_misses: bool = False,
) -> PointCloud:
    """
    Generates a synthetic LiDAR point cloud for a given sensor intrinsic and scene
    at specified sensor xyz position.
    
    Parameters
    ----------
    sensor : LidarProjectionModel
        Sensor intrinsics (elevation min/max, detectors, azimuthal resolution, range).
    scene : Scene
        Scene containing geometric objects with densities.
    sensor_position : array-like of shape (3,)
        Sensor 3D position [x, y, z] in world coordinates.
    sensor_rotation : optional
        Sensor 3D orientation (rotation matrix, Euler angles, or scipy Rotation).
    method : str, optional
        Strategy for generating point returns from the calculated surface probabilities:
        - 'sample' (default): Stochastic sampling. For each ray, a return occurs with
          probability equal to total CDF, and the return surface is sampled from the
          categorical distribution of surface return probabilities (matches physical LiDAR noise).
        - 'strongest' / 'max_prob': Chooses the surface with maximum return probability.
        - 'first_hit': Returns the first contacted surface if its return probability exceeds min_probability.
        - 'all': Returns all contacted surfaces, with return probabilities as intensities.
        - 'threshold': Returns all surfaces whose return probability exceeds min_probability.
    return_placement : str, optional
        Where along the ray chord inside the object to place the return point:
        - 'volume' (default): Points penetrate inside the 3D volume of the shape according
          to its density lambda (using the truncated exponential distribution). Sparse objects
          (e.g. foliage canopy) have points distributed throughout their volume, while dense
          objects concentrate near the surface.
        - 'surface' (or 'near'): Points are placed strictly on the near entry surface.
    min_probability : float, optional
        Minimum return probability threshold (default 0.001).
    random_seed : int, optional
        Random seed for reproducible stochastic sampling.
    return_misses : bool, optional
        If True, rays that do not return are included at max_range with probability 0.
    
    Returns
    -------
    PointCloud
        Synthesized point cloud with 3D points, return probabilities, ranges, normals, and metadata.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # 1. Generate all rays from sensor intrinsics and pose
    rays_o, rays_d = sensor.get_rays(
        sensor_position=sensor_position,
        sensor_rotation=sensor_rotation,
        flatten=True,
    )

    # 2. Trace all rays through the scene and compute return probabilities
    ray_results = trace_scene_rays(
        rays_o=rays_o,
        rays_d=rays_d,
        scene=scene,
        max_range=sensor.max_range,
        min_range=sensor.min_range,
    )

    # 3. Construct point cloud returns according to specified method
    out_points = []
    out_probs = []
    out_ranges = []
    out_normals = []
    out_ray_dirs = []
    out_rays_o = []
    out_obj_idx = []
    out_ray_idx = []

    method = method.lower()
    return_placement = return_placement.lower()

    for ray_i, result in enumerate(ray_results):
        hits = result["hits"]
        if len(hits) == 0:
            if return_misses:
                miss_pt = rays_o[ray_i] + sensor.max_range * rays_d[ray_i]
                out_points.append(miss_pt)
                out_probs.append(0.0)
                out_ranges.append(sensor.max_range)
                out_normals.append(np.zeros(3))
                out_ray_dirs.append(rays_d[ray_i])
                out_rays_o.append(rays_o[ray_i])
                out_obj_idx.append(-1)
                out_ray_idx.append(ray_i)
            continue

        selected_hits = []

        if method == "sample":
            # Total probability of return for this ray
            probs = np.array([h["return_prob"] for h in hits])
            total_return_prob = np.sum(probs)
            # Roll dice: does the ray produce a return?
            if np.random.rand() < total_return_prob:
                # Sample which surface generated the return
                p_normalized = probs / total_return_prob
                hit_idx = np.random.choice(len(hits), p=p_normalized)
                selected_hits.append(hits[hit_idx])
            elif return_misses:
                miss_pt = rays_o[ray_i] + sensor.max_range * rays_d[ray_i]
                out_points.append(miss_pt)
                out_probs.append(0.0)
                out_ranges.append(sensor.max_range)
                out_normals.append(np.zeros(3))
                out_ray_dirs.append(rays_d[ray_i])
                out_rays_o.append(rays_o[ray_i])
                out_obj_idx.append(-1)
                out_ray_idx.append(ray_i)

        elif method in ("strongest", "max_prob"):
            best_hit = max(hits, key=lambda h: h["return_prob"])
            if best_hit["return_prob"] >= min_probability:
                selected_hits.append(best_hit)

        elif method == "first_hit":
            first_hit = hits[0]
            if first_hit["return_prob"] >= min_probability:
                selected_hits.append(first_hit)

        elif method == "all":
            selected_hits.extend(hits)

        elif method == "threshold":
            for h in hits:
                if h["return_prob"] >= min_probability:
                    selected_hits.append(h)
        else:
            raise ValueError(
                f"Unknown method '{method}'. Choose from: 'sample', 'strongest', 'first_hit', 'all', 'threshold'."
            )

        for h in selected_hits:
            density = h["density"]
            chord = h["chord"]
            t_in = h["t_in"]

            if return_placement in ("volume", "penetration", "inside"):
                if density > 1e-7 and chord > 1e-7:
                    p_chord = 1.0 - np.exp(-density * chord)
                    if method == "sample":
                        # Sample penetration depth inside the volume according to exponential distribution
                        u = np.random.rand()
                        s = -np.log(np.maximum(1e-15, 1.0 - u * p_chord)) / density
                    else:
                        # Expected penetration depth for deterministic modes
                        exp_decay = np.exp(-density * chord)
                        if abs(p_chord) > 1e-7:
                            s = (1.0 / density) - (chord * exp_decay / p_chord)
                        else:
                            s = chord / 2.0
                    s = np.clip(s, 0.0, chord)
                else:
                    s = 0.0
                t_ret = t_in + s
                pt_ret = rays_o[ray_i] + t_ret * rays_d[ray_i]
            else:
                # Near surface placement
                t_ret = t_in
                pt_ret = h["point"]

            out_points.append(pt_ret)
            out_probs.append(h["return_prob"])
            out_ranges.append(t_ret)
            out_normals.append(h["normal"])
            out_ray_dirs.append(rays_d[ray_i])
            out_rays_o.append(rays_o[ray_i])
            out_obj_idx.append(h["object_idx"])
            out_ray_idx.append(ray_i)

    metadata = {
        "sensor": sensor,
        "sensor_position": np.asarray(sensor_position),
        "method": method,
        "total_rays": len(rays_o),
    }

    if len(out_points) == 0:
        return PointCloud(
            points=np.empty((0, 3)),
            probabilities=np.empty(0),
            ranges=np.empty(0),
            normals=np.empty((0, 3)),
            ray_directions=np.empty((0, 3)),
            rays_o=np.empty((0, 3)),
            object_indices=np.empty(0, dtype=np.int32),
            ray_indices=np.empty(0, dtype=np.int32),
            metadata=metadata,
        )

    return PointCloud(
        points=np.array(out_points),
        probabilities=np.array(out_probs),
        ranges=np.array(out_ranges),
        normals=np.array(out_normals),
        ray_directions=np.array(out_ray_dirs),
        rays_o=np.array(out_rays_o),
        object_indices=np.array(out_obj_idx, dtype=np.int32),
        ray_indices=np.array(out_ray_idx, dtype=np.int32),
        metadata=metadata,
    )


# Attach generate_point_cloud directly to LidarProjectionModel and Scene for convenience
LidarProjectionModel.generate_point_cloud = generate_point_cloud
Scene.generate_point_cloud = (
    lambda self, sensor, sensor_position, **kwargs: generate_point_cloud(
        sensor=sensor, scene=self, sensor_position=sensor_position, **kwargs
    )
)


# ==============================================================================
# 7. LiZARD CDF & Plücker Utilities
# ==============================================================================

def rays_to_plucker(rays_o: np.ndarray, rays_d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts rays (origin, direction) into Plücker coordinates (d, m).
    
    Parameters
    ----------
    rays_o : np.ndarray of shape (N, 3)
        Ray origins.
    rays_d : np.ndarray of shape (N, 3)
        Unit ray directions.
        
    Returns
    -------
    d : np.ndarray of shape (N, 3)
        Direction vectors.
    m : np.ndarray of shape (N, 3)
        Moment vectors: m = o x d.
    """
    rays_o = np.asarray(rays_o)
    rays_d = np.asarray(rays_d)
    m = np.cross(rays_o, rays_d)
    return rays_d, m


def query_cdf_along_ray(
    scene: Scene,
    ray_o: Union[np.ndarray, List[float]],
    ray_d: Union[np.ndarray, List[float]],
    query_distances: Union[np.ndarray, List[float]],
) -> np.ndarray:
    """
    Evaluates ground-truth return Cumulative Distribution Function CDF(t)
    along a ray at arbitrary query distances t.
    
    Parameters
    ----------
    scene : Scene
        Scene containing geometric objects and densities.
    ray_o : array-like of shape (3,)
        Ray origin.
    ray_d : array-like of shape (3,)
        Ray direction.
    query_distances : array-like of shape (M,)
        Distances along ray at which to evaluate CDF.
        
    Returns
    -------
    cdf_values : np.ndarray of shape (M,)
        CDF(t) values in [0, 1].
    """
    ray_o = np.asarray(ray_o, dtype=np.float64).reshape(1, 3)
    ray_d = np.asarray(ray_d, dtype=np.float64).reshape(1, 3)
    norm = np.linalg.norm(ray_d)
    if norm > 1e-9:
        ray_d = ray_d / norm

    t_eval = np.asarray(query_distances, dtype=np.float64)
    max_t = float(np.max(t_eval)) if len(t_eval) else 100.0

    trace_res = trace_scene_rays(ray_o, ray_d, scene, max_range=max_t + 1.0)[0]
    hits = trace_res["hits"]

    cdf_out = np.zeros_like(t_eval, dtype=np.float64)
    if len(hits) == 0:
        return cdf_out

    # Continuous volumetric CDF evaluation along the ray
    for i, t in enumerate(t_eval):
        T_cum = 1.0
        cdf_val = 0.0
        for h in hits:
            t_in = h["t_in"]
            t_out = h["t_out"]
            density = h["density"]
            if t <= t_in:
                break
            elif t >= t_out:
                p_chord = 1.0 - np.exp(-density * (t_out - t_in))
                cdf_val += T_cum * p_chord
                T_cum *= (1.0 - p_chord)
            else:
                # Currently inside this object at distance (t - t_in)
                p_partial = 1.0 - np.exp(-density * (t - t_in))
                cdf_val += T_cum * p_partial
                T_cum *= (1.0 - p_partial)
                break
        cdf_out[i] = cdf_val

    return cdf_out


# ==============================================================================
# Helper Functions
# ==============================================================================

def _parse_rotation(rotation: Optional[Union[np.ndarray, R, List[float]]]) -> np.ndarray:
    """Parses rotation into a 3x3 orthonormal numpy matrix."""
    if rotation is None:
        return np.eye(3, dtype=np.float64)
    if isinstance(rotation, R):
        return rotation.as_matrix()
    arr = np.asarray(rotation, dtype=np.float64)
    if arr.shape == (3, 3):
        return arr
    if arr.size == 3:
        # Assume Euler angles (degrees)
        return R.from_euler("xyz", arr, degrees=True).as_matrix()
    raise ValueError(f"Invalid rotation specification with shape {arr.shape}.")


def _build_orthonormal_basis(axis: np.ndarray) -> np.ndarray:
    """Builds a 3x3 rotation matrix whose columns are (u, v, axis)."""
    axis = axis / np.linalg.norm(axis)
    if abs(axis[0]) < 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    else:
        ref = np.array([0.0, 1.0, 0.0])
    u = np.cross(axis, ref)
    u = u / np.linalg.norm(u)
    v = np.cross(axis, u)
    return np.column_stack([u, v, axis])


def compute_look_at_rotation(
    eye: Union[np.ndarray, List[float]],
    target: Union[np.ndarray, List[float]],
    up: Union[np.ndarray, List[float]] = (0.0, 0.0, 1.0),
) -> np.ndarray:
    """
    Computes a 3x3 rotation matrix for a sensor at 'eye' looking towards 'target'.
    Sensor frame convention: X = forward (pointing to target), Y = left, Z = up.
    
    Parameters
    ----------
    eye : array-like of shape (3,)
        Position of the sensor in world space.
    target : array-like of shape (3,)
        Point in world space the sensor is aimed at.
    up : array-like of shape (3,)
        World up direction (default [0, 0, 1]).
    
    Returns
    -------
    R : np.ndarray of shape (3, 3)
        Rotation matrix transforming local sensor directions into world directions.
    """
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)

    forward = target - eye
    norm_f = np.linalg.norm(forward)
    if norm_f < 1e-9:
        return np.eye(3, dtype=np.float64)
    forward = forward / norm_f

    # Left vector: Y = up x forward (in frame where X is forward, Z is up)
    left = np.cross(up, forward)
    norm_l = np.linalg.norm(left)
    if norm_l < 1e-9:
        left = np.array([0.0, 1.0, 0.0])
    else:
        left = left / norm_l

    # True up vector: Z = forward x left
    true_up = np.cross(forward, left)
    norm_u = np.linalg.norm(true_up)
    if norm_u > 1e-9:
        true_up = true_up / norm_u

    return np.column_stack([forward, left, true_up])


def generate_multiview_dataset(
    sensor: LidarProjectionModel,
    scene: Scene,
    sensor_positions: Union[np.ndarray, List[List[float]]],
    sensor_rotations: Optional[Union[np.ndarray, List[Any]]] = None,
    targets: Optional[Union[np.ndarray, List[List[float]]]] = None,
    method: str = "sample",
    return_placement: str = "volume",
    evaluate_cdf_knots: Optional[Union[np.ndarray, List[float]]] = None,
    random_seed: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Generates a consolidated multi-view training dataset from multiple sensor poses.
    
    Parameters
    ----------
    sensor : LidarProjectionModel
        Sensor intrinsics.
    scene : Scene
        Populated 3D scene.
    sensor_positions : list or array of shape (P, 3)
        Positions of the sensor across P viewpoints.
    sensor_rotations : list or array of shape (P, 3, 3), optional
        Orientations corresponding to each position.
    targets : list or array of shape (P, 3), optional
        Target points to aim each sensor at (used if sensor_rotations is None).
    method : str
        Point return generation method ('sample', 'all', 'strongest').
    return_placement : str
        'volume' or 'surface'.
    evaluate_cdf_knots : array-like of shape (K,), optional
        Distance values t along each ray at which to evaluate ground-truth CDF(t).
    random_seed : int, optional
        Seed for reproducibility.
        
    Returns
    -------
    dataset : dict of numpy arrays:
        - 'rays_o': (total_points, 3) or per-ray origins
        - 'rays_d': (total_points, 3) unit ray directions
        - 'plucker_d': (total_points, 3) Plücker direction
        - 'plucker_m': (total_points, 3) Plücker moment (o x d)
        - 'points': (total_points, 3) 3D return coordinates
        - 'ranges': (total_points,) distances to returns
        - 'probabilities': (total_points,) return probabilities
        - 'pose_indices': (total_points,) index of sensor pose (0 to P-1)
        - 'poses': (P, 4, 4) SE(3) sensor transformation matrices
        - 'cdf_knots': (P * rays, K) ground-truth CDF values (if evaluate_cdf_knots is given)
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    positions = np.asarray(sensor_positions, dtype=np.float64)
    P = len(positions)

    # Compute rotations if not explicitly provided
    if sensor_rotations is not None:
        rotations = [_parse_rotation(r) for r in sensor_rotations]
    elif targets is not None:
        targets_arr = np.asarray(targets, dtype=np.float64)
        if len(targets_arr) == 1 and P > 1:
            targets_arr = np.repeat(targets_arr, P, axis=0)
        rotations = [compute_look_at_rotation(positions[i], targets_arr[i]) for i in range(P)]
    else:
        rotations = [np.eye(3, dtype=np.float64) for _ in range(P)]

    all_points = []
    all_ranges = []
    all_probs = []
    all_normals = []
    all_rays_o = []
    all_rays_d = []
    all_pose_idx = []
    all_poses = []
    all_cdfs = []

    for pose_i in range(P):
        pos = positions[pose_i]
        rot = rotations[pose_i]

        # SE(3) matrix: [R | t]
        T_matrix = np.eye(4, dtype=np.float64)
        T_matrix[:3, :3] = rot
        T_matrix[:3, 3] = pos
        all_poses.append(T_matrix)

        # Generate point cloud for this pose
        pc = generate_point_cloud(
            sensor=sensor,
            scene=scene,
            sensor_position=pos,
            sensor_rotation=rot,
            method=method,
            return_placement=return_placement,
        )

        N_pts = len(pc)
        if N_pts > 0:
            all_points.append(pc.points)
            all_ranges.append(pc.ranges)
            all_probs.append(pc.probabilities)
            all_normals.append(pc.normals)
            all_rays_o.append(pc.rays_o)
            all_rays_d.append(pc.ray_directions)
            all_pose_idx.append(np.full(N_pts, pose_i, dtype=np.int32))

        # Optional: Ground truth continuous CDF evaluation along rays
        if evaluate_cdf_knots is not None:
            knots = np.asarray(evaluate_cdf_knots, dtype=np.float64)
            rays_o_scan, rays_d_scan = sensor.get_rays(pos, rot, flatten=True)
            pose_cdfs = np.zeros((len(rays_o_scan), len(knots)), dtype=np.float64)
            for r_idx in range(len(rays_o_scan)):
                pose_cdfs[r_idx] = query_cdf_along_ray(
                    scene=scene,
                    ray_o=rays_o_scan[r_idx],
                    ray_d=rays_d_scan[r_idx],
                    query_distances=knots,
                )
            all_cdfs.append(pose_cdfs)

    points_cat = np.concatenate(all_points, axis=0) if all_points else np.empty((0, 3))
    ranges_cat = np.concatenate(all_ranges, axis=0) if all_ranges else np.empty(0)
    probs_cat = np.concatenate(all_probs, axis=0) if all_probs else np.empty(0)
    normals_cat = np.concatenate(all_normals, axis=0) if all_normals else np.empty((0, 3))
    rays_o_cat = np.concatenate(all_rays_o, axis=0) if all_rays_o else np.empty((0, 3))
    rays_d_cat = np.concatenate(all_rays_d, axis=0) if all_rays_d else np.empty((0, 3))
    pose_idx_cat = np.concatenate(all_pose_idx, axis=0) if all_pose_idx else np.empty(0, dtype=np.int32)

    # Compute Plücker coordinates: d and m = o x d
    plucker_d, plucker_m = rays_to_plucker(rays_o_cat, rays_d_cat)

    dataset = {
        "points": points_cat,
        "ranges": ranges_cat,
        "probabilities": probs_cat,
        "normals": normals_cat,
        "rays_o": rays_o_cat,
        "rays_d": rays_d_cat,
        "plucker_d": plucker_d,
        "plucker_m": plucker_m,
        "plucker": np.concatenate([plucker_d, plucker_m], axis=-1) if len(plucker_d) else np.empty((0, 6)),
        "pose_indices": pose_idx_cat,
        "sensor_poses": np.array(all_poses),
    }

    if evaluate_cdf_knots is not None and all_cdfs:
        dataset["cdf_knots"] = np.concatenate(all_cdfs, axis=0)
        dataset["knot_distances"] = np.asarray(evaluate_cdf_knots, dtype=np.float64)

    return dataset

