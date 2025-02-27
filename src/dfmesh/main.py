from __future__ import annotations

import math
from typing import Callable

from ._mesh_tri import MeshTri
import numpy as np
import scipy.spatial

from .geometry import Geometry
from ._helpers import show as show_mesh


def _create_cells(pts, geo: Geometry):
    # compute Delaunay triangulation
    tri = scipy.spatial.Delaunay(pts)
    cells = tri.simplices.copy()

    # kick out all cells whose barycenter is not in the geometry
    bc = np.sum(pts[cells], axis=1) / 3.0
    cells = cells[geo.dist(bc.T) < 0.0]

    return cells


def _recell_and_boundary_step(mesh, geo, flip_tol):
    # We could do a _create_cells() here, but inverted boundary cell removal plus Lawson
    # flips produce the same result and are much cheaper. This is because, most of the
    # time, there are no cells to be removed and no edges to be flipped. (The flip is
    # still a fairly expensive operation.)
    while True:
        idx = mesh.is_boundary_point
        points_new = mesh.points.copy()
        points_new[idx] = geo.boundary_step(points_new[idx].T).T
        mesh.points = points_new
        #
        num_removed_cells = mesh.remove_boundary_cells(
            lambda is_bdry_cell: mesh.compute_signed_cell_volumes(is_bdry_cell)
            < 1.0e-10
        )
        #
        # The flip has to come right after the boundary cell removal to prevent
        # "degenerate cell" errors.
        mesh.flip_until_delaunay(tol=flip_tol)
        #
        if num_removed_cells == 0:
            break

    # Last kick out all boundary cells whose barycenters are not in the geometry.
    mesh.remove_boundary_cells(
        lambda is_bdry_cell: geo.dist(mesh.compute_cell_centroids(is_bdry_cell).T) > 0.0
    )

    return np.min(mesh.q_radius_ratio)


def create_staggered_grid(h, bounding_box):
    x_step = h
    y_step = h * np.sqrt(3) / 2
    bb_width = bounding_box[1] - bounding_box[0]
    bb_height = bounding_box[3] - bounding_box[2]
    midpoint = [
        (bounding_box[0] + bounding_box[1]) / 2,
        (bounding_box[2] + bounding_box[3]) / 2,
    ]

    num_x_steps = int(bb_width / x_step)
    if num_x_steps % 2 == 1:
        num_x_steps -= 1
    num_y_steps = int(bb_height / y_step)
    if num_y_steps % 2 == 1:
        num_y_steps -= 1

    # Generate initial (staggered) point list from bounding box.
    # Make sure that the midpoint is one point in the grid.
    x2 = num_x_steps // 2
    y2 = num_y_steps // 2
    x, y = np.meshgrid(
        midpoint[0] + x_step * np.arange(-x2, x2 + 1),
        midpoint[1] + y_step * np.arange(-y2, y2 + 1),
    )
    # Staggered, such that the midpoint is not moved.
    # Unconditionally move to the right, then add more points to the left.
    offset = (y2 + 1) % 2
    x[offset::2] += h / 2

    out = np.column_stack([x.reshape(-1), y.reshape(-1)])

    # add points in the staggered lines to preserve symmetry
    n = 2 * (-(-y2 // 2))
    extra = np.empty((n, 2))
    extra[:, 0] = midpoint[0] - x_step * x2 - h / 2
    extra[:, 1] = midpoint[1] + y_step * np.arange(-y2 + offset, y2 + 1, 2)

    out = np.concatenate([out, extra])
    return out


def generate(
    geo: Geometry,
    target_edge_size: float | Callable,
    tol: float = 1.0e-5,
    random_seed: int = 0,
    show: bool = False,
    max_steps: int = 1000,
    verbose: bool = False,
    flip_tol: float = 0.0,
):
    target_edge_size_function = (
        target_edge_size
        if callable(target_edge_size)
        else lambda pts: np.full(pts.shape[1], target_edge_size)
    )

    # Find h0 from edge_size (function)
    if callable(target_edge_size):
        # Find h0 by sampling
        h00 = (geo.bounding_box[1] - geo.bounding_box[0]) / 100
        pts = create_staggered_grid(h00, geo.bounding_box)
        sizes = target_edge_size_function(pts.T)
        assert np.all(
            sizes > 0.0
        ), "target_edge_size_function must be strictly positive."
        h0 = np.min(sizes)
    else:
        h0 = target_edge_size

    pts = create_staggered_grid(h0, geo.bounding_box)

    eps = 1.0e-10

    # remove points outside of the region
    pts = pts[geo.dist(pts.T) < eps]

    # evaluate the element size function, remove points according to it
    alpha = 1.0 / target_edge_size_function(pts.T) ** 2
    rng = np.random.default_rng(random_seed)
    pts = pts[rng.random(pts.shape[0]) < alpha / np.max(alpha)]

    num_feature_points = len(geo.feature_points)
    if num_feature_points > 0:
        # remove all points which are equal to a feature point
        diff = np.array([[pt - fp for fp in geo.feature_points] for pt in pts])
        dist = np.einsum("...k,...k->...", diff, diff)
        ftol = h0 / 10
        equals_feature_point = np.any(dist < ftol**2, axis=1)
        pts = pts[~equals_feature_point]
        # Add feature points
        pts = np.concatenate([geo.feature_points, pts])

    cells = _create_cells(pts, geo)
    mesh = MeshTri(pts, cells)
    # When creating a mesh for the staggered grid, degenerate cells can very well occur
    # at the boundary, where points sit in a straight line. Remove those cells.
    mesh.remove_cells(mesh.q_radius_ratio < 1.0e-10)

    # # move boundary points to the boundary exactly
    # is_boundary_point = mesh.is_boundary_point.copy()
    # mesh.points[is_boundary_point] = geo.boundary_step(
    #     mesh.points[is_boundary_point].T
    # ).T

    dim = 2
    mesh = distmesh_smoothing(
        mesh,
        geo,
        num_feature_points,
        target_edge_size_function,
        max_steps,
        tol,
        verbose,
        show,
        delta_t=0.2,
        f_scale=1 + 0.4 / 2 ** (dim - 1),  # from the original article
        flip_tol=flip_tol,
    )
    points = mesh.points
    cells = mesh.cells("points")

    return points, cells


def triangulate(*args, **kwargs):
    """Triangulate a geometry.

    Parameters
    ----------
    geo: Geometry
        The geometry to triangulate.
    target_edge_size: float | Callable
        The size of triangle edges.
    tol: float = 1.0e-5
        Smoothing terminates if nodes move < tol ** 2.
    random_seed: int = 0
    show: bool = False
    max_steps: int = 1000
        Smoothing terminates if exceeded.
    verbose: bool = False
    flip_tol: float = 0.0

    Returns
    -------
    points
    elements

    """
    points, cells = generate(*args, **kwargs)
    return points.T, cells.T


def distmesh_smoothing(
    mesh,
    geo,
    num_feature_points,
    target_edge_size_function,
    max_steps,
    tol,
    verbose,
    show,
    delta_t,
    f_scale,
    flip_tol=0.0,
):
    mesh.create_edges()

    k = 0
    move2 = [0.0]
    prev_max_move = 0.
    while True:
        if verbose:
            print(f"step {k}")

        if k > max_steps:
            print(f"dfmesh: Exceeded max_steps ({max_steps}).")
            break

        k += 1

        if show:
            print(f"max move: {math.sqrt(max(move2)):.3e}")
            show_mesh(mesh.points, mesh.cells("points"), geo)

        edges = mesh.edges["points"]

        edges_vec_normalized = mesh.points[edges[:, 1]] - mesh.points[edges[:, 0]]
        edge_lengths = np.sqrt(
            np.einsum("ij,ij->i", edges_vec_normalized, edges_vec_normalized)
        )
        edges_vec_normalized /= edge_lengths[..., None]

        # Evaluate element sizes at edge midpoints
        edge_midpoints = (mesh.points[edges[:, 1]] + mesh.points[edges[:, 0]]) / 2
        p = target_edge_size_function(edge_midpoints.T)
        target_lengths = (
            f_scale * p * np.sqrt(np.dot(edge_lengths, edge_lengths) / np.dot(p, p))
        )

        force_abs = target_lengths - edge_lengths
        # only consider repulsive forces
        force_abs[force_abs < 0.0] = 0.0

        # force vectors
        force = edges_vec_normalized * force_abs[..., None]

        n = mesh.points.shape[0]
        force_per_point = np.zeros((2, n))
        np.add.at(force_per_point[0], edges[:, 0], -force[:, 0])
        np.add.at(force_per_point[1], edges[:, 0], -force[:, 1])
        np.add.at(force_per_point[0], edges[:, 1], force[:, 0])
        np.add.at(force_per_point[1], edges[:, 1], force[:, 1])
        force_per_point = force_per_point.T

        update = delta_t * force_per_point

        points_old = mesh.points.copy()

        # update coordinates
        points_new = mesh.points + update
        # leave feature points untouched
        points_new[:num_feature_points] = mesh.points[:num_feature_points]
        mesh.points = points_new
        # Some mesh boundary points may have been moved off of the domain boundary,
        # either because they were pushed outside or because they just became boundary
        # points by way of cell removal. Move them all (back) onto the domain boundary.
        # is_outside = geo.dist(points_new.T) > 0.0
        # idx = is_outside
        # Alternative: Push all boundary points (the ones _inside_ the geometry as well)
        # back to the boundary.
        # idx = is_outside | is_boundary_point
        minq = _recell_and_boundary_step(mesh, geo, flip_tol)

        diff = points_new - points_old
        move2 = np.einsum("ij,ij->i", diff, diff)
        max_move = np.sqrt(np.max(move2))
        if np.abs(prev_max_move - max_move) < 1e-8 and minq > 0.5:
            delta_t /= 2
            prev_max_move = 0
        prev_max_move = max_move
        if verbose:
            print(f"max_move: {max_move:.6e}, dt: " + str(delta_t))
        if max_move < tol**2:
            break

    # The cell removal steps in _recell_and_boundary_step() might create points which
    # aren't part of any cell (dangling points). Remove them now.
    mesh.remove_dangling_points()
    return mesh
