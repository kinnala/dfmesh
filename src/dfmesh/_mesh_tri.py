import warnings

import numpy as np

from ._mesh import Mesh
from ._helpers import unique_rows

__all__ = ["MeshTri"]


class MeshTri(Mesh):
    """Class for handling triangular meshes."""

    def __init__(self, points, cells, sort_cells=False):
        super().__init__(points, cells, sort_cells=sort_cells)

        assert self.n == 3

        # some backwards-compatibility fixes
        self.create_edges = super().create_facets
        self.compute_signed_cell_areas = super().compute_signed_cell_volumes

        self.boundary_edges = super().boundary_facets
        self.is_boundary_edge = super().is_boundary_facet

    @property
    def euler_characteristic(self):
        # number of vertices - number of edges + number of faces
        if self._cells_facets is None:
            self.create_facets()
        return (
            self.points.shape[0]
            - self.edges["points"].shape[0]
            + self.cells("points").shape[0]
        )

    @property
    def genus(self):
        # <https://math.stackexchange.com/a/4196497/36678>:
        #
        # chi = 2 - 2 * g - b
        #
        # g = 1 - (chi + b) / 2
        #
        # where b is the number of boundary components.
        if self._is_boundary_facet is None:
            self.create_facets()

        if np.any(self.is_boundary_edge):
            from scipy.sparse import coo_matrix
            from scipy.sparse.csgraph import connected_components

            boundary_edges = self.edges["points"][self.is_boundary_edge]

            n = np.max(boundary_edges) + 1
            val = np.ones(len(boundary_edges), dtype=int)
            M = coo_matrix((val, boundary_edges.T), shape=(n, n))
            # Unfortunately, connected_components() counts orphaned nodes as an
            # individual component. See <https://github.com/scipy/scipy/issues/14391>.
            _, cc = connected_components(M)
            num_cc = np.sum(np.bincount(cc) > 1)

        else:
            num_cc = 0

        chi = self.euler_characteristic
        if (chi + num_cc) % 2 != 0:
            raise RuntimeError("Non-integer genus. Is the mesh perhaps not orientable?")

        return 1 - (chi + num_cc) // 2

    @property
    def angles(self):
        """All angles in the triangle."""
        # The cosines of the angles are the negative dot products of the normalized
        # edges adjacent to the angle.
        norms = self.edge_lengths
        ei_dot_ej = self.ei_dot_ei - np.sum(self.ei_dot_ei, axis=0) / 2
        normalized_ei_dot_ej = np.array(
            [
                ei_dot_ej[0] / norms[1] / norms[2],
                ei_dot_ej[1] / norms[2] / norms[0],
                ei_dot_ej[2] / norms[0] / norms[1],
            ]
        )
        return np.arccos(-normalized_ei_dot_ej)

    def compute_ncurl(self, vector_field):
        """Computes the n.dot.curl of a vector field over the mesh. While the vector
        field is point-based, the curl will be cell-based. The approximation is based on

        .. math::
            n\\cdot curl(F) = \\lim_{A\\to 0} |A|^{-1} \\rangle\\int_{dGamma}, F\\rangle dr;

        see https://en.wikipedia.org/wiki/Curl_(mathematics). Actually, to approximate
        the integral, one would only need the projection of the vector field onto the
        edges at the midpoint of the edges.
        """
        # Compute the projection of A on the edge at each edge midpoint. Take the
        # average of `vector_field` at the endpoints to get the approximate value at the
        # edge midpoint.
        A = 0.5 * np.sum(vector_field[self.idx[-1]], axis=0)
        # sum of <edge, A> for all three edges
        sum_edge_dot_A = np.einsum("ijk, ijk->j", self.half_edge_coords, A)

        # Get normalized vector orthogonal to triangle
        z = np.cross(self.half_edge_coords[0], self.half_edge_coords[1])

        # Now compute
        #
        #    curl = z / ||z|| * sum_edge_dot_A / |A|.
        #
        # Since ||z|| = 2*|A|, one can save a sqrt and do
        #
        #    curl = z * sum_edge_dot_A * 0.5 / |A|^2.
        #
        curl = z * (0.5 * sum_edge_dot_A / self.cell_volumes**2)[..., None]
        return curl

    def show_vertex(self, *args, **kwargs):
        """Show the mesh around a vertex (see plot_vertex())."""
        import matplotlib.pyplot as plt

        self.plot_vertex(*args, **kwargs)
        plt.show()
        plt.close()

    def plot_vertex(self, point_id, show_ce_ratio=True):
        """Plot the vicinity of a point and its covolume/edgelength ratio.

        :param point_id: Node ID of the point to be shown.
        :type point_id: int

        :param show_ce_ratio: If true, shows the ce_ratio of the point, too.
        :type show_ce_ratio: bool, optional
        """
        # Importing matplotlib takes a while, so don't do that at the header.
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.gca()
        plt.axis("equal")

        if self.edges is None:
            self.create_facets()

        # Find the edges that contain the vertex
        edge_gids = np.where((self.edges["points"] == point_id).any(axis=1))[0]
        # ... and plot them
        for point_ids in self.edges["points"][edge_gids]:
            x = self.points[point_ids]
            ax.plot(x[:, 0], x[:, 1], "k")

        # Highlight ce_ratios.
        if show_ce_ratio:
            # Find the cells that contain the vertex
            cell_ids = np.where((self.cells("points") == point_id).any(axis=1))[0]

            for cell_id in cell_ids:
                for edge_gid in self.cells("edges")[cell_id]:
                    if point_id not in self.edges["points"][edge_gid]:
                        continue
                    point_ids = self.edges["points"][edge_gid]
                    edge_midpoint = 0.5 * (
                        self.points[point_ids[0]] + self.points[point_ids[1]]
                    )
                    p = np.stack(
                        [self.cell_circumcenters[cell_id], edge_midpoint], axis=1
                    )
                    q = np.column_stack(
                        [
                            self.cell_circumcenters[cell_id],
                            edge_midpoint,
                            self.points[point_id],
                        ]
                    )
                    ax.fill(q[0], q[1], color="0.5")
                    ax.plot(p[0], p[1], color="0.7")
        return

    def flip_until_delaunay(self, tol=0.0, max_steps=100):
        """Flip edges until the mesh is fully Delaunay (up to `tol`)."""
        num_flips = 0
        assert tol >= 0.0
        # If all circumcenter-facet distances are positive, all cells are Delaunay.
        if np.all(self.circumcenter_facet_distances > -0.5 * tol):
            return num_flips

        # Now compute the boundary facet. A little more costly, but we'd have to do that
        # anyway. If all _interior_ circumcenter-facet distances are positive, all cells
        # are Delaunay.
        if np.all(
            self.circumcenter_facet_distances[~self.is_boundary_facet_local]
            > -0.5 * tol
        ):
            return num_flips

        step = 0

        is_flip_interior_facet = self.signed_circumcenter_distances < -tol

        while True:
            # Don't flip the edges which would flip into existing edges. This can
            # happen, for example, in triangular shell meshes in 3D, and leads to weird
            # behavior down the line. See
            # <https://github.com/nschloe/meshplex/issues/130>.
            flip_filter = np.ones(np.sum(is_flip_interior_facet), dtype=bool)
            #
            facets_cells_flip = self.facets_cells["interior"][:, is_flip_interior_facet]
            # facet_gids = facets_cells_flip[0]
            adj_cells = facets_cells_flip[1:3]
            lids = facets_cells_flip[3:5]
            expected_new_edges = np.array(
                [
                    self.cells("points")[adj_cells[0], lids[0]],
                    self.cells("points")[adj_cells[1], lids[1]],
                ]
            ).T
            expected_new_edges = np.sort(expected_new_edges, axis=1)
            #
            # This isin() call can be quite costly since we're checking against _all_
            # existing edges.

            already_exists = np.isin(
                np.ascontiguousarray(expected_new_edges).view(np.dtype((np.void, 8))),
                np.ascontiguousarray(self.facets["points"]).view(np.dtype((np.void, 8))),
            ).reshape(expected_new_edges.shape[0])
            flip_filter &= ~already_exists
            #
            # Check if some flips would lead to the same flipped edge.
            _, inv = unique_rows(expected_new_edges, return_inverse=True)
            is_unique = np.zeros(len(expected_new_edges), dtype=bool)
            is_unique[inv] = True
            flip_filter &= is_unique
            # apply the filter
            is_flip_interior_facet[is_flip_interior_facet] &= flip_filter

            step += 1
            if not np.any(is_flip_interior_facet):
                break

            if step > max_steps:
                break

            interior_facets_cells = self.facets_cells["interior"][1:3].T
            adj_cells = interior_facets_cells[is_flip_interior_facet].T

            # Check if there are cells for which more than one facet needs to be
            # flipped. For those, only flip one facet, namely that with the smaller
            # (more negative) circumcenter_facet_distance.
            cell_gids, num_flips_per_cell = np.unique(adj_cells, return_counts=True)
            multiflip_cell_gids = cell_gids[num_flips_per_cell > 1]
            while np.any(num_flips_per_cell > 1):
                for cell_gid in multiflip_cell_gids:
                    facet_gids = self.cells("facets")[cell_gid]
                    is_interior_facet = self.is_interior_facet[facet_gids]
                    idx = self.facets_cells_idx[facet_gids[is_interior_facet]]
                    k = np.argmin(self.signed_circumcenter_distances[idx])
                    is_flip_interior_facet[idx] = False
                    is_flip_interior_facet[idx[k]] = True

                adj_cells = interior_facets_cells[is_flip_interior_facet].T
                cell_gids, num_flips_per_cell = np.unique(adj_cells, return_counts=True)
                multiflip_cell_gids = cell_gids[num_flips_per_cell > 1]

            # actually perform the flips
            self.flip_interior_facets(is_flip_interior_facet)
            num_flips += np.sum(is_flip_interior_facet)

            is_flip_interior_facet_old = is_flip_interior_facet.copy()
            # Check which edges need to be flipped next.
            is_flip_interior_facet = self.signed_circumcenter_distances < -tol

            # Don't flip edges which have just been flipped. (This can happen due to
            # round-off errors.)
            is_flip_interior_facet[is_flip_interior_facet_old] = False

        is_negative = self.signed_circumcenter_distances < -tol
        num_nondelaunay_facets = np.sum(is_negative)
        if num_nondelaunay_facets > 0:
            dists = self.signed_circumcenter_distances[is_negative]
            warnings.warn(
                f"After {step} edge flip steps, "
                + f"there are {num_nondelaunay_facets} remaining non-Delaunay facets. "
                f"The signed circumcenter distances are {dists.tolist()}. "
                + "This can happen due to round-off errors or "
                + "to prevent non-manifold edges in shell meshes."
            )

        return num_flips

    def flip_interior_facets(self, is_flip_interior_facet):
        facets_cells_flip = self.facets_cells["interior"][:, is_flip_interior_facet]
        facet_gids = facets_cells_flip[0]
        adj_cells = facets_cells_flip[1:3]
        lids = facets_cells_flip[3:5]

        #        3                   3
        #        A                   A
        #       /|\                 / \
        #     3/ | \2             3/   \2
        #     /  |  \             /  1  \
        #   0/ 0 |   \1   ==>   0/_______\1
        #    \   | 1 /           \       /
        #     \  |  /             \  0  /
        #     0\ | /1             0\   /1
        #       \|/                 \ /
        #        V                   V
        #        2                   2
        #
        v = np.array(
            [
                self.cells("points")[adj_cells[0], lids[0]],
                self.cells("points")[adj_cells[1], lids[1]],
                self.cells("points")[adj_cells[0], (lids[0] + 1) % 3],
                self.cells("points")[adj_cells[0], (lids[0] + 2) % 3],
            ]
        )

        # This must be computed before the points are reset
        equal_orientation = (
            self.cells("points")[adj_cells[0], (lids[0] + 1) % 3]
            == self.cells("points")[adj_cells[1], (lids[1] + 2) % 3]
        )

        # Set up new cells->points relationships.
        # Make sure that positive/negative area orientation is preserved. This is
        # especially important for signed area computations: In a mesh of all positive
        # areas, you don't want a negative area appear after a facet flip.
        self.cells("points")[adj_cells[0]] = v[[0, 2, 1]].T
        self.cells("points")[adj_cells[1]] = v[[0, 1, 3]].T

        # Set up new facet->points relationships.
        self.facets["points"][facet_gids] = np.sort(v[[0, 1]], axis=0).T

        # Set up new cells->facets relationships.
        previous_facets = self.cells("facets")[adj_cells].copy()  # TODO need copy?
        # Do the neighboring cells have equal orientation (both point sets
        # clockwise/counterclockwise)?
        #
        # facets as in the above ascii art
        i0 = np.ones(equal_orientation.shape[0], dtype=int)
        i0[~equal_orientation] = 2
        i1 = np.ones(equal_orientation.shape[0], dtype=int)
        i1[equal_orientation] = 2
        e = [
            np.choose((lids[0] + 2) % 3, previous_facets[0].T),
            np.choose((lids[1] + i0) % 3, previous_facets[1].T),
            np.choose((lids[1] + i1) % 3, previous_facets[1].T),
            np.choose((lids[0] + 1) % 3, previous_facets[0].T),
        ]
        # The order here is tightly coupled to self.cells("points") above
        self.cells("facets")[adj_cells[0]] = np.column_stack([e[1], facet_gids, e[0]])
        self.cells("facets")[adj_cells[1]] = np.column_stack([e[2], e[3], facet_gids])

        # update is_boundary_facet_local
        for k in range(3):
            self.is_boundary_facet_local[k, adj_cells] = self.is_boundary_facet[
                self.cells("facets")[adj_cells, k]
            ]

        # Update the facet->cells relationship. We need to update facets_cells info for
        # all five facets.
        # First update the flipped facet; it's always interior.
        idx = self.facets_cells_idx[facet_gids]
        # cell ids
        self.facets_cells["interior"][1, idx] = adj_cells[0]
        self.facets_cells["interior"][2, idx] = adj_cells[1]
        # local facet ids; see self.cells("facets")
        self.facets_cells["interior"][3, idx] = 1
        self.facets_cells["interior"][4, idx] = 2
        #
        # Now handle the four surrounding facets
        conf = [
            # The data is:
            # (1) facet id
            # (2) previous adjacent cell (adj_cells[0] or adj_cells[1])
            # (3) new adjacent cell (adj_cells[0] or adj_cells[1])
            # (4) local facet index in the new adjacent cell
            (e[0], 0, 0, 2),
            (e[1], 1, 0, 0),
            (e[2], 1, 1, 0),
            (e[3], 0, 1, 1),
        ]
        for facet, prev_adj_idx, new__adj_idx, new_local_facet_index in conf:
            prev_adj = adj_cells[prev_adj_idx]
            new__adj = adj_cells[new__adj_idx]
            idx = self.facets_cells_idx[facet]
            # boundary...
            is_boundary = self.is_boundary_facet[facet]
            idx_bou = idx[is_boundary]
            prev_adjacent = prev_adj[is_boundary]
            new__adjacent = new__adj[is_boundary]
            # The assertion just makes sure we're doing the right thing. It should never
            # trigger.
            assert np.all(prev_adjacent == self.facets_cells["boundary"][1, idx_bou])
            self.facets_cells["boundary"][1, idx_bou] = new__adjacent
            self.facets_cells["boundary"][2, idx_bou] = new_local_facet_index
            # ...or interior?
            prev_adjacent = prev_adj[~is_boundary]
            new__adjacent = new__adj[~is_boundary]
            idx_int = idx[~is_boundary]
            # Interior facets have two neighboring cells in no particular order. Find
            # out if the adj_cell if the flipped facet comes first or second.
            is_first = prev_adjacent == self.facets_cells["interior"][1, idx_int]
            # The following is just a safety net. We could as well take ~is_first.
            is_secnd = prev_adjacent == self.facets_cells["interior"][2, idx_int]
            assert np.all(np.logical_xor(is_first, is_secnd))
            # actually set the data
            idx_first = idx_int[is_first]
            self.facets_cells["interior"][1, idx_first] = new__adjacent[is_first]
            self.facets_cells["interior"][3, idx_first] = new_local_facet_index
            # likewise for when the cell appears in the second column
            idx_secnd = idx_int[~is_first]
            self.facets_cells["interior"][2, idx_secnd] = new__adjacent[~is_first]
            self.facets_cells["interior"][4, idx_secnd] = new_local_facet_index

        # Schedule the cell ids for data updates
        update_cell_ids = np.unique(adj_cells.T.flat)
        # Same for facet ids
        update_facet_gids = self.cells("facets")[update_cell_ids].flat
        facet_cell_idx = self.facets_cells_idx[update_facet_gids]
        update_interior_facet_ids = np.unique(
            facet_cell_idx[self.is_interior_facet[update_facet_gids]]
        )

        self._update_cell_values(update_cell_ids, update_interior_facet_ids)

    def _update_cell_values(self, cell_ids, interior_facet_ids):
        """Updates all sorts of cell information for the given cell IDs."""
        # update idx
        for j in range(1, self.n - 1):
            m = len(self.idx[j - 1])
            r = np.arange(m)
            k = np.array([np.roll(r, -i) for i in range(1, m)])
            self.idx[j][..., cell_ids] = self.idx[j - 1][..., cell_ids][k]

        # update most of the cell-associated values
        self._compute_cell_values(cell_ids)

        if self._signed_cell_volumes is not None:
            self._signed_cell_volumes[cell_ids] = self.compute_signed_cell_volumes(
                cell_ids
            )

        if self._is_boundary_cell is not None:
            self._is_boundary_cell[cell_ids] = np.any(
                self.is_boundary_facet_local[:, cell_ids], axis=0
            )

        if self._cell_centroids is not None:
            self._cell_centroids[cell_ids] = self.compute_cell_centroids(cell_ids)

        # update the signed circumcenter distances for all interior_facet_ids
        if self._signed_circumcenter_distances is not None:
            self._signed_circumcenter_distances[interior_facet_ids] = 0.0
            facet_gids = self.interior_facets[interior_facet_ids]
            adj_cells = self.facets_cells["interior"][1:3, interior_facet_ids].T

            for i in [0, 1]:
                is_facet = np.array(
                    [
                        self.cells("facets")[adj_cells[:, i]][:, k] == facet_gids
                        for k in range(3)
                    ]
                )
                # assert np.all(np.sum(is_facet, axis=0) == 1)
                for k in range(3):
                    self._signed_circumcenter_distances[
                        interior_facet_ids[is_facet[k]]
                    ] += self._circumcenter_facet_distances[
                        k, adj_cells[is_facet[k], i]
                    ]

        # TODO update those values
        self._control_volumes = None
        self._ce_ratios = None
        self._cv_centroids = None
