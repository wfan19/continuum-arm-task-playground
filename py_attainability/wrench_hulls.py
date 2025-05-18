from itertools import product

import numpy as np
from scipy.optimize import minimize
import alphashape
from shapely.geometry import Point, Polygon

from py_attainability.mechanics import ArmDesign, solve_equilibrium_shape, calc_reaction_wrench
from py_attainability.task_attainability import Task
from py_attainability.rod import calc_poses
from py_attainability.lie import Pose2

def check_reactions_attainable_fast(reaction_requirement, segment_twists, struct_design, p_bounds):
    N_nodes = segment_twists.shape[1]
    f_required = reaction_requirement[0, :]
    m_required = reaction_requirement[2, :]

    # Generate pressures along edges of pressure space
    N_ps_per_edge = 2
    ps_bndry = sample_edges_of_cuboid(N_ps_per_edge, p_bounds)
    N_ps = ps_bndry.shape[1]

    # Compute boundary reactions
    reactions_bndry = np.zeros((3, N_nodes, N_ps))
    for i in range(N_ps):
        ps_i = ps_bndry[:, i]
        reactions_bndry[:, :, i] = calc_reaction_wrench(segment_twists, ps_i, struct_design)

    traces_f_bndry, traces_m_bndry = mat_wrenches_to_traces(reactions_bndry)

    # Get convex hull of boundary points for first node
    try:
        i_contour = compute_boundary_indices(traces_f_bndry[0, :], traces_m_bndry[0, :], alpha=0.1)
    except:
        return False

    rltv_f_bndry = traces_f_bndry - traces_f_bndry[0, :]
    rltv_m_bndry = traces_m_bndry - traces_m_bndry[0, :]

    # Test 1: reaction at each node inside boundary polygon
    v_rxn_in_bndry = np.zeros(N_nodes, dtype=bool)
    for i in range(N_nodes):
        v_rxn_in_bndry[i] = inpolygon(
            np.array([f_required[i]]),
            np.array([m_required[i]]),
            traces_f_bndry[i, i_contour],
            traces_m_bndry[i, i_contour]
        )[0]
    if not np.all(v_rxn_in_bndry):
        return False

    # Test 2: relative reactions within convex hull
    rltv_reaction_requirement = reaction_requirement - reaction_requirement[:, [0]]
    rltv_f = rltv_reaction_requirement[0, :]
    rltv_m = rltv_reaction_requirement[2, :]

    try:
        i_contour_all = compute_boundary_indices(rltv_f_bndry[0, :], rltv_m_bndry[0, :], alpha=0.1)
    except:
        return False

    contour_f_bndry = rltv_f_bndry.ravel()[i_contour_all]
    contour_m_bndry = rltv_m_bndry.ravel()[i_contour_all]

    v_rltv_rxn_in_bndry = inpolygon(rltv_f, rltv_m, contour_f_bndry, contour_m_bndry)
    if not np.all(v_rltv_rxn_in_bndry):
        return False

    # Final check per node
    v_rltv_rxn_in_bndry = np.zeros(N_nodes, dtype=bool)
    v_rltv_rxn_in_bndry[0] = True
    for j in range(1, N_nodes):
        try:
            node_points = np.vstack((rltv_f_bndry[j, :], rltv_m_bndry[j, :])).T
            hull_j = ConvexHull(node_points)
            i_contour_j = hull_j.vertices
        except:
            return False

        v_rltv_rxn_in_bndry[j] = inpolygon(
            np.array([rltv_f[j]]),
            np.array([rltv_m[j]]),
            rltv_f_bndry[j, i_contour_j],
            rltv_m_bndry[j, i_contour_j]
        )[0]

    return bool(np.all(v_rxn_in_bndry) and np.all(v_rltv_rxn_in_bndry))

def calc_attainable_wrench_hull(N_segments, arm_design: ArmDesign, task: Task):
    p_maxs = [actuator.model.p_bounds[1] for actuator in arm_design.actuators]
    boundary_ps = sample_edges_of_cuboid(2, p_maxs)

    # Compute boundary reactions
    reactions_bndry = np.zeros((boundary_ps.shape[1], 3, N_segments))
    for i, p_i in enumerate(boundary_ps.T):
        reactions_bndry[i, :, :] = calc_reaction_wrench(task.backbone_shape, p_i, arm_design)

    traces_f_bndry, traces_m_bndry = mat_wrenches_to_traces(reactions_bndry)

    # Get convex hull of boundary points for first node
    i_contours = []
    bndry_fms = []
    for f_bndry_i, m_bndry_i in zip(traces_f_bndry, traces_m_bndry):
        i_contour_i = compute_boundary_indices(f_bndry_i, m_bndry_i, 0.0)
        i_contours.append(i_contour_i)
        bndry_fms.append(np.array([f_bndry_i[i_contour_i], m_bndry_i[i_contour_i]]))

    traces_f_base_bndry = traces_f_bndry[:, i_contours[0]]
    traces_m_base_bndry = traces_m_bndry[:, i_contours[0]]

    rltv_f_bndry = traces_f_base_bndry - traces_f_base_bndry[0, :]
    rltv_m_bndry = traces_m_base_bndry - traces_m_base_bndry[0, :]

    # abs_wrench_hull = [traces_f_bndry[:, i_contour], traces_m_bndry[:, i_contour]]
    abs_wrench_hull = bndry_fms
    rltv_wrench_hull = [rltv_f_bndry, rltv_m_bndry]
    return abs_wrench_hull, rltv_wrench_hull


def mat_wrenches_to_traces(mat_wrenches):
    """
    Extracts force and moment traces from a 3D matrix of wrenches.

    Parameters:
        mat_wrenches (np.ndarray): A 3D NumPy array of shape (3, N_poses, N_ps)

    Returns:
        paths_f (np.ndarray): Force trace, shape (N_poses, N_ps)
        paths_m (np.ndarray): Moment trace, shape (N_poses, N_ps)
    """
    # Validate input dimensions
    if mat_wrenches.shape[0] < 3:
        raise ValueError("mat_wrenches must have at least 3 rows along axis 0 (force, ?, moment).")

    paths_f = mat_wrenches[:, 0, :].T  # Force (index 1 in MATLAB → index 0 in Python)
    paths_m = mat_wrenches[:, 2, :].T  # Moment (index 3 in MATLAB → index 2 in Python)

    return paths_f, paths_m


def sample_edges_of_cuboid(N_points_per_edge, scales):
    """
    Samples points along the edges of an N-dimensional cuboid.

    Parameters:
        N_points_per_edge (int): Number of points to sample along each edge.
        scales (array-like): Lengths of the cuboid sides in each dimension.

    Returns:
        np.ndarray: Array of shape (N_dims, N_points_total)
    """
    scales = np.asarray(scales)
    N_dims = len(scales)
    s_along_edge = np.linspace(0, 1, N_points_per_edge)

    # Generate 2^N_dims vertices as column vectors (shape: N_dims x 2^N_dims)
    vertices = np.array(list(product([0, 1], repeat=N_dims))).T

    points = []

    for i_vert in range(vertices.shape[1]):
        vert_i = vertices[:, i_vert]
        for j_dim in range(N_dims):
            vert_i_repeat = np.tile(vert_i[:, np.newaxis], (1, N_points_per_edge))
            vert_i_repeat[j_dim, :] = s_along_edge
            points.append(vert_i_repeat)

    # Stack all points together and remove duplicates
    all_points = np.hstack(points).T
    unique_points = np.unique(all_points, axis=0).T

    # Scale the points by the cuboid dimensions
    points_out = np.diag(scales) @ unique_points

    return points_out

def compute_boundary_indices(x, y, alpha=0.0):
    """Returns indices of the boundary points using alpha shape."""
    points = np.column_stack((x, y))
    alpha_shape = alphashape.alphashape(points, alpha)

    indices_out = []
    if isinstance(alpha_shape, Polygon):
        # Return indices of input points that are on the boundary
        boundary_coords = np.array(alpha_shape.exterior.coords)
        for i, point in enumerate(points):
            bc_is_point = [np.allclose(point, bc) for bc in boundary_coords]
            if any(bc_is_point):
                indices_out.append(i)
    else:
        print("Failed to find polygon")
    return indices_out

def inpolygon(x, y, xv, yv):
    """Checks whether points (x, y) are inside polygon defined by (xv, yv)."""
    polygon = Polygon(np.column_stack((xv, yv)))
    return np.array([polygon.contains(Point(px, py)) for px, py in zip(x, y)])