import casadi as cs
import urdf2casadi.geometry.transformation_matrix as tm
import numpy as np


def numpy_skew_symmetric(v):
    """Returns a skew symmetric matrix from vector."""
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])


def inertia_matrix(I):
    """Returns the 3x3 rotational inertia matrix given the inertia vector."""
    return np.array([I[0], I[1], I[2]],
                    [I[1], I[3], I[4]],
                    [I[2], I[4], I[5]])


def motion_cross_product(v):
    """Returns the motion cross product matrix of a spatial vector."""

    mcp = cs.SX.zeros(6, 6)

    mcp[0, 1] = -v[2]
    mcp[0, 2] = v[1]
    mcp[1, 0] = v[2]
    mcp[1, 2] = -v[0]
    mcp[2, 0] = -v[1]
    mcp[2, 1] = v[0]

    mcp[3, 4] = -v[2]
    mcp[3, 5] = v[1]
    mcp[4, 3] = v[2]
    mcp[4, 5] = -v[0]
    mcp[5, 3] = -v[1]
    mcp[5, 4] = v[0]

    mcp[3, 1] = -v[5]
    mcp[3, 2] = v[4]
    mcp[4, 0] = v[5]
    mcp[4, 2] = -v[3]
    mcp[5, 0] = -v[4]
    mcp[5, 1] = v[3]

    return mcp


def force_cross_product(v):
    """Returns the force cross product matrix of a spatial vector."""
    return -motion_cross_product(v).T


def spatial_inertia_matrix_Ic(ixx, ixy, ixz, iyy, iyz, izz, mass):
    """Returns the 6x6 spatial inertia matrix expressed at the center of
    mass."""
    Ic = np.zeros([6, 6])
    Ic[:3, :3] = np.array([[ixx, ixy, ixz],
                           [ixy, iyy, iyz],
                           [ixz, iyz, izz]])

    Ic[3, 3] = mass
    Ic[4, 4] = mass
    Ic[5, 5] = mass

    return Ic


def spatial_inertia_matrix_IO(ixx, ixy, ixz, iyy, iyz, izz, mass, c):
    """Returns the 6x6 spatial inertia matrix expressed at the origin."""
    IO = np.zeros([6, 6])
    cx = numpy_skew_symmetric(c)
    inertia_matrix = np.array([[ixx, ixy, ixz],
                               [ixy, iyy, iyz],
                               [ixz, iyz, izz]])

    IO[:3, :3] = inertia_matrix + mass*(np.dot(cx, np.transpose(cx)))
    IO[:3, 3:] = mass*cx
    IO[3:, :3] = mass*np.transpose(cx)

    IO[3, 3] = mass
    IO[4, 4] = mass
    IO[5, 5] = mass

    return IO


def numpy_rotation_rpy(roll, pitch, yaw):
    """Returns a rotation matrix from roll pitch yaw. ZYX convention."""
    cr = np.cos(roll)
    sr = np.sin(roll)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cy = np.cos(yaw)
    sy = np.sin(yaw)
    return np.array([[cy*cp,  cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr],
                     [sy*cp,  sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr],
                     [  -sp,             cp*sr,             cp*cr]])


def spatial_force_transform(R, r):
    """Returns the spatial force transform from a 3x3 rotation matrix
    and a 3x1 displacement vector."""
    X = cs.SX.zeros(6, 6)
    X[:3, :3] = R.T
    X[3:, 3:] = R.T
    X[:3, 3:] = cs.mtimes(cs.skew(r), R.T)
    return X


def spatial_transform(R, r):
    """Returns the spatial motion transform from a 3x3 rotation matrix
    and a 3x1 displacement vector."""
    X = cs.SX.zeros(6, 6)
    X[:3, :3] = R
    X[3:, 3:] = R
    X[3:, :3] = -cs.mtimes(R, cs.skew(r))
    return X


def spatial_transform_BA(R, r):
    """Returns the inverse spatial motion transform from a 3x3 rotation
    matrix and a 3x1 displacement vector."""
    X = cs.SX.zeros(6, 6)
    X[:3, :3] = R.T
    X[3:, 3:] = R.T
    X[3:, :3] = cs.mtimes(cs.skew(r), R.T)
    return X


def XJT_revolute(xyz, rpy, axis, qi):
    """Returns the spatial transform from child link to parent link with
    a revolute connecting joint."""
    T = tm.revolute(xyz, rpy, axis, qi)
    rotation_matrix = T[:3, :3]
    displacement = T[:3, 3]
    return spatial_transform(rotation_matrix.T, displacement)


def XJT_revolute_BA(xyz, rpy, axis, qi):
    """Returns the spatial transform from parent link to child link with
    a revolute connecting joint."""
    T = tm.revolute(xyz, rpy, axis, qi)
    rotation_matrix = T[:3, :3]
    displacement = T[:3, 3]
    return spatial_transform_BA(rotation_matrix, displacement)


def XJT_prismatic(xyz, rpy, axis, qi):
    """Returns the spatial transform from child link to parent link with
    a prismatic connecting joint."""
    T = tm.prismatic(xyz, rpy, axis, qi)
    rotation_matrix = T[:3, :3]
    displacement = T[:3, 3]
    return spatial_transform(rotation_matrix.T, displacement)


def XJT_prismatic_BA(xyz, rpy, axis, qi):
    """Returns the spatial transform from parent link to child link with
    a prismatic connecting joint."""
    T = tm.prismatic(xyz, rpy, axis, qi)
    rotation_matrix = T[:3, :3]
    displacement = T[:3, 3]
    return spatial_transform_BA(rotation_matrix, displacement)


def XT(xyz, rpy):
    """Returns a general spatial transformation matrix matrix"""
    rotation_matrix = numpy_rotation_rpy(rpy[0], rpy[1], rpy[2])
    return spatial_transform(rotation_matrix.T, xyz)
