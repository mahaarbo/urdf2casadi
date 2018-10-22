import casadi as cs
import numpy as np

def numpy_skew_symmetric(v):
    """Returns a skew symmetric matrix from vector. p q r"""
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

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

def inertia_matrix(I):
    """Returns the inertia matrix given the inertia vector """
    return np.array([I[0], I[1], I[2]], [I[1], I[3], I[4]], [I[2], I[4], I[5]])


def spatial_cross_product(v):
    """Returns the cross product matrix of a spatial vector"""
    cross_matrix = np.zeros([6, 6])
    #crp_1 = np.array([[0 -v[2] v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    #crp_2 = np.array([[0 -v[6] v[5]], [v[6], 0, -v[4]], [-v[5], v[4], 0]])

    cross_matrix[:3, :3] = numpy_skew_symmetric(v[:3])
    cross_matrix[3:, 3:] = numpy_skew_symmetric(v[:3])
    cross_matrix[3:, :3] = numpy_skew_symmetric(v[3:])

    return cross_matrix


def spatial_inertia_matrix(ixx, ixy, ixz, iyy, iyz, izz, mass):
    """Returns a spatial inertia matrix expressed at the center of mass """
    Ic = np.zeros([6, 6])
    Ic[:3, :3] = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])

    Ic[3, 3] = mass
    Ic[4, 4] = mass
    Ic[5, 5] = mass

    return Ic

def XL(xyz, rpy):
    """Returns a Plucker transformation matrix on X_L form"""
    X = np.zeros([6, 6])

    #should i not use numpy this way for calculation?
    rotation_matrix = numpy_rotation_rpy(rpy[0], rpy[1], rpy[2])

    X[:3, :3] = rotation_matrix
    X[3:, 3:] = rotation_matrix

    X[3, 0] = -xyz[2]*rotation_matrix[1, 0] + xyz[1]*rotation_matrix[2, 0]
    X[3, 1] = -xyz[2]*rotation_matrix[1, 1] + xyz[1]*rotation_matrix[2, 1]
    X[3, 2] = -xyz[2]*rotation_matrix[1, 2] + xyz[1]*rotation_matrix[2, 2]

    X[4, 0] = xyz[2]*rotation_matrix[0, 0] - xyz[0]*rotation_matrix[2, 0]
    X[4, 1] = xyz[2]*rotation_matrix[0, 1] - xyz[0]*rotation_matrix[2, 1]
    X[4, 2] = xyz[2]*rotation_matrix[0, 2] - xyz[0]*rotation_matrix[2, 2]

    X[5, 0] = -xyz[1]*rotation_matrix[0, 0] + xyz[0]*rotation_matrix[1, 0]
    X[5, 1] = -xyz[1]*rotation_matrix[0, 1] + xyz[0]*rotation_matrix[1, 1]
    X[5, 2] = -xyz[1]*rotation_matrix[0, 2] + xyz[0]*rotation_matrix[1, 2]
    return X

def XJ_prismatic(xyz, rpy, axis, qi):
    """Returns a symbolic Plucker transformation matrix for prismatic joint"""
    X = cs.SX.zeros(6, 6)

    # Origin rotation from RPY ZYX convention
    cr = cs.cos(rpy[0])
    sr = cs.sin(rpy[0])
    cp = cs.cos(rpy[1])
    sp = cs.sin(rpy[1])
    cy = cs.cos(rpy[2])
    sy = cs.sin(rpy[2])
    r00 = cy*cp
    r01 = cy*sp*sr - sy*cr
    r02 = cy*sp*cr + sy*sr
    r10 = sy*cp
    r11 = sy*sp*sr + cy*cr
    r12 = sy*sp*cr - cy*sr
    r20 = -sp
    r21 = cp*sr
    r22 = cp*cr

    #are these position variables correct in plucker?
    p0 = r00*axis[0]*qi + r01*axis[1]*qi + r02*axis[2]*qi
    p1 = r10*axis[0]*qi + r11*axis[1]*qi + r12*axis[2]*qi
    p2 = r20*axis[0]*qi + r21*axis[1]*qi + r22*axis[2]*qi

    sr00 = -p2*r10 + p1*r20
    sr01 = -p2*r11 + p1*r21
    sr02 = -p2*r12 + p1*r22
    sr10 = p2*r00 - p0*r20
    sr11 = p2*r01 - p0*r21
    sr12 = p2*r02 - p0*r22
    sr20 = -p1*r00 + p0*r10
    sr21 = -p1*r01 + p0*r11
    sr22 = -p1*r02 + p0*r12

    # Plucker transformation matrix
    X[0, 0] = r00
    X[0, 1] = r01
    X[0, 2] = r02
    X[1, 0] = r10
    X[1, 1] = r11
    X[1, 2] = r12
    X[2, 0] = r20
    X[2, 1] = r21
    X[2, 2] = r22

    X[3, 0] = sr00
    X[3, 1] = sr01
    X[3, 2] = sr02
    X[4, 0] = sr10
    X[4, 1] = sr11
    X[4, 2] = sr12
    X[5, 0] = sr20
    X[5, 1] = sr21
    X[5, 2] = sr22

    X[3, 3] = r00
    X[3, 4] = r01
    X[3, 5] = r02
    X[4, 3] = r11
    X[4, 4] = r12
    X[4, 5] = r13
    X[5, 3] = r20
    X[5, 4] = r21
    X[5, 5] = r22
    return X


def XJ_revolute(xyz, rpy, axis, qi):
    X = cs.SX.zeros(6, 6)

    # Origin rotation from RPY ZYX convention
    cr = cs.cos(rpy[0])
    sr = cs.sin(rpy[0])
    cp = cs.cos(rpy[1])
    sp = cs.sin(rpy[1])
    cy = cs.cos(rpy[2])
    sy = cs.sin(rpy[2])
    r00 = cy*cp
    r01 = cy*sp*sr - sy*cr
    r02 = cy*sp*cr + sy*sr
    r10 = sy*cp
    r11 = sy*sp*sr + cy*cr
    r12 = sy*sp*cr - cy*sr
    r20 = -sp
    r21 = cp*sr
    r22 = cp*cr

    # joint rotation from skew sym axis angle
    cqi = cs.cos(qi)
    sqi = cs.sin(qi)
    s00 = (1 - cqi)*axis[0]*axis[0] + cqi
    s11 = (1 - cqi)*axis[1]*axis[1] + cqi
    s22 = (1 - cqi)*axis[2]*axis[2] + cqi
    s01 = (1 - cqi)*axis[0]*axis[1] - axis[2]*sqi
    s10 = (1 - cqi)*axis[0]*axis[1] + axis[2]*sqi
    s12 = (1 - cqi)*axis[1]*axis[2] - axis[0]*sqi
    s21 = (1 - cqi)*axis[1]*axis[2] + axis[0]*sqi
    s20 = (1 - cqi)*axis[0]*axis[2] - axis[1]*sqi
    s02 = (1 - cqi)*axis[0]*axis[2] + axis[1]*sqi

    # Homogeneous transformation matrix
    X[0, 0] = r00*s00 + r01*s10 + r02*s20
    X[1, 0] = r10*s00 + r11*s10 + r12*s20
    X[2, 0] = r20*s00 + r21*s10 + r22*s20

    X[0, 1] = r00*s01 + r01*s11 + r02*s21
    X[1, 1] = r10*s01 + r11*s11 + r12*s21
    X[2, 1] = r20*s01 + r21*s11 + r22*s21

    X[0, 2] = r00*s02 + r01*s12 + r02*s22
    X[1, 2] = r10*s02 + r11*s12 + r12*s22
    X[2, 2] = r20*s02 + r21*s12 + r22*s22

    X[3, 3] = r00*s00 + r01*s10 + r02*s20
    X[4, 3] = r10*s00 + r11*s10 + r12*s20
    X[5, 3] = r20*s00 + r21*s10 + r22*s20

    X[3, 4] = r00*s01 + r01*s11 + r02*s21
    X[4, 4] = r10*s01 + r11*s11 + r12*s21
    X[5, 4] = r20*s01 + r21*s11 + r22*s21

    X[3, 5] = r00*s02 + r01*s12 + r02*s22
    X[4, 5] = r10*s02 + r11*s12 + r12*s22
    X[5, 5] = r20*s02 + r21*s12 + r22*s22

    X[3, 0] = -xyz[2]*r10 + xyz[1]*r20
    X[3, 1] = -xyz[2]*r11 + xyz[1]*r21
    X[3, 2] = -xyz[2]*r12 + xyz[1]*r22

    X[4, 0] = xyz[2]*r00 - xyz[0]*r20
    X[4, 1] = xyz[2]*r01 - xyz[0]*r21
    X[4, 2] = xyz[2]*r02 - xyz[0]*r22

    X[5, 0] = -xyz[1]*r00 + xyz[0]*r10
    X[5, 1] = -xyz[1]*r01 + xyz[0]*r11
    X[5, 2] = -xyz[1]*r02 + xyz[0]*r12

    return X
