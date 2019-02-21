import casadi as cs
import numpy as np
import transformation_matrix as tm






def numpy_skew_symmetric(v):
    """Returns a skew symmetric matrix from vector. p q r"""
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])


def inertia_matrix(I):
    """Returns the inertia matrix given the inertia vector """
    return np.array([I[0], I[1], I[2]], [I[1], I[3], I[4]], [I[2], I[4], I[5]])


def motion_cross_product(v):
    """Returns the cross product matrix of a spatial vector"""
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
    return -motion_cross_product(v).T


def spatial_inertia_matrix_Ic(ixx, ixy, ixz, iyy, iyz, izz, mass):
    """Returns a spatial inertia matrix expressed at the center of mass """
    Ic = np.zeros([6, 6])
    Ic[:3, :3] = np.array([[ixx, -ixy, -ixz], [-ixy, iyy, -iyz], [-ixz, -iyz, izz]])

    Ic[3, 3] = mass
    Ic[4, 4] = mass
    Ic[5, 5] = mass

    return Ic


def spatial_inertia_matrix_IO(ixx, ixy, ixz, iyy, iyz, izz, mass, c):
    """Returns a spatial inertia matrix expressed at the origin """
    IO = np.zeros([6, 6])
    cx = numpy_skew_symmetric(c)
    inertia_matrix =np.array([[ixx, -ixy, -ixz], [-ixy, iyy, -iyz], [-ixz, -iyz, izz]])
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
    X = cs.SX.zeros(6,6)
    X[:3, :3] = R.T
    X[3:, 3:] = R.T
    X[:3, 3:] = cs.mtimes(cs.skew(r), R.T)
    return X

def spatial_transform(R, r):
    X = cs.SX.zeros(6,6)
    X[:3, :3] = R
    X[3:, 3:] = R
    X[3:, :3] = -cs.mtimes(R, cs.skew(r))
    return X


def spatial_transform_BA(R, r):
    X = cs.SX.zeros(6,6)
    X[:3, :3] = R.T
    X[3:, 3:] = R.T
    X[3:, :3] = cs.mtimes(cs.skew(r), R.T)
    return X


def XJXT2(xyz, rpy, axis, qi):
    RT = numpy_rotation_rpy(rpy[0], rpy[1], rpy[2])
    XT = spatial_transform(RT.T, xyz)

    cqi = cs.cos(qi)
    sqi = cs.sin(qi)
    R = cs.SX.zeros(3, 3)

    s00 = (1 - cqi)*axis[0]*axis[0] + cqi

    s11 = (1 - cqi)*axis[1]*axis[1] + cqi
    s22 = (1 - cqi)*axis[2]*axis[2] + cqi
    s01 = (1 - cqi)*axis[0]*axis[1] - axis[2]*sqi
    s10 = (1 - cqi)*axis[0]*axis[1] + axis[2]*sqi
    s12 = (1 - cqi)*axis[1]*axis[2] - axis[0]*sqi
    s21 = (1 - cqi)*axis[1]*axis[2] + axis[0]*sqi
    s20 = (1 - cqi)*axis[0]*axis[2] - axis[1]*sqi
    s02 = (1 - cqi)*axis[0]*axis[2] + axis[1]*sqi

    R[0, 0] = s00
    R[0, 1] = s01
    R[0, 2] = s02

    R[1, 0] = s10
    R[1, 1] = s11
    R[1, 2] = s12

    R[2, 0] = s20
    R[2, 1] = s21
    R[2, 2] = s22


    XJ = cs.SX.zeros(6,6)
    XJ[:3, :3] = R.T
    XJ[3:, 3:] = R.T


    return cs.mtimes(XJ, XT)


def XJT_revolute(xyz, rpy, axis, qi):
    T = tm.revolute(xyz,rpy,axis,qi)
    rotation_matrix = T[:3,:3]
    displacement = T[:3, 3]
    return spatial_transform(rotation_matrix.T, displacement)


def XJT_revolute_BA(xyz, rpy, axis, qi):
    T = tm.revolute(xyz,rpy,axis,qi)
    rotation_matrix = T[:3,:3]
    displacement = T[:3, 3]
    return spatial_transform_BA(rotation_matrix, displacement)

def XJT_prismatic(xyz, rpy, axis, qi):
    T = tm.prismatic(xyz,rpy,axis,qi)
    rotation_matrix = T[:3,:3]
    displacement = T[:3, 3]
    return spatial_transform(rotation_matrix.T, displacement)


def XJT_prismatic_BA(xyz, rpy, axis, qi):
    T = tm.prismatic(xyz,rpy,axis,qi)
    rotation_matrix = T[:3,:3]
    displacement = T[:3, 3]
    return spatial_transform_BA(rotation_matrix, displacement)


def XT(xyz, rpy):
    """Returns a general spatial transformation matrix matrix"""
    rotation_matrix = numpy_rotation_rpy(rpy[0], rpy[1], rpy[2])
    return spatial_transform(rotation_matrix.T, xyz)



def XJ_prismatic(axis, qi):
        """Returns a symbolic spatial translation transformation matrix for prismatic joint"""
        R = np.identity(3)
        r = axis*qi
        return spatial_transform(R, r)

def XJ_prismatic_BA(axis, qi):
        """Returns a symbolic spatial translation transformation matrix for prismatic joint"""
        R = np.identity(3)
        r = axis*qi
        return spatial_transform_BA(R, r)

def XJ_revolute(axis, qi):
    """Returns a symbolic spatial rotation transformation matrix for a revolute joint"""
    X = cs.SX.zeros(6, 6)
    R = cs.SX.zeros(3, 3)
    s = cs.sin(qi)
    c = cs.cos(qi)


    if axis[0] == 1:
        R[0, 0] = 1
        R[1, 1] = c
        R[2, 2] = c
        R[1, 2] = -s
        R[2, 1] = s

    elif axis[1] == 1:
        R[0, 0] = c
        R[0, 2] = s
        R[1, 1] = 1
        R[2, 0] = -s
        R[2, 2] = c

    elif axis[2] == 1:
        R[0, 0] = c
        R[0, 1] = -s
        R[1, 0] = s
        R[1, 1] = c
        R[2, 2] = 1

    X[:3, :3] = R.T
    X[3:, 3:] = R.T
    return X


def Rx(qi):
    s = cs.sin(qi)
    c = cs.cos(qi)
    Rx = cs.SX.zeros(3, 3)
    Rx[0, 0] = 1
    Rx[1, 1] = c
    Rx[2, 2] = c
    Rx[1, 2] = -s
    Rx[2, 1] = s

    return Rx

def Ry(qi):
    s = cs.sin(qi)
    c = cs.cos(qi)
    Ry = cs.SX.zeros(3, 3)
    Ry[0, 0] = c
    Ry[0, 2] = s
    Ry[1, 1] = 1
    Ry[2, 0] = -s
    Ry[2, 2] = c

    return Ry

def Rz(qi):
    s = cs.sin(qi)
    c = cs.cos(qi)
    Rz = cs.SX.zeros(3, 3)
    Rz[0, 0] = c
    Rz[0, 1] = -s
    Rz[1, 0] = s
    Rz[1, 1] = c
    Rz[2, 2] = 1

    return Rz

def XJ_revolute_posneg(axis, qi):
    """Returns a symbolic spatial rotation transformation matrix for a revolute joint"""
    X = cs.SX.zeros(6, 6)
    R = cs.SX.zeros(3, 3)


    def Rx(scale):
        s = cs.sin(qi*scale)
        c = cs.cos(qi*scale)
        Rx = cs.SX.zeros(3, 3)
        Rx[0, 0] = 1
        Rx[1, 1] = c
        Rx[2, 2] = c
        Rx[1, 2] = -s
        Rx[2, 1] = s

        return Rx

    def Ry(scale):
        s = cs.sin(qi*scale)
        c = cs.cos(qi*scale)
        Ry = cs.SX.zeros(3, 3)
        Ry[0, 0] = c
        Ry[0, 2] = s
        Ry[1, 1] = 1
        Ry[2, 0] = -s
        Ry[2, 2] = c

        return Ry

    def Rz(scale):
        s = cs.sin(qi*scale)
        c = cs.cos(qi*scale)
        Rz = cs.SX.zeros(3, 3)
        Rz[0, 0] = c
        Rz[0, 1] = -s
        Rz[1, 0] = s
        Rz[1, 1] = c
        Rz[2, 2] = 1

        return Rz

    R = cs.mtimes(Rx(axis[0]), cs.mtimes(Ry(axis[1]), Rz(axis[2])))

    X[:3, :3] = R.T
    X[3:, 3:] = R.T
    return X

def Xrot(axis, qi):
    X = cs.SX.zeros(6, 6)
    R = cs.SX.zeros(3, 3)
    s = cs.sin(qi)
    c = cs.sin(qi)

    R[0, 0] = axis[0] * axis[0] * (1. - c) + c
    R[0, 1] = axis[1] * axis[0] * (1. - c) + axis[2] * s
    R[0, 2] = axis[0] * axis[2] * (1. - c) - axis[1] * s

    R[1, 0] = axis[0] * axis[1] * (1. - c) - axis[2] * s
    R[1, 1] = axis[1] * axis[1] * (1. - c) + c
    R[1, 2] = axis[1] * axis[2] * (1. - c) + axis[0] * s

    R[2, 0] = axis[0] * axis[2] * (1. - c) + axis[1] * s
    R[2, 1] = axis[1] * axis[2] * (1. - c) - axis[0] * s
    R[2, 2] = axis[2] * axis[2] * (1. - c) + c

    X[:3, :3] = R
    X[3:, 3:] = R
    return X


#brukes som ledd i XJXT
def Xrot2(axis, qi):
    cqi = cs.cos(qi)
    sqi = cs.sin(qi)
    R = cs.SX.zeros(3, 3)
    s00 = (1 - cqi)*axis[0]*axis[0] + cqi
    s11 = (1 - cqi)*axis[1]*axis[1] + cqi
    s22 = (1 - cqi)*axis[2]*axis[2] + cqi
    s01 = (1 - cqi)*axis[0]*axis[1] - axis[2]*sqi
    s10 = (1 - cqi)*axis[0]*axis[1] + axis[2]*sqi
    s12 = (1 - cqi)*axis[1]*axis[2] - axis[0]*sqi
    s21 = (1 - cqi)*axis[1]*axis[2] + axis[0]*sqi
    s20 = (1 - cqi)*axis[0]*axis[2] - axis[1]*sqi
    s02 = (1 - cqi)*axis[0]*axis[2] + axis[1]*sqi

    R[0, 0] = s00
    R[0, 1] = s01
    R[0, 2] = s02

    R[1, 0] = s10
    R[1, 1] = s11
    R[1, 2] = s12

    R[2, 0] = s20
    R[2, 1] = s21
    R[2, 2] = s22
    return R



#XJXT = XJT_revolute bare at R-delen er delt opp for testing
def XJXT(xyz, rpy, axis, q):
    XJ_R = Xrot2(axis, q)
    XT_R = numpy_rotation_rpy(rpy[0], rpy[1], rpy[2])
    R = cs.mtimes(XJ_R, XT_R)
    return spatial_transform(R, xyz)


def Xrotkdl(axis, qi):


    X = cs.SX.zeros(6, 6)
    R = cs.SX.zeros(3, 3)
    s = cs.sin(-qi)
    c = cs.sin(-qi)

    R[0, 0] = axis[0] * axis[0] * (1. - c) + c
    R[0, 1] = axis[0] * axis[1] * (1. - c) - axis[2] * s
    R[0, 2] = axis[0] * axis[2] * (1. - c) + axis[1] * s


    R[1, 0] = axis[1] * axis[0] * (1. - c) + axis[2] * s
    R[1, 1] = axis[1] * axis[1] * (1. - c) + c
    R[1, 2] = axis[1] * axis[2] * (1. - c) - axis[0] * s


    R[2, 0] = axis[0] * axis[2] * (1. - c) - axis[1] * s
    R[2, 1] = axis[1] * axis[2] * (1. - c) + axis[0] * s
    R[2, 2] = axis[2] * axis[2] * (1. - c) + c

    X[:3, :3] = R
    X[3:, 3:] = R
    return X
