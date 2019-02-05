import casadi as cs
import numpy as np
import transformation_matrix as tm

def XJT_revolute(xyz, rpy, axis, qi):
    X = cs.SX.zeros(6,6)
    T = tm.revolute(xyz,rpy,axis,qi)
    rotation_matrix = T[:3,:3]
    displacement = T[:3, 3]
    X[:3, :3] = rotation_matrix
    X[:3, 3:] = rotation_matrix
    X[3:, :3] = -cs.mtimes(rotation_matrix, cs.skew(displacement))

    return X

def numpy_skew_symmetric(v):
    """Returns a skew symmetric matrix from vector. p q r"""
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])


def inertia_matrix(I):
    """Returns the inertia matrix given the inertia vector """
    return np.array([I[0], I[1], I[2]], [I[1], I[3], I[4]], [I[2], I[4], I[5]])

#must be verified - check
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

#must be verified - check
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

#must be verified - check
def spatial_inertia_matrix_IO(ixx, ixy, ixz, iyy, iyz, izz, mass, c):
    """Returns a spatial inertia matrix expressed at the origin """
    IO = np.zeros([6, 6])
    cx = numpy_skew_symmetric(c)
    #print "cx: \n", cx, "\n"
    inertia_matrix =np.array([[ixx, -ixy, -ixz], [-ixy, iyy, -iyz], [-ixz, -iyz, izz]])
    #print "3x3 inertia tensor: \n", inertia_matrix, "\n"

    IO[:3, :3] = inertia_matrix + mass*(np.dot(cx, np.transpose(cx)))
    #print "I + (m cx cx^T):\n", IO[:3, :3], "\n"

    IO[:3, 3:] = mass*cx
    #print "mcx: \n", IO[:3, 3:], "\n"

    IO[3:, :3] = mass*np.transpose(cx)
    #print "mcx^T: \n", IO[3:, :3], "\n"

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


def XT(xyz, rpy):
    """Returns a general spatial transformation matrix matrix"""
    X = np.zeros([6, 6])

    rotation_matrix = numpy_rotation_rpy(rpy[0], rpy[1], rpy[2])

    X[:3, :3] = rotation_matrix
    X[3:, 3:] = rotation_matrix
    X[3:, :3] = -cs.mtimes(rotation_matrix, numpy_skew_symmetric(xyz))

    return X


#must be verified (is, but look over)
def XJ_prismatic(axis, qi):
        """Returns a symbolic spatial translation transformation matrix for prismatic joint"""
        X = cs.SX.zeros(6, 6)

        X[0, 0] = 1
        X[1, 1] = 1
        X[2, 2] = 1

        X[3, 1] = axis[2]*qi
        X[3, 2] = -axis[1]*qi
        X[4, 0] = -axis[2]*qi
        X[4, 2] = axis[0]*qi
        X[5, 0] = axis[1]*qi
        X[5, 1] = -axis[0]*qi

        X[3, 3] = 1
        X[4, 4] = 1
        X[5, 5] = 1

        return X

def XJ_prismatic_T(axis, qi):
        """Returns a symbolic spatial translation transformation matrix for prismatic joint"""
        return XJ_prismatic(axis, qi).T


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
        R[1, 2] = s
        R[2, 1] = -s

    elif axis[1] == 1:
        R[0, 0] = c
        R[0, 2] = -s
        R[1, 1] = 1
        R[2, 0] = s
        R[2, 2] = c

    else:
        R[0, 0] = c
        R[0, 1] = s
        R[1, 0] = -s
        R[1, 1] = c
        R[2, 2] = 1

    X[:3, :3] = R.T
    X[3:, 3:] = R.T
    return X

def XJ_revolute_new(axis, qi):
    """Returns a symbolic spatial rotation transformation matrix for a revolute joint"""
    X = cs.SX.zeros(6, 6)
    #R = cs.SX.zeros(3, 3)


    def Rx(scale):
        s = cs.sin(-qi*scale)
        c = cs.cos(-qi*scale)
        Rx = cs.SX.zeros(3, 3)
        Rx[0, 0] = 1
        Rx[1, 1] = c
        Rx[2, 2] = c
        Rx[1, 2] = s
        Rx[2, 1] = -s

        return Rx

    def Ry(scale):
        s = cs.sin(-qi*scale)
        c = cs.cos(-qi*scale)
        Ry = cs.SX.zeros(3, 3)
        Ry[0, 0] = c
        Ry[0, 2] = -s
        Ry[1, 1] = 1
        Ry[2, 0] = s
        Ry[2, 2] = c

        return Ry

    def Rz(scale):
        s = cs.sin(-qi*scale)
        c = cs.cos(-qi*scale)
        Rz = cs.SX.zeros(3, 3)
        Rz[0, 0] = c
        Rz[0, 1] = s
        Rz[1, 0] = -s
        Rz[1, 1] = c
        Rz[2, 2] = 1

        return Rz

    #print Rx(axis[0])
    #print Ry(axis[1])
    #print Rz(axis[2])

    R = cs.mtimes(Rx(axis[0]), cs.mtimes(Ry(axis[1]), Rz(axis[2])))

    X[:3, :3] = R
    X[3:, 3:] = R
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



def XJ_revolute_new2(axis, qi):
    """Returns a symbolic spatial rotation transformation matrix for a revolute joint"""
    X = cs.SX.zeros(6, 6)
    #R = cs.SX.zeros(3, 3)


    def Rx():
        s = cs.sin(-qi)
        c = cs.cos(-qi)
        Rx = cs.SX.zeros(3, 3)
        Rx[0, 0] = 1
        Rx[1, 1] = c
        Rx[2, 2] = c
        Rx[1, 2] = s
        Rx[2, 1] = -s

        return Rx

    def Ry():
        s = cs.sin(-qi)
        c = cs.cos(-qi)
        Ry = cs.SX.zeros(3, 3)
        Ry[0, 0] = c
        Ry[0, 2] = -s
        Ry[1, 1] = 1
        Ry[2, 0] = s
        Ry[2, 2] = c

        return Ry

    def Rz():
        s = cs.sin(-qi)
        c = cs.cos(-qi)
        Rz = cs.SX.zeros(3, 3)
        Rz[0, 0] = c
        Rz[0, 1] = s
        Rz[1, 0] = -s
        Rz[1, 1] = c
        Rz[2, 2] = 1

        return Rz

    #print Rx(axis[0])
    #print Ry(axis[1])
    #print Rz(axis[2])

    R = cs.mtimes(Rx(), cs.mtimes(Ry(), Rz()))

    X[:3, :3] = R
    X[3:, 3:] = R
    return X


def XJ_revolute_T(axis, qi):
    """Returns a symbolic spatial rotation transformation matrix for a revolute joint"""
    X = cs.SX.zeros(6, 6)
    R = cs.SX.zeros(3, 3)
    s = cs.sin(-qi)
    c = cs.cos(-qi)

    if axis[0] == 1:
        R[0, 0] = 1
        R[1, 1] = c
        R[2, 2] = c
        R[1, 2] = s
        R[2, 1] = -s

    elif axis[1] == 1:
        R[0, 0] = c
        R[0, 2] = -s
        R[1, 1] = 1
        R[2, 0] = s
        R[2, 2] = c

    else:
        R[0, 0] = c
        R[0, 1] = s
        R[1, 0] = -s
        R[1, 1] = c
        R[2, 2] = 1

    X[:3, :3] = R
    X[3:, 3:] = R
    return X



def XJ_revolute2(xyz, rpy, axis, qi):
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
