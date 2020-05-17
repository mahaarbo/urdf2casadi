"""Functions for getting casadi expressions for transformation matrices from
joint type."""
import casadi as cs
import numpy as np


def prismatic(xyz, rpy, axis, qi):
    T = cs.SX.zeros(4, 4)

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
    p0 = r00*axis[0]*qi + r01*axis[1]*qi + r02*axis[2]*qi
    p1 = r10*axis[0]*qi + r11*axis[1]*qi + r12*axis[2]*qi
    p2 = r20*axis[0]*qi + r21*axis[1]*qi + r22*axis[2]*qi

    # Homogeneous transformation matrix
    T[0, 0] = r00
    T[0, 1] = r01
    T[0, 2] = r02
    T[1, 0] = r10
    T[1, 1] = r11
    T[1, 2] = r12
    T[2, 0] = r20
    T[2, 1] = r21
    T[2, 2] = r22
    T[0, 3] = xyz[0] + p0
    T[1, 3] = xyz[1] + p1
    T[2, 3] = xyz[2] + p2
    T[3, 3] = 1.0
    return T


def revolute(xyz, rpy, axis, qi):
    T = cs.SX.zeros(4, 4)

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
    T[0, 0] = r00*s00 + r01*s10 + r02*s20
    T[1, 0] = r10*s00 + r11*s10 + r12*s20
    T[2, 0] = r20*s00 + r21*s10 + r22*s20

    T[0, 1] = r00*s01 + r01*s11 + r02*s21
    T[1, 1] = r10*s01 + r11*s11 + r12*s21
    T[2, 1] = r20*s01 + r21*s11 + r22*s21

    T[0, 2] = r00*s02 + r01*s12 + r02*s22
    T[1, 2] = r10*s02 + r11*s12 + r12*s22
    T[2, 2] = r20*s02 + r21*s12 + r22*s22

    T[0, 3] = xyz[0]
    T[1, 3] = xyz[1]
    T[2, 3] = xyz[2]
    T[3, 3] = 1.0
    return T


def full_symbolic(xyz, rpy):
    """Gives a symbolic transformation matrix."""
    T = cs.SX.zeros(4, 4)
    cr = cs.cos(rpy[0])
    sr = cs.sin(rpy[0])
    cp = cs.cos(rpy[1])
    sp = cs.sin(rpy[1])
    cy = cs.cos(rpy[2])
    sy = cs.sin(rpy[2])
    T[0, 0] = cy*cp
    T[0, 1] = cy*sp*sr - sy*cr
    T[0, 2] = cy*sp*cr + sy*sr
    T[1, 0] = sy*cp
    T[1, 1] = sy*sp*sr + cy*cr
    T[1, 2] = sy*sp*cr - cy*sr
    T[2, 0] = -sp
    T[2, 1] = cp*sr
    T[2, 2] = cp*cr
    T[0, 3] = xyz[0]
    T[1, 3] = xyz[1]
    T[2, 3] = xyz[2]
    T[3, 3] = 1.0
    return T


def numpy_normalize(v):
    nv = np.linalg.norm(v)
    if nv > 0.0:
        v[0] = v[0]/nv
        v[1] = v[1]/nv
        v[2] = v[2]/nv
    return v


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


def numpy_rpy(displacement, roll, pitch, yaw):
    """Homogeneous transformation matrix with roll pitch yaw."""
    T = np.zeros([4, 4])
    T[:3, :3] = numpy_rotation_rpy(roll, pitch, yaw)
    T[:3, 3] = displacement
    T[3, 3] = 1.0
    return T


def numpy_rotation_distance_from_identity(R1, R2):
    """Rotation matrix distance based on distance from identity matrix.
    See comparisons at: https://link.springer.com/content/pdf/10.1007%2Fs10851-009-0161-2.pdf"""
    return np.linalg.norm(np.eye(1) - np.dot(R1, R2.T))
