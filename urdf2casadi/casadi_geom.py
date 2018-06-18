"""Functions for getting casadi expressions from joint type."""
import casadi as cs


def T_prismatic(xyz, rpy, axis, qi):
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


def T_revolute(xyz, rpy, axis, qi):
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


def quaternion_revolute(xyz, rpy, axis, qi):
    """Gives a casadi function for the quaternion. [xyz, w] form."""
    #quat = cs.SX.zeros(4, 1)
    roll, pitch, yaw = rpy
    # Origin rotation from RPY ZYX convention
    cr = cs.cos(roll/2.0)
    sr = cs.sin(roll/2.0)
    cp = cs.cos(pitch/2.0)
    sp = cs.sin(pitch/2.0)
    cy = cs.cos(yaw/2.0)
    sy = cs.sin(yaw/2.0)

    # The quaternion associated with the static joint frame rotation
    # Note: quat = [ xyz, w], where w is the scalar part
    x0 = sr*cp*cy - cr*sp*sy
    y0 = cr*sp*cy + sr*cp*sy
    z0 = cr*cp*sy - sr*sp*cy
    w0 = cr*cp*cy + sr*sp*sy

    # Joint rotation from axis angle
    cqi = cs.cos(qi/2.0)
    sqi = cs.sin(qi/2.0)
    x1 = axis[0]*sqi
    y1 = axis[1]*sqi
    z1 = axis[2]*sqi
    w1 = cqi

    # Resulting quaternion
    return quaternion_product([x1, y1, z1, w1], [x0, y0, z0, w0])


def T_full_symbolic(xyz, rpy):
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


def quaternion_product(quat0, quat1):
    """Returns the quaternion product of q0 and q1."""
    quat = cs.SX.zeros(4)
    x0, y0, z0, w0 = quat0[0], quat0[1], quat0[2], quat0[3]
    x1, y1, z1, w1 = quat1[0], quat1[1], quat1[2], quat1[3]
    quat[0] = x1*w0 + y1*z0 - z1*y0 + w1*x0
    quat[1] = -x1*z0 + y1*w0 + z1*x0 + w1*y0
    quat[2] = x1*y0 - y1*x0 + z1*w0 + w1*z0
    quat[3] = -x1*x0 - y1*y0 - z1*z0 + w1*w0
    return quat
