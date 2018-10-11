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
    roll, pitch, yaw = rpy
    # Origin rotation from RPY ZYX convention
    cr = cs.cos(roll/2.0)
    sr = cs.sin(roll/2.0)
    cp = cs.cos(pitch/2.0)
    sp = cs.sin(pitch/2.0)
    cy = cs.cos(yaw/2.0)
    sy = cs.sin(yaw/2.0)

    # The quaternion associated with the origin rotation
    # Note: quat = [ xyz, w], where w is the scalar part
    x_or = cy*sr*cp - sy*cr*sp
    y_or = cy*cr*sp + sy*sr*cp
    z_or = sy*cr*cp - cy*sr*sp
    w_or = cy*cr*cp + sy*sr*sp
    q_or = [x_or, y_or, z_or, w_or]
    # Joint rotation from axis angle
    cqi = cs.cos(qi/2.0)
    sqi = cs.sin(qi/2.0)
    x_j = axis[0]*sqi
    y_j = axis[1]*sqi
    z_j = axis[2]*sqi
    w_j = cqi
    q_j = [x_j, y_j, z_j, w_j]
    # Resulting quaternion
    return quaternion_product(q_or, q_j)


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


def T_denavit_hartenberg(joint_angle, link_length, link_offset, link_twist):
    """Returns a transformation matrix based on denavit hartenberg
    parameters."""
    T = cs.SX.zeros(4, 4)
    T[0, 0] = cs.cos(joint_angle)
    T[0, 1] = -cs.sin(joint_angle)*cs.cos(link_twist)
    T[0, 2] = cs.sin(joint_angle)*cs.sin(link_twist)
    T[0, 3] = link_length*cs.cos(joint_angle)
    T[1, 0] = cs.sin(joint_angle)
    T[1, 1] = cs.cos(joint_angle)*cs.cos(link_twist)
    T[1, 2] = -cs.cos(joint_angle)*cs.sin(link_twist)
    T[1, 3] = link_length*cs.sin(joint_angle)
    T[2, 1] = cs.sin(link_twist)
    T[2, 2] = cs.cos(link_twist)
    T[2, 3] = link_offset
    T[3, 3] = 1.0
    return T


def quaternion_product(quat0, quat1):
    """Returns the quaternion product of q0 and q1."""
    quat = cs.SX.zeros(4)
    x0, y0, z0, w0 = quat0[0], quat0[1], quat0[2], quat0[3]
    x1, y1, z1, w1 = quat1[0], quat1[1], quat1[2], quat1[3]
    quat[0] = w0*x1 + x0*w1 + y0*z1 - z0*y1
    quat[1] = w0*y1 - x0*z1 + y0*w1 + z0*x1
    quat[2] = w0*z1 + x0*y1 - y0*x1 + z0*w1
    quat[3] = w0*w1 - x0*x1 - y0*y1 - z0*z1
    return quat


def quaternion_conj(quat):
    res = cs.SX.zeros(4)
    res[0] = -quat[0]
    res[1] = -quat[1]
    res[2] = -quat[2]
    res[3] = quat[3]
    return res


def dual_quaternion_product(Q, P):
    """Returns the dual quaternion product of two 8 element vectors
    representing a dual quaternions. First four elements are the real
    part, last four elements are the dual part.
    """
    res = cs.SX.zeros(8)
    # Real and dual components
    xr0, yr0, zr0, wr0 = Q[0], Q[1], Q[2], Q[3]
    xd0, yd0, zd0, wd0 = Q[4], Q[5], Q[6], Q[7]
    xr1, yr1, zr1, wr1 = P[0], P[1], P[2], P[3]
    xd1, yd1, zd1, wd1 = P[4], P[5], P[6], P[7]
    # Real part
    xr = wr0*xr1 + xr0*wr1 + yr0*zr1 - zr0*yr1
    yr = wr0*yr1 - xr0*zr1 + yr0*wr1 + zr0*xr1
    zr = wr0*zr1 + xr0*yr1 - yr0*xr1 + zr0*wr1
    wr = wr0*wr1 - xr0*xr1 - yr0*yr1 - zr0*zr1
    # Dual part
    xd = xr0*wd1 + wr0*xd1 + yr0*zd1 - zr0*yd1
    xd += xd0*wr1 + wd0*xr1 + yd0*zr1 - zd0*yr1
    yd = wr0*yd1 - xr0*zd1 + yr0*wd1 + zr0*xd1
    yd += wd0*yr1 - xd0*zr1 + yd0*wr1 + zd0*xr1
    zd = wr0*zd1 + xr0*yd1 - yr0*xd1 + zr0*wd1
    zd += wd0*zr1 + xd0*yr1 - yd0*xr1 + zd0*wr1
    wd = wr1*wd0 - xr1*xd0 - yr1*yd0 - zr1*zd0
    wd += wd1*wr0 - xd1*xr0 - yd1*yr0 - zd1*zr0

    res[0] = xr
    res[1] = yr
    res[2] = zr
    res[3] = wr
    res[4] = xd
    res[5] = yd
    res[6] = zd
    res[7] = wd
    return res


def dual_quaternion_conj(Q):
    """Returns the conjugate of a dual quaternion.
    """
    res = cs.SX.zeros(8)
    res[0] = -Q[0]
    res[1] = -Q[1]
    res[2] = -Q[2]
    res[3] = Q[3]
    res[4] = -Q[4]
    res[5] = -Q[5]
    res[6] = -Q[6]
    res[7] = Q[7]
    return res


def dual_quaternion_norm2(Q):
    """Returns the dual norm of a dual quaternion.
    Based on:
    https://github.com/bobbens/libdq/blob/master/dq.c
    """
    real = cs.SX.zeros(1)
    dual = cs.SX.zeros(1)
    real = Q[0]*Q[0] + Q[1]*Q[1] + Q[2]*Q[2] + Q[3]*Q[3]
    dual = 2.*(Q[3]*Q[7] + Q[0]*Q[4] + Q[1]*Q[5] + Q[2]*Q[6])
    return real, dual


def dual_quaternion_inv(Q):
    """Returns the inverse of a dual quaternion.
    Based on:
    https://github.com/bobbens/libdq/blob/master/dq.c
    """
    res = cs.SX.zeros(8)
    real, dual = dual_quaternion_norm2(Q)
    res[0] = -Q[0] * real
    res[1] = -Q[1] * real
    res[2] = -Q[2] * real
    res[3] = Q[3] * real
    res[4] = Q[4] * (dual-real)
    res[5] = Q[5] * (dual-real)
    res[6] = Q[6] * (dual-real)
    res[7] = Q[7] * (real-dual)
    return res


def dual_quaternion_to_transformation_matrix(Q):
    """Transforms a dual quaternion to a 4x4 transformation matrix.
    """
    res = cs.SX.zeros(4, 4)
    # Rotation part:
    xr, yr, zr, wr = Q[0], Q[1], Q[2], Q[3]
    xd, yd, zd, wd = Q[4], Q[5], Q[6], Q[7]
    res[0, 0] = wr*wr + xr*xr - yr*yr - zr*zr
    res[1, 1] = wr*wr - xr*xr + yr*yr - zr*zr
    res[2, 2] = wr*wr - xr*xr - yr*yr + zr*zr
    res[0, 1] = 2.*(xr*yr - wr*zr)
    res[1, 0] = 2.*(xr*yr + wr*zr)
    res[0, 2] = 2.*(xr*zr + wr*yr)
    res[2, 0] = 2.*(xr*zr - wr*yr)
    res[1, 2] = 2.*(yr*zr - wr*xr)
    res[2, 1] = 2.*(yr*zr + wr*xr)

    # Displacement part:
    res[0, 3] = 2.*(-wd*xr + xd*wr - yd*zr + zd*yr)
    res[1, 3] = 2.*(-wd*yr + xd*zr + yd*wr - zd*xr)
    res[2, 3] = 2.*(-wd*zr - xd*yr + yd*xr + zd*wr)
    res[3, 3] = 1.0
    return res


def dual_quaternion_to_rotation_matrix(Q):
    """Transforms a dual quaternion to 3x3 rotation matrix
    """
    res = cs.MX.zeros(3, 3)
    xr, yr, zr, wr = Q[0], Q[1], Q[2], Q[3]
    res[0, 0] = wr*wr + xr*xr - yr*yr - zr*zr
    res[1, 1] = wr*wr - xr*xr + yr*yr - zr*zr
    res[2, 2] = wr*wr - xr*xr - yr*yr + zr*zr
    res[0, 1] = 2.*(xr*yr - wr*zr)
    res[1, 0] = 2.*(xr*yr + wr*zr)
    res[0, 2] = 2.*(xr*zr + wr*yr)
    res[2, 0] = 2.*(xr*zr - wr*yr)
    res[1, 2] = 2.*(yr*zr - wr*xr)
    res[2, 1] = 2.*(yr*zr + wr*xr)
    return res


def dual_quaternion_to_position(Q):
    """Transforms a dual quaternion to a position.
    """
    res = cs.MX.zeros(3)
    xr, yr, zr, wr = Q[0], Q[1], Q[2], Q[3]
    xd, yd, zd, wd = Q[4], Q[5], Q[6], Q[7]
    res[0] = 2.*(-wd*xr + xd*wr - yd*zr + zd*yr)
    res[1] = 2.*(-wd*yr + xd*zr + yd*wr - zd*xr)
    res[2] = 2.*(-wd*zr - xd*yr + yd*xr + zd*wr)
    return res


def dual_quaternion_rpy(rpy):
    """Returns the dual quaternion for a pure roll-pitch-yaw rotation.
    """
    roll, pitch, yaw = rpy[0], rpy[1], rpy[2]
    # Origin rotation from RPY ZYX convention
    cr = cs.cos(roll/2.0)
    sr = cs.sin(roll/2.0)
    cp = cs.cos(pitch/2.0)
    sp = cs.sin(pitch/2.0)
    cy = cs.cos(yaw/2.0)
    sy = cs.sin(yaw/2.0)

    # The quaternion associated with the origin rotation
    # Note: quat = w + ix + jy + kz
    x_or = cy*sr*cp - sy*cr*sp
    y_or = cy*cr*sp + sy*sr*cp
    z_or = sy*cr*cp - cy*sr*sp
    w_or = cy*cr*cp + sy*sr*sp
    res = cs.SX.zeros(8)
    # Note, our dual quaternions use a different representation
    # dual_quat = [xyz, w, xyz', w']
    # where w + xyz represents the "real" quaternion
    # and w'+xyz' represents the "dual" quaternion
    res[0] = x_or
    res[1] = y_or
    res[2] = z_or
    res[3] = w_or
    return res


def dual_quaternion_translation(xyz):
    """Returns the dual quaternion for a pure translation.
    """
    res = cs.SX.zeros(8)
    res[3] = 1.0
    res[4] = xyz[0]/2.0
    res[5] = xyz[1]/2.0
    res[6] = xyz[2]/2.0
    return res


def dual_quaternion_axis_translation(axis, qi):
    """Returns the dual quaternion for a translation along an axis.
    """
    res = cs.SX.zeros(8)
    res[3] = 1.0
    res[4] = qi*axis[0]/2.0
    res[5] = qi*axis[1]/2.0
    res[6] = qi*axis[2]/2.0
    return res


def dual_quaternion_axis_rotation(axis, qi):
    """Returns the dual quaternion for a rotation along an axis.
    AXIS MUST BE NORMALIZED!
    """
    res = cs.SX.zeros(8)
    cqi = cs.cos(qi/2.0)
    sqi = cs.sin(qi/2.0)
    res[0] = axis[0]*sqi
    res[1] = axis[1]*sqi
    res[2] = axis[2]*sqi
    res[3] = cqi
    return res


def dual_quaternion_prismatic(xyz, rpy, axis, qi):
    """Returns the dual quaternion for a prismatic joint.
    """
    # Joint origin rotation from RPY ZYX convention
    roll, pitch, yaw = rpy[0], rpy[1], rpy[2]
    # Origin rotation from RPY ZYX convention
    cr = cs.cos(roll/2.0)
    sr = cs.sin(roll/2.0)
    cp = cs.cos(pitch/2.0)
    sp = cs.sin(pitch/2.0)
    cy = cs.cos(yaw/2.0)
    sy = cs.sin(yaw/2.0)
    # The quaternion associated with the origin rotation
    # Note: quat = w + ix + jy + kz
    x_or = cy*sr*cp - sy*cr*sp
    y_or = cy*cr*sp + sy*sr*cp
    z_or = sy*cr*cp - cy*sr*sp
    w_or = cy*cr*cp + sy*sr*sp
    # Joint origin translation as a dual quaternion
    x_ot = 0.5*xyz[0]*w_or + 0.5*xyz[1]*z_or - 0.5*xyz[2]*y_or
    y_ot = - 0.5*xyz[0]*z_or + 0.5*xyz[1]*w_or + 0.5*xyz[2]*x_or
    z_ot = 0.5*xyz[0]*y_or - 0.5*xyz[1]*x_or + 0.5*xyz[2]*w_or
    w_ot = - 0.5*xyz[0]*x_or - 0.5*xyz[1]*y_or - 0.5*xyz[2]*z_or
    Q_o = [x_or, y_or, z_or, w_or, x_ot, y_ot, z_ot, w_ot]
    # Joint displacement orientation is just identity
    x_jr = 0.0
    y_jr = 0.0
    z_jr = 0.0
    w_jr = 1.0
    # Joint displacement translation along axis
    x_jt = qi*axis[0]/2.0
    y_jt = qi*axis[1]/2.0
    z_jt = qi*axis[2]/2.0
    w_jt = 0.0
    Q_j = [x_jr, y_jr, z_jr, w_jr, x_jt, y_jt, z_jt, w_jt]
    # Get resulting dual quaternion
    return dual_quaternion_product(Q_o, Q_j)


def dual_quaternion_revolute(xyz, rpy, axis, qi):
    """Returns the dual quaternion for a revolute joint.
    AXIS MUST BE NORMALIZED!
    """
    # Joint origin rotation from RPY ZYX convention
    roll, pitch, yaw = rpy[0], rpy[1], rpy[2]
    # Origin rotation from RPY ZYX convention
    cr = cs.cos(roll/2.0)
    sr = cs.sin(roll/2.0)
    cp = cs.cos(pitch/2.0)
    sp = cs.sin(pitch/2.0)
    cy = cs.cos(yaw/2.0)
    sy = cs.sin(yaw/2.0)
    # The quaternion associated with the origin rotation
    # Note: quat = w + ix + jy + kz
    x_or = cy*sr*cp - sy*cr*sp
    y_or = cy*cr*sp + sy*sr*cp
    z_or = sy*cr*cp - cy*sr*sp
    w_or = cy*cr*cp + sy*sr*sp
    # Joint origin translation as a dual quaternion
    x_ot = 0.5*xyz[0]*w_or + 0.5*xyz[1]*z_or - 0.5*xyz[2]*y_or
    y_ot = - 0.5*xyz[0]*z_or + 0.5*xyz[1]*w_or + 0.5*xyz[2]*x_or
    z_ot = 0.5*xyz[0]*y_or - 0.5*xyz[1]*x_or + 0.5*xyz[2]*w_or
    w_ot = - 0.5*xyz[0]*x_or - 0.5*xyz[1]*y_or - 0.5*xyz[2]*z_or
    Q_o = [x_or, y_or, z_or, w_or, x_ot, y_ot, z_ot, w_ot]
    # Joint displacement rotation is from axis angle
    #nax = cs.sqrt(axis[0]*axis[0] + axis[1]*axis[1] + axis[2]*axis[2])
    cqi = cs.cos(qi/2.0)
    sqi = cs.sin(qi/2.0)
    x_jr = axis[0]*sqi
    y_jr = axis[1]*sqi
    z_jr = axis[2]*sqi
    w_jr = cqi
    # Joint displacement translation is nothing
    x_jt = 0.0
    y_jt = 0.0
    z_jt = 0.0
    w_jt = 0.0
    Q_j = [x_jr, y_jr, z_jr, w_jr, x_jt, y_jt, z_jt, w_jt]
    return dual_quaternion_product(Q_o, Q_j)


def dual_quaternion_to_pos(Q):
    """Returns the Cartesian position in a dual quaternion."""
    quaternion_rot_conj = quaternion_conj(Q[:4])
    quaternion_disp = Q[4:8]
    return 2*quaternion_product(quaternion_disp, quaternion_rot_conj)[:3]


def dual_quaternion_denavit_hartenberg(joint_angle, link_length,
                                       link_offset, link_twist):
    """Returns a transformation matrix based on denavit hartenberg
    parameters."""
    Q_rot_z = dual_quaternion_axis_rotation([0., 0., 1.], joint_angle)
    Q_trans_z = dual_quaternion_axis_translation([0., 0., 1.], link_offset)
    Q_trans_x = dual_quaternion_axis_translation([1., 0., 0.], link_length)
    Q_rot_x = dual_quaternion_axis_rotation([1., 0., 0.], link_twist)
    Q_z = dual_quaternion_product(Q_rot_z, Q_trans_z)
    Q_x = dual_quaternion_product(Q_trans_x, Q_rot_x)
    return dual_quaternion_product(Q_z, Q_x)
