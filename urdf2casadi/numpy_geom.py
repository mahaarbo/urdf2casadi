import numpy as np


def normalize(v):
    nv = np.linalg.norm(v)
    if nv > 0.0:
        v[0] = v[0]/nv
        v[1] = v[1]/nv
        v[2] = v[2]/nv
    return v


def skew_symmetric(v):
    """Returns a skew symmetric matrix from vector. p q r"""
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])


def rotation_rpy(roll, pitch, yaw):
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


def quaternion_rpy(roll, pitch, yaw):
    """Returns a quaternion ([x,y,z,w], w scalar) from roll pitch yaw ZYX
    convention."""
    cr = np.cos(roll/2.0)
    sr = np.sin(roll/2.0)
    cp = np.cos(pitch/2.0)
    sp = np.sin(pitch/2.0)
    cy = np.cos(yaw/2.0)
    sy = np.sin(yaw/2.0)
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    w = cr * cp * cy + sr * sp * sy
    # Remember to normalize:
    nq = np.sqrt(x*x + y*y + z*z + w*w)
    return np.array([x/nq,
                     y/nq,
                     z/nq,
                     w/nq])


def T_rpy(displacement, roll, pitch, yaw):
    """Homogeneous transformation matrix with roll pitch yaw."""
    T = np.zeros([4, 4])
    T[:3, :3] = rotation_rpy(roll, pitch, yaw)
    T[:3, 3] = displacement
    T[3, 3] = 1.0
    return T


def quaternion_product(quat0, quat1):
    """Returns the quaternion product of q0 and q1."""
    quat = np.zeros(4)
    x0, y0, z0, w0 = quat0[0], quat0[1], quat0[2], quat0[3]
    x1, y1, z1, w1 = quat1[0], quat1[1], quat1[2], quat1[3]
    quat[0] = w0*x1 + x0*w1 + y0*z1 - z0*y1
    quat[1] = w0*y1 - x0*z1 + y0*w1 + z0*x1
    quat[2] = w0*z1 + x0*y1 - y0*x1 + z0*w1
    quat[3] = w0*w1 - x0*x1 - y0*y1 - z0*z1
    return quat

def dual_quaternion_product(Q, P):
    """Returns the dual quaternion product of two 8 element vectors
    representing a dual quaternions. First four elements are the real
    part, last four elements are the dual part.
    """
    res = np.zeros(8)
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
    res = np.zeros(8)
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
    real = Q[0]*Q[0] + Q[1]*Q[1] + Q[2]*Q[2] + Q[3]*Q[3]
    dual = 2.*(Q[3]*Q[7] + Q[0]*Q[4] + Q[1]*Q[5] + Q[2]*Q[6])
    return real, dual


def dual_quaternion_inv(Q):
    """Returns the inverse of a dual quaternion.
    Based on:
    https://github.com/bobbens/libdq/blob/master/dq.c
    """
    res = np.zeros(8)
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
    res = np.zeros((4, 4))
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


def dual_quaternion_rpy(rpy):
    """Returns the dual quaternion for a pure roll-pitch-yaw rotation.
    """
    roll, pitch, yaw = rpy
    # Origin rotation from RPY ZYX convention
    cr = np.cos(roll/2.0)
    sr = np.sin(roll/2.0)
    cp = np.cos(pitch/2.0)
    sp = np.sin(pitch/2.0)
    cy = np.cos(yaw/2.0)
    sy = np.sin(yaw/2.0)

    # The quaternion associated with the origin rotation
    # Note: quat = w + ix + jy + kz
    x_or = cy*sr*cp - sy*cr*sp
    y_or = cy*cr*sp + sy*sr*cp
    z_or = sy*cr*cp - cy*sr*sp
    w_or = cy*cr*cp + sy*sr*sp
    res = np.zeros(8)
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
    res = np.zeros(8)
    res[3] = 1.0
    res[4] = xyz[0]/2.0
    res[5] = xyz[1]/2.0
    res[6] = xyz[2]/2.0
    return res


def dual_quaternion_axis_translation(axis, qi):
    """Returns the dual quaternion for a translation along an axis.
    """
    res = np.zeros(8)
    res[3] = 1.0
    res[4] = qi*axis[0]/2.0
    res[5] = qi*axis[1]/2.0
    res[6] = qi*axis[2]/2.0
    return res


def dual_quaternion_axis_rotation(axis, qi):
    """Returns the dual quaternion for a rotation along an axis.
    AXIS MUST BE NORMALIZED!
    """
    res = np.zeros(8)
    cqi = np.cos(qi/2.0)
    sqi = np.sin(qi/2.0)
    res[0] = axis[0]*sqi
    res[1] = axis[1]*sqi
    res[2] = axis[2]*sqi
    res[3] = cqi
    return res


def dual_quaternion_prismatic(xyz, rpy, axis, qi):
    """Returns the dual quaternion for a prismatic joint.
    """
    # Joint origin rotation from RPY ZYX convention
    roll, pitch, yaw = rpy
    # Origin rotation from RPY ZYX convention
    cr = np.cos(roll/2.0)
    sr = np.sin(roll/2.0)
    cp = np.cos(pitch/2.0)
    sp = np.sin(pitch/2.0)
    cy = np.cos(yaw/2.0)
    sy = np.sin(yaw/2.0)
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
    roll, pitch, yaw = rpy
    # Origin rotation from RPY ZYX convention
    cr = np.cos(roll/2.0)
    sr = np.sin(roll/2.0)
    cp = np.cos(pitch/2.0)
    sp = np.sin(pitch/2.0)
    cy = np.cos(yaw/2.0)
    sy = np.sin(yaw/2.0)
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
    cqi = np.cos(qi/2.0)
    sqi = np.sin(qi/2.0)
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



def quaternion_ravani_roth_dist(q1, q2):
    """Quaternion distance designed by ravani and roth.
    See comparisons at: https://link.springer.com/content/pdf/10.1007%2Fs10851-009-0161-2.pdf"""
    return min(np.linalg.norm(q1 - q2), np.linalg.norm(q1 + q2))


def quaternion_inner_product_dist(q1, q2):
    """Quaternion distance based on innerproduct.
    See comparisons at: https://link.springer.com/content/pdf/10.1007%2Fs10851-009-0161-2.pdf"""
    return 1.0 - abs(q1[0]*q2[0] + q1[1]*q2[1] + q1[2]*q2[2] + q1[3]*q2[3])


def rotation_distance_from_identity(R1, R2):
    """Rotation matrix distance based on distance from identity matrix.
    See comparisons at: https://link.springer.com/content/pdf/10.1007%2Fs10851-009-0161-2.pdf"""
    return np.linalg.norm(np.eye(1) - np.dot(R1, R2.T))
