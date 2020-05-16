"""Functions for getting casadi expressions for quaternions from joint type."""
import casadi as cs
import numpy as np


def revolute(xyz, rpy, axis, qi):
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
    return product(q_or, q_j)


def product(quat0, quat1):
    """Returns the quaternion product of q0 and q1."""
    quat = cs.SX.zeros(4)
    x0, y0, z0, w0 = quat0[0], quat0[1], quat0[2], quat0[3]
    x1, y1, z1, w1 = quat1[0], quat1[1], quat1[2], quat1[3]
    quat[0] = w0*x1 + x0*w1 + y0*z1 - z0*y1
    quat[1] = w0*y1 - x0*z1 + y0*w1 + z0*x1
    quat[2] = w0*z1 + x0*y1 - y0*x1 + z0*w1
    quat[3] = w0*w1 - x0*x1 - y0*y1 - z0*z1
    return quat


def numpy_rpy(roll, pitch, yaw):
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


def numpy_product(quat0, quat1):
    """Returns the quaternion product of q0 and q1."""
    quat = np.zeros(4)
    x0, y0, z0, w0 = quat0[0], quat0[1], quat0[2], quat0[3]
    x1, y1, z1, w1 = quat1[0], quat1[1], quat1[2], quat1[3]
    quat[0] = w0*x1 + x0*w1 + y0*z1 - z0*y1
    quat[1] = w0*y1 - x0*z1 + y0*w1 + z0*x1
    quat[2] = w0*z1 + x0*y1 - y0*x1 + z0*w1
    quat[3] = w0*w1 - x0*x1 - y0*y1 - z0*z1
    return quat


def numpy_ravani_roth_dist(q1, q2):
    """Quaternion distance designed by ravani and roth.
    See comparisons at: https://link.springer.com/content/pdf/10.1007%2Fs10851-009-0161-2.pdf"""
    return min(np.linalg.norm(q1 - q2), np.linalg.norm(q1 + q2))


def numpy_inner_product_dist(q1, q2):
    """Quaternion distance based on innerproduct.
    See comparisons at: https://link.springer.com/content/pdf/10.1007%2Fs10851-009-0161-2.pdf"""
    return 1.0 - abs(q1[0]*q2[0] + q1[1]*q2[1] + q1[2]*q2[2] + q1[3]*q2[3])
