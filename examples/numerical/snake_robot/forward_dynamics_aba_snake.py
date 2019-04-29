import numpy as np
import casadi as cs
from urdf_parser_py.urdf import URDF, Pose
import os
import urdf2casadi.urdfparser as u2c

root = "base_link"
tip = "link8"

snake_robot = u2c.URDFparser()
snake_robot.from_file("/home/lillmaria/urdf2casadi/examples/urdf/snake_robot.urdf")

jointlist, names, q_max, q_min = snake_robot.get_joint_info(root, tip)
n_joints = snake_robot.get_n_joints(root, tip)

q = [None]*(n_joints)
qdot = [None]*(n_joints)
tau = [None]*(n_joints)

gravity = [0., 0., -9.81]
error = np.zeros(n_joints)

fd_sym_aba = snake_robot.get_forward_dynamics_aba(root, tip, gravity = gravity)
fd_sym_crba = snake_robot.get_forward_dynamics_crba(root, tip, gravity = gravity)


def u2c2np(asd):
    return cs.Function("temp",[],[asd])()["o0"].toarray()

n_itr = 1000

for i in range(n_itr):
    for j in range(n_joints):
        q[j] = (q_max[j] - q_min[j])*np.random.rand()-(q_max[j] - q_min[j])/2
        qdot[j] = (q_max[j] - q_min[j])*np.random.rand()-(q_max[j] - q_min[j])/2
        tau[j] = (q_max[j] - q_min[j])*np.random.rand()-(q_max[j] - q_min[j])/2

    fd_u2c_crba = fd_sym_crba(q, qdot, tau)
    fd_u2c_aba = fd_sym_aba(q, qdot, tau)

    for fd_idx in range(n_joints):
        error[fd_idx] += np.absolute(u2c2np(fd_u2c_aba[fd_idx]) - u2c2np(fd_u2c_crba)[fd_idx])

print "Errors in forward dynamics aba joint accelerations with",n_itr, "iterations and comparing against u2c crba:\n", error

sum_error = 0
for err in range(n_joints):
    sum_error += error[err]
print "Sum of errors:\n", sum_error
