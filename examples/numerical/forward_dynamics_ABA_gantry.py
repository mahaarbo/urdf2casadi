#import rbdl
import numpy as np
import casadi as cs
from urdf_parser_py.urdf import URDF, Pose
import os
import urdf2casadi.urdfparser as u2c


asd = u2c.URDFparser()
robot_desc = asd.from_file("/home/lillmaria/urdf2casadi/examples/urdf/thrivaldi.urdf")
root = "gantry_root"
tip = "gantry_tool0"

jointlist, names, q_max, q_min = asd.get_joint_info(root, tip)
n_joints = asd.get_n_joints(root, tip)

q = [None]*(n_joints)
qdot = [None]*(n_joints)
tau = [None]*(n_joints)

gravity = [0., 0., -9.81]
error = np.zeros(n_joints)

qddot_sym_aba = asd.get_forward_dynamics_aba(root, tip, gravity = gravity)
qddot_sym_crba = asd.get_forward_dynamics_crba(root, tip, gravity = gravity)


def u2c2np(asd):
    return cs.Function("temp",[],[asd])()["o0"].toarray()

n_itr = 10

for i in range(n_itr):
    for j in range(n_joints):
        q[j] = (q_max[j] - q_min[j])*np.random.rand()-(q_max[j] - q_min[j])/2
        qdot[j] = (q_max[j] - q_min[j])*np.random.rand()-(q_max[j] - q_min[j])/2
        tau[j] = (q_max[j] - q_min[j])*np.random.rand()-(q_max[j] - q_min[j])/2

    qddot_u2c_crba = qddot_sym_crba(q, qdot, tau)
    qddot_u2c_aba = qddot_sym_aba(q, qdot, tau)

    for qddot_idx in range(n_joints):
        error[qddot_idx] += np.absolute(u2c2np(qddot_u2c_aba[qddot_idx]) - u2c2np(qddot_u2c_crba)[qddot_idx])

print "Errors in forward dynamics aba joint accelerations with",n_itr, "iterations and comparing against u2c crba:\n", error

sum_error = 0
for err in range(n_joints):
    sum_error += error[err]
print "Sum of errors:\n", sum_error
