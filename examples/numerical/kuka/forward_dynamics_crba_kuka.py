import rbdl
import numpy as np
import casadi as cs
from urdf_parser_py.urdf import URDF, Pose
import os
import urdf2casadi.urdfparser as u2c

kuka_model = rbdl.loadModel("../../urdf/kuka.urdf")

root = "calib_kuka_arm_base_link"
tip = "kuka_arm_7_link"
kuka = u2c.URDFparser()
kuka.from_file("../../urdf/kuka.urdf")

jointlist, names, q_max, q_min = kuka.get_joint_info(root, tip)
n_joints = kuka.get_n_joints(root, tip)

q_rbdl = np.zeros(n_joints)
qdot_rbdl = np.zeros(n_joints)
tau_rbdl = np.zeros(n_joints)
fd_rbdl = np.zeros(n_joints)

q = [None]*n_joints
qdot = [None]*n_joints
tau = [None]*n_joints

gravity = [0., 0., -9.81]
fd_sym = kuka.get_forward_dynamics_crba(root, tip, gravity)
error = np.zeros(n_joints)

def u2c2np(asd):
    return cs.Function("temp",[],[asd])()["o0"].toarray()

n_itr = 1000
for i in range(n_itr):
    for j in range(n_joints):
        q[j] = (q_max[j] - q_min[j])*np.random.rand()-(q_max[j] - q_min[j])/2
        qdot[j] = (q_max[j] - q_min[j])*np.random.rand()-(q_max[j] - q_min[j])/2
        tau[j] = (q_max[j] - q_min[j])*np.random.rand()-(q_max[j] - q_min[j])/2

        q_rbdl[j] = q[j]
        qdot_rbdl[j] = qdot[j]
        tau_rbdl[j] = tau[j]

    rbdl.ForwardDynamics(kuka_model, q_rbdl, qdot_rbdl, tau_rbdl, fd_rbdl)

    fd_u2c = fd_sym(q, qdot, tau)

    for fd_idx in range(n_joints):
        error[fd_idx] += np.absolute(fd_rbdl[fd_idx] - u2c2np(fd_u2c)[fd_idx])

print "Errors in forward dynamics joint accelerations with",n_itr, "iterations and comparing against rbdl:\n", error

sum_error = 0
for err in range(6):
    sum_error += error[err]
print "Sum of errors:\n", sum_error
