import rbdl
import numpy as np
import casadi as cs
from urdf_parser_py.urdf import URDF, Pose
import os
import urdf2casadi.urdfparser as u2c

pr2_model = rbdl.loadModel("pr2.urdf")

root = "pr2_root"
tip = "pr2_tool0"
pr2 = u2c.URDFparser()
pr2.from_file("pr2.urdf")

jointlist, names, q_max, q_min = pr2.get_joint_info(root, tip)
n_joints = pr2.get_n_joints(root, tip)

q_rbdl = np.zeros(n_joints)
qdot_rbdl = np.zeros(n_joints)
tau_rbdl = np.zeros(n_joints)
fd_rbdl = np.zeros(n_joints)

q = [None]*n_joints
qdot = [None]*n_joints
tau = [None]*n_joints
fd_sym = pr2.get_forward_dynamics_crba(root, tip, gravity)


gravity = [0., 0., -9.81]
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

    rbdl.ForwardDynamics(pr2_model, q_rbdl, qdot_rbdl, tau, fd_rbdl)

    fd_u2c = qddot_sym(q, qdot, tau)

    for fd_idx in range(n_joints):
        error[fd_idx] += np.absolute(fd_rbdl[fd_idx] - u2c2np(fd_u2c)[fd_idx])

print "Errors in forward dynamics joint accelerations with",n_itr, "iterations and comparing against rbdl:\n", error

sum_error = 0
for err in range(6):
    sum_error += error[err]
print "Sum of errors:\n", sum_error
