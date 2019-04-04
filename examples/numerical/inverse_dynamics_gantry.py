import rbdl
import numpy as np
import casadi as cs
from urdf_parser_py.urdf import URDF, Pose
import os
import urdf2casadi.urdfparser as u2c

urmodel = rbdl.loadModel("gantry.urdf")
asd = u2c.URDFparser()
robot_desc = asd.from_file("gantry.urdf")
root = 'gantry_root'
tip = 'gantry_tool0'


jointlist, names, q_max, q_min = asd.get_joint_info(root, tip)
n_joints = asd.get_n_joints(root, tip)

q = np.zeros(n_joints)
qdot = np.zeros(n_joints)
qddot = np.zeros(n_joints)
tau_rbdl = np.zeros(n_joints)

gravity = [0., 0., -9.81]
tau_sym = asd.get_inverse_dynamics_rnea(root, tip, gravity)
error = np.zeros(n_joints)

def u2c2np(asd):
    return cs.Function("temp",[],[asd])()["o0"].toarray()

n_itr = 1000
for i in range(n_itr):
    for j in range(n_joints):
        q[j] = (q_max[j] - q_min[j])*np.random.rand()-(q_max[j] - q_min[j])/2
        qdot[j] = (q_max[j] - q_min[j])*np.random.rand()-(q_max[j] - q_min[j])/2
        qddot[j] = (q_max[j] - q_min[j])*np.random.rand()-(q_max[j] - q_min[j])/2

    rbdl.InverseDynamics(urmodel, q, qdot, qddot, tau_rbdl)
    tau_u2c = tau_sym(q, qdot, qddot)

    for tau_idx in range(n_joints):
        error[tau_idx] += np.absolute(tau_rbdl[tau_idx] - u2c2np(tau_u2c)[tau_idx])

print "Errors in inverse dynamics forces with",n_itr, "iterations and comparing against rbdl:\n", error

sum_error = 0
for err in range(n_joints):
    sum_error += error[err]
print "Sum of errors:\n", sum_error
