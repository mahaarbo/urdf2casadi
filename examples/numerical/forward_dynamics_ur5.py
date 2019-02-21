import rbdl
import numpy as np
import casadi as cs
from urdf_parser_py.urdf import URDF, Pose
import os
import urdf2casadi.urdf2casadi.urdfparser as u2c

urmodel = rbdl.loadModel("ur5_rbdl.urdf")
asd = u2c.URDFparser()
robot_desc = asd.from_file("ur5_rbdl.urdf")
root = "base_link"
tip = "wrist_3_link"


jointlist, names, q_max, q_min = asd.get_joint_info(root, tip)
n_joints = len(jointlist)


q = np.zeros(n_joints)
qdot = np.zeros(n_joints)
qddot_rbdl = np.zeros(n_joints)
tau = np.zeros(n_joints)

gravity = [0., 0., -9.81]
error = np.zeros(n_joints)

def u2c2np(asd):
    return cs.Function("temp",[],[asd])()["o0"].toarray()

n_itr = 100

for i in range(n_itr):
    for j in range(n_joints):
        q[j] = (q_max[j] - q_min[j])*np.random.rand()-(q_max[j] - q_min[j])/2
        qdot[j] = (q_max[j] - q_min[j])*np.random.rand()-(q_max[j] - q_min[j])/2
        #Har tau noen restrictions?
        tau[j] = (q_max[j] - q_min[j])*np.random.rand()-(q_max[j] - q_min[j])/2

    rbdl.ForwardDynamics(urmodel, q, qdot, tau, qddot_rbdl)

    #Skal tau legges inn symbolsk eller ok med den numerisk?
    qddot_sym = asd.get_forward_dynamics_CRBA(root, tip, tau, gravity)
    qddot_u2c = qddot_sym(q, qdot)

    for qddot_idx in range(n_joints):
        error[qddot_idx] += np.absolute(qddot_rbdl[qddot_idx] - u2c2np(qddot_u2c)[qddot_idx])

print "Errors in forward dynamics joint accelerations with",n_itr, "iterations and comparing against rbdl:\n", error

sum_error = 0
for err in range(6):
    sum_error += error[err]
print "Sum of errors:\n", sum_error
