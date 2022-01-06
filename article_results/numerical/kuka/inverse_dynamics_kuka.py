import rbdl
import numpy as np
import casadi as cs
from urdf_parser_py.urdf import URDF, Pose
import os
import urdf2casadi.urdfparser as u2c


root = "calib_kuka_arm_base_link"
tip = "kuka_arm_7_link"

urmodel = rbdl.loadModel("../../urdf/kuka.urdf")
kuka = u2c.URDFparser()
kuka.from_file("../../urdf/kuka.urdf")

jointlist, names, q_max, q_min = kuka.get_joint_info(root, tip)
n_joints = kuka.get_n_joints(root, tip)

q_rbdl = np.zeros(n_joints)
qdot_rbdl = np.zeros(n_joints)
qddot_rbdl = np.zeros(n_joints)
id_rbdl = np.zeros(n_joints)

q = [None]*n_joints
qdot = [None]*n_joints
qddot = [None]*n_joints
gravity = [0., 0., -9.81]
id_sym = kuka.get_inverse_dynamics_rnea(root, tip, gravity)
error = np.zeros(n_joints)

def u2c2np(asd):
    return cs.Function("temp", [], [asd])()["o0"].toarray()

n_itr = 1000
for i in range(n_itr):
    for j in range(n_joints):
        q[j] = (q_max[j] - q_min[j])*np.random.rand()-(q_max[j] - q_min[j])/2
        qdot[j] = (q_max[j] - q_min[j])*np.random.rand()-(q_max[j] - q_min[j])/2
        #qddot[j] = (q_max[j] - q_min[j])*np.random.rand()-(q_max[j] - q_min[j])/2

        q_rbdl[j] = q[j]
        qdot_rbdl[j] = qdot[j]
        #qddot_rbdl[j] = qddot[j]

    rbdl.InverseDynamics(urmodel, q_rbdl, qdot_rbdl, qddot_rbdl, id_rbdl)
    id_u2c = id_sym(q, qdot, qddot_rbdl)

    for id_idx in range(n_joints):
        error[id_idx] += np.absolute(id_rbdl[id_idx] - u2c2np(id_u2c)[id_idx])

print("Errors in inverse dynamics forces with",n_itr, "iterations and comparing against rbdl:\n", error)

sum_error = 0
for err in range(n_joints):
    sum_error += error[err]
print("Sum of errors:\n", sum_error)
