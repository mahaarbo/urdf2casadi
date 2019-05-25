import rbdl
import numpy as np
import casadi as cs
from urdf_parser_py.urdf import URDF, Pose
import os
import urdf2casadi.urdfparser as u2c
import pybullet as pb

path_to_urdf = "/home/lmjohann/urdf2casadi/examples/urdf/gantry.urdf"
root = "gantry_link_base"
tip = "gantry_tool0"

#get robot models
#rbdl

gantry_rbdl = rbdl.loadModel(path_to_urdf)

#u2c
gantry = u2c.URDFparser()
gantry.from_file(path_to_urdf)

#pybullet
sim = pb.connect(pb.DIRECT)
gantry_pb = pb.loadURDF(path_to_urdf, useFixedBase=True, flags = pb.URDF_USE_INERTIA_FROM_FILE)
pb.setGravity(0, 0, -9.81)

#joint info
jointlist, names, q_max, q_min = gantry.get_joint_info(root, tip)
n_joints = gantry.get_n_joints(root, tip)


#u2c & pybullet
q = [None]*n_joints
qdot = [None]*n_joints
qddot = [None]*n_joints
gravity_u2c = [0, 0, -9.81]
id_sym = gantry.get_inverse_dynamics_rnea(root, tip, gravity_u2c)

#rbdl
q_np = np.zeros(n_joints)
qdot_np = np.zeros(n_joints)
qddot_np = np.zeros(n_joints)
id_rbdl = np.zeros(n_joints)


error_rbdl_pb = np.zeros(n_joints)
error_rbdl_u2c = np.zeros(n_joints)
error_pb_u2c = np.zeros(n_joints)

def u2c2np(asd):
    return cs.Function("temp",[],[asd])()["o0"].toarray()

def list2np(asd):
    return np.asarray(asd)

n_itr = 1000
for i in range(n_itr):
    for j in range(n_joints):
        q[j] = (q_max[j] - q_min[j])*np.random.rand()-(q_max[j] - q_min[j])/2
        qdot[j] = (q_max[j] - q_min[j])*np.random.rand()-(q_max[j] - q_min[j])/2
        qddot[j] = (q_max[j] - q_min[j])*np.random.rand()-(q_max[j] - q_min[j])/2

        q_np[j] = q[j]
        qdot_np[j] = qdot[j]
        qddot_np[j] = qddot[j]


    rbdl.InverseDynamics(gantry_rbdl, q_np, qdot_np, qddot_np, id_rbdl)
    id_pb = pb.calculateInverseDynamics(gantry_pb, q, qdot, qddot)
    id_u2c = id_sym(q, qdot, qddot)

    for tau_idx in range(n_joints):
        error_pb_u2c[tau_idx] += np.absolute(list2np(id_pb[tau_idx]) - u2c2np(id_u2c[tau_idx]))
        error_rbdl_pb[tau_idx] += np.absolute(id_rbdl[tau_idx] - list2np(id_pb[tau_idx]))
        error_rbdl_u2c[tau_idx] += np.absolute(id_rbdl[tau_idx] - u2c2np(id_u2c)[tau_idx])


sum_error_rbdl_pb = 0
sum_error_rbdl_u2c = 0
sum_error_pb_u2c = 0

for err in range(n_joints):
    sum_error_rbdl_u2c += error_rbdl_u2c[err]
    sum_error_rbdl_pb += error_rbdl_pb[err]
    sum_error_pb_u2c += error_pb_u2c[err]

print "\nSum of errors pybullet vs. U2C for", n_itr, "iterations:\n", sum_error_pb_u2c
print "\nSum of errors RBDL vs. U2C for", n_itr, "iterations:\n",sum_error_rbdl_u2c
print "\nSum of errors pybullet vs. RBDL for", n_itr, "iterations:\n", sum_error_rbdl_pb
