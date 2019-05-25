import casadi as cs
import rbdl
from urdf_parser_py.urdf import URDF, Pose
import os # For current directory
import urdf2casadi.urdfparser as u2c
from urdf2casadi.geometry import plucker
import numpy as np
import PyKDL as kdl
import kdl_parser_py.urdf as kdlurdf
import pybullet as pb

#root = 'calib_kuka_arm_base_link'
#tip = "kuka_arm_7_link"

root = 'base_link'
tip = 'tool0'
path_to_urdf = '/home/lmjohann/urdf2casadi/examples/urdf/ur5_mod.urdf'

#get robot models

#kdl
ok, ur_tree = kdlurdf.treeFromFile(path_to_urdf)
ur_chain = ur_tree.getChain(root,tip)

#rbdl
urmodel = rbdl.loadModel(path_to_urdf)

#u2c
ur5 = u2c.URDFparser()
ur5.from_file(path_to_urdf)

#pybullet
sim = pb.connect(pb.DIRECT)
pbmodel = pb.loadURDF(path_to_urdf, useFixedBase=True, flags = pb.URDF_USE_INERTIA_FROM_FILE)
pb.setGravity(0, 0, -9.81)

#joint info
jointlist, names, q_max, q_min = ur5.get_joint_info(root, tip)
n_joints = ur5.get_n_joints(root, tip)

q_kdl = kdl.JntArray(n_joints)
#declarations
q_kdl = kdl.JntArray(n_joints)
#kdl
q_kdl = kdl.JntArray(n_joints)
gravity_kdl = kdl.Vector()
gravity_kdl[2] = -9.81
g_kdl = kdl.JntArray(n_joints)

#u2c & pybullet
q = [None]*n_joints
qdot = [None]*n_joints
zeros_pb = [None]*n_joints
gravity_u2c = [0, 0, -9.81]
g_sym = ur5.get_gravity_rnea(root, tip, gravity_u2c)

#rbdl
q_np = np.zeros(n_joints)
qdot_np = np.zeros(n_joints)
qddot_np = np.zeros(n_joints)
g_rbdl = np.zeros(n_joints)
C_rbdl = np.zeros(n_joints)
zeros_rbdl = np.zeros(n_joints)



error_kdl_rbdl = np.zeros(n_joints)
error_kdl_u2c = np.zeros(n_joints)
error_rbdl_u2c = np.zeros(n_joints)
error_pb_u2c = np.zeros(n_joints)
error_pb_kdl = np.zeros(n_joints)
error_pb_rbdl = np.zeros(n_joints)


def u2c2np(asd):
    return cs.Function("temp",[],[asd])()["o0"].toarray()

def list2np(asd):
    return np.asarray(asd)

n_itr = 1000
for i in range(n_itr):
    for j in range(n_joints):
        q[j] = (q_max[j] - q_min[j])*np.random.rand()-(q_max[j] - q_min[j])/2
        q_kdl[j] = q[j]
        q_np[j] = q[j]
        zeros_pb[j] = 0.

    rbdl.InverseDynamics(urmodel, q_np, zeros_rbdl, zeros_rbdl, g_rbdl)
    kdl.ChainDynParam(ur_chain, gravity_kdl).JntToGravity(q_kdl, g_kdl)
    g_pb = pb.calculateInverseDynamics(pbmodel, q, zeros_pb, zeros_pb)
    g_u2c = g_sym(q)
    #print g_u2c

    for tau_idx in range(n_joints):
        error_kdl_rbdl[tau_idx] += np.absolute((list2np(g_kdl[tau_idx]) - g_rbdl[tau_idx]))
        error_kdl_u2c[tau_idx] += np.absolute((list2np(g_kdl[tau_idx]) - u2c2np(g_u2c[tau_idx])))
        error_rbdl_u2c[tau_idx] += np.absolute((u2c2np(g_u2c[tau_idx]) - g_rbdl[tau_idx]))
        error_pb_u2c[tau_idx] += np.absolute((u2c2np(g_u2c[tau_idx]) - list2np(g_pb[tau_idx])))
        error_pb_kdl[tau_idx] += np.absolute((list2np(g_kdl[tau_idx]) - list2np(g_pb[tau_idx])))
        error_pb_rbdl[tau_idx] += np.absolute(g_rbdl[tau_idx] - list2np(g_pb[tau_idx]))


sum_error_kdl_rbdl = 0
sum_error_kdl_u2c = 0
sum_error_rbdl_u2c = 0
sum_error_pb_u2c = 0
sum_error_pb_kdl = 0
sum_error_pb_rbdl = 0

for err in range(n_joints):
    sum_error_kdl_rbdl += error_kdl_rbdl[err]
    sum_error_kdl_u2c += error_kdl_u2c[err]
    sum_error_rbdl_u2c += error_rbdl_u2c[err]
    sum_error_pb_u2c += error_pb_u2c[err]
    sum_error_pb_kdl += error_pb_kdl[err]
    sum_error_pb_rbdl += error_pb_rbdl[err]

print "\nSum of errors KDL vs. RBDL for", n_itr, "iterations:\n", sum_error_kdl_rbdl
print "\nSum of errors KDL vs. U2C for", n_itr, "iterations:\n", sum_error_kdl_u2c
print "\nSum of errors RBDL vs. U2C for", n_itr, "iterations:\n",sum_error_rbdl_u2c
print "\nSum of errors pybullet vs. U2C for", n_itr, "iterations:\n", sum_error_pb_u2c
print "\nSum of errors pybullet vs. KDL for", n_itr, "iterations:\n",sum_error_pb_kdl
print "\nSum of errors pybullet vs. RBDL for", n_itr, "iterations:\n", sum_error_pb_rbdl
