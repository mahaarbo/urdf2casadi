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


path_to_urdf = "../../urdf/gantry.urdf"
root = "gantry_link_base"
tip = "gantry_tool0"
#kdl
ok, gantry_tree = kdlurdf.treeFromFile(path_to_urdf)
gantry_chain = gantry_tree.getChain(root,tip)

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


#declarations

#kdl
q_kdl = kdl.JntArray(n_joints)
gravity_kdl = kdl.Vector()
gravity_kdl[2] = -9.81
g_kdl = kdl.JntArray(n_joints)
M_kdl = kdl.JntSpaceInertiaMatrix(n_joints)

#u2c & pybullet
q = [None]*n_joints
qdot = [None]*n_joints
zeros_pb = [None]*n_joints
M_sym = gantry.get_inertia_matrix_crba(root, tip)

#rbdl
q_np = np.zeros(n_joints)
M_rbdl = (n_joints, n_joints)
M_rbdl = np.zeros(M_rbdl)

#error declarations
error_kdl_rbdl = np.zeros((n_joints, n_joints))
error_kdl_u2c = np.zeros((n_joints, n_joints))
error_rbdl_u2c = np.zeros((n_joints, n_joints))
error_pb_u2c = np.zeros((n_joints, n_joints))
error_pb_kdl = np.zeros((n_joints, n_joints))
error_pb_rbdl = np.zeros((n_joints, n_joints))


def u2c2np(asd):
    return cs.Function("temp",[],[asd])()["o0"].toarray()

def list2np(asd):
    return np.asarray(asd)


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


    rbdl.CompositeRigidBodyAlgorithm(gantry_rbdl, q_np, M_rbdl)
    kdl.ChainDynParam(gantry_chain, gravity_kdl).JntToMass(q_kdl, M_kdl)
    M_pb = pb.calculateMassMatrix(gantry_pb, q)
    M_u2c = M_sym(q)

    for row_idx in range(n_joints):
        for col_idx in range(n_joints):
            error_kdl_u2c[row_idx][col_idx] += np.absolute(M_kdl[row_idx,col_idx] - u2c2np(M_u2c[row_idx, col_idx]))
            error_rbdl_u2c[row_idx][col_idx] += np.absolute((M_rbdl[row_idx,col_idx]) - u2c2np(M_u2c[row_idx, col_idx]))
            error_pb_u2c[row_idx][col_idx] += np.absolute(list2np(M_pb[row_idx][col_idx]) - u2c2np(M_u2c[row_idx, col_idx]))
            error_kdl_rbdl[row_idx][col_idx] += np.absolute((M_rbdl[row_idx,col_idx]) - list2np(M_kdl[row_idx, col_idx]))
            error_pb_kdl[row_idx][col_idx] += np.absolute(list2np(M_pb[row_idx][col_idx]) - list2np(M_kdl[row_idx, col_idx]))
            error_pb_rbdl[row_idx][col_idx] += np.absolute((M_rbdl[row_idx,col_idx]) - list2np(M_pb[row_idx][col_idx]))



sum_error_kdl_rbdl = 0
sum_error_kdl_u2c = 0
sum_error_rbdl_u2c = 0
sum_error_pb_u2c = 0
sum_error_pb_kdl = 0
sum_error_pb_rbdl = 0


for row in range(n_joints):
    for col in range(n_joints):
        sum_error_kdl_rbdl += error_kdl_rbdl[row][col]
        sum_error_kdl_u2c += error_kdl_u2c[row][col]
        sum_error_rbdl_u2c += error_rbdl_u2c[row][col]
        sum_error_pb_u2c += error_pb_u2c[row][col]
        sum_error_pb_kdl += error_pb_kdl[row][col]
        sum_error_pb_rbdl += error_pb_rbdl[row][col]

print("\nSum of errors KDL vs. RBDL for", n_itr, "iterations:\n", sum_error_kdl_rbdl)
print("\nSum of errors KDL vs. U2C for", n_itr, "iterations:\n", sum_error_kdl_u2c)
print("\nSum of errors RBDL vs. U2C for", n_itr, "iterations:\n",sum_error_rbdl_u2c)
print("\nSum of errors pybullet vs. U2C for", n_itr, "iterations:\n", sum_error_pb_u2c)
print("\nSum of errors pybullet vs. KDL for", n_itr, "iterations:\n", sum_error_pb_kdl)
print("\nSum of errors pybullet vs. RBDL for", n_itr, "iterations:\n", sum_error_pb_rbdl)
