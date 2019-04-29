import casadi as cs
from urdf_parser_py.urdf import URDF, Pose
import os
import urdf2casadi.urdfparser as u2c
import numpy as np
import PyKDL as kdl
import kdl_parser.kdl_parser_py.kdl_parser_py.urdf as kdlurdf


root = 'gantry_root'
tip = 'gantry_tool0'

ok, ur_tree = kdlurdf.treeFromFile('/home/lillmaria/urdf2casadi/examples/urdf/thrivaldi.urdf')
gantry_chain = ur_tree.getChain(root,tip)

gantry = u2c.URDFparser()
gantry.from_file("/home/lillmaria/urdf2casadi/examples/urdf/thrivaldi.urdf")


jointlist, names, q_max, q_min = gantry.get_joint_info(root, tip)
n_joints = gantry.get_n_joints(root, tip)

grav = kdl.Vector()
q_kdl = kdl.JntArray(n_joints)
q = [None]*n_joints
M_sym = gantry.get_inertia_matrix_crba(root, tip)
error = np.zeros((n_joints, n_joints))
M_kdl = kdl.JntSpaceInertiaMatrix(n_joints)

def u2c2np(asd):
    return cs.Function("temp",[],[asd])()["o0"].toarray()

def kdl2np(asd):
    return np.asarray(asd)

n_itr = 1000
for i in range(n_itr):
    for j in range(n_joints):
        q[j] = (q_max[j] - q_min[j])*np.random.rand()-(q_max[j] - q_min[j])/2
        q_kdl[j] = q[j]

    kdl.ChainDynParam(gantry_chain, grav).JntToMass(q_kdl, M_kdl)
    M_u2c = M_sym(q)

    for row_idx in range(n_joints):
        for col_idx in range(n_joints):
            error[row_idx][col_idx] += np.absolute((kdl2np(M_kdl[row_idx,col_idx])) - u2c2np(M_u2c[row_idx, col_idx]))

print "Errors in inertia matrix with", n_itr, "iterations and comparing against KDL:\n", error

sum_error = 0
for row in range(n_joints):
    for col in range(n_joints):
        sum_error += error[row][col]
print "Sum of errors:\n", sum_error
