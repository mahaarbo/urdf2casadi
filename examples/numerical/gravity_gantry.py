import casadi as cs
from urdf_parser_py.urdf import URDF, Pose
import os # For current directory
import urdf2casadi.urdfparser as u2c
import numpy as np
import PyKDL as kdl
import kdl_parser.kdl_parser_py.kdl_parser_py.urdf as kdlurdf


root = 'gantry_root'
tip = 'gantry_tool0'
ok, ur_tree = kdlurdf.treeFromFile('/home/lillmaria/urdf2casadi/examples/urdf/thrivaldi.urdf')
gantry_chain = ur_tree.getChain(root,tip)
asd = u2c.URDFparser()
robot_desc = asd.from_file("/home/lillmaria/urdf2casadi/examples/urdf/thrivaldi.urdf")



jointlist, names, q_max, q_min = asd.get_joint_info(root, tip)
n_joints = asd.get_n_joints(root, tip)
q_kdl = kdl.JntArray(n_joints)
q = [None]*n_joints
gravity_kdl = kdl.Vector()
gravity_kdl[2] = -9.81
gravity_u2c = [0., 0., -9.81]
G_kdl = kdl.JntArray(n_joints)
G_sym = asd.get_gravity_rnea(root, tip, gravity_u2c)
error = np.zeros(n_joints)

def u2c2np(asd):
    return cs.Function("temp",[],[asd])()["o0"].toarray()

def kdl2np(asd):
    x = []
    for i in range(n_joints):
        x.append(asd[i])
    return np.asarray(x)


n_itr = 1000
for i in range(n_itr):
    for j in range(n_joints):
        q[j] = (q_max[j] - q_min[j])*np.random.rand()-(q_max[j] - q_min[j])/2
        q_kdl[j] = q[j]

    kdl.ChainDynParam(gantry_chain, gravity_kdl).JntToGravity(q_kdl, G_kdl)
    G_u2c = G_sym(q)

    for tau_idx in range(n_joints):
        error[tau_idx] += np.absolute((kdl2np(G_kdl)[tau_idx] - u2c2np(G_u2c)[tau_idx]))

print "Errors in coriolis forces with",n_itr, "iterations and comparing against KDL:\n", error

sum_error = 0
for err in range(n_joints):
    sum_error += error[err]
print "Sum of errors:\n", sum_error
