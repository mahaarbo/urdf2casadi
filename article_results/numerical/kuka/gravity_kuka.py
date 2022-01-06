import casadi as cs
from urdf_parser_py.urdf import URDF, Pose
import os # For current directory
import urdf2casadi.urdfparser as u2c
import numpy as np
import PyKDL as kdl
import kdl_parser_py.urdf as kdlurdf

root = "calib_kuka_arm_base_link"
tip = "kuka_arm_7_link"

ok, ur_tree = kdlurdf.treeFromFile('../../urdf/kuka.urdf')
kuka_chain = ur_tree.getChain(root,tip)

kuka = u2c.URDFparser()
kuka.from_file("../../urdf/kuka.urdf")


jointlist, names, q_max, q_min = kuka.get_joint_info(root, tip)
n_joints = kuka.get_n_joints(root, tip)

q_kdl = kdl.JntArray(n_joints)
q = [None]*n_joints

gravity_kdl = kdl.Vector()
gravity_kdl[2] = -9.81
gravity_u2c = [0., 0., -9.81]

G_kdl = kdl.JntArray(n_joints)
G_sym = kuka.get_gravity_rnea(root, tip, gravity_u2c)

error = np.zeros(n_joints)


def u2c2np(asd):
    return cs.Function("temp", [], [asd])()["o0"].toarray()

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

    kdl.ChainDynParam(kuka_chain, gravity_kdl).JntToGravity(q_kdl, G_kdl)
    G_u2c = G_sym(q)
    for tau_idx in range(n_joints):
        error[tau_idx] += np.absolute((kdl2np(G_kdl)[tau_idx] - u2c2np(G_u2c)[tau_idx]))

print("Errors in gravity forces with",n_itr, "iterations and comparing against KDL:\n", error)

sum_error = 0
for err in range(n_joints):
    sum_error += error[err]
print("Sum of errors:\n", sum_error)
