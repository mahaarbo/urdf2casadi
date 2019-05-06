import casadi as cs
from urdf_parser_py.urdf import URDF, Pose
import os # For current directory
import urdf2casadi.urdfparser as u2c
import numpy as np
import PyKDL as kdl
import kdl_parser_py.urdf as kdlurdf

root = 'base_link'
tip = 'fl_caster_r_wheel_link'

ok, ur_tree = kdlurdf.treeFromFile('/home/lmjohann/urdf2casadi/examples/urdf/pr2.urdf')
pr2_chain = ur_tree.getChain(root,tip)

pr2 = u2c.URDFparser()
robot_desc = pr2.from_file('/home/lmjohann/urdf2casadi/examples/urdf/pr2.urdf')


jointlist, names, q_max, q_min = pr2.get_joint_info(root, tip)
n_joints = pr2.get_n_joints(root, tip)

q_kdl = kdl.JntArray(n_joints)
qdot_kdl = kdl.JntArray(n_joints)
grav = kdl.Vector()

q = [None]*n_joints
qdot = [None]*n_joints

C_kdl = kdl.JntArray(n_joints)
C_sym = pr2.get_coriolis_rnea(root, tip)

error = np.zeros(n_joints)

def u2c2np(asd):
    return cs.Function("temp",[],[asd])()["o0"].toarray()

def kdl2np(asd):
    x = []
    for i in range(n_joints):
        x.append(asd[i])
    return np.asarray(x)

n_itr = 10000
for i in range(n_itr):
    for j in range(n_joints):
        q[j] = (q_max[j] - q_min[j])*np.random.rand()-(q_max[j] - q_min[j])/2
        qdot[j] = (q_max[j] - q_min[j])*np.random.rand()-(q_max[j] - q_min[j])/2
        q_kdl[j] = q[j]
        qdot_kdl[j] = qdot[j]

    kdl.ChainDynParam(pr2_chain, grav).JntToCoriolis(q_kdl, qdot_kdl, C_kdl)
    C_u2c = C_sym(q,qdot)

    for tau_idx in range(n_joints):
        error[tau_idx] += np.absolute((kdl2np(C_kdl)[tau_idx] - u2c2np(C_u2c)[tau_idx]))

print "Errors in coriolis forces with",n_itr, "iterations and comparing against KDL:\n", error

sum_error = 0
for err in range(n_joints):
    sum_error += error[err]
print "Sum of errors:\n", sum_error
