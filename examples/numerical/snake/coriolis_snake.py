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


root = 'base_link'
tip = 'link16'
path_to_urdf = '../../urdf/snake_robot.urdf'

#get robot models

#kdl
ok, snake_tree = kdlurdf.treeFromFile(path_to_urdf)
snake_chain = snake_tree.getChain(root,tip)

#rbdl
snake_rbdl = rbdl.loadModel(path_to_urdf)

#u2c
snake = u2c.URDFparser()
snake.from_file(path_to_urdf)

#pybullet
sim = pb.connect(pb.DIRECT)
snake_pb = pb.loadURDF(path_to_urdf, useFixedBase=True, flags = pb.URDF_USE_INERTIA_FROM_FILE)

#joint info
jointlist, names, q_max, q_min = snake.get_joint_info(root, tip)
n_joints = snake.get_n_joints(root, tip)


#declarations

#kdl
q_kdl = kdl.JntArray(n_joints)
qdot_kdl = kdl.JntArray(n_joints)
gravity_kdl = kdl.Vector()
gravity_kdl[2] = -9.81
C_kdl = kdl.JntArray(n_joints)
g_kdl = kdl.JntArray(n_joints)

#u2c & pybullet
q = [None]*n_joints
qdot = [None]*n_joints
zeros_pb = [None]*n_joints
gravity_u2c = [0, 0, -9.81]
g_sym = snake.get_gravity_rnea(root, tip, gravity_u2c)
C_sym = snake.get_coriolis_rnea(root, tip)


#rbdlq_kdl = kdl.JntArray(n_joints)
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

        qdot[j] = (q_max[j] - q_min[j])*np.random.rand()-(q_max[j] - q_min[j])/2
        qdot_kdl[j] = qdot[j]
        qdot_np[j] = qdot[j]

        zeros_pb[j] = 0.


    rbdl.NonlinearEffects(snake_rbdl, q_np, qdot_np, C_rbdl)
    kdl.ChainDynParam(snake_chain, gravity_kdl).JntToGravity(q_kdl, g_kdl)
    kdl.ChainDynParam(snake_chain, gravity_kdl).JntToCoriolis(q_kdl, qdot_kdl, C_kdl)
    pb.setGravity(0, 0, 0)
    C_pb = pb.calculateInverseDynamics(snake_pb, q, qdot, zeros_pb)
    pb.setGravity(0, 0, -9.81)
    g_pb = pb.calculateInverseDynamics(snake_pb, q, zeros_pb, zeros_pb)


    g_u2c = g_sym(q)
    C_u2c = C_sym(q, qdot)

    for tau_idx in range(n_joints):
        error_kdl_rbdl[tau_idx] += np.absolute(((list2np(g_kdl[tau_idx]) + list2np(C_kdl[tau_idx])) - C_rbdl[tau_idx]))
        error_kdl_u2c[tau_idx] += np.absolute((list2np(C_kdl[tau_idx]) - u2c2np(C_u2c[tau_idx])))
        error_rbdl_u2c[tau_idx] += np.absolute(((u2c2np(g_u2c[tau_idx]) + u2c2np(C_u2c)[tau_idx]) - C_rbdl[tau_idx]))
        error_pb_u2c[tau_idx] += np.absolute((u2c2np(C_u2c[tau_idx]) - list2np(C_pb[tau_idx])))
        error_pb_kdl[tau_idx] += np.absolute((list2np(C_kdl[tau_idx]) - list2np(C_pb[tau_idx])))
        error_pb_rbdl[tau_idx] += np.absolute(C_rbdl[tau_idx] - (list2np(C_pb[tau_idx]) + list2np(g_pb[tau_idx])))


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
