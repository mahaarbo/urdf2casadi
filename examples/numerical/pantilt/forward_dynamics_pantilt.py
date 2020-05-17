import rbdl
import numpy as np
import casadi as cs
from urdf_parser_py.urdf import URDF, Pose
import os
import urdf2casadi.urdfparser as u2c

path_to_urdf = "../../urdf/pantilt.urdf"
root = "base_link"
tip = "tilt_link"


snake = u2c.URDFparser()
snake.from_file(path_to_urdf)
snake_rbdl = rbdl.loadModel(path_to_urdf)


jointlist, names, q_max, q_min = snake.get_joint_info(root, tip)
n_joints = snake.get_n_joints(root, tip)

q = np.zeros(n_joints)
qdot = np.zeros(n_joints)
tau = np.zeros(n_joints)

fd_rbdl_aba = np.zeros(n_joints)
fd_rbdl_crba = np.zeros(n_joints)

gravity = [0., 0., -9.81]

fd_sym_aba = snake.get_forward_dynamics_aba(root, tip, gravity = gravity)
fd_sym_crba = snake.get_forward_dynamics_crba(root, tip, gravity = gravity)

error_rbdl_u2c_crba = np.zeros(n_joints)
error_rbdl_u2c_aba = np.zeros(n_joints)
error_rbdl_crba_aba = np.zeros(n_joints)
error_u2c_crba_aba = np.zeros(n_joints)
error_u2c_crba_rbdl_aba = np.zeros(n_joints)

def u2c2np(asd):
    return cs.Function("temp",[],[asd])()["o0"].toarray()

n_itr = 1000
for i in range(n_itr):
    for j in range(n_joints):
        q[j] = (q_max[j] - q_min[j])*np.random.rand()-(q_max[j] - q_min[j])/2
        qdot[j] = (q_max[j] - q_min[j])*np.random.rand()-(q_max[j] - q_min[j])/2
        tau[j] = (q_max[j] - q_min[j])*np.random.rand()-(q_max[j] - q_min[j])/2

    fd_u2c_crba = fd_sym_crba(q, qdot, tau)
    fd_u2c_aba = fd_sym_aba(q, qdot, tau)
    rbdl.ForwardDynamics(snake_rbdl, q, qdot, tau, fd_rbdl_aba)

    #rbdl.ForwardDynamicsLagrangian(snake_rbdl, q, qdot, tau, fd_rbdl_crba)

    for qddot_idx in range(n_joints):
        error_rbdl_u2c_aba[qddot_idx] += np.absolute(u2c2np(fd_u2c_aba[qddot_idx]) - fd_rbdl_aba[qddot_idx])
        #error_rbdl_u2c_crba[qddot_idx] += np.absolute(u2c2np(fd_u2c_crba[qddot_idx]) - fd_rbdl_crba[qddot_idx])
        error_u2c_crba_aba[qddot_idx] += np.absolute(u2c2np(fd_u2c_aba[qddot_idx]) - u2c2np(fd_u2c_crba[qddot_idx]))
        #error_rbdl_crba_aba[qddot_idx] += np.absolute(fd_rbdl_crba[qddot_idx] - fd_rbdl_aba[qddot_idx])
        error_u2c_crba_rbdl_aba[qddot_idx] += np.absolute(u2c2np(fd_u2c_crba[qddot_idx]) - fd_rbdl_aba[qddot_idx])


sum_error_rbdl_u2c_crba = 0
sum_error_rbdl_u2c_aba = 0
sum_error_rbdl_crba_aba = 0
sum_error_u2c_crba_aba = 0
sum_error_u2c_crba_rbdl_aba = 0

for err in range(n_joints):
    sum_error_rbdl_u2c_crba += error_rbdl_u2c_crba[err]
    sum_error_rbdl_u2c_aba += error_rbdl_u2c_aba[err]
    sum_error_rbdl_crba_aba += error_rbdl_crba_aba[err]
    sum_error_u2c_crba_aba += error_u2c_crba_aba[err]
    sum_error_u2c_crba_rbdl_aba += error_u2c_crba_rbdl_aba[err]

print "\nSum of errors RBDL vs. U2c using ABA for", n_itr, "iterations:\n", sum_error_rbdl_u2c_aba
print "\nSum of errors RBDL vs. U2C using CRBA for", n_itr, "iterations:\n", sum_error_rbdl_u2c_crba
print "\nSum of errors U2C ABA vs. CRBA for", n_itr, "iterations:\n",sum_error_u2c_crba_aba
print "\nSum of errors RBDL ABA vs. CRBA for", n_itr, "iterations:\n", sum_error_rbdl_crba_aba
print "\nSum of errors RBDL ABA vs. U2C CRBA for", n_itr, "iterations:\n", sum_error_u2c_crba_rbdl_aba
