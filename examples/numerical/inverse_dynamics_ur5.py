import rbdl
import numpy as np
import casadi as cs
from urdf_parser_py.urdf import URDF, Pose
import os
import urdf2casadi.urdf2casadi.urdfparser as u2c

urmodel = rbdl.loadModel("/home/lillmaria/urdf2casadi/examples/urdf/ur5_rbdl.urdf")
asd = u2c.URDFparser()
robot_desc = asd.from_file("/home/lillmaria/urdf2casadi/examples/urdf/ur5_rbdl.urdf")
root = "base_link"
tip = "wrist_3_link"


jointlist, names, q_max, q_min = asd.get_joint_info(root, tip)
n_joints = asd.get_n_joints(root, tip)


q = np.zeros(n_joints)
qdot = np.zeros(n_joints)
qddot = np.zeros(n_joints)
tau_rbdl = np.zeros(n_joints)

gravity = [0., 0., -9.81]
error = np.zeros(n_joints)

def u2c2np(asd):
    return cs.Function("temp",[],[asd])()["o0"].toarray()

n_itr = 100
for i in range(n_itr):
    for j in range(n_joints):
        q[j] = (q_max[j] - q_min[j])*np.random.rand()-(q_max[j] - q_min[j])/2
        qdot[j] = (q_max[j] - q_min[j])*np.random.rand()-(q_max[j] - q_min[j])/2
        #Har tau noen restrictions?
        qddot[j] = 0

    rbdl.InverseDynamics(urmodel, q, qdot, qddot, tau_rbdl)
    #print qddot_rbdl

    #Skal tau legges inn symbolsk eller ok med den numerisk?
    tau_sym = asd.get_inverse_dynamics_RNEA(root, tip, gravity)
    tau_u2c = tau_sym(q, qdot, qddot)
    #print qddot_u2c

    for tau_idx in range(n_joints):
        error[tau_idx] += np.absolute(tau_rbdl[tau_idx] - u2c2np(tau_u2c)[tau_idx])

print "Errors in inverse dynamics forces with",n_itr, "iterations and comparing against rbdl:\n", error

sum_error = 0
for err in range(n_joints):
    sum_error += error[err]
print "Sum of errors:\n", sum_error
