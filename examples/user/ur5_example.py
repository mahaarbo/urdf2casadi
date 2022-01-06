
# coding: utf-8

# In[1]:


import urdf2casadi.urdfparser as u2c
import numpy as np


# ## 1. Load robot from urdf 
# 
# 1. Create urdfparser-class instance. 
# 2. Load model to instance, either from file, string or ros parameter server. Examples uses from file. 

# In[2]:


ur5 = u2c.URDFparser()
import os
path_to_urdf = absPath = os.path.dirname(os.path.abspath(__file__)) + '/../urdf/ur5_mod.urdf' 
ur5.from_file(path_to_urdf)


# ## 2. Get joint information ? 
# Information about the joints of the robot model can be obtained using u2c's get_joint_info(). "root" and "tip" are the inputs, and represent the link names for the root and tip of the kinematic tree one wishes to evaluate. 

# In[3]:


root = "base_link"
tip = "tool0"

joint_list, joint_names, q_max, q_min = ur5.get_joint_info(root, tip)
n_joints = ur5.get_n_joints(root, tip)
print("name of first joint:", joint_names[0], "\n")
print("joint information for first joint:\n", joint_list[0])
print("\n q max:", q_max)
print("\n q min:", q_min)


# # 3. Obtain robot dynamics! 
# After loading the robot model to the urdfparser-instance, the robot dynamics can be obtained from any given root and tip of the robot. 
# 
# ## Dynamic parameters:
# To obtain the dynamic parameters (M, C, G) of the equation of motion $\tau = M(q)\ddot{q} + C(q, \dot{q}) + G(q)$ we use the recursive Newton-Euler algorithm has defined by Roy Featherstone in "Rigid Body Dynamics Algorithms" (2008). The algorithm calculates $\tau = M(q)\ddot{q} + C_2(q,\dot{q})$ where $C_2$ is a bias term containing both the coriolis and gravitational effect. From this we can obtain:
#  
#  1. Coriolis vector term $C(q,\dot{q})\dot{q}$ by setting $\ddot{q}=0$ and gravity to zero.
#  2. Gravity term $G(q)$ by setting $\ddot{q}=0$ and $\dot{q}=0$
# 
# The inertia matrix $M$ is obtained using the composite rigid body algorithm.
# 

# In[4]:


M_sym = ur5.get_inertia_matrix_crba(root, tip)
C_sym = ur5.get_coriolis_rnea(root, tip)

gravity = [0, 0, -9.81]
G_sym = ur5.get_gravity_rnea(root, tip, gravity)


# M_sym, C_sym, G_sym are CasADi symbolic expressions of the ur5's dynamic parameters from given root to tip. The CasADi expressions are C-code generated so they can be numerically evaluated efficiently. Example of numerical evaluation is shown below. 

# In[5]:


q = [None]*n_joints
q_dot = [None]*n_joints
for i in range(n_joints):
    #to make sure the inputs are within the robot's limits:
    q[i] = (q_max[i] - q_min[i])*np.random.rand()-(q_max[i] - q_min[i])/2
    q_dot[i] = (q_max[i] - q_min[i])*np.random.rand()-(q_max[i] - q_min[i])/2
    
M_num = M_sym(q)
C_num = C_sym(q, q_dot)
G_num = G_sym(q)
print("Numerical Inertia Matrx for random input: \n", M_num)
print("\nNumerical Coriolis term for random input: \n", C_num) 
print("\nNumerical gravity term for random input: \n", G_num)


# ## Inverse Dynamics 
# 
# Without accounting for gravitational forces:

# In[6]:


tau_sym = ur5.get_inverse_dynamics_rnea(root, tip)


# Accounting for gravitational forces:

# In[7]:


gravity = [0, 0, -9.81]
tau_g_sym = ur5.get_inverse_dynamics_rnea(root, tip, gravity = gravity)


# External forces can also be accounted for:

# In[8]:


q = [None]*n_joints
q_dot = [None]*n_joints
q_ddot = [None]*n_joints
for i in range(n_joints):
    #to make sure the inputs are within the robot's limits:
    q[i] = (q_max[i] - q_min[i])*np.random.rand()-(q_max[i] - q_min[i])/2
    q_dot[i] = (q_max[i] - q_min[i])*np.random.rand()-(q_max[i] - q_min[i])/2
    q_ddot[i] = (q_max[i] - q_min[i])*np.random.rand()-(q_max[i] - q_min[i])/2

tau_num = tau_sym(q, q_dot, q_ddot)
tau_g_num = tau_g_sym(q, q_dot, q_ddot)
#tau_fext_num = tau_fext_sym(q, q_dot, q_ddot)
print("Numerical inverse dynamics: \n", tau_num)
print("\nNumerical inverse dynamics w/ gravity: \n", tau_g_num)
#print "\nNumerical inverse dynamics w/ external forces: \n", G_num


# ## Forward Dynamics
# 
# urdf2casadi provides two methods for finding the robot's forward dynamics. The first method combines the recursive Newton-Euler algorithm (RNEA) and the composite rigid body algorithm (CRBA) and solves the equation of motion for the joint accelerations. The second method uses the articulated body algorithm (ABA) for forward dynamics. The method that uses ABA is in most cases the most efficient with regard to numerical evaluation, especially if the number of joints are high. (See timing examples for more information.)

# In[9]:


tau = np.zeros(n_joints)
qddot_sym = ur5.get_forward_dynamics_crba(root, tip)


# In[10]:


qddot_g_sym = ur5.get_forward_dynamics_aba(root, tip, gravity = gravity)


# In[11]:


q = [None]*n_joints
q_dot = [None]*n_joints
for i in range(n_joints):
    #to make sure the inputs are within the robot's limits:
    q[i] = (q_max[i] - q_min[i])*np.random.rand()-(q_max[i] - q_min[i])/2
    q_dot[i] = (q_max[i] - q_min[i])*np.random.rand()-(q_max[i] - q_min[i])/2

qddot_num = qddot_sym(q, q_dot, tau)
qddot_g_num = qddot_g_sym(q, q_dot, tau)

print("Numerical inverse dynamics: \n", qddot_num)
print("\nNumerical inverse dynamics w/ gravity: \n", qddot_g_num)


# # 4. Obtain the Derivatives 
# 
# From the dynamics functions, their derivatives can easily be obtained using CasADi`s built-in Jacobian functionality. The user can choose to find the derivative with regard to those variables needed (cs.jacobian()), or to find the time derivative with regard to these variables (cs.jtimes()), i.e the Jacobian times the time derivative of the variables.
# 
# If one are to find the time derivative, jtimes() is recommended over first obtaining the Jacobian, as jtimes() shortens the expressions, thus making the evaluation time of the expressions more efficient. 
# 
# Examples for obtaining the derivatives for the inverse dynamics, using both jacobian() and jtimes() are shown below:
# 
# ## cs.jacobian()
# 
# The following explains how to use CasADi to obtain the derivative of the inverse dynamics with respect to q, qdot, and qddot, using the symbolic function returned by urdf2casadi (tau_sym):
# 
# 
# 1. Import CasADi and declare the symbolic variables needed in the derivative expression. 

# In[12]:


import casadi as cs

q_sym =cs.SX.sym("qs", n_joints)
qdot_sym =cs.SX.sym("qsdot", n_joints)
qddot_sym =cs.SX.sym("qsddot", n_joints)


# 2. Declare the vector of the variables to find the derivatives with respect to using cs.vertcat():

# In[13]:


id_vec = cs.vertcat(q_sym, qdot_sym, qddot_sym)
print(id_vec)


# 3. Obtain the symbolic expression of the derivative of ID with respect to q, qdot, and qddot using cs.jacobian():

# In[14]:


derivative_id_expression = cs.jacobian(tau_sym(q_sym, qdot_sym, qddot_sym), id_vec)


# 4. Use the symbolic expression to make a CasADi function that can be efficiently numerical evaluated:

# In[15]:


derivative_id_function  = cs.Function("did", [q_sym, qdot_sym, qddot_sym], [derivative_id_expression], {"jit": True, "jit_options":{"flags":"-Ofast"}})


# where -Ofast flag is used to C-code generate the function. The derivative function can then be numerically evaluated similar to the functions returned by urdf2casadi, as illustrated in the above. For instance:

# In[16]:


print(derivative_id_function(np.ones(n_joints), np.ones(n_joints), np.ones(n_joints)))


# 
# One can also find the derivative with respect to just one variable:
# 

# In[17]:


derivative_id_expression_dq = cs.jacobian(tau_sym(q_sym, qdot_sym, qddot_sym), q_sym)
derivative_id_function_dq  = cs.Function("didq", [q_sym, qdot_sym, qddot_sym], [derivative_id_expression_dq], {"jit": True, "jit_options":{"flags":"-Ofast"}})

print(derivative_id_function_dq(np.ones(n_joints), np.ones(n_joints), np.ones(n_joints)))


# # cs.jtimes()
# 
# To obtain the time derivative with cs.jtimes, the same procedure as for cs.jacobian() is used with an additional variable, i.e the time derivatives of the varibales:
# 
# 1. Import casadi and declare the symbolic variables needed, also the time derivatives of these: 

# In[18]:


import casadi as cs

q_sym =cs.SX.sym("qs", n_joints)
qdot_sym =cs.SX.sym("qsdot", n_joints)
qddot_sym =cs.SX.sym("qsddot", n_joints)
qdddot_sym = cs.SX.sym("qsdddot", n_joints)


# 2. Declare the vector of the variables to find the derivatives with respect to, and the vector with their time derivatives:

# In[19]:


id_vec = cs.vertcat(q_sym, qdot_sym, qddot_sym)
id_dvec = cs.vertcat(qdot_sym, qddot_sym, qdddot_sym)


# 3. Obtain the symbolic expression of the time derivative of ID with respect to q, qdot, and qddot using cs.jtimes():

# In[20]:


timederivative_id_expression = cs.jtimes(tau_sym(q_sym, qdot_sym, qddot_sym), id_vec, id_dvec)


# 4. Use the symbolic expression to make a CasADi function that can be efficiently numerical evaluated:

# In[21]:


timederivative_id_function = cs.Function("didtimes", [q_sym, qdot_sym, qddot_sym, qdddot_sym], [timederivative_id_expression], {"jit": True, "jit_options":{"flags":"-Ofast"}})
print(timederivative_id_function(np.ones(n_joints), np.ones(n_joints), np.ones(n_joints), np.ones(n_joints)))


# And one can also just find the time derivative of the inverse dynamics with respect to, for instance, q:

# In[22]:


timederivative_id_expression_dq = cs.jtimes(tau_sym(q_sym, qdot_sym, qddot_sym), q_sym, qdot_sym)
timederivative_id_function_dq = cs.Function("dqidtimes", [q_sym, qdot_sym, qddot_sym], [timederivative_id_expression_dq], {"jit": True, "jit_options":{"flags":"-Ofast"}})
print(timederivative_id_function_dq(np.ones(n_joints), np.ones(n_joints), np.ones(n_joints)))

