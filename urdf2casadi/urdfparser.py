"""This module contains a class for turning a chain in a URDF to a
casadi function.
"""

#ToDo:
#1.Raise exeption for alle funcs
#2.Bytte ut uintuitive navn
#3.Lage flere pluckerfuncs, bl.a cross force & motion
#4. Flere fd-algos?

import casadi as cs
import numpy as np
from urdf_parser_py.urdf import URDF, Pose
from geometry import transformation_matrix as T
from geometry import plucker
from geometry import quaternion
from geometry import dual_quaternion

#Should we have more object variables so that these are common for whole object?

class URDFparser(object):
	"""Class that turns a chain from URDF to casadi functions"""
	actuated_types = ["prismatic", "revolute", "continuous"]

	def __init__(self):
		self.robot_desc = None

	#Load robot description from urdf methods
	def from_file(self, filename):
		"""Uses an URDF file to get robot description"""
		print filename
		self.robot_desc = URDF.from_xml_file(filename)
		#self.chain_list = robot_desc.get_chain(root, tip)

	def from_server(self, key="robot_description"):
		"""Uses a parameter server to get robot description"""
		self.robot_desc = URDF.from_parameter_server(key=key)
        #self.chain_list = robot_desc.get_chain(root, tip)

	def from_string(self, urdfstring):
		"""Uses a string to get robot description"""
		self.robot_desc = URDF.from_xml_string(urdfstring)


	#helper methods for other methods
	def _get_joint_info(self, root, tip):
		"""Using an URDF to extract a proper joint_list"""
		chain = self.robot_desc.get_chain(root, tip)
		if self.robot_desc is None:
			raise ValueError('Robot description not loaded from urdf')

		joint_list = []
		upper = []
		lower = []
		actuated_names = []
		chain = self.robot_desc.get_chain(root, tip)
		for item in chain:
			if item in self.robot_desc.joint_map:
				joint = self.robot_desc.joint_map[item]
				joint_list += [joint]
				if joint.type in self.actuated_types:
					actuated_names += [joint.name]
					if joint.type == "continuous":
						upper += [cs.inf]
						lower += [-cs.inf]
					else:
						upper += [joint.limit.upper]
						lower += [joint.limit.lower]
					if joint.axis is None:
						joint.axis = [1., 0., 0.]
					if joint.origin is None:
						joint.origin = Pose(xyz=[0., 0., 0.],
	                                        rpy=[0., 0., 0.])
					elif joint.origin.xyz is None:
						joint.origin.xyz = [0., 0., 0.]
					elif joint.origin.rpy is None:
						joint.origin.rpy = [0., 0., 0.]

		return joint_list, actuated_names, upper, lower

	def _get_spatial_inertias(self, root, tip):
		chain = self.robot_desc.get_chain(root, tip)
		#link_list = []
		spatial_inertias = []
		n_bodies = 0

		for item in chain:
			if item in self.robot_desc.link_map:
				link = self.robot_desc.link_map[item]
				n_bodies += 1
				if link.inertial is not None:
					I = link.inertial.inertia
					spatial_inertia = plucker.spatial_inertia_matrix(I.ixx, I.ixy, I.ixz, I.iyz, I.iyy, I.izz, link.inertial.mass)
					spatial_inertias.append(spatial_inertia)

		return spatial_inertias, n_bodies


	def _get_spatial_transforms_and_joint_space(self, q, joint_list):
		"""Helper function for RNEA which calculates spatial transform matrices and motion subspace matrices"""
		i_X_0 = []
		i_X_p = []
		joint_spaces = []
		joint_space = cs.SX.zeros(6,1)
		i = 0

		for joint in joint_list:
			XL = plucker.XL(joint.origin.xyz, joint.origin.rpy)
			if joint.type == "fixed":
				XJ = plucker.XL(joint.origin.xyz, joint.origin.rpy)

			elif joint.type == "prismatic":
				XJ = plucker.XJ_prismatic(joint.origin.xyz, joint.origin.rpy, joint.axis, q[i])

				if joint.axis.xyz[0] is 1:
					joint_space = cs.SX([1, 0, 0, 0, 0, 0])
				elif joint.axis.xyz[1] is 1:
					joint_space = cs.SX([0, 1, 0, 0, 0, 0])
				elif joint.axis.xyz[2] is 1:
					joint_space = cs.SX([0, 0, 1, 0, 0, 0])

			elif joint.type in ["revolute", "continuous"]:
				XJ = plucker.XJ_revolute(joint.origin.xyz, joint.origin.rpy, joint.axis, q[i])
				if joint.axis[0] is 1:
					joint_space = cs.SX([0, 0, 0, 1, 0, 0])
				elif joint.axis[1] is 1:
					joint_space = cs.SX([0, 0, 0, 0, 1, 0])
				elif joint.axis[2] is 1:
					joint_space = cs.SX([0, 0, 0, 0, 0, 1])

			i_X_p.append(cs.mtimes(XJ, XL))
			joint_spaces.append(joint_space)

			if(i == 0):
				i_X_0.append(i_X_p[i])

			else:
				i_X_0.append(cs.mtimes(i_X_p[i], i_X_0[i-1]))
			i += 1

		return i_X_p, i_X_0, joint_spaces




	def _apply_external_forces(external_f, f):
		for i in range(0, len(f)):
			f[i] += external_f[i]
		return f




	def get_inverse_dynamics_RNEA(self, root, tip, f_ext = None):
		"""Calculates and returns the casadi inverse dynamics, aka tau, using RNEA"""

		joint_list, actuated_names, upper, lower = self._get_joint_info(root, tip)
		n_joints = len(joint_list)
		q = cs.SX.sym("q", n_joints)
		q_dot = cs.SX.sym("q_dot", n_joints)
		q_ddot = cs.SX.sym("q_ddot", n_joints)
		i_X_p, i_X_0, joint_space = self._get_spatial_transforms_and_joint_space(q, joint_list)
		Ic, n_bodies = self._get_spatial_inertias(root, tip)

		v = []
		a = []
		#f = cs.SX.zeros(n_bodies-1)
		f = []
		tau = cs.SX.zeros(n_bodies-1)
		print "n_bodies: ",n_bodies
		print "n_joints: ",len(joint_list)
		#print q

		v0 = cs.SX.zeros(6,1)
		a_gravity = cs.SX([0., 0., 0., 0., 0., 9.81])

		for i in range(0, n_bodies-1):
			if (joint_list[i].type == "fixed"):
				if((i-1) is -1):
					v.append(v0)
					a.append(cs.mtimes(i_X_p[i], -a_gravity))
				else:
					v.append(cs.mtimes(i_X_p[i], v[i-1]))
					a.append(cs.mtimes(i_X_p[i], a[i-1]))
			else:
				vJ = cs.mtimes(joint_space[i],q_dot[i])

				if((i-1) is -1):
					v.append(vJ)
					a.append(cs.mtimes(i_X_p[i], -a_gravity) + vJ)
				else:
					v.append(cs.mtimes(i_X_p[i], v[i-1]) + joint_space[i]*q_dot[i])
					a.append(cs.mtimes(i_X_p[i], a[i-1]) + cs.mtimes(joint_space[i],q_ddot[i]) + cs.mtimes(plucker.motion_cross_product(v[i]),vJ))

			f.append(cs.mtimes(Ic[i], a[i]) + cs.mtimes(plucker.motion_cross_product(v[i]), cs.mtimes(Ic[i], v[i])))#dim 6x1

		if f_ext is not None:
			f = self._apply_external_forces(f_ext, f)

		for i in range((n_bodies-2), -1, -1):
			tau[i] = cs.mtimes(joint_space[i].T, f[i])
			if (i-1) is not -1:
				f[i-1] = f[i-1] + cs.mtimes(i_X_p[i].T, f[i])


		#for i in range(0, n_bodies-1):
			#tau[i] = cs.Function("tau", [q, q_dot, q_ddot], [tau[i]])
		#f = cs.Function("f", [q, q_dot, q_ddot], [f])
		tau = cs.Function("tau", [q, q_dot, q_ddot], [tau])
		return tau, f




	def _get_jointpace_inertia_matrix(self, Ic, i_X_p, joint_space, n_bodies):
			"""Returns the joint space inertia matrix aka the H-component of the equation of motion tau = H(q)q_ddot + C(q, q_dot,fx)"""
			H = cs.SX.zeros(n_bodies-1, n_bodies-1)

			for i in range(n_bodies-2, -1, -1):
				Ic[i-1] += cs.mtimes(i_X_p[i].T, cs.mtimes(Ic[i], i_X_p[i]))

			for i in range(0, n_bodies-1):
				fh = cs.mtimes(Ic[i], joint_space[i])
				H[i, i] = cs.mtimes(joint_space[i].T, fh)
				j = i
				while (j-1) is not -1:
					fh = cs.mtimes(i_X_p[j].T, fh)
					j -= 1
					H[i,j] = cs.mtimes(joint_space[j].T, fh)
					H[j,i] = H[i,j]
			return H




	def _get_jointspace_bias_matrix(self, joint_list, i_X_p, joint_space, Ic, q, q_dot, n_bodies, f_ext = None):
			"""Returns the joint space bias matrix aka the C-component of the equation of motion tau = H(q)q_ddot + C(q, q_dot,fx)"""
			v = []
			a = []
			f = []
			C = cs.SX.zeros(n_bodies-1)
			v0 = cs.SX.zeros(6,1)
			a_gravity = cs.SX([0., 0., 9.81, 0., 0., 0.])

			for i in range(0, n_bodies-1):
				if (joint_list[i].type == "fixed"):
					if((i-1) is -1):
						v.append(v0)
						a.append(cs.mtimes(i_X_p[i], -a_gravity))
					else:
						v.append(cs.mtimes(i_X_p[i], v[i-1]))
						a.append(cs.mtimes(i_X_p[i], a[i-1]))
				else:
					vJ = cs.mtimes(joint_space[i],q_dot[i])

					if((i-1) is -1):
						v.append(vJ)
						a.append(cs.mtimes(i_X_p[i], -a_gravity))
					else:
						v.append(cs.mtimes(i_X_p[i], v[i-1]) + vJ)
						a.append(cs.mtimes(i_X_p[i], a[i-1]) + cs.mtimes(plucker.motion_cross_product(v[i]),vJ))

				f.append(cs.mtimes(Ic[i], a[i]) + cs.mtimes(plucker.force_cross_product(v[i]), cs.mtimes(Ic[i], v[i])))#dim 6x1

			if f_ext is not None:
				f = self._apply_external_forces(f_ext, f)

			for i in range((n_bodies-2), -1, -1):
				C[i] = cs.mtimes(joint_space[i].T, f[i])
	    		#if (i-1) is not 0:
	        		#f[i-1] = f[i-1] + cs.mtimes(i_X_p[i].T, f[i])
			return C



	def get_forward_dynamics_CRBA(self, root, tip, tau):
			"""Returns the casadi forward dynamics using one of the inertia matrix method - composite rigid body algorithm"""

			joint_list, actuated_names, upper, lower = self._get_joint_info(root, tip)
			n_joints = len(joint_list)
			q = cs.SX.sym("q", n_joints)
			q_dot = cs.SX.sym("q_dot", n_joints)
			q_ddot = cs.SX.zeros(n_joints)
			i_X_p, i_X_0, joint_space = self._get_spatial_transforms_and_joint_space(q, joint_list)
			Ic, n_bodies = self._get_spatial_inertias(root, tip)

			H = self._get_jointpace_inertia_matrix(Ic, i_X_p, joint_space, n_bodies)
			H_inv = cs.solve(H, cs.SX.eye(H.size1()))
			C = self._get_jointspace_bias_matrix(joint_list, i_X_p, joint_space, Ic, q, q_dot, n_bodies)

			q_ddot = cs.mtimes(H_inv, (tau - C))
			q_ddot = cs.Function("q_ddot", [q, q_dot], [q_ddot])

			return q_ddot



	def get_forward_dynamics_ABA(self, root, tip, tau, f_ext = None):
		"""Returns the casadi forward dynamics using the inertia articulated rigid body algorithm"""

		joint_list, actuated_names, upper, lower = self._get_joint_info(root, tip)
		n_joints = len(joint_list)
		q = cs.SX.sym("q", n_joints)
		q_dot = cs.SX.sym("q_dot", n_joints)
		q_ddot = cs.SX.zeros(n_joints)
		i_X_p, i_X_0, joint_space = self._get_spatial_transforms_and_joint_space(q, joint_list)
		Ic, n_bodies = self._get_spatial_inertias(root, tip)

		v = []
		c = []
		pA = []

		#Er en av måtene å deklarere bedre på?
		#U = cs.SX.zeros(6, n_joints)
		#d = cs.SX.zeros(1, n_joints)
		u = cs.SX.zeros(1, n_bodies-1)
		U = [None]*n_joints
		d = [None]*n_joints
		zeros = cs.SX.zeros(6)

		for i in range(0, n_joints):
			if (joint_list[i].type == "fixed"):
				if((i-1) is -1):
					v.append(zeros)
					c.append(zeros)
				else:
					v.append(cs.mtimes(i_X_p[i], v[i-1]))
					c.append(cs.mtimes(plucker.motion_cross_product(v[i]), vJ))

			else:
				vJ = cs.mtimes(joint_space[i], q_dot[i])
				if i-1 is -1:
					v.append(vJ)
					c.append(cs.SX.zeros(6))
				else:
					v.append(cs.mtimes(i_X_p[i], v[i-1]) + vJ)
					c.append(cs.mtimes(plucker.motion_cross_product(v[i]), vJ))

			pA.append(cs.mtimes(plucker.force_cross_product(v[i]), cs.mtimes(Ic[i], v[i])))

		if f_ext is not None:
			pA = self._apply_external_forces(f_ext, pa)

		for i in range(n_bodies-2, -1, -1):
			#U[:6,i] = cs.mtimes(Ic[i], joint_space[i])#6x6 * 6x1 = 6x1
			U[i] = cs.mtimes(Ic[i], joint_space[i])#6x6 * 6x1 = 6x1
			#d[i] = cs.mtimes(joint_space[i].T, U[:6,i])#1x6 x 6x1 = 1x1
			d[i] = cs.mtimes(joint_space[i].T, U[i])
			u[i] = tau[i] - cs.mtimes(joint_space[i].T, pA[i]) # 1x1 - 1x6 x 6x1 = 1x1
			if i-1 is not -1:
				#Hva brukes disse til?
				Ia = Ic[i] - (cs.mtimes(U[i], U[i].T)/d[i])
				pa = pA[i] + cs.mtimes(Ic[i], c[i]) + cs.mtimes(U[i], u[i]/d[i])
				Ic[i-1] += cs.mtimes(i_X_p[i].T, cs.mtimes(Ia, i_X_p[i]))
				pA[i-1] += cs.mtimes(i_X_p[i].T, pa)

		a = []
		a_gravity = cs.SX([0., 0., 0., 0., 0., 9.81])

		for i in range(0, n_joints):
			if i-1 is -1:
				a_temp = (cs.mtimes(i_X_p[i], -a_gravity) + c[i])#6x1
			else:
				a_temp = (cs.mtimes(i_X_p[i], a[i-1]) + c[i])#6x1

			if joint_list[i].type is "fixed":
				a.append(a_temp)

			else:
				#q_ddot[i] = u[i] - (cs.mtimes(U[:6,i].T, a_temp)/d[i])
				q_ddot[i] = u[i] - (cs.mtimes(U[i].T, a_temp)/d[i])
				a.append(a_temp + cs.mtimes(joint_space[i], q_ddot[i]))#6x1

		q_ddot = cs.Function("q_ddot", [q, q_dot], [q_ddot])
		return q_ddot




	def get_forward_kinematics(self, root, tip):
		"""Using one of the above to derive info needed for casadi fk"""
		chain = self.robot_desc.get_chain(root, tip)
		#if self.robot_desc is None:
			#raise ValueError('Robot description not loaded from urdf')
		joint_list, nvar, actuated_names, upper, lower = self.get_joint_info(root, tip)

		#make symbolics
		T_fk = cs.SX.eye(4)
		q = cs.SX.sym("q", nvar)
		quaternion_fk = cs.SX.zeros(4)
		quaternion_fk[3] = 1.0
		dual_quaternion_fk = cs.SX.zeros(8)
		dual_quaternion_fk[3] = 1.0
		i = 0
		for joint in joint_list:
			if joint.type == "fixed":
				xyz = joint.origin.xyz
				rpy = joint.origin.rpy
				joint_frame = T.numpy_rpy(xyz, *rpy)
				joint_quaternion = quaternion.numpy_rpy(*rpy)
				joint_dual_quat = dual_quaternion.numpy_prismatic(xyz,
				                                           rpy,
				                                           [1., 0., 0.],
				                                           0.)
				T_fk = cs.mtimes(T_fk, joint_frame)
				quaternion_fk = quaternion.product(quaternion_fk,
				                                   joint_quaternion)
				dual_quaternion_fk = dual_quaternion.product(
				dual_quaternion_fk,
				joint_dual_quat)

			elif joint.type == "prismatic":
				if joint.axis is None:
					axis = cs.np.array([1., 0., 0.])
				else:
					axis = cs.np.array(joint.axis)
	            #axis = (1./cs.np.linalg.norm(axis))*axis
				joint_frame = T.prismatic(joint.origin.xyz,
				                                  joint.origin.rpy,
				                                  joint.axis, q[i])
				joint_quaternion = quaternion.numpy_rpy(*joint.origin.rpy)
				joint_dual_quat = dual_quaternion.prismatic(
				joint.origin.xyz,
				joint.origin.rpy,
				axis, q[i])
				T_fk = cs.mtimes(T_fk, joint_frame)
				quaternion_fk = quaternion.product(quaternion_fk,
				                                           joint_quaternion)
				dual_quaternion_fk = dual_quaternion.product(dual_quaternion_fk, joint_dual_quat)
				i += 1

			elif joint.type in ["revolute", "continuous"]:
				if joint.axis is None:
					axis = cs.np.array([1., 0., 0.])
				else:
					axis = cs.np.array(joint.axis)
				axis = (1./cs.np.linalg.norm(axis))*axis
				joint_frame = T.revolute(joint.origin.xyz, joint.origin.rpy, joint.axis, q[i])
				joint_quaternion = quaternion.revolute(joint.origin.xyz, joint.origin.rpy, axis, q[i])
				joint_dual_quat = dual_quaternion.revolute(joint.origin.xyz, joint.origin.rpy, axis, q[i])
				T_fk = cs.mtimes(T_fk, joint_frame)
				quaternion_fk = quaternion.product(quaternion_fk, joint_quaternion)
				dual_quaternion_fk = dual_quaternion.product(dual_quaternion_fk,joint_dual_quat)
				i += 1

		T_fk = cs.Function("T_fk", [q], [T_fk])
		quaternion_fk = cs.Function("quaternion_fk", [q], [quaternion_fk])
		dual_quaternion_fk = cs.Function("dual_quaternion_fk", [q], [dual_quaternion_fk])

		return {
		    "joint_names": actuated_names,
		    "upper": upper,
		    "lower": lower,
		    "joint_list": joint_list,
		    "q": q,
		    "quaternion_fk": quaternion_fk,
		    "dual_quaternion_fk": dual_quaternion_fk,
		    "T_fk": T_fk
		}
