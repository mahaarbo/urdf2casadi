"""This module contains a class for turning a chain in a URDF to a
casadi function.
"""
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
	def get_joint_info(self, root, tip):
		"""Using an URDF to extract a proper joint_list"""
		chain = self.robot_desc.get_chain(root, tip)
		if self.robot_desc is None:
			raise ValueError('Robot description not loaded from urdf')

		nvar = 0
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
					nvar += 1
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

		return joint_list, nvar, actuated_names, upper, lower

	def get_spatial_inertias(self, root, tip):
		chain = self.robot_desc.get_chain(root, tip)
		#link_list = []
		spatial_inertias = []
		NB = 0

		for item in chain:
			if item in self.robot_desc.link_map:
				link = self.robot_desc.link_map[item]
				NB += 1
				if link.inertial is not None:
					I = link.inertial.inertia
					spatial_inertia = plucker.spatial_inertia_matrix(I.ixx, I.ixy, I.ixz, I.iyz, I.iyy, I.izz, link.inertial.mass)
					spatial_inertias.append(spatial_inertia)

		return spatial_inertias, NB


	def jcalc(self, root, tip, q, joint_list):
		"""Helper function for RNEA which calculates spatial transform matrices and motion subspace matrices"""
		i_X_0 = []
		joint_motions = []
		joint_motion = [0, 0, 0, 0, 0, 0]
		i = 0
		for joint in joint_list:
			XL = plucker.XL(joint.origin.xyz, joint.origin.rpy)
			if joint.type == "fixed":
				XJ = plucker.XL(joint.origin.xyz, joint.origin.rpy)
				joint_motion = [0, 0, 0, 0, 0, 0]

			elif joint.type == "prismatic":
				XJ = plucker.XJ_prismatic(joint.origin.xyz, joint.origin.rpy, joint.axis, q[i])

				if joint.axis.xyz[0] is 1:
					joint_motion = [1, 0, 0, 0, 0, 0]
				elif joint.axis.xyz[1] is 1:
					joint_motion = [0, 1, 0, 0, 0, 0]
				elif joint.axis.xyz[2] is 1:
					joint_motion = [0, 0, 1, 0, 0, 0]

			elif joint.type in ["revolute", "continuous"]:
				XJ = plucker.XJ_revolute(joint.origin.xyz, joint.origin.rpy, joint.axis, q[i])
				if joint.axis[0] is 1:
					joint_motion = [0, 0, 0, 1, 0, 0]
				elif joint.axis[1] is 1:
					joint_motion = [0, 1, 0, 0, 1, 0]
				elif joint.axis[2] is 1:
					joint_motion = [0, 0, 1, 0, 0, 1]

			i_X_p = cs.mtimes(XJ, XL)
			joint_motions.append(joint_motion)

			if(i == 0):
				i_X_0.append(i_X_p)

			else:
				i_X_0.append(cs.mtimes(i_X_p, i_X_0[i-1]))
			i += 1

		return i_X_0, joint_motions


	def get_inverse_dynamics(self, root, tip):
		"""Using one of the above to derive info needed for casadi id"""

		joint_list, nvar, actuated_names, upper, lower = self.get_joint_info(root, tip)
		q = cs.SX.sym("q", nvar)
		q_dot = cs.SX.sym("q_dot", nvar)
		q_ddot = cs.SX.sym("q_ddot", nvar)
		i_X_0, motion_space = self.jcalc(root, tip, q, joint_list)
		Ic, NB = self.get_spatial_inertias(root, tip)

		print(NB)

		v = []
		a = []
		f = []
		v.append(np.zeros(6))#kan man bruke numpy eller vanlig lise?
		a.append([0., 0., 9.81, 0., 0., 0.])

		for i in range(1, NB-2):
			vJ = cs.mtimes(motion_space[i],q_dot[i])
			#vJ = motion_space[i]*q_dot[i]

			if((i-1) is 0):
				v.append(vJ)
				a.append(cs.mtimes(i_X_0[i], (-np.transpose(a[i-1]))) + cs.mtimes(motion_space[i], q_ddot[i]))
			else:
				v.append(cs.mtimes(i_X_0[i], v[i-1]) + motion_space[i]*q_dot[i])
				a.append(cs.mtimes(i_X_0[i], a[i-1]) + cs.mtimes(motion_space[i],q_ddot[i]) + cs.mtimes(plucker.spatial_cross_product(v[i]),vJ))

			f.append(cs.mtimes(i_X_0[i], a[i-1]) - cs.mtimes(plucker.spatial_cross_product(v[i]), cs.mtimes(i_X_0[i], v[i-1])))

		return f


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


	def get_inverse_dynamics_parameters():
		"""Using one of the above to derive info needed for casadi id"""
