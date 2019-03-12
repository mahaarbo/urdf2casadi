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

class URDFparser(object):
	"""Class that turns a chain from URDF to casadi functions"""
	actuated_types = ["prismatic", "revolute", "continuous"]

	def __init__(self):
		self.robot_desc = None


	def from_file(self, filename):
		"""Uses an URDF file to get robot description"""
		print filename
		self.robot_desc = URDF.from_xml_file(filename)

	def from_server(self, key="robot_description"):
		"""Uses a parameter server to get robot description"""
		self.robot_desc = URDF.from_parameter_server(key=key)

	def from_string(self, urdfstring):
		"""Uses a string to get robot description"""
		self.robot_desc = URDF.from_xml_string(urdfstring)



	def get_joint_info(self, root, tip):
		"""Using an URDF to extract joint information, such as a list of joints, actuated names and upper and lower limits for joints"""
		chain = self.robot_desc.get_chain(root, tip)
		if self.robot_desc is None:
			raise ValueError('Robot description not loaded from urdf')

		joint_list = []
		upper = []
		lower = []
		actuated_names = []
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



	def _get_n_joints(self, root, tip):
		"""Returns number of actuated (i.e not fixed) joints"""

		chain = self.robot_desc.get_chain(root, tip)
		#joint_list = []
		n_actuated = 0

		for item in chain:
			if item in self.robot_desc.joint_map:
				joint = self.robot_desc.joint_map[item]
				#joint_list += [joint]
				if (joint.type in self.actuated_types):
					n_actuated += 1
					#if joint.axis is None:
					#	joint.axis = [1., 0., 0.]
					#if joint.origin is None:
					#	joint.origin = Pose(xyz=[0., 0., 0.],
	                #                        rpy=[0., 0., 0.])
					#elif joint.origin.xyz is None:
					#	joint.origin.xyz = [0., 0., 0.]
					#elif joint.origin.rpy is None:
					#	joint.origin.rpy = [0., 0., 0.]

		return n_actuated


	def get_spatial_inertias_old(self, root, tip):
		if self.robot_desc is None:
			raise ValueError('Robot description not loaded from urdf')

		chain = self.robot_desc.get_chain(root, tip)
		spatial_inertias = []

		for item in chain:
			if item in self.robot_desc.link_map:
				link = self.robot_desc.link_map[item]
				if link.inertial is None:
					spatial_inertia = np.zeros((6, 6))
				else:
					I = link.inertial.inertia
					spatial_inertia = plucker.spatial_inertia_matrix_IO(I.ixx, I.ixy, I.ixz, I.iyy, I.iyz, I.izz, link.inertial.mass, link.inertial.origin.xyz)
				spatial_inertias.append(spatial_inertia)

		spatial_inertias.pop(0)
		return spatial_inertias

	def _model_calculation(self, root, tip, q):
		"""Calculates and returns model information, such as transforms, joint space and inertia for actuated joints"""
		if self.robot_desc is None:
		    raise ValueError('Robot description not loaded from urdf')

		chain = self.robot_desc.get_chain(root, tip)
		spatial_inertias = []
		i_X_0 = []
		i_X_p = []
		Sis = []
		prev_joint = None
		n_actuated = 0
		i = 0

		for item in chain:
			#Assuming here that root is always a base link, is this a reasonable assumption?
			if item in self.robot_desc.joint_map:
				joint = self.robot_desc.joint_map[item]
				print joint.type
				if joint.type == "fixed":
					if prev_joint == "fixed":
						XT_prev = cs.mtimes(plucker.XT(joint.origin.xyz, joint.origin.rpy), XT_prev)
					else:
						XT_prev = plucker.XT(joint.origin.xyz, joint.origin.rpy)
					inertia_transform = XT_prev
					prev_inertia = spatial_inertia

				elif joint.type == "prismatic":
					if n_actuated != 0:
						spatial_inertias.append(spatial_inertia)
					n_actuated += 1
					XJT = plucker.XJT_prismatic(joint.origin.xyz, joint.origin.rpy, joint.axis, q[i])
					if (prev_joint == "fixed"):
						XJT = cs.mtimes(XJT, XT_prev)
					Si = cs.SX([0, 0, 0, joint.axis[0], joint.axis[1], joint.axis[2]])
					i_X_p.append(XJT)
					Sis.append(Si)
					i += 1


				elif joint.type in ["revolute", "continuous"]:
					if n_actuated != 0:
						spatial_inertias.append(spatial_inertia)
					n_actuated += 1

					XJT = plucker.XJT_revolute(joint.origin.xyz, joint.origin.rpy, joint.axis, q[i])
					if prev_joint == "fixed":
						XJT = cs.mtimes(XJT, XT_prev)
					Si = cs.SX([joint.axis[0], joint.axis[1], joint.axis[2], 0, 0, 0])
					i_X_p.append(XJT)
					Sis.append(Si)
					i += 1

				prev_joint = joint.type

			if item in self.robot_desc.link_map:
				link = self.robot_desc.link_map[item]
				print link.name
				if link.inertial is None:
					spatial_inertia = np.zeros((6, 6))
				else:
					I = link.inertial.inertia
					spatial_inertia = plucker.spatial_inertia_matrix_IO(I.ixx, I.ixy, I.ixz, I.iyy, I.iyz, I.izz, link.inertial.mass, link.inertial.origin.xyz)
				if prev_joint == "fixed":
					spatial_inertia = prev_inertia + cs.mtimes(inertia_transform.T, cs.mtimes(spatial_inertia, inertia_transform))

				if link.name == tip:
					spatial_inertias.append(spatial_inertia)


		return i_X_p, Sis, spatial_inertias

	def get_spatial_inertias(self, root, tip, inertia_transform):
		if self.robot_desc is None:
			raise ValueError('Robot description not loaded from urdf')

		chain = self.robot_desc.get_chain(root, tip)
		spatial_inertias = []
		n_bodies = (len(chain)-1)/2
		prev_joint = None
		body_i = 0
		f = 0
		for item in chain:
			#Assuming here that root is always a base link, is this a reasonable assumption?
			if item in self.robot_desc.joint_map:
				joint = self.robot_desc.joint_map[item]
				if (joint.type == "fixed") and (body_i is not 1):
					print "constructing first of fixed inertia"
					prev_inertia = spatial_inertia
				elif body_i is not 1 and (joint.type != "fixed"):
					spatial_inertias.append(spatial_inertia)

				prev_joint = joint.type

			if item in self.robot_desc.link_map:
				link = self.robot_desc.link_map[item]


				if link.inertial is None:
					spatial_inertia = np.zeros((6, 6))
				else:
					I = link.inertial.inertia
					spatial_inertia = plucker.spatial_inertia_matrix_IO(I.ixx, I.ixy, I.ixz, I.iyy, I.iyz, I.izz, link.inertial.mass, link.inertial.origin.xyz)

				if prev_joint == "fixed" and (body_i is not 0):
					print "adding fixed inertia to new inertia"
					spatial_inertia = prev_inertia + cs.mtimes(inertia_transform[f].T, cs.mtimes(spatial_inertia, inertia_transform[f]))
					f += 1

				if body_i is n_bodies:
					spatial_inertias.append(spatial_inertia)
				body_i += 1



		return spatial_inertias

	def _get_spatial_transforms_and_Si_old(self, q, joint_list):
		"""Helper function which calculates spatial transform matrices and motion subspace matrices"""
		i_X_0 = []
		i_X_p = []
		Sis = []
		i = 0
		prev_type = None
		inertia_transforms = []

		for joint in joint_list:
			if joint.type == "fixed":
				if prev_type == "fixed":
					XJT_prev = cs.mtimes(plucker.XT(joint.origin.xyz, joint.origin.rpy), XJT_prev)
				else:
					XJT_prev = plucker.XT(joint.origin.xyz, joint.origin.rpy)


			elif joint.type == "prismatic":
				XJT = plucker.XJT_prismatic(joint.origin.xyz, joint.origin.rpy, joint.axis, q[i])
				Si = cs.SX([0, 0, 0, joint.axis[0], joint.axis[1], joint.axis[2]])
				i += 1

			elif joint.type in ["revolute", "continuous"]:
				XJT = plucker.XJT_revolute(joint.origin.xyz, joint.origin.rpy, joint.axis, q[i])
				Si = cs.SX([joint.axis[0], joint.axis[1], joint.axis[2], 0, 0, 0])
				i += 1



			if (prev_type == "fixed") and (joint.type != "fixed"):
				XJT = cs.mtimes(XJT, XJT_prev)

			if joint.type != "fixed":
				i_X_p.append(XJT)
				#i_X_0.append(cs.mtimes(i_X_p[i], i_X_0[i-1]))
				Sis.append(Si)



			prev_type = joint.type



			#obs! must implement for special case of fixed after i=0
			#if(i == 0 and (joint.type != "fixed")):
				#i_X_0.append(i_X_p[i])



		return i_X_p, i_X_0, Sis


	def _get_spatial_transforms_and_Si(self, q, joint_list):
		"""Helper function which calculates spatial transform matrices and motion subspace matrices"""
		i_X_0 = []
		i_X_p = []
		Sis = []
		i = 0
		prev_type = None
		inertia_transforms = []

		for j in range (len(joint_list)):
			joint = joint_list[j]
			if joint.type == "fixed":
				if prev_type == "fixed":
					#eller andre veien? sjekk ut
					XJT_prev = cs.mtimes(plucker.XT(joint.origin.xyz, joint.origin.rpy), XJT_prev)
				else:
					XJT_prev = plucker.XT(joint.origin.xyz, joint.origin.rpy)
					inertia_transforms.append(XJT_prev)


			elif joint.type == "prismatic":
				XJT = plucker.XJT_prismatic(joint.origin.xyz, joint.origin.rpy, joint.axis, q[i])
				Si = cs.SX([0, 0, 0, joint.axis[0], joint.axis[1], joint.axis[2]])
				i += 1

			elif joint.type in ["revolute", "continuous"]:
				XJT = plucker.XJT_revolute(joint.origin.xyz, joint.origin.rpy, joint.axis, q[i])
				Si = cs.SX([joint.axis[0], joint.axis[1], joint.axis[2], 0, 0, 0])
				i += 1

			if (prev_type == "fixed") and (joint.type != "fixed"):
				XJT = cs.mtimes(XJT, XJT_prev)

			if joint.type != "fixed":
				i_X_p.append(XJT)
				#i_X_0.append(cs.mtimes(i_X_p[i], i_X_0[i-1]))
				Sis.append(Si)



			prev_type = joint.type



			#obs! must implement for special case of fixed after i=0
			#if(i == 0 and (joint.type != "fixed")):
				#i_X_0.append(i_X_p[i])



		return i_X_p, i_X_0, Sis, inertia_transforms



	def _apply_external_forces(external_f, f, i_X_0):
		for i in range(0, len(f)):
			f[i] -= cs.mtimes(i_X_0[i], external_f[i])
		return f

	def get_inverse_dynamics_RNEA(self, root, tip, gravity = None, f_ext = None):
		"""Calculates and returns joint torques (inverse dynamics) as a casadi function"""

		if self.robot_desc is None:
			raise ValueError('Robot description not loaded from urdf')

		n_joints = self._get_n_joints(root, tip)
		q = cs.SX.sym("q", n_joints)
		q_dot = cs.SX.sym("q_dot", n_joints)
		q_ddot = cs.SX.sym("q_ddot", n_joints)
		i_X_p, Si, Ic = self._model_calculation(root, tip, q)

		v = []
		a = []
		f = []
		tau = cs.SX.zeros(n_joints)

		for i in range(0, n_joints):
			vJ = cs.mtimes(Si[i],q_dot[i])

			if(i is 0):
				v.append(vJ)
				if gravity is not None:
					ag = np.array([0., 0., 0., gravity[0], gravity[1], gravity[2]])
					a.append(cs.mtimes(i_X_p[i], -ag) + cs.mtimes(Si[i],q_ddot[i]))
				else:
					a.append(cs.mtimes(Si[i],q_ddot[i]))

			else:
				v.append(cs.mtimes(i_X_p[i], v[i-1]) + vJ)
				a.append(cs.mtimes(i_X_p[i], a[i-1]) + cs.mtimes(Si[i],q_ddot[i]) + cs.mtimes(plucker.motion_cross_product(v[i]),vJ))

			f.append(cs.mtimes(Ic[i], a[i]) + cs.mtimes(plucker.force_cross_product(v[i]), cs.mtimes(Ic[i], v[i])))

		if f_ext is not None:
			f = self._apply_external_forces(f_ext, f, i_X_0)

		for i in range(n_joints-1, -1, -1):
			tau[i] = cs.mtimes(Si[i].T, f[i])

			if i is not 0:
				f[i-1] = f[i-1] + cs.mtimes(i_X_p[i].T, f[i])


		tau = cs.Function("C", [q, q_dot, q_ddot], [tau], {"jit": True, "jit_options":{"flags":"-Ofast"}})
		return tau



	def get_gravity_RNEA(self, root, tip, gravity):
		"""Calculates and returns the gravity terms for each joint, given as a casadi function, using RNEA"""

		if self.robot_desc is None:
			raise ValueError('Robot description not loaded from urdf')

		n_joints = self._get_n_joints(root, tip)
		q = cs.SX.sym("q", n_joints)
		i_X_p, Si, Ic = self._model_calculation(root, tip, q)

		v = []
		a = []
		ag = cs.SX([0., 0., 0., gravity[0], gravity[1], gravity[2]])
		f = []
		tau = cs.SX.zeros(n_joints)

		for i in range(0, n_joints):
			if(i is 0):
				a.append(cs.mtimes(i_X_p[i], -ag))
			else:
				a.append(cs.mtimes(i_X_p[i], a[i-1]))
			f.append(cs.mtimes(Ic[i], a[i]))


		for i in range(n_joints-1, -1, -1):
			tau[i] = cs.mtimes(Si[i].T, f[i])

			if i is not 0:
				f[i-1] = f[i-1] + cs.mtimes(i_X_p[i].T, f[i])

		tau = cs.Function("C", [q], [tau], {"jit": True, "jit_options":{"flags":"-Ofast"}})
		return tau



	def _get_M(self, Ic, i_X_p, Si, n_joints, q):
			"""Returns the joint space inertia matrix aka the H-component of the equation of motion tau = H(q)q_ddot + C(q, q_dot,fx)"""
			M = cs.SX.zeros(n_joints, n_joints)
			Ic_composite = [None]*len(Ic)

			for i in range(0, n_joints):
				Ic_composite[i] = Ic[i]

			for i in range(n_joints-1, -1, -1):
				if i is not 0:
					Ic_composite[i-1] = Ic[i-1] + cs.mtimes(i_X_p[i].T, cs.mtimes(Ic_composite[i], i_X_p[i]))

			for i in range(0, n_joints):
				fh = cs.mtimes(Ic_composite[i], Si[i])
				M[i, i] = cs.mtimes(Si[i].T, fh)
				j = i
				while j is not 0:
					fh = cs.mtimes(i_X_p[j].T, fh)
					j -= 1
					M[i,j] = cs.mtimes(Si[j].T, fh)
					M[j,i] = M[i,j]

			return M



	def get_jointspace_inertia_matrix(self, root, tip):
			"""Returns the joint space inertia matrix aka the M-component of the equation of motion tau = M(q)q_ddot + C(q, q_dot) + g(q)"""
			if self.robot_desc is None:
				raise ValueError('Robot description not loaded from urdf')


			n_joints = self._get_n_joints(root, tip)
			q = cs.SX.sym("q", n_joints)
			i_X_p, Si, Ic = self._model_calculation(root, tip, q)
			M = cs.SX.zeros(n_joints, n_joints)
			Ic_composite = [None]*len(Ic)

			for i in range(0, n_joints):
				Ic_composite[i] = Ic[i]

			for i in range(n_joints-1, -1, -1):
				if i is not 0:
					Ic_composite[i-1] = Ic[i-1] + cs.mtimes(i_X_p[i].T, cs.mtimes(Ic_composite[i], i_X_p[i]))

			for i in range(0, n_joints):
				fh = cs.mtimes(Ic_composite[i], Si[i])
				M[i, i] = cs.mtimes(Si[i].T, fh)
				j = i
				while j is not 0:
					fh = cs.mtimes(i_X_p[j].T, fh)
					j -= 1
					M[i,j] = cs.mtimes(Si[j].T, fh)
					M[j,i] = M[i,j]

			M = cs.Function("M", [q], [M], {"jit": True, "jit_options":{"flags":"-Ofast"}})
			return M



	def get_jointspace_inertia_matrix_2point0(self, root, tip):
			"""Returns the joint space inertia matrix aka the H-component of the equation of motion tau = H(q)q_ddot + C(q, q_dot,fx)"""
			if self.robot_desc is None:
				raise ValueError('Robot description not loaded from urdf')


			n_joints = self._get_n_joints(root, tip)
			q = cs.SX.sym("q", n_joints)
			i_X_p, Si, Ic = self._model_calculation(root, tip, q)
			H = cs.SX.zeros(n_joints, n_joints)
			Ic_composite = [None]*len(Ic)

			for i in range(0, n_joints):
				Ic_composite[i] = Ic[i]

			for i in range(n_joints-1, -1, -1):
				if i is not 0:
					Ic_composite[i-1] = Ic[i-1] + cs.mtimes(i_X_p[i].T, cs.mtimes(Ic_composite[i], i_X_p[i]))

			for i in range(n_joints-1, -1, -1):
				fh = cs.mtimes(Ic_composite[i], Si[i])
				H[i, i] = cs.mtimes(Si[i].T, fh)
				j = i
				while j is not 0:
					fh = cs.mtimes(i_X_p[j].T, fh)
					j -= 1
					H[i,j] = cs.mtimes(Si[j].T, fh)
					H[j,i] = H[i,j]

			H = cs.Function("H", [q], [H], {"jit": True, "jit_options":{"flags":"-Ofast"}})
			return M


	def _get_C(self, i_X_p, Si, Ic, q, q_dot, n_joints, gravity = None, f_ext = None):
		"""Returns the coriolis terms, aka the C-componentvof the equation of motion tau = M(q)q_ddot + C(q, q_dot) + g(q)"""

		v = []
		a = []
		f = []
		C = cs.SX.zeros(n_joints)

		for i in range(0, n_joints):

			vJ = cs.mtimes(Si[i],q_dot[i])

			if(i is 0):
				v.append(vJ)
				if gravity is not None:
					ag = np.array([0., 0., 0., gravity[0], gravity[1], gravity[2]])
					#ag = (cs.SX([0., 0., 0., gravity[0], gravity[1], gravity[2]]))
					a.append(cs.mtimes(i_X_p[i], -ag))
				else:
					#a.append(cs.mtimes(Si[i],q_ddot[i]))
					a.append(cs.SX([0., 0., 0., 0., 0., 0.]))


			else:
				v.append(cs.mtimes(i_X_p[i], v[i-1]) + vJ)
				a.append(cs.mtimes(i_X_p[i], a[i-1]) + cs.mtimes(plucker.motion_cross_product(v[i]),vJ))

			f.append(cs.mtimes(Ic[i], a[i]) + cs.mtimes(plucker.force_cross_product(v[i]), cs.mtimes(Ic[i], v[i])))

		if f_ext is not None:
			f = self._apply_external_forces(f_ext, f, i_X_0)

		for i in range(n_joints-1, -1, -1):
			C[i] = cs.mtimes(Si[i].T, f[i])

			if i is not 0:
				f[i-1] = f[i-1] + cs.mtimes(i_X_p[i].T, f[i])

		return C





	def get_jointspace_bias_matrix(self, root, tip, f_ext = None):
		"""Returns the coriolis terms for each joint, that is the C-component of the equation of motion tau = H(q)q_ddot + C(q, q_dot) + g(q), using RNEA"""
		if self.robot_desc is None:
			raise ValueError('Robot description not loaded from urdf')


		n_joints = self._get_n_joints(root, tip)
		q = cs.SX.sym("q", n_joints)
		q_dot = cs.SX.sym("q_dot", n_joints)
		i_X_p, Si, Ic = self._model_calculation(root, tip, q)

		v = []
		a = []
		f = []
		tau = cs.SX.zeros(n_joints)

		for i in range(0, n_joints):
			vJ = cs.mtimes(Si[i],q_dot[i])

			if(i is 0):
				v.append(vJ)
				a.append(cs.SX([0., 0., 0., 0., 0., 0.]))
				#a.append(np.array([0., 0., 0., 0., 0., 0.]))
				#a.append(np.zeros(6))
				#a.append([0., 0., 0., 0., 0., 0.])
			else:
				v.append(cs.mtimes(i_X_p[i], v[i-1]) + vJ)
				a.append(cs.mtimes(i_X_p[i], a[i-1]) + cs.mtimes(plucker.motion_cross_product(v[i]),vJ))

			f.append(cs.mtimes(Ic[i], a[i]) + cs.mtimes(plucker.force_cross_product(v[i]), cs.mtimes(Ic[i], v[i])))

		if f_ext is not None:
			f = self._apply_external_forces(f_ext, f, i_X_0)

		for i in range(n_joints-1, -1, -1):
			tau[i] = cs.mtimes(Si[i].T, f[i])

			if i is not 0:
				f[i-1] = f[i-1] + cs.mtimes(i_X_p[i].T, f[i])

		C = cs.Function("C", [q, q_dot], [tau], {"jit": True, "jit_options":{"flags":"-Ofast"}})
		return C



	def get_forward_dynamics_CRBA(self, root, tip, tau, gravity = None, f_ext = None):
			"""Returns the joint accelerations, i.e the forward dynamics, by solving the equation of motion and combining CRBA and RNEA"""

			if self.robot_desc is None:
				raise ValueError('Robot description not loaded from urdf')

			n_joints = self._get_n_joints(root, tip)
			q = cs.SX.sym("q", n_joints)
			q_dot = cs.SX.sym("q_dot", n_joints)
			q_ddot = cs.SX.zeros(n_joints)

			i_X_p, Si, Ic = self._model_calculation(root, tip, q)
			M = self._get_M(Ic, i_X_p, Si, n_joints, q)
			M_inv = cs.solve(M, cs.SX.eye(M.size1()))
			C = self._get_C(i_X_p, Si, Ic, q, q_dot, n_joints, gravity, f_ext)
			q_ddot = cs.mtimes(M_inv, (tau - C))
			q_ddot = cs.Function("q_ddot", [q, q_dot], [q_ddot], {"jit": True, "jit_options":{"flags":"-Ofast"}})

			return q_ddot



	def get_forward_dynamics_ABA(self, root, tip, tau, gravity = None, f_ext = None):
		"""Returns the joint accelerations, i.e forward dynamics, using the inertia articulated rigid body algorithm"""

		if self.robot_desc is None:
			raise ValueError('Robot description not loaded from urdf')

		n_joints = self._get_n_joints(root, tip)
		q = cs.SX.sym("q", n_joints)
		q_dot = cs.SX.sym("q_dot", n_joints)
		q_ddot = cs.SX.zeros(n_joints)
		i_X_p, Si, Ic = self._model_calculation(root, tip, q)

		v = []
		c = []
		pA = []
		IA = []

		u = [None]*n_joints
		U = [None]*n_joints
		d = [None]*n_joints

		for i in range(0, n_joints):
			vJ = cs.mtimes(Si[i], q_dot[i])
			if i is 0:
				#v0 = S*qdot0 = [0, qdot0, 0, 0, 0, 0] = [0, 1, 0, 0, 0, 0]
				v.append(vJ)
				c.append([0, 0, 0, 0, 0, 0])
				#c.append(cs.SX.zeros(n_joints))
				#c0 = [0, 0, 0, 0, 0, 0]
			else:
				v.append(cs.mtimes(i_X_p[i], v[i-1]) + vJ)
				c.append(cs.mtimes(plucker.motion_cross_product(v[i]), vJ))

			IA.append(Ic[i])
			pA.append(cs.mtimes(plucker.force_cross_product(v[i]), cs.mtimes(Ic[i], v[i])))

		if f_ext is not None:
			pA = self._apply_external_forces(f_ext, pA)

		for i in range(n_joints-1, -1, -1):
			U[i] = cs.mtimes(IA[i], Si[i])
			d[i] = cs.mtimes(Si[i].T, U[i])
			u[i] = tau[i] - cs.mtimes(Si[i].T, pA[i])
			if i is not 0:
				Ia = IA[i] - ((cs.mtimes(U[i], U[i].T)/d[i]))
				pa = pA[i] + cs.mtimes(Ia, c[i]) + (cs.mtimes(U[i], u[i])/d[i])
				IA[i-1] += cs.mtimes(i_X_p[i].T, cs.mtimes(Ia, i_X_p[i]))
				pA[i-1] += cs.mtimes(i_X_p[i].T, pa)

		a = []

		for i in range(0, n_joints):
			if i is 0:
				if gravity is not None:
					ag = np.array([0., 0., 0., gravity[0], gravity[1], gravity[2]])
					a_temp = (cs.mtimes(i_X_p[i], -ag) + c[i])

				else:
					a_temp = c[i]

			else:
				a_temp = (cs.mtimes(i_X_p[i], a[i-1]) + c[i])

			q_ddot[i] = (u[i] - cs.mtimes(U[i].T, a_temp))/d[i]
			a.append(a_temp + cs.mtimes(Si[i], q_ddot[i]))#6x1

		q_ddot = cs.Function("q_ddot", [q, q_dot], [q_ddot], {"jit": True, "jit_options":{"flags":"-Ofast"}})
		return q_ddot




	def get_forward_kinematics(self, root, tip):
		"""Using one of the above to derive info needed for casadi fk"""
		chain = self.robot_desc.get_chain(root, tip)
		if self.robot_desc is None:
			raise ValueError('Robot description not loaded from urdf')
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

		T_fk = cs.Function("T_fk", [q], [T_fk], {"jit": True, "jit_options":{"flags":"-Ofast"}})
		quaternion_fk = cs.Function("quaternion_fk", [q], [quaternion_fk], {"jit": True, "jit_options":{"flags":"-Ofast"}})
		dual_quaternion_fk = cs.Function("dual_quaternion_fk", [q], [dual_quaternion_fk], {"jit": True, "jit_options":{"flags":"-Ofast"}})

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
