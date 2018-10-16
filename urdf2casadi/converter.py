"""This module contains functions for turning a chain in a URDF to a
casadi function.
"""
import casadi as cs
from urdf_parser_py.urdf import URDF, Pose
from urdf2casadi import casadi_geom
from urdf2casadi import numpy_geom


def from_string(root, tip, urdfstring):
    robot_desc = URDF.from_xml_string(urdfstring)
    chain_list = robot_desc.get_chain(root, tip)
    return get_fk_dict(robot_desc, chain_list)


def from_file(root, tip, filename):
    robot_desc = URDF.from_xml_file(filename)
    chain_list = robot_desc.get_chain(root, tip)
    return get_fk_dict(robot_desc, chain_list)


def from_parameter_server(root, tip, key="robot_description"):
    robot_desc = URDF.from_parameter_server(key=key)
    chain_list = robot_desc.get_chain(root, tip)
    return get_fk_dict(robot_desc, chain_list)


def get_fk_dict(robot_desc, chain):
    """Returns an fk_dict from URDF and a chain."""
    # First get simple info and set defaults
    nvar = 0
    joint_list = []
    upper = []
    lower = []
    actuated_types = ["prismatic", "revolute", "continuous"]
    actuated_names = []
    for item in chain:
        if item in robot_desc.joint_map:
            joint = robot_desc.joint_map[item]
            joint_list += [joint]
            if joint.type in actuated_types:
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
    # Then start on symbolics
    q = cs.SX.sym("q", nvar)
    T_fk = cs.SX.eye(4)
    quaternion_fk = cs.SX.zeros(4)
    quaternion_fk[3] = 1.0
    dual_quaternion_fk = cs.SX.zeros(8)
    dual_quaternion_fk[3] = 1.0
    i = 0
    for joint in joint_list:
        if joint.type == "fixed":
            xyz = joint.origin.xyz
            rpy = joint.origin.rpy
            joint_frame = numpy_geom.T_rpy(xyz,
                                           *rpy)
            joint_quaternion = numpy_geom.quaternion_rpy(*rpy)
            joint_dual_quat = numpy_geom.dual_quaternion_prismatic(xyz,
                                                                   rpy,
                                                                   [1., 0., 0.],
                                                                   0.)
            T_fk = cs.mtimes(T_fk, joint_frame)
            quaternion_fk = casadi_geom.quaternion_product(quaternion_fk,
                                                           joint_quaternion)
            dual_quaternion_fk = casadi_geom.dual_quaternion_product(
                dual_quaternion_fk,
                joint_dual_quat)
        elif joint.type == "prismatic":
            if joint.axis is None:
                axis = cs.np.array([1., 0., 0.])
            axis = cs.np.array(joint.axis)
            #axis = (1./cs.np.linalg.norm(axis))*axis
            joint_frame = casadi_geom.T_prismatic(joint.origin.xyz,
                                                  joint.origin.rpy,
                                                  joint.axis, q[i])
            joint_quaternion = numpy_geom.quaternion_rpy(*joint.origin.rpy)
            joint_dual_quat = casadi_geom.dual_quaternion_prismatic(
                joint.origin.xyz,
                joint.origin.rpy,
                axis, q[i])
            T_fk = cs.mtimes(T_fk, joint_frame)
            quaternion_fk = casadi_geom.quaternion_product(quaternion_fk,
                                                           joint_quaternion)
            dual_quaternion_fk = casadi_geom.dual_quaternion_product(
                dual_quaternion_fk,
                joint_dual_quat)
            i += 1
        elif joint.type in ["revolute", "continuous"]:
            if joint.axis is None:
                axis = cs.np.array([1., 0., 0.])
            axis = cs.np.array(joint.axis)
            axis = (1./cs.np.linalg.norm(axis))*axis
            joint_frame = casadi_geom.T_revolute(joint.origin.xyz,
                                                 joint.origin.rpy,
                                                 joint.axis, q[i])
            joint_quaternion = casadi_geom.quaternion_revolute(joint.origin.xyz,
                                                               joint.origin.rpy,
                                                               axis, q[i])
            joint_dual_quat = casadi_geom.dual_quaternion_revolute(
                joint.origin.xyz,
                joint.origin.rpy,
                axis, q[i])
            T_fk = cs.mtimes(T_fk, joint_frame)
            quaternion_fk = casadi_geom.quaternion_product(quaternion_fk,
                                                           joint_quaternion)
            dual_quaternion_fk = casadi_geom.dual_quaternion_product(
                dual_quaternion_fk,
                joint_dual_quat)
            i += 1
    quaternion_fk = cs.Function("quaternion_fk", [q], [quaternion_fk])
    dual_quaternion_fk = cs.Function("dual_quaternion_fk",
                                     [q], [dual_quaternion_fk])
    T_fk = cs.Function("T_fk", [q], [T_fk])
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


def from_denavit_hartenberg(joint_angles, link_lengths, link_offsets,
                            link_twists, joint_names=[],
                            upper_limits=[], lower_limits=[]):
    """Returns a fk_dict from denavit hartenberg parameters. Indicate
    joint variables with \"s\" in the relevant list. The rest should be
    floats.
    """
    all_props = joint_angles + link_lengths + link_offsets + link_twists
    if len(all_props) % 4 != 0:
        raise IndexError("Uneven number of parameters."
                         + " joint_angles="+str(len(joint_angles))
                         + " link_lengths="+str(len(link_lengths))
                         + " link_offsets="+str(len(link_offsets))
                         + " link_twists="+str(len(link_twists)))
    # Check all for strings
    all_robot_vars = []
    for i, element in enumerate(all_props):
        if isinstance(element, str):
            qi = cs.SX.sym("q"+str(i))
            all_props[i] = qi
            all_robot_vars += [qi]
    # Then put them all back
    idx = 0
    joint_angles = all_props[idx:len(joint_angles)]
    idx += len(joint_angles)
    link_lengths = all_props[idx:idx+len(link_lengths)]
    idx += len(link_lengths)
    link_offsets = all_props[idx: idx+len(link_offsets)]
    idx += len(link_offsets)
    link_twists = all_props[idx:]
    q = cs.vertcat(*all_robot_vars)
    # Then start on forming the expressions
    T_fk = cs.SX.eye(4)
    quaternion_fk = cs.SX.zeros(4)
    quaternion_fk[3] = 1.0
    dual_quaternion_fk = cs.SX.zeros(8)
    dual_quaternion_fk[3] = 1.0
    for i in range(len(joint_angles)):
        jai = joint_angles[i]
        lli = link_lengths[i]
        loi = link_offsets[i]
        lti = link_twists[i]
        T_dhi = casadi_geom.T_denavit_hartenberg(jai, lli, loi, lti)
        dual_quat_dhi = casadi_geom.dual_quaternion_denavit_hartenberg(jai,
                                                                       lli,
                                                                       loi,
                                                                       lti)
        T_fk = cs.mtimes(T_fk, T_dhi)
        quaternion_fk = casadi_geom.quaternion_product(quaternion_fk,
                                                       dual_quat_dhi[:4])
        dual_quaternion_fk = casadi_geom.dual_quaternion_product(
            dual_quaternion_fk,
            dual_quat_dhi
        )
    quaternion_fk = cs.Function("quaternion_fk", [q], [quaternion_fk])
    dual_quaternion_fk = cs.Function("dual_quaternion_fk",
                                     [q], [dual_quaternion_fk])
    T_fk = cs.Function("T_fk", [q], [T_fk])
    return {
        "joint_names": joint_names,
        "upper": upper_limits,
        "lower": lower_limits,
        "joint_list": [],
        "q": q,
        "quaternion_fk": quaternion_fk,
        "dual_quaternion_fk": dual_quaternion_fk,
        "T_fk": T_fk
    }
