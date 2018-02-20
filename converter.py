"""This module contains functions for turning a chain in a URDF to a
casadi function.
"""
import casadi as cs
from urdf_parser_py.urdf import URDF, Pose
import casadi_geom
import numpy_geom


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
    q = cs.MX.sym("q", nvar)
    T_fk = cs.MX.eye(4)
    # quaternion_fk = cs.MX.zeros(4, 1)
    # quaternion_fk[3, 0] = 1.0
    i = 0

    for joint in joint_list:
        if joint.type == "fixed":
            joint_frame = numpy_geom.T_rpy(joint.origin.xyz,
                                           *joint.origin.rpy)
            # joint_quaternion = numpy_geom.quaternion_rpy(*joint.origin.rpy)
            T_fk = cs.mtimes(T_fk, joint_frame)
            # quaternion_fk = casadi_geom.quaternion_product(quaternion_fk,
            #                                               joint_quaternion)
        elif joint.type == "prismatic":
            if joint.axis is None:
                joint.axis = [1., 0., 0.]
            joint_frame = casadi_geom.T_prismatic(joint.origin.xyz,
                                                  joint.origin.rpy,
                                                  joint.axis, q[i])
            # joint_quaternion = numpy_geom.quaternion_rpy(*joint.origin.rpy)
            T_fk = cs.mtimes(T_fk, joint_frame)
            # quaternion_fk = casadi_geom.quaternion_product(quaternion_fk,
            #                                               joint_quaternion)
            i += 1
        elif joint.type in ["revolute", "continuous"]:
            if joint.axis is None:
                joint.axis = [1., 0., 0.]
            joint_frame = casadi_geom.T_revolute(joint.origin.xyz,
                                                 joint.origin.rpy,
                                                 joint.axis, q[i])
            # joint_quaternion = casadi_geom.quaternion_revolute(joint.origin.xyz,
            #                                                   joint.origin.rpy,
            #                                                   joint.axis, q[i])
            T_fk = cs.mtimes(T_fk, joint_frame)
            # quaternion_fk = casadi_geom.quaternion_product(quaternion_fk,
            #                                               joint_quaternion)
            i += 1

    return {
        "joint_names": actuated_names,
        "upper": upper,
        "lower": lower,
        "joint_list": joint_list,
        "q": q,
        # "quaternion_fk": quaternion_fk,
        "T_fk": T_fk
    }
