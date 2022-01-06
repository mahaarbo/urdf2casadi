import urdf2casadi.urdfparser as u2c
import numpy as np
import os
import pytest


@pytest.fixture
def ur5():
    ur5 = u2c.URDFparser()
    path_to_urdf = absPath = (
        os.path.dirname(os.path.abspath(__file__)) + "/urdf/ur5_mod.urdf"
    )
    ur5.from_file(path_to_urdf)
    return ur5


def test_jointInfo(ur5):
    root = "base_link"
    tip = "tool0"

    joint_list, joint_names, q_max, q_min = ur5.get_joint_info(root, tip)
    n_joints = ur5.get_n_joints(root, tip)
    assert joint_names[0] == "shoulder_pan_joint"
    assert joint_list[0].child == "shoulder_link"
    assert joint_list[0].dynamics.friction == 0.0
    assert q_max[0] == pytest.approx(6.2831, abs=1e-4)
    assert q_min[4] == pytest.approx(-6.2831, abs=1e-4)
    assert len(q_max) == 6
    assert n_joints == 6


def test_dynamics(ur5):
    root = "base_link"
    tip = "tool0"
    M_sym = ur5.get_inertia_matrix_crba(root, tip)
    C_sym = ur5.get_coriolis_rnea(root, tip)

    gravity = [0, 0, -9.81]
    G_sym = ur5.get_gravity_rnea(root, tip, gravity)

    q = [-3.0, 2.5, 0.21, -4.5, -1.0, 2.0]
    q_dot = [0.1, 1.2, -0.6, -1.3, 0.5, 0.6]

    M_num_ca = M_sym(q)
    print(M_num_ca)
    M_num = np.array(M_num_ca)
    C_num = np.array(C_sym(q, q_dot))
    G_num = np.array(G_sym(q))

    assert M_num.shape == (6, 6)
    assert G_num.shape == (6, 1)
    assert C_num.shape == (6, 1)
    M_rbdl = np.array(
        [
            [
                3.13241106e00,
                2.89631451e-01,
                3.78143202e-02,
                8.07922723e-04,
                5.17346871e-02,
                -4.18539959e-02,
            ],
            [
                2.89631451e-01,
                4.04634894e00,
                1.55862262e00,
                2.51441801e-01,
                1.09647259e-02,
                2.75329805e-02,
            ],
            [
                3.78143202e-02,
                1.55862262e00,
                8.74962680e-01,
                2.62963098e-01,
                7.72580281e-03,
                2.75329805e-02,
            ],
            [
                8.07922723e-04,
                2.51441801e-01,
                2.62963098e-01,
                2.75525773e-01,
                4.52206818e-03,
                2.75329805e-02,
            ],
            [
                5.17346871e-02,
                1.09647259e-02,
                7.72580281e-03,
                4.52206818e-03,
                2.57855217e-01,
                -3.40000975e-14,
            ],
            [
                -4.18539959e-02,
                2.75329805e-02,
                2.75329805e-02,
                2.75329805e-02,
                -3.40000975e-14,
                5.09584731e-02,
            ],
        ]
    )
    for i in range(6):
        for j in range(6):
            assert M_num[i, j] == pytest.approx(M_rbdl[i, j])
    g_rbdl = np.array(
        [
            4.44089210e-16,
            5.00009444e01,
            1.45340187e01,
            -3.68345465e-01,
            8.00043901e-02,
            0.00000000e00,
        ]
    )
    for i in range(6):
        assert G_num[i] == pytest.approx(g_rbdl[i])
    c_rbdl_all = np.array(
        [
            -4.81717356e-02,
            5.01697792e01,
            1.47604220e01,
            -3.56463193e-01,
            1.15180538e-01,
            -9.86274036e-03,
        ]
    )
    c_rbdl = c_rbdl_all - g_rbdl
    for i in range(6):
        assert C_num[i] == pytest.approx(c_rbdl[i])


def test_inversDynamics(ur5):
    root = "base_link"
    tip = "tool0"

    gravity = [0, 0, -9.81]

    tau_g_sym = ur5.get_inverse_dynamics_rnea(root, tip, gravity=gravity)

    q = [-3.0, 2.5, 0.21, -4.5, -1.0, 2.0]
    q_dot = [0.1, 1.2, -0.6, -1.3, 0.5, 0.6]
    q_ddot = np.array([0.5, -0.3, 0.2, 0.1, 0.6, 0.6])

    tau_g_num = np.array(tau_g_sym(q, q_dot, q_ddot))

    tau_rbdl = np.array(
        [
            1.44471643e00,
            4.94606576e01,
            1.45341865e01,
            -3.32113546e-01,
            2.94468961e-01,
            -2.14654411e-04,
        ]
    )
    for i in range(6):
        assert tau_g_num[i] == pytest.approx(tau_rbdl[i])


def test_forwardDynamics(ur5):
    root = "base_link"
    tip = "tool0"

    gravity = [0, 0, -9.81]

    qddot_g_sym = ur5.get_forward_dynamics_aba(root, tip, gravity=gravity)

    q = [-3.0, 2.5, 0.21, -4.5, -1.0, 2.0]
    q_dot = [0.1, 1.2, -0.6, -1.3, 0.5, 0.6]
    tau = np.array([0.4, -0.2, 0.1, 1.2, 0.2, -0.3])

    qddot_g_num = np.array(qddot_g_sym(q, q_dot, tau))
    qddot_rbdl = np.array(
        [1.52072684, -17.46341328, 10.95245985, 11.85747633, 0.23032233, -7.33331383]
    )
    for i in range(6):
        assert qddot_g_num[i] == pytest.approx(qddot_rbdl[i])
