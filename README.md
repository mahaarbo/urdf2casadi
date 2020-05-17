# URDF2CASADI
## Installation
1. [Get ROS](http://www.ros.org/install/) (actually anything that installs `urdfdom_py`/`urdf_parser_py` will do)
2. [Get CasADi](https://github.com/casadi/casadi/wiki/InstallationInstructions) (e.g. `pip install casadi`)
3. run `pip install --user .` in the folder

## Usage example
```python
import casadi as cs
from urdf2casadi import urdfparser as u2c
urdf_path = "../urdf/ur5_mod.urdf"
root_link = "base_link"
end_link = "tool0"
robot_parser = u2c.URDFparser()
fk_dict = robot_parser.get_forward_kinematics(root_link, end_link)
print fk_dict.keys()
# should give ['upper', 'T_fk', 'lower', 'q', 'joint_names', 'joint_list']
forward_kinematics = fk_dict["T_fk"]
print forward_kinematics([0.3, 0.3, 0.3, 0., 0.3, 0.7])
```

## Todo/Implementation status
- [x] Forward kinematics with SE(3) matrix
- [x] Forward kinematics of rotation with quaternion
- [x] Dual Quaternions as alternative to SE(3) matrices
- [x] Dynamics from links and their inertia tags
- [ ] Denavit Hartenberg?
- [ ] Move numerical to a test folder
- [x] Examples
