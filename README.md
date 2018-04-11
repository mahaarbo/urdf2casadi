# URDF2CASADI
## Installation
1. [Get ROS](http://www.ros.org/install/) (actually anything that installs `urdfdom_py`/`urdf_parser_py` will do)
2. [Get CasADi](https://github.com/casadi/casadi/wiki/InstallationInstructions) (e.g. `pip install casadi`)

## Usage example
```python
import casadi as cs
from urdf2casadi import converter
fk_dict = converter.from_file("root", "gantry_tool0", "robot_urdf_file_path.urdf")
print fk_dict.keys()
# should give ['upper', 'T_fk', 'lower', 'q', 'joint_names', 'joint_list']
forward_kinematics = cs.Function("FK",[fk_dict["q"]],[fk_dict["T_fk"]])
print forward_kinematics([0.3, 0.3, 0.3, 0., 0.3, 0.7])
```

## Todo/Implementation status
- [x] Forward kinematics with SE3 matrix
- [ ] Forward kinematics of rotation with quaternion
- [ ] Dynamics from links and their inertia tags
- [ ] Denavit Hartenberg?
