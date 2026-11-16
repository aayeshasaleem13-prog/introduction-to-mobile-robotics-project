
import time
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer

from forward_kinematics import fk_pose
from inverse_kinematics import inverse_kinematics_position


# ============================================================
# 1. LOAD THE ORIGINAL SO-101 MUJOCO MODEL
# ============================================================

MODEL_PATH = Path(
    r"D:\Robotic Data\Project Work"
    r"\SO-101 Utility + Additional Files-20261022"
    r"\model_files\scene.xml"
)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"MuJoCo XML file not found: {MODEL_PATH}"
    )

model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
data = mujoco.MjData(model)


# ============================================================
# 2. IDENTIFY THE ROBOT JOINTS
# ============================================================

joint_names = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]

joint_addresses = []

for name in joint_names:
    joint_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        name
    )

    if joint_id == -1:
        raise ValueError(
            f"Joint '{name}' was not found in the original model."
        )

    joint_addresses.append(
        model.jnt_qposadr[joint_id]
    )

# Set gripper to its neutral position if present.
gripper_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_JOINT,
    "gripper"
)

if gripper_id != -1:
    data.qpos[model.jnt_qposadr[gripper_id]] = 0.0


# ============================================================
# 3. CALCULATE A REACHABLE CARTESIAN TARGET
# ============================================================

target_joint_angles = np.deg2rad(
    [30, -20, 45, 10, 60]
)

target_position, target_rotation = fk_pose(
    target_joint_angles
)


# ============================================================
# 4. SOLVE INVERSE KINEMATICS
# ============================================================

q_initial = np.zeros(5)

q_solution, converged, iterations = (
    inverse_kinematics_position(
        target_position,
        q_initial,
        return_info=True
    )
)

if not converged:
    raise RuntimeError(
        "Inverse kinematics did not converge."
    )


# ============================================================
# 5. APPLY THE SOLUTION TO THE ORIGINAL ROBOT
# ============================================================

for i, address in enumerate(joint_addresses):
    data.qpos[address] = q_solution[i]

mujoco.mj_forward(model, data)


# ============================================================
# 6. FIND THE END-EFFECTOR SITE
# ============================================================

site_id = -1

for site_name in ["gripperframe", "end_effector"]:
    site_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        site_name
    )

    if site_id != -1:
        print("End-effector site:", site_name)
        break

if site_id == -1:
    raise ValueError(
        "Could not find gripperframe or end_effector site."
    )

mujoco_position = data.site_xpos[site_id].copy()

position_error = (
    np.linalg.norm(target_position - mujoco_position) * 1000
)


# ============================================================
# 7. DISPLAY VALIDATION INFORMATION
# ============================================================

print("\n========================================")
print("SO-101 INVERSE KINEMATICS VISUALIZATION")
print("========================================")

print("Target position (m):",
      np.round(target_position, 5))

print("MuJoCo end-effector position (m):",
      np.round(mujoco_position, 5))

print("Solved joint angles (rad):",
      np.round(q_solution, 5))

print("IK iterations:", iterations)

print("Position difference (mm):",
      round(position_error, 4))

print("Robot geometry count:", model.ngeom)
print("Robot mesh count:", model.nmesh)


# ============================================================
# 8. OPEN THE ORIGINAL ROBOT IN THE GRAPHICAL VIEWER
# ============================================================

print("\nOpening original SO-101 MuJoCo model...")
print("Use the Screenshot button to capture the simulation.")

with mujoco.viewer.launch_passive(model, data) as viewer:

    # Configure camera
    mujoco.mjv_defaultFreeCamera(model, viewer.cam)

    viewer.cam.lookat[:] = [0.15, 0.0, 0.12]
    viewer.cam.distance = 0.8
    viewer.cam.azimuth = 135
    viewer.cam.elevation = -20

    # Display all available geometry groups
    viewer.opt.geomgroup[:] = 1

    while viewer.is_running():
        viewer.sync()
        time.sleep(0.02)

print("MuJoCo viewer closed.")
