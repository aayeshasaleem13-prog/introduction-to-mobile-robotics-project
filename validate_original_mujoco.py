
import numpy as np
import mujoco
from pathlib import Path

from forward_kinematics import fk_pose
from inverse_kinematics import inverse_kinematics_position


# Original SO-101 model
MODEL_PATH = Path(
    r"D:\Robotic Data\Project Work"
    r"\SO-101 Utility + Additional Files-20261022"
    r"\model_files\scene.xml"
)

model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
data = mujoco.MjData(model)


# Five arm joints in the correct order
joint_names = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll"
]

joint_addresses = []

for name in joint_names:
    jid = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, name
    )

    if jid == -1:
        raise ValueError(f"Missing joint: {name}")

    joint_addresses.append(model.jnt_qposadr[jid])


# Locate the original end-effector site
site_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe"
)

if site_id == -1:
    raise ValueError("gripperframe site not found")


# Test five reachable targets
targets = [
    ("Target 1", [30, -20, 45, 10, 60]),
    ("Target 2", [-45, -35, 50, -15, 20]),
    ("Target 3", [55, 25, -35, 30, -40]),
    ("Target 4", [15, -55, 80, -25, 0]),
    ("Target 5", [-75, 15, 30, -20, 45])
]

print("\nORIGINAL SO-101 MUJOCO VALIDATION")
print("-" * 75)

for name, angles in targets:

    q_target = np.deg2rad(angles)

    target_position, _ = fk_pose(q_target)

    q_solution, converged, iterations = (
        inverse_kinematics_position(
            target_position,
            np.zeros(5),
            return_info=True
        )
    )

    if not converged:
        print(name, "IK did not converge")
        continue

    # Reset simulation state
    mujoco.mj_resetData(model, data)

    # Apply solution using named joint addresses
    for i, address in enumerate(joint_addresses):
        data.qpos[address] = q_solution[i]

    mujoco.mj_forward(model, data)

    actual_position = data.site_xpos[site_id].copy()

    error_mm = (
        np.linalg.norm(target_position - actual_position) * 1000
    )

    print(f"\n{name}")
    print("Target position:", np.round(target_position, 5))
    print("MuJoCo position:", np.round(actual_position, 5))
    print("Position error (mm):", round(error_mm, 4))
    print("IK iterations:", iterations)

print("\nValidation completed.")


# ------------------------------------------------
# Unreachable target validation
# ------------------------------------------------

print("\nUNREACHABLE TARGET VALIDATION")

unreachable_target = np.array([0.80, 0.0, 0.35])

q_unreachable, converged, iterations = (
    inverse_kinematics_position(
        unreachable_target,
        np.zeros(5),
        return_info=True
    )
)

mujoco.mj_resetData(model, data)

for i, address in enumerate(joint_addresses):
    data.qpos[address] = q_unreachable[i]

mujoco.mj_forward(model, data)

actual_position = data.site_xpos[site_id].copy()

error_mm = (
    np.linalg.norm(unreachable_target - actual_position) * 1000
)

print("Target:", unreachable_target)
print("MuJoCo position:", np.round(actual_position, 5))
print("Converged:", converged)
print("Iterations:", iterations)
print("Remaining error (mm):", round(error_mm, 2))

