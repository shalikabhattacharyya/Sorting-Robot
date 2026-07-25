"""Stage B: the vision-guided sorting robot.

Loads the trained classifier, spawns a physical scene, photographs and
classifies each object, then sorts each into its labeled bin — using ONLY
the model's prediction as identity. Run generate_data.py and train.py first.
"""

import os
import time
import numpy as np
import pybullet as p
import pybullet_data
import matplotlib.pyplot as plt
from fastai.vision.all import load_learner, PILImage

from common import (CATEGORIES, MODEL_PATH, make_pyramid_obj,
                    random_color, crop_object)

# ============================================================
#  ROBOT-SCALE OBJECTS
#  ~10x smaller than detector scale so the arm can grasp them
# ============================================================
SPHERE_RADIUS, CUBE_HALF, PYRAMID_SCALE = 0.05, 0.05, 0.1
MESH_SCALES = {'duck.obj': 0.075, 'bunny.obj': 0.075}


def mesh_fit(filename, scale, orientation, probe_position=(0, 0, 10)):
    """Measure a mesh: box half-extents + offset to recenter the visual on the origin.
    Fixes ducks wobbling, tipping, and being grabbed off to the side."""
    collision_shape = p.createCollisionShape(p.GEOM_MESH, fileName=filename,
                                             meshScale=[scale] * 3)
    temporary_body = p.createMultiBody(0, collision_shape, -1, list(probe_position), orientation)
    bounding_box_min, bounding_box_max = p.getAABB(temporary_body)
    p.removeBody(temporary_body)

    half_extents = []
    visual_offset = []
    for i in range(3):
        size_on_axis = bounding_box_max[i] - bounding_box_min[i]
        half_extents.append(size_on_axis / 2)
        middle_on_axis = (bounding_box_min[i] + bounding_box_max[i]) / 2
        visual_offset.append(probe_position[i] - middle_on_axis)
    return half_extents, visual_offset


def spawn(kind, position):
    """Spawn a physical object the arm can pick up."""
    color = random_color()
    x, y = position[0], position[1]

    if kind == 'sphere':
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=SPHERE_RADIUS, rgbaColor=color)
        col = p.createCollisionShape(p.GEOM_SPHERE, radius=SPHERE_RADIUS)
        body = p.createMultiBody(0.1, col, vis, [x, y, SPHERE_RADIUS + 0.002])

    elif kind == 'cube':
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[CUBE_HALF] * 3, rgbaColor=color)
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[CUBE_HALF] * 3)
        body = p.createMultiBody(0.1, col, vis, [x, y, CUBE_HALF + 0.002])

    elif kind == 'pyramid':
        vis = p.createVisualShape(p.GEOM_MESH, fileName='pyramid.obj',
                                  meshScale=[PYRAMID_SCALE] * 3, rgbaColor=color)
        col = p.createCollisionShape(p.GEOM_MESH, fileName='pyramid.obj',
                                     meshScale=[PYRAMID_SCALE] * 3)
        body = p.createMultiBody(0.1, col, vis, [x, y, 0.002])

    else:   # duck / bunny — mesh visual, box collision, recentered
        scale = MESH_SCALES[kind]
        degree_90 = p.getQuaternionFromEuler([np.pi / 2, 0, 0])   # they're modeled lying down
        half_extents, visual_offset = mesh_fit(kind, scale, degree_90)
        vis = p.createVisualShape(p.GEOM_MESH, fileName=kind, meshScale=[scale] * 3,
                                  rgbaColor=color, visualFrameOrientation=degree_90,
                                  visualFramePosition=visual_offset)
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
        body = p.createMultiBody(0.1, col, vis, [x, y, half_extents[2] + 0.002])

    # grippy floor friction so nudged objects don't roll away (lowered later in the bin)
    p.changeDynamics(body, -1, lateralFriction=0.8, spinningFriction=0.02,
                     rollingFriction=0.01, restitution=0.0)
    return body


def reachable_spots(n, radius_min=0.38, radius_max=0.65, separation=0.16,
                    y_min=-0.14, tries=12000):
    """Random spots in the reachable ring, clear of the bins and not overlapping."""
    spots = []
    while len(spots) < n and tries > 0:
        tries -= 1
        radius = np.random.uniform(radius_min, radius_max)
        theta = np.random.uniform(0, 2 * np.pi)
        x, y = radius * np.cos(theta), radius * np.sin(theta)   # polar -> cartesian
        if y < y_min:                       # too close to the bin row — reject
            continue
        if all(np.hypot(x - sx, y - sy) >= separation for sx, sy, _ in spots):
            spots.append([x, y, 0.1])
    if len(spots) < n:
        print(f"only placed {len(spots)}/{n}")
    return spots


# ============================================================
#  BINS
# ============================================================
BIN_Y, BIN_WIDTH, BIN_DEPTH = -0.50, 0.26, 0.34
X_POSITIONS = [-0.42, -0.14, 0.14, 0.42]
PANEL_COLORS = {'sphere':  [0.9, 0.1, 0.1, 1], 'cube':    [0.1, 0.8, 0.1, 1],
                'pyramid': [0.1, 0.3, 0.9, 1], 'other':   [0.5, 0.5, 0.5, 1]}


def make_clear_bin(center, label_color, width=0.26, depth=0.34, height=0.24, wall=0.005):
    """Build one clear storage bin: frosted walls plus bright edges."""
    center_x, center_y, center_z = center
    FROST = [0.88, 0.92, 0.96, 0.30]
    EDGE = [0.97, 0.98, 1.00, 0.70]
    FLOOR = [0.86, 0.90, 0.95, 0.50]

    parts = []
    half_width, half_depth, half_height = width / 2, depth / 2, height / 2
    half_wall = wall / 2

    def add(half_extents, position, color, collide=True):
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color)
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents) if collide else -1
        parts.append(p.createMultiBody(0, col, vis, position))

    # floor
    add([half_width, half_depth, half_wall],
        [center_x, center_y, center_z + half_wall], FLOOR)

    # 4 walls
    add([half_width, half_wall, half_height], [center_x, center_y + half_depth, center_z + half_height], FROST)
    add([half_width, half_wall, half_height], [center_x, center_y - half_depth, center_z + half_height], FROST)
    add([half_wall, half_depth, half_height], [center_x + half_width, center_y, center_z + half_height], FROST)
    add([half_wall, half_depth, half_height], [center_x - half_width, center_y, center_z + half_height], FROST)

    # 4 corner posts (cosmetic — the edges are what make it read as plastic)
    post = wall * 0.9
    for sign_x in (+1, -1):
        for sign_y in (+1, -1):
            add([post, post, half_height],
                [center_x + sign_x * half_width, center_y + sign_y * half_depth, center_z + half_height],
                EDGE, collide=False)

    # top rim (cosmetic)
    rim = wall * 1.2
    add([half_width, post, rim], [center_x, center_y + half_depth, center_z + height - rim], EDGE, collide=False)
    add([half_width, post, rim], [center_x, center_y - half_depth, center_z + height - rim], EDGE, collide=False)
    add([post, half_depth, rim], [center_x + half_width, center_y, center_z + height - rim], EDGE, collide=False)
    add([post, half_depth, rim], [center_x - half_width, center_y, center_z + height - rim], EDGE, collide=False)

    # colored label panel on the front face
    label_half_extents = [width / 3, wall * 0.4, height / 10]
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=label_half_extents, rgbaColor=label_color)
    parts.append(p.createMultiBody(0, -1, vis,
                 [center_x, center_y - half_depth - wall, center_z + height * 0.75]))
    return parts


# ============================================================
#  MODULE-LEVEL STATE
#  Set up inside main(); the motion helpers read these globals.
# ============================================================
robot = None
movable = []
EE_LINK = None
DOWN = None
HOME_UP = [0, 0, 0, 0, 0, 0, 0]
LOWER_LIMITS = [-2.97, -2.09, -2.97, -2.09, -2.97, -2.09, -3.05]
UPPER_LIMITS = [2.97, 2.09, 2.97, 2.09, 2.97, 2.09, 3.05]
REST_POSE = [0, 0, 0, -np.pi / 2, 0, np.pi / 2, 0]
JOINT_RANGES = [u - l for l, u in zip(LOWER_LIMITS, UPPER_LIMITS)]
JOINT_DAMPING = None

SPEED = 0.8          # not too slow, not too fast
MAX_REACH = 0.78     # the KUKA's reach limit
TRAVEL_Z = 0.35      # hover height above an object
SAFE_Z = 0.45        # horizontal travel height, above everything on the table
DROP_Z = 0.45        # release height over a bin
GRAB_GAP = 0.02      # gap between flange and object top
ACCEPT = 0.05        # tip must land within this, else don't grasp


# ============================================================
#  MOTION HELPERS
# ============================================================
def move_to_config(goal, steps=60, speed=SPEED):
    """Drive the joints to `goal` with sinusoidal easing (no jerking)."""
    start_angles = [p.getJointState(robot, joint)[0] for joint in movable]
    num_steps = max(2, int(steps / speed))     # at least 2 steps
    for step in range(1, num_steps + 1):
        ease = 0.5 * (1 - np.cos(np.pi * step / num_steps))   # 0 -> 1 S-curve
        blended = [start + (target - start) * ease
                   for start, target in zip(start_angles, goal)]
        p.setJointMotorControlArray(robot, movable, p.POSITION_CONTROL,
                                    targetPositions=blended,
                                    forces=[200] * len(movable))
        p.stepSimulation()
        time.sleep(1 / 480)


def inverse_kinematics(position, orientation=None):
    """Joint angles that put the tip at `position` (null-space IK)."""
    if orientation is None:
        orientation = DOWN
    solver_args = (robot, EE_LINK, position, orientation)
    return list(p.calculateInverseKinematics(
        *solver_args, lowerLimits=LOWER_LIMITS, upperLimits=UPPER_LIMITS,
        jointRanges=JOINT_RANGES, restPoses=REST_POSE,
        jointDamping=JOINT_DAMPING, maxNumIterations=200, residualThreshold=0.0001))


def move_ee_to(position, orientation=None, steps=60):
    """Move the end-effector to a position, rejecting anything past MAX_REACH."""
    if orientation is None:
        orientation = DOWN
    if np.hypot(position[0], position[1]) > MAX_REACH:
        rounded = tuple(round(v, 2) for v in position)
        print(f"    ! {rounded} out of reach — skipping")
        return False
    move_to_config(inverse_kinematics(position, orientation), steps=steps)
    return True


def grasp(obj_id):
    """Constraint-based 'suction': weld the object to the flange at its current pose."""
    ee_position, ee_orientation = p.getLinkState(robot, EE_LINK,
                                                 computeForwardKinematics=True)[4:6]
    obj_position, obj_orientation = p.getBasePositionAndOrientation(obj_id)
    inverse = p.invertTransform(ee_position, ee_orientation)
    relative_position, relative_orientation = p.multiplyTransforms(
        inverse[0], inverse[1], obj_position, obj_orientation)
    constraint_id = p.createConstraint(robot, EE_LINK, obj_id, -1, p.JOINT_FIXED,
                                       [0, 0, 0], relative_position,
                                       [0, 0, 0], relative_orientation, [0, 0, 0, 1])
    p.changeConstraint(constraint_id, maxForce=80)
    return constraint_id


def obj_center(obj_id):
    """Visible center + top from the bounding box. ALWAYS use this, never the base
    position — mesh origins sit off to the side of the visible geometry."""
    bounding_box_min, bounding_box_max = p.getAABB(obj_id)
    return ((bounding_box_min[0] + bounding_box_max[0]) / 2,
            (bounding_box_min[1] + bounding_box_max[1]) / 2,
            bounding_box_max[2])


def settle(frames=30):
    for _ in range(frames):
        p.stepSimulation()
        time.sleep(1 / 480)


def solve_and_check(target_x, target_y, z, orientation):
    """Move there, then report how far the tip ACTUALLY landed — the honesty check."""
    move_ee_to([target_x, target_y, z], orientation=orientation)
    ee = p.getLinkState(robot, EE_LINK)[4]
    return np.hypot(ee[0] - target_x, ee[1] - target_y)


# ============================================================
#  PICK
# ============================================================
def attempt_grab(center_x, center_y, top_z):
    """One approach: straight down, then lean the wrist if that missed."""
    offset = solve_and_check(center_x, center_y, top_z + GRAB_GAP, DOWN)
    if offset > ACCEPT:
        reach_direction = np.arctan2(center_y, center_x)
        for tilt in (0.4, 0.8, 1.2):
            move_ee_to([center_x, center_y, TRAVEL_Z])
            offset = solve_and_check(center_x, center_y, top_z + GRAB_GAP,
                                     p.getQuaternionFromEuler([np.pi, tilt, reach_direction]))
            if offset <= ACCEPT:
                break
    return offset


def pick(obj_id):
    """Grasp an object, or return None if it can't be reached cleanly."""
    center_x, center_y, top_z = obj_center(obj_id)
    move_ee_to([center_x, center_y, TRAVEL_Z])

    offset = attempt_grab(center_x, center_y, top_z)

    # missed — reset to upright and re-approach cleanly
    if offset > ACCEPT:
        print(f"   tip {offset*100:.0f} cm off — resetting to home and re-approaching")
        move_to_config(HOME_UP, speed=0.4)
        center_x, center_y, top_z = obj_center(obj_id)   # object may have shifted
        move_ee_to([center_x, center_y, TRAVEL_Z])
        offset = attempt_grab(center_x, center_y, top_z)

    # CRITICAL: without this guard, grasp() pins a distant object and drags it around
    if offset > ACCEPT:
        print(f"   tip {offset*100:.0f} cm from object — refusing (would fling it)")
        move_to_config(HOME_UP, speed=0.4)
        return None

    constraint_id = grasp(obj_id)
    move_ee_to([center_x, center_y, TRAVEL_Z])
    return constraint_id


# ============================================================
#  DETECT
# ============================================================
def detect(bodies, learn):
    """Photograph each object, classify it. Returns predictions {body_id: label}."""
    # park the arm straight up so it's never between camera and object
    for _ in range(120):
        p.setJointMotorControlArray(robot, movable, p.POSITION_CONTROL,
                                    targetPositions=[0] * len(movable),
                                    forces=[200] * len(movable))
        p.stepSimulation()

    projection_matrix = p.computeProjectionMatrixFOV(fov=60, aspect=1.0, nearVal=0.01, farVal=5)
    predictions = {}
    results = []

    for body_id in bodies:
        bounding_box_min, bounding_box_max = p.getAABB(body_id)
        center = [(bounding_box_min[i] + bounding_box_max[i]) / 2 for i in range(3)]
        center_x, center_y, center_z = center

        # camera outside the object looking inward and down at 45 deg (matches training pitch -45)
        horizontal_distance = np.hypot(center_x, center_y)
        if horizontal_distance > 1e-6:
            unit_x, unit_y = center_x / horizontal_distance, center_y / horizontal_distance
        else:
            unit_x, unit_y = 1.0, 0.0

        camera_offset = 0.5
        eye = [center_x + unit_x * camera_offset,
               center_y + unit_y * camera_offset,
               center_z + camera_offset]
        view_matrix = p.computeViewMatrix(eye, [center_x, center_y, center_z], [0, 0, 1])

        _, _, rgb, _, seg = p.getCameraImage(768, 768, viewMatrix=view_matrix,
                                             projectionMatrix=projection_matrix)
        rgb_img = np.reshape(rgb, (768, 768, 4))[:, :, :3].astype(np.uint8)
        seg_mask = np.reshape(seg, (768, 768))

        crop = crop_object(rgb_img, seg_mask, body_id)
        if crop is None:
            print(f"   id {body_id}: couldn't frame it")
            continue

        prediction, class_index, probabilities = learn.predict(PILImage.create(crop))
        predictions[body_id] = str(prediction)
        results.append((crop, str(prediction), float(probabilities[class_index])))
        print(f"   id {body_id}: model says '{prediction}' ({probabilities[class_index]:.0%})")

    print(f"\nClassified {len(predictions)} objects.")

    # show the crops with predictions (no answer key)
    if results:
        fig, axes = plt.subplots(1, len(results), figsize=(3 * len(results), 3.2))
        if len(results) == 1:
            axes = [axes]
        for ax, (crop, prediction, confidence) in zip(axes, results):
            ax.imshow(crop); ax.axis('off')
            ax.set_title(f"{prediction} ({confidence:.0%})", fontsize=10)
        plt.tight_layout(); plt.show()

    return predictions


# ============================================================
#  SORT
# ============================================================
def sweep_key(body_id):
    """Order picks by angle around the base, then distance — a smooth sweep."""
    center_x, center_y, _ = obj_center(body_id)
    return (round(np.arctan2(center_y, center_x), 1), np.hypot(center_x, center_y))


def find_unscored(bodies, predictions, bin_centers):
    """Check the ground: which objects are loose or in the wrong bin?
    Uses the rectangular footprint — distance-from-center wrongly rejects corners."""
    half_width, half_depth = BIN_WIDTH / 2, BIN_DEPTH / 2
    problems = []
    for body_id in bodies:
        x, y, _ = p.getBasePositionAndOrientation(body_id)[0]
        prediction = predictions.get(body_id)

        which_bin = None
        for category, (bin_x, bin_y, _) in bin_centers.items():
            if abs(x - bin_x) < half_width and abs(y - bin_y) < half_depth:
                which_bin = category
                break

        if which_bin is None:
            print(f"   id {body_id} not in any bin, at ({x:.2f}, {y:.2f})")
            problems.append(body_id)
        elif which_bin != prediction:
            print(f"   id {body_id} in WRONG bin ({which_bin}, should be {prediction})")
            problems.append(body_id)
    return problems


def try_sort(body_id, predictions, bin_centers):
    """Pick one object and drop it in its predicted bin. True if it landed inside."""
    prediction = predictions.get(body_id)
    if prediction is None:
        print(f"   id {body_id}: no prediction, skipping")
        return False

    pickup_x, pickup_y, _ = obj_center(body_id)      # pickup spot, before grabbing
    constraint_id = pick(body_id)
    if constraint_id is None:
        return False

    bin_x, bin_y, _ = bin_centers[prediction]

    # scatter the drop point so objects spread out instead of stacking into towers
    drop_x = bin_x + np.random.uniform(-0.09, 0.09)
    drop_y = bin_y + np.random.uniform(-0.10, 0.05)

    # tall objects (ducks) dangle below the flange — raise carry height by object height
    bounding_box_min, bounding_box_max = p.getAABB(body_id)
    object_height = bounding_box_max[2] - bounding_box_min[2]
    carry_z = SAFE_Z + object_height

    move_ee_to([pickup_x, pickup_y, carry_z])            # up
    move_ee_to([drop_x, drop_y, carry_z])                # over, above the walls
    offset = solve_and_check(drop_x, drop_y, DROP_Z, DOWN)   # down into the bin

    # reset instead of tilting — tilting next to the bins knocks neighbors out
    if offset > 0.06:
        print(f"   drop {offset*100:.0f} cm off — resetting to upright and re-approaching")
        move_to_config(HOME_UP, speed=0.4)
        move_ee_to([drop_x, drop_y, carry_z])
        offset = solve_and_check(drop_x, drop_y, DROP_Z, DOWN)

    settle(40)                                    # STOP before releasing, or it flings
    p.removeConstraint(constraint_id)
    p.changeDynamics(body_id, -1, lateralFriction=0.2, spinningFriction=0.002,
                     rollingFriction=0.001)       # looser bin friction, settles naturally
    settle(20)
    move_ee_to([drop_x, drop_y, carry_z])

    # judge by where the OBJECT landed, not the flange — they can differ
    object_x, object_y, _ = p.getBasePositionAndOrientation(body_id)[0]
    in_bin = abs(object_x - bin_x) < BIN_WIDTH / 2 and abs(object_y - bin_y) < BIN_DEPTH / 2
    if not in_bin:
        print(f"   ! id {body_id} landed outside the bin — missed")
        return False
    print(f"   id {body_id} -> {prediction} bin")
    return True


def sort_all(bodies, predictions, bin_centers):
    """Sweep pass, retry pass, then up to 3 verification rounds."""
    move_to_config(REST_POSE)                     # start from a tidy ready pose

    # first pass, in sweep order
    skipped = []
    for body_id in sorted(bodies, key=sweep_key):
        print(f"-> Pick {body_id}")
        if not try_sort(body_id, predictions, bin_centers):
            skipped.append(body_id)

    # retry anything skipped
    if skipped:
        print(f"\nRetrying {len(skipped)} skipped: {skipped}")
        for body_id in skipped:
            print(f"-> Retry {body_id}")
            try_sort(body_id, predictions, bin_centers)

    # verify: settle, check, fix, repeat
    for attempt in range(3):
        settle(120)
        problems = find_unscored(bodies, predictions, bin_centers)
        if not problems:
            print("\nAll objects verified in correct bins")
            break
        print(f"\nattempt {attempt+1}: {len(problems)} to fix: {problems}")
        for body_id in problems:
            try_sort(body_id, predictions, bin_centers)
    else:
        print(f"\nStill unresolved after 3 rounds: "
              f"{find_unscored(bodies, predictions, bin_centers)}")

    print("\nDone!")


# ============================================================
#  MAIN
# ============================================================
def main():
    global robot, movable, EE_LINK, DOWN, JOINT_DAMPING

    if not os.path.exists(MODEL_PATH):
        print(f"! {MODEL_PATH} not found — run generate_data.py then train.py first")
        return

    make_pyramid_obj()

    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, -9.8)
    p.loadURDF("plane.urdf")

    robot = p.loadURDF("kuka_iiwa/model.urdf", basePosition=[0, 0, 0], useFixedBase=True)
    print(f"KUKA arm loaded (id {robot}), {p.getNumJoints(robot)} joints")

    # movable joints, end-effector, orientations, damping
    movable = [j for j in range(p.getNumJoints(robot))
               if p.getJointInfo(robot, j)[2] != p.JOINT_FIXED]
    EE_LINK = movable[-1]
    DOWN = p.getQuaternionFromEuler([0, np.pi, 0])
    JOINT_DAMPING = [0.08] * len(movable)

    # build the object pool: at most MAX_PER_KIND of each category
    N_OBJECTS, MAX_PER_KIND = 12, 4
    pool = []
    for category in CATEGORIES:
        pool += [category] * MAX_PER_KIND
    np.random.shuffle(pool)
    chosen = []
    for category in pool[:N_OBJECTS]:
        if category != 'other':
            chosen.append(category)
        else:
            chosen.append(np.random.choice(['duck.obj', 'bunny.obj']))

    spots = reachable_spots(N_OBJECTS)

    # spawn objects — only IDs stored, no labels (the model decides identity)
    bodies = []
    for kind, position in zip(chosen, spots):
        body_id = spawn(kind, position)
        bodies.append(body_id)
        print(f"spawned id {body_id} at ({position[0]:.2f}, {position[1]:.2f})")

    # build the four bins
    bin_centers = {}
    for category, x in zip(PANEL_COLORS, X_POSITIONS):
        label_color = PANEL_COLORS[category]
        make_clear_bin([x, BIN_Y, -0.003], label_color,
                       width=BIN_WIDTH, depth=BIN_DEPTH, height=0.24)
        bin_centers[category] = [x, BIN_Y, 0]
    print("bin_centers:", {c: [round(v, 2) for v in xyz]
                           for c, xyz in bin_centers.items()})

    for _ in range(240):          # settle before the arm moves
        p.stepSimulation()
        time.sleep(1 / 480)
    print("Scene ready")

    # detect, then sort
    learn = load_learner(MODEL_PATH)
    predictions = detect(bodies, learn)
    sort_all(bodies, predictions, bin_centers)

    input("\nPress Enter to close the simulation...")
    p.disconnect()


if __name__ == "__main__":
    main()
