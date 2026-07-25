"""Shared constants and helpers used by more than one script.

Defined once here so training and the robot run can't drift apart —
crop_object in particular MUST be identical between data generation and
detection, or the classifier gets confident wrong answers.
"""

import os
import numpy as np
import pybullet as p
import pybullet_data
import matplotlib.pyplot as plt

# ---- paths / categories ----
DATASET_DIRECTORY = 'shape_dataset'
MODEL_PATH = 'shape_classifier.pkl'      # project root, NOT inside the dataset folder
CATEGORIES = ['sphere', 'cube', 'pyramid', 'other']
KINDS = ['sphere', 'cube', 'pyramid', 'duck.obj', 'bunny.obj']

# 3x3 grid of spawn spots with the center left out
GRID_SPOTS = []
for _x in [-2.5, 0, 2.5]:
    for _y in [-2.5, 0, 2.5]:
        if not (_x == 0 and _y == 0):
            GRID_SPOTS.append([_x, _y, 0.5])


def make_pyramid_obj():
    """Write pyramid.obj if it's missing (PyBullet has no built-in pyramid)."""
    if not os.path.exists('pyramid.obj'):
        with open('pyramid.obj', 'w') as f:
            f.write("# pyramid\nv -0.5 -0.5 0.0\nv 0.5 -0.5 0.0\nv 0.5 0.5 0.0\n"
                    "v -0.5 0.5 0.0\nv 0.0 0.0 1.0\nf 1 2 5\nf 2 3 5\nf 3 4 5\n"
                    "f 4 1 5\nf 1 4 3\nf 1 3 2\n")


def random_color():
    """Random opaque RGBA — stops the model learning 'cubes are green'."""
    return [np.random.rand(), np.random.rand(), np.random.rand(), 1]


def category_of(kind):
    """Map a spawn kind to its training category (duck/bunny -> 'other')."""
    return kind if kind in ['sphere', 'cube', 'pyramid'] else 'other'


def start_headless():
    """Connect PyBullet with no window and load the ground plane."""
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")


def photograph(yaw):
    """Render the Stage-A camera: angled-from-above (pitch -45) at the given yaw.
    Returns (rgb_img, seg_mask). Detection must reproduce this geometry."""
    view = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=[0, 0, 0.5], distance=12.0,
        yaw=yaw, pitch=-45, roll=0, upAxisIndex=2)
    proj = p.computeProjectionMatrixFOV(fov=60, aspect=1.0, nearVal=0.1, farVal=30)
    w, h, rgb, _, seg = p.getCameraImage(768, 768, viewMatrix=view, projectionMatrix=proj)
    rgb_img = np.reshape(rgb, (h, w, 4))[:, :, :3].astype(np.uint8)
    seg_mask = np.reshape(seg, (h, w))
    return rgb_img, seg_mask


def show_scene_and_mask(rgb_img, seg_mask):
    """Draw the scene and its segmentation mask side by side."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 6))
    a1.imshow(rgb_img);  a1.set_title("Scene (RGB)");       a1.axis('off')
    a2.imshow(seg_mask); a2.set_title("Segmentation mask"); a2.axis('off')
    plt.show()


def crop_object(rgb, mask, body_id, pad_ratio=0.6):
    """Cut one object out of a rendered scene onto a gray square.

    Used by BOTH training and detection — these must stay identical.
    Returns a square crop of just this object, or None if not visible.
    """
    rows, cols = np.where(mask == body_id)      # pixel coordinates of this object
    if len(cols) == 0:
        return None                             # object not visible

    clean = np.full_like(rgb, 200)                      # gray canvas
    clean[mask == body_id] = rgb[mask == body_id]       # copy back only this object's pixels

    left, right = cols.min(), cols.max()        # bounding box edges
    top, bottom = rows.min(), rows.max()
    object_width = right - left
    object_height = bottom - top

    pad_x = int(object_width * pad_ratio)       # pad by a % of the object's size
    pad_y = int(object_height * pad_ratio)
    left = max(left - pad_x, 0)                  # expand box, clamped to image edges
    right = min(right + pad_x, rgb.shape[1])
    top = max(top - pad_y, 0)
    bottom = min(bottom + pad_y, rgb.shape[0])
    crop = clean[top:bottom, left:right]

    crop_height, crop_width = crop.shape[:2]    # pad to a square so the resize doesn't stretch
    size = max(crop_height, crop_width)
    square = np.full((size, size, 3), 200, dtype=np.uint8)
    offset_y = (size - crop_height) // 2        # offsets to center the crop
    offset_x = (size - crop_width) // 2
    square[offset_y:offset_y + crop_height, offset_x:offset_x + crop_width] = crop
    return square
