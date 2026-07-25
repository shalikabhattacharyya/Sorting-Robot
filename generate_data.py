"""Stage A, part 1: generate synthetic training data.

Renders 200 random scenes of 3D shapes, crops each object, and saves the
crops into shape_dataset/<category>/. Run this once before train.py.
"""

import os
import shutil
import numpy as np
import pybullet as p
from PIL import Image

from common import (DATASET_DIRECTORY, CATEGORIES, GRID_SPOTS,
                    make_pyramid_obj, random_color, category_of,
                    start_headless, photograph, crop_object)

SCENES = 200

# detector-scale sizes — large, for clear photos
DET_SPHERE_RADIUS, DET_CUBE_HALF, DET_PYRAMID_SCALE = 0.5, 0.4, 0.7
DET_MESH_SCALES = {'duck.obj': 0.8, 'bunny.obj': 0.8}


def spawn_for_dataset(kind, position):
    """Spawn a static (massless) object just to photograph it."""
    color = random_color()
    orientation = p.getQuaternionFromEuler([0, 0, 0])   # euler -> quaternion

    if kind == 'sphere':
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=DET_SPHERE_RADIUS, rgbaColor=color)
    elif kind == 'cube':
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[DET_CUBE_HALF] * 3, rgbaColor=color)
    elif kind == 'pyramid':
        vis = p.createVisualShape(p.GEOM_MESH, fileName='pyramid.obj',
                                  meshScale=[DET_PYRAMID_SCALE] * 3, rgbaColor=color)
    else:
        vis = p.createVisualShape(p.GEOM_MESH, fileName=kind,
                                  meshScale=[DET_MESH_SCALES[kind]] * 3, rgbaColor=color)
        orientation = p.getQuaternionFromEuler([np.pi / 2, 0, 0])   # stand mesh upright

    return p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis,
                             basePosition=position, baseOrientation=orientation)


def pick_kind():
    """Pick a CATEGORY first (equal odds) so 'other' doesn't get 40% and starve pyramids."""
    category = np.random.choice(CATEGORIES)
    if category == 'other':
        return 'duck.obj' if np.random.rand() < 0.5 else 'bunny.obj'
    return category


def main():
    make_pyramid_obj()

    # wipe and rebuild the dataset folder — the model lives elsewhere, so it's safe
    if os.path.exists(DATASET_DIRECTORY):
        shutil.rmtree(DATASET_DIRECTORY)
    for category in CATEGORIES:
        os.makedirs(os.path.join(DATASET_DIRECTORY, category), exist_ok=True)

    start_headless()

    counts = {c: 0 for c in CATEGORIES}          # images saved per category
    spawn_positions = GRID_SPOTS.copy()          # copy — shuffled in place

    for s in range(SCENES):
        np.random.shuffle(spawn_positions)
        n = np.random.randint(3, 6)              # 3-5 objects this scene
        scene_objects = []
        for pos in spawn_positions[:n]:
            kind = pick_kind()
            body = spawn_for_dataset(kind, pos)
            scene_objects.append((body, category_of(kind)))

        # random yaw so the model learns every rotation
        rgb_img, seg_mask = photograph(np.random.uniform(0, 360))

        for body, category in scene_objects:
            crop = crop_object(rgb_img, seg_mask, body)
            if crop is not None:
                Image.fromarray(crop).save(
                    os.path.join(DATASET_DIRECTORY, category,
                                 f'{category}_{counts[category]}.png'))
                counts[category] += 1
            p.removeBody(body)

        if (s + 1) % 25 == 0:
            print(f"scene {s+1}/{SCENES} — counts: {counts}")

    p.disconnect()
    print(f"\nDone — {counts}")


if __name__ == "__main__":
    main()
