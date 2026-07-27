"""Stage A, part 1: generate synthetic training data.

Renders random scenes of 3D shapes, crops each object, and saves the
crops into shape_dataset/<category>/. Run this once before train.py.

Usage:
    python generate_data.py                 # default 200 scenes, random data
    python generate_data.py --scenes 20     # quick test run
    python generate_data.py --seed 42       # reproducible dataset
"""

import os
import shutil
import argparse
import numpy as np
import pybullet as p
from PIL import Image

from common import (DATASET_DIRECTORY, CATEGORIES, GRID_SPOTS,
                    make_pyramid_obj, random_color, category_of,
                    start_headless, photograph, crop_object)

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


def main(scenes=200, seed=None):
    if seed is not None:
        np.random.seed(seed)                     # reproducible dataset
        print(f"random seed set to {seed}")

    make_pyramid_obj()

    # wipe and rebuild the dataset folder — the model lives elsewhere, so it's safe
    if os.path.exists(DATASET_DIRECTORY):
        shutil.rmtree(DATASET_DIRECTORY)
    for category in CATEGORIES:
        os.makedirs(os.path.join(DATASET_DIRECTORY, category), exist_ok=True)

    start_headless()

    counts = {c: 0 for c in CATEGORIES}          # images saved per category
    spawn_positions = GRID_SPOTS.copy()          # copy — shuffled in place

    for s in range(scenes):
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
            print(f"scene {s+1}/{scenes} — counts: {counts}")

    p.disconnect()
    print(f"\nDone — {counts}")

    # sanity check: warn if any category came out too small to train well
    total = sum(counts.values())
    if total == 0:
        print("! WARNING: no images were saved — check the crop/camera setup.")
    else:
        for category, count in counts.items():
            if count < max(10, total * 0.1):
                print(f"! WARNING: '{category}' has only {count} images "
                      f"({count/total:.0%}) — training may be unbalanced.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic shape training data.")
    parser.add_argument('--scenes', type=int, default=200,
                        help="number of scenes to render (default 200)")
    parser.add_argument('--seed', type=int, default=None,
                        help="random seed for a reproducible dataset")
    args = parser.parse_args()
    main(scenes=args.scenes, seed=args.seed)
