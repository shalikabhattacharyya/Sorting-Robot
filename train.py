"""Stage A, part 2: train the shape classifier.

Reads the crops in shape_dataset/, trains a ResNet-18, and exports
shape_classifier.pkl to the project root. Run generate_data.py first.
"""

from fastai.vision.all import (ImageDataLoaders, Resize, aug_transforms,
                               vision_learner, resnet18, accuracy,
                               load_learner, Path)

from common import DATASET_DIRECTORY, MODEL_PATH


def main():
    dls = ImageDataLoaders.from_folder(
        Path(DATASET_DIRECTORY),
        valid_pct=0.2,                # hold out 20% for validation
        item_tfms=Resize(128),        # uniform 128x128
        batch_tfms=aug_transforms(),  # random flips/rotations = free variety
        seed=42                       # reproducible split
    )
    dls.show_batch(max_n=9)

    learn = vision_learner(dls, resnet18, metrics=accuracy)
    learn.fine_tune(4)

    learn.path = Path('.')            # export to project root, not the dataset folder
    learn.export(MODEL_PATH)
    print(f"saved to {MODEL_PATH}")

    # quick reload check
    learn = load_learner(MODEL_PATH)
    print("vocab:", learn.dls.vocab)


if __name__ == "__main__":
    main()
