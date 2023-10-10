import numpy as np
import pandas as pd
import albumentations as A


original_height, original_width = 320, 320
desired_height, desired_width = 256, 256


def get_info(target, classifier):
    if target == "chirality":
        assert classifier == True
        class_labels = ["left", "right"]
        class_weights = None
        target_name = "chirality"
        mean_std = None
    else:
        raise NotImplementedError
    image_shape = desired_height, desired_width

    return (
        class_labels,
        class_weights,
        target_name,
        mean_std,
        image_shape,
    )


def get_augmentation_list(subset_name):

    if "train" in subset_name:
        return [
            A.RandomResizedCrop(
                original_height, original_width, scale=(0.8, 1.2), p=0.5
            ),
            A.OneOf(
                [
                    A.ElasticTransform(),
                    A.GridDistortion(),
                ],
                p=0.8,
            ),
            A.Resize(desired_height, desired_width),
        ]
    else:
        return [A.Resize(desired_height, desired_width)]


def other_augmentation(labels, images):
    label, *_ = labels
    flip = bool(np.random.randint(2))
    if flip:
        images = tuple(np.flip(image, axis=-1) for image in images)
    flipped_label = label ^ flip

    return (flipped_label,), images


def get_parameters(target, classifier):
    assert target == "chirality"
    assert classifier

    def label_selector(df):
        return df["is_left_leg"].astype(int)

    def dataframe_filter(x):
        return x

    return (
        get_augmentation_list,
        other_augmentation,
        dataframe_filter,
        label_selector,
    )
