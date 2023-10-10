import numpy as np
import pandas as pd
import cv2
import albumentations as A


original_height, original_width = 240, 176
desired_height, desired_width = 256, 256


def get_info(target, classifier):

    if "alzheimers" in target:
        assert classifier
        if target == "alzheimers":
            class_labels = ["CN", "MCI", "AD"]
            class_weights = [0.626714, 0.924362, 3.100295]
        elif target == "alzheimers_binary":
            class_labels = ["CN", "AD"]
            class_weights = [0.626714, 3.100295]

        target_name = "diagnosis"
        mean_std = None

    elif target == "age":
        assert not classifier
        class_labels = ["age"]
        class_weights = None
        target_name = "age"
        mean_std = 75, 10

    elif target == "brain_vol":
        assert not classifier
        class_labels = ["brain_vol"]
        class_weights = None
        target_name = "vol"
        mean_std = 1000, 100

    elif target == "sex":
        assert classifier
        class_labels = ["F", "M"]
        class_weights = None
        target_name = "sex"
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
            A.PadIfNeeded(
                min_height=desired_height,
                min_width=desired_width,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=1,
            ),
            A.Rotate(10),
            A.HorizontalFlip(),
        ]
    else:
        return [
            A.PadIfNeeded(
                min_height=desired_height,
                min_width=desired_width,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=1,
            ),
        ]


def other_augmentation(labels, images):
    label, *_ = labels
    return (label,), images


def get_parameters(target, classifier):

    if "alzheimers" in target:
        assert classifier
        if target == "alzheimers":
            class_labels = ["CN", "MCI", "AD"]
        elif target == "alzheimers_binary":
            class_labels = ["CN", "AD"]

        def label_selector(df):
            categorical_column = pd.Categorical(
                df["diagnosis_cleaned"], class_labels, ordered=True
            )
            return pd.Series(categorical_column.codes, index=df.index)

    elif target == "age":

        def label_selector(df):
            return df["Age"]

    elif target == "brain_vol":

        def label_selector(df):
            return df["BRAINVOL"]

    elif target == "sex":
        class_labels = ["F", "M"]

        def label_selector(df):
            categorical_column = pd.Categorical(df["Sex"], class_labels)
            return pd.Series(categorical_column.codes, index=df.index)

    else:
        raise NotImplementedError

    def get_single_space_indices(df):
        return np.isclose(
            df[["spacing_x", "spacing_y", "spacing_z"]],
            1,
            rtol=1e-3,
            atol=1e-3,
        ).all(1)

    if target == "alzheimers_binary":

        def get_clear_alzheimers_separation_indices(df):
            return df["diagnosis_cleaned"].isin({"CN", "AD"})

        def get_filtered_boolean_indices(df):
            return get_single_space_indices(
                df
            ) & get_clear_alzheimers_separation_indices(df)

    else:
        get_filtered_boolean_indices = get_single_space_indices

    def dataframe_filter(df):
        before = len(df)
        df = df.loc[get_filtered_boolean_indices(df)]
        print(f"filtering before: {before} after: {len(df)}")
        return df

    return (
        get_augmentation_list,
        other_augmentation,
        dataframe_filter,
        label_selector,
    )
