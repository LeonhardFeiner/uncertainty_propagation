from pathlib import Path
from operator import itemgetter
from contextlib import ExitStack
import numpy as np
import pandas as pd
import h5py
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import (
    BatchSampler,
    RandomSampler,
    SequentialSampler,
)
import albumentations as A
from jax import tree_util
from data.common import (
    get_generator_with_jax_rng_key,
    singleton_select_collate_fn,
    tree_collate_function,
)


class CommonDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df
        self.transform = transform

    def __getitem__(self, index):
        data = self.df.iloc[index]
        return self.transform(data)

    def __len__(self):
        return len(self.df)


def open_numpy(file_path):
    return np.load(file_path, mmap_mode="r")


def open_hdf5(file_path):
    return h5py.File(file_path, "r")


def get_image_getter(opener, file_selector, slice_selector, keys, array_changer):
    def get_image(file_path_list, labels):
        file_path = file_selector(file_path_list)
        with opener(file_path) as file:
            volumes = [file[key] for key in keys]
            slices = slice_selector(volumes, labels)
            kwargs = {key: single_slice for key, single_slice in zip(keys, slices)}
            return array_changer(**kwargs)

    return get_image


def get_image_list_getter(opener, slice_selector, keys, array_changer):
    def get_images(file_path_list, labels):
        with ExitStack() as stack:
            files = [
                stack.enter_context(opener(file_path)) for file_path in file_path_list
            ]
            volumes = [[file[key] for key in keys] for file in files]

            slices_list = slice_selector(volumes, labels)
            return [
                array_changer(
                    **{key: single_slice for key, single_slice in zip(keys, slices)}
                )
                for slices in slices_list
            ]

    return get_images


def get_image_getter_wrapper(
    get_images, input_modifier_function_dict, file_selector=None
):
    def inner_get_images(file_path_list, labels):
        tuple_list = get_images(file_path_list, labels)

        replacement_images = {
            key: image_function(
                np.stack([image_dict[key] for image_dict in tuple_list])
            )
            for key, image_function in input_modifier_function_dict.items()
        }

        def get_new_tuple(i, single_tuple):
            single_list = list(single_tuple)
            for key, new_array in replacement_images.items():
                single_list[key] = new_array[i]

            return tuple(single_list)

        new_tuple_list = [
            get_new_tuple(i, single_tuple) for i, single_tuple in enumerate(tuple_list)
        ]

        if file_selector is None:
            return new_tuple_list
        else:
            return new_tuple_list[file_selector(np.arange(len(new_tuple_list)))]

    return inner_get_images


def slice_selector_info_selector(index_selector, *args_indices, **kwargs_indices):
    def slice_selector(volumes_tree, labels):

        index_args = (labels[index] for index in args_indices)
        index_kwargs = {key: labels[index] for key, index in kwargs_indices.items()}

        num_slices = tree_util.tree_leaves(volumes_tree)[0].shape[0]

        index = index_selector(num_slices, *index_args, **index_kwargs)

        return tree_util.tree_map(itemgetter(index), volumes_tree)

    return slice_selector


def middle_slice_index_getter(num_slices):
    return num_slices // 2


def index_slice_index_getter(num_slices, index):
    return index


def relative_position_slice_index_getter(num_slices, relative_position):
    return int(np.rint(num_slices * relative_position))


def get_random_slice_index_getter(a, b):
    def random_slice_index(num_slices):
        possible_slices = np.arange(num_slices)
        p = get_slice_weight(len(possible_slices), a, b)
        return np.random.choice(possible_slices, p=p)

    return random_slice_index


def get_normalized_image_train_data_params():
    def array_changer(*, reconstruction_esc):
        flipped_image = np.flip(reconstruction_esc, axis=0)
        return (flipped_image / flipped_image.max(),)

    return dict(
        keys=["reconstruction_esc"],
        array_changer=array_changer,
    )


def get_gnd_prediction_data_params():
    def array_changer(*, gnd):
        return (np.squeeze(gnd, 0),)

    return dict(
        keys=["gnd"],
        array_changer=array_changer,
    )


def get_mean_image_prediction_data_params():
    def array_changer(*, recon):
        return (np.squeeze(recon, 0),)

    return dict(
        keys=["recon"],
        array_changer=array_changer,
    )


def get_mean_var_image_prediction_data_params():
    def array_changer(*, recon, aleatoric_std):
        return (np.squeeze(recon, 0), np.squeeze(np.square(aleatoric_std), 0))

    return dict(
        keys=["recon", "aleatoric_std"],
        array_changer=array_changer,
    )


def get_mean_image_shape(file_path_list):
    file_path = (
        file_path_list[0]
        if isinstance(file_path_list, (list, tuple))
        else file_path_list
    )
    with np.load(file_path, mmap_mode="r") as f:
        return f[list(f.keys())[0]].shape


def get_slice_weight(slice_count, a, b):
    am1 = a - 1
    bm1 = b - 1

    linspace = np.linspace(0, 1, slice_count + 2)[1:-1]
    not_normalized = linspace**am1 * (1 - linspace) ** bm1
    return not_normalized / np.sum(not_normalized)


def get_sampler(ds, batch_size, random, generator):
    if random:
        sampler = RandomSampler(ds, generator=generator)
    else:
        sampler = SequentialSampler(ds)
    batch_sampler = BatchSampler(sampler, batch_size=batch_size, drop_last=True)

    return batch_sampler


def get_augmentation(aug_list, other_augmentation, image_count=1):
    additional_targets = [f"image{i}" for i in range(1, image_count)]
    image_names = ["image"] + additional_targets

    augmentation = A.Compose(
        aug_list, additional_targets={name: "image" for name in additional_targets}
    )

    def augment(slices, labels):
        assert len(slices) == image_count

        slices = tuple(
            np.abs(selected_slice) for selected_slice in slices
        )  #  todo remove

        correct_dtype_slices = tuple(
            single_slice.astype(np.float32)
            if single_slice.dtype.kind == "f"
            else single_slice
            for single_slice in slices
        )

        albumentations_out = augmentation(
            **dict(zip(image_names, correct_dtype_slices))
        )
        augmented_slices = tuple(
            albumentations_out[image_name] for image_name in image_names
        )

        augmented_labels, fully_augmented_slices = other_augmentation(
            labels, augmented_slices
        )

        expanded_labels = [
            np.array([augmented_label]) for augmented_label in augmented_labels
        ]
        expanded_slices = [
            np.expand_dims(single_slice, 2) for single_slice in fully_augmented_slices
        ]

        return expanded_slices, expanded_labels

    return augment


def get_transform(
    aug_list,
    other_augmentation,
    add_var,
    recon,
    use_gt,
    multi_transform,
    slice_list,
    suffix,
    input_modifier_function_dict=None,
):

    opener = {".npz": open_numpy, ".hdf5": open_hdf5}[suffix]

    if slice_list is None:
        slice_selector = slice_selector_info_selector(
            get_random_slice_index_getter(3, 3)
        )
    elif slice_list.dtype.kind == "f":
        slice_selector = slice_selector_info_selector(
            relative_position_slice_index_getter, 1
        )
    else:
        slice_selector = slice_selector_info_selector(index_slice_index_getter, 1)
    if recon:
        if use_gt:
            assert not add_var
            file_selector = itemgetter(0)
            loader_params = get_gnd_prediction_data_params()
        else:
            file_selector = np.random.choice
            if not add_var:
                loader_params = get_mean_image_prediction_data_params()
            else:
                loader_params = get_mean_var_image_prediction_data_params()

    else:
        file_selector = lambda x: x
        loader_params = get_normalized_image_train_data_params()

    if multi_transform:
        loader = get_image_list_getter(
            slice_selector=slice_selector, opener=opener, **loader_params
        )
        if input_modifier_function_dict:
            loader = get_image_getter_wrapper(loader, input_modifier_function_dict)
    else:
        if input_modifier_function_dict is None:
            loader = get_image_getter(
                slice_selector=slice_selector,
                file_selector=file_selector,
                opener=opener,
                **loader_params,
            )
        else:
            loader = get_image_list_getter(
                slice_selector=slice_selector, opener=opener, **loader_params
            )

            loader = get_image_getter_wrapper(
                loader, input_modifier_function_dict, file_selector=file_selector
            )

    image_count = 2 if add_var else 1
    if multi_transform:
        image_count *= multi_transform

    augmentation = get_augmentation(aug_list, other_augmentation, image_count)

    if multi_transform:

        def multivolume_augment(volumes, labels):
            assert len(volumes) == multi_transform
            single_len = len(volumes[0])
            flat_volumes = [volume for volume_list in volumes for volume in volume_list]
            flat_volume_results, augmented_labels = augmentation(flat_volumes, labels)
            unflatted_volumes = [
                flat_volume_results[i : i + single_len]
                for i in range(0, len(flat_volume_results), single_len)
            ]

            label_array = tuple(
                np.stack([augmented_label] * multi_transform)
                for augmented_label in augmented_labels
            )

            volume_stacks = tuple(
                np.stack(volumes) for volumes in zip(*unflatted_volumes)
            )
            return volume_stacks, label_array

        inner_transform = multivolume_augment
    else:
        inner_transform = augmentation

    def transform(data):
        file_path, *label_data = data
        slices = loader(file_path, label_data)
        return inner_transform(slices, label_data)

    return transform


def get_joint_df(file_pathes, labels, file_exists, slice_col=None, overfit=False):
    assert file_exists.sum() > 0, file_pathes.iloc[0]
    joined_df = pd.concat(dict(path=file_pathes, label=labels), axis=1)
    filtered_df = joined_df.loc[file_exists & labels.notna()]

    if slice_col is not None:
        slice_col = np.array(slice_col)
        if np.any(slice_col > 1):
            filtered_df = filtered_df.copy()
            filtered_df["num_slices"] = filtered_df.path.apply(
                lambda x: get_mean_image_shape(x)[0]
            )
            filtered_df = filtered_df.merge(
                pd.Series(slice_col, name="slice"), how="cross"
            )

            keep = filtered_df.num_slices > filtered_df.slice
            filtered_df_slices = filtered_df.loc[keep, ["path", "label", "slice"]]

            if len(filtered_df_slices) < len(filtered_df):
                print("Warning: some files are less large")

            filtered_df = filtered_df_slices
        else:
            filtered_df = filtered_df.merge(
                pd.Series(slice_col, name="slice"), how="cross"
            )

    assert len(filtered_df)
    if overfit:
        filtered_df = filtered_df.iloc[:overfit]

    return filtered_df


def get_ideal_file(dataset_path, file_name):
    return dataset_path / file_name


def get_file_series(dataset_path: Path, file_series: pd.Series):
    file_pathes = file_series.apply(
        lambda file_name: get_ideal_file(dataset_path, file_name)
    )
    files_exist = file_pathes.apply(lambda file_path: file_path.exists())
    return file_pathes, files_exist


def get_recon_file(prediction_path, file_name, acceleration_name, suffix):
    return (
        prediction_path.parent
        / f"{prediction_path.name}{acceleration_name}"
        / ("test_" + Path(file_name).with_suffix(suffix).name)
    )


def get_augmentation_files_list(
    prediction_path: Path, file_series: pd.Series, acceleration_names, suffix
):

    file_pathes = file_series.apply(
        lambda file_name: [
            get_recon_file(prediction_path, file_name, acceleration_name, suffix)
            for acceleration_name in acceleration_names
        ]
    )

    files_exist = file_pathes.apply(
        lambda file_pathes: all(file_path.exists() for file_path in file_pathes)
    )

    return file_pathes, files_exist


def get_dataset(
    *,
    dataframe_filter,
    label_selector,
    transform,
    csv_path,
    prediction_path,
    prediction_suffix,
    dataset_path,
    accelerations,
    batch_size,
    random,
    rng_key,
    recon,
    multi_transform,
    overfit=0,
    slice_list=None,
):

    df = dataframe_filter(pd.read_csv(csv_path))
    labels = label_selector(df)

    if recon:
        file_pathes, file_exists = get_augmentation_files_list(
            prediction_path, df["filename"], accelerations, suffix=prediction_suffix
        )
    else:
        file_pathes, file_exists = get_file_series(dataset_path, df["filename"])

    generator = get_generator_with_jax_rng_key(rng_key)

    collate_fn = (
        singleton_select_collate_fn if multi_transform else tree_collate_function
    )

    filtered_df = get_joint_df(
        file_pathes, labels, file_exists, slice_col=slice_list, overfit=overfit
    )
    ds = CommonDataset(filtered_df, transform)

    sampler = get_sampler(ds, batch_size=batch_size, random=random, generator=generator)
    return ds, sampler, collate_fn, filtered_df
