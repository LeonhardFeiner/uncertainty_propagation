import itertools
from pathlib import Path
from absl import flags
import numpy as np
import pandas as pd
import jax.numpy as jnp
from jax import random
import jax
import augmax
from augmax import InputType
from torch.utils.data import DataLoader, RandomSampler, BatchSampler, SequentialSampler

from data.common import (
    NoiseImages,
    GaussianNoiseImages,
    BetaNoiseImages,
    convert_prngkey_to_uint,
    get_generator_with_jax_rng_key,
    get_worker_init_fn_with_jax_rng_key,
    get_noise_shape_calculator,
    identity,
    get_regression_label_scaler,
    get_regression_label_rescaler,
    TransformDataset,
    PickleDataset,
    ceildiv,
    get_batch_sampler,
    tree_collate_function,
    SliceDataset,
)

from data.recon_downstream import EpochDataset, AllEpochsDataset, RandomEpochDataset
from data.mri.common import get_dataset, get_transform


from data.mri.adni import (
    get_info as get_adni_info,
    get_parameters as get_adni_parameters,
)
from data.mri.knee import (
    get_info as get_knee_info,
    get_parameters as get_knee_parameters,
)

FLAGS = flags.FLAGS


def get_dataset_containers(train_ds, val_ds):
    batches_per_epoch = len(train_ds)
    train_ds_generator = itertools.repeat(train_ds)

    return train_ds_generator, batches_per_epoch, val_ds


def get_source_dataset(subset, rng_key, return_variance=False, multi_noise=False):
    multi_transform = FLAGS.multi_augmentation_batch and FLAGS.num_augmentation_degrees
    assumed_noise = None
    random_order = "train" in subset and not multi_transform
    fix = not random_order
    batch_size = 1 if multi_transform else FLAGS.batch_size
    num_workers = FLAGS.num_workers
    persistent_workers = bool(num_workers) and not fix

    if FLAGS.dataset == "luna":
        root_path = Path(FLAGS.luna_data_path).expanduser().resolve()
        image_parameters = pd.read_csv(root_path / "image_parameters.csv", index_col=0)
        slice_parameters = pd.read_csv(root_path / "slice_parameters.csv", index_col=0)
        joint_slice_parameters = pd.merge(
            image_parameters,
            slice_parameters,
            on="seriesuid",
            suffixes=(None, "_slice"),
        ).drop(columns=["subset_slice", "total_index_slice", "subset_index_slice"])

        associations = {
            "trainfull": {f"subset{i}" for i in range(0, 6)},
            "train0": {f"subset{i}" for i in range(0, 3)},
            "train1": {f"subset{i}" for i in range(3, 6)},
            "valfull": {f"subset{i}" for i in range(6, 8)},
            "val0": {f"subset{i}" for i in range(6, 7)},
            "val1": {f"subset{i}" for i in range(7, 8)},
            "test": {f"subset{i}" for i in range(8, 10)},
        }

        for association, subsets in associations.items():
            joint_slice_parameters[association] = joint_slice_parameters.subset.isin(
                subsets
            )

        # foreground_slice_parameters = joint_slice_parameters.query("nodules_slice>0")
        # foreground_slice_parameters = joint_slice_parameters.query(
        #     "voxel_with_nodules_slice>=32"
        # )

        foreground_slice_parameters = joint_slice_parameters.query(
            "(distance_to_nodule_center_in_slices<=1.5)&(voxel_with_nodules_slice>=32)"
        )

        nodule_mask_name = "nodule_mask"

        def transform_raw(x):
            normalized_image = np.expand_dims((x["image"] + 1350) / 1500, -1)
            mask = np.expand_dims((x[nodule_mask_name] > 0).astype(np.int8), -1)

            return normalized_image, mask

        if assumed_noise:
            raise NotImplementedError

        else:
            transform = transform_raw

        filtered_df = foreground_slice_parameters.query(subset)
        subfolder_names = ["image", nodule_mask_name]  # "anatomy_mask"
        id_name = "seriesuid"
        index_name = "slice_index"

        ds = SliceDataset(
            filtered_df,
            root_path,
            subfolder_names,
            id_name,
            index_name,
            transform=transform,
        )

        generator = get_generator_with_jax_rng_key(rng_key)
        if random_order:
            sampler = RandomSampler(ds, generator=generator)
        else:
            sampler = SequentialSampler(ds)
        batch_sampler = BatchSampler(sampler, batch_size=batch_size, drop_last=True)

        dl = DataLoader(
            ds,
            collate_fn=tree_collate_function,
            num_workers=num_workers,
            # prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            worker_init_fn=get_worker_init_fn_with_jax_rng_key(rng_key),
            batch_sampler=batch_sampler,
        )
        df = filtered_df

    else:
        raise NotImplementedError

    if fix:
        dl = list(dl)

    return dl, df


def get_mri_dataloader(rng_key, subset):
    if FLAGS.dataset == "adni":
        (
            get_augmentation_list,
            other_augmentation,
            dataframe_filter,
            label_selector,
        ) = get_adni_parameters(FLAGS.adni_target, FLAGS.classifier)

    elif FLAGS.dataset == "knee":
        (
            get_augmentation_list,
            other_augmentation,
            dataframe_filter,
            label_selector,
        ) = get_knee_parameters("chirality", FLAGS.classifier)
    else:
        raise NotImplementedError

    mask_foreground = FLAGS.masked_foreground
    original_data = FLAGS.original_data

    csv_path_raw = Path(FLAGS.csv_path).expanduser().resolve()
    csv_path = csv_path_raw.parent / f"{csv_path_raw.name}_{subset}.csv"

    if original_data:
        dataset_path = Path(FLAGS.original_data_path).expanduser().resolve()
        prediction_path = None
        suffix = ".npz"
    else:
        dataset_path = None
        prediction_suffix = ".npz"
        suffix = prediction_suffix

        if "train" in subset:
            prediction_path = Path(FLAGS.train_prediction_path).expanduser().resolve()
        elif "val" in subset:
            prediction_path = Path(FLAGS.val_prediction_path).expanduser().resolve()
        elif "test" in subset:
            prediction_path = Path(FLAGS.test_prediction_path).expanduser().resolve()
        else:
            raise NotImplementedError

    accelerations = FLAGS.mri_accelerations
    add_var = FLAGS.input_distribution == "mvnd"
    use_gt = FLAGS.use_gt
    multi_transform = FLAGS.multi_augmentation_batch and len(accelerations)
    random = "train" in subset and not multi_transform
    fix = not random
    batch_size = 1 if multi_transform else FLAGS.batch_size
    overfit = FLAGS.overfit
    # drop_last = not overfit
    num_workers = FLAGS.num_workers
    persistent_workers = bool(num_workers) and not fix

    if len(FLAGS.slice_list) == 0:
        slice_list = None
    elif all(num.isdigit() for num in FLAGS.slice_list):
        slice_list = np.array([int(num) for num in FLAGS.slice_list], dtype=np.int32)
    else:
        slice_list = np.array(
            [float(num) for num in FLAGS.slice_list], dtype=np.float32
        )

    if original_data:
        assert not add_var
        assert not multi_transform

    recon = not original_data

    aug_list = get_augmentation_list(subset)
    transform = get_transform(
        aug_list=aug_list,
        other_augmentation=other_augmentation,
        add_var=add_var,
        recon=recon,
        use_gt=use_gt,
        multi_transform=multi_transform,
        slice_list=slice_list,
        suffix=suffix,
    )

    ds, batch_sampler, collate_fn, filtered_df = get_dataset(
        dataframe_filter=dataframe_filter,
        label_selector=label_selector,
        transform=transform,
        csv_path=csv_path,
        prediction_path=prediction_path,
        prediction_suffix=prediction_suffix,
        dataset_path=dataset_path,
        accelerations=accelerations,
        batch_size=batch_size,
        random=random,
        rng_key=rng_key,
        recon=recon,
        multi_transform=multi_transform,
        overfit=overfit,
        slice_list=slice_list,
    )

    dl = DataLoader(
        ds,
        collate_fn=collate_fn,
        num_workers=num_workers,
        # prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        worker_init_fn=get_worker_init_fn_with_jax_rng_key(rng_key),
        batch_sampler=batch_sampler,
    )

    if fix:
        dl = list(dl)

    return dl, filtered_df


def get_dataloaders(name_seeds, multi_augmentation_batch=False):
    if FLAGS.overfit:
        original_names = name_seeds.keys()
        name_seeds = dict(
            [[(key, seed) for key, seed in name_seeds.items() if "train" not in key][0]]
        )

    assert FLAGS.dataset in {"knee", "adni"}
    dls_dfs = [get_mri_dataloader(seed, name) for name, seed in name_seeds.items()]

    if FLAGS.overfit:
        ((dl, df),) = dls_dfs

        dls = [
            list(dl) * 32 if "train" in name else list(dl) for name in original_names
        ]

        dfs = [
            (pd.concat([df] * 32) if "train" in name and dl is not None else dl)
            for name in original_names
        ]

    else:
        dls, dfs = (list(item) for item in zip(*dls_dfs))

    joint_df = (
        None
        if dfs[0] is None
        else pd.concat({name: df for name, df in zip(name_seeds.keys(), dfs)})
    )

    return dls, joint_df


def binary_segmentation_transform(x):
    return {**x, "segmentation": np.clip(x["segmentation"], 0, 1)}


def get_noise_adder(noise_images, train=False, squeeze_dims=0):
    if FLAGS.multi_augmentation_batch:
        batch_size = FLAGS.num_augmentation_degrees
    else:
        batch_size = FLAGS.batch_size  # if train else FLAGS.test_batch_size

    def default_key_splitter(key):
        return jax.random.split(key, batch_size)

    def broadcast_key_splitter(key):
        return jnp.broadcast_to(key, (batch_size, 2))

    key_splitter = default_key_splitter

    if FLAGS.multi_augmentation_batch:
        if not train:
            key_splitter = broadcast_key_splitter

        def add_noise(noise_key, sharp_img, labels_raw):
            orig_imgs = jnp.broadcast_to(
                sharp_img, (batch_size, *jnp.squeeze(sharp_img, squeeze_dims).shape)
            )
            noise_keys = key_splitter(noise_key)

            labels = jax.tree_map(
                lambda label: jnp.broadcast_to(
                    label, (batch_size, *jnp.squeeze(label, squeeze_dims).shape)
                ),
                labels_raw,
            )

            noise_imgs, sharp_imgs, *others = noise_images.get_fix_noise_images(
                noise_keys, orig_imgs
            )
            return noise_imgs, *others, sharp_imgs, labels

    else:

        def add_noise(noise_key, orig_imgs, labels):
            noise_keys = key_splitter(noise_key)

            noise_imgs, sharp_imgs, *others = noise_images.get_random_noise_images(
                noise_keys, orig_imgs
            )
            return noise_imgs, *others, sharp_imgs, labels

    return add_noise


def get_dataset_info():
    def default_pre_augmentation(rng, batch):
        return batch

    def default_post_augmentation(image):
        return image

    train_pre_augmentation = default_pre_augmentation
    val_pre_augmentation = default_pre_augmentation
    image_post_augmentation = default_post_augmentation
    label_rescaler = identity
    label_scaler = identity
    extra_types = ()

    class_weights = None
    num_input_channels = 1
    image_output = False

    if FLAGS.dataset in {"knee", "adni"}:
        image_res_factor = 2
        plot_res_factor = 2
        full_batches_keep_count = 4 if FLAGS.multi_augmentation_batch else 1
        add_barplot_numbers = True
        formatting = "0" if FLAGS.classifier else ".00f"
        image_shape = 256, 256

    else:
        image_res_factor = 4
        plot_res_factor = 2
        full_batches_keep_count = 4 if FLAGS.multi_augmentation_batch else 1
        add_barplot_numbers = False
        formatting = "0" if FLAGS.classifier else "+.02f"
        image_shape = 28, 28

    if FLAGS.dataset in {"knee", "adni"}:
        classifier = FLAGS.classifier
        if FLAGS.dataset == "adni":
            target = FLAGS.adni_target

            (
                class_labels,
                class_weights,
                target_name,
                mean_std,
                image_shape,
            ) = get_adni_info(target, classifier)
        else:
            target = "chirality"

            (
                class_labels,
                class_weights,
                target_name,
                mean_std,
                image_shape,
            ) = get_knee_info(target, classifier)

        if mean_std is not None:
            mean, std = mean_std
            label_rescaler = get_regression_label_rescaler(mean, std)
            label_scaler = get_regression_label_scaler(mean, std)

        if FLAGS.input_distribution == "dirac":

            def batch_selector(batch):
                (inputs,), (target, *_) = batch
                return (inputs,), (label_scaler(target),), ()

        elif FLAGS.input_distribution == "mvnd":

            def batch_selector(batch):
                inputs, (target, *_) = batch
                return (tuple(inputs),), (label_scaler(target),), ()

        else:
            raise NotImplementedError

    else:
        raise NotImplementedError

    data_shape = (FLAGS.batch_size, *image_shape, num_input_channels)

    if FLAGS.binary and FLAGS.classifier:
        assert len(class_labels) == 2
        num_outputs = 1
    else:
        num_outputs = len(class_labels)

    num_console_predictions = 12 if FLAGS.classifier else 8
    loss_kwargs = dict(
        image_output=image_output,
        full_batches_keep_count=full_batches_keep_count,
        foreground_classes=None if image_output else False,
        class_weights=class_weights,
        label_rescaler=label_rescaler,
        concatenate_batch_metrics=not FLAGS.multi_augmentation_batch,
        stack_batch_metrics=FLAGS.multi_augmentation_batch,
    )

    if FLAGS.dataset in {"adni", "knee"}:
        if FLAGS.multi_augmentation_batch:
            if all(num.isdigit() for num in FLAGS.mri_accelerations):
                accelerations = [int(num) for num in FLAGS.mri_accelerations]
            else:
                accelerations = [float(num) for num in FLAGS.mri_accelerations]

            loss_kwargs["extra_infos"] = {
                "element_wise_scalar": {"accelerations": np.array(accelerations)}
            }

    eval_kwargs = dict(
        class_labels=class_labels,
        target_name=target_name,
        image_output=image_output,
        image_res_factor=image_res_factor,
        plot_res_factor=plot_res_factor,
        add_barplot_numbers=add_barplot_numbers,
    )

    eval_epoch_kwargs = dict(
        **eval_kwargs,
        num_console_predictions=num_console_predictions,
        num_visual_images=full_batches_keep_count
        * (1 if FLAGS.multi_augmentation_batch else FLAGS.batch_size),
        formatting=formatting,
    )

    if FLAGS.input_distribution == "mvnd":

        def post_augmentation(x):
            image, diag, *other = x
            new_image = image_post_augmentation(image)
            new_diag = jnp.clip(diag, 1e-6, None)
            return new_image, new_diag, *other

    else:
        post_augmentation = image_post_augmentation

    def input_wise_post_augmentation(input_array):
        return tuple(post_augmentation(x) for x in input_array)

    target_type = (
        (InputType.MASK if FLAGS.classifier else InputType.IMAGE)
        if image_output
        else InputType.IGNORE
    )

    target_types = (target_type,)

    if FLAGS.input_distribution == "dirac":
        input_types = (InputType.IMAGE,)
    elif FLAGS.input_distribution == "mvnd":
        input_types = ((InputType.IMAGE, InputType.IMAGE),)
    else:
        raise NotImplementedError

    augmax_types = (input_types, target_types, extra_types)

    augmentations = []

    train_augmentations = list(augmentations)
    val_augmentations = list(augmentations)

    if FLAGS.augment:
        train_augmentations += [
            augmax.RandomSizedCrop(256, 256, (0.8, 1.25)),
            augmax.Rotate(),
        ]

    train_augment = get_augmentation(augmax_types, train_augmentations, seed_split=True)
    val_augment = get_augmentation(augmax_types, val_augmentations, seed_split=True)

    def train_batch_modifier(rng, batch):
        pre_augmented_batch = train_pre_augmentation(rng, batch)
        selected_batch = batch_selector(pre_augmented_batch)
        inputs, targets, extras = train_augment(rng, selected_batch)
        post_augmented_inputs = input_wise_post_augmentation(inputs)
        return post_augmented_inputs, targets, extras

    def val_batch_modifier(rng, batch):
        pre_augmented_batch = val_pre_augmentation(rng, batch)
        selected_batch = batch_selector(pre_augmented_batch)
        inputs, targets, extras = val_augment(rng, selected_batch)
        post_augmented_inputs = input_wise_post_augmentation(inputs)
        return post_augmented_inputs, targets, extras

    return (
        data_shape,
        num_outputs,
        train_batch_modifier,
        val_batch_modifier,
        loss_kwargs,
        eval_epoch_kwargs,
    )


def get_augmentation(input_types, augmentations, seed_split=True):
    if seed_split:
        seed_multiplier = random.split
    else:
        seed_multiplier = lambda key, length: jnp.broadcast_to(key, (length, 2))

    if augmentations:
        augmax_chain = jax.vmap(
            augmax.Chain(
                *augmentations,
                input_types=input_types,
            )
        )

        def augmax_function(rng, data):
            return augmax_chain(
                seed_multiplier(rng, len(jax.tree_util.tree_leaves(data)[0])), data
            )

    else:

        def augmax_function(rng, data):
            return data

    return augmax_function


def get_reconstruction_dataset(
    subset,
    target,
    info,
    random=True,
    epoch_count=None,
    rng_key=None,
    balanced=False,
    transform=None,
):
    """
    reconstruction dataset
    """
    data_path = Path(FLAGS.reconstruction_data_path).expanduser().resolve() / subset
    files = sorted(data_path.glob("*.pickle"))
    assert len(files), f"{data_path} does not contain predictions"
    pickle_ds = PickleDataset(files)
    required_tensors = [
        ("mean", True),
        ("aleatoric", True),
        (target, False),
        ("original", False),
        ("augmented", True),
        (info, False),
    ]

    if transform is not None:
        pickle_ds = TransformDataset(pickle_ds, transform)

    if FLAGS.artificial_uncertainty_gradient:

        def factor_transform(x):
            mean = np.broadcast_to(
                np.mean(x["mean"], 0, keepdims=True), x["mean"].shape
            )

            aleatoric = np.mean(x["aleatoric"], 0, keepdims=True)

            aleatoric_range = (
                aleatoric.T
                * np.linspace(0, FLAGS.input_uncertainty_factor, len(mean), True)
            ).T

            return {**x, "mean": mean, "aleatoric": aleatoric_range}

        pickle_ds = TransformDataset(pickle_ds, factor_transform)

    elif FLAGS.input_uncertainty_factor != 1:

        def factor_transform(x):
            return {**x, "aleatoric": FLAGS.input_uncertainty_factor * x["aleatoric"]}

        pickle_ds = TransformDataset(pickle_ds, factor_transform)

    required_tensors += []
    if random:
        ds_raw = RandomEpochDataset(pickle_ds, required_tensors, epoch_count)

        worker_kwargs = (
            dict(
                num_workers=FLAGS.num_workers,
                prefetch_factor=ceildiv(4 * FLAGS.batch_size, FLAGS.num_workers),
                persistent_workers=True,
            )
            if FLAGS.num_workers
            else dict()
        )
        if balanced:

            def balance_label_getter(x):
                return np.squeeze(x[-1], -1)

        else:
            balance_label_getter = None

        dl = DataLoader(
            ds_raw,
            collate_fn=tree_collate_function,
            worker_init_fn=get_worker_init_fn_with_jax_rng_key(rng_key),
            batch_sampler=get_batch_sampler(
                rng_key,
                ds_raw,
                FLAGS.batch_size,
                balance_label_getter=balance_label_getter,
            ),
            **worker_kwargs,
        )
    elif epoch_count is None:
        dl = EpochDataset(pickle_ds, required_tensors)
    else:
        dl = AllEpochsDataset(pickle_ds, required_tensors, epoch_count)

    return dl, None
