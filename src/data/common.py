from abc import abstractmethod
from operator import itemgetter
from typing import Callable, Iterator, Optional, Sequence, Sized, Tuple, List, Any
from functools import partial
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import jax
import jax.numpy as jnp
from jax import random
from torch.utils.data import Dataset, DataLoader, RandomSampler, BatchSampler, Sampler

from common.utils import rayleigh


def convert_prngkey_to_uint(jax_seed):
    return np.asarray(jax_seed).view(dtype=np.uint64).item()


def ceildiv(a, b):
    return -(-a // b)


def get_generator(seed):
    return torch.Generator().manual_seed(seed)


def get_generator_with_jax_rng_key(key):
    return get_generator(convert_prngkey_to_uint(key))


def get_worker_init_fn(seed):
    def worker_init_fn(worker_seed):
        np.random.seed((worker_seed + seed) % (2**32))

    return worker_init_fn


def get_worker_init_fn_with_jax_rng_key(key):
    return get_worker_init_fn(convert_prngkey_to_uint(key))


def get_multitransform_fn(transform_fn_list, collate_fn):
    def transform(data):
        transformed = [transform_fn(data) for transform_fn in transform_fn_list]
        return collate_fn(transformed)

    return transform


def get_constant_seed_multitransform_fn(transform_fn_list, collate_fn):
    def transform(data):
        seed = np.random.randint(np.iinfo(np.int32).max)

        def seed_apply(fn, *args, **kwargs):
            np.random.seed(seed)
            return fn(*args, **kwargs)

        transformed = [
            seed_apply(transform_fn, data) for transform_fn in transform_fn_list
        ]
        np.random.seed(seed)

        return collate_fn(transformed)

    return transform


def get_batch_sampler(
    rng_key,
    ds,
    batch_size,
    for_storage=False,
    storage_repeats=None,
    balance_label_getter=None,
):
    if for_storage:
        storage_repeats |= batch_size
        sample_count = len(ds)
        batch_sampler = np.tile(
            np.expand_dims(np.arange(sample_count), 1), (1, storage_repeats)
        )
    else:
        generator = get_generator_with_jax_rng_key(rng_key)
        if balance_label_getter is not None:
            sampler = BalanceSampler(
                ds,
                label_getter=balance_label_getter,
                generator=generator,
            )
        else:
            sampler = RandomSampler(ds, generator=generator)
        batch_sampler = BatchSampler(sampler, batch_size=batch_size, drop_last=True)
    return batch_sampler


def metadict_collate_fn(data):
    images, metadicts = zip(*data)
    images_combined = (np.stack(part) for part in zip(*images))
    metadicts_combined = {
        k: np.array([dic[k] for dic in metadicts]) for k in metadicts[0].keys()
    }

    return (*images_combined, metadicts_combined)


def elementwise_collate_fn(data):
    return tuple(np.stack(part) for part in zip(*data))


def singleton_select_collate_fn(batch):
    return batch[0]


def stack_except_string(*x):
    if isinstance(x[0], str):
        return list(x)
    else:
        return np.stack(x)


def tree_collate_function(data):
    return jax.tree_map(stack_except_string, *data)


def get_noise_shape_calculator(remove_dims=(), equal_dims=()):
    def noise_shape_calculator(image_shape):
        equal_dims_set = set(
            dim if dim >= 0 else len(image_shape) + dim for dim in equal_dims
        )
        remove_dims_set = set(
            dim if dim >= 0 else len(image_shape) + dim for dim in remove_dims
        )

        return tuple(
            image_shape[i] if i not in equal_dims_set else 1
            for i in range(len(image_shape))
            if i not in remove_dims_set
        )

    return noise_shape_calculator


def identity(x):
    return x


def get_regression_label_scaler(mean, std):
    def label_scaler(x):
        return (x - mean) / std

    return label_scaler


def get_regression_label_rescaler(mean, std):
    var = np.square(std)

    def label_rescaler(result_dict):
        scaled_labels = {
            key: (value * std + mean)
            for key, value in result_dict.items()
            if key in {"gt", "pr"}
        }

        scaled_stds = {
            key: (value * std)
            for key, value in result_dict.items()
            if key.endswith("std")
        }

        scaled_vars = {
            key: (value * var)
            for key, value in result_dict.items()
            if key.endswith("var")
        }

        unscaled_losses = {
            "unscaled_" + key: value
            for key, value in result_dict.items()
            if key.endswith("l1") or key.endswith("l2")
        }

        scaled_l1 = {
            key: (value * std)
            for key, value in result_dict.items()
            if key.endswith("l1")
        }

        scaled_l2 = {
            key: (value * var)
            for key, value in result_dict.items()
            if key.endswith("l2")
        }

        return {
            **result_dict,
            **scaled_labels,
            **scaled_stds,
            **scaled_vars,
            **unscaled_losses,
            **scaled_l1,
            **scaled_l2,
        }

    return label_rescaler


class TransformDataset:
    def __init__(
        self,
        dataset,
        transform,
    ) -> None:
        self.dataset = dataset
        self.transform = transform

    @property
    def epoch(self):
        return self.dataset.epoch

    @epoch.setter
    def epoch(self, epoch):
        self.dataset.epoch = epoch

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.transform(self.dataset[index])


class PickleDataset(Dataset):
    file_list: List[Path]
    transform: Callable

    def __init__(
        self, file_list: List[Path], transform: Callable = lambda x: x
    ) -> None:
        super().__init__()
        self.file_list = file_list
        self.transform = transform

    def __getitem__(self, index: int) -> Any:
        with open(self.file_list[index], "rb") as file:
            return self.transform(pickle.load(file))

    def __len__(self) -> int:
        return len(self.file_list)


class ListDataset(Dataset):
    dataset_list: List[Dataset]
    collate_fn: Callable

    def __init__(
        self, dataset_list: List[Dataset], collate_fn: Callable = lambda x: x
    ) -> None:
        super().__init__()
        self.dataset_list = dataset_list
        self.collate_fn = collate_fn

        assert all(
            len(self.dataset_list[0]) == len(dataset) for dataset in self.dataset_list
        )

    def __getitem__(self, index: int) -> Any:
        self.collate_fn([dataset[index] for dataset in self.dataset_list])

    def __len__(self) -> int:
        return len(self.dataset_list[0])


class BalanceSampler(Sampler):
    data_source: Sized
    replacement: bool

    def __init__(
        self,
        data_source: Sized,
        label_getter: Callable[[Sequence], Any] = itemgetter(-1),
        replacement: bool = False,
        num_samples: Optional[int] = None,
        generator=None,
    ) -> None:
        # super(BalanceSampler, self).__init__(data_source)
        self.data_source = data_source
        self.replacement = replacement
        self.generator = generator

        index_arrays = np.array([label_getter(sample) for sample in data_source])
        labels = np.unique(index_arrays)
        self.label_indices = [np.where(index_arrays == label)[0] for label in labels]
        self.min_length = min(len(index_array) for index_array in self.label_indices)
        max_num_samples = self.min_length * len(labels)
        if num_samples is None:
            self.num_samples = max_num_samples
        else:
            assert num_samples < max_num_samples
            self.num_samples = num_samples

        if not isinstance(self.replacement, bool):
            raise TypeError(
                "replacement should be a boolean value, but got "
                f"replacement={self.replacement}"
            )

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError(
                "num_samples should be a positive integer "
                f"value, but got num_samples={self.num_samples}"
            )

    def __iter__(self) -> Iterator[int]:
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        if self.replacement:
            for _ in range(self.num_samples):
                label_index = torch.randint(
                    high=len(self.labels_indices),
                    dtype=torch.int64,
                    generator=generator,
                ).item()
                item_index = torch.randint(
                    high=len(self.labels_indices[label_index]),
                    dtype=torch.int64,
                    generator=generator,
                ).item()
                yield self.label_indices[label_index][item_index]
        else:
            label_type_index_indices = torch.repeat_interleave(
                torch.arange(len(self.label_indices)), self.min_length
            )[
                torch.randperm(
                    len(self.label_indices) * self.min_length,
                    generator=generator,
                )
            ].numpy()

            item_indices = np.stack(
                [
                    self.label_indices[i][
                        torch.randperm(len(self.label_indices[i]), generator=generator)[
                            : self.min_length
                        ]
                    ]
                    for i in range(len(self.label_indices))
                ]
            )

            selected_item_index = (
                np.cumsum(
                    label_type_index_indices[:, np.newaxis]
                    == np.arange(len(self.label_indices)),
                    0,
                )[np.arange(len(label_type_index_indices)), label_type_index_indices]
                - 1
            )

            yield from item_indices[label_type_index_indices, selected_item_index][
                : self.num_samples
            ].tolist()

    def __len__(self) -> int:
        return self.num_samples


class NoiseImages:
    def __init__(
        self,
        noise_shape_calculator=get_noise_shape_calculator((0,)),
        clip_min=None,
        clip_max=None,
    ) -> None:
        self.noise_shape_calculator = noise_shape_calculator
        self.clip_min = clip_min
        self.clip_max = clip_max

    @abstractmethod
    def get_fix_noise_images(self, rng_keys, sharp_imgs):
        pass

    @abstractmethod
    def get_random_noise_images(self, rng_keys, sharp_imgs):
        pass

    @abstractmethod
    def get_fix_noises(self, rng_keys, shape):
        pass

    @abstractmethod
    def get_random_noises(self, rng_keys, shape):
        pass

    def clip_image(self, sharp_imgs, noise_imgs, *others):
        sum_image = sharp_imgs + noise_imgs
        broadcasted_others = (
            jnp.broadcast_to(
                jnp.expand_dims(other, range(1, sharp_imgs.ndim)), sharp_imgs.shape
            )
            for other in others
        )
        if self.clip_min is None and self.clip_max is None:
            return sum_image, sharp_imgs, *broadcasted_others
        else:
            return (
                jnp.clip(sum_image, a_min=self.clip_min, a_max=self.clip_max),
                jnp.clip(sharp_imgs, a_min=self.clip_min, a_max=self.clip_max),
                *broadcasted_others,
            )

    def get_fix_noise_images(self, rng_keys, sharp_imgs):
        shape = self.noise_shape_calculator(sharp_imgs.shape)
        noise_imgs, *other = self.get_fix_noises(rng_keys, shape)
        return self.clip_image(sharp_imgs, noise_imgs, *other)

    def get_random_noise_images(self, rng_keys, sharp_imgs):
        shape = self.noise_shape_calculator(sharp_imgs.shape)
        noise_imgs, *other = self.get_random_noises(rng_keys, shape)
        return self.clip_image(sharp_imgs, noise_imgs, *other)


class BetaNoiseImages(NoiseImages):
    def __init__(
        self,
        min_a=2,
        max_a=8,
        b=2,
        noise_shape_calculator=get_noise_shape_calculator((0,)),
        clip_min=None,
        clip_max=None,
    ) -> None:
        self.min_a = min_a
        self.max_a = max_a
        self.b = b
        super(BetaNoiseImages, self).__init__(
            noise_shape_calculator, clip_min, clip_max
        )

    def get_fix_noises(self, rng_keys, shape):
        a_list = jnp.linspace(self.min_a, self.max_a, len(rng_keys))

        return (
            jax.vmap(partial(random.beta, b=self.b, shape=shape), (0, 0))(
                rng_keys, a_list
            ),
        )

    def get_random_noises(self, rng_keys, shape):
        return jax.vmap(partial(self.get_random_noise, shape=shape))(rng_keys)

    def get_random_noise(self, rng_key, shape):
        noise_degree_key, noise_key = random.split(rng_key)
        noise_degree = random.uniform(noise_degree_key)

        a = self.min_a + (self.max_a - self.min_a) * (1 - noise_degree)

        return (random.beta(noise_key, a, self.b, shape),)

    def get_fix_noise_images(self, rng_keys, sharp_imgs):
        shape = self.noise_shape_calculator(sharp_imgs.shape)
        noise_imgs, *other = self.get_fix_noises(rng_keys, shape)
        # return jnp.where(sharp_imgs, noise_imgs, 1 - noise_imgs)
        return (
            sharp_imgs * noise_imgs + (1 - sharp_imgs) * (1 - noise_imgs),
            sharp_imgs,
            *other,
        )

    def get_random_noise_images(self, rng_keys, sharp_imgs):
        shape = self.noise_shape_calculator(sharp_imgs.shape)
        noise_imgs, *other = self.get_random_noises(rng_keys, shape)
        # TODO check wheter this generalization makes sense
        # return jnp.where(sharp_imgs, noise_imgs, 1 - noise_imgs)
        return (
            sharp_imgs * noise_imgs + (1 - sharp_imgs) * (1 - noise_imgs),
            sharp_imgs,
            *other,
        )


class GaussianNoiseImages(NoiseImages):
    def __init__(
        self,
        min_sigma=0.00,
        max_sigma=0.16,
        noise_shape_calculator=get_noise_shape_calculator((0,)),
        clip_min=None,
        clip_max=None,
        return_variance=False,
    ) -> None:
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.return_variance = return_variance

        super(GaussianNoiseImages, self).__init__(
            noise_shape_calculator, clip_min, clip_max
        )

    def get_fix_noises(self, rng_keys, shape):
        sigma_list = jnp.linspace(self.max_sigma, self.min_sigma, len(rng_keys))
        raw_noise = jax.vmap(partial(random.normal, shape=shape))(rng_keys)
        noise_images = raw_noise * jnp.expand_dims(sigma_list, range(1, raw_noise.ndim))

        if self.return_variance:
            return noise_images, jnp.square(sigma_list)
        else:
            return (noise_images,)

    def get_random_noises(self, rng_keys, shape):
        return jax.vmap(partial(self.get_random_noise, shape=shape))(rng_keys)

    def get_random_noise(self, rng_key, shape):
        noise_degree_key, noise_key = random.split(rng_key)
        noise_degree = random.uniform(noise_degree_key)

        sigma = self.min_sigma + (self.max_sigma - self.min_sigma) * (1 - noise_degree)
        noise_images = random.normal(noise_key, shape) * sigma

        if self.return_variance:
            return noise_images, jnp.square(sigma)
        else:
            return (noise_images,)


class RayleighNormalNoiseImages(NoiseImages):
    def __init__(
        self,
        min_scale=0.0,
        max_scale=1.0,
        noise_shape_calculator=get_noise_shape_calculator((0,)),
        clip_min=None,
        clip_max=None,
    ) -> None:
        self.min_scale = min_scale
        self.max_scale = max_scale
        super(RayleighNormalNoiseImages, self).__init__(
            noise_shape_calculator, clip_min, clip_max
        )

    def get_fix_noises(self, rng_keys, shape):
        scale_list = jnp.linspace(self.max_scale, self.min_scale, len(rng_keys))

        rayleigh_keys, normal_keys = jax.vmap(random.split, out_axes=1)(rng_keys)
        sigmas = jax.vmap(partial(rayleigh, shape=shape))(rayleigh_keys, scale_list)
        raw_noise = jax.vmap(partial(random.normal, shape=shape))(normal_keys)
        return (raw_noise * sigmas,)

    def get_random_noises(self, rng_keys, shape):
        return jax.vmap(partial(self.get_random_noise, shape=shape))(rng_keys)

    def get_random_noise(self, rng_key, shape):
        noise_degree_key, rayleigh_key, normal_key = random.split(rng_key, 3)
        noise_degree = random.uniform(noise_degree_key)

        scale = self.min_scale + (self.max_scale - self.min_scale) * (1 - noise_degree)
        sigma = rayleigh(rayleigh_key, scale, shape)
        return (random.normal(normal_key, shape) * sigma,)


class MultiImageDataset(Sequence):
    def __init__(
        self,
        dataset,
        batch_size,
        noise_images: NoiseImages = BetaNoiseImages(),
        shuffle=False,
        key=random.PRNGKey(0),
    ) -> None:
        self.dataset = dataset
        self.batch_size = jnp.array(batch_size)
        self.index_range = jnp.arange(self.batch_size)
        self.noise_images = noise_images
        self.shuffle = shuffle
        self.key = key
        self.reshuffle()

    def reshuffle(self):
        key = random.PRNGKey(self.epoch) + self.key
        if self.shuffle:
            indices = jnp.arange(len(self.dataset))
            self.shuffled_indices = random.permutation(key, indices)
        else:
            self.shuffled_indices = None

    @property
    def epoch(self):
        return self.dataset.epoch

    @epoch.setter
    def epoch(self, epoch):
        self.dataset.epoch = epoch
        self.reshuffle()

    def __len__(self):
        return len(self.dataset) // self.batch_size

    def __getitem__(self, index):
        try:
            len(self.dataset)
        except:
            pass
        else:
            if index >= len(self):
                raise IndexError

        def inner_get_item(index, shuffled_indices):
            indices = self.index_range + jnp.array(index) * self.batch_size
            resulting_shuffled_indices = (
                indices if shuffled_indices is None else shuffled_indices[indices]
            )

            orig_imgs, labels, noise_keys = jax.vmap(
                self.dataset.__getitem__, 0, (0, 0, 0)
            )(resulting_shuffled_indices)
            noise_imgs, sharp_imgs, *others = self.noise_images.get_random_noise_images(
                noise_keys, orig_imgs
            )
            return noise_imgs, *others, sharp_imgs, labels

        return jax.jit(inner_get_item)(index, self.shuffled_indices)


class MultiNoiseDataset:
    def __init__(
        self,
        dataset,
        batch_size,
        noise_images: NoiseImages = BetaNoiseImages(
            noise_shape_calculator=get_noise_shape_calculator((0,))
        ),
    ) -> None:
        self.dataset = dataset
        self.batch_size = jax.Array(batch_size)
        self.index_range = jnp.arange(self.batch_size)
        self.noise_images = noise_images

    @property
    def epoch(self):
        return self.dataset.epoch

    @epoch.setter
    def epoch(self, epoch):
        self.dataset.epoch = epoch

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        try:
            if index >= len(self):
                raise IndexError
        except TypeError:
            pass

        def inner_get_item(index):
            sharp_img, labels_raw, noise_key = self.dataset[index]
            orig_imgs = jnp.broadcast_to(sharp_img, (self.batch_size, *sharp_img.shape))
            noise_keys = jnp.broadcast_to(noise_key, (self.batch_size, 2))

            labels = {
                key: jnp.broadcast_to(label, (self.batch_size, *label.shape))
                for key, label in labels_raw.items()
            }
            noise_imgs, sharp_imgs, *others = self.noise_images.get_fix_noise_images(
                noise_keys, orig_imgs
            )
            return noise_imgs, *others, sharp_imgs, labels

        return jax.jit(inner_get_item)(index)


class PrngKeyDataset:
    "dataset"

    def __init__(self, dataset, rng_key=random.PRNGKey(0), key_shape=()):
        self.rng_key = rng_key
        if isinstance(key_shape, int):
            key_shape = (key_shape,)
        self.key_shape = (*key_shape, 2)
        self.key_count = np.prod(key_shape, dtype=int)
        self.dataset = dataset
        self.epoch = 0

    @property
    def epoch(self):
        if hasattr(self.dataset, "epoch"):
            return self.dataset.epoch
        else:
            return self._epoch

    @epoch.setter
    def epoch(self, epoch):
        if hasattr(self.dataset, "epoch"):
            self.dataset.epoch = epoch
        else:
            self._epoch = epoch

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        "get sample"
        if isinstance(index, tuple):
            assert len(index) == 2
            index, epoch = index
        else:
            epoch = self.epoch
        combined_key = random.fold_in(random.fold_in(self.rng_key, index), epoch)
        shaped_key = random.split(combined_key, self.key_count).reshape(self.key_shape)
        img, target = self.dataset[index]
        return img, target, shaped_key


class DatasetEpochSetter:
    def __init__(self, dataset) -> None:
        self.dataset = dataset

    @property
    def inner_length(self):
        return len(self.dataset)

    def __iter__(self):
        self.dataset.epoch = 0
        return self

    def __next__(self):
        self.dataset.epoch += 1
        return self.dataset


class DatasetEpochSetterDataloader:
    def __init__(self, dataset, *args, **kwargs) -> None:
        self.dataset = dataset
        self.dataloader = DataLoader(self.dataset, *args, **kwargs)
        # self.args = args
        # self.kwargs = kwargs

    @property
    def inner_length(self):
        return len(self.dataset)

    def __iter__(self):
        self.dataset.epoch = 0
        return self

    def __next__(self):
        self.dataset.epoch += 1
        return self.dataloader


class LengthLimitDataset:
    def __init__(self, dataset, length) -> None:
        self.dataset = dataset
        self.length = length

    @property
    def inner_length(self):
        return len(self.dataset)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if index < self.length:
            return self.dataset[index]
        else:
            raise IndexError


class JitDataset:
    def __init__(
        self,
        dataset,
    ) -> None:
        self.dataset = dataset

    @property
    def epoch(self):
        return self.dataset.epoch

    @epoch.setter
    def epoch(self, epoch):
        self.dataset.epoch = epoch

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return jax.jit(self.dataset.__getitem__)(index)


def load_all(base_path, subfolder_list, image_id, index=slice(None)):
    filename = image_id + ".npy"
    return {
        subfolder: np.asarray(
            np.load(base_path / subfolder / filename, mmap_mode="r")[index]
        )
        for subfolder in subfolder_list
    }


class SliceDataset:
    def __init__(
        self,
        dataframe: pd.DataFrame,
        root_path: Path,
        subfolder_names: List[str],
        id_col,
        slice_col=None,
        transform=lambda x: x,
    ) -> None:
        self.root_path = Path(root_path)
        self.dataframe = dataframe
        self.subfolder_names = subfolder_names
        self.id_col = id_col
        self.slice_col = slice_col
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        image_id = row[self.id_col]
        slice_index = row[self.slice_col] if self.slice_col else slice(None)
        image_dict = load_all(
            self.root_path, self.subfolder_names, image_id, slice_index
        )
        return self.transform({**image_dict, **row})
