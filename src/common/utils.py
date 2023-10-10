from typing import Any, Sequence
from jax._src import dtypes
import jax.numpy as jnp
from jax import random
import more_itertools
from einops import rearrange


DTypeLikeFloat = Any


def rayleigh(
    key, scale, shape: Sequence[int] = (), dtype: DTypeLikeFloat = dtypes.float_
):
    return random.weibull_min(
        key,
        scale * jnp.sqrt(2),
        2,
        shape=shape,
        dtype=dtype,
    )


def split_attenuation_channels(tensor):
    return rearrange(tensor, "... (u c) -> u ... c", u=2)


def split_temperature_channel(tensor):
    return jnp.split(tensor, [tensor.shape[-1] - 1], -1)


def rng_generator(rng):
    while True:
        yield_rng, rng = random.split(rng)
        yield yield_rng


def zip_dict_iterable(dict_iterable):
    head, dict_iterable = more_itertools.spy(dict_iterable)
    keys = head[0].keys()
    value_iterables = more_itertools.unzip(
        ([the_dict[key] for key in keys] for the_dict in dict_iterable)
    )
    return dict(zip(keys, value_iterables))
