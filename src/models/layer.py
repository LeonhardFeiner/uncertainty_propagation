from typing import Any, Callable, Sequence, Optional, Tuple, Union, Iterable
from numbers import Number

import flax.linen as nn
import flax.linen.initializers as initializers

import jax.numpy as jnp


# calculate SAME-like symmetric padding for a convolution
def get_like_padding(kernel_size: int, stride: int = 1, dilation: int = 1, **_) -> int:
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


def to_tuple(v: Union[Tuple[Number, ...], Number, Iterable], n: int):
    """Converts input to tuple."""
    if isinstance(v, tuple):
        return v
    elif isinstance(v, Number):
        return (v,) * n
    else:
        return tuple(v)


PRNGKey = Any
Shape = Tuple[int]
Dtype = Any  # this could be a real type?
Array = Any

default_kernel_init = initializers.kaiming_normal()


def conv2d(
    features: int,
    kernel_size: int,
    stride: Optional[int] = None,
    padding: Union[str, Tuple[int, int]] = 0,
    dilation: Optional[int] = None,
    groups: int = 1,
    bias: bool = False,
    dtype: Dtype = jnp.float32,
    precision: Any = None,
    name: Optional[str] = None,
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init,
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.zeros,
):

    stride = stride or 1
    dilation = dilation or 1
    if isinstance(padding, str):
        if padding == "LIKE":
            padding = get_like_padding(kernel_size, stride, dilation)
            padding = to_tuple(padding, 2)
            padding = [padding, padding]
    else:
        padding = to_tuple(padding, 2)
        padding = [padding, padding]
    return nn.Conv(
        features=features,
        kernel_size=to_tuple(kernel_size, 2),
        strides=to_tuple(stride, 2),
        padding=padding,
        kernel_dilation=to_tuple(dilation, 2),
        feature_group_count=groups,
        use_bias=bias,
        dtype=dtype,
        precision=precision,
        name=name,
        kernel_init=kernel_init,
        bias_init=bias_init,
    )


def transposed_conv2d(
    features: int,
    kernel_size: int,
    stride: Optional[int] = None,
    padding: Union[str, Tuple[int, int]] = 0,
    dilation: Optional[int] = None,
    groups: int = 1,
    bias: bool = False,
    dtype: Dtype = jnp.float32,
    precision: Any = None,
    name: Optional[str] = None,
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init,
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.zeros,
):

    stride = stride or 1
    dilation = dilation or 1
    if isinstance(padding, str):
        if padding == "LIKE":
            padding = get_like_padding(kernel_size, stride, dilation)
            padding = to_tuple(padding, 2)
            padding = [padding, padding]
    else:
        padding = to_tuple(padding, 2)
        padding = [padding, padding]
    return nn.ConvTranspose(
        features=features,
        kernel_size=to_tuple(kernel_size, 2),
        strides=to_tuple(stride, 2),
        padding=padding,
        kernel_dilation=to_tuple(dilation, 2),
        # feature_group_count=groups,
        use_bias=bias,
        dtype=dtype,
        precision=precision,
        name=name,
        kernel_init=kernel_init,
        bias_init=bias_init,
    )
