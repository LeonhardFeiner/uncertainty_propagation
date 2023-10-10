from typing import Any, Callable, Dict, List, Tuple, Union
from functools import partial
import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
from flax.core.frozen_dict import FrozenDict
from flax.training import train_state
import optax

import jax_resnet
from jax_resnet import pretrained_resnest, ResNeSt50
from jax_resnet import pretrained_resnet, ResNet50
from models.common import replace_equal_shape


class TrainState(train_state.TrainState):
    batch_stats: FrozenDict[str, Any] = FrozenDict()

    @property
    def variables(self):
        return FrozenDict(params=self.params, batch_stats=self.batch_stats)


def variables_extractor(state):
    return FrozenDict(batch_stats=state.batch_stats), state.params


train_kwargs_batch_norm = dict(
    mutable=["batch_stats"],
    variables_extractor=variables_extractor,
)
train_kwargs_no_batch_norm = dict(
    variables_extractor=variables_extractor,
)
val_kwargs = dict(variables_extractor=variables_extractor)


class AdaptedGroupNorm(nn.GroupNorm):
    def __init__(self, use_running_average=None, **kwargs):
        super().__init__(**kwargs)


def get_model(
    num_classes,
    name="ResNeSt",
    size=50,
    use_batch_norm=False,
    use_max_pool=False,
    batch_norm_reduction_axis_name=None,
):

    model_class = getattr(jax_resnet, f"{name}{size}")
    norm_cls = (
        partial(nn.BatchNorm, momentum=0.9, axis_name=batch_norm_reduction_axis_name)
        if use_batch_norm
        else AdaptedGroupNorm
    )
    pool_fn = partial(
        partial(
            nn.max_pool, window_shape=(3, 3), strides=(2, 2), padding=((1, 1), (1, 1))
        )
        if use_max_pool
        else nn.avg_pool,
        window_shape=(3, 3),
        strides=(2, 2),
        padding=((1, 1), (1, 1)),
    )
    model = model_class(
        n_classes=num_classes,
        pool_fn=pool_fn,
        norm_cls=norm_cls,
    )
    return model


def default_wrapper(x, y=None):
    if y is None:
        return x
    else:
        return x, y


def get_model_variables(
    rng,
    data_shape,
    num_classes,
    name="ResNeSt",
    size=50,
    wrapper=default_wrapper,
    pretrained=True,
    variables=None,
    **kwargs,
):
    model: nn.Module = wrapper(get_model(num_classes, name=name, size=size, **kwargs))

    if variables is None:
        init_data = jnp.ones(data_shape, jnp.float32)
        variables = model.init({"dropout": rng, "params": rng}, init_data)

        if pretrained:
            if name == "ResNeSt":
                _, pretrained_variables = wrapper(*pretrained_resnest(size))
            elif name == "ResNet":
                _, pretrained_variables = wrapper(*pretrained_resnet(size))
            else:
                raise NotImplementedError
            variables = replace_equal_shape(
                variables, pretrained_variables, layer_key_renamer
            )

    return model, variables


def get_model_state(*args, batches_per_epoch, use_batch_norm, **kwargs):
    model, variables = get_model_variables(
        *args, **kwargs, use_batch_norm=use_batch_norm
    )

    state = TrainState.create(
        apply_fn=model.apply,
        **variables,
        tx=optax.adamw(optax.exponential_decay(1e-4, batches_per_epoch, 0.9)),
    )

    train_kwargs = (
        train_kwargs_batch_norm if use_batch_norm else train_kwargs_no_batch_norm
    )
    return model, state, train_kwargs, val_kwargs


def layer_key_renamer(key):
    if any(
        "GroupNorm" in key_element or "LayerNorm" in key_element for key_element in key
    ):
        new_key = key
        key = tuple(
            key_element.replace("GroupNorm", "BatchNorm")
            .replace("LayerNorm", "BatchNorm")
            .replace("Adapted", "")
            for key_element in key
        )
        print(f"use {key} as pretraining for {new_key}")

    return key
