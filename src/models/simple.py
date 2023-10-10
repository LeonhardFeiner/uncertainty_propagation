from typing import Callable, Dict, List, NamedTuple, Tuple, Type, Union, Any
import jax.numpy as jnp
from flax import linen as nn
from flax.core.frozen_dict import FrozenDict
from flax.training import train_state
import optax
from einops import rearrange, reduce, repeat


class TrainState(train_state.TrainState):
    batch_stats: FrozenDict[str, Any] = FrozenDict()

    @property
    def variables(self):
        return FrozenDict(params=self.params, batch_stats=self.batch_stats)


train_kwargs = dict(training=True)
val_kwargs = dict(training=False)


class GlobalPool(nn.Module):
    operator: Union[str, Callable] = "mean"

    def __call__(self, x):
        return reduce(x, "b ... c ->b c", self.operator)


class Flatten(nn.Module):
    def __call__(self, x):
        return rearrange(x, "b ... -> b (...)")


class SimplePredictor(nn.Module):
    num_classes: int = 1
    global_pool: bool = False
    dropout_rate: float = 0.0
    channels: int = 64
    conv_layer: Type = nn.Conv
    activation: Callable = staticmethod(nn.relu)
    dropout_layer: Type = nn.Dropout
    global_pooling_layer: Type = GlobalPool
    flatten_layer: Type = Flatten
    dense: Type = nn.Dense

    @nn.compact
    def __call__(self, x, training=False):

        x = self.conv_layer(self.channels, (3, 3), 2, "SAME", use_bias=True)(x)
        x = self.activation(x)

        if self.dropout_rate:
            x = self.dropout_layer(self.dropout_rate, deterministic=not training)(x)

        x = self.conv_layer(self.channels * 2, (3, 3), 2, "SAME", use_bias=True)(x)
        x = self.activation(x)

        if self.dropout_rate:
            x = self.dropout_layer(self.dropout_rate, deterministic=not training)(x)

        x = self.conv_layer(self.channels * 4, (3, 3), 2, "SAME", use_bias=True)(x)
        x = self.activation(x)

        if self.dropout_rate:
            x = self.dropout_layer(self.dropout_rate, deterministic=not training)(x)

        if self.global_pool:
            x = self.global_pooling_layer()(x)
        else:
            x = self.flatten_layer()(x)

        output = self.dense(self.num_classes)(x)

        return output


def get_model(num_classes):
    return SimplePredictor(
        num_classes=num_classes,
    )


def get_model_variables(
    rng, data_shape, num_classes, wrapper=lambda x: x, variables=None
):
    model = wrapper(get_model(num_classes))
    if variables is None:
        init_data = jnp.ones(data_shape, jnp.float32)
        variables = model.init({"dropout": rng, "params": rng}, init_data)
    return model, variables


def get_model_state(*args, batches_per_epoch, learning_rate=1e-4, **kwargs):
    model, variables = get_model_variables(*args, **kwargs)
    state = TrainState.create(
        apply_fn=model.apply,
        **variables,
        tx=optax.adamw(optax.exponential_decay(learning_rate, batches_per_epoch, 0.9)),
    )

    return model, state, train_kwargs, val_kwargs
