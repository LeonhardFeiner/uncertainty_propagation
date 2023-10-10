from functools import partial
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple, Type, Union, Any
import numpy as np
import jax.numpy as jnp
import jax
from flax.training import train_state
from flax.core.frozen_dict import FrozenDict
from flax import linen as nn
import optax
from models.layer import transposed_conv2d, conv2d
from models.common import replace_equal_shape
from models.pretrain_weights import get_pretrained_weights

ModuleDef = Callable[..., Callable]


class AdaptedGroupNorm(nn.GroupNorm):
    def __init__(self, use_running_average=None, **kwargs):
        super().__init__(**kwargs)


class EncoderSmall(nn.Module):
    dropout_rate: float
    channels: int = 256

    activation: Callable = nn.relu
    conv_cls: ModuleDef = conv2d
    norm_cls: ModuleDef = nn.BatchNorm
    dropout_cls: ModuleDef = nn.Dropout

    # pylint: disable=arguments-differ
    @nn.compact
    def __call__(self, x, training=False, deterministic=None):
        use_running_average = not training
        deterministic = not training if deterministic is None else deterministic

        x = self.conv_cls(self.channels, 3, 2, "SAME", bias=True)(x)
        x = self.activation(x)

        if self.dropout_rate:
            x = self.dropout_cls(self.dropout_rate, deterministic=deterministic)(x)

        x = self.conv_cls(self.channels * 2, 3, 2, "SAME", bias=True)(x)
        x = self.activation(x)

        if self.dropout_rate:
            x = self.dropout_cls(self.dropout_rate, deterministic=deterministic)(x)
        x = self.conv_cls(self.channels * 4, 3, 2, "LIKE", bias=True)(x)
        x = self.activation(x)

        if self.dropout_rate:
            x = self.dropout_cls(self.dropout_rate, deterministic=deterministic)(x)

        return x


class DecoderSmall(nn.Module):
    dropout_rate: float
    channels: int = 256
    out_channels: int = 1

    activation: Callable = nn.relu
    transposed_conv_cls: ModuleDef = transposed_conv2d
    dropout_cls: ModuleDef = nn.Dropout

    # pylint: disable=arguments-differ
    @nn.compact
    def __call__(self, x, training=False, deterministic=None):
        use_running_average = not training
        deterministic = not training if deterministic is None else deterministic

        x = self.transposed_conv_cls(self.channels * 2, 3, 2, "LIKE", bias=True)(x)
        x = self.activation(x)

        if self.dropout_rate:
            x = self.dropout_cls(self.dropout_rate, deterministic=deterministic)(x)
        x = self.transposed_conv_cls(self.channels, 3, 2, "SAME", bias=True)(x)
        x = self.activation(x)

        if self.dropout_rate:
            x = self.dropout_cls(self.dropout_rate, deterministic=deterministic)(x)

        x = self.transposed_conv_cls(self.out_channels, 3, 2, "SAME", bias=True)(x)
        return x


class EncoderLarge(nn.Module):
    dropout_rate: float = 0.5
    channels: int = 64
    blocks: int = 5
    dropout_blocks: int = 2

    activation: Callable = nn.relu
    conv_cls: ModuleDef = nn.Conv
    norm_cls: ModuleDef = nn.BatchNorm
    pool_fn: Callable = partial(nn.avg_pool, window_shape=(2, 2), strides=(2, 2))
    dropout_cls: ModuleDef = nn.Dropout

    def get_block(
        self,
        channels,
        use_pool,
        use_dropout,
        use_running_average,
        deterministic,
    ):
        kernel_size = (3, 3)

        layers = [
            self.conv_cls(channels, kernel_size=kernel_size),
            self.activation,
            self.conv_cls(channels, kernel_size=kernel_size),
            self.norm_cls(use_running_average=use_running_average),
            self.activation,
        ]

        if use_pool:
            layers = [self.pool_fn] + layers

        if use_dropout:
            layers += [self.dropout_cls(self.dropout_rate, deterministic=deterministic)]

        return nn.Sequential(layers)

    # pylint: disable=arguments-differ
    @nn.compact
    def __call__(self, x, training=False, deterministic=None):
        use_running_average = not training
        deterministic = not training if deterministic is None else deterministic
        kwargs = dict(
            use_running_average=use_running_average, deterministic=deterministic
        )

        latents = []
        for i in range(self.blocks):
            x = self.get_block(
                2**i * self.channels,
                use_pool=i != 0,
                use_dropout=i >= self.blocks - self.dropout_blocks,
                **kwargs,
            )(x)
            latents += [x]

        return latents


class DecoderBlock(nn.Module):
    dropout_rate: float = 0.5
    channels: int = 64

    activation: Callable = nn.relu
    conv_cls: ModuleDef = nn.Conv
    norm_cls: ModuleDef = nn.BatchNorm
    concatenation_fn: Callable = partial(jnp.concatenate, axis=3)
    resize_fn: Callable = partial(jax.image.resize, method="nearest")
    shape_calculator: Callable = jnp.shape

    # pylint: disable=arguments-differ
    @nn.compact
    def __call__(self, x, skip, use_running_average):
        batch_size, *spatial, channels = self.shape_calculator(x)
        new_shape = batch_size, *(n * 2 for n in spatial), channels

        x = self.resize_fn(x, shape=new_shape)
        x = self.conv_cls(self.channels, kernel_size=(2, 2))(x)
        x = self.activation(x)
        x = self.concatenation_fn([skip, x])
        x = self.conv_cls(self.channels, kernel_size=(3, 3))(x)
        x = self.activation(x)
        x = self.conv_cls(self.channels, kernel_size=(3, 3))(x)
        x = self.norm_cls(use_running_average=use_running_average)(x)
        x = self.activation(x)
        return x


class DecoderLarge(nn.Module):
    dropout_rate: float = 0.5
    channels: int = 64
    out_channels: int = 1

    conv_cls: ModuleDef = nn.Conv
    decoder_block: ModuleDef = DecoderBlock

    # pylint: disable=arguments-differ
    @nn.compact
    def __call__(self, latent, training=False, deterministic=None):
        use_running_average = not training
        deterministic = not training if deterministic is None else deterministic

        *latent, x = latent

        num_skip = len(latent)
        for i, skip in enumerate(reversed(latent)):
            factor = 2 ** (num_skip - i - 1)
            x = self.decoder_block(self.dropout_rate, self.channels * factor)(
                x, skip, use_running_average
            )

        return self.conv_cls(self.out_channels, kernel_size=(1, 1))(x)


class EncoderDummy(nn.Module):
    dropout_rate: float = 0
    channels: int = 1

    def __call__(self, x, training=False, deterministic=None) -> Any:
        return x


class DecoderDummy(nn.Module):
    dropout_rate: float = 0.5
    channels: int = 64
    out_channels: int = 1

    conv_cls: ModuleDef = nn.Conv

    @nn.compact
    def __call__(self, x, training=False, deterministic=None) -> Any:
        kernel_size = (3, 3)

        return self.conv_cls(self.out_channels, kernel_size=kernel_size)(x)


class Autoencoder(nn.Module):
    dropout_rate: float = 0.0
    encoder_type: ModuleDef = EncoderSmall
    decoder_type: ModuleDef = DecoderSmall
    channels: int = 256
    out_channels: Optional[int] = None
    out_channel_factor: Optional[int] = 1
    residual: bool = False

    # pylint: disable=arguments-differ
    @nn.compact
    def __call__(self, x, training=False, deterministic=None):
        out_channels = (
            x.shape[-1] * self.out_channel_factor
            if self.out_channels is None
            else self.out_channels
        )
        encoder = self.encoder_type(self.dropout_rate, self.channels)
        decoder = self.decoder_type(self.dropout_rate, self.channels, out_channels)
        latent = encoder(x, training=training, deterministic=deterministic)
        decoded = decoder(latent, training=training, deterministic=deterministic)

        if self.residual:
            return decoded.at[..., : x.shape[-1]].add(x)
        else:
            return decoded


def calc_blocks(data_shape):
    return min(int(np.log2(np.min(data_shape[1:-1]))) - 1, 5)


class TrainState(train_state.TrainState):
    batch_stats: FrozenDict[str, Any] = FrozenDict()

    @property
    def variables(self):
        return FrozenDict(params=self.params, batch_stats=self.batch_stats)


def variables_extractor(state):
    return FrozenDict(batch_stats=state.batch_stats), state.params


def get_model(
    out_channels,
    blocks=None,
    channels=None,
    use_batch_norm=False,
    use_max_pool=False,
    residual=False,
    dropout_rate=0,
    model_name="unet",
    batch_norm_reduction_axis_name=None,
):
    norm_cls = (
        partial(nn.BatchNorm, axis_name=batch_norm_reduction_axis_name)
        if use_batch_norm
        else AdaptedGroupNorm
    )
    pool_fn = partial(
        nn.max_pool if use_max_pool else nn.avg_pool,
        window_shape=(2, 2),
        strides=(2, 2),
    )

    if model_name == "unet":
        if blocks is None:
            blocks = 5
        if channels is None:
            channels = 64
        kwargs = dict(
            encoder_type=partial(
                EncoderLarge, blocks=blocks, norm_cls=norm_cls, pool_fn=pool_fn
            ),
            decoder_type=partial(
                DecoderLarge, decoder_block=partial(DecoderBlock, norm_cls=norm_cls)
            ),
            channels=channels,
        )
    elif model_name == "simpleauto":
        if channels is None:
            channels = 256
        kwargs = dict(
            encoder_type=partial(EncoderSmall, norm_cls=norm_cls),
            decoder_type=DecoderSmall,
            channels=channels,
        )
    elif model_name == "dummyauto":
        kwargs = dict(encoder_type=EncoderDummy, decoder_type=DecoderDummy)
    else:
        raise NotImplementedError

    return Autoencoder(
        dropout_rate, out_channels=out_channels, residual=residual, **kwargs
    )


def get_model_state(
    model_key,
    data_shape,
    out_channels,
    batches_per_epoch,
    wrapper=lambda x: x,
    model_name="unet",
    pretrained=False,
    use_batch_norm=False,
    use_max_pool=False,
    dropout_rate=0,
    variables=None,
    **kwargs
):

    model = wrapper(
        get_model(
            out_channels,
            **kwargs,
            model_name=model_name,
            use_batch_norm=use_batch_norm,
            use_max_pool=use_max_pool,
            dropout_rate=dropout_rate,
        )
    )
    if variables is None:
        init_data = jnp.ones(data_shape, jnp.float32)
        variables = model.init({"dropout": model_key, "params": model_key}, init_data)

        if pretrained:
            pretrained_variables = get_pretrained_weights(
                model_name, use_max_pool, use_batch_norm, dropout_rate
            )
            variables = replace_equal_shape(variables, pretrained_variables)

    learning_rate = 1e-4

    state = TrainState.create(
        apply_fn=model.apply,
        **variables,
        tx=optax.adamw(optax.exponential_decay(learning_rate, batches_per_epoch, 0.9)),
    )

    if model_name == "unet":
        train_kwargs = dict(
            training=True,
            mutable=["batch_stats"],
            variables_extractor=variables_extractor,
        )
        val_kwargs = dict(
            training=False,
            variables_extractor=variables_extractor,
        )
    elif model_name == "simpleauto":
        train_kwargs = dict(training=True)
        val_kwargs = dict(training=False)
    elif model_name == "dummyauto":
        train_kwargs = dict()
        val_kwargs = dict()
    else:
        raise NotImplementedError

    return model, state, train_kwargs, val_kwargs
