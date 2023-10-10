from abc import ABC, abstractmethod
from typing import Any, Tuple
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn


class SamplerBase(nn.Module, ABC):
    def get_sliced_samples(
        self, model, parameters, *args, sample_slice=slice(None), **kwargs
    ):
        carry, array, weights, ddof = self(model, parameters, *args, **kwargs)
        returned_weights = None if weights is None else weights[sample_slice]
        return self.get_samples(carry, array[sample_slice]), returned_weights, ddof

    @abstractmethod
    def __call__(self, model, parameters, *args, **kwargs) -> Tuple[Any, Any, Any, int]:
        pass

    @staticmethod
    @abstractmethod
    def get_samples(carry, array):
        pass

    def calc_mean_var(self, predictions, ddof, weights):
        mean_prediction = jnp.average(predictions, 0, weights)

        if weights is None:
            var_prediction = jnp.var(predictions, 0, ddof=ddof)
        else:
            bessel_correction = len(predictions) / (len(predictions) - ddof)
            var_prediction = (
                jnp.average(jnp.square(predictions - mean_prediction), 0, weights)
                * bessel_correction
            )
        return mean_prediction, var_prediction


class DiagNormalMonteCarloSampler(SamplerBase):
    num_samples: int
    epsilon: float = 1e-8

    def __call__(self, model, parameters, *args, **kwargs):
        mean, var = parameters
        rngs = random.split(self.make_rng("data_sample"), self.num_samples)
        std = jnp.sqrt(var + self.epsilon)
        carry = mean, std
        weights = None
        ddof = 1
        return carry, rngs, weights, ddof

    @staticmethod
    def _get_sample(mean, std, rng):
        shape = np.broadcast_shapes(mean.shape, std.shape)
        noise = random.normal(rng, shape, dtype=mean.dtype)
        return mean + noise * std

    @staticmethod
    def _get_samples(mean, std, rngs):
        return jax.vmap(DiagNormalMonteCarloSampler._get_sample, (None, None, 0))(
            mean, std, rngs
        )

    @staticmethod
    def get_samples(carry, array):
        mean, std = carry
        return DiagNormalMonteCarloSampler._get_samples(mean, std, array)
