from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Tuple, Type, Union
import jax
import jax.numpy as jnp
from flax import linen as nn
from einops import rearrange
from models.common import custom_split
from models.sampler import SamplerBase
from more_itertools import unique_everseen, one


class ModelContainer(nn.Module):
    model: nn.Module
    args: Tuple
    kwargs: Dict

    def __call__(self, x):
        retval = self.model(x, *self.args, **self.kwargs)
        return retval


class InnerSamplePropagator(nn.Module):
    model: nn.Module
    sample_function: Callable[[Tuple, jax.Array], jax.Array]
    variable_axes: Dict = field(
        default_factory=lambda: {"params": None, "batch_stats": None}
    )
    split_rngs: Dict = field(
        default_factory=lambda: {"params": False, "dropout": False}
    )
    out_axes: Union[int, Tuple[int, ...]] = 1
    axis_name: str = "propagator_samples"

    @nn.compact
    def __call__(self, carry, inputs):
        sampler_carry, args, kwargs = carry
        samples = self.sample_function(sampler_carry, inputs)
        # return carry, jax.vmap(lambda x: self.model(x, *args, **kwargs))(samples)

        vmap_model = nn.vmap(
            ModelContainer,
            variable_axes=self.variable_axes,
            split_rngs=self.split_rngs,
            out_axes=self.out_axes,
            axis_name=self.axis_name,
        )(self.model, args, kwargs)
        return (
            carry,
            vmap_model(samples),
        )


ScanSamplePropagator: Type = nn.scan(
    InnerSamplePropagator,
    variable_broadcast=("params", "batch_stats"),
    split_rngs={"params": False},
    in_axes=0,
    out_axes=1,
)


class PropagatorBase(nn.Module, ABC):
    model: nn.Module

    @abstractmethod
    def predict_samples(self, parameters, *args, **kwargs):
        ...

    @abstractmethod
    def predict_normal(self, parameters, *args, **kwargs):
        ...

    @abstractmethod
    def predict(self, parameters, *args, **kwargs):
        ...

    def __call__(
        self,
        parameters,
        *args,
        predict_parameters=True,
        predict_samples=False,
        **kwargs
    ):
        assert predict_parameters or predict_samples

        if predict_parameters and predict_samples:
            return self.predict(parameters, *args, **kwargs)
        elif predict_parameters:
            return self.predict_normal(parameters, *args, **kwargs)
        else:
            return self.predict_samples(parameters, *args, **kwargs)


class SamplePropagator(PropagatorBase):
    sampler: SamplerBase
    max_vectorized_samples: int

    def setup(self) -> None:
        self.scan_sample_propagator = ScanSamplePropagator(
            self.model,
            self.sampler.get_samples,
        )
        self.sample_propagator = InnerSamplePropagator(
            self.model,
            self.sampler.get_samples,
        )
        return super().setup()

    def get_samples(self, parameters, *args, sample_slice=slice(None), **kwargs):
        return self.sampler.get_sliced_samples(
            self.model, parameters, sample_slice=sample_slice, **kwargs
        )

    def predict_samples(self, parameters, *args, **kwargs):
        sampler_carry, array, weights, ddof = self.sampler(
            self.model, parameters, *args, **kwargs
        )

        vectorized, remaining = custom_split(
            array, max_split_size=self.max_vectorized_samples
        )

        carry = sampler_carry, args, kwargs
        result = []
        if len(vectorized):
            _, predictions_raw = self.scan_sample_propagator(carry, vectorized)
            result += [rearrange(predictions_raw, "b m v ... -> (m v) b ...")]
        if len(remaining):
            _, remaining_predictions_raw = self.sample_propagator(carry, remaining)
            result += [rearrange(remaining_predictions_raw, "b m ... -> m b ...")]

        predictions = jnp.concatenate(result)

        return predictions, weights, ddof

    def predict_normal(self, parameters, *args, **kwargs):
        mean_prediction, var_prediction, predictions, weights, ddof = self.predict(
            parameters, *args, **kwargs
        )

        return mean_prediction, var_prediction

    def predict(self, parameters, *args, **kwargs):
        predictions, weights, ddof = self.predict_samples(parameters, *args, **kwargs)

        mean_prediction, var_prediction = self.sampler.calc_mean_var(
            predictions, ddof, weights
        )

        return (
            mean_prediction,
            var_prediction,
            predictions,
            weights,
            ddof,
        )
