import jax
from flax import linen as nn


def get_apply_samples_fn(model, sample_slice=slice(8), jit=True, **kwargs):
    assert hasattr(model, "get_samples")

    def apply_samples(variables, data, rngs):
        def get_samples_inner(model, parameters, *args, **kwargs):
            return model.get_samples(
                parameters, *args, sample_slice=sample_slice, **kwargs
            )

        return nn.apply(get_samples_inner, model)(
            variables,
            *data,
            rngs=rngs,
            **kwargs,
        )

    if jit:
        return jax.jit(apply_samples)
    else:
        return apply_samples
