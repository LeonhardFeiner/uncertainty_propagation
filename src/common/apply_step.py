import jax
from flax.core.frozen_dict import FrozenDict


def default_variables_extractor(state):
    return FrozenDict(), state.params


def default_batch_modifier(batch, rng):
    inputs, outputs = batch
    return (inputs,), (outputs,)


def get_apply_step(
    loss_and_metrics_fn,
    batch_modifier=default_batch_modifier,
    variables_extractor=default_variables_extractor,
    required_rngs=("dropout", "data_sample"),
    mutable=False,
    update_state=True,
    device=None,
    jit=True,
    **apply_kwargs,
):

    if not update_state:
        mutable = False

    def apply_step(state, batch, rng):

        inputs, targets, extra = batch_modifier(rng, jax.device_put(batch, device))
        rngs = dict(
            zip(
                required_rngs,
                jax.random.split(jax.device_put(rng, device), len(required_rngs)),
            )
        )
        remaining_variables, params = jax.device_put(variables_extractor(state), device)

        def loss_fn(params):
            apply_result = state.apply_fn(
                FrozenDict(params=params, **remaining_variables),
                *inputs,
                mutable=mutable,
                rngs=rngs,
                **apply_kwargs,
            )
            if mutable:
                outputs, variables = apply_result
            else:
                outputs = apply_result
                variables = {}

            loss, metrics = loss_and_metrics_fn(
                inputs, outputs, targets, extra, variables, rngs
            )
            return loss, (variables, metrics)

        if not update_state:
            _, (_, metrics) = loss_fn(params)
            new_state = state
        else:
            grads, (variables, metrics) = jax.grad(loss_fn, has_aux=True)(params)
            new_state = state.apply_gradients(grads=grads, **variables)

        return new_state, metrics

    if jit:
        return jax.jit(apply_step, donate_argnums=0, device=device)
    else:
        return apply_step
