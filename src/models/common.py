import dataclasses
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict


def custom_split(array, max_split_size=0, prefer_single_remain=True):
    if prefer_single_remain and len(array) == max_split_size:
        first_half = 0
    else:
        first_half = max_split_size and len(array) // max_split_size * max_split_size

    vectorized_flat, remaining = jnp.split(array, [first_half])
    vectorized = jnp.reshape(
        vectorized_flat,
        (
            vectorized_flat.size and -1,
            max_split_size,
            *vectorized_flat.shape[1:],
        ),
    )
    return vectorized, remaining


def add_outer_layer(variables, layer_name="model"):
    return FrozenDict(
        {key: FrozenDict(**{layer_name: value}) for key, value in variables.items()}
    )


def remove_outer_layer(variables, layer_name="model"):
    assert all(len(variable_dict) == 1 for variable_dict in variables.values())
    return FrozenDict({key: value[layer_name] for key, value in variables.items()})


def get_params(model):
    return {
        field.name: model.__dict__[field.name]
        for field in dataclasses.fields(model)
        if field.name not in {"name", "parent"}
    }


def conditional_replace(key, init_array, load_dict, renamer=lambda x: x):
    key = renamer(key)

    if key not in load_dict:
        print(f"{key} not in pretrained weights")
        return init_array
    else:
        load_array = load_dict[key]
    if init_array.shape != load_array.shape:
        print(
            f"{key} has different shapes in initialization {init_array.shape} and pretraining {load_array.shape}"
        )
        return init_array
    else:
        return load_array


def replace_equal_shape(init_state, load_state, renamer=lambda x: x):
    flat_init_state = flatten_dict(unfreeze(init_state))
    flat_load_state = flatten_dict(unfreeze(load_state))

    flat_new_state = {
        key: conditional_replace(key, init_array, flat_load_state)
        for key, init_array in flat_init_state.items()
    }

    return freeze(unflatten_dict(flat_new_state))
