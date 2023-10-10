from typing import Dict, List, Tuple, Union
from collections import defaultdict
import jax
import jax.numpy as jnp
from absl import flags
from common.loss_and_metrics import (
    get_loss_fn,
    get_input_dict_fn,
    get_calibration_metrics_fn,
    get_accumulated_image_metrics_fn,
    get_output_size_metrics_fn,
    get_scalar_metrics_fn,
    get_class_wise_metrics_fn,
    get_class_wise_metrics_aggregator_fn,
    get_class_wise_accuracy_aggregator_fn,
    get_class_wise_metrics_average_calculator_fn,
)

FLAGS = flags.FLAGS


def get_loss_and_metrics(
    *,
    get_samples_fn=None,
    sampling_model,
    mean_var_model,
    sampling_loss,
    class_weights=None,
    binary_threshold=0.5,
    image_output=True,
    calc_metrics=True,
    calc_f_scores=True,
    concatenate_batch_metrics=True,
    stack_batch_metrics=False,
    full_batches_keep_count=1,
    label_rescaler=lambda x: x,
    foreground_classes=None,
    extra_infos=None,
):
    class_weights = None if class_weights is None else jnp.array(class_weights)

    loss_fn = get_loss_fn(
        classifier=FLAGS.classifier,
        logits=FLAGS.logits,
        binary=FLAGS.binary,
        dice_loss=FLAGS.dice_loss,
        class_weights=class_weights,
        stage_aleatoric=FLAGS.stage_aleatoric,
        temperature_scaling=FLAGS.temperature_scaling,
        detach_loss=FLAGS.detach_loss,
        l2=FLAGS.l2,
        sampling_loss=sampling_loss,
    )

    calibration_fn = get_calibration_metrics_fn(
        FLAGS.classifier,
        FLAGS.logits,
        FLAGS.binary,
        FLAGS.stage_aleatoric,
        FLAGS.l2,
    )

    classifier = FLAGS.classifier
    binary = FLAGS.binary
    calc_f_scores = calc_f_scores and calc_metrics and classifier

    output_size_metrics_fn = get_output_size_metrics_fn(
        classifier=FLAGS.classifier,
        binary=FLAGS.binary,
        binary_threshold=binary_threshold,
        stage_aleatoric=FLAGS.stage_aleatoric,
        temperature_scaling=FLAGS.temperature_scaling,
        l2=FLAGS.l2,
    )

    input_wise_scalar_fn = get_scalar_metrics_fn(
        FLAGS.classifier,
        FLAGS.binary,
        class_weights,
        binary_threshold,
    )

    class_wise_metrics_fn = get_class_wise_metrics_fn(
        FLAGS.binary, binary_threshold=binary_threshold
    )

    accumulated_image_metrics_fn = get_accumulated_image_metrics_fn()
    input_dict_fn = get_input_dict_fn()

    def loss_and_metrics_fn(inputs, outputs, targets, extra, variables, rngs):
        (target,) = targets

        if sampling_model:
            (
                mean_output,
                var_output,
                sample_output,
                sample_weights,
                bessel_correction_ddof,
            ) = outputs
            sample_axis = 0
        elif mean_var_model:
            mean_output, var_output = outputs
            sample_output = mean_output
            sample_weights = None
            bessel_correction_ddof = 0
            sample_axis = None
        else:
            mean_output = outputs
            sample_output = mean_output
            var_output = None
            sample_weights = None
            bessel_correction_ddof = 0
            sample_axis = None

        loss = loss_fn(
            target,
            mean_output,
            var_output,
            sample_output,
            sample_weights,
            bessel_correction_ddof,
            sample_axis,
        )

        batch_wise_metrics = {"loss": loss}

        if calc_metrics:
            (
                input_wise_output_size,
                input_class_wise_output_size,
                sample_size,
            ) = output_size_metrics_fn(
                target,
                mean_output=mean_output,
                var_output=var_output,
                sample_output=sample_output,
                sample_weights=sample_weights,
                bessel_correction_ddof=bessel_correction_ddof,
                sample_axis=sample_axis,
            )

            if image_output:
                input_wise_scalar_raw = input_wise_scalar_fn(
                    input_wise_output_size, input_class_wise_output_size
                )

                input_wise_scalar = input_wise_scalar_raw
                input_class_wise_scalar = {}
                input_class_wise_full_size = input_class_wise_output_size
                input_wise_full_size = input_wise_output_size

            else:
                input_wise_scalar = jax.tree_map(
                    lambda x: jnp.squeeze(x, -1), input_wise_output_size
                )

                input_class_wise_scalar = input_class_wise_output_size
                input_class_wise_full_size = {}
                input_wise_full_size = {}

            input_wise_full_size.update(input_dict_fn(inputs))
            input_wise_scalar.update(accumulated_image_metrics_fn(input_wise_full_size))
            input_wise_scalar.update(
                calibration_fn(
                    target,
                    mean_output=mean_output,
                    var_output=var_output,
                    sample_output=sample_output,
                    sample_weights=sample_weights,
                    bessel_correction_ddof=bessel_correction_ddof,
                    sample_axis=sample_axis,
                )
            )

            if classifier:
                if image_output:
                    input_class_wise_scalar.update(
                        class_wise_metrics_fn(
                            input_wise_full_size, input_class_wise_full_size
                        )
                    )
                else:
                    input_class_wise_scalar.update(
                        class_wise_metrics_fn(
                            input_wise_output_size, input_class_wise_scalar
                        )
                    )

            all_metrics = {
                "batch_wise_scalar": batch_wise_metrics,
                "input_wise_scalar": input_wise_scalar,
                "input_class_wise_scalar": input_class_wise_scalar,
                "input_wise_full_size": input_wise_full_size,
                "input_class_wise_full_size": input_class_wise_full_size,
                "sample_size": sample_size,
                "inputs": inputs,
                "variables": variables,
                "rngs": rngs,
            }

        else:
            all_metrics = {"batch_wise_scalar": batch_wise_metrics}

        return loss, all_metrics

    if calc_metrics:

        def get_empty_metrics_aggregator():
            return {
                "batch_wise_scalar": [],
                "input_wise_scalar": [],
                "input_class_wise_scalar": [],
                "input_wise_full_size": [],
                "input_class_wise_full_size": [],
                "sample_size": [],
                "inputs": [],
                "variables": [],
                "rngs": [],
            }

        batch_counts = {
            "batch_wise_scalar": None,
            "input_wise_scalar": None,
            "input_class_wise_scalar": None,
            "input_wise_full_size": full_batches_keep_count,
            "input_class_wise_full_size": full_batches_keep_count,
            "sample_size": full_batches_keep_count,
            "inputs": full_batches_keep_count,
            "variables": 1,
            "rngs": full_batches_keep_count,
        }

        target_cpu = {
            "batch_wise_scalar": False,
            "input_wise_scalar": False,
            "input_class_wise_scalar": False,
            "input_wise_full_size": True,
            "input_class_wise_full_size": True,
            "sample_size": True,
            "inputs": True,
            "variables": True,
            "rngs": True,
        }

    else:

        def get_empty_metrics_aggregator():
            return {
                "batch_wise_scalar": [],
            }

        batch_counts = {
            "batch_wise_scalar": None,
        }

        target_cpu = {
            "batch_wise_scalar": False,
        }

    def metrics_combiner_wrapper(fun, metrics_name):
        def inner_function(*x):
            return fun(x)

        def full_function(metrics_aggregator):
            return (
                label_rescaler(
                    jax.tree_map(inner_function, *metrics_aggregator[metrics_name])
                )
                if metrics_name in metrics_aggregator
                else {}
            )

        return jax.jit(full_function)

    def metrics_extender_wrapper(fun, metrics_name):
        def full_function(metrics_aggregator):
            return (
                jax.tree_map(fun, metrics_aggregator[metrics_name])
                if metrics_name in metrics_aggregator
                else {}
            )

        return jax.jit(full_function)

    metrics_combiner_functions = {
        "batch_wise_scalar": metrics_combiner_wrapper(jnp.stack, "batch_wise_scalar"),
        # "variables": lambda x: x["variables"][0],
        # "rngs": lambda x: x["rngs"],
    }

    if get_samples_fn is not None:

        def get_samples(x):
            variables = x["variables"][0]
            inputs_list = x["inputs"]
            rngs_list = x["rngs"]

            return [
                jax.device_get(get_samples_fn(variables, inputs, rngs)[0])
                for inputs, rngs in zip(inputs_list, rngs_list)
            ]

    metrics_extender_functions = [
        (
            "scalar",
            metrics_extender_wrapper(lambda x: jnp.mean(x, 0), "batch_wise_scalar"),
        ),
    ]

    if concatenate_batch_metrics:
        metrics_combiner_functions.update(
            {
                "input_wise_scalar": metrics_combiner_wrapper(
                    jnp.concatenate, "input_wise_scalar"
                ),
                "input_class_wise_scalar": metrics_combiner_wrapper(
                    jnp.concatenate, "input_class_wise_scalar"
                ),
                "input_wise_full_size": metrics_combiner_wrapper(
                    jnp.concatenate, "input_wise_full_size"
                ),
                "input_class_wise_full_size": metrics_combiner_wrapper(
                    jnp.concatenate, "input_class_wise_full_size"
                ),
                "sample_size": metrics_combiner_wrapper(
                    lambda x: jnp.concatenate(x, axis=1), "sample_size"
                ),
            }
        )

        if get_samples_fn is not None:
            metrics_combiner_functions.update(
                {
                    "sample_full_size": lambda x: jnp.concatenate(
                        get_samples(x), axis=1
                    ),
                }
            )

        metrics_extender_functions.append(
            (
                "scalar",
                metrics_extender_wrapper(lambda x: jnp.mean(x, 0), "input_wise_scalar"),
            )
        )

        if calc_f_scores:
            class_wise_accuracy_aggregator_fn = get_class_wise_accuracy_aggregator_fn(
                (0,)
            )

            accuracy_aggregator_fn = get_class_wise_accuracy_aggregator_fn((0, -1))

            metrics_extender_functions.extend(
                [
                    (
                        "class_wise",
                        lambda x: class_wise_accuracy_aggregator_fn(
                            x["input_class_wise_scalar"]
                        ),
                    ),
                    (
                        "scalar",
                        lambda x: accuracy_aggregator_fn(x["input_class_wise_scalar"]),
                    ),
                ]
            )

            class_wise_metrics_average_calculator_fn = (
                get_class_wise_metrics_average_calculator_fn(
                    binary=binary, foreground_classes=foreground_classes
                )
            )

            if image_output:
                image_class_wise_metrics_aggregator_fn = (
                    get_class_wise_metrics_aggregator_fn()
                )
                metrics_extender_functions.extend(
                    [
                        (
                            "input_class_wise_scalar",
                            lambda x: image_class_wise_metrics_aggregator_fn(
                                x["input_class_wise_scalar"]
                            ),
                        ),
                        (
                            "class_wise",
                            lambda metrics: jax.tree_map(
                                lambda score, validity: jnp.average(score, 0, validity),
                                metrics["input_class_wise_scalar"]["score"],
                                metrics["input_class_wise_scalar"]["validity"],
                            ),
                        ),
                    ]
                )
            else:
                image_class_wise_metrics_aggregator_fn = (
                    get_class_wise_metrics_aggregator_fn(
                        aggregation_axes=0, return_validity=False
                    )
                )
                metrics_extender_functions.append(
                    (
                        "class_wise",
                        lambda x: image_class_wise_metrics_aggregator_fn(
                            x["input_class_wise_scalar"]
                        ),
                    )
                )

            metrics_extender_functions.append(
                (
                    "scalar",
                    lambda metrics: class_wise_metrics_average_calculator_fn(
                        metrics["class_wise"]
                    ),
                )
            )

    if stack_batch_metrics:
        metrics_combiner_functions.update(
            {
                "element_input_wise_scalar": metrics_combiner_wrapper(
                    jnp.stack, "input_wise_scalar"
                ),
                "element_input_class_wise_scalar": metrics_combiner_wrapper(
                    jnp.stack, "input_class_wise_scalar"
                ),
                "element_wise_full_size": metrics_combiner_wrapper(
                    jnp.stack, "input_wise_full_size"
                ),
                "element_class_wise_full_size": metrics_combiner_wrapper(
                    jnp.stack, "input_class_wise_full_size"
                ),
                "element_wise_sample_size": metrics_combiner_wrapper(
                    lambda x: jnp.stack(x, axis=1), "sample_size"
                ),
            }
        )

        if get_samples_fn is not None:
            metrics_combiner_functions.update(
                {
                    "element_sample_wise_full_size": lambda x: jnp.stack(
                        get_samples(x), 1
                    ),
                }
            )

        metrics_extender_functions.append(
            (
                "element_wise_scalar",
                metrics_extender_wrapper(
                    lambda x: jnp.mean(x, 0), "element_input_wise_scalar"
                ),
            )
        )

        if classifier:
            class_wise_accuracy_aggregator_fn = get_class_wise_accuracy_aggregator_fn(
                (0,)
            )

            accuracy_aggregator_fn = get_class_wise_accuracy_aggregator_fn((0, -1))

            metrics_extender_functions.extend(
                [
                    (
                        "element_class_wise_scalar",
                        lambda x: class_wise_accuracy_aggregator_fn(
                            x["element_input_class_wise_scalar"]
                        ),
                    ),
                    (
                        "element_wise_scalar",
                        lambda x: accuracy_aggregator_fn(
                            x["element_input_class_wise_scalar"]
                        ),
                    ),
                ]
            )

        if calc_f_scores:
            if image_output:
                image_class_wise_metrics_aggregator_fn = (
                    get_class_wise_metrics_aggregator_fn()
                )
                metrics_extender_functions.extend(
                    [
                        (
                            "element_input_class_wise_scalar",
                            lambda x: image_class_wise_metrics_aggregator_fn(
                                x["element_input_class_wise_scalar"]
                            ),
                        ),
                        (
                            "element_class_wise_scalar",
                            lambda metrics: jax.tree_map(
                                lambda score, validity: jnp.average(score, 0, validity),
                                metrics["element_input_class_wise_scalar"]["score"],
                                metrics["element_input_class_wise_scalar"]["validity"],
                            ),
                        ),
                    ]
                )
            else:
                class_wise_metrics_aggregator_fn = get_class_wise_metrics_aggregator_fn(
                    aggregation_axes=0, return_validity=False
                )
                metrics_extender_functions.append(
                    (
                        "element_class_wise_scalar",
                        lambda x: class_wise_metrics_aggregator_fn(
                            x["element_input_class_wise_scalar"]
                        ),
                    )
                )

            class_wise_metrics_average_calculator_fn = (
                get_class_wise_metrics_average_calculator_fn(
                    binary=binary, foreground_classes=foreground_classes
                )
            )
            metrics_extender_functions.append(
                (
                    "element_wise_scalar",
                    lambda metrics: class_wise_metrics_average_calculator_fn(
                        metrics["element_class_wise_scalar"]
                    ),
                )
            )

    if extra_infos is not None:
        for key, value in extra_infos.items():
            metrics_extender_functions.append((key, lambda x: value))

    def metrics_aggregator(step, aggregator, metrics):
        for key in aggregator:
            if batch_counts[key] is None or step < batch_counts[key]:
                metric = (
                    jax.device_get(metrics[key]) if target_cpu[key] else metrics[key]
                )
                aggregator[key].append(metric)

    def metrics_combiner(metrics_aggregator):
        result_dict = defaultdict(
            dict,
            (
                (key, function(metrics_aggregator))
                for key, function in metrics_combiner_functions.items()
            ),
        )

        for key, function in metrics_extender_functions:
            result_dict[key].update(function(result_dict))

        result_dict = jax.device_get(
            {key: value for key, value in result_dict.items() if len(value)}
        )

        return result_dict

    def metrics_plotter(name, tqdm_object, step, metrics):
        loss = metrics["batch_wise_scalar"]["loss"]
        tqdm_object.set_postfix({"loss": loss.item()})

    return (
        loss_and_metrics_fn,
        get_empty_metrics_aggregator,
        metrics_aggregator,
        metrics_combiner,
        metrics_plotter,
    )
