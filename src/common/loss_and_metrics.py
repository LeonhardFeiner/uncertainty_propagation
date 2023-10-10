from typing import Dict, List, Tuple, Union
from collections import defaultdict
import jax
import jax.numpy as jnp
from flax import linen as nn
from absl import flags
from common.types import *
from common.loss import (
    l1 as l1_fn,
    l2 as l2_fn,
    ce_with_logits,
    bce_with_logits,
    ce_with_confidence,
    bce_with_confidence,
    f_score_loss,
    get_weighted_mean_variance,
    get_one_hot,
    get_tp_gt_pr_tn_tot_from_one_hot,
    f_score_and_validity_from_counts,
    accuracy_from_counts,
)
from common.metrics import (
    compute_attenuated_l2_regression_metrics,
    compute_attenuated_l1_regression_metrics,
    compute_bce_classification_metrics_with_confidence,
    compute_ce_classification_metrics_with_confidence,
    compute_regression_metrics,
    get_confidence_entropy_mutual,
    class_entropy,
)
from common.utils import split_attenuation_channels, split_temperature_channel

FLAGS = flags.FLAGS


def get_output_size_metrics_fn(
    *,
    classifier,
    binary,
    binary_threshold,
    stage_aleatoric,
    temperature_scaling,
    l2,
):
    def full_size_metrics(
        target,
        mean_output,
        var_output=None,
        sample_output=None,
        sample_weights=None,
        bessel_correction_ddof=0,
        sample_axis=None,
    ):
        gt = target
        if classifier:
            if temperature_scaling:
                raw_logits, raw_temperature = split_temperature_channel(mean_output)
                temperature = jax.nn.softplus(raw_temperature)
                logits = raw_logits / temperature
                if var_output is not None:
                    class_wise_logit_var_raw, _ = split_temperature_channel(var_output)
                    class_wise_logit_var = class_wise_logit_var_raw / jnp.square(
                        temperature
                    )
                else:
                    class_wise_logit_var = None

            else:
                logits = mean_output
                class_wise_logit_var = var_output

            if sample_axis is None:
                if binary:
                    confidence = nn.sigmoid(logits)
                    class_wise_confidence = jnp.concatenate(
                        (1 - confidence, confidence), -1
                    )
                    class_wise_logits = jnp.concatenate(
                        jnp.zeros_like(logits), logits, -1
                    )
                else:
                    confidence = nn.softmax(logits, -1)
                    class_wise_logits = logits
                    class_wise_confidence = confidence

                entropy = class_entropy(class_wise_confidence)

            else:
                if temperature_scaling:
                    (
                        raw_sample_logits,
                        raw_sample_temperature,
                    ) = split_temperature_channel(sample_output)
                    sample_temperature = jax.nn.softplus(raw_sample_temperature)
                    sample_logits = raw_sample_logits / sample_temperature

                    temperature_mean, temperature_var = get_weighted_mean_variance(
                        sample_temperature,
                        sample_axis=sample_axis,
                        sample_weights=sample_weights,
                        bessel_correction_ddof=bessel_correction_ddof,
                    )
                    temperature_std = jnp.sqrt(temperature_var)
                else:
                    sample_logits = sample_output

                if binary:
                    sample_confidence = nn.sigmoid(sample_logits)
                    class_wise_sample_confidence = jnp.concatenate(
                        (1 - sample_confidence, sample_confidence), -1
                    )
                else:
                    sample_confidence = nn.softmax(sample_logits, -1)
                    class_wise_sample_confidence = sample_confidence

                (
                    class_wise_confidence,
                    entropy,
                    downstream_entropy,
                    mutual_info,
                ) = get_confidence_entropy_mutual(
                    class_wise_sample_confidence,
                    sample_weights,
                    sample_axis=sample_axis,
                )

                if binary:
                    assert class_wise_confidence.shape[-1] == 2
                    confidence = class_wise_confidence[..., -1:]
                else:
                    confidence = class_wise_confidence

                _, class_wise_confidence_var = get_weighted_mean_variance(
                    sample_confidence,
                    sample_axis=sample_axis,
                    sample_weights=sample_weights,
                    bessel_correction_ddof=bessel_correction_ddof,
                )

                confidence_var = jnp.mean(class_wise_confidence_var, -1, keepdims=True)
                confidence_std = jnp.sqrt(confidence_var)

            if class_wise_logit_var is not None:
                logit_var = jnp.mean(class_wise_logit_var, -1, keepdims=True)
                logit_std = jnp.sqrt(logit_var)

            if binary:
                pr = (confidence > binary_threshold).astype(confidence.dtype)
            else:
                pr = jnp.argmax(confidence, -1, keepdims=True)

        else:
            if stage_aleatoric:
                (
                    mean,
                    log_b_or_log_sigma_squared,
                ) = split_attenuation_channels(mean_output)
                if var_output is not None:
                    propagated_var, _ = split_attenuation_channels(var_output)
                else:
                    propagated_var = None

                b_or_sigma_squared = jnp.exp(log_b_or_log_sigma_squared)

                if l2:
                    stage_var = b_or_sigma_squared
                else:
                    stage_var = 2 * jnp.square(b_or_sigma_squared)
                stage_std = jnp.sqrt(stage_var)

            else:
                stage_var = None
                mean = mean_output
                propagated_var = var_output

            if propagated_var is not None:
                propagated_std = jnp.sqrt(propagated_var)
            else:
                propagated_var = None

            all_vars = [var for var in (stage_var, propagated_var) if var is not None]
            if all_vars:
                joint_var = jnp.sum(jnp.stack(all_vars), 0)
                joint_std = jnp.sqrt(joint_var)
            else:
                joint_var = None
                joint_std = None

            pr = mean

        sample_size_return_dict = {}
        full_size_return_dict = {}
        full_size_class_wise_return_dict = {}
        full_size_return_dict["gt"] = gt
        if classifier:
            if binary:
                full_size_return_dict["confidence"] = confidence

            full_size_class_wise_return_dict["confidence"] = class_wise_confidence

            full_size_return_dict["entropy"] = entropy
            full_size_return_dict["pr"] = pr
            if sample_axis is None:
                full_size_class_wise_return_dict["logits"] = class_wise_logits
                if temperature_scaling:
                    full_size_return_dict["temperature"] = temperature
            else:
                if temperature_scaling:
                    sample_size_return_dict["temperature"] = sample_temperature
                    full_size_return_dict["temperature_mean"] = temperature_mean
                    full_size_return_dict["temperature_var"] = temperature_var
                    full_size_return_dict["temperature_std"] = temperature_std
                sample_size_return_dict["sample_logits"] = sample_logits
                sample_size_return_dict["sample_confidence"] = sample_confidence
                full_size_return_dict["mutual_info"] = mutual_info
                full_size_return_dict["stage_aleatoric_entropy"] = downstream_entropy
                full_size_return_dict["confidence_std"] = confidence_std
                full_size_return_dict["logit_var"] = logit_var
                full_size_return_dict["logit_std"] = logit_std

        else:
            full_size_return_dict["pr"] = pr
            if joint_var is not None:
                full_size_return_dict["joint_var"] = joint_var
                full_size_return_dict["joint_std"] = joint_std
            if stage_aleatoric:
                full_size_return_dict["stage_aleatoric_var"] = stage_var
                full_size_return_dict["stage_aleatoric_std"] = stage_std
            if propagated_var is not None:
                full_size_return_dict["propagated_var"] = propagated_var
                full_size_return_dict["propagated_std"] = propagated_std
            if sample_axis is not None:
                sample_size_return_dict["sample_value"] = sample_output

        # if sample_axis is not None:
        #     sample_description_dict = {}
        #     sample_description_dict["sample_axis"] = sample_axis
        #     sample_description_dict["sample_weights"] = sample_weights
        #     sample_description_dict["bessel_correction_ddof"] = bessel_correction_ddof

        return (
            full_size_return_dict,
            full_size_class_wise_return_dict,
            sample_size_return_dict,
        )

    return full_size_metrics


def cross_entropy_loss(
    sample_output,
    target,
    temperature=1.0,
    *,
    class_weights,
    sample_axis,
    sample_weights,
    aggregation_axis,
    binary,
    logits,
):
    assert logits or temperature == 1.0
    if binary:
        if logits:
            loss = bce_with_logits(
                sample_output,
                target,
                temperature,
                class_weights,
                sample_axis,
                sample_weights,
                aggregation_axis,
            )
        else:
            loss = bce_with_confidence(
                sample_output,
                target,
                class_weights,
                sample_axis,
                sample_weights,
                aggregation_axis,
            )
    else:
        if logits:
            loss = ce_with_logits(
                sample_output,
                target,
                temperature,
                class_weights,
                sample_axis,
                sample_weights,
                aggregation_axis,
            )
        else:
            loss = ce_with_confidence(
                sample_output,
                target,
                class_weights,
                sample_axis,
                sample_weights,
                aggregation_axis,
            )

    return loss


def get_loss_fn(
    *,
    classifier,
    logits,
    binary,
    dice_loss,
    class_weights,
    stage_aleatoric,
    temperature_scaling,
    detach_loss,
    l2,
    sampling_loss,
    aggregation_axis=None,
    custom_regression_loss_fn=None,
):
    tuple_class_weights = None if class_weights is None else jnp.array(class_weights)
    jax_class_weights = None if class_weights is None else jnp.array(class_weights)

    def loss_fn(
        target,
        mean_output,
        var_output=None,
        sample_output=None,
        sample_weights=None,
        bessel_correction_ddof=0,
        sample_axis=None,
    ):
        if not sampling_loss:
            sample_output = mean_output
            var_output = None
            sample_weights = None
            bessel_correction_ddof = 0
            sample_axis = None

        if classifier:
            if FLAGS.temperature_scaling:
                prediction, raw_temperature = split_temperature_channel(sample_output)
                temperature = jax.nn.softplus(raw_temperature)
            else:
                prediction = sample_output
                temperature = 1.0

            loss = 0

            if dice_loss:
                dice_loss_value = f_score_loss(
                    prediction,
                    target,
                    beta=1,
                    class_weights=tuple_class_weights,
                    sample_axis=sample_axis,
                    sample_weights=sample_weights,
                    binary=binary,
                    logits=logits,
                    aggregation_axis=aggregation_axis,
                )

                loss += dice_loss_value

                if detach_loss:
                    prediction = jax.lax.stop_gradient(prediction)
                else:
                    assert not detach_loss

            if temperature_scaling or not dice_loss:
                cross_entropy_loss_value = cross_entropy_loss(
                    prediction,
                    target,
                    temperature=temperature,
                    class_weights=jax_class_weights,
                    sample_axis=sample_axis,
                    sample_weights=sample_weights,
                    aggregation_axis=aggregation_axis,
                    binary=binary,
                    logits=logits,
                )

                loss += cross_entropy_loss_value
        else:
            if stage_aleatoric:
                prediction, log_b_or_log_sigma_squared = split_attenuation_channels(
                    mean_output
                )
                if var_output is None:
                    prediction_variance = None
                else:
                    prediction_variance, _ = split_attenuation_channels(var_output)
            else:
                prediction = sample_output
                log_b_or_log_sigma_squared = 1 if var_output is None else 0
                prediction_variance = var_output

            loss = 0

            if custom_regression_loss_fn:
                custom_loss_value = custom_regression_loss_fn(
                    prediction, target, aggregation_axis=aggregation_axis
                )
                loss += custom_loss_value
                if detach_loss:
                    prediction = jax.lax.stop_gradient(prediction)
                    prediction_variance = jax.lax.stop_gradient(prediction_variance)
            else:
                assert not detach_loss

            if l2:
                loss += l2_fn(
                    prediction,
                    target,
                    prediction_variance,
                    log_b_or_log_sigma_squared,
                    aggregation_axis=aggregation_axis,
                )
            else:
                loss += l1_fn(
                    prediction,
                    target,
                    prediction_variance,
                    log_b_or_log_sigma_squared,
                    aggregation_axis=aggregation_axis,
                )

        return loss

    return loss_fn


def get_calibration_metrics_fn(
    classifier,
    logits,
    binary,
    stage_aleatoric,
    l2,
):
    def calibration_metrics_fn(
        target,
        mean_output,
        var_output=None,
        sample_output=None,
        sample_weights=None,
        bessel_correction_ddof=0,
        sample_axis=None,
    ):
        metrics = {}
        if classifier:
            pr, gt = get_one_hot(sample_output, target, binary=binary, logits=logits)

            ce = ce_with_confidence(
                pr,
                target if sample_axis is None else jnp.expand_dims(target, sample_axis),
                aggregation_axis=(),
            )

            if sample_axis is not None:
                sample_average_ce = jnp.average(
                    ce, np.array(sample_axis) % pr.ndim, sample_weights
                )

                metrics["sample_average_cross_entropy"] = sample_average_ce

                ce = ce_with_confidence(
                    pr,
                    target,
                    sample_axis=sample_axis,
                    sample_weights=sample_weights,
                    aggregation_axis=(),
                )
                pr = jnp.average(pr, sample_axis, sample_weights)

            brier = jnp.sum(
                l2_fn(pr, gt, const=0, factor=1, aggregation_axis=()), -1, keepdims=True
            )

            metrics.update(dict(cross_entropy=ce, brier=brier))
        else:
            if stage_aleatoric:
                prediction, log_b_or_log_sigma_squared = split_attenuation_channels(
                    mean_output
                )
                if var_output is None:
                    prediction_variance = None
                else:
                    prediction_variance, _ = split_attenuation_channels(var_output)
            else:
                prediction = sample_output
                log_b_or_log_sigma_squared = 0
                prediction_variance = var_output

            l2_distance = jnp.sqrt(l2_fn(prediction, target, aggregation_axis=()))
            l1_distance = l1_fn(prediction, target, aggregation_axis=())

            metrics = dict(
                l1=l1_distance,
                l2=l2_distance,
            )

            if l2:
                l2_distribution = l2_fn(
                    prediction,
                    target,
                    prediction_variance,
                    log_b_or_log_sigma_squared,
                    aggregation_axis=(),
                )
                metrics["l2_distribution"] = l2_distribution
            else:
                l1_distribution = l1_fn(
                    prediction,
                    target,
                    prediction_variance,  # TODO fix
                    log_b_or_log_sigma_squared,
                    aggregation_axis=(),
                )
                metrics["l1_distribution"] = l1_distribution

        return jax.tree_map(jax.vmap(jnp.mean), metrics)

    return calibration_metrics_fn


def get_input_dict_fn():
    def input_dict_fn(inputs):
        input_names = [""] if len(inputs) == 1 else range(len(inputs))
        full_size_return_dict = {}
        for input_name, input_tensor in zip(input_names, inputs):
            if isinstance(input_tensor, tuple):
                assert len(input_tensor) == 2
                input_mean, input_var = input_tensor

                full_size_return_dict[f"input{input_name}_mean"] = input_mean
                full_size_return_dict[f"input{input_name}_var"] = input_var
                full_size_return_dict[f"input{input_name}_std"] = jnp.sqrt(input_var)
            else:
                full_size_return_dict[f"input{input_name}_mean"] = input_tensor

            return full_size_return_dict

    return jax.jit(input_dict_fn)


def get_accumulated_image_metrics_fn():
    def accumulated_image_metrics(image_dict):
        sqrt_of_mean_vars = {
            key.replace("var", "sqrt_of_mean_var"): jnp.sqrt(
                jax.vmap(jnp.mean)(var_images)
            )
            for key, var_images in image_dict.items()
            if key.endswith("var")
        }

        sqrt_of_max_vars = {
            key.replace("var", "sqrt_of_max_var"): jnp.sqrt(
                jax.vmap(jnp.max)(var_images)
            )
            for key, var_images in image_dict.items()
            if key.endswith("var")
        }

        mean_std = {
            key.replace("std", "mean_std"): jnp.sqrt(jax.vmap(jnp.mean)(var_images))
            for key, var_images in image_dict.items()
            if key.endswith("std")
        }

        max_std = {
            key.replace("std", "max_std"): jnp.sqrt(jax.vmap(jnp.max)(var_images))
            for key, var_images in image_dict.items()
            if key.endswith("std")
        }

        return {**sqrt_of_mean_vars, **sqrt_of_max_vars, **mean_std, **max_std}

    return jax.jit(accumulated_image_metrics)


def get_class_wise_metrics_fn(
    binary,
    sample_wise=True,
    binary_threshold=0.5,
    hard=True,
    soft=False,
    square_soft=False,
):
    def class_wise_metrics(metrics, class_wise_metrics):
        target = metrics["gt"]
        confidence = class_wise_metrics["confidence"]
        metrics = {}
        if soft:
            soft_pr, soft_gt = get_one_hot(
                confidence,
                target,
                hard=False,
                binary=binary,
                logits=False,
                threshold=binary_threshold,
            )

            (
                metrics["soft_tp"],
                metrics["soft_gt"],
                metrics["soft_pr"],
                metrics["soft_tn"],
                metrics["soft_total"],
            ) = get_tp_gt_pr_tn_tot_from_one_hot(
                soft_pr, soft_gt, square=square_soft, efficient_tn=False
            )

        if hard:
            pr, gt = get_one_hot(
                confidence,
                target,
                hard=True,
                binary=binary,
                logits=False,
                threshold=binary_threshold,
            )

            (
                metrics["tp"],
                metrics["gt"],
                metrics["pr"],
                metrics["tn"],
                metrics["total"],
            ) = get_tp_gt_pr_tn_tot_from_one_hot(pr, gt)

        return metrics

    if sample_wise:
        return jax.jit(jax.vmap(class_wise_metrics))
    else:
        return jax.jit(class_wise_metrics)


def get_class_wise_accuracy_aggregator_fn(
    aggregation_axes=(),
):
    def calc_accuracy(full_size_metrics):
        tp = jnp.sum(full_size_metrics["tp"], axis=aggregation_axes)
        tn = jnp.sum(full_size_metrics["tn"], axis=aggregation_axes)
        total = jnp.sum(full_size_metrics["total"], axis=aggregation_axes)
        return {"accuracy": (tp + tn) / total}

    return calc_accuracy


def get_class_wise_metrics_aggregator_fn(
    betas=(1,),
    aggregation_axes=(),
    hard=True,
    soft=False,
    return_validity=True,
):
    vmap_f_score_and_validity_from_counts = jax.vmap(
        f_score_and_validity_from_counts, (None, None, None, 0), (0, 0)
    )

    beta_array = jnp.array(betas)

    def calc_f_scores(full_size_metrics, extra_name):
        tp = jnp.sum(full_size_metrics[f"{extra_name}tp"], axis=aggregation_axes)
        gt = jnp.sum(full_size_metrics[f"{extra_name}gt"], axis=aggregation_axes)
        pr = jnp.sum(full_size_metrics[f"{extra_name}pr"], axis=aggregation_axes)

        scores, validities = vmap_f_score_and_validity_from_counts(
            tp, pr, gt, beta_array
        )
        scores_dict = {
            f"{extra_name}f{beta}_score": score for beta, score in zip(betas, scores)
        }
        validity_dict = {
            f"{extra_name}f{beta}_score": validity
            for beta, validity in zip(betas, validities)
        }
        return {"score": scores_dict, "validity": validity_dict}

    def class_wise_metric_aggregator(full_size_metrics):
        result_metrics = defaultdict(dict)

        if soft:
            for key, value in calc_f_scores(full_size_metrics, "soft_").items():
                result_metrics[key].update(value)

        if hard:
            for key, value in calc_f_scores(full_size_metrics, "").items():
                result_metrics[key].update(value)

        if return_validity:
            return result_metrics
        else:
            # is_valid = jax.tree_util.tree_all(result_metrics["validity"])
            return result_metrics["score"]

    return jax.jit(class_wise_metric_aggregator)


def get_class_wise_metrics_average_calculator_fn(
    *, foreground_classes=None, binary=False, betas=(1,), hard=True, soft=False
):
    def get_foreground_indices(class_shape):
        if foreground_classes is None:
            return (
                np.arange(class_shape // 2, class_shape)
                if binary
                else np.arange(1, class_shape)
            )
        else:
            return foreground_classes

    extra_names = []
    if hard:
        extra_names.append("")
    if soft:
        extra_names.append("soft_")

    names = [
        f"{extra_name}f{beta}_score" for extra_name in extra_names for beta in betas
    ]

    def calc_averages(metrics, name):
        class_f_scores = metrics[name]

        result_dict = {name: jnp.mean(class_f_scores, axis=-1)}

        if foreground_classes is not False:
            selected_foreground_classes = get_foreground_indices(
                class_f_scores.shape[-1]
            )

            foreground_mean = jnp.mean(
                class_f_scores[..., selected_foreground_classes], axis=-1
            )

            result_dict[f"{name}_foreground"] = foreground_mean

        return result_dict

    def class_wise_metrics_average_calculator_fn(metrics):
        return {
            key: value
            for name in names
            for key, value in calc_averages(metrics, name).items()
        }

    return jax.jit(class_wise_metrics_average_calculator_fn)


def get_scalar_metrics_fn(classifier, binary, class_weights, binary_threshold):
    class_weights = None if class_weights is None else jnp.array(class_weights)

    def scalar_metrics(full_size_metrics, full_size_class_wise):
        target = full_size_metrics["gt"]

        if classifier:
            confidence = full_size_class_wise["confidence"]

            if binary:
                metrics = compute_bce_classification_metrics_with_confidence(
                    confidence,
                    target,
                    class_weights=class_weights,
                    binary_threshold=binary_threshold,
                )
            else:
                metrics = compute_ce_classification_metrics_with_confidence(
                    confidence, target, class_weights=class_weights
                )
        else:
            pr = full_size_metrics["pr"]
            metrics = compute_regression_metrics(pr, target)

        return metrics

    return jax.jit(scalar_metrics)
