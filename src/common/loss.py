from typing import Sequence, Tuple
from functools import partial
import jax
import numpy as np
import jax.numpy as jnp
from flax import linen as nn
from einops import rearrange


def accuracy_from_counts(tp, tn, total):
    return (tp + tn) / total


def f_score_fraction_from_counts(tp, pr, gt, beta):
    beta_squared = beta * beta
    nominator = (1 + beta_squared) * tp
    denominator = beta_squared * pr + gt
    return nominator, denominator


def f_score_from_counts(tp, pr, gt, beta, nominator_epsilon=0, denominator_epsilon=0):
    nominator, denominator = f_score_fraction_from_counts(tp, pr, gt, beta)
    return (nominator + nominator_epsilon) / (denominator + denominator_epsilon)


def f_score_and_validity_from_counts(tp, pr, gt, beta, fallback=1e-8):
    nominator, denominator = f_score_fraction_from_counts(tp, pr, gt, beta)
    validity = denominator != 0
    f_score = jnp.where(
        validity, nominator / jnp.where(validity, denominator, 1), fallback
    )
    return f_score, validity


@partial(jax.jit, static_argnames=("square", "efficient_tn"))
def get_tp_gt_pr_tn_tot_from_one_hot(
    gt_one_hot, pr_one_hot, *, square=False, efficient_tn=True, aggregation_axis=None
):
    gt_one_hot, pr_one_hot = jnp.broadcast_arrays(gt_one_hot, pr_one_hot)

    assert aggregation_axis is None
    keep_dim_index = pr_one_hot.ndim - 1
    if aggregation_axis is not None:
        aggregation_axis = np.array(aggregation_axis) % pr_one_hot.ndim
        assert keep_dim_index not in aggregation_axis
    else:
        aggregation_axis = np.arange(keep_dim_index)

    tp_sum = jnp.sum(gt_one_hot * pr_one_hot, aggregation_axis)
    if square:
        gt_one_hot = jnp.square(gt_one_hot)
        pr_one_hot = jnp.square(pr_one_hot)

    pr_sum = jnp.sum(pr_one_hot, aggregation_axis)
    gt_sum = jnp.sum(gt_one_hot, aggregation_axis)
    total = jnp.full_like(gt_sum, np.prod(np.array(gt_one_hot.shape)[aggregation_axis]))
    if efficient_tn:
        tn_sum = total - gt_sum - pr_sum + tp_sum
    else:
        tp_sum = jax.vmap(jnp.sum, -1)((1 - gt_one_hot) * (1 - pr_one_hot))

    return tp_sum, gt_sum, pr_sum, tn_sum, total


def fscores_from_one_hot(
    gt_one_hot,
    pr_one_hot,
    beta=1,
    *,
    aggregation_axis=None,
    square=False,
    nominator_epsilon=0,
    denominator_epsilon=0,
):
    tp_sum, gt_sum, pr_sum, tn_sum, total = get_tp_gt_pr_tn_tot_from_one_hot(
        gt_one_hot,
        pr_one_hot,
        square=square,
        aggregation_axis=aggregation_axis,
    )

    return f_score_from_counts(
        tp_sum, pr_sum, gt_sum, beta, nominator_epsilon, denominator_epsilon
    )


def get_confidence(confidence_or_logits, *, binary=False, logits=True):
    if binary:
        confidence = (
            jax.nn.sigmoid(confidence_or_logits) if logits else confidence_or_logits
        )
    else:
        confidence = (
            jax.nn.softmax(confidence_or_logits, axis=-1)
            if logits
            else confidence_or_logits
        )

    return confidence


@partial(jax.jit, static_argnames=("hard", "binary", "logits", "threshold"))
def get_one_hot(
    confidence_or_logits,
    labels,
    *,
    hard=False,
    binary=False,
    logits=False,
    threshold=0.5,
):
    dtype = labels.dtype if hard else confidence_or_logits.dtype
    confidence = get_confidence(confidence_or_logits, binary=binary, logits=logits)

    if binary:
        confidence_foreground = (
            (confidence > threshold).astype(dtype) if hard else confidence
        )

        pr_raw = jnp.stack((1 - confidence_foreground, confidence_foreground), -2)
        gt_raw = jax.vmap(lambda x: jax.nn.one_hot(x, 2, axis=-1, dtype=dtype), -1, -1)(
            labels
        )
        pr = rearrange(pr_raw, "... c t -> ... (c t)")
        gt = rearrange(gt_raw, "... c t -> ... (c t)")

    else:
        n_classes = confidence.shape[-1]
        if hard:
            pr = jax.nn.one_hot(
                jnp.argmax(confidence, -1),
                n_classes,
                axis=-1,
                dtype=dtype,
            )
        else:
            pr = confidence

        gt = jax.nn.one_hot(
            jnp.squeeze(labels, -1),
            n_classes,
            axis=-1,
            dtype=dtype,
        )

    return pr, gt


def f_score_loss(
    confidence_or_logits,
    labels,
    *,
    beta=1,
    class_weights=None,
    sample_axis=None,
    sample_weights=None,
    aggregation_axis=None,
    binary=False,
    logits=False,
    epsilon=1e-5,
    square=False,
):
    beta = jnp.squeeze(beta)
    assert beta.size == 1

    confidence = get_confidence(confidence_or_logits, binary=binary, logits=logits)

    if sample_axis is not None:
        confidence = jnp.average(confidence, sample_axis, sample_weights)

    pr, gt = get_one_hot(
        confidence,
        labels,
        binary=binary,
        logits=False,  # converted to confidence beforehand
    )

    if class_weights is not None:
        interesting_classes = np.array(class_weights) != 0

        gt = gt[..., interesting_classes]
        pr = pr[..., interesting_classes]
        class_weights = class_weights[interesting_classes]

    f_score_values = fscores_from_one_hot(
        gt,
        pr,
        beta=beta,
        square=square,
        nominator_epsilon=0,
        denominator_epsilon=epsilon,
        aggregation_axis=aggregation_axis,
    )

    return 1 - jnp.average(f_score_values, -1, class_weights)


def negative_weighted_mean(nll, labels, class_weights, axis=None):
    if class_weights is None:
        return -jnp.mean(nll, axis=axis)
    else:
        selected_weights = class_weights[labels]
        return -jnp.sum(nll * selected_weights, axis=axis) / jnp.clip(
            jnp.sum(jnp.broadcast_to(selected_weights, nll.shape), axis=axis),
            a_min=1e-8,
        )


def bce_with_logits(
    logits,
    labels,
    temperature=1.0,
    class_weights=None,
    sample_axis=None,
    sample_weights=None,
    aggregation_axis=None,
):
    if sample_axis is not None:
        confidences = nn.sigmoid(logits / temperature)
        return bce_with_confidence(
            confidences,
            labels,
            class_weights,
            sample_axis,
            sample_weights,
            aggregation_axis,
        )
    else:
        log_confidences = nn.log_sigmoid(logits / temperature)
        nll = labels * log_confidences + (1.0 - labels) * jnp.log(
            -jnp.expm1(log_confidences)
        )
        return negative_weighted_mean(nll, labels, class_weights, axis=aggregation_axis)


def ce_with_logits(
    logits,
    labels,
    temperature=1.0,
    class_weights=None,
    sample_axis=None,
    sample_weights=None,
    aggregation_axis=None,
):
    if sample_axis is not None:
        confidences = nn.softmax(logits / temperature, axis=-1)
        return ce_with_confidence(
            confidences,
            labels,
            class_weights,
            sample_axis,
            sample_weights,
            aggregation_axis,
        )
    else:
        log_confidences = nn.log_softmax(logits / temperature, axis=-1)
        nll = jnp.take_along_axis(log_confidences, labels, axis=-1)
        return negative_weighted_mean(nll, labels, class_weights, axis=aggregation_axis)


def calc_cross_entropy_input(confidences, sample_axis, sample_weights):
    if sample_axis is not None:
        confidences = jnp.average(confidences, sample_axis, sample_weights)

    return confidences


def clipped_log(confidence):
    return jnp.log(jnp.clip(confidence, a_min=jnp.finfo(confidence).eps))


def bce_with_confidence(
    confidences,
    labels,
    class_weights=None,
    sample_axis=None,
    sample_weights=None,
    aggregation_axis=None,
):
    confidences = calc_cross_entropy_input(confidences, sample_axis, sample_weights)
    nll = labels * clipped_log(confidences) + (1.0 - labels) * clipped_log(
        1.0 - confidences
    )
    return negative_weighted_mean(nll, labels, class_weights, axis=aggregation_axis)


def ce_with_confidence(
    confidences,
    labels,
    class_weights=None,
    sample_axis=None,
    sample_weights=None,
    aggregation_axis=None,
):
    confidences = calc_cross_entropy_input(confidences, sample_axis, sample_weights)
    nll = clipped_log(jnp.take_along_axis(confidences, labels, axis=-1))
    return negative_weighted_mean(nll, labels, class_weights, axis=aggregation_axis)


def get_weighted_mean_variance(
    output, sample_axis, sample_weights, bessel_correction_ddof=0
):
    if isinstance(sample_axis, Sequence):
        sample_axis = tuple(sample_axis)
    sample_mean = jnp.average(output, sample_axis, sample_weights)

    if sample_weights is not None:
        sample_variance = jnp.average(
            jnp.square(output - sample_mean), sample_axis, sample_weights
        )

        if bessel_correction_ddof:
            n = np.prod(output.shape[sample_axis])
            correction_factor = n / (n - bessel_correction_ddof)
            sample_variance *= correction_factor
    else:
        sample_variance = jnp.var(output, sample_axis, ddof=bessel_correction_ddof)

    return sample_mean, sample_variance


# pylint: disable=invalid-name
def l2(
    output_mean,
    gt,
    output_var=None,
    sigma_squared_or_log_sigma_squared=0,
    *,
    log_space=True,
    aggregation_axis=None,
    const=0.5 * np.log(2 * np.pi),
    factor=0.5,
):
    if output_var is not None:
        sigma_squared = (
            jnp.exp(sigma_squared_or_log_sigma_squared)
            if log_space
            else sigma_squared_or_log_sigma_squared
        )
        joint_var = output_var + sigma_squared
        inverse_joint_var = 1 / joint_var
        log_joint_var = jnp.log(joint_var)
    elif log_space:
        inverse_joint_var = jnp.exp(-sigma_squared_or_log_sigma_squared)
        log_joint_var = sigma_squared_or_log_sigma_squared
    else:
        inverse_joint_var = 1 / sigma_squared_or_log_sigma_squared
        log_joint_var = jnp.log(sigma_squared_or_log_sigma_squared)

    return factor * jnp.mean(
        jnp.square(output_mean - gt) * inverse_joint_var + log_joint_var + const,
        aggregation_axis,
    )


# pylint: disable=invalid-name
def l1(
    output_median,
    gt,
    output_scale=None,
    log_b=0,
    *,
    aggregation_axis=None,
):
    if output_scale is not None:
        joint_scale = output_scale + jnp.exp(log_b)
        inverse_joint_scale = 1 / joint_scale
        log_joint_scale = jnp.log(joint_scale)
    else:
        inverse_joint_scale = jnp.exp(-log_b)
        log_joint_scale = log_b

    return jnp.mean(
        jnp.abs(output_median - gt) * inverse_joint_scale + log_joint_scale,
        aggregation_axis,
    )


def raw_l2(output: jax.Array, gt):
    return jnp.mean(jnp.square(output - gt))


def raw_l1(output: jax.Array, gt):
    return jnp.mean(jnp.abs(output - gt))
