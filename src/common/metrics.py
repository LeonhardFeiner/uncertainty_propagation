from typing import Dict, List, Tuple, Union
from functools import partial
import jax.numpy as jnp
import jax
from einops import reduce
from common.loss import (
    ce_with_confidence,
    l1,
    l2,
    raw_l1,
    raw_l2,
    ce_with_logits,
    bce_with_confidence,
    bce_with_logits,
    fscores_from_one_hot,
    clipped_log,
)


def inverse_sigmoid(y):
    return jnp.log(y / (1 - y))


def compute_ce_classification_metrics_with_logits(logits, target, class_weights=None):
    weighted_ce_with_logits = partial(ce_with_logits, class_weights=class_weights)
    ce_loss = jax.vmap(weighted_ce_with_logits)(logits, target)
    accuracy = (jnp.argmax(logits, -1, keepdims=True) == target).astype(jnp.float32)
    return {"ce": ce_loss, "accuracy": accuracy}


def compute_ce_classification_metrics_with_confidence(
    confidence,
    target,
    class_weights=None,
):
    weighted_ce_with_confidence = partial(
        ce_with_confidence, class_weights=class_weights
    )

    ce_loss = jax.vmap(weighted_ce_with_confidence)(confidence, target)
    accuracy_raw = (jnp.argmax(confidence, -1, keepdims=True) == target).astype(
        jnp.float32
    )
    accuracy = jax.vmap(jnp.mean)((accuracy_raw))
    return {"ce": ce_loss, "accuracy": accuracy}


def compute_bce_classification_metrics_with_logits(
    logits, target, class_weights=None, binary_threshold=0.5
):
    weighted_ce_with_logits = partial(bce_with_logits, class_weights=class_weights)
    ce_loss = jax.vmap(weighted_ce_with_logits)(logits, target)

    logits_threshold = inverse_sigmoid(binary_threshold)

    accuracy_raw = ((logits > logits_threshold).astype(target.dtype) == target).astype(
        jnp.float32
    )
    accuracy = jax.vmap(jnp.mean)((accuracy_raw))
    return {"ce": ce_loss, "accuracy": accuracy}


def compute_bce_classification_metrics_with_confidence(
    confidence, target, class_weights=None, binary_threshold=0.5
):
    weighted_ce_with_confidence = partial(
        bce_with_confidence, class_weights=class_weights
    )

    ce_loss = jax.vmap(weighted_ce_with_confidence)(confidence, target)
    accuracy = ((confidence > binary_threshold).astype(target.dtype) == target).astype(
        jnp.float32
    )
    return {"ce": ce_loss, "accuracy": accuracy}


def compute_regression_metrics(prediction, target):
    l1_loss = jax.vmap(raw_l1)(prediction, target)
    l2_loss = jax.vmap(raw_l2)(prediction, target)
    return {"l1": l1_loss, "l2": l2_loss}


def compute_attenuated_l2_regression_metrics(
    output_mean, gt, output_var=None, log_sigma_squared=0
):
    l2_loss = jax.vmap(l2)(output_mean, gt, output_var, log_sigma_squared)
    return {"attenuated_l2": l2_loss}


def compute_attenuated_l1_regression_metrics(
    output_median, gt, output_scale=None, log_b=0
):
    l1_loss = jax.vmap(l1)(output_median, gt, output_scale, log_b)
    return {"attenuated_l1": l1_loss}


def class_entropy(confidence):
    return -jnp.sum(confidence * clipped_log(confidence), axis=-1, keepdims=True)


def get_confidence_entropy_mutual(
    sample_confidence,
    sample_weight=None,
    sample_axis=None,
):
    mean_confidence = jnp.average(sample_confidence, sample_axis, sample_weight)
    entropy = class_entropy(mean_confidence)
    sample_entropy = class_entropy(sample_confidence)
    not_sampled_entropy = jnp.average(sample_entropy, sample_axis, sample_weight)

    mutual_info = entropy - not_sampled_entropy
    return mean_confidence, entropy, not_sampled_entropy, mutual_info


def get_f_scores(pr, gt, class_labels, betas=(1, 2), fg_indices=slice(1, None)):
    num_classes = len(class_labels)
    pr_one_hot = jax.nn.one_hot(pr, num_classes)
    gt_one_hot = jax.nn.one_hot(gt, num_classes)
    vectorized_f_scores_from_one_hot = jax.vmap(fscores_from_one_hot, (None, None, 0))
    fscores = vectorized_f_scores_from_one_hot(gt_one_hot, pr_one_hot, jnp.array(betas))

    mean_fscores = {
        f"f{beta}_mean": jnp.mean(f_score_array)
        for beta, f_score_array in zip(betas, fscores)
    }

    mean_fg_fscores = {
        f"f{beta}_fg_mean": jnp.mean(f_score_array[..., fg_indices])
        for beta, f_score_array in zip(betas, fscores)
    }

    class_wise_fscores = {
        f"f{beta}_{class_label}": value
        for beta, f_score_array in zip(betas, fscores)
        for class_label, value in zip(class_labels, f_score_array)
    }
    return mean_fscores, mean_fg_fscores, class_wise_fscores


def get_class_portions(pr, gt, class_labels):
    gt_pixel_count = jnp.bincount(gt.flatten(), length=len(class_labels))
    pr_pixel_count = jnp.bincount(pr.flatten(), length=len(class_labels))
    gt_pixel_portion = gt_pixel_count / jnp.sum(gt_pixel_count)
    pr_pixel_portion = pr_pixel_count / jnp.sum(pr_pixel_count)

    class_wise_gt_pixel_portion = {
        f"gt_pixel_portion_{class_label}": value
        for class_label, value in zip(class_labels, gt_pixel_portion)
    }

    class_wise_pr_pixel_portion = {
        f"pr_pixel_portion_{class_label}": value
        for class_label, value in zip(class_labels, pr_pixel_portion)
    }
    return {**class_wise_gt_pixel_portion, **class_wise_pr_pixel_portion}


def get_slice_portions(pr, gt, class_labels):
    eye = jnp.eye(len(class_labels))
    gt_slice_portion = reduce(eye[gt], "b ... c -> b c", jnp.any).mean(0)
    pr_slice_portion = reduce(eye[pr], "b ... c -> b c", jnp.any).mean(0)

    class_wise_gt_slice_portion = {
        f"gt_slice_portion_{class_label}": value
        for class_label, value in zip(class_labels, gt_slice_portion)
    }

    class_wise_pr_slice_portion = {
        f"pr_slice_portion_{class_label}": value
        for class_label, value in zip(class_labels, pr_slice_portion)
    }

    return {**class_wise_gt_slice_portion, **class_wise_pr_slice_portion}
