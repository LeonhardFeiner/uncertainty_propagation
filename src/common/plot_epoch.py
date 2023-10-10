from functools import partial
from operator import itemgetter
from typing import Dict, List, Tuple, Union
from absl import flags
import itertools
import numpy as np
import jax.numpy as jnp
import jax
from PIL import Image
from einops import rearrange, reduce
from tensorboardX import SummaryWriter
from common.plot_utils import (
    get_multi_roc,
    plot_bar,
    plot_confusion_matrix,
    plot_2d_scatter,
    plot_lines,
    plot_violin,
)
from common.image_plot_utils import (
    combine_image_collection,
    get_image,
    get_barplots_img,
    get_labelside_img,
    get_text_img,
)


tqdm_ncols = 100
FLAGS = flags.FLAGS


def plot_imagewise(
    writer,
    image_path,
    target_name,
    class_labels,
    augmentation_index,
    image_dict,
    scalar_dict,
    class_wise_dict,
    image_res_factor,
    plot_res_factor,
    *,
    add_barplot_numbers=True,
    writer_index=0,
    image_output,
    postfix="",
    normalize=True,
    io_domains_equal=True,
):
    """
    plot example images to tensorboard
    """

    raw_image_plots = {}

    mean_input_image = image_dict["input_mean"]
    dpi = mean_input_image.shape[-2] * image_res_factor // plot_res_factor

    min_output_image = min_input_image = jnp.min(mean_input_image)
    max_input_image = jnp.max(mean_input_image)
    range_output_image = range_input_image = max_input_image - min_input_image
    if normalize:
        normalized_mean_input_image = (
            mean_input_image - min_input_image
        ) / range_input_image
    else:
        normalized_mean_input_image = mean_input_image
    raw_image_plots["mean_input_image"] = get_image(
        normalized_mean_input_image, bounded_greyscale=True, scale=image_res_factor
    )

    if "input_var" in image_dict:
        std_input_image = image_dict["input_std"]
        if normalize:
            normalized_std_input_image = std_input_image / range_input_image
        else:
            normalized_std_input_image = std_input_image
        raw_image_plots["std_input_image"] = get_image(
            normalized_std_input_image, bounded_greyscale=True, scale=image_res_factor
        )

    if image_output:
        gt = image_dict["gt"]
        pr = image_dict["pr"]

        raw_image_plots["gt"] = get_image(
            gt, segmentation_num_classes=len(class_labels), scale=image_res_factor
        )
        raw_image_plots["pr"] = get_image(
            pr, segmentation_num_classes=len(class_labels), scale=image_res_factor
        )

        if "entropy" in image_dict:
            uncertainty_normalizer = jnp.log(len(class_labels))
            raw_image_plots["entropy"] = get_image(
                image_dict["entropy"] / uncertainty_normalizer,
                bounded_greyscale=True,
                scale=image_res_factor,
            )
            if "mutual_info" in image_dict:
                raw_image_plots["mutual_info"] = get_image(
                    image_dict["mutual_info"] / uncertainty_normalizer,
                    bounded_greyscale=True,
                    scale=image_res_factor,
                )
                raw_image_plots["stage_aleatoric_entropy"] = get_image(
                    image_dict["stage_aleatoric_entropy"] / uncertainty_normalizer,
                    bounded_greyscale=True,
                    scale=image_res_factor,
                )
                raw_image_plots["confidence_std"] = get_image(
                    image_dict["confidence_std"]
                    / jnp.max(image_dict["confidence_std"]),
                    bounded_greyscale=True,
                    scale=image_res_factor,
                )
                raw_image_plots["logit_std_norm"] = get_image(
                    image_dict["logit_std"] / jnp.max(image_dict["logit_std"]),
                    bounded_greyscale=True,
                    scale=image_res_factor,
                )
                raw_image_plots["logit_std"] = get_image(
                    image_dict["logit_std"],
                    bounded_greyscale=True,
                    scale=image_res_factor,
                )
            if "temperature" in image_dict:
                raw_image_plots["temperature"] = get_image(
                    image_dict["temperature"],
                    bounded_greyscale=True,
                    scale=image_res_factor,
                )
            elif "temperature_mean" in image_dict:
                raw_image_plots["temperature_mean"] = get_image(
                    image_dict["temperature_mean"],
                    bounded_greyscale=True,
                    scale=image_res_factor,
                )
                raw_image_plots["temperature_std"] = get_image(
                    image_dict["temperature_std"],
                    bounded_greyscale=True,
                    scale=image_res_factor,
                )

        else:
            if normalize:
                min_output_image = jnp.min(gt)
                max_output_image = jnp.max(gt)
                range_output_image = max_output_image - min_output_image
                image_gt = (gt - min_output_image) / range_output_image
                image_pr = (pr - min_output_image) / range_output_image
            else:
                image_gt = gt
                image_pr = pr
            gt_diff = image_gt - image_pr

            raw_image_plots["gt"] = get_image(
                image_gt, bounded_greyscale=True, scale=image_res_factor
            )
            raw_image_plots["pr"] = get_image(
                image_pr, bounded_greyscale=True, scale=image_res_factor
            )

            if "stage_aleatoric_var" in image_dict:
                if normalize:
                    joint_std = image_dict["joint_std"] / range_output_image
                else:
                    joint_std = image_dict["joint_std"]
                raw_image_plots["joint_std"] = get_image(
                    joint_std, bounded_greyscale=True, scale=image_res_factor
                )

            raw_image_plots["gt_diff"] = get_image(
                gt_diff, difference_color=True, scale=image_res_factor
            )

            if io_domains_equal:
                input_diff = normalized_mean_input_image - image_pr
                raw_image_plots["input_diff"] = get_image(
                    input_diff, difference_color=True, scale=image_res_factor
                )

    else:
        gt = scalar_dict["gt"]
        pr = scalar_dict["pr"]

        if "entropy" in scalar_dict:
            confidence = class_wise_dict["confidence"]
            entropy = scalar_dict["entropy"]
            plot_images = [("entropy", entropy)]  # ("inp_std", inp_mean_std)

            if FLAGS.propagator != "normal":
                mutual_info = scalar_dict["mutual_info"]
                plot_images += [("mutual", mutual_info)]
                # std_in = scalar_dict["logit_std"]
                confidence_std = scalar_dict["confidence_std"]
                fig = plot_bar(
                    jnp.arange(len(gt)),
                    mutual_info,
                    first_name=target_name,
                    second_name="mutual_info",
                )
                writer.add_figure(
                    f"uncertainty_aug_mutualinfo/{target_name}{augmentation_index}{postfix}",
                    fig,
                    writer_index,
                )
                fig = plot_bar(
                    jnp.arange(len(gt)),
                    confidence_std,
                    first_name=target_name,
                    second_name="std",
                )
                writer.add_figure(
                    f"uncertainty_aug_confidencestd/{target_name}{augmentation_index}{postfix}",
                    fig,
                    writer_index,
                )

            fig = plot_bar(
                jnp.arange(len(gt)),
                entropy,
                first_name=target_name,
                second_name="entropy",
            )
            writer.add_figure(
                f"uncertainty_aug_entropy/{target_name}{augmentation_index}{postfix}",
                fig,
                writer_index,
            )

            raw_image_plots["label_img"] = get_labelside_img(
                class_labels,
                gt,
                dpi=dpi,
                plot_res_factor=plot_res_factor,
            )
            raw_image_plots["confidence"] = get_barplots_img(
                class_labels,
                confidence,
                dpi=dpi,
                plot_res_factor=plot_res_factor,
                add_numbers=add_barplot_numbers,
            )

            uncertainty_names, uncertainty_values = zip(*plot_images)

            raw_image_plots["uncertainty"] = get_barplots_img(
                uncertainty_names,
                jnp.stack(uncertainty_values, -1),
                formater=".2e",
                dpi=dpi,
                plot_res_factor=plot_res_factor,
            )
        else:
            raw_image_plots["prediction"] = get_barplots_img(
                ["gt", "pr"],
                jnp.stack((gt, pr), -1),
                dpi=dpi,
                plot_res_factor=plot_res_factor,
                add_numbers=True,
            )
            if FLAGS.propagator != "normal" or FLAGS.stage_aleatoric:
                joint_std = scalar_dict["joint_std"]
                plot_images = [("joint", joint_std)]
                fig = plot_bar(
                    jnp.arange(len(gt)),
                    joint_std,
                    first_name=target_name,
                    second_name="joint_std",
                )
                writer.add_figure(
                    f"uncertainty_aug_jointstd/{target_name}{augmentation_index}{postfix}",
                    fig,
                    writer_index,
                )

                if FLAGS.propagator != "normal":
                    propagated_std = scalar_dict["propagated_std"]
                    plot_images += [("prop.", propagated_std)]
                    fig = plot_bar(
                        jnp.arange(len(gt)),
                        propagated_std,
                        first_name=target_name,
                        second_name="propagated_std",
                    )
                    writer.add_figure(
                        f"uncertainty_aug_propagatedstd/{target_name}{augmentation_index}{postfix}",
                        fig,
                        writer_index,
                    )

                if FLAGS.stage_aleatoric:
                    stage_std = scalar_dict["stage_aleatoric_std"]
                    plot_images += [("stage", stage_std)]
                    fig = plot_bar(
                        jnp.arange(len(gt)),
                        stage_std,
                        first_name=target_name,
                        second_name="stage_aleatoric_std",
                    )
                    writer.add_figure(
                        f"uncertainty_aug_stagestd/{target_name}{augmentation_index}{postfix}",
                        fig,
                        writer_index,
                    )

                uncertainty_names, uncertainty_values = zip(*plot_images)

                raw_image_plots["uncertainty"] = get_barplots_img(
                    uncertainty_names,
                    jnp.stack(uncertainty_values, -1),
                    formater="=8.04f",
                    dpi=dpi,
                    plot_res_factor=plot_res_factor,
                )

    for order_index, (key, image_stack) in enumerate(raw_image_plots.items()):
        for image_index, image in enumerate(image_stack):
            path = (
                image_path
                / f"{writer_index:03}_{augmentation_index:02}_{image_index:02}_{order_index:02}-{key}.png"
            )
            Image.fromarray(np.array((image * 255).astype(jnp.uint8))).save(path)

    image = combine_image_collection(list(raw_image_plots.values()))

    text_image = get_text_img(
        raw_image_plots.keys(),
        dpi=dpi,
        image_res_factor=image_res_factor,
        plot_res_factor=plot_res_factor,
    )

    image = jnp.concatenate((text_image, image), -2)

    writer.add_image(
        f"image/{augmentation_index}{postfix}", image, writer_index, dataformats="HWC"
    )

    if FLAGS.input_distribution == "mvnd":
        inp_mean_std = scalar_dict["input_sqrt_of_mean_var"]
        inp_max_std = scalar_dict["input_sqrt_of_max_var"]
        fig = plot_bar(
            jnp.arange(len(gt)),
            inp_mean_std,
            first_name=target_name,
            second_name="sqrt of mean variance",
        )

        writer.add_figure(
            f"uncertainty_aug_imagemeanstd/{target_name}{augmentation_index}{postfix}",
            fig,
            writer_index,
        )

        fig = plot_bar(
            jnp.arange(len(gt)),
            inp_max_std,
            first_name=target_name,
            second_name="max std",
        )

        writer.add_figure(
            f"uncertainty_aug_imagemaxstd/{target_name}{augmentation_index}{postfix}",
            fig,
            writer_index,
        )


def get_singleclass_calibration(y_true, y_prob, n_bins, strategy="quantile"):

    if strategy == "quantile":  # Determine bin edges by distribution of data
        quantiles = jnp.linspace(0, 1, n_bins + 1)
        bins = jnp.percentile(y_prob, quantiles * 100).at[-1].add(1e-8)
    elif strategy == "uniform":
        bins = jnp.linspace(0.0, 1.0 + 1e-8, n_bins + 1)

    binids = jnp.digitize(y_prob, bins) - 1

    bin_sums = jnp.bincount(binids, weights=y_prob, length=len(bins))
    bin_true = jnp.bincount(binids, weights=y_true, length=len(bins))
    bin_total = jnp.bincount(binids, length=len(bins))

    nonzero = bin_total != 0
    prob_true = bin_true / jnp.where(nonzero, bin_total, 1)
    prob_pred = bin_sums / jnp.where(nonzero, bin_total, 1)
    return prob_true, prob_pred


def get_multiclass_calibration(gt, pr, n_classes, n_bins=10, strategy="quantile"):
    classwise_gt = (jnp.expand_dims(gt, -1) == jnp.arange(n_classes)).astype(jnp.int32)

    get_classwise_calibration = jax.vmap(
        partial(get_singleclass_calibration, n_bins=n_bins, strategy=strategy),
        (-1, -1),
        (0, 0),
    )

    return get_classwise_calibration(classwise_gt, pr)


def sample_resolution(tensor_structure, rng=jax.random.PRNGKey(0)):
    def reshape_tensor(tensor):  # TODO sampling
        if tensor.ndim >= 4:
            w, h = tensor.shape[2:4]
            start_w = w // 4
            start_h = h // 4

            center = tensor[
                :,
                :,
                start_w:-start_w,
                start_h:-start_h,
            ]
            return rearrange(center, "s a w h ... -> (s w h) a ...")
        else:
            return tensor

    return jax.tree_map(reshape_tensor, tensor_structure)


def plot_stats(writer, target_name, class_labels, all_aug_result_dict, postfix=""):  #
    """
    plot all stats to tensorboard
    """

    all_aug_result_dict = jax.tree_map(
        lambda x: jnp.squeeze(x, -1).T if x.shape[-1] == 1 else jnp.moveaxis(x, 1, 0),
        all_aug_result_dict,
    )

    gt = all_aug_result_dict["gt"]
    pr = all_aug_result_dict["pr"]
    category = all_aug_result_dict.get("category", itertools.repeat(None))

    if "entropy" in all_aug_result_dict:
        entropy = all_aug_result_dict["entropy"]

        fig = plot_violin(entropy, first_name="augmentation", second_name="entropy")
        writer.add_figure(f"uncertainty_violin/output_entropy{postfix}", fig, 0)

        if "mutual_info" in all_aug_result_dict:
            mutual_info = all_aug_result_dict["mutual_info"]
            confidence_std = all_aug_result_dict["confidence_std"]

            fig = plot_violin(
                mutual_info, first_name="augmentation", second_name="mutual_info"
            )
            writer.add_figure(f"uncertainty_violin/mutual_info{postfix}", fig, 0)

            fig = plot_violin(
                confidence_std,
                first_name="augmentation",
                second_name="confidence_std",
            )
            writer.add_figure(f"uncertainty_violin/confidence_std{postfix}", fig, 0)

    elif "joint_std" in all_aug_result_dict:
        joint_std = all_aug_result_dict["joint_std"]
        fig = plot_violin(joint_std, first_name="augmentation", second_name="joint_std")
        writer.add_figure(f"uncertainty_violin/joint_std{postfix}", fig, 0)
        if "propagated_std" in all_aug_result_dict:
            propagated_std = all_aug_result_dict["propagated_std"]
            fig = plot_violin(
                propagated_std, first_name="augmentation", second_name="propagated_std"
            )
            writer.add_figure(f"uncertainty_violin/propagated_std{postfix}", fig, 0)
        if "stage_aleatoric_std" in all_aug_result_dict:
            stage_std = all_aug_result_dict["stage_aleatoric_std"]
            fig = plot_violin(
                stage_std, first_name="augmentation", second_name="stage_aleatoric_std"
            )
            writer.add_figure(f"uncertainty_violin/stage_std{postfix}", fig, 0)

    if "input_sqrt_of_mean_var" in all_aug_result_dict:
        inp_mean_std = all_aug_result_dict["input_sqrt_of_mean_var"]
        inp_max_std = all_aug_result_dict["input_sqrt_of_max_var"]

        fig = plot_violin(
            inp_mean_std, first_name="augmentation", second_name="input_mean_std"
        )
        writer.add_figure(f"uncertainty_violin/input_meanstd{postfix}", fig, 0)
        fig = plot_violin(
            inp_max_std, first_name="augmentation", second_name="input_max_std"
        )
        writer.add_figure(f"uncertainty_violin/input_maxstd{postfix}", fig, 0)

    max_val = max(jnp.max(gt), jnp.max(pr))
    for aug_degree, selected_gt, selected_pr, selected_category in zip(
        itertools.count(), gt, pr, category
    ):

        if "entropy" in all_aug_result_dict:
            fig = plot_confusion_matrix(selected_gt, selected_pr, class_labels)
            writer.add_figure(f"cm/{target_name}{postfix}", fig, aug_degree)
        else:
            fig = plot_2d_scatter(
                selected_gt,
                selected_pr,
                color=selected_category,
                first_max=max_val,
                second_max=max_val,
                first_name="gt",
                second_name="pr",
                color_name=target_name,
            )
            writer.add_figure("accuracy_gt_pr", fig, aug_degree)

    if "confidence" in all_aug_result_dict:  # calibration

        confidence = all_aug_result_dict["confidence"]
        bin_prob_true, bin_prob_pred = jax.vmap(
            lambda gt, confidence: get_multiclass_calibration(
                gt, confidence, len(class_labels), 10
            ),
            (0, 0),
            (0, 0),
        )(gt, confidence)

        calibration_error = jnp.abs(bin_prob_true - bin_prob_pred)
        for aug_degree, selected_gt, selected_pr, selected_confidence in zip(
            itertools.count(), gt, pr, confidence
        ):
            fig = get_multi_roc(selected_gt, selected_confidence, class_labels)
            writer.add_figure("roc", fig, aug_degree)

        aug_degree_names = [str(i) for i in range(len(gt))]

        for (
            class_label,
            selected_bin_prob_true,
            selected_bin_prob_pred,
            selected_calibration_error,
        ) in zip(
            class_labels,
            jnp.moveaxis(bin_prob_true, 1, 0),
            jnp.moveaxis(bin_prob_pred, 1, 0),
            jnp.moveaxis(calibration_error, 1, 0),
        ):

            for bin_error, aug_degree in zip(
                selected_bin_prob_true - selected_bin_prob_pred,
                aug_degree_names,
            ):
                for i, error in enumerate(bin_error):
                    writer.add_scalar(
                        f"calibration/class_{class_label}_aug_degree_{aug_degree}{postfix}",
                        error,
                        i,
                    )
            fig = plot_lines(
                selected_bin_prob_pred,
                selected_bin_prob_true,
                aug_degree_names,
                "aug_degree",
                first_name="mean predicted probability",
                second_name="fraction of positives",
                add_line=True,
            )

            writer.add_figure(f"calibration/class_{class_label}{postfix}", fig)

            fig = plot_lines(
                selected_bin_prob_pred,
                selected_bin_prob_true - selected_bin_prob_pred,
                aug_degree_names,
                "aug_degree",
                first_name="mean predicted probability",
                second_name="difference between fraction of positives and predicted probability",
                add_line=True,
                horizontal=True,
            )

            writer.add_figure(
                f"calibration_difference/class_{class_label}{postfix}", fig
            )

            fig = plot_bar(
                range(len(selected_calibration_error)),
                jnp.nanmean(selected_calibration_error, 1),
                first_name="augmentation degree",
                second_name="expected calibration error",
                add_numbers=True,
            )

            writer.add_figure(f"expected_calibration/class_{class_label}{postfix}", fig)

    elif "joint_std" in all_aug_result_dict:
        pred_error = jnp.abs(gt - pr)
        bin_true, bin_pred = jax.vmap(
            lambda pred_std, pred_error: get_singleclass_calibration(
                pred_std, pred_error, 10
            ),
            (0, 0),
            (0, 0),
        )(joint_std, pred_error)

        calibration_offset = bin_true - bin_pred
        calibration_error = jnp.abs(calibration_offset)
        aug_degree_names = [str(i) for i in range(len(gt))]

        fig = plot_lines(
            bin_pred,
            bin_true,
            aug_degree_names,
            "aug_degree",
            first_name="mean predicted std",
            second_name="average absolute error",
            add_line=True,
        )
        writer.add_figure(f"calibration/joint{postfix}", fig)

        fig = plot_lines(
            bin_pred,
            calibration_offset,
            aug_degree_names,
            "aug_degree",
            first_name="mean predicted std",
            second_name="deviation of error to std",
            add_line=True,
            horizontal=True,
        )

        writer.add_figure(f"calibration_difference/joint{postfix}", fig)

        if "propagated_std" in all_aug_result_dict:
            bin_true, bin_pred = jax.vmap(
                lambda pred_std, pred_error: get_singleclass_calibration(
                    pred_std, pred_error, 10
                ),
                (0, 0),
                (0, 0),
            )(propagated_std, pred_error)

            calibration_error = jnp.abs(bin_true - bin_pred)

            fig = plot_lines(
                bin_pred,
                bin_true,
                aug_degree_names,
                "aug_degree",
                first_name="mean predicted std",
                second_name="average absolute error",
                add_line=True,
            )
            writer.add_figure(f"calibration/propagated{postfix}", fig)

            fig = plot_lines(
                bin_pred,
                calibration_error,
                aug_degree_names,
                "aug_degree",
                first_name="mean predicted std",
                second_name="deviation of error to std",
                add_line=True,
                horizontal=True,
            )

            writer.add_figure(f"calibration_difference/propagated{postfix}", fig)

        if "stage_aleatoric_std" in all_aug_result_dict:
            bin_true, bin_pred = jax.vmap(
                lambda pred_std, pred_error: get_singleclass_calibration(
                    pred_std, pred_error, 10
                ),
                (0, 0),
                (0, 0),
            )(stage_std, pred_error)

            calibration_error = jnp.abs(bin_true - bin_pred)

            fig = plot_lines(
                bin_pred,
                bin_true,
                aug_degree_names,
                "aug_degree",
                first_name="mean predicted std",
                second_name="average absolute error",
                add_line=True,
            )
            writer.add_figure(f"calibration/stage{postfix}", fig)

            fig = plot_lines(
                bin_pred,
                calibration_error,
                aug_degree_names,
                "aug_degree",
                first_name="mean predicted std",
                second_name="deviation of error to std",
                add_line=True,
                horizontal=True,
            )

            writer.add_figure(f"calibration_difference/stage{postfix}", fig)

        fig = plot_bar(
            range(len(calibration_error)),
            jnp.nanmean(calibration_error, 1),
            first_name="augmentation degree",
            second_name="expected calibration error",
            add_numbers=True,
        )

        writer.add_figure(f"expected_calibration/joint{postfix}", fig)

        max_error = max(jnp.max(jnp.abs(gt - pr)).item(), 1e-8)
        max_std = max(jnp.max(joint_std).item(), 1e-8)  # TODO

        for (
            aug_degree,
            selected_gt,
            selected_pr,
            selected_std,
            selected_category,
        ) in zip(itertools.count(), gt, pr, joint_std, category):
            fig = plot_2d_scatter(
                jnp.abs(selected_gt - selected_pr),
                selected_std,
                color=selected_category,
                first_max=max_error,
                second_max=max_std,
                first_name="error",
                second_name="standard deviation",
                color_name=target_name,
                add_line=False,
            )
            writer.add_figure(f"uncertainty_err_std{postfix}", fig, aug_degree)

    if FLAGS.propagator != "normal":
        input_mean_std = jnp.sqrt(all_aug_result_dict["input_var"])
        max_input_std = jnp.max(input_mean_std)

        acceleration_max = len(gt)
        acceleration_indices = jnp.broadcast_to(
            jnp.expand_dims(jnp.arange(len(gt)), tuple(range(1, gt.ndim))), gt.shape
        )

        if FLAGS.classifier:
            # TODO propagation classification i/o uncertainty

            max_mutual_info = jnp.max(mutual_info)
            max_entropy = jnp.max(entropy)
            max_confidence_std = jnp.max(confidence_std)
            max_class = len(class_labels)

            fig = plot_2d_scatter(
                input_mean_std.flatten(),
                mutual_info.flatten(),
                color=acceleration_indices.flatten(),
                first_max=max_input_std,
                second_max=max_mutual_info,
                color_max=acceleration_max,
                first_name="input standard deviation",
                second_name="mutual info",
                color_name="augmentation_index",
                add_line=False,
            )
            writer.add_figure(
                f"uncertainty_inputstd_mutualinfo/augmentation{postfix}",
                fig,
                aug_degree,
            )

            fig = plot_2d_scatter(
                input_mean_std.flatten(),
                confidence_std.flatten(),
                color=acceleration_indices.flatten(),
                first_max=max_input_std,
                second_max=max_confidence_std,
                color_max=acceleration_max,
                first_name="input standard deviation",
                second_name="standard deviation of confidence",
                color_name="augmentation_index",
                add_line=False,
            )
            writer.add_figure(
                f"uncertainty_inputstd_confidencestd/augmentation{postfix}",
                fig,
                aug_degree,
            )

            fig = plot_2d_scatter(
                input_mean_std.flatten(),
                entropy.flatten(),
                color=acceleration_indices.flatten(),
                first_max=max_input_std,
                second_max=max_entropy,
                color_max=acceleration_max,
                first_name="input standard deviation",
                second_name="entropy",
                color_name="augmentation_index",
                add_line=False,
            )
            writer.add_figure(
                f"uncertainty_inputstd_entropy/augmentation{postfix}", fig, aug_degree
            )

            for (
                aug_degree,
                selected_mutual,
                selected_entropy,
                selected_confidence_std,
                selected_input_std,
                selected_target,
            ) in zip(
                itertools.count(),
                mutual_info,
                entropy,
                confidence_std,
                input_mean_std,
                gt,
            ):

                fig = plot_2d_scatter(
                    selected_input_std,
                    selected_mutual,
                    color=selected_target,
                    first_max=max_input_std,
                    second_max=max_mutual_info,
                    color_max=max_class,
                    first_name="input standard deviation",
                    second_name="mutual info",
                    color_name=target_name,
                    add_line=False,
                )
                writer.add_figure(
                    f"uncertainty_inputstd_mutualinfo/{target_name}{postfix}",
                    fig,
                    aug_degree,
                )

                fig = plot_2d_scatter(
                    selected_input_std,
                    selected_entropy,
                    color=selected_target,
                    first_max=max_input_std,
                    second_max=max_entropy,
                    color_max=max_class,
                    first_name="input standard deviation",
                    second_name="entropy",
                    color_name=target_name,
                    add_line=False,
                )
                writer.add_figure(
                    f"uncertainty_inputstd_entropy/{target_name}{postfix}",
                    fig,
                    aug_degree,
                )
        else:
            max_joint_std = jnp.max(joint_std)
            max_propagated_std = jnp.max(propagated_std)
            max_stage_std = jnp.max(stage_std)

            fig = plot_2d_scatter(
                input_mean_std.flatten(),
                joint_std.flatten(),
                color=acceleration_indices.flatten(),
                first_max=max_input_std,
                second_max=max_joint_std,
                color_max=acceleration_max,
                first_name="input standard deviation",
                second_name="joint output standard deviation",
                color_name="augmentation_index",
                add_line=False,
            )
            writer.add_figure(
                f"uncertainty_inputstd_jointstd/augmentation{postfix}", fig, aug_degree
            )

            fig = plot_2d_scatter(
                input_mean_std.flatten(),
                propagated_std.flatten(),
                color=acceleration_indices.flatten(),
                first_max=max_input_std,
                second_max=max_propagated_std,
                color_max=acceleration_max,
                first_name="input standard deviation",
                second_name="propagated output standard deviation",
                color_name="augmentation_index",
                add_line=False,
            )
            writer.add_figure(
                f"uncertainty_inputstd_propstd/augmentation{postfix}", fig, aug_degree
            )

            fig = plot_2d_scatter(
                input_mean_std.flatten(),
                stage_std.flatten(),
                color=acceleration_indices.flatten(),
                first_max=max_input_std,
                second_max=max_stage_std,
                color_max=acceleration_max,
                first_name="input standard deviation",
                second_name="stage output standard deviation",
                color_name="augmentation_index",
                add_line=False,
            )
            writer.add_figure(
                f"uncertainty_inputstd_stagestd/augmentation{postfix}", fig, aug_degree
            )

            for (
                aug_degree,
                selected_joint_std,
                selected_propagated_std,
                selected_stage_std,
                selected_input_std,
                selected_category,
            ) in zip(
                itertools.count(),
                joint_std,
                propagated_std,
                stage_std,
                input_mean_std,
                category,
            ):
                fig = plot_2d_scatter(
                    selected_input_std,
                    selected_joint_std,
                    color=selected_category,
                    first_max=max_input_std,
                    second_max=max_joint_std,
                    first_name="input standard deviation",
                    second_name="joint output standard deviation",
                    color_name=target_name,
                    add_line=False,
                )
                writer.add_figure(
                    f"uncertainty_inputstd_jointstd/{target_name}{postfix}",
                    fig,
                    aug_degree,
                )

                fig = plot_2d_scatter(
                    selected_input_std,
                    selected_propagated_std,
                    color=selected_category,
                    first_max=max_input_std,
                    second_max=max_joint_std,
                    first_name="input standard deviation",
                    second_name="propagated output standard deviation",
                    color_name=target_name,
                    add_line=False,
                )
                writer.add_figure(
                    f"uncertainty_inputstd_propstd/{target_name}{postfix}",
                    fig,
                    aug_degree,
                )

                fig = plot_2d_scatter(
                    selected_input_std,
                    selected_stage_std,
                    color=selected_category,
                    first_max=max_input_std,
                    second_max=max_stage_std,
                    first_name="input standard deviation",
                    second_name="stage output standard deviation",
                    color_name=target_name,
                    add_line=False,
                )
                writer.add_figure(
                    f"uncertainty_inputstd_stagestd/{target_name}{postfix}",
                    fig,
                    aug_degree,
                )


def calc_calibration(all_aug_dict):

    if "confidence" in all_aug_dict:
        gt = all_aug_dict["gt"]

        confidence = all_aug_dict["confidence"]
        bin_true, bin_pred = jax.vmap(get_multiclass_calibration, (0, 0), (0, 0))(
            gt, confidence
        )

        calibration_error = jnp.abs(bin_true - bin_pred)
    elif "joint_std" in all_aug_dict:
        gt = all_aug_dict["gt"]
        pr = all_aug_dict["pr"]

        abs_error = jnp.abs(gt - pr)


def get_log_epoch_fn(
    writer,
    image_path,
    class_labels,
    target_name,
    image_res_factor=2,
    plot_res_factor=2,
    image_output=False,
    postfix="",
    add_barplot_numbers=True,
    # multi_uncertainty_compare=False,
    num_visual_images=1,
    num_console_predictions=0,
    formatting="",
):
    def log_epoch(name, tqdm_ds, epoch_index, metrics):

        if "input_wise_scalar" in metrics:
            scalar_dict = metrics["scalar"]
            input_wise_scalar_dict = metrics["input_wise_scalar"]
            input_class_wise_scalar_dict = metrics.get("input_class_wise_scalar", {})
            input_wise_full_size_dict = metrics["input_wise_full_size"]

            plot_imagewise(
                writer,
                image_path,
                target_name,
                class_labels,
                0,
                jax.tree_map(
                    itemgetter(slice(num_visual_images)), input_wise_full_size_dict
                ),
                jax.tree_map(
                    itemgetter(slice(num_visual_images)), input_wise_scalar_dict
                ),
                jax.tree_map(
                    itemgetter(slice(num_visual_images)), input_class_wise_scalar_dict
                ),
                image_res_factor=image_res_factor,
                plot_res_factor=plot_res_factor,
                writer_index=epoch_index,
                image_output=image_output,
                add_barplot_numbers=add_barplot_numbers,
                postfix=postfix,
            )

            if "sample_full_size" in metrics:
                np.save(
                    image_path / f"samples_{epoch_index:04}.npy",
                    metrics["sample_full_size"],
                )
                for key, value in input_wise_full_size_dict.items():
                    np.save(image_path / f"{key}_{epoch_index:04}.npy", value)

                sample_images = rearrange(
                    metrics["sample_full_size"], "s b w h c -> (b w) (s h) c"
                )

                sample_images_scaled = sample_images - sample_images.min()
                sample_images_scaled = sample_images_scaled / sample_images_scaled.max()

                writer.add_image(
                    f"input_samples/scaled{postfix}",
                    get_image(sample_images_scaled),
                    dataformats="HWC",
                )

                writer.add_image(
                    f"input_samples/clipped{postfix}",
                    get_image(sample_images, clipped_grayscale=True),
                    dataformats="HWC",
                )
                writer.add_image(
                    f"input_samples/colored{postfix}",
                    get_image(sample_images, bounded_greyscale=True),
                    dataformats="HWC",
                )

            for metric_name, metric_value in scalar_dict.items():
                if metric_value.size == 1:
                    writer.add_scalar(
                        f"{name}/{metric_name}{postfix}",
                        metric_value.item(),
                        epoch_index,
                    )

            tqdm_ds.write(
                f"eval epoch:{epoch_index:>3}, "
                + ", ".join(
                    f"{name}: {value:0.4f}{postfix}"
                    for name, value in scalar_dict.items()
                    if value.size == 1
                )
            )

            if "pr" in input_wise_scalar_dict:
                gt_array = input_wise_scalar_dict["gt"]
                pr_array = input_wise_scalar_dict["pr"]
                tqdm_ds.write(
                    ", ".join(
                        f"{gt.item():{formatting}}={pr.item():{formatting}}"
                        for gt, pr in zip(
                            gt_array[:num_console_predictions],
                            pr_array[:num_console_predictions],
                        )
                    )
                )

        if "element_input_wise_scalar" in metrics:
            element_wise_scalar_dict = metrics["element_wise_scalar"]
            element_input_class_wise_scalar_dict = metrics.get(
                "element_input_class_wise_scalar", {}
            )
            element_input_wise_scalar_dict = metrics["element_input_wise_scalar"]
            element_wise_full_size_dict = metrics["element_wise_full_size"]

            for i in range(num_visual_images):
                plot_imagewise(
                    writer,
                    image_path,
                    target_name,
                    class_labels,
                    i,
                    jax.tree_map(itemgetter(i), element_wise_full_size_dict),
                    jax.tree_map(itemgetter(i), element_input_wise_scalar_dict),
                    jax.tree_map(itemgetter(i), element_input_class_wise_scalar_dict),
                    image_res_factor=image_res_factor,
                    plot_res_factor=plot_res_factor,
                    writer_index=epoch_index,
                    image_output=image_output,
                    add_barplot_numbers=add_barplot_numbers,
                    postfix=postfix,
                )

            if "element_sample_wise_full_size" in metrics:
                sample_images = rearrange(
                    metrics["element_sample_wise_full_size"],
                    "s b i w h c -> i (b w) (s h) c",
                )

                sample_images_scaled = sample_images - sample_images.min()
                sample_images_scaled = sample_images_scaled / sample_images_scaled.max()

                for i, (sample_image, scaled_sample_image) in enumerate(
                    zip(sample_images, sample_images_scaled)
                ):
                    writer.add_image(
                        f"input_samples/scaled_{i}{postfix}",
                        get_image(scaled_sample_image),
                        dataformats="HWC",
                    )

                    writer.add_image(
                        f"input_samples/clipped_{i}{postfix}",
                        get_image(sample_image, clipped_grayscale=True),
                        dataformats="HWC",
                    )
                    writer.add_image(
                        f"input_samples/colored_{i}{postfix}",
                        get_image(sample_image, bounded_greyscale=True),
                        dataformats="HWC",
                    )

            for metric_name, metric_list in element_wise_scalar_dict.items():
                for aug_degree, metric_value in enumerate(metric_list):
                    if metric_value.size == 1:
                        writer.add_scalar(
                            f"data/{metric_name}{postfix}", metric_value, aug_degree
                        )

            if image_output:
                element_scalar_stats_dict = sample_resolution(
                    element_wise_full_size_dict
                )
            else:
                element_scalar_stats_dict = element_input_wise_scalar_dict

            plot_stats(
                writer,
                target_name,
                class_labels,
                element_scalar_stats_dict,
                postfix=postfix,
            )

    return log_epoch
