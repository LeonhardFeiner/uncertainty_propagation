import itertools
from operator import itemgetter
from jax import random
import jax
import jax.numpy as jnp
import tqdm
from common.image_plot_utils import get_image, combine_image_collection, get_text_img
import numpy as np
from common.utils import rng_generator

tqdm_ncols = 100


def get_predict_epoch_fn(
    batch_selector,
    label_rescaler=lambda x: x,
    writer=None,
    postfix="",
    num_visual_imgs=8,
    result_device=None,
    image_res_factor=1,
    plot_res_factor=1,
    **unused,
):
    def predict_epoch(
        eval_fn,
        variables,
        rng,
        dataset,
        name,
    ):
        tqdm_epoch = tqdm.tqdm(
            dataset, desc=f"predict {name}", leave=False, ncols=tqdm_ncols
        )

        inference_ds, eval_ds = itertools.tee(tqdm_epoch)

        evaluation_results_images = (
            jax.device_put(
                label_rescaler(eval_fn(variables, batch_selector(batch), rng)),
                result_device,
            )
            for batch, rng in zip(inference_ds, rng_generator(rng))
        )

        cpu_eval_ds = (jax.device_put(x, result_device) for x in eval_ds)
        zipped_eval_images = zip(
            itertools.count(), cpu_eval_ds, evaluation_results_images
        )

        if writer is not None and "test" not in name:
            zipped_eval_images, zipped_visualize_images = itertools.tee(
                zipped_eval_images
            )

            for i, batch, prediction in itertools.islice(
                zipped_visualize_images, num_visual_imgs
            ):
                augmented, original, labels = batch

                dpi = original.shape[-2] * image_res_factor // plot_res_factor

                min_original = jnp.min(original)
                max_original = jnp.max(original)
                range_original = max_original - min_original

                normalized_original = (original - min_original) / range_original
                normalized_augmented = (augmented - min_original) / range_original
                normalized_prediction = (
                    prediction["pr"] - min_original
                ) / range_original

                raw_image_plots = {
                    "original": get_image(
                        normalized_original,
                        bounded_greyscale=True,
                        scale=image_res_factor,
                    ),
                    "augmented": get_image(
                        normalized_augmented,
                        bounded_greyscale=True,
                        scale=image_res_factor,
                    ),
                    "prediction": get_image(
                        normalized_prediction,
                        bounded_greyscale=True,
                        scale=image_res_factor,
                    ),
                }
                if "stage_aleatoric_std" in prediction:
                    raw_image_plots["stage_aleatoric_std"] = get_image(
                        prediction["stage_aleatoric_std"] / range_original,
                        bounded_greyscale=True,
                        scale=image_res_factor,
                    )

                if "stage_epistemic_std" in prediction:
                    raw_image_plots["stage_epistemic_std"] = get_image(
                        prediction["stage_epistemic_std"] / range_original,
                        bounded_greyscale=True,
                        scale=image_res_factor,
                    )

                # raw_image_plots["gt_diff"] = get_image(
                #     gt_diff, difference_color=True, scale=image_res_factor
                # )

                # input_diff = normalized_mean_input_image - image_pr
                # raw_image_plots["input_diff"] = get_image(
                #     input_diff, difference_color=True, scale=image_res_factor
                # )

                image = combine_image_collection(list(raw_image_plots.values()))

                text_image = get_text_img(
                    raw_image_plots.keys(),
                    dpi=dpi,
                    image_res_factor=image_res_factor,
                    plot_res_factor=plot_res_factor,
                )

                image = jnp.concatenate((text_image, image), -2)

                writer.add_image(
                    f"predicted_image/{name}_{i}{postfix}",
                    image,
                    0,
                    dataformats="HWC",
                )

        for i, batch, prediction in zipped_eval_images:

            variances_dict = dict()
            if "stage_aleatoric_std" in prediction:
                variances_dict["aleatoric"] = jnp.square(
                    prediction["stage_aleatoric_std"]
                )
            if "stage_epistemic_std" in prediction:
                variances_dict["epistemic"] = jnp.square(
                    prediction["stage_epistemic_std"]
                )

            metrics_dict = {
                key: value for key, value in prediction.items() if value.ndim == 1
            }
            augmented, original, labels = batch

            label_dict = jax.tree_util.tree_map(lambda x: np.array(x[0]), labels)

            original_single = original[0]

            yield dict(
                index=i,
                original=original_single,
                augmented=augmented,
                mean=prediction["pr"],
                **variances_dict,
                **label_dict,
                **metrics_dict,
            )

    return predict_epoch
