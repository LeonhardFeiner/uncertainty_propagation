from typing import Any, Dict, List, Tuple, Union
from datetime import datetime
import pickle
import numpy as np
from numpy.typing import NDArray
from absl import app
from absl import flags
import jax.numpy as jnp
import jax
from jax import random
import tqdm

from tensorboardX import SummaryWriter
from pathlib import Path
from common.predict_epoch import get_predict_epoch_fn
from models.model_creation import get_train_model
from data.dataset_creation import (
    get_dataset_info,
    get_dataset_containers,
    get_dataloaders,
)


from models.uncertainty_propagator_creation import (
    remove_outer_layer,
    get_uncertainty_propagator,
)
from common.plot_epoch import get_log_epoch_fn
from common.loss_and_metrics_creator import get_loss_and_metrics
from common.apply_step import get_apply_step
from common.apply_samples import get_apply_samples_fn
from common.apply_epoch import get_apply_epoch, get_predict_epoch
from common.utils import rng_generator
import common.flags

AnyArray = Union[NDArray, jax.Array]
AnyFloatingArray = Union[NDArray[np.floating], jax.Array]


tqdm_ncols = 100
FLAGS = flags.FLAGS


def main(argv):
    print("Device used:", jax.lib.xla_bridge.get_backend().platform)
    del argv
    run_name = f"{FLAGS.run_name}{datetime.now():%Y-%m-%d_%H-%M-%S}"
    parent_path = Path(FLAGS.log_path).expanduser() / FLAGS.dataset

    if FLAGS.eval:
        path = parent_path / "_".join(FLAGS.subset) / run_name
    else:
        path = parent_path / run_name

    print()
    print(path)
    print()

    image_path = path / "images"
    image_path.mkdir(parents=True)
    writer = SummaryWriter(path)
    writer.add_text("flags", FLAGS.flags_into_string())

    if FLAGS.train:
        model_path = path / "model"
        model_path.mkdir()
    if FLAGS.predict:
        prediction_path = path / "prediction"
        prediction_path.mkdir()

    rng = random.PRNGKey(int(FLAGS.seed))
    (train_rng, model_key, dataloader_key, eval_key, prediction_key) = random.split(
        rng, 5
    )

    (
        data_shape,
        num_outputs,
        batch_modifier_train,
        batch_modifier_eval,
        loss_kwargs,
        eval_epoch_kwargs,
    ) = get_dataset_info()

    if FLAGS.train:
        print("Load datasets")

        if FLAGS.full_data:
            subsets = "trainfull", "valfull"
        else:
            subsets = "train1", "val1"

        subset_name_seeds = dict(
            zip(subsets, random.split(dataloader_key, len(subsets)))
        )

        (train_ds, val_ds), df = get_dataloaders(subset_name_seeds)

        (
            train_ds_generator,
            batches_per_epoch,
            val_ds,
        ) = get_dataset_containers(train_ds, val_ds)

        print("Load datasets finished")

        if df is not None:
            df.to_csv(path / "data.csv")
    else:
        batches_per_epoch = 1

    calc_device = None

    if FLAGS.classifier:
        num_output_channels = num_outputs + (1 if FLAGS.temperature_scaling else 0)
    else:
        num_output_channels = num_output_channels * (2 if FLAGS.stage_aleatoric else 1)

    get_model_kwargs = {}
    if FLAGS.propagator in {"mc"}:
        get_model_kwargs["batch_norm_reduction_axis_name"] = "propagator_samples"

    if FLAGS.load_run_parent_path is not None:
        load_model_path = (
            Path(FLAGS.load_run_parent_path).expanduser().resolve()
            / FLAGS.load_run_name
            / "model"
        )
        downstream_model_state_file = max(load_model_path.glob("epoch*.pickle"))
        with open(downstream_model_state_file, "rb") as file:
            get_model_kwargs["variables"] = pickle.load(file)

    (model, state, train_kwargs, val_kwargs) = get_train_model(
        model_key,
        data_shape,
        num_output_channels,
        batches_per_epoch=batches_per_epoch,
        **get_model_kwargs,
    )

    apply_samples_fn = None
    if FLAGS.propagator == "mc":
        model, variables = get_uncertainty_propagator(
            model, state.variables, max_vectorized_samples=64
        )
        state = type(state).create(apply_fn=model.apply, **variables, tx=state.tx)
        apply_samples_fn = get_apply_samples_fn(model)
    else:
        assert FLAGS.input_distribution == "dirac"

    if FLAGS.propagator != "normal":
        train_kwargs["predict_parameters"] = True
        train_kwargs["predict_samples"] = True
        val_kwargs["predict_parameters"] = True
        val_kwargs["predict_samples"] = True

        wrapper_loss_kwargs = dict(
            sampling_model=True,
            mean_var_model=False,
            sampling_loss=FLAGS.sampling_loss,
        )
    else:
        wrapper_loss_kwargs = dict(
            sampling_model=False,
            mean_var_model=False,
            sampling_loss=False,
        )

    param_counts = {
        key: sum(x.size for x in jax.tree_util.tree_leaves(part))
        for key, part in state.variables.items()
    }

    print("params count:", param_counts)

    tqdm_kwargs = {"ncols": 100, "leave": False}
    val_loss_kwargs = dict(calc_metrics=True, get_samples_fn=apply_samples_fn)

    (
        eval_loss_and_metrics_fn,
        get_eval_empty_metrics_aggregator,
        eval_metrics_aggregator,
        eval_metrics_combiner,
        eval_metrics_plotter,
    ) = get_loss_and_metrics(**loss_kwargs, **wrapper_loss_kwargs, **val_loss_kwargs)

    eval_step_fn = get_apply_step(
        loss_and_metrics_fn=eval_loss_and_metrics_fn,
        batch_modifier=batch_modifier_eval,
        **val_kwargs,
        device=calc_device,
        update_state=False,
        jit=FLAGS.jit,
    )

    eval_predict_epoch = get_predict_epoch("val", eval_step_fn, tqdm_kwargs=tqdm_kwargs)

    log_epoch = get_log_epoch_fn(
        writer,
        **eval_epoch_kwargs,
        image_path=image_path,
        postfix="",
    )

    if FLAGS.train:
        train_loss_kwargs = dict(calc_metrics=False)

        (
            train_loss_and_metrics_fn,
            get_train_empty_metrics_aggregator,
            train_metrics_aggregator,
            train_metrics_combiner,
            train_metrics_plotter,
        ) = get_loss_and_metrics(
            **loss_kwargs, **wrapper_loss_kwargs, **train_loss_kwargs
        )

        train_step_fn = get_apply_step(
            loss_and_metrics_fn=train_loss_and_metrics_fn,
            batch_modifier=batch_modifier_train,
            **train_kwargs,
            device=calc_device,
            update_state=True,
            jit=FLAGS.jit,
        )

        train_predict_epoch = get_predict_epoch(
            "train", train_step_fn, tqdm_kwargs=tqdm_kwargs
        )

        train_epoch_fn = get_apply_epoch(
            train_predict_epoch,
            get_empty_metrics_aggregator=get_train_empty_metrics_aggregator,
            metrics_aggregator=train_metrics_aggregator,
            metrics_combiner=train_metrics_combiner,
            metrics_plotter=train_metrics_plotter,
        )

        eval_epoch_fn = get_apply_epoch(
            eval_predict_epoch,
            get_empty_metrics_aggregator=get_eval_empty_metrics_aggregator,
            metrics_aggregator=eval_metrics_aggregator,
            metrics_combiner=eval_metrics_combiner,
            metrics_plotter=eval_metrics_plotter,
            epoch_metrics_plotter=log_epoch,
        )

        epoch_range = tqdm.trange(
            1, FLAGS.num_epochs + 1, desc="epochs", ncols=tqdm_ncols
        )

        offset = 0

        if FLAGS.eval_at_beginning:
            state, _, metrics = eval_epoch_fn(state, val_ds, eval_key, offset)

        variables = state.variables
        if FLAGS.propagator != "normal":
            variables = remove_outer_layer(variables)

        for epoch, epoch_train_ds, epoch_rng in zip(
            epoch_range, train_ds_generator, rng_generator(train_rng)
        ):
            state, offset, metrics = train_epoch_fn(
                state, epoch_train_ds, epoch_rng, offset
            )
            if 0 == FLAGS.eval_every or 0 == (epoch % FLAGS.eval_every):
                state, _, metrics = eval_epoch_fn(state, val_ds, eval_key, offset)

            writer.flush()

            variables = state.variables

            if FLAGS.propagator != "normal":
                variables = remove_outer_layer(variables)

            with open(model_path / f"epoch{epoch:03}.pickle", "wb") as file:
                pickle.dump(variables, file)

            if FLAGS.store_every and epoch > 0 and ((epoch - 1) % FLAGS.store_every):
                (model_path / f"epoch{epoch - 1:03}.pickle").unlink()

    if FLAGS.predict:
        # predict_epoch = get_predict_epoch_fn(
        #     **eval_epoch_kwargs, writer=writer
        # )

        subsets = ["train1", "val1", "test"]
        joint_prediction_dataset_key, joint_prediction_inference_key = random.split(
            prediction_key
        )
        prediction_dataset_keys = random.split(
            joint_prediction_dataset_key, len(subsets)
        )
        prediction_inference_keys = random.split(
            joint_prediction_inference_key, len(subsets)
        )

        prediction_datasets = get_dataloaders(
            dict(zip(subsets, prediction_dataset_keys)), multi_augmentation_batch=True
        )

        for subset, prediction_dataset, prediction_inference_key in zip(
            subsets, prediction_datasets, prediction_inference_keys
        ):
            current_prediction_path = prediction_path / subset
            current_prediction_path.mkdir()

            for prediction_dict in predict_epoch(
                eval_fn,
                variables,
                prediction_inference_key,
                prediction_dataset,
                subset,
            ):
                index = prediction_dict["index"]
                file_path = current_prediction_path / f"{index:06}.pickle"
                with open(file_path, "wb") as file:
                    pickle.dump(prediction_dict, file)

        print()
        print(prediction_path)
        print()

    if FLAGS.eval:
        key = random.PRNGKey(int(FLAGS.seed))
        propagation_key, dataloader_key = random.split(key)

        assert len(FLAGS.subset) == 1
        subset_name_seeds = dict(
            zip(FLAGS.subset, random.split(dataloader_key, len(FLAGS.subset)))
        )

        (dataset,), df = get_dataloaders(subset_name_seeds)

        (
            eval_loss_and_metrics_fn,
            get_eval_empty_metrics_aggregator,
            eval_metrics_aggregator,
            eval_metrics_combiner,
            eval_metrics_plotter,
        ) = get_loss_and_metrics(
            **loss_kwargs, **wrapper_loss_kwargs, **val_loss_kwargs
        )

        eval_predict_epoch = get_predict_epoch(
            "val", eval_step_fn, tqdm_kwargs=tqdm_kwargs
        )

        eval_epoch_fn = get_apply_epoch(
            eval_predict_epoch,
            get_empty_metrics_aggregator=get_eval_empty_metrics_aggregator,
            metrics_aggregator=eval_metrics_aggregator,
            metrics_combiner=eval_metrics_combiner,
            metrics_plotter=eval_metrics_plotter,
            epoch_metrics_plotter=log_epoch,
        )

        if df is not None:
            df.to_csv(path / "data.csv")

        state, _, metrics = eval_epoch_fn(state, dataset, eval_key, 0)

        jnp.savez(path / "results.npz", **metrics)

    print()
    print(path)
    print()


if __name__ == "__main__":
    jax.config.config_with_absl()
    app.run(main)
