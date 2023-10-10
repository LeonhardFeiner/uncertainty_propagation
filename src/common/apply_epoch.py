from tqdm import tqdm
from common.utils import rng_generator


def get_apply_epoch(
    predict_epoch,
    *,
    get_empty_metrics_aggregator,
    metrics_aggregator,
    metrics_combiner,
    metrics_plotter,
    epoch_metrics_plotter=lambda name, tqdm_ds, step, metrics: None,
):
    def apply_epoch(
        state,
        epoch_ds,
        rng,
        offset=0,
    ):

        micro_step = None
        aggregator = get_empty_metrics_aggregator()

        name, tqdm_ds, iterator = predict_epoch(state, epoch_ds, rng, offset)

        for micro_step, step, state, metrics in iterator:
            metrics_aggregator(micro_step, aggregator, metrics)
            metrics_plotter(name, tqdm_ds, step, metrics)

        combined_metrics = metrics_combiner(aggregator)
        assert micro_step is not None, f"dataset {name} is empty"

        epoch_metrics_plotter(name, tqdm_ds, step + 1, combined_metrics)

        return state, step + 1, combined_metrics

    return apply_epoch


def get_predict_epoch(name, apply_step, tqdm_kwargs={"ncols": 100, "leave": False}):
    def inner_predict(state, tqdm_ds, rng, offset):
        for step, (batch, rng) in enumerate(zip(tqdm_ds, rng_generator(rng))):
            state, all_metrics = apply_step(state, batch, rng)
            yield step, step + offset, state, all_metrics

    def predict_epoch(
        state,
        epoch_ds,
        rng,
        offset=0,
    ):
        tqdm_ds = tqdm(epoch_ds, name, **tqdm_kwargs)

        return name, tqdm_ds, inner_predict(state, tqdm_ds, rng, offset)

    return predict_epoch
