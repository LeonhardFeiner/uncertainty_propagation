from absl import flags
from models.uncertainty_propagator import (
    SamplePropagator,
)
from models.sampler import DiagNormalMonteCarloSampler
from models.common import add_outer_layer, remove_outer_layer

FLAGS = flags.FLAGS


def get_uncertainty_propagator(model, variables, max_vectorized_samples=16):

    if FLAGS.propagator == "normal":
        assert FLAGS.input_distribution == "dirac"
        return model, variables
    elif FLAGS.propagator == "mc":
        propagator_class = SamplePropagator
        assert FLAGS.input_distribution == "mvnd"
        sampler = DiagNormalMonteCarloSampler(FLAGS.num_samples)
        args = []
        kwargs = {
            "max_vectorized_samples": max_vectorized_samples,
            "sampler": sampler,
        }
    else:
        raise NotImplementedError

    combined_model = propagator_class(model, *args, **kwargs)

    changed_variables = add_outer_layer(variables)

    return combined_model, changed_variables
