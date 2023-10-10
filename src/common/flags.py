from pathlib import Path
from absl import flags

flags.DEFINE_string(
    "run_name",
    default="",
    help=("name of tensorboard logs and checkpoints"),
)


flags.DEFINE_integer("num_epochs", default=40, help=("Number of training epochs."))

flags.DEFINE_integer("seed", default=0, help=("Seed for randomness."))
flags.DEFINE_integer("batch_size", default=8, help=("Seed for randomness."))

flags.DEFINE_list("mri_accelerations", default=["02", "04", "06", "08"], help="")

flags.DEFINE_list("slice_list", default=[], help="")

flags.DEFINE_float("dropout_rate", 0, "Dropout rate.")
flags.DEFINE_integer("overfit", 0, "overfit samples")
flags.DEFINE_bool("use_gt", False, "")
flags.DEFINE_bool("original_data", False, "")
flags.DEFINE_bool("full_data", False, "")
flags.DEFINE_bool("augment", False, "")

flags.DEFINE_bool("pretrained", True, "whether to use a pretrained network if possible")

flags.DEFINE_bool(
    "classifier", True, "whether to use predict classification instead of regression"
)

flags.DEFINE_bool(
    "dice_loss", False, "whether to use predict classification instead of regression"
)

flags.DEFINE_bool(
    "l2", True, "whether to use predict classification instead of regression"
)

flags.DEFINE_bool(
    "jit", True, "whether to use predict classification instead of regression"
)

flags.DEFINE_bool("eval_at_beginning", True, "")


flags.DEFINE_bool(
    "binary", False, "whether to use predict classification instead of regression"
)

flags.DEFINE_bool("logits", True, "whether to use predict logits instead of confidence")

flags.DEFINE_bool(
    "stage_aleatoric", False, "whether to use predict variance of predictions"
)

flags.DEFINE_bool(
    "temperature_scaling",
    False,
    "whether to use temperature scaling for classification",
)

flags.DEFINE_bool(
    "detach_loss",
    False,
    "whether to a detached version of prediction to compute loss for uncertainty",
)


flags.DEFINE_bool(
    "multi_augmentation_batch", False, "whether to use predict variance of predictions"
)

flags.DEFINE_enum(
    "model",
    "simple",
    ["simple", "resnet", "resnest", "simpleauto", "unet", "dummyauto"],
    help=("name of model"),
)

flags.DEFINE_bool("use_max_pool", False, "")
flags.DEFINE_bool("use_batch_norm", False, "")

flags.DEFINE_enum(
    "dataset",
    "knee",
    [
        "knee",
        "adni",
    ],
    "dataset to use",
    case_sensitive=False,
)

flags.DEFINE_bool("masked_foreground", False, "")


flags.DEFINE_enum(
    "adni_target",
    "alzheimers",
    ["alzheimers", "alzheimers_binary", "age", "brain_vol", "sex"],
    "later",
)



flags.DEFINE_float("input_uncertainty_factor", 1, "multiplier to input uncertainty")
flags.DEFINE_bool(
    "artificial_uncertainty_gradient",
    False,
    "Whether to use artificial uncertainty gradient instead of original data",
)


flags.DEFINE_string(
    "pytorch_data_path",
    default="~/datasets",
    help="pytorch data path",
)


flags.DEFINE_multi_enum(
    "subset",
    ("train1", "val1"),
    ["train0", "val0", "train1", "val1", "test"],
    "subsets to use",
    case_sensitive=False,
)

flags.DEFINE_string(
    "csv_path",
    None,
    "dataset to use",
)

flags.DEFINE_string(
    "train_prediction_path",
    None,
    "dataset to use",
)

flags.DEFINE_string(
    "val_prediction_path",
    None,
    "dataset to use",
)

flags.DEFINE_string(
    "test_prediction_path",
    None,
    "dataset to use",
)

flags.DEFINE_string(
    "original_data_path",
    None,
    "dataset to use",
)

flags.DEFINE_float(
    "learning_rate", default=1e-3, help="The learning rate for the Adam optimizer."
)

flags.DEFINE_integer("test_batch_size", default=1024, help="Batch size for testing.")

flags.DEFINE_integer("eval_every", default=0, help=".")
flags.DEFINE_integer("store_every", default=10, help=".")


flags.DEFINE_string(
    "log_path",
    None,
    "Parent path of tensorboard logs and checkpoints",
)


flags.DEFINE_string(
    "load_run_parent_path",
    None,
    "name of tensorboard logs and checkpoints",
)

flags.DEFINE_string(
    "load_run_name",
    default="",
    help=("name of tensorboard logs and checkpoints"),
)


flags.DEFINE_string(
    "reconstruction_data_path",
    None,
    "Path of the data",
)

flags.DEFINE_integer(
    "num_augmentation_degrees",
    default=8,
    help="Number of different augmentation degrees.",
)

flags.DEFINE_enum(
    "propagator",
    "normal",
    [
        "normal",
        "mc",
    ],
    help="type of uncertainty propagator",
)

flags.DEFINE_enum(
    "input_distribution",
    "mvnd",
    [
        "dirac",
        "mvnd",
    ],
    help="type of uncertainty distribution",
)

flags.DEFINE_integer(
    "num_samples", 8, help="number of samples of uncertainty propagator"
)


flags.DEFINE_bool(
    "sampling_loss", True, "whether to incorporate the propagation in the loss"
)



flags.DEFINE_boolean(
    "train",
    default=False,
    help=("Wether we want to train."),
)

flags.DEFINE_boolean(
    "eval",
    default=False,
    help=("Wether we want to eval at the end."),
)

flags.DEFINE_boolean(
    "predict",
    default=False,
    help=("Wether we want to create predictions for downstream tasks."),
)


flags.DEFINE_integer(
    "num_workers",
    default=0,
    help="Number of workers used in dataloaders",
)

