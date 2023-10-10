from pathlib import Path
import pickle

pretrainings = {
    ("unet", False, False, False): Path(
        "./weights/no_maxpool_no_batchnorm_epoch40.pickle"
    ).expanduser(),
    ("unet", True, True, False): Path(
        "./weights/maxpool_batchnorm_epoch40.pickle"
    ).expanduser(),
}


def get_pretrained_weights(model_name, use_max_pool, use_batch_norm, dropout_rate):
    path = pretrainings[(model_name, use_max_pool, use_batch_norm, bool(dropout_rate))]

    with open(path, "rb") as file:
        return pickle.load(file)
