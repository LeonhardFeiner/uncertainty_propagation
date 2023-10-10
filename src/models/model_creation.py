from absl import flags

import models.simple
import models.resnet
import models.autoencoder

FLAGS = flags.FLAGS


def get_evalmodel(num_output_channels):
    if FLAGS.model == "simple":
        get_model = models.simple.get_model
        model_kwargs = {}
        val_kwargs = models.simple.val_kwargs
        max_vectorized_samples = 256
    elif FLAGS.model == "resnest":
        get_model = models.resnet.get_model
        model_kwargs = {
            "name": "ResNeSt",
            "use_batch_norm": FLAGS.use_batch_norm,
            "use_max_pool": FLAGS.use_max_pool,
        }
        val_kwargs = models.resnet.val_kwargs
        max_vectorized_samples = 64
    elif FLAGS.model == "resnet":
        get_model = models.resnet.get_model
        model_kwargs = {
            "name": "ResNet",
            "use_batch_norm": FLAGS.use_batch_norm,
            "use_max_pool": FLAGS.use_max_pool,
        }
        val_kwargs = models.resnet.val_kwargs
        max_vectorized_samples = 64
    elif FLAGS.model == "simpleauto":
        get_model = models.autoencoder.get_model
        val_kwargs = dict(training=False)
        model_kwargs = {
            "residual": not FLAGS.classifier,
            "dropout_rate": FLAGS.dropout_rate,
            "model_name": FLAGS.model,
        }
        max_vectorized_samples = 64
    elif FLAGS.model == "unet":
        get_model = models.autoencoder.get_model
        val_kwargs = dict(training=False)
        model_kwargs = {
            "residual": not FLAGS.classifier,
            "dropout_rate": FLAGS.dropout_rate,
            "model_name": FLAGS.model,
        }
        max_vectorized_samples = 64
    elif FLAGS.model == "dummyauto":
        get_model = models.autoencoder.get_model
        val_kwargs = dict()
        model_kwargs = {
            "residual": not FLAGS.classifier,
            "dropout_rate": FLAGS.dropout_rate,
            "model_name": FLAGS.model,
        }
        max_vectorized_samples = 64
    else:
        raise NotImplementedError

    model = get_model(num_output_channels, **model_kwargs)
    return model, val_kwargs, max_vectorized_samples


def get_train_model(
    model_key,
    data_shape,
    num_output_channels,
    batches_per_epoch,
    batch_norm_reduction_axis_name=None,
    **kwargs,
):
    if FLAGS.model == "simple":
        get_model_state = models.simple.get_model_state
        model_kwargs = {}
    elif FLAGS.model == "resnest":
        get_model_state = models.resnet.get_model_state
        model_kwargs = {
            "name": "ResNeSt",
            "use_batch_norm": FLAGS.use_batch_norm,
            "use_max_pool": FLAGS.use_max_pool,
            "pretrained": FLAGS.pretrained,
            "batch_norm_reduction_axis_name": batch_norm_reduction_axis_name,
        }
    elif FLAGS.model == "resnet":
        get_model_state = models.resnet.get_model_state
        model_kwargs = {
            "name": "ResNet",
            "use_batch_norm": FLAGS.use_batch_norm,
            "use_max_pool": FLAGS.use_max_pool,
            "pretrained": FLAGS.pretrained,
            "batch_norm_reduction_axis_name": batch_norm_reduction_axis_name,
        }
    elif FLAGS.model == "simpleauto":
        get_model_state = models.autoencoder.get_model_state
        model_kwargs = {
            "residual": not FLAGS.classifier,
            "dropout_rate": FLAGS.dropout_rate,
            "model_name": FLAGS.model,
        }
    elif FLAGS.model == "unet":
        get_model_state = models.autoencoder.get_model_state
        model_kwargs = {
            "residual": not FLAGS.classifier,
            "dropout_rate": FLAGS.dropout_rate,
            "model_name": FLAGS.model,
            "pretrained": FLAGS.pretrained,
            "use_batch_norm": FLAGS.use_batch_norm,
            "use_max_pool": FLAGS.use_max_pool,
            "batch_norm_reduction_axis_name": batch_norm_reduction_axis_name,
        }
    elif FLAGS.model == "dummyauto":
        get_model_state = models.autoencoder.get_model_state
        model_kwargs = {
            "residual": not FLAGS.classifier,
            "dropout_rate": FLAGS.dropout_rate,
            "model_name": FLAGS.model,
        }
    else:
        raise NotImplementedError

    return get_model_state(
        model_key,
        data_shape,
        num_output_channels,
        batches_per_epoch=batches_per_epoch,
        **model_kwargs,
        **kwargs,
    )
