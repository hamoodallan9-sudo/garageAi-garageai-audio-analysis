# -*- coding: utf-8 -*-
"""
Registers the custom Keras layer/loss used by the currently-trained cnn_model.keras
(a BiGRU temporal head on top of the conv blocks, trained with focal loss), so
tf.keras.models.load_model() can deserialize it. This file must be imported
(even if nothing in it is called directly) BEFORE loading cnn_model.keras --
Keras resolves custom classes by the registered name baked into the saved
model's config, and that registration only happens when this module runs.

Mirrors car_test/cnn_model.py exactly (the script that trains and saves this
model) -- kept as its own copy here instead of importing across projects,
since this microservice is meant to be deployable independently of the
training-side project being present on the same machine.
"""

import tensorflow as tf
import keras
from tensorflow.keras import layers


@keras.saving.register_keras_serializable(package="car_test", name="FrequencyAveragePooling")
class FrequencyAveragePooling(layers.Layer):
    """Averages out the frequency axis (axis=1) of a (batch, freq, time, channels)
    log-mel feature map, leaving a (batch, time, channels) sequence for the BiGRU."""

    def call(self, inputs):
        return tf.reduce_mean(inputs, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2], input_shape[3])


@keras.saving.register_keras_serializable(package="car_test", name="SparseCategoricalFocalLoss")
class SparseCategoricalFocalLoss(tf.keras.losses.Loss):
    """Focal loss the model was trained with. Not actually needed for inference
    (model_loader.py loads with compile=False, so the loss is never
    reconstructed), but registered anyway for completeness / in case anything
    ever loads with compile=True."""

    def __init__(self, gamma=2.0, name="sparse_categorical_focal_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.gamma = gamma

    def call(self, y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        true_class_probs = tf.gather(y_pred, y_true, batch_dims=1)
        cross_entropy = -tf.math.log(true_class_probs)
        modulating_factor = tf.pow(1.0 - true_class_probs, self.gamma)
        return modulating_factor * cross_entropy

    def get_config(self):
        config = super().get_config()
        config.update({"gamma": self.gamma})
        return config
