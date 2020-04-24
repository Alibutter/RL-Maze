import tensorflow as tf
from tensorflow.keras import layers
weights = tf.random_normal_initializer(0., 0.3)
bias = tf.constant_initializer(0.1)


class EvalModel(tf.keras.Model):
    def __init__(self, num_actions):
        super().__init__('eval_network')
        self.layer1 = layers.Dense(30, activation='relu', name='layer1',
                                   use_bias=True, kernel_initializer=weights, bias_initializer=bias)
        self.logits = layers.Dense(num_actions, activation=None, name='eval_output')

    def call(self, inputs):
        x = tf.convert_to_tensor(inputs)
        layer1 = self.layer1(x)
        logits = self.logits(layer1)
        return logits


class TargetModel(tf.keras.Model):
    def __init__(self, num_actions):
        super().__init__('target_network')
        self.layer1 = layers.Dense(30, trainable=False, activation='relu', name='layer1',
                                   use_bias=True, kernel_initializer=weights, bias_initializer=bias)
        self.logits = layers.Dense(num_actions, trainable=False, activation=None, name='target_output')

    def call(self, inputs):
        x = tf.convert_to_tensor(inputs)
        layer1 = self.layer1(x)
        logits = self.logits(layer1)
        return logits
