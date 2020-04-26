import tensorflow as tf
from tensorflow.keras import layers


class EvalModel(tf.keras.Model):
    def __init__(self, num_actions, weights, bias):
        super().__init__('eval_network')
        self.layer1 = layers.Dense(30, activation=tf.nn.relu, name='layer1',
                                   use_bias=True, kernel_initializer=weights, bias_initializer=bias)
        self.logits = layers.Dense(num_actions, activation=None, name='layer2',
                                   use_bias=True, kernel_initializer=weights, bias_initializer=bias)

    def call(self, inputs):
        x = tf.convert_to_tensor(inputs)
        layer1 = self.layer1(x)
        logits = self.logits(layer1)
        return logits


class TargetModel(tf.keras.Model):
    def __init__(self, num_actions, weights, bias):
        super().__init__('target_network')
        self.layer1 = layers.Dense(30, trainable=False, activation=tf.nn.relu, name='layer1',
                                   use_bias=True, kernel_initializer=weights, bias_initializer=bias)
        self.logits = layers.Dense(num_actions, trainable=False, activation=None, name='layer2',
                                   use_bias=True, kernel_initializer=weights, bias_initializer=bias)

    def call(self, inputs):
        x = tf.convert_to_tensor(inputs)
        layer1 = self.layer1(x)
        logits = self.logits(layer1)
        return logits
