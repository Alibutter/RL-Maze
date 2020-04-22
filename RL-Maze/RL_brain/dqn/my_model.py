import tensorflow as tf
from tensorflow.keras import layers


class EvalModel(tf.keras.Model):
    def __init__(self, num_actions):
        super().__init__('eval_network')
        self.layer1 = layers.Dense(40, activation='relu', name='layer1')
        self.logits = layers.Dense(num_actions, activation=None, name='eval_output')

    def call(self, inputs):
        x = tf.convert_to_tensor(inputs)
        layer1 = self.layer1(x)
        logits = self.logits(layer1)
        return logits


class TargetModel(tf.keras.Model):
    def __init__(self, num_actions):
        super().__init__('target_network')
        self.layer1 = layers.Dense(40, trainable=False, activation='relu', name='layer1')
        self.logits = layers.Dense(num_actions, trainable=False, activation=None, name='target_output')

    def call(self, inputs):
        x = tf.convert_to_tensor(inputs)
        layer1 = self.layer1(x)
        logits = self.logits(layer1)
        return logits
