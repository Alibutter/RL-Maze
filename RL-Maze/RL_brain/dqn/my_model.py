import tensorflow as tf
from tensorflow.keras import layers


class EvalModel(tf.keras.Model):
    def __init__(self, num_actions):
        super().__init__('mlp_q_network')
        self.layer1 = layers.Dense(40, activation='relu')
        self.logits = layers.Dense(num_actions, activation=None)

    def call(self, inputs):
        x = tf.convert_to_tensor(inputs)
        layer1 = self.layer1(x)
        logits = self.logits(layer1)
        return logits


class TargetModel(tf.keras.Model):
    def __init__(self, num_actions):
        super().__init__('mlp_q_network_1')
        self.layer1 = layers.Dense(40, trainable=False, activation='relu')
        self.logits = layers.Dense(num_actions, trainable=False, activation=None)

    def call(self, inputs):
        x = tf.convert_to_tensor(inputs)
        layer1 = self.layer1(x)
        logits = self.logits(layer1)
        return logits
