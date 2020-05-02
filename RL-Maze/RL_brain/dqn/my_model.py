import tensorflow as tf
from tensorflow.keras import layers


class EvalModel(tf.keras.Model):
    def __init__(self, num_actions):
        super().__init__('eval_network')
        self.layer1 = layers.Dense(256, activation=tf.nn.relu, name='layer1')
        self.layer2 = layers.Dense(512, activation=tf.nn.relu, name='layer2')
        self.layer3 = layers.Dense(512, activation=tf.nn.relu, name='layer3')
        self.logits = layers.Dense(num_actions, activation=None, name='output')

    def call(self, inputs):
        x = tf.convert_to_tensor(inputs)
        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        logits = self.logits(layer3)
        return logits


class TargetModel(tf.keras.Model):
    def __init__(self, num_actions):
        super().__init__('target_network')
        self.layer1 = layers.Dense(256, activation=tf.nn.relu, name='layer1')
        self.layer2 = layers.Dense(512, activation=tf.nn.relu, name='layer2')
        self.layer3 = layers.Dense(512, activation=tf.nn.relu, name='layer3')
        self.logits = layers.Dense(num_actions, activation=None, name='output')

    def call(self, inputs):
        x = tf.convert_to_tensor(inputs)
        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        logits = self.logits(layer3)
        return logits
