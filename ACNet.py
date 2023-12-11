import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import numpy as np


class ActorNetwork(tf.keras.Model):
    def __init__(self, action_dim, hidden_layer_size, num_hidden_layers):
        super(ActorNetwork, self).__init__()

        # Initialize Parameters
        self.num_hidden_layers = num_hidden_layers
        self.hidden_layer_size = hidden_layer_size
        self.action_dim = action_dim

        self.build_net()

    def build_net(self):

        # Shared Layers
        self.shared_layers = tf.keras.Sequential()
        for _ in range(self.num_hidden_layers):
            self.shared_layers.add(layers.Dense(self.hidden_layer_size, activation='relu'))

        # Actor Layers for Mean
        self.actor_mean = tf.keras.Sequential([
            #layers.Dense(32, activation='relu'),
            layers.Dense(self.action_dim, activation='tanh')
        ])

        # Actor Layers for LOG STD
        self.actor_std = tf.keras.Sequential([
            #layers.Dense(32, activation='relu'),
            layers.Dense(self.action_dim, activation='softplus')
        ])

    def call(self, state_input):

        z = self.shared_layers(state_input)

        mean = self.actor_mean(z)

        std = self.actor_std(z)

        return mean, std


class CriticNetwork(tf.keras.Model):
    def __init__(self, hidden_layer_size, num_hidden_layers):
        super(CriticNetwork, self).__init__()

        # Initialize Parameters
        self.hidden_layer_size = hidden_layer_size
        self.num_hidden_layers = num_hidden_layers
        
        self.build_net()

    def build_net(self):
        
        # Value Layers
        self.value_layers = tf.keras.Sequential()
        for _ in range(self.num_hidden_layers):
            self.value_layers.add(layers.Dense(self.hidden_layer_size, activation='relu'))
        
        self.value_layers.add(layers.Dense(1, activation=None))
    
    def call(self, state_input):

        value = self.value_layers(state_input)

        return value