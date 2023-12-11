
import tensorflow as tf
#import tensorflow.keras as keras
#import tensorflow.keras.layers as layers
import numpy as np

class PPOAgent():
    def __init__(self,
                 actor_model,
                 critic_model,
                 epsilon,
                 target_kl_div,
                 max_policy_iters,
                 max_value_iters,
                 policy_lr,
                 value_lr,
                 batch_size=16,
                 max_grad_norm=None,
                 checkpoint_dir='new_model/'):
        
        # Model Save Directory 
        #self.checkpoint_dir = 'models/'
        self.checkpoint_dir = checkpoint_dir
        
        # Initialize PPO Parameters   
        self.actor = actor_model
        self.critic = critic_model
        self.epsilon = epsilon
        self.target_kl_div = target_kl_div
        self.max_policy_train_iters = max_policy_iters
        self.max_value_train_iters = max_value_iters
        self.policy_lr = policy_lr
        self.value_lr = value_lr
        self.max_grad_norm = max_grad_norm

        # Initialize PPO Memory
        self.states = []
        self.actions = []
        self.values = []
        self.log_probs = []
        self.rewards = []

        self.batch_size = batch_size

        # Initialize Optimizers
        #self.actor_optimizer = tf.keras.optimizers.Adam(policy_lr, clipnorm=self.max_grad_norm)
        #self.critic_optimizer = tf.keras.optimizers.Adam(value_lr, clipnorm=self.max_grad_norm)

        # Compiling the Model Optimizers
        self.actor.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.policy_lr, clipnorm=self.max_grad_norm))
        self.critic.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.value_lr, clipnorm=self.max_grad_norm))
    
    def save_models(self):
        print('....saving weights....')
        self.actor.save(self.checkpoint_dir + 'actor')
        self.critic.save(self.checkpoint_dir + 'critic')

    def load_models(self):
        print('....loading_models....')
        self.actor = tf.keras.models.load_model(self.checkpoint_dir + 'actor')
        self.critic = tf.keras.models.load_model(self.checkpoint_dir + 'critic')

    def store_memory(self, state, action, value, log_prob, reward):
        self.states.append(state)
        self.actions.append(action)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)

    def access_memory(self):
        return np.asarray(self.states), np.asarray(self.actions), np.asarray(self.values), np.asarray(self.log_probs), np.asarray(self.rewards)
    
    def clear_memory(self):
        self.states = []
        self.actions = []
        self.values = []
        self.log_probs = []
        self.rewards = []

    def get_action_value(self, state):

        mean, std = self.actor(tf.convert_to_tensor(state[None, :], dtype=tf.float32))
        value = self.critic(tf.convert_to_tensor(state[None, :], dtype=tf.float32))

        action = tf.random.normal(shape=tf.shape(mean), mean=mean, stddev=std)
        action = tf.clip_by_value(action, -1, 1)

        log_prob = self.get_log_prob(mean, std, action)

        return action, value, log_prob

    def get_log_prob(self, mean, std, action):

        pre_sum = -0.5 * (((action - mean) / std) ** 2 + 2 * tf.math.log(std) + tf.math.log(2 * np.pi))
        log_prob = tf.reduce_sum(pre_sum, axis=-1)

        return log_prob
    
    def train_policy(self, states, actions, advantages, old_log_probs):
        for _ in range(self.max_policy_train_iters):

            with tf.GradientTape(persistent=True) as tape:
                batch_states = tf.convert_to_tensor(states, dtype=tf.float32)
                batch_actions = tf.convert_to_tensor(actions, dtype=tf.float32)
                batch_advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)
                batch_old_log_probs = tf.convert_to_tensor(old_log_probs, dtype=tf.float32)

                mean, std = self.actor(batch_states)

                new_log_probs = self.get_log_prob(mean, std, batch_actions)

                policy_ratio = tf.exp(new_log_probs - batch_old_log_probs)
                clipped_ratio = tf.clip_by_value(policy_ratio, 1 - self.epsilon, 1 + self.epsilon)

                surrogate1 = policy_ratio * batch_advantages
                surrogate2 = clipped_ratio * batch_advantages

                policy_loss = -tf.minimum(surrogate1, surrogate2)
                policy_loss = tf.reduce_mean(policy_loss)

                # Add entropy to policy loss
                #policy_loss = policy_loss + -0.005 * tf.reduce_mean(new_log_probs)

            actor_params = self.actor.trainable_variables
            gradients = tape.gradient(policy_loss, actor_params)

            self.actor.optimizer.apply_gradients(zip(gradients, actor_params))

            kl_div = tf.reduce_mean(batch_old_log_probs - new_log_probs)

            if kl_div >= self.target_kl_div:
                break

        return policy_loss

    def train_value(self, states, returns):
        for _ in range(self.max_value_train_iters):

            with tf.GradientTape(persistent=True) as tape:

                batch_states = tf.convert_to_tensor(states, dtype=tf.float32)
                batch_returns = tf.convert_to_tensor(returns, dtype=tf.float32)

                values = self.critic(batch_states)

                #value_loss = 0.9 * tf.keras.losses.MSE(values, batch_returns)
                value_loss =  0.5 * tf.reduce_mean(tf.square(values - batch_returns))

            critic_params = self.critic.trainable_variables
            gradients = tape.gradient(value_loss, critic_params)

            self.critic.optimizer.apply_gradients(zip(gradients, critic_params))

        return value_loss

