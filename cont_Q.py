import gym
from keras.models import Model
from ops import createLayers
from keras.optimizers import Adam
from keras import backend as K
import numpy as np

batch_size = 100
hidden_size = 100
layers = 2
batch_norm = False
min_train = 10
train_repeat = 10
gamma = 0.9
tau = 0.001
episodes = 500
activation = 'tanh'
optimizer = 'adam'
optimizer_lr = 0.001
noise_decay = 'linear'
fixed_noise  = 0.1
display = False
environment = 'Pendulum-v0'
gym_monitor = False
max_timesteps = 200

env = gym.make(environment)
num_actuators = env.action_space.shape[0]


x, u, m, v, q = createLayers('model1', env)
_mu = K.function([K.learning_phase(), x], m)
mu = lambda x: _mu([0] + [x])
model = Model(input=[x,u], output=q)
model.summary()
optimizer = Adam(optimizer_lr)
model.compile(optimizer=optimizer, loss='mse')

x, u, m, v, q = createLayers('model2', env)
_V = K.function([K.learning_phase(), x], v)
V = lambda x: _V([0] + [x])
target_model = Model(input=[x,u], output=q)
target_model.set_weights(model.get_weights())

prestates = []
actions = []
rewards = []
poststates = []
terminals = []
episode_reward_list =[]
total_reward = 0
for i_episode in xrange(episodes):
    observation = env.reset()
    print "initial state:", observation, max_timesteps
    episode_reward = 0
    for t in xrange(max_timesteps):
        if display:
          env.render()

        x = np.array([observation])
        u = mu(x)
        if noise_decay == 'linear':
          noise = 1. / (i_episode + 1)
        elif noise_decay == 'exp':
          noise = 10 ** -i_episode
        elif noise_decay == 'fixed':
          noise = fixed_noise
        else:
          assert False
        action = u[0] + np.random.randn(num_actuators) * noise
        prestates.append(observation)
        actions.append(action)
        observation, reward, done, info = env.step(action)
        episode_reward += reward
        rewards.append(reward)
        poststates.append(observation)
        terminals.append(done)
        if len(prestates) > min_train:
         loss =0
         for k in xrange(train_repeat):
            if len(prestates) > batch_size:
              indexes = np.random.choice(len(prestates), size=batch_size)
            else:
              indexes = range(len(prestates))

            v = V(np.array(poststates)[indexes])
            y = np.array(rewards)[indexes] + gamma * np.squeeze(v)
            loss += model.train_on_batch([np.array(prestates)[indexes], np.array(actions)[indexes]], y)
            weights = model.get_weights()
            target_weights = target_model.get_weights()
            for i in xrange(len(weights)):
              target_weights[i] = tau * weights[i] + (1 - tau) * target_weights[i]
            target_model.set_weights(target_weights)
        if done:
            break

    episode_reward = episode_reward / float(t + 1)
    print "Episode {} finished after {} timesteps, average reward {}".format(i_episode + 1, t + 1, episode_reward)
    total_reward += episode_reward
    episode_reward_list.append(episode_reward)

episode_reward_list = np.asarray(episode_reward_list)
np.savez('episode_reward_list_Cont_q_500.npz', episode_reward_list)
print "Average reward per episode {}".format(total_reward / episodes)
