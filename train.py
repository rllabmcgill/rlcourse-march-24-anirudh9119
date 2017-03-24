import numpy as np
import theano
import theano.tensor as T
from util import  norm_weight, _p , itemlist  #load_params, create_log_dir,  save_params
from collections import OrderedDict
import gym
import optimizers
batch_size = 100
hidden_size = 100
batch_norm = False
min_train = 10
train_repeat = 10
gamma = 0.9
tau = 0.001
episodes = 200
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

if gym_monitor:
    env.monitor.start(gym_monitor)


def param_init_fflayer(options, params, prefix='ff',
                       nin=None, nout=None, ortho=True, flag=False):
    params[_p(prefix, 'W')] = norm_weight(nin, nout, scale=0.01, ortho=ortho)
    params[_p(prefix, 'b')] = np.zeros((nout,)).astype('float32')
    return params

def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams

layers = {'ff': ('param_init_fflayer', 'fflayer')}

def get_layer(name):
        fns = layers[name]
        return (eval(fns[0]), eval(fns[1]))


def fflayer(tparams, state_below, options, prefix='rconv',
            activ='lambda x: tensor.tanh(x)', **kwargs):
    return T.dot(state_below, tparams[_p(prefix, 'W')]) + tparams[_p(prefix, 'b')]


use_conv = False
def init_params(options):
    params = OrderedDict()
    print 'Initializing Params'
    params = get_layer('ff')[0](options, params, prefix='layer_1', nin=3, nout=100, ortho=False)
    params = get_layer('ff')[0](options, params, prefix='layer_2', nin=100, nout=100, ortho=False)
    params = get_layer('ff')[0](options, params, prefix='layer_v', nin=100, nout=1, ortho=False)
    params = get_layer('ff')[0](options, params, prefix='layer_m', nin=100, nout=1, ortho=False)
    params = get_layer('ff')[0](options, params, prefix='layer_l', nin=100, nout=1, ortho=False)
    return params

def init_params_target(options):
    params = OrderedDict()
    print 'Initializing Params'
    params = get_layer('ff')[0](options, params, prefix='target_layer_1', nin=3, nout=100, ortho=False)
    params = get_layer('ff')[0](options, params, prefix='target_layer_2', nin=100, nout=100, ortho=False)
    params = get_layer('ff')[0](options, params, prefix='target_layer_v', nin=100, nout=1, ortho=False)
    params = get_layer('ff')[0](options, params, prefix='target_layer_m', nin=100, nout=1, ortho=False)
    params = get_layer('ff')[0](options, params, prefix='target_layer_l', nin=100, nout=1, ortho=False)
    return params
from ops import batchnorm
def build_model(tparams, model_options):
    x = T.matrix('x', dtype='float64')
    u = T.matrix('u', dtype='float64')
    h_1 = batchnorm(T.nnet.relu(fflayer(tparams, x, model_options, prefix='layer_1', activ='linear')))
    h_2 = batchnorm(T.nnet.relu(fflayer(tparams, h_1, model_options, prefix='layer_2', activ='linear')))
    h_L = h_2
    v = fflayer(tparams, h_L, model_options, prefix='layer_v', activ='linear')
    m = fflayer(tparams, h_L, model_options, prefix='layer_m', activ='linear')
    l = fflayer(tparams, h_L, model_options, prefix='layer_l', activ='linear')
    p = T.sqr(l)
    a = (-(u - m)**2 * p)
    q = v + a
    return x, u, m, v, q

def build_model_target(tparams, model_options):
    x = T.matrix('target_x', dtype='float64')
    u = T.matrix('target_u', dtype='float64')
    h_1 = batchnorm(T.tanh(fflayer(tparams, x, model_options, prefix='target_layer_1', activ='linear')))
    h_2 = batchnorm(T.tanh(fflayer(tparams, h_1, model_options, prefix='target_layer_2', activ='linear')))
    h_L = h_2
    v = fflayer(tparams, h_L, model_options, prefix='target_layer_v', activ='linear')
    m = fflayer(tparams, h_L, model_options, prefix='target_layer_m', activ='linear')
    l = fflayer(tparams, h_L, model_options, prefix='target_layer_l', activ='linear')
    p = T.sqr(l)
    a = (-(u - m)**2 * p)
    q = v + a
    return x, u, m, v, q

def train():
    model = 'theano'
    if model == 'theano':
        model_options = locals().copy()
        params = init_params(model_options)
        tparams = init_tparams(params)
        x, u, m, v, q = build_model(tparams, model_options)

        reward = T.matrix('rewards')
        gamma_t = T.scalar('reward')
        cost = T.sqr(q - (reward + gamma_t * v))
        inps = [x, u, q, reward, gamma_t , v]
        inps_cost = [q, reward, gamma_t , v]

        _mu = theano.function([x], m)
        mu = lambda x: _mu(x)
        model = theano.function([x,u], q)

        params_target = init_params_target(model_options)
        tparams_target = init_tparams(params_target)

        x, u, m, v, q = build_model_target(tparams_target, model_options)
        _V = theano.function([x], v)
        V = lambda x: _V(x)


        lr = T.scalar(name='lr')
        f_cost = theano.function(inps_cost, cost, on_unused_input='warn')
        grads = T.grad(cost.mean(), wrt=itemlist(tparams))
        optimizer = 'adam'
        f_grad_shared, f_update = getattr(optimizers, optimizer)(lr, tparams, grads, inps, cost)


        for key, key2  in zip(tparams_target.items(), tparams.items()):
            a,b = key
            c,d = key2
            params_target[a] = params[c]


    prestates = []
    actions = []
    rewards = []
    poststates = []
    terminals = []
    total_reward = 0
    lrate = optimizer_lr
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
                loss = 0
                for k in xrange(train_repeat):
                    if len(prestates) > batch_size:
                        indexes = np.random.choice(len(prestates), size=batch_size)
                    else:
                        indexes = range(len(prestates))

                    v = V(np.array(poststates)[indexes])
                    y = np.array(rewards)[indexes] + gamma * np.squeeze(v)
                    q = model(np.array(prestates)[indexes], np.array(actions)[indexes])
                    out = f_cost(q, np.array(rewards)[indexes].reshape((y.shape[0],1)).astype('float32'), gamma, np.squeeze(v).reshape((y.shape[0],1)))
                    loss += out.mean()
                    f_grad_shared(np.array(prestates)[indexes], np.array(actions)[indexes], q, np.array(rewards)[indexes].reshape((y.shape[0],1)).astype('float32'), gamma, np.squeeze(v).reshape((y.shape[0],1)))
                    f_update(lrate)
                    for key, key2  in zip(tparams_target.items(), tparams.items()):
                        a,b = key
                        c,d = key2
                        params_target[a] = params[c] * tau  + params_target[a] * (1. - tau)

            if done:
                break

        episode_reward = episode_reward / float(t + 1)
        print "Episode {} finished after {} timesteps, average reward {}".format(i_episode + 1, t + 1, episode_reward)
        total_reward += episode_reward

    print "Average reward per episode {}".format(total_reward / episodes)

train()
