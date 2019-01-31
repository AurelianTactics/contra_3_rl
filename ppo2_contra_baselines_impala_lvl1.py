#!/usr/bin/env python

"""
Train an agent on contra III using PPO implementation by OpenAI baselines
"""


#to do:
#TEST various custom models

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

import tensorflow as tf
import baselines.ppo2.ppo2 as ppo2
from baselines.common.atari_wrappers import *
from baselines.common.retro_wrappers import *
from baselines import bench
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common import set_global_seeds
import time
import retro
import gym
from baselines.common.models import register
from baselines.a2c import utils
from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch
import numpy as np
import argparse
import os
from baselines import logger

os.environ['OPENAI_LOG_FORMAT']='stdout,log,csv,tensorboard'


#To do multiple inputs:
#after it works, test with all inputs
#add normalizer into the model with all inputs


#to do impala:
#DONE turn the Config. into parameters
#NOT DOING DROPOUT figure out dropout assigns
    #coinrun has this: sess.run([model.train_model.dropout_assign_ops]), I need to add that somewhere if I'm doing it
    #if I was going to do it, would have to add something in ppo policy that looks for drpout and calls it if there is any
        #looks like the way the model works is if dropout, makes an array of tf.assign, then those assigns are passsed to policy which calls them
#is my weird impala shit like this going to work?

#step 1:
#basic env, feed out the obs and feed out something from info appended to the obs
    #lives (maybe), x, y, weapon1, weapon2, is scrollable, scroll value (meh), vert_scroll_value (meh)
        #unsure about scroll value since no LSTM
        #thinking treat all with /256
        #not sure about lives since i'm using multi lives envs and due to extra lives/ shit glitching this could be misleading
#step 2:
    #feed both visual obs and side obs into a model, side by side as inputs/array checkable
#step 3:
    #make it work with wrappers
#step 4:
    #custom model that takes in the extra inputs
#step 5:
    #embedding for features






#for info just doing last step, not doing a sequence of steps
class FrameStackWithInfo(gym.Wrapper):
    def __init__(self, env, k, info_keys = []):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        self.obs_info = []
        self.info_keys = info_keys
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[:-1] + (shp[-1] * k,)), dtype=env.observation_space.dtype)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        self.obs_info = []
        for i in self.info_keys:
            self.obs_info.append(0)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        self.obs_info = []
        for i in self.info_keys:
            self.obs_info.append(info[i])
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        if len(info_keys) > 0:
            return LazyFrames(list(self.frames)), self.obs_info
        else:
            return LazyFrames(list(self.frames))



# #various NN architectures
# def contra_nature_cnn(unscaled_images,batch_norm=True,snnn=True,use_loc=False,use_loc_r=False, **conv_kwargs):
#     """
#     CNN from Nature paper.
#     """
#
#     scaled_images = tf.cast(unscaled_images, tf.float32) / 255.
#     if use_loc:
#         addcoords = AddCoords(x_dim=int(np.shape(scaled_images)[1]), y_dim=int(np.shape(scaled_images)[1]), with_r=use_loc_r)
#         scaled_images = addcoords(scaled_images)
#         print("CNN: Added coordinate filters tensor size is now {}".format(np.shape(scaled_images)))
#
#     activ = tf.nn.relu
#     h = activ(conv(scaled_images, 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2),
#                    **conv_kwargs))
#     if batch_norm:
#         h = tf.layers.batch_normalization(h)
#     h2 = activ(conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2), **conv_kwargs))
#     if batch_norm:
#         h2 = tf.layers.batch_normalization(h2)
#     h3 = activ(conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2), **conv_kwargs))
#     if batch_norm:
#         h3 = tf.layers.batch_normalization(h3)
#     h3 = conv_to_fc(h3)
#     if snnn:
#         h4 = tf.layers.dense(h3,512,activation=tf.nn.selu,kernel_initializer=tf.variance_scaling_initializer(scale=1.0,mode='fan_in'))
#         h5 = tf.contrib.nn.alpha_dropout(h4,keep_prob=0.8)
#         return h5
#     else:
#         return activ(fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2)))


def contra_mixed_cnn(X, use_loc=False,use_loc_r=False,**conv_kwargs):
    """
    CNN from Nature paper with additional features for last two FC layers.
    """
    print("testing contra_mixed_cnn ")
    print(X)
    print(np.shape(X))
    unscaled_images = X[0]
    info_features = X[1]


    scaled_images = tf.cast(unscaled_images, tf.float32) / 255.
    if use_loc:
        addcoords = AddCoords(x_dim=int(np.shape(scaled_images)[1]), y_dim=int(np.shape(scaled_images)[1]), with_r=use_loc_r)
        scaled_images = addcoords(scaled_images)
        print("CNN: Added coordinate filters tensor size is now {}".format(np.shape(scaled_images)))

    activ = tf.nn.relu
    h = activ(conv(scaled_images, 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2),
                   **conv_kwargs))
    h2 = activ(conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2), **conv_kwargs))
    h3 = activ(conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2), **conv_kwargs))
    h3 = conv_to_fc(h3)
    h4 = tf.concat([h3,info_features],axis=1)
    h5 = activ(fc(h4, 'fc1', nh=256, init_scale=np.sqrt(2)))
    return activ(fc(h5, 'fc2', nh=256, init_scale=np.sqrt(2)))



def impala_cnn(unscaled_images, depths=[16, 32, 32], use_batch_norm=False, dropout=0.0):
    """
    Model used in the paper "IMPALA: Scalable Distributed Deep-RL with
    Importance Weighted Actor-Learner Architectures" https://arxiv.org/abs/1802.01561

    code from coinrun: https://github.com/openai/coinrun/blob/master/coinrun/policies.py
    """
    #use_batch_norm = Config.USE_BATCH_NORM == 1

    dropout_layer_num = [0]
    dropout_assign_ops = []

    def dropout_layer(out):
        #if Config.DROPOUT > 0:
        if dropout > 0:
            print("DROPOUT HAS NOT BEEN ADDED")
            out_shape = out.get_shape().as_list()
            num_features = np.prod(out_shape[1:])

            var_name = 'mask_' + str(dropout_layer_num[0])
            batch_seed_shape = out_shape[1:]
            batch_seed = tf.get_variable(var_name, shape=batch_seed_shape,
                                         initializer=tf.random_uniform_initializer(minval=0, maxval=1), trainable=False)
            batch_seed_assign = tf.assign(batch_seed, tf.random_uniform(batch_seed_shape, minval=0, maxval=1))
            dropout_assign_ops.append(batch_seed_assign)

            curr_mask = tf.sign(tf.nn.relu(batch_seed[None, ...] - dropout))

            curr_mask = curr_mask * (1.0 / (1.0 - dropout))

            out = out * curr_mask

        dropout_layer_num[0] += 1

        return out

    def conv_layer(out, depth):
        out = tf.layers.conv2d(out, depth, 3, padding='same')
        out = dropout_layer(out)

        if use_batch_norm:
            out = tf.contrib.layers.batch_norm(out, center=True, scale=True, is_training=True)

        return out

    def residual_block(inputs):
        depth = inputs.get_shape()[-1].value

        out = tf.nn.relu(inputs)

        out = conv_layer(out, depth)
        out = tf.nn.relu(out)
        out = conv_layer(out, depth)
        return out + inputs

    def conv_sequence(inputs, depth):
        out = conv_layer(inputs, depth)
        out = tf.layers.max_pooling2d(out, pool_size=3, strides=2, padding='same')
        out = residual_block(out)
        out = residual_block(out)
        return out

    #out = images #coinrun scales images outside this function call,
    scaled_images = tf.cast(unscaled_images, tf.float32) / 255.
    out = scaled_images
    for depth in depths:
        out = conv_sequence(out, depth)

    out = tf.layers.flatten(out)
    out = tf.nn.relu(out)
    out = tf.layers.dense(out, 256, activation=tf.nn.relu)

    #return out, dropout_assign_ops
    return out #not doing dropout

@register("impala_cnn")
def your_network_define(**conv_kwargs):
    def network_fn(X):
        #def impala_cnn(unscaled_images, depths=[16, 32, 32], use_batch_norm=False, dropout=0.0): #DROPOUT ISN'T ADDED
        return impala_cnn(X,depths=[16, 32, 32],use_batch_norm=False)
    return network_fn


# def contra_mixed_norm_cnn(X, use_loc=False,use_loc_r=False,**conv_kwargs):
#     """
#     CNN from Nature paper with batch norm
#     FC net with SNN
#     """
#     unscaled_images = X[0]
#     info_features = X[1]
#
#     scaled_images = tf.cast(unscaled_images, tf.float32) / 255.
#     if use_loc:
#         addcoords = AddCoords(x_dim=int(np.shape(scaled_images)[1]), y_dim=int(np.shape(scaled_images)[1]), with_r=use_loc_r)
#         scaled_images = addcoords(scaled_images)
#         print("CNN: Added coordinate filters tensor size is now {}".format(np.shape(scaled_images)))
#
#     activ = tf.nn.relu
#     h = activ(conv(scaled_images, 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2),
#                    **conv_kwargs))
#     h = tf.layers.batch_normalization(h)
#     h2 = activ(conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2), **conv_kwargs))
#     h2 = tf.layers.batch_normalization(h2)
#     h3 = activ(conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2), **conv_kwargs))
#     h3 = tf.layers.batch_normalization(h3)
#     h3 = conv_to_fc(h3)
#     h4 = tf.concat([h3,info_features],axis=1)
#     h5 = tf.layers.dense(h4,512,activation=tf.nn.selu,kernel_initializer=tf.variance_scaling_initializer(scale=1.0,mode='fan_in'))
#     h6 = tf.contrib.nn.alpha_dropout(h5,keep_prob=0.8)
#     h7 = tf.layers.dense(h6,512,activation=tf.nn.selu,kernel_initializer=tf.variance_scaling_initializer(scale=1.0,mode='fan_in'))
#     h8 = tf.contrib.nn.alpha_dropout(h7,keep_prob=0.8)
#     return h8
#
#
# def contra_mixed_norm_embed_cnn(X, use_loc=False,use_loc_r=False,**conv_kwargs):
#     """
#     CNN from Nature paper with batch norm
#     FC net with SNN
#     embedding for the weapons
#     """
#     unscaled_images = X[0]
#     info_inputs = X[1]
#     embed_inputs = X[2]
#
#     scaled_images = tf.cast(unscaled_images, tf.float32) / 255.
#     if use_loc:
#         addcoords = AddCoords(x_dim=int(np.shape(scaled_images)[1]), y_dim=int(np.shape(scaled_images)[1]), with_r=use_loc_r)
#         scaled_images = addcoords(scaled_images)
#         print("CNN: Added coordinate filters tensor size is now {}".format(np.shape(scaled_images)))
#
#     activ = tf.nn.relu
#     h = activ(conv(scaled_images, 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2),
#                    **conv_kwargs))
#     h = tf.layers.batch_normalization(h)
#     h2 = activ(conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2), **conv_kwargs))
#     h2 = tf.layers.batch_normalization(h2)
#     h3 = activ(conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2), **conv_kwargs))
#     h3 = tf.layers.batch_normalization(h3)
#     h3 = conv_to_fc(h3)
#
#     #normal,s,l,h,c,f
#     #embed_inputs = tf.placeholder(tf.int32)
#     embeddings = tf.Variable(tf.random_normal([None,3],-1.0,1.0))
#     embed = tf.nn.embedding_lookup(embeddings,embed_inputs)
#
#     h4 = tf.concat([h3,info_inputs,embed],axis=1)
#     h5 = tf.layers.dense(h4,512,activation=tf.nn.selu,kernel_initializer=tf.variance_scaling_initializer(scale=1.0,mode='fan_in'))
#     h6 = tf.contrib.nn.alpha_dropout(h5,keep_prob=0.8)
#     h7 = tf.layers.dense(h6,512,activation=tf.nn.selu,kernel_initializer=tf.variance_scaling_initializer(scale=1.0,mode='fan_in'))
#     h8 = tf.contrib.nn.alpha_dropout(h7,keep_prob=0.8)
#     return h8

@register("contra_mixed_cnn")
def your_network_define(**conv_kwargs):
    def network_fn(X):
        return contra_mixed_cnn(X,use_loc=False,use_loc_r=False, **conv_kwargs)
        #return contra_mixed_cnn(X, use_loc=True, use_loc_r=True, **conv_kwargs)
    return network_fn



# @register("contra_nature_cnn_batch_snnn")
# def your_network_define(**conv_kwargs):
#     def network_fn(X):
#         return contra_nature_cnn(X, batch_norm=True,snnn=True,use_loc=False,use_loc_r=False, **conv_kwargs)
#         #return contra_nature_cnn(X, batch_norm=True, snnn=True, use_loc=True, use_loc_r=True, **conv_kwargs)
#     return network_fn
#
# @register("contra_nature_cnn_batch")
# def your_network_define(**conv_kwargs):
#     def network_fn(X):
#         return contra_nature_cnn(X, batch_norm=True, snnn=False, use_loc=False, use_loc_r=False, **conv_kwargs)
#         #return contra_nature_cnn(X, batch_norm=True,snnn=False,use_loc=True,use_loc_r=True, **conv_kwargs)
#     return network_fn
#
# @register("contra_nature_cnn_snnn")
# def your_network_define(**conv_kwargs):
#     def network_fn(X):
#         return contra_nature_cnn(X, batch_norm=False,snnn=True,use_loc=False,use_loc_r=False, **conv_kwargs)
#         #return contra_nature_cnn(X, batch_norm=False, snnn=True, use_loc=True, use_loc_r=True, **conv_kwargs)
#     return network_fn
#
# @register("contra_mixed_norm_cnn")
# def your_network_define(**conv_kwargs):
#     def network_fn(X):
#         return contra_mixed_norm_cnn(X, use_loc=False, use_loc_r=False, **conv_kwargs)
#         #return contra_mixed_norm_cnn(X,use_loc=True,use_loc_r=True, **conv_kwargs)
#     return network_fn
#
# @register("contra_mixed_norm_cnn")
# def your_network_define(**conv_kwargs):
#     def network_fn(X):
#         return contra_mixed_norm_embed_cnn(X, use_loc=False, use_loc_r=False, **conv_kwargs)
#         #return contra_mixed_norm_embed_cnn(X,use_loc=True,use_loc_r=True, **conv_kwargs)
#     return network_fn





def main():
    """Run PPO until the environment throws an exception."""
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True # pylint: disable=E1101

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--num_env',default=1,type=int)
    parser.add_argument('--seed', default=None,type=int)
    parser.add_argument('--game',default='ContraIII-Snes')
    parser.add_argument('--state', default='level1.1player.easy.100lives') #state=retro.State.DEFAULT
    parser.add_argument('--scenario', default='scenario')
    parser.add_argument('--discrete_actions', default=0, type=int)
    parser.add_argument('--bk2dir', default='videos')
    parser.add_argument('--monitordir', default='logs')
    parser.add_argument('--sonic_discretizer', default=1, type=int)
    parser.add_argument('--clip_rewards', default=0, type=int)
    parser.add_argument('--stack', default=4,type=int)
    parser.add_argument('--time_limit', default=8000,type=int)
    parser.add_argument('--scale_reward', default=0.01,type=float)
    parser.add_argument('--warp_frame', default=1,type=int)
    parser.add_argument('--stochastic_frame_skip', default=4,type=int)
    parser.add_argument('--skip_prob', default=0.0,type=float)
    parser.add_argument('--network', default='cnn')
    parser.add_argument('--scenario_number', default=1,type=int)
    parser.add_argument('--load_path',default=None)
    args = parser.parse_args()
    time_int = int(time.time())

    env_vec = make_vec_env(args,time_int)
    logger.configure(dir='./log/{}'.format(time_int), format_strs=['stdout', 'log', 'csv', 'tensorboard'])
    with tf.Session(config=config):
        ppo2.learn(network='impala_cnn',#args.network, #network='contra_net', #network='cnn',
                   env=env_vec,
                   nsteps=1024,#1024,
                   nminibatches=128,#16,256,512,64,128
                   lam=0.95,
                   gamma=0.997, #0.99
                   noptepochs=3, #3,
                   log_interval=100,
                   ent_coef=0.003,#0.003,#0.003, 0.001, 0.005 #many actions #0.01
                   lr=lambda _: 5e-5,#2e-4,1e-4,5e-5
                   cliprange=0.1,
                   save_interval=100,
                   seed=args.seed,
                   vf_coef=0.5,
                   max_grad_norm=0.5,
                   save_path='ppo_save/{}'.format(time_int),
                   #load_path=args.load_path,
                   total_timesteps=int(2e10))

                   #total_timesteps=int(10000))

def make_vec_env(args, time_int, start_index=0):
    """
    Create a wrapped, monitored SubprocVecEnv
    """
    mpi_rank = MPI.COMM_WORLD.Get_rank() if MPI else 0
    seed = args.seed + 10000 * mpi_rank if args.seed is not None else None

    def make_thunk(rank):
        return lambda: make_env(time_int=time_int,seed=args.seed,subrank=rank, game=args.game,state=args.state,scenario=args.scenario,
                                discrete_actions=bool(args.discrete_actions),bk2dir=args.bk2dir,monitordir=args.monitordir,
                                sonic_discretizer=bool(args.sonic_discretizer),clip_rewards=bool(args.clip_rewards), stack=args.stack,
                                time_limit=args.time_limit,scale_reward=args.scale_reward,warp_frame=bool(args.warp_frame),
                                stochastic_frame_skip=args.stochastic_frame_skip,skip_prob=args.skip_prob,scenario_number=args.scenario_number)

    set_global_seeds(seed)
    if args.num_env > 1:
        return SubprocVecEnv([make_thunk(i + start_index) for i in range(args.num_env)])
    else:
        return DummyVecEnv([make_thunk(start_index)])

def make_env(time_int, seed=None, subrank=0, game='ContraIII-Snes', state='level1.1player.easy.100lives', scenario='scenario', discrete_actions=False,
             bk2dir='videos',monitordir='logs', sonic_discretizer=True, clip_rewards=False,
             stack=4,time_limit=8000,scale_reward=0.01,warp_frame=True,stochastic_frame_skip=4,skip_prob=0.0,scenario_number=1):

    use_restricted_actions = retro.Actions.FILTERED  # retro.ACTIONS_FILTERED
    if discrete_actions:
        use_restricted_actions = retro.Actions.DISCRETE  # retro.ACTIONS_DISCRETE

    if scenario_number <= 1:
        print("scenario is {}".format(scenario))
        env = retro.make(game, state, scenario=scenario, use_restricted_actions=use_restricted_actions)
    else:
        scenario = scenario + "_{}".format(subrank%scenario_number)
        print("scenario is {}".format(scenario))
        env = retro.make(game, state, scenario=scenario, use_restricted_actions=use_restricted_actions)

    if bk2dir:
        #env.auto_record(bk2dir)
        bk2dir = bk2dir + "/{}_{}".format(time_int,subrank)
        print(bk2dir)

        if not os.path.exists(bk2dir):
            os.makedirs(bk2dir)

        env = MovieRecord(env,bk2dir,k=10)#k is how often to save and record the video

    if monitordir:
        #print("test path {}".format(os.path.join(monitordir, 'log_{}_{}'.format(time_int,subrank))))
        env = bench.Monitor(env,os.path.join(monitordir, 'log_{}_{}'.format(time_int,subrank)),allow_early_resets=True)

    if stochastic_frame_skip:
        #print("skip prob is {}".format(skip_prob))
        env = StochasticFrameSkip(env,stochastic_frame_skip,skip_prob)

    if time_limit:
        env = TimeLimit(env,max_episode_steps=time_limit)

    if sonic_discretizer:
        #env = SonicDiscretizer(env)
        env = ContraDiscretizer(env)
    if scale_reward:
        env = RewardScaler(env,scale=scale_reward)
    if clip_rewards:
        env = ClipRewardEnv(env)
    if warp_frame:
        env = WarpFrame(env,width=112,height=128) #,grayscale=False

    if stack:
        env = FrameStack(env, stack)

    #env = FrameStackWithInfo(env,4,["x1","y1"])

    env.seed(seed + subrank if seed is not None else None)

    return env



# def make_vec_env(env_id, env_type, num_env, seed, wrapper_kwargs=None, start_index=0, reward_scale=1.0, gamestate=None):
#     """
#     Create a wrapped, monitored SubprocVecEnv for Atari and MuJoCo.
#     """
#     if wrapper_kwargs is None: wrapper_kwargs = {}
#     mpi_rank = MPI.COMM_WORLD.Get_rank() if MPI else 0
#     seed = seed + 10000 * mpi_rank if seed is not None else None
#     def make_thunk(rank):
#         return lambda: make_env(
#             env_id=env_id,
#             env_type=env_type,
#             subrank = rank,
#             seed=seed,
#             reward_scale=reward_scale,
#             gamestate=gamestate,
#             wrapper_kwargs=wrapper_kwargs
#         )
#
#     set_global_seeds(seed)
#     if num_env > 1:
#         return SubprocVecEnv([make_thunk(i + start_index) for i in range(num_env)])
#     else:
#         return DummyVecEnv([make_thunk(start_index)])
#
#
# def make_env(env_id, env_type, subrank=0, seed=None, reward_scale=1.0, gamestate=None, wrapper_kwargs={}):
#     mpi_rank = MPI.COMM_WORLD.Get_rank() if MPI else 0
#     if env_type == 'atari':
#         env = make_atari(env_id)
#     elif env_type == 'retro':
#         import retro
#         gamestate = gamestate or retro.State.DEFAULT
#         env = retro_wrappers.make_retro(game=env_id, max_episode_steps=10000, use_restricted_actions=retro.Actions.DISCRETE, state=gamestate)
#     else:
#         env = gym.make(env_id)
#
#     env.seed(seed + subrank if seed is not None else None)
#     env = Monitor(env,
#                   logger.get_dir() and os.path.join(logger.get_dir(), str(mpi_rank) + '.' + str(subrank)),
#                   allow_early_resets=True)
#
#     if env_type == 'atari':
#         env = wrap_deepmind(env, **wrapper_kwargs)
#     elif env_type == 'retro':
#         env = retro_wrappers.wrap_deepmind_retro(env, **wrapper_kwargs)
#
#     if reward_scale != 1:
#         env = retro_wrappers.RewardScaler(env, reward_scale)
#
#     return env

#def contra_discretizer():
class ContraDiscretizer(gym.ActionWrapper):
    """
    Wrap a gym-retro environment and make it use discrete
    actions for the Sonic game.
    """
    def __init__(self, env):
        super(ContraDiscretizer, self).__init__(env)
        # SNES keys
        buttons = ["B", "Y", "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT", "A", "X", "L", "R"]
        #lvl 2
        actions = [['Y'], ['UP'], ['RIGHT'], ['DOWN'], ['LEFT'], ['A'], ['R'], ['L'],
                   # 3 button combos optional but might be useful for harder parts
                   ['Y','R','UP'],['Y','R','RIGHT'],['Y','R','DOWN'],['Y','R','LEFT'],
                   ['Y', 'L', 'UP'], ['Y', 'L', 'RIGHT'], ['Y', 'L', 'DOWN'], ['Y', 'L', 'LEFT'],
                   ['Y','X'], ['Y','UP'], ['Y','RIGHT'], ['Y','DOWN'], ['Y','LEFT'], ['Y','R'], ['Y','L'], ['Y','B']]

        #level 3
        # actions = [['Y'], ['UP'], ['RIGHT'], ['DOWN'], ['LEFT'], ['A'],
        #             #['Y', 'DOWN', 'B'],
        #            ['Y', 'LEFT'], ['Y', 'RIGHT'], ['Y', 'X'], ['Y', 'UP'], ['Y', 'DOWN'],
        #            ['Y', 'B', 'LEFT'], ['Y', 'B', 'RIGHT'], ['Y', 'UP', 'RIGHT'], ['Y', 'DOWN', 'RIGHT'],
        #            ['Y', 'UP', 'LEFT'], ['Y', 'DOWN', 'LEFT']
        #            ] #
                    #['Y', 'R', 'UP', 'LEFT'], ['Y', 'R', 'UP', 'RIGHT'], ['Y', 'R', 'DOWN', 'LEFT'], ['Y', 'R', 'DOWN', 'RIGHT']
                    #['Y', 'L', 'R', 'RIGHT'], ['Y', 'L', 'R', 'LEFT'],  ['Y', 'UP', 'LEFT'], ['Y', 'DOWN', 'LEFT'], ['A'],
                     # ['Y', 'R', 'UP', 'LEFT'], ['Y', 'R', 'UP', 'RIGHT'], ['Y', 'X', 'R', 'UP', 'LEFT'], ['Y', 'X', 'R', 'UP', 'RIGHT'], #level 4 specific

        #23 actions, more than I would like but certain ones needed at specific moments
        self._actions = []
        for action in actions:
            arr = np.array([False] * 12)
            if action == ['NOOP']:
                self._actions.append(arr)
                continue
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a): # pylint: disable=W0221
        return self._actions[a].copy()


if __name__ == '__main__':
    main()
