#!/usr/bin/env python

"""
Train an agent on contra III using PPO implementation by OpenAI baselines
"""

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
#from coord_conv import AddCoords
import os
from baselines import logger
import cv2
cv2.ocl.setUseOpenCL(False)


os.environ['OPENAI_LOG_FORMAT']='stdout,log,csv,tensorboard'


def contra_mixed_cnn(X, use_loc=False,use_loc_r=False,**conv_kwargs):
    """
    CNN from Nature paper with additional features for last two FC layers.
    """
    print("testing contra_mixed_cnn ")
    print(X)
    #a[0,0:1,0:15,-1] #the last one
    #a[:,0:1,0:15,0:12:3] #all of them
    info_features = tf.squeeze(X[:,0:1,0:15,-1],axis=1)
    print("size of info features")
    print(info_features)
    b = X[:, :, :, 0:2]
    c = X[:, :, :, 3:5]
    d = X[:, :, :, 6:8]
    e = X[:, :, :, 9:11]
    unscaled_images = tf.concat([b,c,d,e],axis=3)
    print("size of unscaled images")
    print(unscaled_images)

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
    # print("initial fc net ")
    # print(h3)
    h4 = tf.concat([h3,info_features],axis=1)
    # print("size of h4")
    # print(h4)
    # h5 = activ(fc(h4, 'fc1', nh=512, init_scale=np.sqrt(2)))
    # return activ(fc(h5, 'fc2', nh=512, init_scale=np.sqrt(2)))
    return activ(fc(h4, 'fc1', nh=512, init_scale=np.sqrt(2)))



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


@register("contra_mixed_cnn")
def your_network_define(**conv_kwargs):
    def network_fn(X):
        return contra_mixed_cnn(X,use_loc=False,use_loc_r=False, **conv_kwargs)
        #return contra_mixed_cnn(X, use_loc=True, use_loc_r=True, **conv_kwargs)
    return network_fn

@register("impala_cnn")
def your_network_define(**conv_kwargs):
    def network_fn(X):
        #def impala_cnn(unscaled_images, depths=[16, 32, 32], use_batch_norm=False, dropout=0.0): #DROPOUT ISN'T ADDED
        return impala_cnn(X,depths=[16, 32, 32],use_batch_norm=False)
    return network_fn



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
        ppo2.learn(network=args.network,  # network='contra_net', #network='cnn',
                   env=env_vec,
                   nsteps=1024,  # 1024,
                   nminibatches=128,  # 16,256,512,64,128
                   lam=0.95,
                   gamma=0.997,  # 0.99
                   noptepochs=3,  # 3,
                   log_interval=100,
                   ent_coef=0.003,  # 0.003, 0.001, 0.005 #many actions #0.01
                   lr=lambda _: 5e-4,  # 2e-4,1e-4,5e-5
                   cliprange=0.15,
                   save_interval=100,
                   seed=args.seed,
                   vf_coef=0.5,
                   max_grad_norm=0.5,
                   save_path='ppo_save/{}'.format(time_int),
                   # load_path=args.load_path,
                   total_timesteps=int(2e8))

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

    additional_layers = 2
    if warp_frame:
        #env = WarpFrame(env, width=128, height=112)  # ,grayscale=False
        scalar_layer = True
        print("using new WarpFrameFeature, additional_layers {} and scalar layer is {}".format(additional_layers,scalar_layer))
        env = WarpFrameFeature(env, width=128, height=112,additional_layers=additional_layers,
                               info_keys=['weapon_active','weapon1','weapon2','bomb_count','bomb_active_side','is_scrollable'])  # ,grayscale=False

    if stack:
        #env = FrameStack(env, stack)
        env = FrameStackMedium(env, stack,additional_layers=additional_layers)
        #print("Using FrameStackWithInfo")
        #env = FrameStackWithInfo(env,k=stack,info_keys=['weapon_active','weapon1','weapon2','bomb_count','bomb_active_side','is_scrollable'],info_k=1)

    env.seed(seed + subrank if seed is not None else None)

    return env



class ContraDiscretizer(gym.ActionWrapper):
    """
    Wrap a gym-retro environment and make it use discrete
    actions for the Sonic game.
    """
    def __init__(self, env):
        super(ContraDiscretizer, self).__init__(env)
        # SNES keys
        buttons = ["B", "Y", "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT", "A", "X", "L", "R"]

        #lvl1
        # actions = [['Y'], ['UP'], ['RIGHT'], ['DOWN'], ['LEFT'], ['A'],
        #            ['Y', 'LEFT'], ['Y', 'RIGHT'], ['Y', 'X'], ['Y', 'UP'], ['Y', 'DOWN'],
        #            ['Y', 'B', 'LEFT'], ['Y', 'B', 'RIGHT'], ['Y', 'UP', 'RIGHT'], ['Y', 'DOWN', 'RIGHT'],
        #            ['Y', 'DOWN', 'B'], ['Y', 'UP', 'LEFT'], ['Y', 'DOWN', 'LEFT']
        #           ]

        #lvl4
        actions = [['Y'], ['UP'], ['RIGHT'], ['DOWN'], ['A'],
                   ['LEFT'], ['Y', 'R', 'UP', 'LEFT'], ['Y', 'R', 'UP', 'RIGHT'], ['Y', 'X', 'R', 'UP', 'LEFT'],
                   ['Y', 'X', 'R', 'UP', 'RIGHT'],  # level 4 specific
                   # ['Y', 'DOWN', 'B'], removed for level 4
                   ['Y', 'LEFT'], ['Y', 'RIGHT'], ['Y', 'X'], ['Y', 'UP'], ['Y', 'DOWN'],
                   ['Y', 'B', 'LEFT'], ['Y', 'B', 'RIGHT'], ['Y', 'UP', 'RIGHT'], ['Y', 'DOWN', 'RIGHT'],
                   ['Y', 'UP', 'LEFT'], ['Y', 'DOWN', 'LEFT']
                   ]  #

        #lvl 2
        # actions = [['Y'], ['UP'], ['RIGHT'], ['DOWN'], ['LEFT'], ['A'], ['R'], ['L'],
        #            # 3 button combos optional but might be useful for harder parts
        #            ['Y','R','UP'],['Y','R','RIGHT'],['Y','R','DOWN'],['Y','R','LEFT'],
        #            ['Y', 'L', 'UP'], ['Y', 'L', 'RIGHT'], ['Y', 'L', 'DOWN'], ['Y', 'L', 'LEFT'],
        #            ['Y','X'], ['Y','UP'], ['Y','RIGHT'], ['Y','DOWN'], ['Y','LEFT'], ['Y','R'], ['Y','L'], ['Y','B']]

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

class WarpFrameFeature(gym.ObservationWrapper):
#warps the frame and optional grayscale as WarpFrame does
#also preprocesses info features for an additional CNN layer and appends that to end
    def __init__(self, env, width=84, height=84, grayscale=True,additional_layers=1,info_keys = []):
        """Warp frames to 84x84 as done in the Nature paper and later work."""
        gym.ObservationWrapper.__init__(self, env)
        self.width = width
        self.height = height
        self.grayscale = grayscale
        self.additional_layers = additional_layers
        self.info_keys = info_keys

        if self.grayscale:
            self.observation_space = gym.spaces.Box(low=0, high=255,
                shape=(self.height, self.width, 1+self.additional_layers), dtype=np.uint8)
            #print("obs space is {}".format(self.observation_space))
        else:
            self.observation_space = gym.spaces.Box(low=0, high=255,
                shape=(self.height, self.width, 3+self.additional_layers), dtype=np.uint8)

    def step(self,ac):
        ob, rew, done, info = self.env.step(ac)
        #return self.feature_process(ob,info), rew, done, info
        #return ob, rew, done, info
        #print("in warpframefeature, step was overriden succesfully")
        return self.feature_process(self.observation(ob),info), rew, done, info
        #return self.observation(ob), rew, done, info


    def preprocess_info_features_side(self,info):
        feature_array = np.zeros((15,), dtype=np.uint8)

        # optional other features not used:
        # score (need difference), lives (doing episode end on death), level (same level throughout episodes)

        # which weapon is active, 2 for weapon 2 being active 0 for weapon 1
        if info['weapon_active'] > 0:
            feature_array[0] = 1

        # weapon1, weapon2
        # This is one hot encoding. other options: hard coding in that C/L are better, embeddings
        # 0 is default, 1 S, 2 C, 3 H, 4 F, 5 L
        if info['weapon1'] > 0:
            feature_array[info['weapon1']] = 1
        if info['weapon2'] > 0:
            feature_array[info['weapon2'] + 5] = 1

        # bomb count same on each level
        if info['bomb_count'] > 0:
            feature_array[11] = 1

        # if info['level'] == 2 or info['level'] == 5:
        #     #ret_list.append(['bomb_active_top']/255.0)
        #     feature_array[12] = info['bomb_active_top']/255.0
        #     feature_array = feature_array[0:13] #less features in top down levels
        # else:

        if info['bomb_active_side'] > 0:
            feature_array[12] = 1

        x_scroll_constant = 1
        y_scroll_constant = 3
        if info['is_scrollable'] == x_scroll_constant:
            feature_array[13] = 1
        if info['is_scrollable'] == y_scroll_constant:
            feature_array[14] = 1

        # scroll constant values, only useful if some memory to see difference
        # ret_list.append(info['scroll_value']/255.0)
        # ret_list.append(info['vert_scroll_value'] / 255.0)

        #print("testing feature array for preprocessing features ", feature_array)
        return feature_array

    def feature_process(self,frame,info):
        #code take from lua script:

        #https://github.com/EmulatorArchive/bizhawk/blob/master/output/Lua/SNES/Contra%203.lua
        #enemy feature constants
        enemy_inv = 40
        enemy_touch = 50
        enemy_proj = 55
        enemy_hp = 45

        #player constants
        # not active 0
        # active 1
        # invulnerable 3

        #print("testing warp frame feature preprocessing")
        # turn the info into obs shit
        # concatenate them
        # feed them to wrappers
        # obs image size x,y is 256x224
        # however not sure what that is in contra coordinates

        #Contra Rom coordinates go on x scale from 0 to 255, on y from 0 to 248
        #for each available item in info, turning the coordinates into boxes on an array
        #frame for player, and the 3 enemy types. do not add overplays withing types but add overlaps between types
        frame_player = np.zeros((256, 256),dtype=np.uint8)
        frame_enemy_touch = np.zeros((256, 256),dtype=np.uint8)
        frame_enemy_proj = np.zeros((256, 256),dtype=np.uint8)
        frame_enemy_inv = np.zeros((256, 256),dtype=np.uint8)
        frame_enemy_hp = np.zeros((256, 256),dtype=np.uint8)

        #player
        player_feature = info['active'] * 10 + 5 #0: not active, 1: active, 3: invulnerable
        x1 = self.check_offscreen(info['x1'], info['x1_offset'])
        x2 = self.check_offscreen(info['x2'], info['x2_offset'])
        y1 = self.check_offscreen(info['y1'], info['y1_offset'])
        y2 = self.check_offscreen(info['y2'], info['y2_offset'])
        #y1 is bottom of player, y2 is top. x1 is lhs of player, x2 is rhs of player
        frame_player[y2:y1,x1:x2] = player_feature

        #enemy frames
        for i in range(33):
            temp_string = str(i)
            active = info["active_"+temp_string]
            if active > 0:
                has_hp = info["hp_"+temp_string] > 0
                touch, proj, inv = self.check_enemy_type(active)
                x_proj = self.check_offscreen(info['xproj_' + temp_string], info['xproj_offset_' + temp_string])
                y_proj = self.check_offscreen(info['yproj_' + temp_string], info['yproj_offset_' + temp_string])
                x_rad = info['xrad_'+temp_string]
                y_rad = info['yrad_' + temp_string]

                #x coordinates on screen are the column values for my array
                #y coordinates are the row but in game 0,0 is lower left, in my array this is the upper left index
                row_bottom = y_proj  # 255 - y_proj
                row_top = row_bottom + y_rad  # row_bottom + y_rad #y_rad is negative
                if row_top < 0:
                    row_top = 0
                x_right = x_proj + x_rad

                if touch:
                    frame_enemy_touch[row_top:row_bottom, x_proj:x_right] = enemy_touch
                if proj:
                    frame_enemy_proj[row_top:row_bottom, x_proj:x_right] = enemy_proj
                if inv:
                    frame_enemy_inv[row_top:row_bottom, x_proj:x_right] = enemy_inv
                if has_hp:
                    frame_enemy_hp[row_top:row_bottom, x_proj:x_right] = enemy_hp


        frame_end = frame_player + frame_enemy_inv + frame_enemy_touch + frame_enemy_proj + frame_enemy_hp

        #resize frame end and concat onto other frame observation
        frame_end = cv2.resize(frame_end, (self.width, self.height), interpolation=cv2.INTER_AREA)
        frame_end = np.expand_dims(frame_end,-1)
        #print("frame shape is {} {}".format(np.shape(frame),np.shape(frame_end)))
        frame = np.concatenate((frame, frame_end), axis=2)
        #print("frame shape is {} {}".format(np.shape(frame), np.shape(frame_end)))

        if len(self.info_keys) > 0:
            #adds scalar features to final frame, fucking stupid but baselines doesn't take spaces.Dict at this time
            frame_scalar = np.zeros((self.height, self.width),dtype=np.uint8)
            preprocessed_features = self.preprocess_info_features_side(info)
            frame_scalar[0, 0:len(preprocessed_features)] = preprocessed_features
            frame_scalar = np.expand_dims(frame_scalar, -1)
            #print("shapes of frame and frame_scalar")
            #print(np.shape(frame))
            #print(np.shape(frame_scalar))
            frame = np.concatenate((frame, frame_scalar), axis=2)

        return frame

    def check_offscreen(self,pos,val):
        #val can be 0, 1, or 255
        if val != 0:
            if val == 1:
                pos = 255 + pos
            elif val == 255:
                pos = 0 - (255 - pos)

        pos = np.clip(pos,0,255)

        return pos

    def check_enemy_type(self,a):
        # touch_constant = 8
        # proj_constant = 16
        # inv_constant = 32
        return self.hasbit(a,8), self.hasbit(a,16), self.hasbit(a,32)

    def hasbit(self, x, p):
        return x % (p + p) >= p

    def observation(self, frame):
        if self.grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        if self.grayscale:
            frame = np.expand_dims(frame, -1)
        return frame


class FrameStackMedium(gym.Wrapper):
    def __init__(self, env, k,additional_layers):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[:-1] + (shp[-1] * k,)), dtype=env.observation_space.dtype)
        #print("FrameStackMedium testing obs shape {} {}".format(shp,self.observation_space))
        self.additional_layers=additional_layers #on a reset have to concat empty layers

    def reset(self):
        ob = self.env.reset()
        ob = np.tile(ob,self.additional_layers+1)
        #print("FrameStack resetting ob, ob shape is {}".format(np.shape(ob)))
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))



if __name__ == '__main__':
    main()



