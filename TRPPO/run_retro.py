#!/usr/bin/env python3
import ast
import os
import sys
import os.path as osp
from baselines.common import tools

from baselines import logger
#from baselines.common.cmd_util import make_atari_env, arg_parser
#from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.TRPPO import ppo2
from baselines.TRPPO.policies import CnnPolicy, MlpPolicy
import multiprocessing
import tensorflow as tf

#retro imports
# import tensorflow as tf
# import baselines.ppo2.ppo2 as ppo2
from baselines.common.atari_wrappers import *
from baselines.common.retro_wrappers import *
from baselines import bench
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common import set_global_seeds
import time
import retro
import gym
# from baselines.common.models import register
# from baselines.a2c import utils
# from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch
import numpy as np
#import argparse
from baselines.common.cmd_util import arg_parser
# import os
# from baselines import logger

try:
    from mpi4py import MPI
except ImportError:
    MPI = None


#python -m baselines.TRPPO.run_atari  --clipped_type=kl2clip --delta_kl=0.001 --use_tabular=True --num-timesteps= --clip-range=  --seed=

#hard code the env in
#def train(env_id, clipped_type, num_timesteps, seed, args, policy):
def train(clipped_type, num_timesteps, seed, args, policy):
    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    config.gpu_options.allow_growth = True
    tf.Session(config=config).__enter__()

	#TRPPO env creation
    #env = VecFrameStack(make_atari_env(env_id, num_env=8, seed=seed), 4)  # TODO: 注意是8个进程
	#I run in through my make_env functions
    time_int = int(time.time())
    env = make_vec_env(seed, time_int)


    policy = {'cnn': CnnPolicy, 'mlp': MlpPolicy}[policy]
    ent_coef = 0.01 if args.clipped_type == 'origin' else 0
    ppo2.learn(policy=policy, env=env, nsteps=1024, nminibatches=128,
               lam=0.95, gamma=0.997, noptepochs=4, log_interval=25,
               ent_coef=ent_coef,
               lr=lambda _: 5e-5,#lambda f: f * 2.5e-4,
               total_timesteps=int(2e8 * 1.1),
               clipped_type=clipped_type, args=args,
               save_interval=100,
               )

               #ppo2.learn(network=args.network, #network='contra_net', #network='cnn',
                #   env=env_vec,
                 #  nsteps=1024,#1024,
                  # nminibatches=128,#16,256,512,64,128
                  # lam=0.95,
                   #gamma=0.997, #0.99
                   #noptepochs=4, #3,
                   #log_interval=25,
                   #ent_coef=0.003,#0.003, 0.001, 0.005 #many actions #0.01
                   #lr=lambda _: 5e-5,#2e-4,1e-4,5e-5
                   #cliprange=0.1,
                   #save_interval=100,
                   #seed=args.seed,
                   #vf_coef=0.5,
                   #max_grad_norm=0.5,
                   #save_path='ppo_save/{}'.format(time_int),
                   #load_path=args.load_path,
                   #total_timesteps=int(2e15))


def atari_arg_parser():
    """
    Create an argparse.ArgumentParser for run_atari.py.
    """
    parser = arg_parser()
    parser.add_argument('--env', help='environment ID', default='BreakoutNoFrameskip-v4')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(10e6))
    parser.add_argument('--clipped_type', default='kl2clip', type=str)
    parser.add_argument('--use_tabular', default=False, type=ast.literal_eval)
    parser.add_argument('--cliprange', default=0.1, type=ast.literal_eval)
    parser.add_argument('--delta_kl', default=0.001, type=float)
    root_dir_default = '/tmp/baselines'
    if not os.path.exists(root_dir_default):
        tools.mkdir(root_dir_default)

    parser.add_argument('--root_dir', default=root_dir_default, type=str)
    parser.add_argument('--sub_dir', default=None, type=str)
    parser.add_argument('--force_write', default=1, type=int)
    return parser

from baselines.common import tools
import json

def main():
    parser = atari_arg_parser()
    parser.add_argument('--policy', help='Policy architecture', choices=['cnn', 'lstm', 'lnlstm', 'mlp'], default='cnn')
    args = parser.parse_args()
    if args.clipped_type == 'kl2clip':
        name_tmp = ''
        if args.cliprange and 'NoFrameskip-v4' not in args.env:
            args.kl2clip_clipcontroltype = 'base-clip'
        else:
            args.kl2clip_clipcontroltype = 'none-clip'
    else:
        name_tmp = ''
        assert args.cliprange, "PPO has to receive a cliprange parameter, the default one is 0.2"
    # --- Generate sub_dir of log dir and model dir
    split = ','
    if args.sub_dir is None:
        keys_except = ['env', 'play', 'root_dir', 'sub_dir', 'force_write', 'lr', 'kl2clip_clipcontroltype']
        # TODO: tmp for kl2clip_sharelogsigma
        keys_fmt = {'num_timesteps': '.0e'}
        args_dict = vars(args)
        sub_dir = args.env
        if not args.clipped_type in ['kl2clip']:
            keys_except += ['delta_kl']
        if not args.clipped_type in ['origin', 'kl2clip', 'a2c']:
            keys_except += ['cliprange']

        # --- add keys common
        for key in args_dict.keys():
            if key not in keys_except and key not in keys_fmt.keys():
                sub_dir += f'{split} {key}={args_dict[key]}'
        # --- add keys which has specific format
        for key in keys_fmt.keys():
            sub_dir += f'{split} {key}={args_dict[key]:{keys_fmt[key]}}'
        sub_dir += ('' if name_tmp == '' else f'{split} {name_tmp}')
        args.sub_dir = sub_dir

    tools.mkdir(f'{args.root_dir}/log')
    tools.mkdir(f'{args.root_dir}/model')
    args.log_dir = f'{args.root_dir}/log/{args.sub_dir}'
    args.model_dir = f'{args.root_dir}/model/{args.sub_dir}'
    force_write = args.force_write
    # Move Dirs
    if osp.exists(args.log_dir) or osp.exists(args.model_dir):  # modify name if exist
        print(
            f"Exsits directory! \n log_dir:'{args.log_dir}' \n model_dir:'{args.model_dir}'\nMove to discard(y or n)?",
            end='')
        if force_write > 0:
            cmd = 'y'
        elif force_write < 0:
            exit()
        else:
            cmd = input()
        if cmd == 'y':
            log_dir_new = args.log_dir.replace('/log/', '/log_discard/')
            model_dir_new = args.model_dir.replace('/model/', '/model_discard/')
            import itertools
            if osp.exists(log_dir_new) or osp.exists(model_dir_new):
                for i in itertools.count():
                    suffix = f' {split} {i}'
                    log_dir_new = f'{args.root_dir}/log_discard/{args.sub_dir}{suffix}'
                    model_dir_new = f'{args.root_dir}/model_discard/{args.sub_dir}{suffix}'
                    if not osp.exists(log_dir_new) and not osp.exists(model_dir_new):
                        break
            print(f"Move log_dir '{args.log_dir}' \n   to '{log_dir_new}'. \n"
                  f"Move model_dir '{args.model_dir}' \n to '{model_dir_new}'"
                  f"\nConfirm move(y or n)?", end='')
            if force_write > 0:
                cmd = 'y'
            elif force_write < 0:
                exit()
            else:
                cmd = input()
            if cmd == 'y':
                import shutil
                if osp.exists(args.log_dir):
                    shutil.move(args.log_dir, log_dir_new)
                if osp.exists(args.model_dir):
                    shutil.move(args.model_dir, model_dir_new)
            else:
                print("Please Rename 'name_tmp'")
                exit()
        else:
            print("Please Rename 'name_tmp'")
            exit()

    os.mkdir(args.log_dir)
    os.mkdir(args.model_dir)
    # exit()

    os.mkdir(osp.join(args.model_dir, 'cliprange_max'))
    os.mkdir(osp.join(args.model_dir, 'cliprange_min'))
    os.mkdir(osp.join(args.model_dir, 'actions'))
    # os.mkdir(osp.join(args.model_dir, 'mu0_logsigma0'))
    os.mkdir(osp.join(args.model_dir, 'kls, ratios'))
    os.mkdir(osp.join(args.model_dir, 'advs'))

    args_str = vars(args)
    with open(f'{args.log_dir}/args.json', 'w') as f:
        json.dump(args_str, f, indent=4, separators=(',', ':'))

    logger.configure(args.log_dir)
    train(clipped_type=args.clipped_type, num_timesteps=args.num_timesteps,
		  seed=args.seed, args=args, policy=args.policy)
    # train(env_id=args.env, clipped_type=args.clipped_type, num_timesteps=args.num_timesteps,
    #       seed=args.seed, args=args, policy=args.policy)



#retro functions
def make_vec_env(seed, time_int, start_index=0):
    """
    Create a wrapped, monitored SubprocVecEnv
    """
    mpi_rank = MPI.COMM_WORLD.Get_rank() if MPI else 0
    seed = seed + 10000 * mpi_rank if seed is not None else None

    def make_thunk(rank):
        return lambda: make_env(time_int=time_int, seed=seed, subrank=rank, game='ContraIII-Snes', state='level3.1player.easy.100lives',
								scenario='scenario20',
								discrete_actions=bool(0), bk2dir='videos',
								monitordir='logs',
								sonic_discretizer=bool(1), clip_rewards=bool(0),
								stack=4,
								time_limit=8000, scale_reward=0.01,
								warp_frame=bool(1),
								stochastic_frame_skip=4, skip_prob=0.25,
								scenario_number=1)
                                # return lambda: make_env(time_int=time_int,seed=args.seed,subrank=rank, game=args.game,state=args.state,scenario=args.scenario,
                        		#         #                         discrete_actions=bool(args.discrete_actions),bk2dir=args.bk2dir,monitordir=args.monitordir,
                        		#         #                         sonic_discretizer=bool(args.sonic_discretizer),clip_rewards=bool(args.clip_rewards), stack=args.stack,
                        		#         #                         time_limit=args.time_limit,scale_reward=args.scale_reward,warp_frame=bool(args.warp_frame),
                        		#         #                         stochastic_frame_skip=args.stochastic_frame_skip,skip_prob=args.skip_prob,scenario_number=args.scenario_number)

                        		# parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
                        		# parser.add_argument('--num_env', default=1, type=int)
                        		# parser.add_argument('--seed', default=None, type=int)
                        		# parser.add_argument('--game', default='ContraIII-Snes')
                        		# parser.add_argument('--state', default='level1.1player.easy.100lives')  # state=retro.State.DEFAULT
                        		# parser.add_argument('--scenario', default='scenario')
                        		# parser.add_argument('--discrete_actions', default=0, type=int)
                        		# parser.add_argument('--bk2dir', default='videos')
                        		# parser.add_argument('--monitordir', default='logs')
                        		# parser.add_argument('--sonic_discretizer', default=1, type=int)
                        		# parser.add_argument('--clip_rewards', default=0, type=int)
                        		# parser.add_argument('--stack', default=4, type=int)
                        		# parser.add_argument('--time_limit', default=8000, type=int)
                        		# parser.add_argument('--scale_reward', default=0.01, type=float)
                        		# parser.add_argument('--warp_frame', default=1, type=int)
                        		# parser.add_argument('--stochastic_frame_skip', default=4, type=int)
                        		# parser.add_argument('--skip_prob', default=0.0, type=float)
                        		# parser.add_argument('--network', default='cnn')
                        		# parser.add_argument('--scenario_number', default=1, type=int)
                        		# parser.add_argument('--load_path', default=None)

                                		#coding arguments in, only doing one run
                                		#python3 ppo2_contra_baselines_agent.py --game ContraIII-Snes
                                        #--state level3.1player.easy.100lives --num_env 42 --seed 3933 --scenario_number 1
                                        #--stochastic_frame_skip 4 --scale_reward 0.01 --skip_prob 0.25 --scenario scenario20


    set_global_seeds(seed)
    num_envs = 42
    if num_envs > 1:
        return SubprocVecEnv([make_thunk(i + start_index) for i in range(num_envs)])
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
        #lvl 2
        # actions = [['Y'], ['UP'], ['RIGHT'], ['DOWN'], ['LEFT'], ['A'], ['R'], ['L'],
        #            # 3 button combos optional but might be useful for harder parts
        #            ['Y','R','UP'],['Y','R','RIGHT'],['Y','R','DOWN'],['Y','R','LEFT'],
        #            ['Y', 'L', 'UP'], ['Y', 'L', 'RIGHT'], ['Y', 'L', 'DOWN'], ['Y', 'L', 'LEFT'],
        #            ['Y','X'], ['Y','UP'], ['Y','RIGHT'], ['Y','DOWN'], ['Y','LEFT'], ['Y','R'], ['Y','L'], ['Y','B']]

        #level 3
        actions = [['Y'], ['UP'], ['RIGHT'], ['DOWN'], ['LEFT'], ['A'], ['X'],
                    #['Y', 'DOWN', 'B'],
                   ['Y', 'LEFT'], ['Y', 'RIGHT'], ['Y', 'X'], ['Y', 'UP'], ['Y', 'DOWN'],
                   ['Y', 'B', 'LEFT'], ['Y', 'B', 'RIGHT'], ['Y', 'UP', 'RIGHT'], ['Y', 'DOWN', 'RIGHT'],
                   ['Y', 'UP', 'LEFT'], ['Y', 'DOWN', 'LEFT']
                   ] #
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
