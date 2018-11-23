import sys
import retro
import csv
import time
import pickle
#from baselines.common.atari_wrappers import WarpFrame, FrameStack
import numpy as np
#from sonic_util_test import AllowBacktracking #, make_env
from collections import OrderedDict

#1 for viewing videos, 0 for creating waypoints
debug = 1 #int(sys.argv[1])
video_arg = int(sys.argv[1])
video_arg = "%06d" % video_arg

#level_string = 'BustAMove.1pplay.Level1'#'BustAMove.Challengeplay0'
level_string = 'level6.1player.easy.100lives' #'level6.1player.easy.100lives' level1boss.1player.easy.100lives
#/home/jim/projects/contra/videos/1542495791_1 #full level video to test
#python3 render.py videos/1542738976_20/ 19

movie_path = 'videos/1542738976_19/ContraIII-Snes-{}-{}.bk2'.format(level_string,video_arg)
#ContraIII-Snes-level1.1player.easy-000000
print(movie_path)
movie = retro.Movie(movie_path)
movie.step()

scenario_string= 'scenario16_0'#'test_retro' #'trajectory_max'
env = retro.make(game=movie.get_game(), state=level_string, scenario=scenario_string)
env.initial_state = movie.get_state()
env.reset()

#button_dict = ['B', 'A', 'MODE', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'C', 'Y', 'X', 'Z']
#button_dict = ['A', 'B', 'X', 'Y', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'MODE', 'START', 'SELECT', 'L', 'R']
button_dict = ["B", "Y", "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT", "A", "X", "L", "R"]

# inputs = {
#             'A': keycodes.X in keys_pressed or buttoncodes.A in buttons_pressed,
#             'B': keycodes.Z in keys_pressed or buttoncodes.B in buttons_pressed,
#             #'C': keycodes.C in keys_pressed,
#             'X': keycodes.S in keys_pressed or buttoncodes.X in buttons_pressed,
#             'Y': keycodes.A in keys_pressed or buttoncodes.Y in buttons_pressed,
#             #'Z': keycodes.D in keys_pressed,
#
#             'UP': keycodes.UP in keys_pressed or buttoncodes.D_UP in buttons_pressed,
#             'DOWN': keycodes.DOWN in keys_pressed or buttoncodes.D_DOWN in buttons_pressed,
#             'LEFT': keycodes.LEFT in keys_pressed or buttoncodes.D_LEFT in buttons_pressed,
#             'RIGHT': keycodes.RIGHT in keys_pressed or buttoncodes.D_RIGHT in buttons_pressed,
#
#             'MODE': keycodes.TAB in keys_pressed or buttoncodes.SELECT in buttons_pressed,
#             'START': keycodes.ENTER in keys_pressed or buttoncodes.START in buttons_pressed,
#             'SELECT': keycodes.L in keys_pressed,
#             'L': keycodes.W in keys_pressed,
#             'R': keycodes.E in keys_pressed,
#        }

num_buttons = len(env.buttons)#12#len(button_dict)

num_steps = 0
total_reward = 0.
# keys_file = open('keys.csv','w')
# keys_csv = csv.DictWriter(keys_file,fieldnames=['step','keys','action','r','x','y','rings'])
# keys_csv.writeheader()
#
# trajectory_steps = 0
# traj_dict = OrderedDict()
#
# #creates a waypoint for the reward function when the user holds waypoint_key down for waypoint_threshold number of steps
# waypoint_steps = 0
# waypointx_dict = OrderedDict()
# waypointy_dict = OrderedDict()
# waypoint_key = 'DOWN'
# waypoint_this_frame = 0
# waypoint_press = 0
# waypoint_threshold = 30
# prev_waypoint_x = 0
# prev_waypoint_y = 0

boss_health_keys = []
for i in range(33):
    boss_health_keys.append("hp_" + str(i))

print('stepping movie')
death_count = 0
scroll_count = 0
while movie.step():
    if debug:
        env.render()
        time.sleep(0.001)
    keys = []
    #key_string = '_'
    #waypoint_this_frame = 0
    for i in range(num_buttons):
        #print(i)
        keys.append(movie.get_key(i,0))
        #if movie.get_key(i):
            #key_string += button_dict[i] + "_"
        #    if button_dict[i] == waypoint_key:
                #print(movie.get_key(i),button_dict[i])
        #        waypoint_this_frame = 1

    _obs, _rew, _done, _info = env.step(keys)
    _rew = np.round(_rew,6)
    num_steps += 1
    total_reward += _rew
    total_reward = np.round(total_reward,6)
    if _rew > 1:
        scroll_count += 1
    elif _rew < -1:
        death_count += 1

    #keys_csv.writerow({'step': num_steps, 'keys':key_string, 'action':key_string, 'r':_rew, 'x':current_x, 'y':current_y, 'rings':_info['rings']})
    if debug: #and _rew > -1000:
        #if _rew != 0:
            #print(_info)
        #print("reward: {}--{}: {} -- {}: {} -- {}: {} total_reward: {}".format(_rew,'x1',_info['x1'],'score',_info['score'],'is_scroll',_info['is_scrollable'],total_reward))
            #print(np.round(_rew,2), "_",np.round(total_reward,0),"_", _info['x1'],"--{},{}--{}".format(_done,_done,_done))
        # print("r: {}--{}: {} -- {}: {} -- {}: {} tot_r: {} {} {}".format(_rew, 'x1', _info['x1'], 'score',
        #                                                                        _info['score'], 'is_scroll',
        #                                                                        _info['is_scrollable'], total_reward,scroll_count,death_count))
        if _rew > 10:
            print("r: {}--{}: {} -- {}: {} tot_r: {} sv {} {}".format(_rew, 'x1', _info['x1'], 'is_scroll',
                                                                             _info['is_scrollable'], total_reward,
                                                                             _info['scroll_value'],_info['level']))
        # for i in boss_health_keys:
        #     if _info[i] > 0:
        #         print("Python boss hp value {} {}".format(i, _info[i]) )
#keys_file.close()

# if not debug:
#     t_file = open('traj_dict_{}_{}.csv'.format(level_string,replay_number),'w')
#     t_csv = csv.DictWriter(t_file,fieldnames=['line'])
#     for key,value in traj_dict.items():
#         zString = "    ret_value[{}] = {}".format(key, value)
#         t_csv.writerow({'line':zString})
#     t_file.close()
#
#     w_file = open('waypoint_dict_{}_{}.csv'.format(level_string,replay_number),'w')
#     w_csv = csv.DictWriter(w_file,fieldnames=['line'])
#     for key,value in waypointx_dict.items():
#         zString = "    ret_x[{}] = {}".format(key,value)
#         w_csv.writerow({'line':zString})
#     for key,value in waypointy_dict.items():
#         zString = "    ret_y[{}] = {}".format(key,value)
#         w_csv.writerow({'line':zString})
#     w_file.close()
