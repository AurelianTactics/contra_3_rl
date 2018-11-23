#!/usr/bin/python

import sys
import retro
from os import listdir
from os.path import isfile, join, isdir
import time

def render(file):
    print(file)
    movie = retro.Movie(file)
    movie.step()

    env = retro.make(game=movie.get_game(), state=retro.State.NONE, use_restricted_actions=retro.Actions.ALL)
    #env = retro.make(game='SonicTheHedgehog-Genesis', state=retro.STATE_NONE, use_restricted_actions=retro.ACTIONS_ALL)
    env.initial_state = movie.get_state()
    env.reset()
    num_buttons = len(env.buttons)
    frame = 0
    framerate = 4
    while movie.step():
        time.sleep(0.001)
        if frame == framerate:
            env.render()
            frame = 0
        else:
            frame += 1

        keys = []
        for i in range(num_buttons):
            keys.append(movie.get_key(i,0))
        _obs, _rew, _done, _info = env.step(keys)
    env.close()
if isdir(sys.argv[1]):
    onlyfiles = [f for f in listdir(sys.argv[1]) if isfile(join(sys.argv[1], f))]
    onlyfiles.sort()
    min_file_number = int(sys.argv[2])
    file_count = 0
    for file in onlyfiles:
        if ".bk2" in file :
            file_count += 1
            # if min_file_number > file_count:
            #     continue
            temp_substring = file[-10:-4] #these 6 characters of the string have the video number
            temp_int = int(temp_substring)
            # if str(min_file_number) not in file:
            #     continue
            if temp_int != min_file_number:
                continue
            print('playing', file)
            render(sys.argv[1]+file)
            min_file_number += 1
else:
    print('playing', sys.argv[1])
    render(sys.argv[1])
