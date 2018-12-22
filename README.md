# contra_3_rl
* Reinforcement Learning attempts to beat Contra 3 for the SNES
* Run using Retro Gym (https://github.com/openai/retro)
* Levels 1-6 cleared using OpenAI Baselines PPO implementation

To run levels 1-6 easy clears:
* Install Open AI Baselines (https://github.com/openai/baselines). These runs use the November 11th, 2018 version of PPO with some slight modifications (see the misc directory)
* Place Contra III-Snes directory in corresponding Retro Gym directory and modify scenario.json and data.json if necessary (see readme.md)
* run ppo2_contra_baselines_agent (...).py from the command line

Explanation of directories:
* /log: stores TensorBoard data for successful level clears
* /logs: stores .csv files with basic stats like reward and timesteps
* /videos: videos of cleared runs
* /ppo_save: saved models

Explantion of files:
* ppo2_contra_baselines_agent (...).py: launches the runs
* render.py: view videos of episodes
* retro_movie_debugger.py: view videos of episodes and examing env outputs like rewards, info, lua script print messages etc. Useful for debugging
* monitor_graphs.ipynb: Jupyter notebook that shows data from the /logs files to see results
* image_test.ipynb: visual testing of observation space so you can see what your agent sees
