import datetime
import json
from expectation_player import ExpectationPlayer
from greedy_player import GreedyWordlePlayer
from wordle_environment import WordleEnvironment
import numpy as np
import torch
from pathlib import Path

from rl_player import RLPlayer
from game_room import evaluate_player, train_player

use_only_solutions = True
word_len = 2
num_episodes_to_train = 10000000
greedy_player = False
reward_success = 15
reward_failure = -15
reward_yellow = 0.5
reward_green = 1.5
rounds_to_failure = 6
sync_every = 1000
learn_every = 3
lr = 0.00025
gamma = 0
n_embd = 16
n_head = 2
n_layer = 2
dropout = 0
save_every = 50000
burnin = 1000
learn_every = 3
exploration_rate_decay = 0.999995
exploration_rate_min = 0.001
batch_size = 32
memory_size = 1000
save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir.mkdir(parents=True)
debug = False
log_every = 5000
# create config variable from global variables (except for those starting with _)
config = {k: v for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))}
with open(save_dir / "config", 'w') as f:
    json.dump(config, f)
config['save_dir'] = save_dir
print(f"config = {config}")
device = 'cpu'

env = WordleEnvironment(config)

# get greedy score for comparison
greedy_agent = GreedyWordlePlayer(config)
print("evaluating greedy agent")
__ = evaluate_player(greedy_agent, env, num_games_to_evaluate=2000, should_print=True)

expectation_player = ExpectationPlayer(config)
print("evaluating expectation agent")
__ = evaluate_player(expectation_player, env, num_games_to_evaluate=200, should_print=True)

# now train RL agent
agent = GreedyWordlePlayer(config) if config['greedy_player'] else RLPlayer(config, device)
train_player(agent, env, config, save_dir)

