import datetime
import json
from expectation_player import ExpectationPlayer
from greedy_player import GreedyWordlePlayer
import sys
from ast import literal_eval
from wordle_environment import WordleEnvironment
import numpy as np
import torch
from pathlib import Path

from rl_player import RLPlayer
from game_room import evaluate_player, train_player

use_only_solutions = True
word_len = 5
num_episodes_to_train = 10000000
greedy_player = False
reward_success = 15
reward_failure = -15
reward_yellow = 0.2
reward_green = 0.5
num_words_to_take = 100
rounds_to_failure = 6
sync_every = 1000
lr = 0.00025
gamma = 0.6
n_embd = 32
n_head = 2
n_layer = 4
dropout = 0
save_every = 100000
log_every = 10000
burnin = 5000
learn_every = 3
exploration_rate_decay = 0.9999
exploration_rate_min = 0.08
batch_size = 32
memory_size = 5000
save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir.mkdir(parents=True)
debug = False
pre_calc_guess_emb = True
device = 'cpu'
average_out_words = False
max_turn_to_give_non_success_rewards = -1
sample_from_top_n = -1
# create config variable from global variables (except for those starting with _)
def override_config():
    for arg in sys.argv[1:]:
        assert arg.startswith('--')
        key, val = arg.split('=')
        key = key[2:]
        if key in globals():
            try:
                # attempt to eval it (e.g. if bool, number, or etc)
                attempt = literal_eval(val)
            except (SyntaxError, ValueError):
                # if that goes wrong, just use the string
                attempt = val
            # ensure the types match ok
            assert type(attempt) == type(globals()[key])
            # cross fingers
            print(f"Overriding: {key} = {attempt}")
            globals()[key] = attempt
        else:
            raise ValueError(f"Unknown config key: {key}")

override_config()
config = {k: v for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))}
with open(save_dir / "config", 'w') as f:
    json.dump(config, f)
config['save_dir'] = save_dir
print(f"config = {config}")


env = WordleEnvironment(config)

# # get greedy score for comparison
# greedy_agent = GreedyWordlePlayer(config)
# print("evaluating greedy agent")
# __ = evaluate_player(greedy_agent, env, should_print=True)
#
# expectation_player = ExpectationPlayer(config)
# print("evaluating expectation agent")
# __ = evaluate_player(expectation_player, env, should_print=True)

# now train RL agent
agent = GreedyWordlePlayer(config) if config['greedy_player'] else RLPlayer(config, device)
train_player(agent, env, config, save_dir)
