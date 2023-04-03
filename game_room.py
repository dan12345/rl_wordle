import numpy as np
import json
from metric_logger import MetricLogger
from rl_player import RLPlayer
from wordle_environment import WordleEnvironment
from pathlib import Path
import argparse


def train_player(agent, env, config, save_dir):
    """ train loop adapted from https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html"""
    if config['greedy_player']:
        return
    logger = MetricLogger(save_dir, config)
    percent_win = 0
    repeat_last_solution = False
    for e in range(config['num_episodes_to_train']):
        state = env.reset(repeat_last_solution=repeat_last_solution)
        n_turn = 0
        while True:
            action = agent.act(state, n_turn)
            n_turn += 1
            next_state, reward, done = env.step(action)
            if config['debug']:
                print(
                    f"solution is {env.solution} state is {state} action is {action} reward is {reward} next_state is {next_state} done is {done}")
            agent.cache(state, next_state, action, reward, done)
            q, loss = agent.learn()
            logger.log_step(reward, loss, q)
            state = next_state
            if done:
                # repeat the last failure with some probability to focus on hard words. Reduce 0.2 to not overdo it
                if config['should_repeat_failures'] and reward < 0 and np.random.rand() < percent_win/100-0.2:
                    repeat_last_solution = True
                else:
                    repeat_last_solution = False
                break
        logger.log_episode()
        if e % config['log_every'] == 0 and e > 0:
            percent_win, average_win_len = evaluate_player(agent, env)
            logger.record(episode=e, epsilon=np.round(agent.exploration_rate, 5), step=agent.curr_step,
                          percent_win=percent_win, average_win_len=average_win_len)


def evaluate_player(player, env, should_print=False, print_failures=False):
    n_games_played = 0
    n_games_won = 0
    sum_game_lengths = 0
    dist = {i: 0 for i in range(1, 7)}
    for sol in env.valid_solutions:
        state = env.reset(sol)
        player.reset()
        n_turn = 0
        is_done = False
        n_games_played += 1
        while not is_done:
            n_turn += 1
            new_guess = player.act(state, n_turn, force_exploit=True)
            state, reward, is_done = env.step(new_guess)
            if is_done:
                if reward > 0:  # agent won! good job agent
                    n_games_won += 1
                    sum_game_lengths += n_turn
                    dist[n_turn] += 1
                if print_failures: # and reward < 0:
                    print("state is %s reward is %s solution is %s" % (state, reward, env.solution))
                if n_games_played < 5:
                    print("state is %s reward is %s solution is %s" % (state, reward, env.solution))
    if should_print:
        dist = {k: np.round(v / n_games_played * 100, 1) for k, v in dist.items()}
        print(f"win distribution is {dist}")
        print(
            f"played {n_games_played} games, won {np.round(n_games_won / n_games_played * 100, 1)}% of games, average game length for wins {sum_game_lengths / n_games_won}")

    percent_win = np.round(n_games_won / n_games_played * 100, 3)
    average_win_len = -1 if n_games_won == 0 else np.round(sum_game_lengths / n_games_won, 3)
    return percent_win, average_win_len


def evaluate_saved_player(save_dir, checkpoint):
    with open(save_dir + "/config", 'r') as f:
        config = json.load(f)
    print(config)
    config['save_dir'] = save_dir
    env = WordleEnvironment(config)
    agent = RLPlayer(config, 'cpu', save_dir + "/" + checkpoint)
    evaluate_player(agent, env, should_print=True, print_failures=True)


def play_against_player(save_dir, checkpoint):
    with open(save_dir + "/config", 'r') as f:
        config = json.load(f)

    agent = RLPlayer(config, 'cpu', save_dir + "/" + checkpoint)
    n_turn = 0
    success = False
    print("think of a 5 letter word, I will try to guess it")
    state = '!'
    while n_turn <= 6 and not success:
        n_turn += 1
        action = agent.act(state, n_turn, force_exploit=True)
        print(f"my guess is {action}, please input the evaluation of my guess")
        eval = input()
        state += eval
        if eval == 'G' * config['word_len']:
            success = True
    if success:
        print(f"I win! I succeeded in guessing {action} in {n_turn} turns")
    else:
        print(f"I lose! :(")


def debug_q_values_of_saved_model(s, save_dir, checkpoint):
    with open(save_dir + "/config", 'r') as f:
        config = json.load(f)
    agent = RLPlayer(config, 'cpu', save_dir + "/" + checkpoint)
    words_qs = agent.debug_q_values_of_state(s)
    print(f"for state {s} the q values are: ")
    for word, q in sorted(words_qs, key=lambda x: x[1], reverse=True):
        print(f"{word} {q}")

def continue_training(save_dir, checkpoint):
    with open(save_dir + "/config", 'r') as f:
        config = json.load(f)
    path = Path(save_dir)
    config['save_dir'] = path
    config['should_repeat_failures'] = False
    env = WordleEnvironment(config)
    agent = RLPlayer(config, 'cpu', save_dir + "/" + checkpoint)
    train_player(agent, env, config, path)

if __name__ == '__main__':
    # evaluate_saved_player('checkpoints/2023-03-20T08-53-12', 'wordle_net_55.chkpt', 5000)

    import datetime
    now = datetime.datetime.now()

    parser = argparse.ArgumentParser(description='RL game room')

    parser.add_argument('-m', '--chkpt', type=str, help='Name of chkpt file')
    parser.add_argument('-d', '--dir', type=str,  help='directory of chkpt file')
    parser.add_argument('-a', '--action', type=str,  help='which action to perform')
    parser.add_argument('-s', '--state', type=str,   help='state for debug')

    args = parser.parse_args()

    assert(args.dir is not None)
    assert(args.chkpt is not None)
    assert(args.action is not None)

    if args.action == 'evaluate':
        evaluate_saved_player(args.dir, args.chkpt)
    elif args.action == 'play':
        play_against_player(args.dir, args.chkpt)
    elif args.action == 'debug':
        assert(args.state is not None)
        debug_q_values_of_saved_model(args.state, args.dir, args.chkpt)
    elif args.action == 'continue':
        continue_training(args.dir, args.chkpt)
    print((datetime.datetime.now() - now).total_seconds())