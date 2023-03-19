import numpy as np

from metric_logger import MetricLogger


def train_player(agent, env, config, save_dir):
    """ train loop adapted from https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html"""
    if config['greedy_player']:
        return
    logger = MetricLogger(save_dir, config)
    for e in range(config['num_episodes_to_train']):
        state = env.reset()
        n_turn = 0
        while True:
            action = agent.act(state, n_turn)
            n_turn += 1
            next_state, reward, done = env.step(action)
            if config['debug']:
                print(f"solution is {env.solution} state is {state} action is {action} reward is {reward} next_state is {next_state} done is {done}")
            agent.cache(state, next_state, action, reward, done)
            q, loss = agent.learn()
            logger.log_step(reward, loss, q)
            state = next_state
            if done:
                break
        logger.log_episode()
        if e % config['log_every'] == 0 and e > 0:
            percent_win, average_win_len = evaluate_player(agent, env, num_games_to_evaluate=500)
            logger.record(episode=e, epsilon=np.round(agent.exploration_rate, 5), step=agent.curr_step, percent_win=percent_win, average_len=average_win_len)



def evaluate_player(player, env, num_games_to_evaluate):
    n_games_played = 0
    n_games_won = 0
    sum_game_lengths = 0
    for i in range(num_games_to_evaluate):
        state = env.reset()
        player.reset()
        n_turn = 0
        is_done = False
        n_games_played += 1
        while not is_done:
            n_turn += 1
            new_guess = player.act(state, n_turn)
            state, reward, is_done = env.step(new_guess)
            if is_done:
                if reward > 0:  # agent won! good job agent
                    n_games_won += 1
                    sum_game_lengths += n_turn
                if i < 5:
                    print("state is %s reward is %s solution is %s" % (state, reward, env.solution))
        # if n_games_played % 1000 == 0:
        #     print(f"played {n_games_played} games, won {np.round(n_games_won/n_games_played*100, 1)}% of games, average game length for wins {sum_game_lengths / n_games_won}")

    percent_win = np.round(n_games_won / n_games_played * 100, 1)
    average_win_len = sum_game_lengths / n_games_won
    print(f"played {n_games_played} games, won {percent_win}% of games, average game length for wins {average_win_len}")
    return percent_win, average_win_len

def play_against_player(save_dir):
    print("think of a 5 letter word, I will try to guess it")
    agent = RLPlayer(save_dir, config, device)
    print(f)

