import pickle
import random
from utils import get_words, get_eval_dict, START_TOKEN


class WordleEnvironment:

    def __init__(self, config):
        self.solution = None
        self.reward_success = config['reward_success']
        self.reward_failure = config['reward_failure']
        self.reward_yellow = config['reward_yellow']
        self.reward_green = config['reward_green']
        self.rounds_to_failure = config['rounds_to_failure']
        self.state = None
        self.valid_solutions = get_words(config['word_len'], get_solutions=True, num_words_to_take=config['num_words_to_take'])
        self.valid_guesses = get_words(config['word_len'], config['use_only_solutions'], config['num_words_to_take'])
        self.eval_dict = get_eval_dict(config['word_len'], config['use_only_solutions'])
        self.nround = 0
        self.config = config

    def reset(self, sol=None, repeat_last_solution=False):
        """Initializes a new game with a random solution"""
        if not repeat_last_solution:
            self.solution = random.choice(self.valid_solutions) if sol is None else sol
        self.nround = 0
        self.state = START_TOKEN
        return self.state

    def step(self, guess):
        # the assumption here is that the agent will only guess valid words. A harder version would be that the agent needs to learn that as well from scratch, but that makes the game much harder
        assert (guess in self.valid_guesses)
        self.nround += 1
        self.state = self.state + guess
        if guess == self.solution:
            return self.state, self.reward_success, True
        else:
            if self.nround == self.rounds_to_failure:
                return self.state, self.reward_failure, True

        assert((self.solution, guess) in self.eval_dict)
        eval = self.eval_dict[(self.solution, guess)]
        self.state = self.state + eval
        if self.config['max_turn_to_give_non_success_rewards'] == -1 or self.nround <= self.config['max_turn_to_give_non_success_rewards']:
            reward = eval.count('Y') * self.reward_yellow + eval.count('G') * self.reward_green
        else:
            reward = 0
        return self.state, reward, False
