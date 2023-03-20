import random
from utils import get_words, get_eval_dict


class GreedyWordlePlayer:
    """ A player that always guesses the solution if it is in the list of possible solutions, otherwise it guesses a random word"""

    def __init__(self, config):
        self.word_len = config['word_len']
        self.all_solutions = get_words(self.word_len, config['use_only_solutions'])
        self.possible_solutions = self.all_solutions.copy()  # copy of all solutions, so that we can remove words from it
        self.eval_dict = get_eval_dict(self.word_len, config['use_only_solutions'])

    def reset(self):
        self.possible_solutions = self.all_solutions.copy()

    def filter_possible_solutions(self, guess, eval):
        """ remove all elements that don't align with the evaluation """
        self.possible_solutions = [s for s in self.possible_solutions if self.eval_dict[(s, guess)] == eval]

    def act(self, state, n_turn, force_exploit=False):
        if not n_turn == 1:
            last_guess = state[-2 * self.word_len:-self.word_len]
            last_eval = state[-self.word_len:]
            self.filter_possible_solutions(last_guess, last_eval)
            assert (len(self.possible_solutions) > 0)

        # return random word (that can be a solution)
        return random.choice(self.possible_solutions)
