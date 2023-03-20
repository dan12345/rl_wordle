import random
from utils import get_words, get_eval_dict

BEST_FIRST_GUESS = {5: 'raise', 4: 'sale', 3: 'one'}
class ExpectationPlayer:
    """ A player that always guesses the solution that by expectation will minimize number of remaining solutions"""

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
        else:
            if self.word_len in BEST_FIRST_GUESS:
                return BEST_FIRST_GUESS[self.word_len]  # found by best first guess by running below once, here to save time
        best_solution = None
        for possible_guess in self.possible_solutions:
            num_remaining = 0
            for sol in self.possible_solutions:
                eval = self.eval_dict[(sol, possible_guess)]
                num_remaining += len([s for s in self.possible_solutions if eval == self.eval_dict[(s, possible_guess)]])
            num_remaining /= len(self.possible_solutions) # normalize
            if best_solution is None or num_remaining < best_solution[1]:
                best_solution = (possible_guess, num_remaining)
                # print(best_solution)
        return best_solution[0]
