import pickle

START_TOKEN = '!'


def get_eval_dict(word_length, use_only_solutions):
    if use_only_solutions:
        fname = f'words/ss{word_length}.pickle'
    else:
        fname = f'words/sg{word_length}.pickle'
    with open(fname, 'rb') as f:
        return pickle.load(f)

def get_words(word_length, get_solutions=True):
    if get_solutions:
        fname = f'words/solutions{word_length}.csv'
    else:
        fname = f'words/guesses{word_length}.csv'
    with open(fname, 'r') as f:
        return f.read().splitlines()

def get_eval_dict(word_len, use_only_solutions):
    if use_only_solutions:
        fname = f'words/ss{word_len}.pickle'
    else:
        fname = f'words/sg{word_len}.pickle'
    with open(fname, 'rb') as f:
        return pickle.load(f)

def calc_eval_dict(solutions, guesses, fname):
    """
    write to pickle file a dictionary containing the evaluation of each guess given each solution
    """
    d = {}
    for word1 in solutions:
        for word2 in guesses:
            d[(word1, word2)] = get_eval(word1, word2, [])
    with open(fname, 'wb') as f:
        pickle.dump(d, f)


def get_eval(sol, guess, eval_dict):
    """
    returns a string containing the response wordle would give to guess given solution. example = given solution = "abcde" and guess = "aooeo", response would be "GWWEW"
    Green means the letter is in the correct position, yellow means the letter is in the word but not in the correct position
    and white means the letter is not in the word
    """
    assert len(sol) == len(guess)
    word_len = len(sol)
    res = ['W'] * word_len
    if (sol, guess) in eval_dict:
        return eval_dict[(sol, guess)]
    non_green_counts = {}
    # first fill in greens
    for i in range(0, word_len):
        if guess[i] == sol[i]:
            res[i] = 'G'
        else:
            non_green_counts[sol[i]] = 1 if sol[i] not in non_green_counts else non_green_counts[sol[i]] + 1

    # now fill in yellows - needed to do separately because of tricky logic to take into account duplicates
    for i in range(0, word_len):
        if res[i] != 1:
            if guess[i] in non_green_counts and non_green_counts[guess[i]] > 0:
                res[i] = 'Y'
                non_green_counts[guess[i]] -= 1
    return "".join(res)