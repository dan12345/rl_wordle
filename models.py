import torch.nn as nn
import torch
import math
from torch.nn import functional as F

from utils import ABC, CHARS, PADDING, START_TOKEN, EVAL_CHARS


class TransformerModel(nn.Module):

    def __init__(self, config, possible_guesses, n_chars, device):
        super().__init__()
        self.config = config
        n_guesses = len(possible_guesses)
        if not config['encode_as_char_positions']:
            self.token_embedding_table = nn.Embedding(n_chars, config['n_embd'])
        # each round need double the characters in order to contain eval result
        if config['average_out_words']:
            self.max_sequence_size = config['word_len'] * 2
        elif config['encode_as_char_positions']:
            self.max_sequence_size = 2 * config['rounds_to_failure'] + 1
            self.input_to_token_embedding = nn.Linear(len(ABC + EVAL_CHARS)*config['word_len'], config['n_embd'])
        else:
            self.max_sequence_size = config['word_len'] * 2 * config['rounds_to_failure'] + 1
        self.position_embedding_table = nn.Embedding(self.max_sequence_size, config['n_embd'])
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config['n_layer'])])
        self.ln_f = nn.LayerNorm(config['n_embd'])  # final layer norm
        self.char_to_idx = {char: idx for idx, char in enumerate(CHARS)}
        if config['pre_calc_guess_emb']:
            # create a matrix where each row represents a binary vector of a possible guess based on the its character breakdown
            self.guess_embedding_table = torch.zeros(n_guesses, len(ABC) * config['word_len'], device=device, requires_grad=False)
            char_to_idx_for_encode = {char: idx for idx, char in enumerate(ABC)}
            for i, guess in enumerate(possible_guesses):
                for j in range(config['word_len']):
                    self.guess_embedding_table[i][char_to_idx_for_encode[guess[j]] * config['word_len'] + j] = 1

            # projection to let the model decide how it want to use the precalced embeddings
            self.proj = nn.Linear(len(ABC) * config['word_len'], config['n_embd'])
        else:
            self.lm_head = nn.Linear(config['n_embd'], n_guesses)
        self.apply(self._init_weights)
        print("transformer model initialized, number parameters = ",
              sum(p.numel() for p in self.parameters() if p.requires_grad))
        self.device = device

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, states, lens):
        batch_size = states.shape[0]
        # idx and targets are both (B,T) tensor of integers
        if self.config['average_out_words']:
            states, n_words_mini_states = self.build_new_states(states, lens)

        if self.config['encode_as_char_positions']:
            tok_emb = self.input_to_token_embedding(states)  # (B,T,C)
            lens = (lens - 1) // self.config['word_len'] + 1
        else:
            tok_emb = self.token_embedding_table(states)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(self.max_sequence_size, device=self.device))  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        x = self.ln_f(x)  # (B,T,C)
        # get rid of all dimensions besides the last which we care about. We substract 1, since if the state is of length 1 we want the 0'th logit
        if self.config['average_out_words']:
            x = x[:, -1, :]  # (B,C)
            x = self.average_out_mini_states(x, n_words_mini_states)
        else:
            x = x[torch.arange(batch_size), lens - 1, :]  # (B,C)
        if self.config['pre_calc_guess_emb']:
            guess_embeddings = self.proj(self.guess_embedding_table)  # (n_guesses, C)
            logits = x @ guess_embeddings.t()  # (B,T,n_guesses)
        else:
            logits = self.lm_head(x)
        return logits

    def average_out_mini_states(self, x, n_word_mini_states):
        new_x = torch.tensor([], device=self.device)
        idx = 0
        for n_word_evals in n_word_mini_states:
            next_idx = idx + n_word_evals
            avg = torch.mean(x[idx:next_idx, :], 0, keepdim=True)
            new_x = torch.cat((new_x, avg), 0)
            idx = next_idx
        return new_x

    def build_new_states(self, states, lens):
        # built a new batch composed of single word evaluation
        new_batch = torch.tensor([], device=self.device, dtype=torch.int64)
        num_mini_states = []  # keep track so we know what to average afterwards
        for i in range(len(states)):
            if lens[i] == 1:
                new_batch = torch.cat((new_batch, torch.ones(1, self.max_sequence_size, dtype=torch.int64) * self.char_to_idx[START_TOKEN]), 0)  # this is ugly, I guess it should work but not sure
                num_mini_states.append(1)
            else:
                n_word_evals = (lens[i] - 1) // self.max_sequence_size  # number of word + word evaluations, each one will be a mini state
                split_states = states[i, 1:lens[i]].view(n_word_evals, self.max_sequence_size)
                new_batch = torch.cat((new_batch, split_states), 0)
                num_mini_states.append(n_word_evals)
        return new_batch, num_mini_states



class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, config):
        super().__init__()
        self.sa = CausalSelfAttention(config)
        self.ffwd = FeedFoward(config)
        self.ln1 = nn.LayerNorm(config['n_embd'])
        self.ln2 = nn.LayerNorm(config['n_embd'])

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config['n_embd'], 4 * config['n_embd']),
            nn.ReLU(),
            nn.Linear(4 * config['n_embd'], config['n_embd']),
            nn.Dropout(config['dropout']),
        )

    def forward(self, x):
        return self.net(x)


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        assert config['n_embd'] % config['n_head'] == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config['n_embd'], 3 * config['n_embd'], bias=False)
        # output projection
        self.c_proj = nn.Linear(config['n_embd'], config['n_embd'])
        # regularization
        if config['average_out_words']:
            max_sequence_size = config['word_len'] * 2
        else:
            max_sequence_size = config['word_len'] * 2 * config['rounds_to_failure'] + 1
        self.register_buffer("bias", torch.tril(torch.ones(max_sequence_size, max_sequence_size))
                             .view(1, 1, max_sequence_size, max_sequence_size))
        self.attn_dropout = nn.Dropout(config['dropout'])

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.config['n_embd'], dim=2)
        k = k.view(B, T, self.config['n_head'], C // self.config['n_head']).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.config['n_head'], C // self.config['n_head']).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.config['n_head'], C // self.config['n_head']).transpose(1, 2)  # (B, nh, T, hs)

        # manual implementation of attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.c_proj(y)
        return y
