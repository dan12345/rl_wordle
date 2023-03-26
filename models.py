import torch.nn as nn
import torch
import math
from torch.nn import functional as F

from utils import ABC


class TransformerModel(nn.Module):

    def __init__(self, config, possible_guesses, n_chars, device):
        super().__init__()
        self.config = config
        n_guesses = len(possible_guesses)
        self.token_embedding_table = nn.Embedding(n_chars, config['n_embd'])
        # each round need double the characters in order to contain eval result
        max_sequence_size = config['word_len'] * 2 * config['rounds_to_failure'] + 1
        self.position_embedding_table = nn.Embedding(max_sequence_size, config['n_embd'])
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config['n_layer'])])
        self.ln_f = nn.LayerNorm(config['n_embd'])  # final layer norm
        if False: #config['pre_calc_guess_emb']:
            self.guess_embedding_table = torch.zeros(n_guesses, ABC * config['word_len'],  device=device, requires_grad=False)
            for i in range(0, n_guesses):
                for j in range(0, len(ABC)):
                    if possible_guesses[i][j] == ABC[j]:
                        self.guess_embedding_table[i][j] = 1
            self.proj = nn.Linear(ABC * config['word_len'], config['n_embd'])
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
        B, T = states.shape
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(states)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device))  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        x = self.ln_f(x)  # (B,T,C)
        if False: #self.config['pre_calc_guess_emb']:
            logits = None
        else:
            logits = self.lm_head(x)
        # get rid of all dimensions besides the last which we care about. We substract 1, since if the state is of length 1 we want the 0'th logit
        return logits[torch.arange(B), lens-1, :]


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
        max_sequence_size = config['word_len'] * config['rounds_to_failure'] * 2 + 1
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
