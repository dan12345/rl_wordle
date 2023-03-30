from collections import deque

import torch
import numpy as np
import torch.nn as nn
from models import TransformerModel
from utils import get_words, CHARS, START_TOKEN, ABC
import copy
import random

EVAL_CHARS = 'WYG'  # the characters used to evaluate a guess compared to the solution
PADDING = '.'
chars = PADDING + ABC + EVAL_CHARS + START_TOKEN


class DQN(nn.Module):

    def __init__(self, config, possible_guess, device):
        super(DQN, self).__init__()
        self.online = TransformerModel(config, possible_guess, len(CHARS), device).to(device)
        self.target = copy.deepcopy(self.online)

        # Q target parameters should not be updated
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, state, lens, model):
        if model == 'online':
            return self.online(state, lens)
        elif model == 'target':
            return self.target(state, lens)
        else:
            raise ValueError('model must be either online or target')


class RLPlayer:
    def __init__(self, config, device, load_path=None):
        self.config = config
        self.device = device
        self.loss_fn = torch.nn.SmoothL1Loss()
        self.possible_guesses = get_words(self.config['word_len'], config['use_only_solutions'], config['num_words_to_take'])
        self.net = DQN(config, self.possible_guesses, device).to(device)
        if load_path is not None:
            checkpoint = torch.load(load_path)
            self.net.load_state_dict(checkpoint['model'])
            self.exploration_rate = checkpoint['exploration_rate']
        self.optimizer = torch.optim.Adam(self.net.parameters(), config['lr'])
        self.exploration_rate = 1
        self.char_to_idx = {char: idx for idx, char in enumerate(CHARS)}
        self.guess_to_idx = {guess: idx for idx, guess in enumerate(self.possible_guesses)}
        self.curr_step = 0
        self.memory = deque(maxlen=config['memory_size'])

    def learn(self):
        if self.curr_step % self.config['sync_every'] == 0:
            self.sync_Q_target()
        if self.curr_step % self.config['save_every'] == 0:
            self.save()
        if self.curr_step < self.config['burnin']:
            return None, None
        if self.curr_step % self.config['learn_every'] != 0:
            return None, None
        states, state_lens, next_states, next_state_lens, actions, rewards, dones = self.recall()
        td_estimates = self.td_estimate(states, state_lens, actions)
        td_targets = self.td_target(next_states, next_state_lens, rewards, dones)
        loss = self.update_Q_online(td_estimates, td_targets)
        return td_estimates.mean().item(), loss

    def reset(self):
        return

    def save(self):
        save_path = (
                self.config['save_dir'] / f"wordle_net_{int(self.curr_step // self.config['save_every'])}.chkpt"
        )
        torch.save(
            dict(model=self.net.state_dict(), exploration_rate=self.exploration_rate),
            save_path,
        )
        print(f"WordleNet saved to {save_path} at step {self.curr_step}")

    def act(self, state, _, force_exploit=False):
        state_length = torch.tensor([len(state)]).to(self.device)
        state = self.encode(state)
        state = state.unsqueeze(0)
        rand = np.random.rand()
        if not force_exploit and rand < self.exploration_rate:
            if self.config['sample_from_top_n'] != -1 and rand < self.exploration_rate / 2:  # half of exploration explore top possibility
                action_values = self.net(state, state_length, model='online').squeeze()
                top_k = torch.topk(action_values, self.config['sample_from_top_n'])[1]
                action_idx = top_k[np.random.randint(0, len(top_k))]
            else:
                action_idx = np.random.randint(0, len(self.possible_guesses))
        else:  # exploit
            action_values = self.net(state, state_length, model='online').squeeze()
            action_idx = torch.argmax(action_values).item()
        self.exploration_rate = max(self.config['exploration_rate_min'],
                                    self.exploration_rate * self.config['exploration_rate_decay'])
        self.curr_step += 1

        return self.possible_guesses[action_idx]

    def encode(self, state):
        tensor = torch.tensor([self.char_to_idx[c] for c in state], requires_grad=False).to(self.device)
        # pad tensor to max length
        max_len = self.config['word_len'] * 2 * self.config['rounds_to_failure'] + 1
        tensor = torch.cat((tensor, torch.ones(max_len - len(state), dtype=torch.long, requires_grad=False).to(self.device) * self.char_to_idx[PADDING]))
        return tensor

    def cache(self, state, next_state, action, reward, done):
        state_length = torch.tensor(len(state), device=self.device)
        state = self.encode(state)
        if done:  # if done, then we won't need to calculate the next state Q value, so we can just use a dummy value
            next_state = self.encode(START_TOKEN)
            next_state_length = torch.tensor(1, device=self.device)
        else:
            next_state_length = torch.tensor(len(next_state)).to(self.device)
            next_state = self.encode(next_state)
        action = torch.tensor([self.guess_to_idx[action]], dtype=torch.long).to(self.device)
        reward = torch.tensor([reward], dtype=torch.long).to(self.device)
        done = torch.tensor([done], dtype=torch.long).to(self.device)
        self.memory.append((state, state_length, next_state, next_state_length, action, reward, done))


    def recall(self):
        batch = random.sample(self.memory, self.config['batch_size'])
        states, state_lengths, next_states, next_state_lengths, actions, rewards, dones = map(torch.stack, zip(*batch))
        return states, state_lengths, next_states, next_state_lengths, actions.squeeze(), rewards.squeeze(), dones.squeeze()

    def td_estimate(self, state, state_lens, action):
        current_Q = self.net(state, state_lens, model='online')[np.arange(0, self.config['batch_size']), action]  # Q(s,a)
        return current_Q

    @torch.no_grad()
    def td_target(self, next_state, next_state_lens, reward, done):
        next_state_Q = self.net(next_state, next_state_lens, model='online')
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, next_state_lens, model='target')[np.arange(0, self.config['batch_size']), best_action]
        target_Q = (reward + (1 - done.float()) * self.config['gamma'] * next_Q).float()
        return target_Q

    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())

    def debug_q_values_of_state(self, state):
        state_length = torch.tensor([len(state)]).to(self.device)
        state = self.encode(state)
        state = state.unsqueeze(0)
        action_values = self.net(state, state_length, model='online').squeeze().tolist()
        return zip(self.possible_guesses, action_values)
