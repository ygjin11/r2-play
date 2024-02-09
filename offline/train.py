import sys
import os
current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
parent_directory = os.path.abspath(os.path.join(current_directory, '..'))
grandparent_directory = os.path.abspath(os.path.join(parent_directory, '..'))
sys.path.append(parent_directory)
sys.path.append(grandparent_directory)
from model.dt import GPTConfig, GPT
from model.dt_condition import GPTConfig_condition, GPT_condition
from offline.trainer import Trainer, TrainerConfig
from tool.utils import set_seed
from create_dataset import create_dataset
import csv
import logging
import numpy as np
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset,  random_split
import os
import yaml
from collections import deque
import os
import yaml
from PIL import Image
import clip
from rich.console import Console
console = Console()
from multiprocessing import Pool
import multiprocessing
import sys
from tqdm import tqdm, trange
import argparse

# our dataset
class StateActionReturnDataset(Dataset):
    def __init__(self, data, block_size, actions, done_idxs, rtgs, timesteps,\
                game, all_games, condition_type, instruct_dir):        
        self.block_size = block_size
        self.vocab_size = 18
        self.data = data
        self.actions = actions
        self.done_idxs = done_idxs
        self.rtgs = rtgs
        self.timesteps = timesteps
        self.game = game
        self.all_games = all_games
        self.condition_type = condition_type
        self.device = 'cpu'
        if self.condition_type == 'guide':
            self.game_dict_desc = dict()
            self.game_dict_traj = dict()
            self.game_dict_guid = dict()
            for game in all_games:
                self.game_dict_desc[game] = torch.load(instruct_dir + f'/embed/desc/{game}.pt', self.device)
                self.game_dict_traj[game] = torch.load(instruct_dir + f'/embed/traj/{game}.pt', self.device)
                self.game_dict_guid[game] = torch.load(instruct_dir + f'/embed/guid/{game}.pt', self.device) 
                    
    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        block_size = self.block_size // 3
        done_idx = idx + block_size
        for i in self.done_idxs:
            if i > idx: # first done_idx greater than idx
                done_idx = min(int(i), done_idx)
                break
        idx = done_idx - block_size
        states = torch.tensor(np.array(self.data[idx:done_idx]), \
                            dtype=torch.float32).reshape(block_size, -1) 
                             # (block_size, 4*84*84)
        states = states / 255.
        # (block_size, 1)
        actions = torch.tensor(self.actions[idx:done_idx], dtype=torch.long).unsqueeze(1) 
        rtgs = torch.tensor(self.rtgs[idx:done_idx], dtype=torch.float32).unsqueeze(1)
        timesteps = torch.tensor(self.timesteps[idx:idx+1], dtype=torch.int64).unsqueeze(1)
         
        if self.condition_type == 'raw':
            return states, actions, rtgs, timesteps
        else:
            game = self.game[idx:done_idx][0]
            desc = self.game_dict_desc[game].reshape(-1)
            traj = self.game_dict_traj[game]
            guid = self.game_dict_guid[game]
            return states, actions, rtgs, timesteps, desc, traj, guid


if __name__ == '__main__':     
    current_file = os.path.abspath(__file__)
    current_directory = os.path.dirname(current_file)
    parent_directory = os.path.dirname(current_directory)
    config_train_path = parent_directory + f'/config/config_main/train.yaml'
    config_path = parent_directory + '/config/config_main/public.yaml'
    with open(config_path, 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)
    with open(config_train_path, 'r') as yaml_file:
        config_train = yaml.safe_load(yaml_file)
    context_length = config['context_length']
    model_type = config['model_type'] 
    data_dir_prefix = config['dataset_dir']
    instruct_dir = config['instruct_dir']
    print(instruct_dir)
    epochs = config_train['epochs'] 
    num_steps = config_train['num_steps']
    num_buffers = config_train['num_buffers']
    batch_size = config_train['batch_size']
    trajectories_per_buffer = config_train['trajectories_per_buffer']
    ckpt_path = config_train['ckpt_path']
    condition_type = config_train['condition_type']
    game_list =config_train['game_list']
    game_path = parent_directory + '/config/config_game/' + game_list + '.yaml'
    condition_dim = config_train['condition_dim']
    train_info = config_train['train_info']
    with open(game_path, 'r') as yaml_file:
        config_game = yaml.safe_load(yaml_file)    
    all_games = config_game['train']
    seed = config['seed'] 
    set_seed(seed)

    console.log(f'load data...')
    # mp to load dataset
    num_game_batch = config_train['num_game_batch']
    obss, actions, returns, done_idxs, rtgs, timesteps, game = [], [], [], [], [], [], []
    def load_game_data(game, num_buffers, num_steps, data_dir_prefix, trajectories_per_buffer):
        return create_dataset(num_buffers, num_steps, [game], data_dir_prefix, trajectories_per_buffer)
    def get_data(num_steps, all_games, data_dir_prefix, num_game_batch, num_buffers, trajectories_per_buffer):
        obss, actions, returns, done_idxs, rtgs, timesteps, game = [], [], [], [], [], [], []
        list_game = []

        for i in range(int(len(all_games) / num_game_batch) + 1):
            if i * num_game_batch < len(all_games):
                list_game.append(all_games[i*num_game_batch:min(i*num_game_batch+num_game_batch, len(all_games))])
        print(f"game batch: {list_game}")
        for i in range(len(list_game)):
            all_games_batch = list_game[i]
            print(f'load game batch {all_games_batch}')
            pool = Pool(processes=len(all_games_batch))
            game_data = pool.starmap(load_game_data, [(game, num_buffers, num_steps, data_dir_prefix, trajectories_per_buffer) for game in all_games_batch])
            pool.close()
            pool.join()
            for data in game_data:
                obss.extend(data[0])
                actions.extend(data[1])
                returns.extend(data[2])
                done_idxs.extend(data[3])
                rtgs.extend(data[4])
                timesteps.extend(data[5])
                game.extend(data[6])
            console.log(f'len of dataset: {len(obss)}')
        return obss, actions, returns, done_idxs, rtgs, timesteps, game
    
    # trian data
    obss, actions, returns, done_idxs, rtgs, timesteps, game = \
        get_data(num_steps, all_games, data_dir_prefix, num_game_batch,\
                 num_buffers, trajectories_per_buffer)

    
    console.log(f'generat dataset for train...')
    dataset = StateActionReturnDataset(obss, context_length*3, actions, done_idxs, rtgs, timesteps,\
                                      game, all_games, condition_type, instruct_dir)  
    
    dataset_size = len(dataset)
    train_size = int(0.7 * dataset_size)
    test_size = dataset_size - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    console.log(f'train-initialize...')  
    
    # DT
    if condition_type == 'raw':
        mconf = GPTConfig(dataset.vocab_size, dataset.block_size,\
                        n_layer=6, n_head=8, n_embd=128,\
                        model_type=model_type, max_timestep=max(timesteps))
        model = GPT(mconf)

    # task conditioned DT
    else:
        mconf = GPTConfig_condition(dataset.vocab_size, dataset.block_size,
                n_layer=6, n_head=8, n_embd=128, model_type=model_type, max_timestep=max(timesteps),\
                condition_dim=condition_dim) 
        model = GPT_condition(mconf) 

    tconf = TrainerConfig(max_epochs=epochs, batch_size=batch_size,\
                          learning_rate=6e-4, lr_decay=True, warmup_tokens=512*20,\
                          final_tokens=2*len(train_dataset)*context_length*3, num_workers=4,\
                          seed=seed, model_type=model_type, game=game,\
                          max_timestep=max(timesteps), ckpt_path=ckpt_path)
    trainer = Trainer(model, train_dataset, test_dataset, tconf, game_list,\
                      condition_type=condition_type, train_info=train_info,\
                      bs=batch_size, num_steps=num_steps) 

    console.log(f'train...') 
    trainer.train()


