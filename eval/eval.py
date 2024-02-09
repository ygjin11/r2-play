import sys
import os
current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
parent_directory = os.path.abspath(os.path.join(current_directory, '..'))
grandparent_directory = os.path.abspath(os.path.join(parent_directory, '..'))
sys.path.append(parent_directory)
sys.path.append(grandparent_directory)
import logging
import torch
import cv2
from tool.utils import sample
import atari_py
from collections import deque
import random
import yaml
import os
from rich.console import Console
console = Console()
import torch.cuda as cuda
import argparse
import json
import clip
import torch
import numpy as np
from PIL import Image
# our package
from model.dt import GPTConfig, GPT
from offline.trainer import Trainer, TrainerConfig
from model.dt_condition import GPTConfig_condition, GPT_condition

# write json
def write_json(file, data):
    with open(file, 'w') as file:
        json.dump(data, file, indent=1) 

# env
class Env():

    def __init__(self, args):
        self.device = args.device
        self.ale = atari_py.ALEInterface()
        self.ale.setInt('random_seed', args.seed)
        self.ale.setInt('max_num_frames_per_episode', args.max_episode_length)
        self.ale.setFloat('repeat_action_probability', 0)  # Disable sticky actions
        self.ale.setInt('frame_skip', 0)
        self.ale.setBool('color_averaging', False)
        self.ale.loadROM(atari_py.get_game_path(args.game))  # ROM loading must be done after setting options
        actions = self.ale.getMinimalActionSet()
        self.actions = dict([i, e] for i, e in zip(range(len(actions)), actions))
        self.lives = 0  # Life counter (used in DeepMind training)
        self.life_termination = False  # Used to check if resetting only from loss of life
        self.window = args.history_length  # Number of frames to concatenate
        self.state_buffer = deque([], maxlen=args.history_length)
        self.training = True  # Consistent with model training mode

    def _get_state(self):
        state = cv2.resize(self.ale.getScreenGrayscale(), (84, 84), interpolation=cv2.INTER_LINEAR)
        return torch.tensor(state, dtype=torch.float32, device=self.device).div_(255)

    def _reset_buffer(self):
        for _ in range(self.window):
            self.state_buffer.append(torch.zeros(84, 84, device=self.device))

    def reset(self):
        if self.life_termination:
            self.life_termination = False  # Reset flag
            self.ale.act(0)  # Use a no-op after loss of life
        else:
            # Reset internals
            self._reset_buffer()
            self.ale.reset_game()
            # Perform up to 30 random no-ops before starting
            for _ in range(random.randrange(30)):
                self.ale.act(0)  # Assumes raw action 0 is always no-op
                if self.ale.game_over():
                    self.ale.reset_game()
        # Process and return "initial" state
        observation = self._get_state()
        self.state_buffer.append(observation)
        self.lives = self.ale.lives()
        return torch.stack(list(self.state_buffer), 0)

    def step(self, action):
        # Repeat action 4 times, max pool over last 2 frames
        frame_buffer = torch.zeros(2, 84, 84, device=self.device)
        reward, done = 0, False
        for t in range(4):
            reward += self.ale.act(self.actions.get(action))
            if t == 2:
                frame_buffer[0] = self._get_state()
            elif t == 3:
                frame_buffer[1] = self._get_state()
            done = self.ale.game_over()
            if done:
                break
        observation = frame_buffer.max(0)[0]
        self.state_buffer.append(observation)
        # Detect loss of life as terminal in training mode
        if self.training:
            lives = self.ale.lives()
            if lives < self.lives and lives > 0:  # Lives > 0 for Q*bert
                self.life_termination = not done  # Only set flag when not truly done
                done = True
            self.lives = lives
        # Return state, reward, done
        return torch.stack(list(self.state_buffer), 0), reward, done

    # Uses loss of life as terminal signal
    def train(self):
        self.training = True

    # Uses standard terminal signal
    def eval(self):
        self.training = False

    def action_space(self):
        return len(self.actions)

    def render(self):
        cv2.imshow('screen', self.ale.getScreenRGB()[:, :, ::-1])
        cv2.waitKey(1)

    def close(self):
        cv2.destroyAllWindows()

#parameters of env
class Args:
    def __init__(self, game, seed, max_episode_length=5120):
        self.device = torch.device('cuda')
        self.seed = seed
        self.max_episode_length = max_episode_length
        self.game = game
        self.history_length = 4

# whole evaluation process 
def interact(device, ret, game, seed, model, max_timestep, condition_dim, action_dim, d=None, t=None, g=None, condition_type='raw'):
    done = True
    args = Args(game.lower(), seed)
    env = Env(args)
    env.eval()
    state = env.reset()
    state_tensor = torch.rand(20, 84, 84).to(device)
    n = state_tensor.size(0)
    indices = torch.arange(n, device=state_tensor.device)  # Create indices for all elements
    rolled_indices = torch.remainder(indices - 1, n)  # Compute rolled indices
    state_tensor = state_tensor[rolled_indices]
    state_tensor[-1] = state[0]

    state = state.type(torch.float32).to(device).unsqueeze(0).unsqueeze(0)
    rtgs = [ret]

    # first state is from env, first rtg is target return, and first timestep is 0
    sampled_action = sample(model, state, 1, temperature=1.0, sample=True, actions=None, 
        rtgs=torch.tensor(rtgs, dtype=torch.long).to(device).unsqueeze(0).unsqueeze(-1), 
        timesteps=torch.zeros((1, 1, 1), dtype=torch.int64).to(device),
        d=d, t=t, g=g, condition_type=condition_type)
    
    # if beyond action dim, limit in the scope of action dim
    if sampled_action[0][0].item() >= action_dim:
        sampled_action[0][0] = 0
    j = 0
    all_states = state
    actions = []
    time = 0
    act_reward = 0
    while True:
        time += 1
        if done:
            state, reward_sum, done = env.reset(), 0, False
        action = sampled_action.cpu().numpy()[0,-1]
        actions += [sampled_action]
        state, reward, done = env.step(action)
        reward += act_reward 
        reward_sum += reward
        j += 1
        if done:
            break
        state = state.unsqueeze(0).unsqueeze(0).to(device)
        all_states = torch.cat([all_states, state], dim=0)
        rtgs += [rtgs[-1] - reward]
        # all_states has all previous states and rtgs has all previous rtgs (will be cut to block_size in utils.sample)
        # timestep is just current timestep
        sampled_action = sample(model, all_states.unsqueeze(0), 1, temperature=1.0, sample=True, \
            actions=torch.tensor(actions, dtype=torch.long).to(device).unsqueeze(1).unsqueeze(0), 
            rtgs=torch.tensor(rtgs, dtype=torch.long).to(device).unsqueeze(0).unsqueeze(-1), 
            timesteps=(min(j, max_timestep) * torch.ones((1, 1, 1), dtype=torch.int64).to(device)),
            d=d, t=t, g=g, condition_type=condition_type)

        # if beyond action dim, limit in the scope of action dim
        act_reward = 0
        if sampled_action[0][0].item() >= action_dim:
            sampled_action[0][0] = 0
    print(reward_sum)
    return reward_sum

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--info', type=str, default = 'eval')
    args = parser.parse_args()
    current_file = os.path.abspath(__file__)
    current_directory = os.path.dirname(current_file)
    parent_directory = os.path.dirname(current_directory)
    config_eval_path = parent_directory + f'/config/config_main/{args.info}.yaml'
    config_path = parent_directory + '/config/config_main/public.yaml'
    with open(config_path, 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)
    with open(config_eval_path, 'r') as yaml_file:
        config_eval = yaml.safe_load(yaml_file)
    context_length = config['context_length'] 
    model_type = config['model_type']
    game_list = config_eval['game_list']
    game_path = parent_directory + '/config/config_game/' + game_list + '.yaml'
    ckpt_eval = config_eval['ckpt_eval']
    action_space = config['action_space']
    eval_rtg = config_eval['eval_rtg']
    condition_dim = config_eval['condition_dim']
    instruct_dir = config['instruct_dir']
    with open(game_path, 'r') as yaml_file:
        config_game = yaml.safe_load(yaml_file)
    games_id = config_game['eval_id']
    games_ood = config_game['eval_ood']

    condition_type = config_eval['condition_type']
    eval_num = config_eval['eval_num']
    save_path = config_eval['save_path']
    eval_info = config_eval['eval_info']
    max_timestep = config_eval['max_timestep']

    seed = config['seed']
    console.print(f'seed: {seed}')

    if condition_type == 'raw':
        mconf = GPTConfig(action_space, context_length*3,\
                n_layer=6, n_head=8, n_embd=128,\
                model_type=model_type, max_timestep=max_timestep)
        model = GPT(mconf)
        
    else:
        mconf = GPTConfig_condition(action_space, context_length*3,\
                n_layer=6, n_head=8, n_embd=128, model_type=model_type, max_timestep=max_timestep,\
                condition_dim=condition_dim) 
        model = GPT_condition(mconf) 

    # laod ckpt
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    console.log('load ckpt')
    model.train(False)
    state_dict = torch.load(ckpt_eval)   
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    from datetime import datetime
    parsed_time_date = datetime.now().date()
    parsed_time_hour = datetime.now().hour
    parsed_time_minute = datetime.now().minute
    parsed_time_second = datetime.now().second
    parsed_time = f'{parsed_time_date}/{parsed_time_hour}_{parsed_time_minute}_{parsed_time_second}'
    if not os.path.exists(save_path + str(parsed_time)):
        os.makedirs(save_path + str(parsed_time) +  f'_{condition_type}')
    log_path = save_path + str(parsed_time) + f'_{condition_type}' + '/log.txt'

    res = {}
    clip_model, clip_process = clip.load("ViT-B/32", \
    device="cuda" if torch.cuda.is_available() else "cpu")
    # interact
    for item in games_id:
        game = item['name']
        action_dim = item['action_dim']
        sum_retrun = 0
        console.log(f'eval {item} id')
        if condition_type == 'raw':
           eval_return, eval_var = interact(device, eval_rtg, game, \
                                   seed, model, max_timestep, condition_dim, action_dim, \
                                    d=None, t=None, g=None, condition_type='raw')
        else:
            # init condition
            d = torch.load(instruct_dir + f'/embed/desc/{game}.pt', device)
            t = torch.load(instruct_dir + f'/embed/traj/{game}.pt', device)
            g = torch.load(instruct_dir + f'/embed/guid/{game}.pt', device) 
            eval_return = interact(device, eval_rtg, game, seed, \
                                   model, max_timestep, condition_dim, action_dim, \
                                    d=d, t=t, g=g, condition_type='guide')
        res[game] = eval_return
    res_path = save_path + '/' + str(parsed_time) + f'_{condition_type}' + f'/id_{seed}.json'
    write_json(res_path, res)

    res = {}
    for item in games_ood:
        game = item['name']
        action_dim = item['action_dim']
        sum_retrun = 0
        console.log(f'eval {item} ood')
        if condition_type == 'raw':
            eval_return, eval_var = interact(device, eval_rtg, game, \
                                   seed, model, max_timestep, condition_dim, action_dim, \
                                    d=None, t=None, g=None, condition_type=None)
        else:
            # init condition
            d = torch.load(instruct_dir + f'/embed/desc/{game}.pt', device)
            t = torch.load(instruct_dir + f'/embed/traj/{game}.pt', device)
            g = torch.load(instruct_dir + f'/embed/guid/{game}.pt', device) 
            eval_return = interact(device, eval_rtg, game, seed, \
                                   model, max_timestep, condition_dim, action_dim,  \
                                    d=d, t=t, g=g, condition_type='guide')
        res[game] = eval_return
    res_path = save_path + '/' + str(parsed_time)+ f'_{condition_type}' + f'/ood_{seed}.json'
    write_json(res_path, res)  
