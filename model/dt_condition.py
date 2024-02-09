# external package
import math
import logging
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import os
import yaml
from rich.console import Console
console = Console()
# our package
from model.adapter_generators import ParameterGenerator
from model.adaptors_generators_moe import ParameterGenerator as ParameterGenerator_moe
from tool.module import Attention_Seqtovec, MLP
C_TYPE = 3

class GELU(nn.Module):
    def forward(self, input):
        return F.gelu(input)

class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1
    adapter_norm_input = False
    adapter_dim = 64
    att_input_dim = 512
    l_embed_dim = 128  # layer embedding dim
    condition_dim = 128  # multimodal embedding dim

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k,v in kwargs.items():
            setattr(self, k, v)

class GPTConfig_condition(GPTConfig):
    """ GPT-1 like network roughly 125M params """
    n_layer = 12
    n_head = 12
    n_embd = 768

class AdapterLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.adapter_dim = config.adapter_dim
        hidden_size = config.n_embd
        self.input_dim = config.n_embd
        self.output_dim = config.n_embd
        # insertion weights
        self.adapter_down_weight = None
        self.adapter_down_bias = None
        self.adapter_up_weight = None
        self.adapter_up_bias = None
        self.hidden_act = nn.ReLU()
        # learnt adapter + inits for it
        self.adapter_down_manual = nn.Linear(hidden_size, self.adapter_dim)
        self.adapter_up_manual = nn.Linear(self.adapter_dim, hidden_size)
        nn.init.xavier_uniform_(self.adapter_up_manual.weight, gain=1e-4)
        nn.init.xavier_uniform_(self.adapter_down_manual.weight, gain=1e-4)
        nn.init.constant_(self.adapter_up_manual.bias, 0.0)
        nn.init.constant_(self.adapter_down_manual.bias, 0.0)

    def clear_adapter(self):
        self.adapter_down_weight = None
        self.adapter_down_bias = None
        self.adapter_up_weight = None
        self.adapter_up_bias = None

    def apply_adapter_params(self, bsz, uw, dw, ub, db):

        self.adapter_down_weight = dw.view(bsz, self.input_dim, self.adapter_dim)
        self.adapter_down_bias = db.view(bsz, self.adapter_dim)
        self.adapter_up_weight = uw.view(bsz, self.adapter_dim, self.output_dim)
        self.adapter_up_bias = ub.view(bsz, self.output_dim)

    def forward(self, x):
        if self.adapter_down_weight is not None:
            x = (x @ self.adapter_down_weight) + self.adapter_down_bias.unsqueeze(1)  
            ## x:batch * length * hid_dim  @  weight:batch * hid_dim * adapter_dim
            ##  =  batch * length * adapter_dim
            x = self.hidden_act(x)
            x = (x @ self.adapter_up_weight) + self.adapter_up_bias.unsqueeze(1)
        else:
            x = self.adapter_down_manual(x)
            x = self.hidden_act(x)
            x = self.adapter_up_manual(x)
        return x  ## no residual connection - we let the user of this layer decide that

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        # self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
        #                              .view(1, 1, config.block_size, config.block_size))
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size + 1, config.block_size + 1))
                                     .view(1, 1, config.block_size + 1, config.block_size + 1))
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) ## (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) ## (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) ## (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.drop = nn.Dropout(config.embd_pdrop)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ln3 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )
        self.adapter_layer = AdapterLayer(config)


    def forward(self, hidden_states):
        normed_states = self.ln3(hidden_states)
        hidden_states = hidden_states + self.attn(self.ln1(hidden_states))
        forward_states = hidden_states + self.mlp(self.ln2(hidden_states))
        adapter_input = (
            normed_states)
        x = (
            hidden_states
            + self.drop(forward_states)
            + self.adapter_layer(adapter_input)
        )
        return x

class GPT_condition(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.model_type = config.model_type
        # input embedding stem
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        # self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size + 1, config.n_embd))
        self.global_pos_emb = nn.Parameter(torch.zeros(1, config.max_timestep+1, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.block_size = config.block_size
        self.param_gen = ParameterGenerator_moe(
            config, config.n_embd
        )

        self.trajtovector = Attention_Seqtovec(512, config.condition_dim, 1, 1) 
        self.insttovector = Attention_Seqtovec(512, config.condition_dim, 1, 1) 
        if C_TYPE == 1:
            self.fusion = MLP(1536, 1024, 512)
        elif C_TYPE == 3:
            self.alpha = nn.Parameter(torch.tensor(0.5))
            self.beta = nn.Parameter(torch.tensor(0.5))
            self.gamma = nn.Parameter(torch.tensor(0.5))
            self.fusion = MLP(512, 1024, 512)

        self.apply(self._init_weights)
        console.print(f"number of parameters: {sum(p.numel() for p in self.parameters())}")
        self.state_encoder = nn.Sequential(nn.Conv2d(4, 32, 8, stride=4, padding=0), nn.ReLU(),
                                 nn.Conv2d(32, 64, 4, stride=2, padding=0), nn.ReLU(),
                                 nn.Conv2d(64, 64, 3, stride=1, padding=0), nn.ReLU(),
                                 nn.Flatten(), nn.Linear(3136, config.n_embd), nn.Tanh())

        self.ret_emb = nn.Sequential(nn.Linear(1, config.n_embd), nn.Tanh())
        self.action_embeddings = nn.Sequential(nn.Embedding(config.vocab_size, config.n_embd), nn.Tanh())
        nn.init.normal_(self.action_embeddings[0].weight, mean=0.0, std=0.02)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        # whitelist_weight_modules = (torch.nn.Linear, )
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')
        no_decay.add('global_pos_emb')

        # weight_decay of trajtovector
        decay.add('insttovector.transformer.layers.0.self_attn.in_proj_weight')
        decay.add('trajtovector.transformer.layers.0.self_attn.in_proj_weight')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s\
               made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, \
               "parameters %s were not separated into either decay/no_decay set!" \
               % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], \
             "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    # state, action, and return
    def forward(self, states, actions, targets=None, rtgs=None, timesteps=None, \
                desc=None, traj=None, guid=None):
        # states: (batch, block_size, 4*84*84)
        # actions: (batch, block_size, 1)
        # targets: (batch, block_size, 1)
        # rtgs: (batch, block_size, 1)
        # timesteps: (batch, 1, 1)
        # instuction：
        self.clear_adapters()
        state_embeddings = self.state_encoder\
            (states.reshape(-1, 4, 84, 84).type(torch.float32).contiguous()) 
            # (batch * block_size, n_embd)
        state_embeddings = state_embeddings.reshape\
            (states.shape[0], states.shape[1], self.config.n_embd) 
        # (batch, block_size, n_embd)

        if actions is not None and self.model_type == 'reward_conditioned':
            rtg_embeddings = self.ret_emb(rtgs.type(torch.float32))
            action_embeddings = self.action_embeddings(actions.type(torch.long).squeeze(-1)) 
            # (batch, block_size, n_embd)

            token_embeddings = torch.zeros((states.shape[0], states.shape[1]*3 - \
                                            int(targets is None), self.config.n_embd),\
                                            dtype=torch.float32, device=state_embeddings.device)
            token_embeddings[:,::3,:] = rtg_embeddings
            token_embeddings[:,1::3,:] = state_embeddings
            token_embeddings[:,2::3,:] = action_embeddings[:,-states.shape[1] + int(targets is None):,:]
            # token_embeddings=[r_hat,s,a,r,s,a,....,]
        elif actions is None and self.model_type == 'reward_conditioned': 
            # only happens at very first timestep of evaluation
            rtg_embeddings = self.ret_emb(rtgs.type(torch.float32))

            token_embeddings = torch.zeros((states.shape[0], states.shape[1]*2, \
                                            self.config.n_embd), dtype=torch.float32, \
                                                device=state_embeddings.device)
            token_embeddings[:,::2,:] = rtg_embeddings # really just [:,0,:]
            token_embeddings[:,1::2,:] = state_embeddings # really just [:,1,:]
        elif actions is not None and self.model_type == 'naive':
            action_embeddings = self.action_embeddings(actions.type(torch.long).squeeze(-1)) 
            # (batch, block_size, n_embd)

            token_embeddings = torch.zeros((states.shape[0], states.shape[1]*2 - int(targets is None),\
                                            self.config.n_embd), dtype=torch.float32,\
                                            device=state_embeddings.device)
            token_embeddings[:,::2,:] = state_embeddings
            token_embeddings[:,1::2,:] = action_embeddings[:,-states.shape[1] + int(targets is None):,:]
        elif actions is None and self.model_type == 'naive': 
            # only happens at very first timestep of evaluation
            token_embeddings = state_embeddings
        else:
            raise NotImplementedError()

        batch_size = states.shape[0]
        all_global_pos_emb = torch.repeat_interleave(self.global_pos_emb, batch_size, dim=0) 
        # batch_size, traj_length, n_embd
        position_embeddings = torch.gather(all_global_pos_emb, 1,\
                                        torch.repeat_interleave(timesteps, self.config.n_embd, dim=-1)) \
                                        + self.pos_emb[:, :token_embeddings.shape[1], :]
        # extract feature of traj and inst
        traj = traj.reshape(-1, 20, 512)
        guid = guid.reshape(-1, 20, 512)
        t_embed = self.trajtovector(traj).reshape(-1, 50, 512).view(-1, traj.size(-1))
        g_embed = self.insttovector(guid).reshape(-1, 50 ,512).view(-1, guid.size(-1)) 
        d_embed = desc.repeat(1, 50).view(-1, desc.size(-1))
        if C_TYPE == 1:
            cond = self.fusion(torch.cat([d_embed, t_embed, g_embed], dim=1))
        elif C_TYPE == 2:
            cond = self.fusion(d_embed + t_embed + g_embed)
        elif C_TYPE == 3:
            cond = self.fusion(d_embed * self.alpha + t_embed * self.alpha + g_embed * self.gamma)
        cond = cond.view(-1, 50, cond.size(-1))

        # calulate similarity
        x1 = cond.unsqueeze(1) 
        x2 = cond.unsqueeze(2)
        similarity_matrix = torch.matmul(x1, x2.transpose(-1, -2)).squeeze(-1)
        eye = torch.eye(similarity_matrix.size(-1)).to(similarity_matrix.device)
        similarity_matrix -= eye.unsqueeze(0) * similarity_matrix
        similarity = similarity_matrix.sum(dim=-1).unsqueeze(1) / 49
        max_v, _ = torch.max(similarity, dim=-1, keepdim=True)
        similarity = similarity - max_v
        similarity = nn.Softmax(dim=-1)(similarity * 10)
        self.apply_params_to_adapters(
            token_embeddings.size(0),
            self.param_gen(cond, similarity),
        )        
                
        x = self.drop(token_embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
    
        if actions is not None and self.model_type == 'reward_conditioned':
            logits = logits[:, 1::3, :] # only keep predictions from state_embeddings
        elif actions is None and self.model_type == 'reward_conditioned':
            logits = logits[:, 1:, :]
        elif actions is not None and self.model_type == 'naive':
            logits = logits[:, ::2, :] # only keep predictions from state_embeddings
        elif actions is None and self.model_type == 'naive':
            logits = logits # for completeness
        else:
            raise NotImplementedError()

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

        return logits, loss

    def clear_adapters(self):
        for block in self.blocks:
            block.adapter_layer.clear_adapter()

    def apply_params_to_adapters(self, batch_size, generated_params):
        for param, block in zip(generated_params, self.blocks):
            block.adapter_layer.apply_adapter_params(batch_size, *param)  # param：batch * weight