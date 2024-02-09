import sys
import os
import json
import yaml
from rich.console import Console
import numpy as np
import torch
import torch.nn as nn
import clip
  
# MLP: concat all info: 512, 256, 256, 512
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# encoder：encode seq to vector
class Attention_Seqtovec(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads, num_layers, dim_feedforward=512, dropout=0):
        super(Attention_Seqtovec, self).__init__()
        # Define the Transformer Encoder
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=input_dim, 
                nhead=num_heads, 
                dim_feedforward=dim_feedforward,
                dropout=dropout
            ),
            num_layers=num_layers
        )
        # Define a Linear layer for the final transformation
        self.fc = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        # x: (batch_size, seq_length, input_dim)
        # Transformer expects input as (seq_length, batch_size, input_dim), so permute
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        # x: (seq_length, batch_size, input_dim)
        # Use mean pooling over sequence dimension to get a single vector
        x = x.mean(dim=0)
        # x: (batch_size, input_dim)
        # Linear transformation to output_dim
        output_vector = self.fc(x)
        return output_vector

if __name__ == '__main__':
    pass