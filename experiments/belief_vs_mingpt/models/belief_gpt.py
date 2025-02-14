"""Belief-enhanced GPT model that integrates dynamic belief system with minGPT."""

import torch
import torch.nn as nn
from mingpt.model import GPT, Block, CausalSelfAttention
from src.dynamic_belief_system_v2 import DynamicBeliefSystem
import math
import torch.nn.functional as F

class BeliefEnhancedAttention(CausalSelfAttention):
    """Enhanced attention mechanism that incorporates belief system dynamics."""
    
    def __init__(self, config):
        super().__init__(config)
        self.belief_system = DynamicBeliefSystem(
            dimension=config.belief_dimension,
            learning_rate=config.belief_learning_rate,
            temporal_decay_rate=config.belief_temporal_decay_rate,
            activation_threshold=config.belief_activation_threshold,
            dominance_threshold=config.belief_dominance_threshold
        )
        
        # Additional projection for belief integration
        self.belief_proj = nn.Linear(config.n_embd, config.belief_dimension)
        self.belief_output = nn.Linear(config.belief_dimension, config.n_embd)
    
    def forward(self, x, layer_past=None, return_present=False):
        B, T, C = x.size()
        
        # Original attention computation
        q = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        # Compute attention scores
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        
        # Reshape and project
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Belief system integration
        belief_patterns = self.belief_proj(y)  # Project to belief dimension
        
        # Process each sequence position through belief system
        enhanced_patterns = torch.zeros_like(belief_patterns)
        for t in range(T):
            pattern_state = self.belief_system.process_expression(
                belief_patterns[:, t].detach().cpu().numpy()
            )
            enhanced_patterns[:, t] = torch.tensor(
                pattern_state['strength'],
                device=belief_patterns.device
            )
        
        # Project back to embedding dimension and combine
        belief_enhanced = self.belief_output(enhanced_patterns)
        y = y + belief_enhanced  # Residual connection
        
        # Final projection
        y = self.proj(y)
        y = self.proj_drop(y)
        
        return y

class BeliefEnhancedBlock(Block):
    """GPT block with belief-enhanced attention."""
    
    def __init__(self, config):
        super().__init__(config)
        self.attn = BeliefEnhancedAttention(config)

class BeliefGPT(GPT):
    """GPT model enhanced with dynamic belief system."""
    
    def __init__(self, config):
        super().__init__(config)
        # Replace standard blocks with belief-enhanced blocks
        self.blocks = nn.ModuleList([BeliefEnhancedBlock(config) for _ in range(config.n_layer)])
        
        # Initialize belief system metrics
        self.belief_stability = 1.0
        self.pattern_strength = 0.0
        
    def forward(self, idx, targets=None):
        """Forward pass with additional belief system metrics."""
        outputs = super().forward(idx, targets)
        
        # Update belief system metrics
        if isinstance(self.blocks[0].attn.belief_system, DynamicBeliefSystem):
            self.belief_stability = self.blocks[0].attn.belief_system.stability
            pattern_stats = self.blocks[0].attn.belief_system.get_pattern_stats()
            self.pattern_strength = pattern_stats['mean_strength']
        
        return outputs
    
    def get_belief_metrics(self):
        """Return current belief system metrics."""
        return {
            'stability': self.belief_stability,
            'pattern_strength': self.pattern_strength
        } 