import torch
import torch.nn as nn
from mingpt.model import GPT, Block, CausalSelfAttention
from src.dynamic_belief_system_v2 import DynamicBeliefSystem
import numpy as np

class BeliefEnhancedGPT(GPT):
    """
    GPT model enhanced with Dynamic Belief System for improved consistency and reasoning.
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        # Initialize belief system
        self.belief_dim = config.n_embd // 4  # Use reduced dimension for beliefs
        self.belief_system = DynamicBeliefSystem(
            dimension=self.belief_dim,
            learning_rate=0.2,
            temporal_decay_rate=0.1,
            activation_threshold=0.15,
            dominance_threshold=0.6
        )
        
        # Additional layers for belief integration
        self.belief_projection = nn.Linear(config.n_embd, self.belief_dim)
        self.belief_expansion = nn.Linear(self.belief_dim, config.n_embd)
        
        # Belief state tracking
        self.belief_states = []
        self.stability_history = []
        
        # Ensure all parameters are float32
        self.to(torch.float32)
    
    def _extract_belief_patterns(self, hidden_states):
        """Extract belief patterns from transformer hidden states."""
        with torch.no_grad():  # No need for gradients here
            # Project to belief space
            belief_space = self.belief_projection(hidden_states)  # [B, T, belief_dim]
            
            # Extract patterns between adjacent positions
            patterns = []
            # Process in batches for efficiency
            source = belief_space[:, :-1, :]  # [B, T-1, belief_dim]
            target = belief_space[:, 1:, :]   # [B, T-1, belief_dim]
            
            # Get indices of maximum values for all positions at once
            src_indices = torch.argmax(source, dim=2)  # [B, T-1]
            tgt_indices = torch.argmax(target, dim=2)  # [B, T-1]
            
            # Convert to CPU for list processing (faster than processing on GPU)
            src_indices = src_indices.cpu()
            tgt_indices = tgt_indices.cpu()
            
            # Create patterns
            for b in range(belief_space.size(0)):
                for t in range(src_indices.size(1)):
                    patterns.append((int(src_indices[b, t]), int(tgt_indices[b, t])))
            
            return patterns
    
    def _apply_belief_enhancement(self, hidden_states):
        """Apply belief system insights to enhance hidden states."""
        with torch.no_grad():  # No need for gradients in belief computation
            batch_size, seq_len, _ = hidden_states.shape
            device = hidden_states.device
            
            # Project to belief space
            belief_space = self.belief_projection(hidden_states)
            
            # Get dominant belief indices for all positions at once
            belief_indices = torch.argmax(belief_space, dim=2)  # [B, T]
            
            # Create a mask for connected beliefs
            enhancement_mask = torch.zeros_like(belief_space)
            
            # Convert belief system P to tensor on correct device
            P_tensor = torch.tensor(self.belief_system.P, dtype=torch.float32, device=device)
            
            # Process each position
            for b in range(batch_size):
                for t in range(seq_len):
                    belief_idx = belief_indices[b, t].item()
                    
                    # Find connected beliefs efficiently using tensor operations
                    connected_mask = P_tensor[belief_idx] > self.belief_system.activation_threshold
                    if connected_mask.any():
                        enhancement_mask[b, t, connected_mask] = P_tensor[belief_idx, connected_mask] * 0.5
        
        # Apply enhancements with gradients
        enhanced_belief_space = belief_space + enhancement_mask
        enhanced_states = self.belief_expansion(enhanced_belief_space)
        
        # Residual connection with smaller scale
        return hidden_states + 0.1 * enhanced_states  # Reduced from 0.2 to 0.1 for stability
    
    def forward(self, idx, targets=None):
        # Token embeddings
        b, t = idx.size()
        token_embeddings = self.transformer.wte(idx)  # [B, T, n_embd]
        position_embeddings = self.transformer.wpe(torch.arange(0, t, dtype=torch.long, device=idx.device)) # [T, n_embd]
        x = self.transformer.drop(token_embeddings + position_embeddings)
        
        # Extract and process belief patterns
        patterns = self._extract_belief_patterns(x)
        
        # Update belief system (batch updates if possible)
        unique_patterns = list(set(patterns))  # Remove duplicates
        for source, target in unique_patterns:
            state = self.belief_system.process_expression(f"{source}+{target}")
            self.belief_states.append(state)
            self.stability_history.append(self.belief_system.stability)
        
        # Apply transformer blocks with belief enhancement
        for block in self.transformer.h:
            x = block(x)
            x = self._apply_belief_enhancement(x)
        
        x = self.transformer.ln_f(x)
        
        if targets is not None:
            # Calculate loss
            logits = self.lm_head(x)
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            
            # Add stability regularization (ensure float32)
            stability_factor = torch.tensor(self.belief_system.stability, dtype=torch.float32, device=x.device)
            stability_loss = 0.1 * (1 - stability_factor)  # Penalize low stability
            
            return logits, loss + stability_loss
        else:
            # Inference-only forward pass
            logits = self.lm_head(x)
            return logits
    
    def get_belief_metrics(self):
        """Return current belief system metrics."""
        return {
            'stability': float(self.belief_system.stability),  # Convert to Python float
            'pattern_stats': self.belief_system.get_pattern_stats(),
            'stability_history': [float(x) for x in self.stability_history]  # Convert to Python floats
        } 