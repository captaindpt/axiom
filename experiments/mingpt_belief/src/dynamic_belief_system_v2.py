import numpy as np
from typing import List, Tuple, Dict
import logging
from datetime import datetime
from enum import Enum
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PatternState(Enum):
    """Enum representing possible states of a pattern."""
    INACTIVE = 0    # Pattern exists but is not currently influencing behavior
    ACTIVE = 1      # Pattern is actively influencing behavior
    DOMINANT = 2    # Pattern is strongly influencing behavior

class DynamicBeliefSystem:
    """
    Enhanced implementation of a dynamic belief system using activation states
    instead of pruning, allowing for pattern persistence and reactivation.
    """
    
    def __init__(self, dimension=10, learning_rate=0.1, temporal_decay_rate=0.1,
                 activation_threshold=0.1, dominance_threshold=0.5):
        self.dimension = dimension
        self.learning_rate = learning_rate
        self.temporal_decay_rate = temporal_decay_rate
        self.activation_threshold = activation_threshold
        self.dominance_threshold = dominance_threshold
        
        # Initialize matrices
        self.P = np.zeros((dimension, dimension))
        self.stability = 1.0
        self.activation_state = np.full((dimension, dimension), PatternState.INACTIVE.value)
        
        # Pattern metadata
        self.last_update = np.zeros((dimension, dimension))
        self.frequency = np.zeros((dimension, dimension))
        self.confidence = np.zeros((dimension, dimension))
        
        # Setup logging
        self.history = []
        
    def process_expression(self, expression):
        # Parse expression (assuming format "X+Y")
        source, target = map(int, expression.split('+'))
        
        # Apply temporal decay
        self._apply_temporal_decay()
        
        # Update pattern strength with initial weak reinforcement
        current_strength = self.P[source, target]
        if current_strength == 0:
            # Initial pattern is weak
            self.P[source, target] = self.learning_rate * 0.5
        else:
            # Enhanced reinforcement calculation
            freq = self.frequency[source, target]
            
            # Slower initial growth, faster later
            if freq <= 3:
                freq_boost = 1.0 + freq * 0.1
            else:
                freq_boost = min(2.0, 1.0 + freq * 0.2)
                
            conf_boost = 1.0 + self.confidence[source, target] * 0.5
            
            # Base reinforcement with controlled boosts
            reinforcement = (self.learning_rate * 
                           (1 - current_strength) * 
                           freq_boost * 
                           conf_boost)
            
            # Additional boost only after sufficient reinforcement
            if freq > 3 and 0.3 <= current_strength <= self.dominance_threshold:
                reinforcement *= 1.5
                
            self.P[source, target] = min(1.0, current_strength + reinforcement)
        
        # Update metadata
        self.last_update[source, target] = time.time()
        self.frequency[source, target] += 1
        self.confidence[source, target] = min(1.0, self.frequency[source, target] / 5)
        
        # Update activation state
        self._update_activation_state(source, target)
        
        # Update stability
        self._update_stability()
        
        # Store history
        self.history.append({
            'P': self.P.copy(),
            'stability': self.stability,
            'timestamp': time.time()
        })
        
        return self.get_pattern_state(source, target)
    
    def _update_activation_state(self, source, target):
        strength = self.P[source, target]
        confidence = self.confidence[source, target]
        freq = self.frequency[source, target]
        
        # Enhanced state transition logic with hysteresis
        current_state = PatternState(self.activation_state[source, target])
        
        if current_state == PatternState.DOMINANT:
            # Dominant patterns need to fall significantly to lose status
            if strength < self.dominance_threshold * 0.8:
                self.activation_state[source, target] = PatternState.ACTIVE.value
        elif current_state == PatternState.ACTIVE:
            # Active patterns can become dominant or inactive
            if strength >= self.dominance_threshold and confidence >= 0.6 and freq > 4:
                self.activation_state[source, target] = PatternState.DOMINANT.value
            elif strength < self.activation_threshold * 0.8:
                self.activation_state[source, target] = PatternState.INACTIVE.value
        else:  # INACTIVE
            # Inactive patterns need significant strength to become active
            if strength >= self.activation_threshold:
                self.activation_state[source, target] = PatternState.ACTIVE.value
    
    def _apply_temporal_decay(self):
        current_time = time.time()
        time_delta = current_time - self.last_update
        decay_factor = np.exp(-self.temporal_decay_rate * time_delta)
        
        # Apply decay only to non-dominant patterns
        dominant_mask = self.activation_state == PatternState.DOMINANT.value
        self.P[~dominant_mask] *= decay_factor[~dominant_mask]
        
        # Reset states for significantly decayed patterns
        inactive_mask = self.P < (self.activation_threshold * 0.8)
        self.activation_state[inactive_mask] = PatternState.INACTIVE.value
    
    def _update_stability(self):
        """Update system stability based on current state and changes."""
        # Calculate active patterns
        active_patterns = (self.activation_state >= PatternState.ACTIVE.value).sum()
        
        # Base stability calculation
        if active_patterns == 0:
            raw_stability = 1.0
        else:
            # Calculate stability components
            active_mask = self.activation_state >= PatternState.ACTIVE.value
            dominant_mask = self.activation_state == PatternState.DOMINANT.value
            
            # Strength stability (how close patterns are to their target states)
            mean_strength = np.mean(self.P[active_mask])
            strength_stability = mean_strength
            
            # State stability (ratio of dominant to active patterns)
            dominant_count = np.sum(dominant_mask)
            state_stability = dominant_count / active_patterns if active_patterns > 0 else 0
            
            # Pattern consistency (how focused the system is)
            total_possible = self.dimension * self.dimension
            pattern_consistency = 1.0 - (active_patterns / total_possible)
            
            # Learning progress (how many patterns have reached their target states)
            progress = (np.sum(self.P[active_mask] >= self.activation_threshold) / 
                       active_patterns)
            
            # Combine base stability components
            raw_stability = (0.25 * strength_stability + 
                           0.25 * state_stability + 
                           0.2 * pattern_consistency +
                           0.3 * progress +
                           0.4)  # Lower base stability
        
        # Calculate penalties
        total_penalty = 0.0
        
        # Pattern conflict detection
        conflict_penalty = 0.0
        conflict_count = 0
        max_conflict_strength = 0.0
        
        # Look for contradictory patterns (bidirectional connections)
        for i in range(self.dimension):
            for j in range(i + 1, self.dimension):
                if self.P[i,j] > 0 and self.P[j,i] > 0:
                    # Calculate contradiction strength based on both patterns
                    forward_strength = self.P[i,j]
                    backward_strength = self.P[j,i]
                    
                    # Stronger penalty when patterns are more similar in strength
                    strength_ratio = min(forward_strength, backward_strength) / max(forward_strength, backward_strength)
                    conflict_strength = min(forward_strength, backward_strength) * (0.5 + 0.5 * strength_ratio)
                    
                    # Track maximum conflict strength
                    max_conflict_strength = max(max_conflict_strength, conflict_strength)
                    
                    conflict_penalty += conflict_strength * 0.6  # Increased multiplier
                    conflict_count += 1
        
        # Scale conflict penalty based on number of conflicts and maximum strength
        if conflict_count > 0:
            conflict_penalty *= (1 + 0.3 * (conflict_count - 1))  # Increased scaling
            conflict_penalty = min(0.7, conflict_penalty)  # Increased maximum
            
            # Add immediate stability impact for strong conflicts
            if max_conflict_strength > 0.3:
                conflict_penalty = max(conflict_penalty, 0.4)
            
            total_penalty += conflict_penalty
        
        # Change detection penalty
        if len(self.history) > 0:
            last_state = self.history[-1]
            pattern_diff = np.abs(self.P - last_state['P'])
            significant_changes = pattern_diff > 0.01
            
            # Penalize changes based on type
            new_patterns = significant_changes & (last_state['P'] == 0)
            reinforced_patterns = significant_changes & (last_state['P'] > 0)
            
            change_penalty = (
                np.sum(new_patterns) * 0.3 +  # Increased penalty for new patterns
                np.sum(reinforced_patterns) * 0.15  # Increased penalty for reinforcement
            )
            total_penalty += min(0.4, change_penalty)  # Increased maximum
        
        # Ensure minimum penalty for any conflicts
        if conflict_count > 0:
            total_penalty = max(total_penalty, 0.3)  # Increased minimum penalty
        
        # Calculate final stability with penalties
        target_stability = max(0.0, raw_stability - total_penalty)
        
        # Apply smooth transition with faster decrease for instability
        if len(self.history) > 0:
            prev_stability = self.history[-1]['stability']
            max_decrease = 0.5 if target_stability < prev_stability else 0.2  # Increased maximum decrease
            
            self.stability = prev_stability + np.clip(
                target_stability - prev_stability,
                -max_decrease,
                0.2
            )
        else:
            self.stability = target_stability
        
        # Final bounds check
        self.stability = min(1.0, max(0.0, self.stability))
    
    def get_pattern_state(self, source, target):
        return {
            'state': PatternState(self.activation_state[source, target]),
            'strength': float(self.P[source, target]),
            'confidence': float(self.confidence[source, target]),
            'frequency': int(self.frequency[source, target]),
            'stability': float(self.stability)
        }
    
    def get_pattern_stats(self):
        active_mask = self.activation_state == PatternState.ACTIVE.value
        dominant_mask = self.activation_state == PatternState.DOMINANT.value
        
        return {
            'active_patterns': int(np.sum(active_mask)),
            'dominant_patterns': int(np.sum(dominant_mask)),
            'mean_strength': float(np.mean(self.P[self.P > 0])),
            'mean_confidence': float(np.mean(self.confidence[self.confidence > 0]))
        } 