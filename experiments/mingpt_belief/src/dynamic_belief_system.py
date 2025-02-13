import numpy as np
from typing import List, Tuple
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MinimalDynamicBeliefSystem:
    """
    A minimal implementation of a dynamic belief system that learns patterns through
    attention mechanisms and feedback loops.
    """
    
    def __init__(self, dimension: int = 2, learning_rate: float = 0.1,
                 importance_threshold: float = 0.01,
                 temporal_decay_rate: float = 0.001):
        """
        Initialize the belief system.
        
        Args:
            dimension (int): Size of the belief matrix (default: 2)
            learning_rate (float): Initial learning rate (default: 0.1)
            importance_threshold (float): Minimum importance to maintain a pattern
            temporal_decay_rate (float): Rate at which old patterns decay
        """
        self.dimension = dimension
        self.P = np.zeros((dimension, dimension))  # Belief matrix
        self.learning_rate = learning_rate
        self.stability = 0.0
        self.history: List[Tuple[np.ndarray, float]] = []  # [(P_matrix, stability)]
        self.importance_threshold = importance_threshold
        self.temporal_decay_rate = temporal_decay_rate
        self.pattern_timestamps = np.zeros((dimension, dimension))  # Track pattern age
        self.pattern_frequency = np.zeros((dimension, dimension))  # Track pattern frequency
        
        logger.info(f"Initialized belief system with dimension {dimension}")
    
    def process_expression(self, expr: str) -> np.ndarray:
        """
        Process an input expression and update the belief matrix.
        
        Args:
            expr (str): Input expression (e.g., "2+3")
            
        Returns:
            np.ndarray: Current state of belief matrix
        """
        # Apply temporal decay before processing new pattern
        self._apply_temporal_decay()
        
        # Convert expression to attention pattern
        pattern = self._compute_attention_pattern(expr)
        
        # Update pattern frequency
        self.pattern_frequency += pattern
        
        # Update timestamps for active patterns
        current_time = datetime.now().timestamp()
        self.pattern_timestamps[pattern > 0] = current_time
        
        # Update beliefs through feedback
        self._update_with_feedback(pattern)
        
        # Prune weak patterns
        self._prune_weak_patterns()
        
        # Compute new stability
        self.stability = self._compute_stability()
        
        # Store history
        self.history.append((self.P.copy(), self.stability))
        
        # Force additional pruning after processing
        self._force_prune_inactive_patterns()
        
        logger.info(f"Processed expression: {expr}, Current stability: {self.stability:.4f}")
        return self.P
    
    def _compute_attention_pattern(self, expr: str) -> np.ndarray:
        """
        Convert an expression into an attention pattern matrix.
        
        Args:
            expr (str): Input expression
            
        Returns:
            np.ndarray: Attention pattern matrix
        """
        pattern = np.zeros((self.dimension, self.dimension))
        
        try:
            # Simple parsing for basic arithmetic expressions
            # Currently handles only addition format "a+b"
            parts = expr.split("+")
            if len(parts) == 2:
                a, b = int(parts[0]), int(parts[1])
                if 0 <= a < self.dimension and 0 <= b < self.dimension:
                    pattern[a, b] = 1.0
                    # For commutative properties, also mark symmetric position
                    pattern[b, a] = 1.0
        except (ValueError, IndexError) as e:
            logger.warning(f"Failed to parse expression {expr}: {e}")
            
        return pattern
    
    def _compute_pattern_importance(self, pattern: np.ndarray) -> np.ndarray:
        """
        Compute importance weights for patterns based on frequency and recency.
        
        Args:
            pattern: Current pattern being processed
            
        Returns:
            np.ndarray: Importance weights for each connection
        """
        current_time = datetime.now().timestamp()
        time_since_update = current_time - self.pattern_timestamps
        
        # Compute recency factor (exponential decay)
        recency = np.exp(-self.temporal_decay_rate * time_since_update)
        
        # Compute frequency factor (normalized)
        max_freq = np.max(self.pattern_frequency) if np.max(self.pattern_frequency) > 0 else 1
        frequency = self.pattern_frequency / max_freq
        
        # Combine recency and frequency with adjusted weights
        # Give more weight to frequency for stability
        importance = (0.4 * recency + 0.6 * frequency) * pattern
        
        # Normalize importance to [0, 1] range
        max_importance = np.max(importance) if np.max(importance) > 0 else 1
        importance = importance / max_importance
        
        return importance
    
    def _update_with_feedback(self, pattern: np.ndarray) -> None:
        """
        Update belief matrix using the attention pattern and feedback loop.
        
        Args:
            pattern (np.ndarray): Current attention pattern
        """
        # Compute pattern importance
        importance = self._compute_pattern_importance(pattern)
        
        # Simple feedback loop: move current beliefs toward the pattern
        delta = pattern - self.P
        
        # Adjust learning rate based on stability and importance
        effective_rate = self.learning_rate * (1 - self.stability) * importance
        
        # Update beliefs with minimum learning rate to prevent complete stagnation
        min_rate = self.learning_rate * 0.01  # 1% minimum learning rate
        effective_rate = np.maximum(effective_rate, min_rate * pattern)
        
        # Update beliefs
        self.P += effective_rate * delta
        
        # Ensure beliefs stay in valid range [0, 1]
        self.P = np.clip(self.P, 0, 1)
    
    def _apply_temporal_decay(self) -> None:
        """Apply temporal decay to all patterns based on their age."""
        current_time = datetime.now().timestamp()
        time_since_update = current_time - self.pattern_timestamps
        
        # Compute decay factor with smaller minimum retention
        min_retention = 0.0001  # 0.01% minimum retention
        decay = min_retention + (1 - min_retention) * np.exp(-self.temporal_decay_rate * time_since_update)
        
        # Only apply decay to non-zero elements
        mask = self.P > 0
        self.P[mask] *= decay[mask]
        
        # If pattern is very weak after decay, prune it immediately
        self.P[self.P < self.importance_threshold / 2] = 0
    
    def _prune_weak_patterns(self) -> None:
        """Remove patterns below importance threshold."""
        # Compute current importance of all patterns
        importance = self._compute_pattern_importance(np.ones_like(self.P))
        
        # Identify weak patterns considering both strength and importance
        pattern_strength = self.P * importance
        
        # Explicit pruning of all patterns below threshold
        self.P[pattern_strength < self.importance_threshold] = 0
        
        # Clear metadata for pruned patterns
        pruned = self.P == 0
        self.pattern_timestamps[pruned] = 0
        self.pattern_frequency[pruned] = 0
    
    def _compute_stability(self) -> float:
        """
        Compute the current stability of the belief system.
        
        Returns:
            float: Stability metric between 0 and 1
        """
        if len(self.history) < 2:
            return 0.0
            
        # Compute change from previous state
        prev_P = self.history[-1][0]
        delta = np.abs(self.P - prev_P).mean()
        
        # Convert to stability metric (1 - normalized change)
        stability = 1.0 - min(delta / self.learning_rate, 1.0)
        return stability
    
    def get_history(self) -> List[Tuple[np.ndarray, float]]:
        """
        Get the history of belief matrix states and stability values.
        
        Returns:
            List[Tuple[np.ndarray, float]]: List of (P_matrix, stability) pairs
        """
        return self.history.copy()
    
    def get_pattern_stats(self) -> dict:
        """
        Get statistics about current patterns.
        
        Returns:
            dict: Dictionary containing pattern statistics
        """
        return {
            'active_patterns': np.count_nonzero(self.P > 0),
            'max_strength': np.max(self.P),
            'mean_strength': np.mean(self.P[self.P > 0]) if np.any(self.P > 0) else 0,
            'pattern_frequency': self.pattern_frequency.copy(),
            'oldest_pattern_age': datetime.now().timestamp() - np.min(self.pattern_timestamps[self.pattern_timestamps > 0])
            if np.any(self.pattern_timestamps > 0) else 0
        }
    
    def _force_prune_inactive_patterns(self) -> None:
        """Force pruning of patterns that haven't been reinforced recently."""
        current_time = datetime.now().timestamp()
        inactive_time = current_time - self.pattern_timestamps
        
        # Prune patterns that haven't been updated recently and are weak
        inactive_patterns = (inactive_time > 0.1) & (self.P < self.importance_threshold)
        self.P[inactive_patterns] = 0
        self.pattern_timestamps[inactive_patterns] = 0
        self.pattern_frequency[inactive_patterns] = 0 