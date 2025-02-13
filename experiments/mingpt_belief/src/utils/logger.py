import os
import csv
import numpy as np
from datetime import datetime

class BeliefSystemLogger:
    """
    Logs belief system test data in readable CSV format.
    Tracks:
    1. Pattern strengths and changes
    2. System stability
    3. Attack impacts
    4. Key metrics at each significant step
    """
    
    def __init__(self, test_name):
        """Initialize the logger with a test name."""
        self.test_name = test_name
        self.log_dir = 'logs'
        self.log_file = self._create_log_file()
        self.steps = []
        self.phase_summaries = []
        
        # Create headers
        headers = [
            'Step', 'Phase', 'Action', 'System_Stability',
            'Top_Pattern_Strength', 'Top_Pattern',
            'Attack_Impact', 'Active_Patterns'
        ]
        
        # Create log directory if it doesn't exist
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        # Initialize CSV file with headers
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
    
    def _create_log_file(self):
        """Create a unique log file name."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return os.path.join(self.log_dir, f'{self.test_name}_{timestamp}.csv')
    
    def log_step(self, phase, action, belief_matrix, attack_matrix=None):
        """Log a single step in the belief system's evolution."""
        step = len(self.steps) + 1
        
        # Calculate metrics
        top_pattern, top_strength = self._get_top_pattern(belief_matrix)
        active_patterns = self._count_active_patterns(belief_matrix)
        stability = self._calculate_stability(belief_matrix)
        
        # Calculate attack impact if there's an attack matrix
        attack_impact = 0
        if attack_matrix is not None:
            attack_impact = np.sum(attack_matrix)
        
        # Create log entry
        log_entry = {
            'step': step,
            'phase': phase,
            'action': action,
            'stability': stability,
            'top_pattern_strength': top_strength,
            'top_pattern': top_pattern,
            'attack_impact': attack_impact,
            'active_patterns': active_patterns
        }
        
        self.steps.append(log_entry)
        
        # Write to CSV
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                step, phase, action, stability,
                top_strength, top_pattern,
                attack_impact, active_patterns
            ])
        
        # Print progress
        self.print_progress(log_entry)
    
    def _get_top_pattern(self, belief_matrix):
        """Find the strongest pattern in the belief matrix."""
        if belief_matrix.size == 0:
            return "None", 0.0
        
        max_idx = np.argmax(belief_matrix)
        i, j = np.unravel_index(max_idx, belief_matrix.shape)
        strength = belief_matrix[i, j]
        
        if strength > 0:
            return f"{i}→{j}", strength
        return "None", 0.0
    
    def _count_active_patterns(self, belief_matrix, threshold=0.1):
        """Count patterns with strength above threshold."""
        return np.sum(belief_matrix > threshold)
    
    def _calculate_stability(self, belief_matrix):
        """Calculate system stability based on pattern strengths."""
        if belief_matrix.size == 0:
            return 0.0
        
        # Normalize by max possible value
        max_possible = belief_matrix.size
        total_strength = np.sum(belief_matrix)
        return total_strength / max_possible if max_possible > 0 else 0.0
    
    def log_phase_summary(self, phase, data):
        """Log a summary of a testing phase."""
        summary = {
            'phase': phase,
            'data': data,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        self.phase_summaries.append(summary)
    
    def get_summary_stats(self):
        """Get summary statistics for the entire test."""
        if not self.steps:
            return {
                'avg_stability': 0.0,
                'min_stability': 0.0,
                'max_stability': 0.0,
                'final_stability': 0.0
            }
        
        stabilities = [step['stability'] for step in self.steps]
        return {
            'avg_stability': np.mean(stabilities),
            'min_stability': min(stabilities),
            'max_stability': max(stabilities),
            'final_stability': stabilities[-1]
        }
    
    def print_progress(self, log_entry):
        """Print progress information to console."""
        print(f"\nStep {log_entry['step']} - {log_entry['phase']}")
        print(f"Action: {log_entry['action']}")
        print(f"Stability: {log_entry['stability']:.3f}")
        print(f"Top Pattern: {log_entry['top_pattern']} ({log_entry['top_pattern_strength']:.3f})")
        if log_entry['attack_impact'] > 0:
            print(f"Attack Impact: {log_entry['attack_impact']:.3f}")
        print(f"Active Patterns: {log_entry['active_patterns']}") 